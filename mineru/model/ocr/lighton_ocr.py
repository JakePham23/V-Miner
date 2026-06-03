# Copyright (c) Opendatalab. All rights reserved.
"""
LightOnOCR — Unified OCR client cho MinerU.

Kiến trúc:
  LightOnOCR (facade)
    ├─ LightOnOCRAPI   — gọi API (OpenAI-compatible, Ollama, LM Studio, v.v.)
    └─ LightOnModelLocal — chạy model local (mlx / transformers)

Logic ưu tiên:
  1. Nếu LLM_SERVICE != "local" → thử API trước
  2. Nếu API fail hoặc LLM_SERVICE == "local" → dùng local model
  3. Nếu không có backend nào → raise RuntimeError

Vietnamese table:
  Khi nhận ra bảng có khả năng chứa tiếng Việt (lang='vi' hoặc detect ký tự),
  tự động dùng prompt tiếng Việt để đảm bảo đúng dấu thanh.

CHANGES vs original:
  - Thêm LightOnOCRAPI (đổi tên từ LightOnOCR cũ)
  - Thêm LightOnOCR facade với API-first + local fallback
  - Đọc config qua lighton_config.get_lighton_config()
  - Thêm _is_vietnamese_content() + _vietnamese_table_prompt
  - recognize_table nhận tham số vietnamese=True/auto
"""
import os
import re
import base64
import requests
from io import BytesIO
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
from PIL import Image
from loguru import logger


# ── Image processing constants ──────────────────────────────────────────────────
_PAD_RATIO   = 0.04  # 4% padding
_PAD_MIN_PX  = 6
_PAD_MAX_PX  = 20
_MIN_DIM_PX  = 320   # Giảm từ 640 xuống 320
_MAX_DIM_PX  = 1280  # Giảm từ 2048 xuống 1280

# Regex nhận diện ký tự tiếng Việt có dấu
_VIET_DIACRITIC_RE = re.compile(
    r"[àáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắặẳẵắặẻẽẹếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ"
    r"ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẶẲẴẮẶẺẼẸẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỶỸỴ]",
    re.UNICODE,
)


# ── HTML to Markdown Conversion Helpers ──────────────────────────────────────

def _get_html_text(html_fragment: str) -> str:
    text = re.sub(r'<br\s*/?>', ' ', html_fragment, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    return (text
            .replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            .replace('&nbsp;', ' ').replace('&ge;', '≥').replace('&le;', '≤')
            .replace('&#8805;', '≥').replace('&#8804;', '≤')).strip()


def _parse_section_to_grid(section_html: str) -> list[list[str]]:
    rows_raw = re.findall(r'<tr[^>]*>(.*?)</tr>', section_html, re.IGNORECASE | re.DOTALL)
    if not rows_raw:
        return []

    grid: list[dict] = []
    occupied: dict = {}

    for row_idx, row_html in enumerate(rows_raw):
        while len(grid) <= row_idx:
            grid.append({})

        col_cursor = 0
        for m in re.finditer(r'<t[hd]([^>]*)>(.*?)</t[hd]>', row_html, re.IGNORECASE | re.DOTALL):
            attrs = m.group(1)
            text = _get_html_text(m.group(2))

            colspan = int(c.group(1)) if (c := re.search(r'colspan=["\']?(\d+)["\']?', attrs, re.IGNORECASE)) else 1
            rowspan = int(r.group(1)) if (r := re.search(r'rowspan=["\']?(\d+)["\']?', attrs, re.IGNORECASE)) else 1

            while (row_idx, col_cursor) in occupied:
                col_cursor += 1

            for c_off in range(colspan):
                col = col_cursor + c_off
                grid[row_idx][col] = text if c_off == 0 else ""
                for r_off in range(1, rowspan):
                    occupied[(row_idx + r_off, col)] = ""

            col_cursor += colspan

        for (r, c), text in occupied.items():
            if r == row_idx and c not in grid[row_idx]:
                grid[row_idx][c] = text

    if not grid:
        return []

    n_cols = max((max(row.keys()) + 1 for row in grid if row), default=0)
    return [[row.get(c, '') for c in range(n_cols)] for row in grid]


def _parse_table_to_grid(html: str) -> tuple[list[list[str]], list[list[str]]]:
    thead_m = re.search(r'<thead[^>]*>(.*?)</thead>', html, re.IGNORECASE | re.DOTALL)
    tbody_m = re.search(r'<tbody[^>]*>(.*?)</tbody>', html, re.IGNORECASE | re.DOTALL)

    thead_html = thead_m.group(1) if thead_m else ''
    tbody_html = tbody_m.group(1) if tbody_m else ''

    if not thead_html and not tbody_html:
        tbody_html = html

    header_rows = _parse_section_to_grid(thead_html) if thead_html else []
    body_rows   = _parse_section_to_grid(tbody_html) if tbody_html else []

    if not header_rows and body_rows:
        first_tr = re.search(r'<tr[^>]*>(.*?)</tr>', tbody_html, re.IGNORECASE | re.DOTALL)
        if first_tr and '<th' in first_tr.group(1).lower():
            header_rows = body_rows[:1]
            body_rows   = body_rows[1:]

    return header_rows, body_rows


def _merge_multirow_header(header_rows: list[list[str]]) -> list[str]:
    if not header_rows:
        return []
    if len(header_rows) == 1:
        return header_rows[0]

    n_cols = max(len(r) for r in header_rows)
    rows = [r + [''] * (n_cols - len(r)) for r in header_rows]
    n_rows = len(rows)

    merged = []
    parent_map = {}
    last_parent = ''
    for col in range(n_cols):
        top_val = rows[0][col]
        if top_val:
            last_parent = top_val
        parent_map[col] = last_parent

    for col in range(n_cols):
        values = [rows[r][col] for r in range(n_rows)]
        unique = list(dict.fromkeys(v for v in values if v))

        if not unique:
            merged.append('')
        elif len(unique) == 1:
            merged.append(unique[0])
        else:
            parent = unique[0]
            sub    = ' / '.join(unique[1:])
            merged.append(f'{parent} / {sub}')

    return merged


def html_table_to_markdown(html: str) -> str:
    html = html.strip()
    if not html.lower().startswith('<table'):
        return html

    header_rows, body_rows = _parse_table_to_grid(html)
    if not header_rows and not body_rows:
        return html

    header = _merge_multirow_header(header_rows) if header_rows else []

    n_cols = max(
        len(header),
        max((len(r) for r in body_rows), default=0)
    )
    if n_cols == 0:
        return html

    def pad(row: list[str]) -> list[str]:
        return (row + [''] * n_cols)[:n_cols]

    lines = []
    if header:
        lines.append('| ' + ' | '.join(pad(header)) + ' |')
        lines.append('| ' + ' | '.join(['---'] * n_cols) + ' |')
    else:
        lines.append('| ' + ' | '.join([f'Col{i+1}' for i in range(n_cols)]) + ' |')
        lines.append('| ' + ' | '.join(['---'] * n_cols) + ' |')

    for row in body_rows:
        lines.append('| ' + ' | '.join(pad(row)) + ' |')

    return '\n'.join(lines)


# ── Flat multi-row header fixer ──────────────────────────────────────────────

def _parse_md_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip('|').split('|')]


def _is_md_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith('|') and s.endswith('|') and len(s) > 2


def _is_md_separator(line: str) -> bool:
    return bool(re.match(r'^\|[-| :]+\|$', line.strip()))


def _fix_flat_multirow_header(md_text: str) -> str:
    lines = md_text.splitlines()
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if not _is_md_table_row(line):
            result.append(line)
            i += 1
            continue

        next1 = lines[i + 1] if i + 1 < len(lines) else ''
        next2 = lines[i + 2] if i + 2 < len(lines) else ''

        if not (_is_md_table_row(next1) and _is_md_separator(next2)):
            result.append(line)
            i += 1
            continue

        row1 = _parse_md_row(line)
        row2 = _parse_md_row(next1)

        if len(row1) != len(row2):
            result.append(line)
            i += 1
            continue

        n = len(row1)
        leading_empty = 0
        for c in row2:
            if c == '':
                leading_empty += 1
            else:
                break

        has_sub_labels = any(c for c in row2[leading_empty:])

        if leading_empty < 2 or not has_sub_labels:
            result.append(line)
            i += 1
            continue

        merged = []
        last_parent = ''
        for c in range(n):
            r1 = row1[c]
            r2 = row2[c]
            if r1:
                last_parent = r1
            if r2:
                merged.append(f'{last_parent} / {r2}' if last_parent else r2)
            else:
                merged.append(r1)

        result.append('| ' + ' | '.join(merged) + ' |')
        result.append(next2)
        i += 3
        continue

    return '\n'.join(result)


# ── Prompt echo cleaner ───────────────────────────────────────────────────────
_PROMPT_ECHO_PATTERNS = [
    "Extract all text from this image accurately. Output only the extracted text.",
    "Extract the table from this image.",
    "Output ONLY a standard Markdown table",
    "Output ONLY a valid HTML",
    "Do not add any explanation or preamble.",
    "Ensure Vietnamese text is accurate.",
    "Đây là bảng tiếng Việt.",
]

_ARTIFACT_PATTERNS = [
    (re.compile(r'^\$\^ \+\$\s*$', re.MULTILINE), ''),
    (re.compile(r'^#\s*$', re.MULTILINE), ''),
    (re.compile(r'^Note:.*$', re.MULTILINE | re.IGNORECASE), ''),
]

def _clean_ocr_response(text: str, prompt: str = "") -> str:
    if not text:
        return ""

    t = text.strip()
    if prompt and t.startswith(prompt.strip()):
        t = t[len(prompt.strip()):].strip()

    # Check for common VLM "no text" placeholders
    lower_t = t.lower()
    no_text_indicators = [
        "no text visible",
        "no visible text",
        "no text found",
        "does not contain any text",
        "empty image",
        "blank image",
        "no readable text"
    ]
    if any(ind in lower_t for ind in no_text_indicators):
        return ""

    lines = t.splitlines()
    clean_lines = []

    # Generic instruction/prompt lines to drop
    drop_patterns = [
        "do not output",
        "extract all text",
        "output only",
        "raw text (extracted",
        "note:",
        "output format:",
        "additional notes:",
        "end of document",
        "do not add any",
        "preamble",
        "explanation",
        "code blocks",
        "html tables",
        "latex formulas"
    ]

    for line in lines:
        s_line = line.strip().lower()
        if not s_line:
            continue
        if any(pat in s_line for pat in drop_patterns):
            continue
        if s_line.startswith("---") or s_line.startswith("***"):
            continue
        clean_lines.append(line)

    cleaned = "\n".join(clean_lines).strip()

    # Final check: if the cleaned text consists only of markdown artifacts/notes, return empty
    lower_cleaned = cleaned.lower()
    if not cleaned or any(ind in lower_cleaned for ind in no_text_indicators):
        return ""

    return cleaned


# ── Smart crop helper ────────────────────────────────────────────────────────

def crop_for_lighton(
    poly: list,
    page_img: np.ndarray,
    pad_ratio: float = _PAD_RATIO,
    pad_min: int = _PAD_MIN_PX,
    pad_max: int = _PAD_MAX_PX,
) -> np.ndarray:
    if len(poly) >= 8:
        xs = [poly[i]     for i in range(0, 8, 2)]
        ys = [poly[i + 1] for i in range(0, 8, 2)]
    elif len(poly) == 4:
        xs = [poly[0], poly[2]]
        ys = [poly[1], poly[3]]
    else:
        raise ValueError(f"poly phải có 4 hoặc 8 phần tử, nhận {len(poly)}")

    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys)), int(max(ys))

    short_side = min(x1 - x0, y1 - y0)
    pad = int(short_side * pad_ratio)
    pad = max(pad_min, min(pad, pad_max))

    h_img, w_img = page_img.shape[:2]
    x0c = max(0, x0 - pad)
    y0c = max(0, y0 - pad)
    x1c = min(w_img, x1 + pad)
    y1c = min(h_img, y1 + pad)

    cropped = page_img[y0c:y1c, x0c:x1c]
    if cropped.size == 0:
        logger.warning(f"crop_for_lighton: crop rỗng tại poly={poly}")
        return page_img
    return cropped


def _preprocess_image(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image

    short, long_ = min(h, w), max(h, w)

    if short < _MIN_DIM_PX:
        scale = _MIN_DIM_PX / short
        new_w = int(w * scale)
        new_h = int(h * scale)
        if max(new_w, new_h) > _MAX_DIM_PX:
            scale = _MAX_DIM_PX / long_
            new_w = int(w * scale)
            new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    elif long_ > _MAX_DIM_PX:
        scale = _MAX_DIM_PX / long_
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image


# ── Vietnamese detection helper ───────────────────────────────────────────────

def _is_vietnamese_content(image: Union[np.ndarray, Image.Image]) -> bool:
    """
    Heuristic: phát hiện nội dung tiếng Việt.
    EasyOCR đã bị xóa — trả về False để pipeline dùng lang mặc định của caller.
    """
    return False


# ── LightOnOCRAPI (internal, API-only) ───────────────────────────────────────

class LightOnOCRAPI:
    """OCR via OpenAI-compatible API (LM Studio, Ollama, OpenAI, Azure, v.v.)."""

    def __init__(self, cfg=None, server_url: str = None, model_name: str = None, timeout: int = 180, **kwargs):
        from mineru.utils.lighton_config import get_lighton_config, build_api_headers

        self._cfg = cfg or get_lighton_config()
        self.server_url = server_url or self._cfg.chat_completions_url
        self.model_name = model_name or self._cfg.model
        self.timeout    = timeout
        self.drop_score = kwargs.get("drop_score", 0.5)
        self._headers   = build_api_headers(self._cfg)
        logger.debug(f"[OCR API] URL={self.server_url!r} model={self.model_name!r} service={self._cfg.llm_service!r}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _to_numpy_rgb(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            if image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _call_api(self, image: Union[np.ndarray, Image.Image], prompt: str, temperature: float = 0.0) -> str:
        # Chuyển ảnh thành PIL để lấy base64 chuẩn
        if isinstance(image, np.ndarray):
            pil = Image.fromarray(self._to_numpy_rgb(image))
        else:
            pil = image.convert("RGB")
        pil = Image.fromarray(_preprocess_image(np.array(pil)))
        
        buffered = BytesIO()
        pil.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{img_str}"

        if "/api/" in self.server_url:
            # Ollama style
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt, "images": [img_str]}],
                "stream": False,
                "options": {"temperature": temperature}
            }
            try:
                r = requests.post(self.server_url, json=payload, headers=self._headers, timeout=self.timeout)
                if r.status_code == 200:
                    res  = r.json()
                    text = res.get("response", res.get("message", {}).get("content", ""))
                    if not text:
                        logger.warning(f"[OCR API] Ollama: response 200 nhưng content rỗng. raw={res}")
                    return text
                else:
                    logger.error(f"[OCR API] Ollama: HTTP {r.status_code} - {r.text[:200]}")
            except Exception as e:
                logger.error(f"[OCR API] Ollama error: {e}")
                raise
        else:
            # OpenAI / LM Studio / Azure / ... style
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",      "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                "temperature": temperature,
                "max_tokens":  4096,
            }
            try:
                r = requests.post(self.server_url, json=payload, headers=self._headers, timeout=self.timeout)
                if r.status_code == 200:
                    text = r.json()["choices"][0]["message"]["content"]
                    if not text:
                        logger.warning(f"[OCR API] OpenAI: response 200 nhưng content rỗng.")
                    return text
                else:
                    logger.error(f"[OCR API] OpenAI: HTTP {r.status_code} - {r.text[:200]}")
                    raise RuntimeError(f"API HTTP {r.status_code}")
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                logger.warning(f"[OCR API] Connection error: {e}")
                raise
            except Exception as e:
                logger.error(f"[OCR API] Error: {e}")
                raise
        return ""

    # ── Public API ────────────────────────────────────────────────────────────

    def recognize_text(
        self,
        image: Union[np.ndarray, Image.Image],
        *,
        page_img: Optional[np.ndarray] = None,
        poly: Optional[list] = None,
        vietnamese: bool = False,
        lang: str = "",
    ) -> Tuple[str, float]:
        if page_img is not None and poly is not None:
            image = crop_for_lighton(poly, page_img)

        if lang and lang.startswith('vi'):
            vietnamese = True

        if vietnamese:
            prompt = (
                "Trích xuất chính xác toàn bộ văn bản từ hình ảnh này. "
                "Đảm bảo chính xác dấu thanh tiếng Việt (á, à, ả, ã, ạ, ắ, ặ, ẳ, ẵ, ề, ế, ệ, ể, ễ, ổ, ỗ, ộ, v.v.). "
                "CHỈ xuất ra văn bản thô được trích xuất. "
                "KHÔNG xuất ra bất kỳ ghi chú, lời giải thích, bình luận hoặc khối mã markdown nào."
            )
        else:
            prompt = (
                "Extract all text from this image accurately. "
                "Output ONLY the raw extracted text. "
                "DO NOT output any notes, explanations, comments, or markdown code blocks."
            )
        text = self._call_api(image, prompt)
        text = _clean_ocr_response(text, prompt)
        return text.strip(), (0.95 if text else 0.0)

    def recognize_table(
        self,
        image: Union[np.ndarray, Image.Image],
        bbox_coords=None,
        *,
        page_img: Optional[np.ndarray] = None,
        poly: Optional[list] = None,
        vietnamese: bool = False,
        lang: str = "",
        skeleton_html: str = "",
    ) -> str:
        if page_img is not None and poly is not None:
            image = crop_for_lighton(poly, page_img)

        if lang and lang.startswith('vi'):
            vietnamese = True
            
        if skeleton_html and "<table>" in skeleton_html:
            prompt = (
                "Dưới đây là một ảnh chứa bảng và cấu trúc HTML khung của bảng đó đã được dựng sẵn (skeleton). "
                "CẢNH BÁO: Các chữ hiện có bên trong các thẻ <td>, <th> của SKELETON HTML là KẾT QUẢ NHẬN DIỆN BỊ LỖI (MẤT DẤU, SAI CHÍNH TẢ). "
                "NHIỆM VỤ CỦA BẠN: Hãy nhìn trực tiếp vào ảnh, tự đọc lại toàn bộ các chữ tiếng Việt một cách chính xác, "
                "và THAY THẾ HOÀN TOÀN các chữ lỗi xuất hiện trong SKELETON bằng chữ bạn vừa đọc được từ ảnh. "
                "Tuyệt đối KHÔNG ĐƯỢC sao chép lại chữ lỗi từ skeleton. Nếu SKELETON bị sai cấu trúc (như gộp cột sai, thiếu cột so với ảnh), bạn ĐƯỢC PHÉP sửa lại các thẻ <td>, <tr>, rowspan, colspan cho đúng với thực tế ảnh. "
                " "
                "Đảm bảo chính xác dấu thanh tiếng Việt (á, à, ả, ã, ạ, ắ, ặ, ẳ, ẵ, ề, ế, ệ, ể, ễ, ổ, ỗ, ộ, v.v.). "
                "Output ONLY a valid HTML <table>...</table>, KHÔNG kèm giải thích.\n"
                f"SKELETON HTML:\n{skeleton_html}"
            )
        elif vietnamese:
            prompt = (
                "Đây là bảng tiếng Việt (vietnamese). Trích xuất toàn bộ nội dung bảng. Header của bảng có bao nhiêu cột thì hàng dưới cũng phải có bấy nhiêu cột, không được thiếu. Chỉ có thể ít hơn cột header (nếu có tồn tại) "
                "Đảm bảo chính xác dấu thanh tiếng Việt (á, à, ả, ã, ạ, ắ, ặ, ẳ, ẵ, ề, ế, ệ, ể, ễ, ổ, ỗ, ộ, v.v.). "
                "Output ONLY a valid HTML <table> with <thead> and <tbody>. "
                "Do not add any explanation or preamble."
            )
        else:
            prompt = (
                "Extract the table from this image. "
                "Output ONLY a valid HTML <table> with <thead> and <tbody>. "
                "Do not add any explanation or preamble. "
                "Ensure Vietnamese text is accurate."
            )

        import time
        for attempt in range(3):
            current_temp = 0.0 if attempt == 0 else 0.4 + (attempt * 0.2)
            
            # Biến hình prompt để bẻ gãy cache/hallucination của model
            current_prompt = prompt
            if attempt > 0:
                current_prompt += f"\n(Lần thử {attempt + 1}: Tuyệt đối KHÔNG ĐƯỢC lặp lại câu lệnh này. Chỉ xuất ra thẻ <table> hoặc LaTeX cho hình ảnh phía trên!)"
                
            result = self._call_api(image, current_prompt, temperature=current_temp)
            result = _clean_ocr_response(result, current_prompt).strip()

            if result:
                start = result.find("<table")
                end   = result.rfind("</table>")
                if start != -1 and end != -1:
                    result = result[start: end + len("</table>")]
                    break
                elif "\\begin{table}" in result or "\\begin{tabular}" in result:
                    logger.info("[OCR API] recognize_table: Nhận diện được định dạng LaTeX, giữ nguyên.")
                    break
                else:
                    # Phát hiện lỗi "nhại lại" Prompt
                    if len(result) < 200 and ("Đây là bảng" in result or "Extract the table" in result or "Output ONLY a valid HTML" in result):
                        logger.warning("[OCR API] AI nhại lại prompt thay vì nhận diện bảng -> Ép chạy lại.")
                        result = ""  # Cố tình gán bằng rỗng để trigger block 'if not result' phía dưới
                    else:
                        logger.warning(
                            f"[OCR API] recognize_table: output không chứa <table>, "
                            f"trả về văn bản thuần. preview={repr(result[:80])}"
                        )
                        break

            if not result:
                logger.warning(f"[OCR API] recognize_table: API trả về kết quả rỗng (lần {attempt + 1}, temp={current_temp}). Tọa độ bảng (poly): {poly}")
                if attempt < 2:
                    logger.info("Đang chờ 5s trước khi gọi lại API với độ sáng tạo (temperature) cao hơn...")
                    time.sleep(5)

        return result

    def recognize_page(self, image: Union[np.ndarray, Image.Image]) -> str:
        """OCR toàn bộ trang PDF thành Markdown sử dụng VLM API."""
        prompt = (
            "Bạn là một trợ lý OCR chuyên nghiệp. Hãy đọc toàn bộ văn bản, cấu trúc bảng biểu, "
            "danh sách và hình ảnh trong bức ảnh trang tài liệu này. "
            "Hãy giữ nguyên tiếng Việt có dấu và cấu trúc trang. "
            "LƯU Ý: Chỉ trả về nội dung văn bản kết quả, KHÔNG thêm bất kỳ câu chào hỏi, giải thích hay bọc trong thẻ code block nào."
        )
        result = self._call_api(image, prompt, temperature=0.0)
        return _clean_ocr_response(result, prompt).strip()

    def ocr(
        self,
        img,
        det=True,
        rec=True,
        mfd_res=None,
        tqdm_enable=False,
        tqdm_desc="OCR-rec Predict",
        **kwargs,
    ) -> List:
        page_img = kwargs.get("page_img")
        poly     = kwargs.get("poly")
        vietnamese = kwargs.get("vietnamese", False)

        imgs = [img] if isinstance(img, np.ndarray) else img
        is_table = "table" in (tqdm_desc or "").lower()

        if is_table:
            return [
                [
                    (self.recognize_table(image, page_img=page_img, poly=poly, vietnamese=vietnamese), 1.0)
                    for image in imgs
                ]
            ]

        ocr_res = []
        if det and rec:
            for image in imgs:
                text, score = self.recognize_text(image, page_img=page_img, poly=poly)
                if text:
                    h, w = (
                        image.shape[:2]
                        if isinstance(image, np.ndarray)
                        else image.size[::-1]
                    )
                    box = [[0, 0], [w, 0], [w, h], [0, h]]
                    ocr_res.append([[box, (text, score)]])
                else:
                    ocr_res.append(None)
        elif not det and rec:
            ocr_res.append(
                [
                    self.recognize_text(image, page_img=page_img, poly=poly)
                    for image in imgs
                ]
            )

        return ocr_res

    def __call__(
        self,
        img: np.ndarray,
        mfd_res: List = None,
        *,
        page_img: Optional[np.ndarray] = None,
        poly: Optional[list] = None,
    ) -> Tuple[List, List]:
        if img is None:
            return None, None
        text, score = self.recognize_text(img, page_img=page_img, poly=poly)
        if not text:
            return None, None
        h, w = img.shape[:2]
        return (
            [np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)],
            [(text, score)],
        )


# ── LightOnOCR Facade (API-first + local fallback) ────────────────────────────

class LightOnOCR:
    """
    Facade OCR: thử API trước, fallback sang local nếu API không khả dụng.

    Config qua env vars (xem mineru/utils/lighton_config.py):
      LLM_SERVICE      → "openai" / "azure" / "local" / ...
      OPENAI_API_KEY   → API key
      OPENAI_API_BASE  → Base URL
      OPENAI_MODEL     → Model name

    Vietnamese table: tự động phát hiện và dùng prompt tiếng Việt chuyên biệt.
    """

    _api_failed = False
    _shared_local_model = None

    def __init__(
        self,
        server_url: str = None,
        model_name: str = None,
        timeout: int = 180,
        **kwargs,
    ):
        from mineru.utils.lighton_config import get_lighton_config

        self._cfg = get_lighton_config()
        self.drop_score = kwargs.get("drop_score", 0.5)

        # ── Khởi tạo API backend ──────────────────────────────────────────────
        if self._cfg.use_api:
            try:
                self._api = LightOnOCRAPI(
                    cfg=self._cfg,
                    server_url=server_url,
                    model_name=model_name,
                    timeout=timeout,
                    **kwargs,
                )
                logger.debug("[LightOnOCR] API backend initialized.")
            except Exception as e:
                logger.warning(f"[LightOnOCR] API backend init failed: {e}")
                self._api = None
        else:
            self._api = None

        # ── Khởi tạo local backend (lazy) ────────────────────────────────────
        # Chỉ load model khi thực sự cần (tránh chiếm RAM nếu API luôn hoạt động)
        self._local = None
        self._local_kwargs = kwargs

        logger.debug(
            f"[LightOnOCR] service={self._cfg.llm_service!r} "
            f"use_api={self._cfg.use_api} use_local={self._cfg.use_local}"
        )

    # ── Lazy local loader ─────────────────────────────────────────────────────

    def _get_local(self):
        if LightOnOCR._shared_local_model is None:
            if not self._cfg.use_local:
                raise RuntimeError(
                    "[LightOnOCR] Local backend is disabled. "
                    "Set LLM_SERVICE=local or check config."
                )
            logger.info("[LightOnOCR] Loading local model (first use)...")
            from mineru.model.ocr.lighton_model_local import LightOnModelLocal
            LightOnOCR._shared_local_model = LightOnModelLocal(**self._local_kwargs)
        return LightOnOCR._shared_local_model

    # ── Offload ───────────────────────────────────────────────────────────────

    @classmethod
    def offload_local_model(cls):
        """Free the shared local model from memory."""
        if cls._shared_local_model is not None:
            logger.info("[LightOnOCR] Offloading shared local model...")
            cls._shared_local_model.offload()
            cls._shared_local_model = None

    def offload(self):
        """Instance method to offload the shared local model."""
        self.offload_local_model()

    # ── API availability check ────────────────────────────────────────────────

    def _check_api_alive(self) -> bool:
        """Kiểm tra nhanh xem API có sẵn sàng không (GET /models)."""
        try:
            from mineru.utils.lighton_config import build_api_headers
            base = self._cfg.api_base.rstrip("/")
            r = requests.get(
                f"{base}/models",
                headers=build_api_headers(self._cfg),
                timeout=2,
            )
            return r.status_code == 200
        except Exception:
            return False

    # ── Vietnamese table detection ────────────────────────────────────────────

    def _should_use_vietnamese_prompt(
        self,
        image: Union[np.ndarray, Image.Image],
        lang: Optional[str] = None,
        auto_detect: bool = True,
    ) -> bool:
        """
        Trả về True nếu nên dùng prompt tiếng Việt chuyên biệt.
        - Nếu lang bắt đầu bằng 'vi' → True ngay
        - Nếu auto_detect=True → scan nhanh ảnh để phát hiện ký tự tiếng Việt
        """
        if lang and lang.lower().startswith("vi"):
            return True
        if auto_detect:
            return _is_vietnamese_content(image)
        return False

    # ── Generic fallback dispatcher ───────────────────────────────────────────

    def _with_fallback(self, api_fn, local_fn):
        """
        Thử gọi api_fn. Nếu fail (exception hoặc không có API) → gọi local_fn.
        """
        if self._api is not None and not getattr(LightOnOCR, '_api_failed', False):
            try:
                return api_fn()
            except Exception as e:
                logger.warning(f"[LightOnOCR] API failed ({e}). Đã tắt API cho các block tiếp theo, chuyển sang dùng local.")
                LightOnOCR._api_failed = True

        if self._cfg.use_local:
            return local_fn()

        raise RuntimeError(
            "[LightOnOCR] API failed và local backend bị tắt. "
            "Kiểm tra kết nối đến API hoặc set LLM_SERVICE=local."
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def recognize_text(
        self,
        image: Union[np.ndarray, Image.Image],
        *,
        page_img: Optional[np.ndarray] = None,
        poly: Optional[list] = None,
        vietnamese: Optional[bool] = None,
        lang: Optional[str] = None,
    ) -> Tuple[str, float]:
        if vietnamese is None and lang is None:
            vietnamese = self._should_use_vietnamese_prompt(image)
        return self._with_fallback(
            api_fn   = lambda: self._api.recognize_text(image, page_img=page_img, poly=poly, vietnamese=vietnamese, lang=lang),
            local_fn = lambda: self._get_local().recognize_text(image, page_img=page_img, poly=poly),
        )

    def recognize_table(
        self,
        image: Union[np.ndarray, Image.Image],
        bbox_coords=None,
        *,
        page_img: Optional[np.ndarray] = None,
        poly: Optional[list] = None,
        lang: Optional[str] = None,
        vietnamese: Optional[bool] = None,   # None = auto-detect
        skeleton_html: str = "",
    ) -> str:
        # Auto-detect Vietnamese nếu không chỉ định tường minh
        if vietnamese is None:
            vietnamese = self._should_use_vietnamese_prompt(image, lang=lang, auto_detect=True)
            if vietnamese:
                logger.info("[LightOnOCR] Detected Vietnamese table → using VI prompt")

        return self._with_fallback(
            api_fn   = lambda: self._api.recognize_table(
                image, bbox_coords, page_img=page_img, poly=poly, vietnamese=vietnamese, lang=lang or "", skeleton_html=skeleton_html
            ),
            local_fn = lambda: self._get_local().recognize_table(
                image, bbox_coords, page_img=page_img, poly=poly, vietnamese=vietnamese, skeleton_html=skeleton_html
            ),
        )

    def recognize_page(self, image: Union[np.ndarray, Image.Image]) -> str:
        """Dịch ảnh toàn trang sang Markdown sử dụng VLM API."""
        return self._with_fallback(
            api_fn   = lambda: self._api.recognize_page(image),
            local_fn = lambda: "",  # local model chưa hỗ trợ OCR full trang
        )

    def ocr(
        self,
        img,
        det=True,
        rec=True,
        mfd_res=None,
        tqdm_enable=False,
        tqdm_desc="OCR-rec Predict",
        **kwargs,
    ) -> List:
        page_img   = kwargs.get("page_img")
        poly       = kwargs.get("poly")
        vietnamese = kwargs.get("vietnamese")   # None = auto

        imgs = [img] if isinstance(img, np.ndarray) else img
        is_table = "table" in (tqdm_desc or "").lower()

        if getattr(LightOnOCR, '_api_failed', False) or getattr(self, '_api', None) is None:
            try:
                local_name = self._get_local()._model_id.split('/')[-1]
                if " " in tqdm_desc:
                    tqdm_desc = f"{local_name} " + tqdm_desc.split(" ", 1)[1]
                else:
                    tqdm_desc = f"{local_name} {tqdm_desc}"
            except Exception:
                pass

        if tqdm_enable:
            from tqdm import tqdm
            iterator = tqdm(imgs, desc=tqdm_desc)
        else:
            iterator = imgs

        def update_desc_if_local():
            if tqdm_enable and (getattr(LightOnOCR, '_api_failed', False) or getattr(self, '_api', None) is None):
                try:
                    local_name = self._get_local()._model_id.split('/')[-1]
                    if " " in iterator.desc:
                        new_desc = f"{local_name} " + iterator.desc.split(" ", 1)[1]
                    else:
                        new_desc = f"{local_name} {iterator.desc}"
                    iterator.set_description(new_desc)
                except Exception:
                    pass

        if is_table:
            results = []
            for image in iterator:
                update_desc_if_local()
                viet = vietnamese
                if viet is None:
                    viet = self._should_use_vietnamese_prompt(image, auto_detect=True)
                    if viet:
                        logger.info("[LightOnOCR.ocr] Vietnamese table detected")
                res = self.recognize_table(image, page_img=page_img, poly=poly, vietnamese=viet)
                results.append((res, 1.0))
            return [results]

        ocr_res = []
        if det and rec:
            for image in iterator:
                update_desc_if_local()
                text, score = self.recognize_text(image, page_img=page_img, poly=poly, vietnamese=vietnamese, lang=kwargs.get('lang'))
                if text:
                    h, w = (
                        image.shape[:2]
                        if isinstance(image, np.ndarray)
                        else image.size[::-1]
                    )
                    box = [[0, 0], [w, 0], [w, h], [0, h]]
                    ocr_res.append([[box, (text, score)]])
                else:
                    ocr_res.append(None)
        elif not det and rec:
            res_list = []
            for image in iterator:
                update_desc_if_local()
                res_list.append(self.recognize_text(image, page_img=page_img, poly=poly, vietnamese=vietnamese, lang=kwargs.get('lang')))
            ocr_res.append(res_list)

        return ocr_res

    def __call__(
        self,
        img: np.ndarray,
        mfd_res: List = None,
        *,
        page_img: Optional[np.ndarray] = None,
        poly: Optional[list] = None,
    ) -> Tuple[List, List]:
        if img is None:
            return None, None
        text, score = self.recognize_text(img, page_img=page_img, poly=poly)
        if not text:
            return None, None
        h, w = img.shape[:2]
        return (
            [np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)],
            [(text, score)],
        )