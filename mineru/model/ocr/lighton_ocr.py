# Copyright (c) Opendatalab. All rights reserved.
"""
LightOnOCR client for MinerU integration.
API-only implementation (Ollama, LM Studio, OpenAI compatible).

CHANGES vs original:
- Added crop_for_lighton() helper: minimal adaptive padding thay vì 50px cứng
- recognize_text / recognize_table nhận thêm tham số page_img + poly để
  tự crop từ ảnh gốc, tránh double-padding từ crop_img() của pipeline
- Nếu không truyền page_img thì fallback về hành vi cũ (tương thích ngược)
- Thêm _preprocess_image(): scale nhỏ lên ≥ 640px để model đọc rõ hơn
- recognize_table: thêm post-process kiểm tra <table> tag hợp lệ
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


# ── Constants ────────────────────────────────────────────────────────────────
# Padding tỉ lệ theo kích thước bbox thay vì giá trị cứng
_PAD_RATIO   = 0.04   # 4% cạnh ngắn hơn của bbox
_PAD_MIN_PX  = 6      # tối thiểu 6px
_PAD_MAX_PX  = 20     # tối đa 20px (thay vì 50px)

# Scale nhỏ lên để model đọc rõ hơn
_MIN_DIM_PX  = 640    # cạnh ngắn nhất sau scale
_MAX_DIM_PX  = 2048   # cạnh dài nhất tối đa (tránh quá nặng)


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
                grid[row_idx][col] = text
                for r_off in range(1, rowspan):
                    occupied[(row_idx + r_off, col)] = text

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
]

_ARTIFACT_PATTERNS = [
    (re.compile(r'^\$\^ \+\$\s*$', re.MULTILINE), ''),
    (re.compile(r'^#\s*$', re.MULTILINE), ''),
    (re.compile(r'\n{3,}'), '\n\n'),
]

def _clean_ocr_response(text: str, prompt: str = "") -> str:
    """Xóa phần prompt bị model echo lại và các artifacts trong response."""
    if not text:
        return text
    t = text.strip()
    if prompt and t.startswith(prompt.strip()):
        t = t[len(prompt.strip()):].strip()

    # Xóa prompt echo từng dòng
    lines = t.splitlines()
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        is_prompt = any(stripped.startswith(p[:30]) for p in _PROMPT_ECHO_PATTERNS)
        if not is_prompt:
            clean_lines.append(line)
    t = "\n".join(clean_lines).strip()

    # Xóa các artifacts đặc thù
    for pattern, replacement in _ARTIFACT_PATTERNS:
        t = pattern.sub(replacement, t)

    return t.strip()


# ── Smart crop helper ────────────────────────────────────────────────────────

def crop_for_lighton(
    poly: list,
    page_img: np.ndarray,
    pad_ratio: float = _PAD_RATIO,
    pad_min: int = _PAD_MIN_PX,
    pad_max: int = _PAD_MAX_PX,
) -> np.ndarray:
    """
    Crop vùng bbox từ ảnh trang gốc với padding tỉ lệ nhỏ.

    Dùng thay thế cho crop_img() của pipeline (vốn dùng 50px cứng).
    poly: list 8 giá trị [x0,y0, x1,y1, x2,y2, x3,y3] hoặc
          list 4 giá trị [x0,y0, x1,y1] (xmin,ymin,xmax,ymax).
    page_img: numpy array RGB của toàn trang.
    """
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

    # Padding tỉ lệ theo cạnh ngắn hơn
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
        return page_img  # fallback ảnh toàn trang

    return cropped


def _preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Scale ảnh lên nếu quá nhỏ để model đọc rõ hơn.
    Scale xuống nếu quá lớn để giảm payload.
    Giữ nguyên aspect ratio.
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image

    short, long_ = min(h, w), max(h, w)

    # Cần scale lên?
    if short < _MIN_DIM_PX:
        scale = _MIN_DIM_PX / short
        new_w = int(w * scale)
        new_h = int(h * scale)
        # Nhưng không để cạnh dài vượt MAX
        if max(new_w, new_h) > _MAX_DIM_PX:
            scale = _MAX_DIM_PX / long_
            new_w = int(w * scale)
            new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        logger.debug(f"_preprocess_image: scaled up {w}x{h} → {new_w}x{new_h}")

    # Cần scale xuống?
    elif long_ > _MAX_DIM_PX:
        scale = _MAX_DIM_PX / long_
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug(f"_preprocess_image: scaled down {w}x{h} → {new_w}x{new_h}")

    return image


# ── Main class ────────────────────────────────────────────────────────────────

class LightOnOCR:
    """OCR client using API backends (OpenAI/Ollama compatible)."""

    def __init__(
        self,
        server_url: str = None,
        model_name: str = None,
        timeout: int = 180,
        **kwargs
    ):
        self.server_url = (
            server_url
            or os.getenv("LIGHTON_SERVER_URL", "http://localhost:1234/v1/chat/completions")
        )
        self.model_name = (
            model_name
            or os.getenv("LIGHTON_MODEL_NAME", "lightonocr")
        )
        self.timeout   = timeout
        self.drop_score = kwargs.get("drop_score", 0.5)
        logger.info(f"Initialized LightOnOCR API: URL={self.server_url}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _to_numpy_rgb(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Chuẩn hóa về numpy RGB."""
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:                      # grayscale
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            if image.shape[2] == 4:                        # RGBA
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            if image.shape[2] == 3:
                # Giả sử BGR (OpenCV convention) → RGB
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _image_to_base64(self, image: Union[np.ndarray, Image.Image]) -> str:
        rgb = self._to_numpy_rgb(image)
        rgb = _preprocess_image(rgb)        # scale nếu cần
        pil = Image.fromarray(rgb)
        buffered = BytesIO()
        pil.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _call_api(self, image: Union[np.ndarray, Image.Image], prompt: str) -> str:
        img_b64 = self._image_to_base64(image)

        if "/api/" in self.server_url:
            # Ollama style
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
                "stream": False,
            }
            try:
                r = requests.post(self.server_url, json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    res  = r.json()
                    text = res.get("response", res.get("message", {}).get("content", ""))
                    if not text:
                        logger.warning(f"LightOnOCR Ollama: response 200 nhưng content rỗng. raw={res}")
                    return text
                else:
                    logger.error(f"LightOnOCR Ollama: HTTP {r.status_code} - {r.text[:200]}")
            except Exception as e:
                logger.error(f"LightOnOCR Ollama error: {e}")
        else:
            # LM Studio / OpenAI style
            image_url = f"data:image/jpeg;base64,{img_b64}"
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
                "temperature": 0.0,
                "max_tokens":  4096,
            }
            try:
                r = requests.post(self.server_url, json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    text = r.json()["choices"][0]["message"]["content"]
                    if not text:
                        logger.warning(f"LightOnOCR OpenAI: response 200 nhưng content rỗng. raw={r.json()}")
                    return text
                else:
                    logger.error(f"LightOnOCR OpenAI: HTTP {r.status_code} - {r.text[:200]}")
            except Exception as e:
                logger.error(f"LightOnOCR OpenAI error: {e}")
        return ""

    # ── Public API ────────────────────────────────────────────────────────────

    def recognize_text(
        self,
        image: Union[np.ndarray, Image.Image],
        *,
        page_img: Optional[np.ndarray] = None,
        poly: Optional[list] = None,
    ) -> Tuple[str, float]:
        """
        Nhận diện text.

        Nếu truyền page_img + poly: tự crop từ ảnh gốc với padding nhỏ.
        Nếu không: dùng image đã truyền vào (hành vi cũ, tương thích ngược).
        """
        if page_img is not None and poly is not None:
            image = crop_for_lighton(poly, page_img)

        prompt = (
            "Extract all text from this image accurately. "
            "Output only the extracted text, preserving line breaks. "
            "Ensure Vietnamese diacritics are accurate."
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
    ) -> str:
        """
        Trích xuất bảng, trả về HTML table string.

        Nếu truyền page_img + poly: tự crop từ ảnh gốc với padding nhỏ.
        bbox_coords giữ lại để tương thích ngược (không dùng).
        """
        if page_img is not None and poly is not None:
            image = crop_for_lighton(poly, page_img)

        prompt = (
            "Extract the table from this image. "
            "Output ONLY a valid HTML <table> with <thead> and <tbody>. "
            "Do not add any explanation or preamble. "
            "Ensure Vietnamese text is accurate."
        )
        result = self._call_api(image, prompt)
        result = _clean_ocr_response(result, prompt).strip()

        # ── Post-process: đảm bảo output là HTML table hợp lệ ──────────────
        if result:
            # Cắt lấy phần <table>...</table> nếu model output thêm text thừa
            start = result.find("<table")
            end   = result.rfind("</table>")
            if start != -1 and end != -1:
                result = result[start : end + len("</table>")]
            elif "<table" not in result:
                # Model trả về plain text thay vì HTML → trả về text thuần thay vì wrap table
                logger.warning(
                    "LightOnOCR recognize_table: output không chứa <table>, "
                    f"trả về văn bản thuần. preview={repr(result[:80])}"
                )
                return result

            # Chuyển đổi sang Markdown chuẩn
            result = html_table_to_markdown(result)
            # Fix flat multi-row headers (nếu có)
            result = _fix_flat_multirow_header(result)

            logger.debug(
                f"LightOnOCR recognize_table: {len(result)} ký tự (Markdown), "
                f"preview={repr(result[:80])}"
            )
        else:
            logger.warning("LightOnOCR recognize_table: API trả về kết quả rỗng.")

        return result

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
        """
        Interface tương thích với PaddleOCR.
        Thêm tham số page_img / poly qua kwargs để bypass double-cropping.
        """
        page_img = kwargs.get("page_img")
        poly     = kwargs.get("poly")

        imgs = [img] if isinstance(img, np.ndarray) else img
        ocr_res = []
        is_table = "table" in (tqdm_desc or "").lower()

        if is_table:
            return [
                [
                    (
                        self.recognize_table(image, page_img=page_img, poly=poly),
                        1.0,
                    )
                    for image in imgs
                ]
            ]

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