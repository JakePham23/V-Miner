# Copyright (c) Opendatalab. All rights reserved.
"""
LightOnOCR Local Model Runner.

Chạy model LightOnOCR trực tiếp trên máy (không cần server).

Backends được hỗ trợ:
  - mlx       : Apple Silicon (M1/M2/M3/M4) — nhanh, ít RAM
                Model: mlx-community/LightOnOCR-2-1B-bf16
                Cài:   pip install mlx-lm
  - transformers: Cross-platform fallback (CPU / CUDA)
                Model: lightonai/LightOnOCR-2-1B (float16)
                Cài:   pip install transformers torch

Model tự động tải về HuggingFace cache (~/.cache/huggingface) lần đầu.
"""
from __future__ import annotations

import os
# if "HF_ENDPOINT" not in os.environ:
#     os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


import base64
import re
import sys
from io import BytesIO
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from loguru import logger


# ── Constants (dùng chung với lighton_ocr.py) ─────────────────────────────────
_PAD_RATIO   = 0.04
_PAD_MIN_PX  = 6
_PAD_MAX_PX  = 20
_MIN_DIM_PX  = 320   # Giảm từ 640 xuống 320 để tiết kiệm RAM cho các crop nhỏ
_MAX_DIM_PX  = 1280  # Giảm từ 2048 xuống 1280 (giảm ~60% lượng token vision)

# HuggingFace model IDs
_MLX_MODEL_ID  = "mlx-community/LightOnOCR-2-1B-4bit"
_HF_MODEL_ID   = "lightonai/LightOnOCR-2-1B"


# ── Image helpers ─────────────────────────────────────────────────────────────

def _to_pil_rgb(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)
    raise TypeError(f"Unsupported image type: {type(image)}")


def _preprocess_pil(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    short, long_ = min(w, h), max(w, h)
    if short < _MIN_DIM_PX:
        scale = _MIN_DIM_PX / short
        new_w, new_h = int(w * scale), int(h * scale)
        if max(new_w, new_h) > _MAX_DIM_PX:
            scale = _MAX_DIM_PX / long_
            new_w, new_h = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    elif long_ > _MAX_DIM_PX:
        scale = _MAX_DIM_PX / long_
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return pil_img


# ── Post-process helpers (same as API version) ────────────────────────────────

_PROMPT_ECHO_PATTERNS = [
    "Extract all text from this image accurately.",
    "Extract the table from this image.",
    "Output ONLY",
    "Do not add any explanation",
    "Đây là bảng tiếng Việt",
]


def _clean_response(text: str) -> str:
    if not text:
        return ""

    t = text.strip()
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


# ── MLX backend ───────────────────────────────────────────────────────────────

class _MLXRunner:
    """Thin wrapper around mlx-vlm for LightOnOCR inference."""

    def __init__(self, model_id: str):
        try:
            from mlx_vlm import load  # type: ignore
        except ImportError:
            raise ImportError(
                "mlx-vlm not installed. Run: pip install mlx-vlm"
            )
        logger.info(f"[Local/MLX] Loading model {model_id!r} ...")
        self._model, self._processor = load(model_id)
        logger.info("[Local/MLX] Model loaded.")

    def generate(self, prompt: str, image: Image.Image, max_tokens: int = 16384) -> str:
        """Generate text from a multimodal prompt+image using mlx-vlm."""
        try:
            from mlx_vlm import generate  # type: ignore
        except ImportError:
            raise ImportError("mlx-vlm not installed.")

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}]
        formatted_prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)

        res = generate(
            self._model,
            self._processor,
            prompt=formatted_prompt,
            image=image,
            max_tokens=max_tokens,
            verbose=False,
        )
        if hasattr(res, "text"):
            return res.text
        return str(res)

    def offload(self):
        """Free memory by deleting model and processor, clearing MLX cache."""
        self._model = None
        self._processor = None
        try:
            import mlx.core as mx
            if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except ImportError:
            pass
        import gc
        gc.collect()


# ── Transformers backend ──────────────────────────────────────────────────────

class _TransformersRunner:
    """Thin wrapper around HuggingFace transformers for LightOnOCR inference."""

    def __init__(self, model_id: str):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError:
            raise ImportError(
                "transformers and torch not installed. "
                "Run: pip install transformers torch"
            )

        logger.info(f"[Local/Transformers] Loading model {model_id!r} ...")
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
        ).to(self._device)
        logger.info(f"[Local/Transformers] Model loaded on {self._device}.")

    def generate(self, prompt: str, image: Image.Image, max_tokens: int = 16384) -> str:
        import torch

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}]
        formatted_prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self._processor(
            text=formatted_prompt,
            images=image,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[:, input_len:]
        output = self._processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        return output

    def offload(self):
        """Free memory by deleting model and emptying CUDA cache."""
        self._model = None
        self._processor = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        import gc
        gc.collect()


# ── Main local class ──────────────────────────────────────────────────────────

class LightOnModelLocal:
    """
    Run LightOnOCR model locally (no server needed).

    Backends:
      "mlx"          → mlx-lm (Apple Silicon, fast)
      "transformers" → HuggingFace transformers (CPU / CUDA)

    Usage:
        local = LightOnModelLocal()           # auto-detect backend
        text, score = local.recognize_text(image)
        html  = local.recognize_table(image)
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        backend: Optional[str] = None,
        **kwargs,
    ):
        from mineru.utils.lighton_config import get_lighton_config, _default_local_backend

        cfg = get_lighton_config()
        self._model_id = model_id or cfg.local_model_id
        self._backend  = backend  or cfg.local_backend
        self.drop_score = kwargs.get("drop_score", 0.5)

        logger.info(
            f"[LightOnModelLocal] backend={self._backend!r}, model={self._model_id!r}"
        )
        self._runner = self._init_runner()

    def _init_runner(self):
        if self._backend == "mlx":
            try:
                return _MLXRunner(self._model_id)
            except ImportError as e:
                logger.warning(f"[Local] MLX unavailable ({e}), falling back to transformers")
                self._backend = "transformers"
                hf_id = _HF_MODEL_ID if "mlx-community" in self._model_id else self._model_id
                return _TransformersRunner(hf_id)
        elif self._backend == "transformers":
            # Use HF float model id (not mlx-community quantized)
            hf_id = _HF_MODEL_ID if "mlx-community" in self._model_id else self._model_id
            return _TransformersRunner(hf_id)
        else:
            raise ValueError(f"Unknown local backend: {self._backend!r}. Use 'mlx' or 'transformers'.")

    def offload(self):
        """Offload the local model to free memory."""
        if self._runner is not None and hasattr(self._runner, "offload"):
            logger.info(f"[LightOnModelLocal] Offloading {self._backend} model to free memory...")
            self._runner.offload()
            self._runner = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _prepare_image(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        pil = _to_pil_rgb(image)
        return _preprocess_pil(pil)

    def _infer(self, image: Union[np.ndarray, Image.Image], prompt: str) -> str:
        pil = self._prepare_image(image)
        try:
            raw = self._runner.generate(prompt, pil)
            return _clean_response(raw)
        except Exception as e:
            logger.error(f"[LightOnModelLocal] Inference error: {e}")
            return ""

    # ── Public API (mirrors LightOnOCR interface) ─────────────────────────────

    def recognize_text(
        self,
        image: Union[np.ndarray, Image.Image],
        *,
        page_img: Optional[np.ndarray] = None,
        poly: Optional[list] = None,
    ) -> Tuple[str, float]:
        if page_img is not None and poly is not None:
            from mineru.model.ocr.lighton_ocr import crop_for_lighton
            image = crop_for_lighton(poly, page_img)

        prompt = (
            "Extract all text from this image accurately. "
            "Output ONLY the raw extracted text. "
            "DO NOT output any notes, explanations, comments, or markdown code blocks."
        )
        text = self._infer(image, prompt)
        return text, (0.95 if text else 0.0)

    def recognize_table(
        self,
        image: Union[np.ndarray, Image.Image],
        bbox_coords=None,
        *,
        page_img: Optional[np.ndarray] = None,
        poly: Optional[list] = None,
        vietnamese: bool = False,
        skeleton_html: str = "",
    ) -> str:
        if page_img is not None and poly is not None:
            from mineru.model.ocr.lighton_ocr import crop_for_lighton
            image = crop_for_lighton(poly, page_img)

        if skeleton_html and "<table>" in skeleton_html:
            prompt = (
                "Dưới đây là một ảnh chứa bảng và cấu trúc HTML khung của bảng đó đã được dựng sẵn (skeleton). "
                "Hãy nhìn vào ảnh, đọc các chữ tiếng Việt và ĐIỀN ĐÚNG các chữ đó vào các ô <td> tương ứng trong khung HTML này. "
                "Giữ nguyên cấu trúc thẻ <tr>, <td>, rowspan, colspan của khung HTML (trừ khi sai khác quá lớn so với ảnh). "
                "Đảm bảo chính xác dấu thanh tiếng Việt (á, à, ả, ã, ạ, ắ, ặ, ẳ, ẵ, ề, ế, ệ, ể, ễ, ổ, ỗ, ộ, v.v.). "
                "Output ONLY a valid HTML <table>...</table>, KHÔNG kèm giải thích.\n"
                f"SKELETON HTML:\n{skeleton_html}"
            )
        elif vietnamese:
            prompt = (
                "Đây là bảng tiếng Việt. Trích xuất toàn bộ nội dung bảng. "
                "Đảm bảo chính xác dấu thanh tiếng Việt (á, à, ả, ã, ạ, v.v.). "
                "Output ONLY a valid HTML <table> with <thead> and <tbody>. "
                "Do not add any explanation or preamble."
            )
        else:
            prompt = (
                "Extract the table from this image. "
                "Output ONLY a valid HTML <table> with <thead> and <tbody>. "
                "Do not add any explanation or preamble."
            )

        result = self._infer(image, prompt)

        # Post-process: extract <table>...</table>
        if result:
            start = result.find("<table")
            end   = result.rfind("</table>")
            if start != -1 and end != -1:
                result = result[start: end + len("</table>")]
            elif "<table" not in result:
                logger.warning("[LightOnModelLocal] recognize_table: no <table> tag, returning plain text")
                return result

        return result

    def ocr(
        self,
        img,
        det: bool = True,
        rec: bool = True,
        mfd_res=None,
        tqdm_enable: bool = False,
        tqdm_desc: str = "OCR-rec Predict",
        **kwargs,
    ) -> List:
        """Interface tương thích với PaddleOCR / LightOnOCR."""
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


# ── Singleton loader (lazy) ───────────────────────────────────────────────────

_local_instance: Optional["LightOnModelLocal"] = None


def get_local_model(**kwargs) -> "LightOnModelLocal":
    """Return a shared LightOnModelLocal instance (loaded once)."""
    global _local_instance
    if _local_instance is None:
        _local_instance = LightOnModelLocal(**kwargs)
    return _local_instance
