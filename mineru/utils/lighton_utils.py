"""
LightOnOCR utility helpers.

Kiểm tra xem LightOnOCR backend (API hoặc local) có khả dụng không.
"""
import os
import requests
from loguru import logger


def is_api_available() -> bool:
    """
    Kiểm tra xem API endpoint có sẵn sàng không.
    Đọc config từ lighton_config để lấy đúng URL + headers.
    """
    try:
        from mineru.utils.lighton_config import get_lighton_config, build_api_headers
        cfg = get_lighton_config()
        if cfg.llm_service == "local":
            return False   # Service được đặt là local → không dùng API

        base = cfg.api_base.rstrip("/")
        url  = f"{base}/models"
        headers = build_api_headers(cfg)
        response = requests.get(url, headers=headers, timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def is_local_available() -> bool:
    """
    Kiểm tra xem local backend (mlx hoặc transformers) có sẵn không.
    Không load model — chỉ check xem thư viện có installed không.
    """
    try:
        from mineru.utils.lighton_config import get_lighton_config, _default_local_backend
        cfg = get_lighton_config()

        if cfg.local_backend == "mlx":
            import importlib
            return importlib.util.find_spec("mlx_vlm") is not None
        else:
            import importlib
            has_transformers = importlib.util.find_spec("transformers") is not None
            has_torch        = importlib.util.find_spec("torch") is not None
            return has_transformers and has_torch
    except Exception:
        return False


def is_lighton_available() -> bool:
    """
    Kiểm tra tổng hợp: có ít nhất một backend khả dụng không?
    Ưu tiên API, sau đó local.
    """
    if is_api_available():
        logger.debug("[lighton_utils] API backend available.")
        return True
    if is_local_available():
        logger.debug("[lighton_utils] Local backend available.")
        return True
    logger.warning("[lighton_utils] Không có backend nào khả dụng (API down + local not installed).")
    return False


def should_use_lighton(lang: str) -> bool:
    """
    Chỉ sử dụng LightOnOCR nếu:
      1. Ngôn ngữ là tiếng Việt (lang bắt đầu bằng 'vi')
      2. Có ít nhất một backend khả dụng (API hoặc local)
    """
    if not lang or not lang.lower().startswith('vi'):
        logger.debug(f"[lighton_utils] Bỏ qua (lang={lang!r} không phải tiếng Việt)")
        return False

    available = is_lighton_available()
    if not available:
        logger.warning(
            "[lighton_utils] Bỏ qua LightOnOCR: "
            "API không kết nối được và local backend chưa installed."
        )
    return available
