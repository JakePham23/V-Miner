import os
import requests
from loguru import logger

def is_lighton_available():
    """
    Kiểm tra xem LM Studio (LightOnOCR) có đang chạy không.
    """
    server_url = os.getenv('LIGHTON_SERVER_URL', 'http://localhost:1234/v1/models')
    try:
        response = requests.get(server_url, timeout=1)
        return response.status_code == 200
    except Exception:
        return False

def should_use_lighton(lang):
    """
    Chỉ sử dụng LightOn nếu là tiếng Việt và server sẵn sàng.
    """
    if not lang or not lang.startswith('vi'):
        logger.info(f"LightOnOCR: Bỏ qua (Ngôn ngữ '{lang}' không phải Tiếng Việt)")
        return False
        
    if is_lighton_available():
        return True
    else:
        logger.warning(f"LightOnOCR: Bỏ qua (Không kết nối được server LM Studio tại localhost:1234)")
        return False
