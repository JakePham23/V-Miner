import os
import requests
import re
from loguru import logger

# --- PROMPTS INSPIRED BY MARKER PROJECT ---

PROMPTS = {
    "PAGE_ANALYSIS": """You are a layout analysis expert. You will receive an image of a PDF page and a JSON list of detected blocks (text, tables, images).
Your task is to:
1. Correct the reading order of the blocks.
2. Refine the block types (e.g., change 'Text' to 'SectionHeader' or 'List').
3. Fix any obvious OCR errors in the text content.
4. Output a refined JSON list in the correct reading order.

Return only the JSON list.""",

    "COMPLEX_REGION": """You are a text reconstruction expert. You will receive an image of a complex region (containing forms, tables, or mixed layout).
Your task is to generate high-quality Markdown that faithfully represents the content and structure in the image.
- Use tables for form-like data.
- Use proper headers (#, ##, etc.).
- Maintain bold/italic formatting.
- Ensure no information is lost.

Output only the Markdown content.""",

    "TABLE_REFINE": """Compare the provided image of a table with the initial OCR/HTML output.
Fix any alignment issues, merged cells (colspan/rowspan), or misread characters.
Ensure headers are correctly identified.

Output the corrected Markdown table only."""
}

def construct_vlm_messages(prompt_type, image_base64, extra_context=""):
    """
    Tạo payload messages cho VLM (OpenAI-compatible) theo phong cách Marker.
    """
    prompt = PROMPTS.get(prompt_type, PROMPTS["COMPLEX_REGION"])
    if extra_context:
        prompt += f"\n\nContext/Initial Text:\n{extra_context}"
        
    messages = [
        {
            "role": "system",
            "content": "You are a professional PDF-to-Markdown conversion assistant."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        }
    ]
    return messages

def clean_markdown_response(text):
    """
    Làm sạch kết quả Markdown từ LLM (loại bỏ block code, khoảng trắng thừa).
    """
    # Loại bỏ ```markdown và ```
    text = re.sub(r"```markdown\n?", "", text)
    text = re.sub(r"```html\n?", "", text)
    text = re.sub(r"```\n?", "", text)
    return text.strip()

# --- SURYA LAYOUT INTEGRATION (INSPIRED BY MARKER) ---

class SuryaLayoutWrapper:
    """
    Wrapper để tích hợp Surya Layout giống như dự án Marker.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SuryaLayoutWrapper, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.det_model = None
        return cls._instance

    def load_models(self):
        if self.model is not None:
            return True
        try:
            from surya.layout import LayoutPredictor
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.settings import settings as surya_settings
            
            logger.info("Loading Surya Layout models...")
            self.det_model = DetectionPredictor()
            self.model = LayoutPredictor(FoundationPredictor(checkpoint=surya_settings.LAYOUT_MODEL_CHECKPOINT))
            return True
        except ImportError:
            logger.warning("Surya-ocr không được cài đặt. Vui lòng chạy 'pip install surya-ocr'")
            return False
        except Exception as e:
            logger.error(f"Lỗi khi tải Surya models: {e}")
            return False

    def predict(self, image):
        if not self.load_models():
            return None
        
        # Surya expects a list of images.
        predictions = self.model([image])
        if not predictions:
            return None
            
        layout_result = predictions[0]
        
        # MinerU Category IDs (from YOLO model):
        # 0: title, 1: plain_text, 2: abandon, 3: figure, 4: figure_caption, 
        # 5: table, 6: table_caption, 7: table_footnote, 8: isolate_formula, 9: formula_caption
        SURYA_MAP = {
            "Title": 0,
            "Section-header": 0,
            "Text": 1,
            "List-item": 1,
            "Table": 5,
            "Picture": 3,
            "Figure": 3,
            "Caption": 4,
            "Formula": 8,
            "Equation": 8,
            "Page-header": 2,
            "Page-footer": 2,
            "Footnote": 2,
        }
        
        vminer_blocks = []
        # Support different surya versions (bboxes might be named differently but usually it's bboxes)
        boxes = getattr(layout_result, 'bboxes', [])
        for box in boxes:
            label = getattr(box, 'label', 'Text')
            cat_id = SURYA_MAP.get(label, 1) # Default to text
            
            # Determine image dimensions for coordinate-based filtering
            img_h = image.shape[0] if hasattr(image, 'shape') else image.size[1]
            
            # Extract y-coordinates safely
            if hasattr(box, 'bbox'):
                y_min, y_max = box.bbox[1], box.bbox[3]
            elif hasattr(box, 'polygon'):
                ys = [pt[1] for pt in box.polygon]
                y_min, y_max = min(ys), max(ys)
            else:
                y_min, y_max = 0, img_h
                
            # Aggressively filter out Text blocks that are at the extreme top (<8%) or bottom (>90%)
            # This catches headers/footers (like "Trang 20/21") that Surya misclassifies as regular Text
            is_extreme_edge = (y_min < img_h * 0.08) or (y_max > img_h * 0.90)
            if (cat_id == 2) or (cat_id == 1 and is_extreme_edge):
                continue
                
            # Surya polygon is usually list of lists: [[x,y], [x,y], [x,y], [x,y]]
            poly = []
            if hasattr(box, 'polygon'):
                for pt in box.polygon:
                    poly.extend([float(pt[0]), float(pt[1])])
            else:
                # Fallback if only bbox [x1, y1, x2, y2] is available
                b = box.bbox
                poly = [b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]]
                
            score = 1.0
            if hasattr(box, 'top_k') and box.top_k:
                score = max(box.top_k.values())
                
            vminer_blocks.append({
                "category_id": cat_id,
                "poly": poly,
                "score": float(score)
            })
            
        return vminer_blocks

def analyze_layout_with_llm(image_base64, current_blocks_json=None):
    """
    Sử dụng VLM (Qwen2-VL) để phân tích lại layout (Page Correction)
    đây là tính năng 'VIP' của Marker giúp sửa lỗi cho YOLO/Surya.
    """
    from mineru.model.ocr.lighton_ocr import LightOnOCR
    
    extra_context = ""
    if current_blocks_json:
        extra_context = f"Current Detected Blocks (JSON):\n{current_blocks_json}"
        
    messages = construct_vlm_messages("PAGE_ANALYSIS", image_base64, extra_context)
    
    try:
        import base64, requests, os
        client = LightOnOCR()
        # _call_api expects (image_array, prompt_str) — decode base64 to numpy for it
        import numpy as np
        from PIL import Image
        from io import BytesIO
        img_bytes = base64.b64decode(image_base64)
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        np_img = np.array(pil_img)
        prompt = PROMPTS.get("PAGE_ANALYSIS", PROMPTS["COMPLEX_REGION"])
        if extra_context:
            prompt += f"\n\nContext/Initial Text:\n{extra_context}"
        response = client._call_api(np_img, prompt)
        if response:
            return clean_markdown_response(response)
    except Exception as e:
        logger.error(f"Lỗi khi phân tích layout bằng LLM: {e}")
    return None

# --- EXISTING UTILS ---

def is_lighton_available():
    """
    Kiểm tra xem API service (LightOnOCR) có đang chạy không.
    
    Logic:
    - Nếu OPENAI_API_BASE là URL remote (không phải localhost/127.0.0.1):
      chỉ cần có OPENAI_API_KEY là xem như available (tránh ping vào cloud).
    - Nếu là localhost: ping thử /models endpoint.
    """
    api_base = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
    api_key  = os.getenv("OPENAI_API_KEY", "")

    # Remote cloud API (SiliconFlow, OpenAI, etc.) — không cần ping
    is_remote = not ("localhost" in api_base or "127.0.0.1" in api_base)
    if is_remote:
        if api_key:
            return True
        else:
            logger.warning("LightOnOCR: OPENAI_API_BASE là remote nhưng không có OPENAI_API_KEY")
            return False

    # Localhost: thử ping /models endpoint
    server_url = f"{api_base.rstrip('/')}/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(server_url, headers=headers, timeout=3)
        return response.status_code == 200
    except Exception:
        return False

def should_use_lighton(lang):
    """
    Sử dụng LightOn nếu:
    - Người dùng explicitly set MINERU_TEXT_BACKEND=lighton, HOỊ C
    - Là tiếng Việt và API server sẵn sàng.
    """
    # Override cứng qua env var — không cần check ngon ngữ
    explicit_backend = os.getenv('MINERU_TEXT_BACKEND', '').lower()
    if explicit_backend == 'lighton':
        if is_lighton_available():
            return True
        else:
            logger.warning("LightOnOCR: MINERU_TEXT_BACKEND=lighton nhưng không kết nối được tới API Server")
            return False

    # Chế độ tự động: chỉ dùng lighton cho tiếng Việt nếu server sẵn sàng
    if not lang or not lang.startswith('vi'):
        return False

    if is_lighton_available():
        return True
    else:
        logger.warning("LightOnOCR: Bỏ qua (Không kết nối được tới API Server)")
        return False
