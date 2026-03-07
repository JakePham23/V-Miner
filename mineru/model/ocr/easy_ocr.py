"""
EasyOCR wrapper for MinerU pipeline.
Provides better Vietnamese text recognition compared to PaddleOCR.
"""
import os
import cv2
import numpy as np
from loguru import logger

try:
    import easyocr
except ImportError:
    logger.error("EasyOCR not installed. Please run: pip install easyocr")
    raise


class EasyOCR:
    """EasyOCR wrapper compatible with MinerU OCR interface."""
    
    # Language mapping from MinerU to EasyOCR
    LANG_MAP = {
        'vi': ['vi'],
        'vie': ['vi'],
        'vietnamese': ['vi'],
        'en': ['en'],
        'ch': ['ch_sim'],
        'chinese': ['ch_sim'],
        'ch_sim': ['ch_sim'],
        'ch_tra': ['ch_tra'],
        'chinese_cht': ['ch_tra'],
        'ja': ['ja'],
        'japan': ['ja'],
        'ko': ['ko'],
        'korean': ['ko'],
        'th': ['th'],
        'thai': ['th'],
        'ar': ['ar'],
        'arabic': ['ar'],
        'hi': ['hi'],
        'hindi': ['hi'],
        'latin': ['en'],  # Use English for latin scripts
    }
    
    def __init__(self, lang='vi', device=None, **kwargs):
        """
        Initialize EasyOCR reader.
        
        Args:
            lang: Language code (MinerU format)
            device: 'cpu', 'cuda', 'mps', or None (auto-detect)
            **kwargs: Additional arguments (for compatibility)
        """
        self.lang = lang
        self.drop_score = kwargs.get('drop_score', 0.5)
        
        # Map language
        easy_lang = self.LANG_MAP.get(lang, ['en'])
        
        # Auto-detect device if not specified
        if device is None:
            import sys
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                gpu = True
            elif sys.platform == 'darwin' and torch.backends.mps.is_available():
                # Use MPS on Apple Silicon
                device = 'mps'
                gpu = True
                logger.info("Apple MPS (Metal Performance Shaders) detected, using GPU acceleration")
            else:
                device = 'cpu'
                gpu = False
        else:
            gpu = device != 'cpu' if isinstance(device, str) else False
        
        logger.info(f"Initializing EasyOCR with languages: {easy_lang}, GPU: {gpu}, Device: {device}")
        
        try:
            self.reader = easyocr.Reader(
                easy_lang,
                gpu=gpu,
                download_enabled=True,
                model_storage_directory=os.path.expanduser('~/.EasyOCR/model')
            )
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise
    
    def ocr(self, img, det=True, rec=True, mfd_res=None, tqdm_enable=False, tqdm_desc="OCR-rec Predict", **kwargs):
        """
        Perform OCR on image(s).
        
        Args:
            img: Input image (numpy array, list of arrays, or bytes)
            det: Whether to perform detection (always True for EasyOCR)
            rec: Whether to perform recognition
            mfd_res: Math formula detection results (unused for now)
            tqdm_enable: Show progress bar
            tqdm_desc: Progress bar description
            
        Returns:
            List of OCR results in MinerU format: [[[box, (text, confidence)], ...]]
        """
        # Handle different input types
        if isinstance(img, list):
            # List of cropped images for recognition only
            if not det and rec:
                ocr_res_list = []
                for single_img in img:
                    if isinstance(single_img, bytes):
                        # Decode bytes to image
                        nparr = np.frombuffer(single_img, np.uint8)
                        single_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # EasyOCR expects RGB
                    if len(single_img.shape) == 3 and single_img.shape[2] == 3:
                        rgb_img = cv2.cvtColor(single_img, cv2.COLOR_BGR2RGB)
                    else:
                        rgb_img = single_img
                    
                    # Perform recognition
                    results = self.reader.readtext(rgb_img, detail=1)
                    
                    # Extract text and confidence (ignore bounding boxes for rec-only)
                    if results:
                        # Use the result with highest confidence
                        best_result = max(results, key=lambda x: x[2])
                        text, conf = best_result[1], best_result[2]
                        ocr_res_list.append((text, conf))
                    else:
                        ocr_res_list.append(("", 0.0))
                
                return [ocr_res_list]
            else:
                logger.error("EasyOCR does not support list input with det=True")
                return [[]]
        
        # Single image detection + recognition
        if isinstance(img, bytes):
            nparr = np.frombuffer(img, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB for EasyOCR
        if len(img.shape) == 3 and img.shape[2] == 3:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = img
        
        if det and rec:
            # Full OCR: detection + recognition
            results = self.reader.readtext(rgb_img, detail=1)
            
            # Transform EasyOCR output to MinerU format
            # EasyOCR: [(bbox, text, confidence), ...]
            # MinerU: [[[box, (text, confidence)], ...]]
            ocr_res = []
            for bbox, text, conf in results:
                if conf >= self.drop_score:
                    # Convert bbox to numpy array
                    box = np.array(bbox, dtype=np.float32)
                    ocr_res.append([box.tolist(), (text, conf)])
            
            return [ocr_res]
        
        elif det and not rec:
            # Detection only
            results = self.reader.readtext(rgb_img, detail=1)
            ocr_res = []
            for bbox, _, _ in results:
                box = np.array(bbox, dtype=np.float32)
                ocr_res.append(box.tolist())
            
            return [ocr_res]
        
        else:
            logger.warning("rec=True with det=False not properly supported")
            return [[]]
    
    def __call__(self, img, mfd_res=None):
        """
        Simplified interface for compatibility with PytorchPaddleOCR.
        
        Args:
            img: Input image
            mfd_res: Math formula detection results
            
        Returns:
            (dt_boxes, rec_res): Tuple of detection boxes and recognition results
        """
        ocr_result = self.ocr(img, det=True, rec=True, mfd_res=mfd_res)
        
        if not ocr_result or not ocr_result[0]:
            return None, None
        
        # Split into dt_boxes and rec_res
        dt_boxes = []
        rec_res = []
        
        for item in ocr_result[0]:
            box, (text, conf) = item
            dt_boxes.append(np.array(box, dtype=np.float32))
            rec_res.append((text, conf))
        
        return dt_boxes, rec_res


if __name__ == '__main__':
    # Test EasyOCR
    import sys
    if len(sys.argv) > 1:
        test_img_path = sys.argv[1]
        easy_ocr = EasyOCR(lang='vi')
        img = cv2.imread(test_img_path)
        dt_boxes, rec_res = easy_ocr(img)
        
        print(f"Found {len(dt_boxes) if dt_boxes else 0} text regions")
        if rec_res:
            for i, (text, conf) in enumerate(rec_res):
                print(f"{i+1}. {text} (confidence: {conf:.3f})")
    else:
        print("Usage: python easy_ocr.py <image_path>")
