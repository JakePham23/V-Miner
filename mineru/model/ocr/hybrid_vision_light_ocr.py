"""
Hybrid Vision + LightOnOCR wrapper for MinerU pipeline.
Uses Apple Vision Framework for text recognition and LightOnOCR for table extraction.
Best accuracy combination: Vision for Vietnamese text + LightOnOCR for complex tables.
"""
import numpy as np
from loguru import logger
from typing import Union, List
from PIL import Image
import sys


class HybridVisionLightOCR:
    """
    Hybrid OCR that combines:
    - Apple Vision Framework: For accurate Vietnamese text recognition
    - LightOnOCR: For table extraction via LM Studio
    """
    
    def __init__(self, **kwargs):
        """Initialize both Vision Framework and LightOnOCR backends."""
        self.is_table_mode = False
        
        # Check if running on macOS
        if sys.platform != 'darwin':
            raise RuntimeError("Vision Framework is only available on macOS")
        
        # Initialize Vision Framework OCR for text
        logger.info("Initializing Vision Framework for text recognition")
        from mineru.model.ocr.vision_ocr import VisionFrameworkOCR
        self.vision_ocr = VisionFrameworkOCR(**kwargs)
        
        # Initialize LightOnOCR for tables
        logger.info("Initializing LightOnOCR for table processing")
        import os
        from mineru.model.ocr.lighton_ocr import LightOnOCR
        self.light_ocr = LightOnOCR(
            server_url=os.getenv('LIGHTON_SERVER_URL', 'http://localhost:1234/v1/chat/completions'),
            model_name=os.getenv('LIGHTON_MODEL_NAME', 'lightonocr'),
            **kwargs
        )
        
        logger.info("HybridVisionLightOCR initialized successfully")
    
    def set_table_mode(self, is_table: bool):
        """Set whether we're processing tables (use LightOnOCR) or text (use Vision)."""
        self.is_table_mode = is_table
    
    def ocr(self, img, det=True, rec=True, mfd_res=None, tqdm_enable=False, tqdm_desc="OCR-rec Predict", **kwargs):
        """
        Perform OCR on image(s).
        Routes to Vision Framework for text, LightOnOCR for tables.
        
        Args:
            img: Input image (numpy array, list of arrays, or bytes)
            det: Whether to perform detection
            rec: Whether to perform recognition
            mfd_res: Math formula detection results
            tqdm_enable: Show progress bar
            tqdm_desc: Progress bar description
            
        Returns:
            List of OCR results in MinerU format
        """
        # Check if this is table OCR based on the description
        is_table_ocr = 'table' in tqdm_desc.lower() if tqdm_desc else False
        
        if is_table_ocr or self.is_table_mode:
            # Use LightOnOCR for tables
            logger.debug("Using LightOnOCR for table processing")
            return self.light_ocr.ocr(img, det, rec, mfd_res, tqdm_enable, tqdm_desc, **kwargs)
        else:
            # Use Vision Framework for regular text
            logger.debug("Using Vision Framework for text recognition")
            return self.vision_ocr.ocr(img, det, rec, mfd_res, tqdm_enable, tqdm_desc, **kwargs)
    
    def __call__(self, img, mfd_res=None):
        """
        Simplified interface for compatibility.
        
        Args:
            img: Input image
            mfd_res: Math formula detection results
            
        Returns:
            (dt_boxes, rec_res): Tuple of detection boxes and recognition results
        """
        # Use Vision Framework for regular text (default call is for text, not tables)
        return self.vision_ocr(img, mfd_res)
    
    def recognize_table(self, image: Union[np.ndarray, Image.Image]) -> str:
        """
        Recognize table and return HTML format.
        Delegates to LightOnOCR.
        
        Args:
            image: Input image containing a table
            
        Returns:
            HTML string representation of the table
        """
        return self.light_ocr.recognize_table(image)
    
    def recognize_text(self, image: Union[np.ndarray, Image.Image]) -> tuple:
        """
        Recognize text using Vision Framework.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (text, confidence)
        """
        return self.vision_ocr.recognize_text(image)


if __name__ == '__main__':
    import sys
    import cv2
    
    if len(sys.argv) > 1:
        test_img_path = sys.argv[1]
        
        if sys.platform != 'darwin':
            print("Error: This OCR backend only works on macOS")
            sys.exit(1)
            
        hybrid_ocr = HybridVisionLightOCR()
        img = cv2.imread(test_img_path)
        
        dt_boxes, rec_res = hybrid_ocr(img)
        
        print(f"Found {len(dt_boxes) if dt_boxes else 0} text regions")
        if rec_res:
            for i, (text, conf) in enumerate(rec_res):
                print(f"{i+1}. {text} (confidence: {conf:.3f})")
    else:
        print("Usage: python hybrid_vision_light_ocr.py <image_path>")
