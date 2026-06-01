"""
Hybrid OCR wrapper: EasyOCR for text + LightOnOCR for tables.
Used for vi-light-ocr mode in MinerU.

- Regular text, headers, footers: EasyOCR (better Vietnamese)
- Tables: LightOnOCR via LM Studio (better table understanding)
- Formulas: Default MinerU pipeline
"""
import os
import cv2
import numpy as np
from loguru import logger

# Import both OCR backends
from .easy_ocr import EasyOCR
from .lighton_ocr import LightOnOCR


class HybridLightOCR:
    """
    Hybrid OCR combining EasyOCR and LightOnOCR.
    
    - Uses EasyOCR for general text recognition
    - Uses LightOnOCR only for table processing
    """
    
    def __init__(self, **kwargs):
        """
        Initialize both OCR backends.
        """
        self.drop_score = kwargs.get('drop_score', 0.5)
        
        # Check if we should use LightOn for EVERYTHING
        self.use_lighton_for_all = os.getenv('MINERU_TEXT_BACKEND', 'easyocr').lower() == 'lighton'
        
        if self.use_lighton_for_all:
            logger.info("Initializing LightOnOCR for ALL recognition tasks (Text/Table/Image)")
            self.light_ocr = LightOnOCR(**kwargs)
            self.easy_ocr = None # Không cần khởi tạo EasyOCR để tiết kiệm bộ nhớ
        else:
            # Initialize EasyOCR for text
            logger.info("Initializing EasyOCR for text recognition")
            self.easy_ocr = EasyOCR(lang='vi', **kwargs)
            
            # Initialize LightOnOCR for tables only
            logger.info("Initializing LightOnOCR for table processing")
            self.light_ocr = LightOnOCR(**kwargs)
        
        # Flag to track if we're processing tables
        self.is_table_mode = False
    
    def set_table_mode(self, is_table):
        """Set whether we're currently processing tables."""
        self.is_table_mode = is_table
    
    def ocr(self, img, det=True, rec=True, mfd_res=None, tqdm_enable=False, tqdm_desc="OCR-rec Predict", **kwargs):
        """
        Perform OCR using appropriate backend.
        """
        is_table_ocr = 'table' in tqdm_desc.lower() if tqdm_desc else False
        
        if is_table_ocr or self.is_table_mode or self.use_lighton_for_all:
            # Use LightOnOCR
            logger.debug(f"Using LightOnOCR for {tqdm_desc}")
            return self.light_ocr.ocr(img, det, rec, mfd_res, tqdm_enable, tqdm_desc, **kwargs)
        else:
            # Use EasyOCR
            logger.debug("Using EasyOCR for text recognition")
            return self.easy_ocr.ocr(img, det, rec, mfd_res, tqdm_enable, tqdm_desc, **kwargs)
    
    def __call__(self, img, mfd_res=None):
        """
        Simplified interface for compatibility.
        
        Args:
            img: Input image
            mfd_res: Math formula detection results
            
        Returns:
            (dt_boxes, rec_res): Tuple of detection boxes and recognition results
        """
        # Default to EasyOCR for general calls
        if self.is_table_mode:
            return self.light_ocr(img, mfd_res)
        else:
            return self.easy_ocr(img, mfd_res)
    
    def recognize_table(self, img, bbox_coords=None):
        """
        Process table using LightOnOCR.
        
        Args:
            img: Table image
            bbox_coords: Bounding box coordinates
            
        Returns:
            Table in markdown format
        """
        return self.light_ocr.recognize_table(img, bbox_coords)
