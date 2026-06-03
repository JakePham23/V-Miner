"""
Hybrid OCR wrapper: LightOnOCR for all content types.
Used for vi-light-ocr and vi-hybrid modes in MinerU.

- Regular text, headers, footers: LightOnOCR (via LM Studio)
- Tables: LightOnOCR via LM Studio (better table understanding)
- Formulas: Default MinerU pipeline
"""
import os
import cv2
import numpy as np
from loguru import logger

from .lighton_ocr import LightOnOCR


class HybridLightOCR:
    """
    OCR backend using LightOnOCR for all content types.
    EasyOCR has been removed — LightOnOCR handles both text and tables.
    """

    def __init__(self, **kwargs):
        self.drop_score = kwargs.get('drop_score', 0.5)
        logger.info("Initializing LightOnOCR for ALL recognition tasks (Text/Table/Image)")
        self.light_ocr = LightOnOCR(**kwargs)
        self.is_table_mode = False

    def set_table_mode(self, is_table):
        """Set whether we're currently processing tables."""
        self.is_table_mode = is_table

    def ocr(self, img, det=True, rec=True, mfd_res=None, tqdm_enable=False, tqdm_desc="OCR-rec Predict", **kwargs):
        """Perform OCR using LightOnOCR."""
        logger.debug(f"HybridLightOCR → LightOnOCR for: {tqdm_desc}")
        return self.light_ocr.ocr(img, det, rec, mfd_res, tqdm_enable, tqdm_desc, **kwargs)

    def __call__(self, img, mfd_res=None):
        return self.light_ocr(img, mfd_res)

    def recognize_table(self, img, bbox_coords=None):
        return self.light_ocr.recognize_table(img, bbox_coords)
