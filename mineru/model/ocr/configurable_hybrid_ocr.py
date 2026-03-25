"""
Configurable Hybrid OCR wrapper for MinerU pipeline.
Allows flexible OCR backend selection for different content types:
- Body text: PaddleOCR / EasyOCR / Vision / LightOnOCR (configurable)
- Tables: configurable (lighton recommended)
- Images: configurable (lighton recommended)
"""
import os
import sys
import numpy as np
from loguru import logger
from typing import Union, List, Optional

# Keys only valid for PaddleOCR init — must be filtered out for other backends
PADDLE_ONLY_KEYS = {'det_db_box_thresh', 'det_db_unclip_ratio', 'enable_merge_det_boxes', 'use_dilation'}


class ConfigurableHybridOCR:
    """
    Hybrid OCR with configurable backends for different content types.
    
    Allows you to choose the best OCR engine for body text while automatically
    using LightOnOCR for tables and images for optimal results.
    """
    
    SUPPORTED_TEXT_BACKENDS = ['paddle', 'easyocr', 'vision', 'lighton', 'paddle-vi']
    SUPPORTED_TABLE_BACKENDS = ['paddle', 'easyocr', 'vision', 'lighton', 'rapidtable', 'default']
    SUPPORTED_IMAGE_BACKENDS = ['paddle', 'easyocr', 'vision', 'lighton']
    
    def __init__(
        self,
        text_backend: str = 'easyocr',
        table_backend: str = 'lighton',
        image_backend: str = 'lighton',
        **kwargs
    ):
        """
        Initialize configurable hybrid OCR.
        
        Args:
            text_backend: OCR backend for body text ('paddle', 'easyocr', 'vision', 'lighton')
            table_backend: OCR backend for tables ('paddle', 'easyocr', 'vision', 'lighton')
            image_backend: OCR backend for images ('paddle', 'easyocr', 'vision', 'lighton')
            **kwargs: Additional arguments passed to OCR engines
        """
        self.text_backend_name = text_backend.lower()
        self.table_backend_name = table_backend.lower()
        self.image_backend_name = image_backend.lower()
        self.kwargs = kwargs
        # Generic kwargs stripped of Paddle-only keys (passed to EasyOCR / LightOnOCR)
        self._generic_kwargs = {k: v for k, v in kwargs.items() if k not in PADDLE_ONLY_KEYS}
        self.is_table_mode = False
        self.is_image_mode = False
        
        # Validate backends
        if self.text_backend_name not in self.SUPPORTED_TEXT_BACKENDS:
            logger.warning(
                f"Unsupported text backend '{text_backend}'. "
                f"Supported: {self.SUPPORTED_TEXT_BACKENDS}. Falling back to 'easyocr'"
            )
            self.text_backend_name = 'easyocr'
        
        if self.table_backend_name not in self.SUPPORTED_TABLE_BACKENDS:
            logger.warning(
                f"Unsupported table backend '{table_backend}'. "
                f"Supported: {self.SUPPORTED_TABLE_BACKENDS}. Falling back to 'lighton'"
            )
            self.table_backend_name = 'lighton'
        
        if self.image_backend_name not in self.SUPPORTED_IMAGE_BACKENDS:
            logger.warning(
                f"Unsupported image backend '{image_backend}'. "
                f"Supported: {self.SUPPORTED_IMAGE_BACKENDS}. Falling back to 'lighton'"
            )
            self.image_backend_name = 'lighton'
        
        # Initialize text backend
        self.text_ocr = self._init_backend(self.text_backend_name, 'text')
        
        # Initialize table backend (if different from text)
        if self.table_backend_name == self.text_backend_name:
            self.table_ocr = self.text_ocr
            logger.info(f"Table backend reusing text backend: {self.table_backend_name}")
        else:
            self.table_ocr = self._init_backend(self.table_backend_name, 'table')
        
        # Initialize image backend (if different from text and table)
        if self.image_backend_name == self.text_backend_name:
            self.image_ocr = self.text_ocr
            logger.info(f"Image backend reusing text backend: {self.image_backend_name}")
        elif self.image_backend_name == self.table_backend_name:
            self.image_ocr = self.table_ocr
            logger.info(f"Image backend reusing table backend: {self.image_backend_name}")
        else:
            self.image_ocr = self._init_backend(self.image_backend_name, 'image')
        
        logger.info(
            f"ConfigurableHybridOCR initialized: "
            f"text={self.text_backend_name}, table={self.table_backend_name}, image={self.image_backend_name}"
        )
    
    def _init_backend(self, backend_name: str, backend_type: str = 'text'):
        """
        Initialize an OCR backend based on configuration.
        
        Args:
            backend_name: Name of the backend ('paddle', 'easyocr', 'vision', 'lighton')
            backend_type: Type of content ('text', 'table', 'image')
        
        Returns:
            OCR instance
        """
        logger.info(f"Initializing {backend_name} for {backend_type}")
        
        if backend_name == 'paddle':
            from .pytorch_paddle import PytorchPaddleOCR
            # Pass all kwargs (including paddle-specific ones) to PaddleOCR
            return PytorchPaddleOCR(**self.kwargs)

        elif backend_name == 'paddle-vi':
            from .simple_paddle import SimplePaddle
            # Uses the simple wrapper for proper Vietnamese support
            return SimplePaddle(lang='vi', **self._generic_kwargs)
            
        elif backend_name == 'easyocr':
            from .easy_ocr import EasyOCR
            # Only pass generic kwargs (no paddle-specific args)
            return EasyOCR(lang='vi', **self._generic_kwargs)
            
        elif backend_name == 'vision':
            # Check if running on macOS
            if sys.platform != 'darwin':
                logger.warning(
                    f"Vision Framework requires macOS for {backend_type}, but running on {sys.platform}. "
                    "Falling back to EasyOCR"
                )
                from .easy_ocr import EasyOCR
                return EasyOCR(lang='vi', **self._generic_kwargs)
            
            from .vision_ocr import VisionFrameworkOCR
            return VisionFrameworkOCR(**self._generic_kwargs)
        
        elif backend_name == 'lighton':
            from .lighton_ocr import LightOnOCR
            # Only pass generic kwargs (no paddle-specific args)
            return LightOnOCR(
                server_url=os.getenv('LIGHTON_SERVER_URL', 'http://localhost:1234/v1/chat/completions'),
                model_name=os.getenv('LIGHTON_MODEL_NAME', 'lightonai/LightOnOCR-2-1B'),
                **self._generic_kwargs
            )
        
        elif backend_name in ['rapidtable', 'default']:
            # Use MinerU's default RapidTable mechanism
            # For OCR text within tables/images, this falls back to PaddleOCR
            # but table structure recognition uses RapidTable (the MinerU default)
            logger.info(f"Using MinerU default (RapidTable + PaddleOCR) for {backend_type}")
            from .pytorch_paddle import PytorchPaddleOCR
            return PytorchPaddleOCR(**self.kwargs)
        
        else:
            # Should never reach here due to validation, but just in case
            logger.error(f"Unknown backend: {backend_name} for {backend_type}, using EasyOCR")
            from .easy_ocr import EasyOCR
            return EasyOCR(lang='vi', **self._generic_kwargs)
    
    @property
    def has_text_detector(self) -> bool:
        """
        Return True if the active text backend supports batch detection
        (i.e. has a .text_detector attribute like PytorchPaddleOCR).
        EasyOCR and LightOnOCR do NOT support this — they do full-image OCR.
        """
        return hasattr(self.text_ocr, 'text_detector')
    
    def full_page_ocr(self, img: np.ndarray, mfd_res=None):
        """
        Perform full-page OCR (detect + recognize in one shot) using the text backend.
        Used by batch_analyze when text backend doesn't support separate det/rec.

        Returns:
            List of (poly, text, score) tuples in a format batch_analyze can consume,
            OR None if no text found.
        """
        dt_boxes, rec_res = self.text_ocr(img, mfd_res)
        if dt_boxes is None or rec_res is None:
            return None
        return list(zip(dt_boxes, rec_res))

    def set_table_mode(self, is_table: bool):
        """
        Set whether we're processing tables.
        
        Args:
            is_table: True if processing tables, False for regular text
        """
        self.is_table_mode = is_table
    
    def set_image_mode(self, is_image: bool):
        """
        Set whether we're processing images.
        
        Args:
            is_image: True if processing images, False for regular text
        """
        self.is_image_mode = is_image
    
    def ocr(
        self,
        img,
        det: bool = True,
        rec: bool = True,
        mfd_res=None,
        tqdm_enable: bool = False,
        tqdm_desc: str = "OCR-rec Predict",
        **kwargs
    ):
        """
        Perform OCR using appropriate backend.
        
        Routes to configured backends for text, tables, or images.
        
        Args:
            img: Input image (numpy array, list of arrays, or bytes)
            det: Whether to perform detection
            rec: Whether to perform recognition
            mfd_res: Math formula detection results
            tqdm_enable: Show progress bar
            tqdm_desc: Progress bar description
            **kwargs: Additional arguments
            
        Returns:
            List of OCR results in MinerU format
        """
        # Detect content type from description
        is_table = 'table' in tqdm_desc.lower() if tqdm_desc else False
        is_image = any(keyword in tqdm_desc.lower() for keyword in ['image', 'figure', 'img']) if tqdm_desc else False
        
        # Route to appropriate backend
        if is_table or self.is_table_mode:
            # Use table backend
            logger.debug(f"Using {self.table_backend_name} for table: {tqdm_desc}")
            return self.table_ocr.ocr(img, det, rec, mfd_res, tqdm_enable, tqdm_desc, **kwargs)
        elif is_image or self.is_image_mode:
            # Use image backend
            logger.debug(f"Using {self.image_backend_name} for image: {tqdm_desc}")
            return self.image_ocr.ocr(img, det, rec, mfd_res, tqdm_enable, tqdm_desc, **kwargs)
        else:
            # Use text backend for body text
            logger.debug(f"Using {self.text_backend_name} for text recognition")
            return self.text_ocr.ocr(img, det, rec, mfd_res, tqdm_enable, tqdm_desc, **kwargs)
    
    def __call__(self, img, mfd_res=None):
        """
        Simplified interface for compatibility.
        
        Args:
            img: Input image
            mfd_res: Math formula detection results
            
        Returns:
            (dt_boxes, rec_res): Tuple of detection boxes and recognition results
        """
        # Route based on mode flags
        if self.is_table_mode:
            return self.table_ocr(img, mfd_res)
        elif self.is_image_mode:
            return self.image_ocr(img, mfd_res)
        else:
            return self.text_ocr(img, mfd_res)
    
    def recognize_table(self, img, bbox_coords=None):
        """
        Process table using configured table backend.
        
        Args:
            img: Table image
            bbox_coords: Bounding box coordinates
            
        Returns:
            Table in markdown format
        """
        logger.debug(f"Using {self.table_backend_name} for table recognition")
        
        # If table backend has recognize_table method, use it
        if hasattr(self.table_ocr, 'recognize_table'):
            return self.table_ocr.recognize_table(img, bbox_coords)
        else:
            # Otherwise use standard OCR
            dt_boxes, rec_res = self.table_ocr(img)
            # Return simple text representation
            if rec_res:
                return '\n'.join([text for text, conf in rec_res])
            return ''


if __name__ == '__main__':
    import sys
    import cv2
    
    if len(sys.argv) > 1:
        test_img_path = sys.argv[1]
        backend = sys.argv[2] if len(sys.argv) > 2 else 'easyocr'
        
        print(f"Testing ConfigurableHybridOCR with text_backend={backend}")
        
        hybrid_ocr = ConfigurableHybridOCR(text_backend=backend)
        img = cv2.imread(test_img_path)
        
        if img is None:
            print(f"Error: Could not read image from {test_img_path}")
            sys.exit(1)
        
        dt_boxes, rec_res = hybrid_ocr(img)
        
        print(f"\nFound {len(dt_boxes) if dt_boxes else 0} text regions")
        if rec_res:
            for i, (text, conf) in enumerate(rec_res):
                print(f"{i+1}. {text} (confidence: {conf:.3f})")
    else:
        print("Usage: python configurable_hybrid_ocr.py <image_path> [text_backend]")
        print("text_backend options: paddle, easyocr, vision, lighton (default: easyocr)")
