"""
Apple Vision Framework OCR wrapper for MinerU pipeline.
Provides high-quality Vietnamese text recognition on macOS using native Vision framework.

NOTE: Vision Framework uses bottom-left origin coordinate system, 
while MinerU expects top-left origin. This module handles the coordinate conversion.
"""
import os
import sys
import cv2
import numpy as np
import re
from loguru import logger

# Check if running on macOS
if sys.platform != 'darwin':
    logger.error("Vision Framework is only available on macOS")
    raise ImportError("Vision Framework requires macOS")

try:
    import Vision
    import Quartz
    import Cocoa
    from Foundation import NSURL
except ImportError:
    logger.error("PyObjC frameworks not installed. Please run: pip install pyobjc-framework-Vision pyobjc-framework-Quartz pyobjc-framework-Cocoa")
    raise


class VisionFrameworkOCR:
    """
    Apple Vision Framework OCR wrapper compatible with MinerU OCR interface.
   
    Coordinate System Conversion:
    - Vision Framework: bottom-left origin (0,0), Y-axis increases upward
    - Min erU expected: top-left origin (0,0), Y-axis increases downward
    - Conversion formula: y_top_left = image_height - y_bottom_left
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Vision Framework OCR.
        
        Args:
            **kwargs: Additional arguments (for compatibility with other OCR engines)
        """
        self.lang = kwargs.get('lang', 'vi')
        self.device = kwargs.get('device', 'cpu')
        self.drop_score = kwargs.get('drop_score', 0.5)
        
        if Vision is None:
            raise ImportError("Apple Vision Framework is not available. Please install pyobjc-framework-Vision.")
        
        logger.info("Initialized Apple Vision Framework OCR")
    
    def __call__(self, img, mfd_res=None):
        """
        Main OCR method compatible with PytorchPaddleOCR interface.
        
        Args:
            img: Input image (numpy array or file path)
            mfd_res: Math formula detection results (unused)
            
        Returns:
            (dt_boxes, rec_res): Tuple of detection boxes and recognition results
        """
        if img is None:
            return [], []
        
        # Convert input to CGImage
        cg_image = None
        
        if isinstance(img, str):
            # File path input
            if not os.path.exists(img):
                logger.error(f"Image path does not exist: {img}")
                return [], []
            
            file_url = NSURL.fileURLWithPath_(img)
            img_source = Quartz.CGImageSourceCreateWithURL(file_url, None)
            if img_source is None:
                logger.error(f"Failed to create CGImageSource from file: {img}")
                return [], []
            
            cg_image = Quartz.CGImageSourceCreateImageAtIndex(img_source, 0, None)
            if cg_image is None:
                logger.error(f"Failed to create CGImage from file: {img}")
                return [], []
        
        elif isinstance(img, np.ndarray):
            # NumPy array input - encode to PNG first for reliability
            is_success, buffer = cv2.imencode(".png", img)
            if not is_success:
                logger.error("Failed to encode numpy image")
                return [], []
            
            ns_data = Cocoa.NSData.dataWithBytes_length_(buffer.tobytes(), len(buffer))
            
            # Use ImageIO to create CGImage
            img_source = Quartz.CGImageSourceCreateWithData(ns_data, None)
            if img_source is None:
                logger.error("Failed to create CGImageSource")
                return [], []
            
            cg_image = Quartz.CGImageSourceCreateImageAtIndex(img_source, 0, None)
            if cg_image is None:
                logger.error("Failed to create CGImage from source")
                return [], []
        
        else:
            logger.error(f"Unsupported image type: {type(img)}")
            return [], []
        
        if cg_image is None:
            logger.error("Failed to get CGImage")
            return [], []
        
        # Prepare to collect results
        results = []
        
        def handle_request(request, error):
            """Completion handler for Vision request."""
            if error:
                logger.error(f"Vision request failed: {error}")
                return
            
            observations = request.results()
            if not observations:
                return
            
            width = Quartz.CGImageGetWidth(cg_image)
            height = Quartz.CGImageGetHeight(cg_image)
            
            for observation in observations:
                # Get top candidate
                candidate = observation.topCandidates_(1)[0]
                text = candidate.string()
                confidence = candidate.confidence()
                
                # Get corner points (normalized, origin bottom-left)
                corners = [
                    observation.topLeft(),
                    observation.topRight(),
                    observation.bottomRight(),
                    observation.bottomLeft()
                ]
                
                # Convert to pixel coordinates and flip Y-axis
                # Vision: Y increases upward from bottom
                # MinerU: Y increases downward from top
                pixel_corners = []
                for point in corners:
                    x = int(round(point.x * width))
                    y = int(round((1.0 - point.y) * height))  # Flip Y
                    pixel_corners.append([x, y])
                
                results.append({
                    'text': text,
                    'confidence': confidence,
                    'box': pixel_corners
                })
        
        # Create Vision request with completion handler
        req = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handle_request)
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setRecognitionLanguages_(['vi-VN', 'en-US'])  # Prioritize Vietnamese
        req.setUsesLanguageCorrection_(True)
        
        # Perform OCR request
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
        success, error = handler.performRequests_error_([req], None)
        
        if not success:
            logger.error(f"Failed to perform Vision request: {error}")
            return [], []
        
        # Filter out Chinese characters (fix hallucinations)
        dt_boxes = []
        rec_res = []
        
        for item in results:
            text = item['text']
            # Remove Chinese characters
            original_text = text
            text = re.sub(r'[\u4e00-\u9fff]', '', text).strip()
            if original_text != text:
                logger.warning(f"Filtered Chinese: '{original_text}' -> '{text}'")
            
            if not text:
                continue
            
            # Add to results with confidence filtering
            if item['confidence'] >= self.drop_score:
                box = np.array(item['box'], dtype=np.int32)
                dt_boxes.append(box)
                rec_res.append((text, item['confidence']))
        
        return dt_boxes, rec_res
    
    def ocr(self, img, det=True, rec=True, mfd_res=None, tqdm_enable=False, tqdm_desc="OCR-rec Predict", **kwargs):
        """
        Perform OCR on image(s).
        
        Args:
            img: Input image (numpy array, list of arrays, or bytes)
            det: Whether to perform detection (always True for Vision)
            rec: Whether to perform recognition
            mfd_res: Math formula detection results (unused)
            tqdm_enable: Show progress bar
            tqdm_desc: Progress bar description
            
        Returns:
            List of OCR results in MinerU format: [[[box, (text, confidence)], ...]]
        """
        logger.info(f"VisionFrameworkOCR.ocr called with input type: {type(img)}")
        
        # Handle list of images (batch processing for cropped regions)
        if isinstance(img, list):
            logger.info(f"Processing batch of {len(img)} crops")
            rec_results = []
            for i, sub_img in enumerate(img):
                # Handle bytes
                if isinstance(sub_img, bytes):
                    nparr = np.frombuffer(sub_img, np.uint8)
                    sub_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                dt_boxes, sub_rec_res = self.__call__(sub_img)
                if sub_rec_res:
                    # If multiple texts found in crop, join them
                    text = " ".join([res[0] for res in sub_rec_res])
                    score = sum([res[1] for res in sub_rec_res]) / len(sub_rec_res)
                    rec_results.append((text, score))
                else:
                    rec_results.append(("", 0.0))
            return [rec_results]
        
        # Single image processing
        if isinstance(img, bytes):
            nparr = np.frombuffer(img, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        dt_boxes, rec_res = self.__call__(img)
        
        if not dt_boxes and not rec_res:
            return [None]
        
        res_item = []
        for box, (text, score) in zip(dt_boxes, rec_res):
            res_item.append([box.tolist(), (text, score)])
        
        return [res_item]


if __name__ == '__main__':
    # Test Vision Framework OCR
    import sys
    if len(sys.argv) > 1:
        test_img_path = sys.argv[1]
        vision_ocr = VisionFrameworkOCR()
        dt_boxes, rec_res = vision_ocr(test_img_path)
        
        print(f"Found {len(dt_boxes) if dt_boxes else 0} text regions")
        if rec_res:
            for i, (text, conf) in enumerate(rec_res):
                print(f"{i+1}. {text} (confidence: {conf:.3f})")
    else:
        print("Usage: python vision_ocr.py <image_path>")
