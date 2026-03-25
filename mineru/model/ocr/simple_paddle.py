"""
A simple PaddleOCR wrapper that lets the official library handle model downloads.
This is used to properly support Vietnamese, which is mishandled by the
complex, manual model management in `pytorch_paddle.py`.
"""
import numpy as np
from loguru import logger

try:
    from paddleocr import PaddleOCR as OfficialPaddleOCR
except ImportError:
    logger.error("PaddleOCR not installed. Please run: pip install paddleocr")
    raise

class SimplePaddle:
    """
    A simplified PaddleOCR wrapper that initializes with a specific language
    and relies on PaddleOCR's built-in model management.
    """
    def __init__(self, lang='vi', **kwargs):
        """
        Initializes the official PaddleOCR instance.

        Args:
            lang (str): The language code to be passed to PaddleOCR (e.g., 'vi').
            **kwargs: Ignored, for compatibility with the project's OCR factory.
        """
        logger.info(f"Initializing official PaddleOCR with lang='{lang}'. "
                    f"This may trigger a model download on first use.")
        # We let the official library handle everything by just specifying the language.
        self.ocr_engine = OfficialPaddleOCR(use_angle_cls=True, lang=lang)

    def __call__(self, img, mfd_res=None):
        """
        Performs OCR and adapts the output to the format expected by the MinerU pipeline,
        which is a tuple of (detection_boxes, recognition_results).

        Args:
            img: The input image as a numpy array.
            mfd_res: Ignored, for compatibility.

        Returns:
            A tuple of (dt_boxes, rec_res) or (None, None) if no text is found.
            - dt_boxes: A list of numpy arrays, where each array represents a bounding box.
            - rec_res: A list of tuples, where each tuple is (text, confidence_score).
        """
        if img is None:
            logger.warning("SimplePaddle received a null image.")
            return None, None

        # The official paddleocr().ocr() returns a list containing one list of results.
        # e.g., [[[[box1], (text1, score1)], [[box2], (text2, score2)]]]
        result = self.ocr_engine.ocr(img)

        if not result or not result[0]:
            return None, None

        dt_boxes = []
        rec_res = []
        # We iterate through the inner list of results.
        for line in result[0]:
            box_coords = line[0]
            text, score = line[1]

            dt_boxes.append(np.array(box_coords, dtype=np.float32))
            rec_res.append((text, score))

        return dt_boxes, rec_res

    def ocr(self, img, det=True, rec=True, *args, **kwargs):
        """
        Provides a more detailed ocr method for compatibility with other wrappers.
        For this simple wrapper, it just calls the main __call__ method.
        """
        # This simple wrapper doesn't distinguish between det/rec modes.
        if not det:
            logger.warning("SimplePaddle does not support recognition-only mode and will perform full OCR.")

        dt_boxes, rec_res = self(img)
        if dt_boxes is None:
            return [[]] # Return format expected by pipeline for no results

        # Reconstruct the [[[box, (text, confidence)], ...]] format
        ocr_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
        return [ocr_res]
