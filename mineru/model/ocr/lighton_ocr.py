# Copyright (c) Opendatalab. All rights reserved.
"""
LightOnOCR client for MinerU integration.
Uses LM Studio API with lightonocr vision model for Vietnamese text extraction.
"""
import os
import base64
import requests
from io import BytesIO
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
from PIL import Image
from loguru import logger


class LightOnOCR:
    """OCR client using LM Studio with LightOnOCR vision model."""
    
    def __init__(
        self,
        server_url: str = None,
        model_name: str = None,
        timeout: int = 180,
        **kwargs
    ):
        """
        Initialize LightOnOCR client.
        
        Args:
            server_url: API endpoint (auto-corrected based on api_type)
            model_name: Model name
            timeout: Request timeout in seconds
            
        Environment Variables:
            LIGHTON_SERVER_URL: API endpoint URL
            LIGHTON_MODEL_NAME: Model name
            LIGHTON_API_TYPE: 'openai' or 'ollama' (default: 'openai')
        """
        self.server_url = server_url or os.getenv(
            'LIGHTON_SERVER_URL', 
            'http://localhost:1234/v1/chat/completions'
        )
        self.model_name = model_name or os.getenv(
            'LIGHTON_MODEL_NAME', 
            'lightonocr'
        )
        self.timeout = timeout
        self.drop_score = kwargs.get('drop_score', 0.5)
        
        # Get API type from env var (explicit configuration)
        self.api_type = os.getenv('LIGHTON_API_TYPE', 'openai').lower()
        
        # Auto-correct URL based on API type
        if self.api_type == 'ollama':
            # Ensure Ollama URL ends with /api/chat for vision models
            if not self.server_url.endswith('/api/chat') and not self.server_url.endswith('/api/generate'):
                if self.server_url.endswith('/'):
                    self.server_url += 'api/chat'
                else:
                    self.server_url += '/api/chat'
        elif self.api_type == 'openai':
            # Ensure OpenAI-compatible URL ends with /v1/chat/completions
            if not self.server_url.endswith('/v1/chat/completions') and not self.server_url.endswith('/chat/completions'):
                if self.server_url.endswith('/'):
                    self.server_url += 'v1/chat/completions'
                else:
                    self.server_url += '/v1/chat/completions'
        
        backend_name = 'Ollama' if self.api_type == 'ollama' else 'LM Studio/OpenAI'
        logger.info(f"Initialized LightOnOCR: URL={self.server_url}, Model={self.model_name}, API Type={self.api_type}")
        
        # Test connection on init
        try:
            if self.api_type == 'ollama':
                # Ollama: test /api/tags endpoint
                base_url = self.server_url.rsplit('/api/', 1)[0]
                test_url = base_url + '/api/tags'
            else:
                # OpenAI: test /v1/models endpoint
                test_url = self.server_url.replace('/chat/completions', '/models')
            
            response = requests.get(test_url, timeout=5)
            if response.status_code == 200:
                logger.info(f"{backend_name} connection successful")
        except Exception as e:
            logger.warning(f"Cannot connect to {backend_name}: {e}")
    
    def _image_to_base64(self, image: Union[np.ndarray, Image.Image]) -> str:
        """Convert image to base64 string."""
        if isinstance(image, np.ndarray):
            # Convert BGR (OpenCV) to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _detect_api_type(self) -> str:
        """Auto-detect API type from server URL."""
        if '/api/' in self.server_url:
            return 'ollama'
        elif '/v1/' in self.server_url:
            return 'openai'
        else:
            # Default to ollama if unclear
            return 'ollama'
    
    def _call_api(self, image: Union[np.ndarray, Image.Image], prompt: str) -> str:
        """
        Call OCR API with image and prompt, auto-detecting backend type.
        
        Args:
            image: Image to process (numpy array or PIL Image)
            prompt: Text prompt for the model
            
        Returns:
            Extracted text from the model
        """
        api_type = self._detect_api_type()
        
        if api_type == 'ollama':
            return self._call_ollama_api(image, prompt)
        else:
            return self._call_lmstudio_api(image, prompt)
    
    def _call_ollama_api(self, image: Union[np.ndarray, Image.Image], prompt: str) -> str:
        """
        Call Ollama API with image and prompt.
        
        Ollama uses /api/generate or /api/chat endpoints with 'images' array.
        """
        img_b64 = self._image_to_base64(image)
        
        # Determine endpoint type from URL
        if '/api/chat' in self.server_url:
            # Chat endpoint format
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_b64]
                    }
                ],
                "stream": False
            }
        else:
            # Generate endpoint format (/api/generate)
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False
            }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(
                self.server_url, 
                headers=headers, 
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                # Handle both /api/generate and /api/chat response formats
                if 'response' in result:
                    return result['response']
                elif 'message' in result:
                    return result['message'].get('content', '')
                else:
                    logger.warning(f"Unexpected Ollama response format: {result.keys()}")
                    return str(result)
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return ""
                
        except requests.exceptions.Timeout:
            logger.error(f"Ollama API timeout after {self.timeout}s")
            return ""
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {self.server_url}")
            return ""
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            return ""
    
    def _call_lmstudio_api(self, image: Union[np.ndarray, Image.Image], prompt: str) -> str:
        """
        Call LM Studio (OpenAI-compatible) API with image and prompt.
        """
        img_b64 = self._image_to_base64(image)
        image_url = f"data:image/jpeg;base64,{img_b64}"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 4096
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(
                self.server_url, 
                headers=headers, 
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"LM Studio API error: {response.status_code} - {response.text}")
                return ""
                
        except requests.exceptions.Timeout:
            logger.error(f"LM Studio API timeout after {self.timeout}s")
            return ""
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to LM Studio at {self.server_url}")
            return ""
        except Exception as e:
            logger.error(f"LM Studio API call failed: {e}")
            return ""
    
    def recognize_text(self, image: Union[np.ndarray, Image.Image]) -> Tuple[str, float]:
        """
        Recognize text from a single image crop.
        
        Args:
            image: Cropped text region image
            
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        prompt = (
            "Extract all text from this image. "
            "Output only the extracted text, nothing else. "
            "Be accurate with Vietnamese text and diacritics."
        )
        
        text = self._call_api(image, prompt)
        
        # Clean up response - remove any echoed prompt text
        text = text.strip()
        
        # Common prompt fragments that might be echoed
        prompt_fragments = [
            "Extract all text from this image.",
            "Output only the extracted text, nothing else.",
            "Be accurate with Vietnamese text and diacritics.",
            "Extract all text from this image",
            "Output only the extracted text",
            "Be accurate with Vietnamese text",
            "Return only the extracted text",
            "nothing else",
        ]
        
        for fragment in prompt_fragments:
            text = text.replace(fragment, "")
        
        # Also remove lines that look like they're part of a prompt
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and lines that look like instructions
            if not line:
                continue
            if line.lower().startswith("extract") and "text" in line.lower():
                continue
            if line.lower().startswith("output only"):
                continue
            if "be accurate with" in line.lower():
                continue
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines).strip()
        
        # Return with high confidence if we got text, low if empty
        confidence = 0.95 if text.strip() else 0.0
        return text.strip(), confidence
    
    def recognize_table(self, image: Union[np.ndarray, Image.Image], bbox_coords=None) -> str:
        """
        Recognize table content and return as HTML.
        
        Args:
            image: Table image
            bbox_coords: Bounding box coordinates (ignored)
            
        Returns:
            HTML table string
        """
        prompt = (
            "Extract the table from this image. "
            "Output as HTML table format with <table>, <tr>, <th>, <td> tags. "
            "Keep all text content accurate, especially Vietnamese text. "
            "Do not add any explanation, only output the HTML table."
        )
        
        html = self._call_api(image, prompt)
        
        # Clean up response - ensure it's valid table HTML
        html = html.strip()
        
        # Nếu model trả về chuỗi rỗng / lỗi
        if not html:
            return ""
            
        if not html.startswith("<table"):
            # Try to extract table from response
            if "<table>" in html:
                start = html.find("<table>")
                end = html.rfind("</table>") + len("</table>")
                html = html[start:end]
            else:
                # Wrap in table if needed
                html = f"<table><tr><td>{html}</td></tr></table>"
        
        return html
    
    def ocr(
        self,
        img: Union[np.ndarray, List[np.ndarray]],
        det: bool = True,
        rec: bool = True,
        mfd_res: List = None,
        tqdm_enable: bool = False,
        tqdm_desc: str = "LightOnOCR",
        **kwargs
    ) -> List:
        """
        OCR interface compatible with PytorchPaddleOCR.
        
        For LightOnOCR, we process the entire image at once rather than
        detecting individual text boxes, so det is effectively ignored.
        
        Args:
            img: Image or list of images to process
            det: Detection flag (ignored for LightOnOCR - we process full image)
            rec: Recognition flag
            mfd_res: Math formula detection results (ignored)
            tqdm_enable: Show progress bar
            tqdm_desc: Progress bar description
            
        Returns:
            OCR results in format compatible with PytorchPaddleOCR
        """
        # Handle single image
        if isinstance(img, np.ndarray) and len(img.shape) == 3:
            imgs = [img]
        elif isinstance(img, list):
            imgs = img
        else:
            imgs = [img]
        
        ocr_res = []
        
        if det and rec:
            # Full OCR: For LightOnOCR, we return text without bounding boxes
            # since the model processes the entire image
            for image in imgs:
                text, score = self.recognize_text(image)
                if text:
                    # Create a synthetic bounding box covering the whole image
                    h, w = image.shape[:2] if isinstance(image, np.ndarray) else image.size[::-1]
                    box = [[0, 0], [w, 0], [w, h], [0, h]]
                    ocr_res.append([[box, (text, score)]])
                else:
                    ocr_res.append(None)
                    
        elif not det and rec:
            # Recognition only: process cropped images
            results = []
            if tqdm_enable:
                from tqdm import tqdm
                imgs = tqdm(imgs, desc=tqdm_desc)
            
            for image in imgs:
                text, score = self.recognize_text(image)
                results.append((text, score))
            
            ocr_res.append(results)
            
        elif det and not rec:
            # Detection only: LightOnOCR doesn't do detection separately
            # Return empty to fall back to other detection methods
            for image in imgs:
                ocr_res.append(None)
        
        return ocr_res
    
    def __call__(self, img: np.ndarray, mfd_res: List = None) -> Tuple[List, List]:
        """
        Process image and return detected boxes and recognized text.
        Compatible with PytorchPaddleOCR interface.
        
        Args:
            img: Input image
            mfd_res: Math formula detection results (ignored)
            
        Returns:
            Tuple of (detected_boxes, recognition_results)
        """
        if img is None:
            logger.debug("No valid image provided to LightOnOCR")
            return None, None
        
        text, score = self.recognize_text(img)
        
        if not text:
            return None, None
        
        # Create synthetic bounding box for the whole image
        h, w = img.shape[:2]
        box = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        
        return [box], [(text, score)]
