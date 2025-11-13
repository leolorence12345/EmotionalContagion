"""
Image Processing Module
Handles emotion extraction from images using DeepFace and text extraction using Pytesseract
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import Dict, Optional, List
import os


class ImageEmotionExtractor:
    """Extract emotions from facial expressions in images using DeepFace"""
    
    def __init__(self, model_name='VGG-Face', enforce_detection=False):
        """
        Initialize DeepFace emotion extractor
        
        Args:
            model_name: DeepFace model name (VGG-Face, Facenet, etc.)
            enforce_detection: Whether to enforce face detection
        """
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            self.model_name = model_name
            self.enforce_detection = enforce_detection
        except ImportError:
            raise ImportError("DeepFace is not installed. Install it using: pip install deepface")
    
    def extract_emotion(self, image_path: str) -> Optional[Dict]:
        """
        Extract emotion from image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with emotion predictions or None if face not detected
        """
        try:
            result = self.DeepFace.analyze(
                img_path=image_path,
                actions=['emotion'],
                model_name=self.model_name,
                enforce_detection=self.enforce_detection
            )
            
            if isinstance(result, list):
                result = result[0]
            
            # Extract dominant emotion
            if 'emotion' in result:
                emotions = result['emotion']
                dominant_emotion = max(emotions, key=emotions.get)
                return {
                    'dominant_emotion': dominant_emotion,
                    'emotion_scores': emotions,
                    'confidence': emotions[dominant_emotion]
                }
            return None
        except Exception as e:
            print(f"Error extracting emotion from image: {e}")
            return None
    
    def extract_emotions_batch(self, image_paths: List[str]) -> List[Optional[Dict]]:
        """
        Extract emotions from multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of emotion dictionaries
        """
        results = []
        for image_path in image_paths:
            results.append(self.extract_emotion(image_path))
        return results


class ImageTextExtractor:
    """Extract text from images using Pytesseract (OCR)"""
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize OCR text extractor
        
        Args:
            tesseract_cmd: Path to tesseract executable (if not in PATH)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    def extract_text(self, image_path: str, preprocess: bool = True) -> str:
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to image file
            preprocess: Whether to preprocess image for better OCR
            
        Returns:
            Extracted text string
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            
            if image is None:
                # Try with PIL if OpenCV fails
                image = np.array(Image.open(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if preprocess:
                # Preprocess image for better OCR
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Apply thresholding
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # Denoise
                denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
                image = denoised
            
            # Extract text
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from image: {e}")
            return ""
    
    def extract_text_batch(self, image_paths: List[str]) -> List[str]:
        """
        Extract text from multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of extracted text strings
        """
        results = []
        for image_path in image_paths:
            results.append(self.extract_text(image_path))
        return results


class ImageMetadataExtractor:
    """Extract both emotion and text from images"""
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize combined image metadata extractor
        
        Args:
            tesseract_cmd: Path to tesseract executable
        """
        self.emotion_extractor = ImageEmotionExtractor()
        self.text_extractor = ImageTextExtractor(tesseract_cmd)
    
    def extract_metadata(self, image_path: str) -> Dict:
        """
        Extract both emotion and text from image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with emotion and text metadata
        """
        emotion = self.emotion_extractor.extract_emotion(image_path)
        text = self.text_extractor.extract_text(image_path)
        
        return {
            'image_path': image_path,
            'emotion': emotion,
            'extracted_text': text
        }
    
    def extract_metadata_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Extract metadata from multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of metadata dictionaries
        """
        results = []
        for image_path in image_paths:
            results.append(self.extract_metadata(image_path))
        return results


def main():
    """Example usage"""
    # Example: Extract metadata from an image
    extractor = ImageMetadataExtractor()
    
    # Assuming you have an image file
    image_path = "data/raw/sample_image.jpg"
    
    if os.path.exists(image_path):
        metadata = extractor.extract_metadata(image_path)
        print(f"Emotion: {metadata['emotion']}")
        print(f"Extracted text: {metadata['extracted_text']}")
    else:
        print(f"Image not found: {image_path}")


if __name__ == "__main__":
    main()

