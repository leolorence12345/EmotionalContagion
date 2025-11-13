"""
Emotional Contagion Analysis Package
A comprehensive toolkit for measuring emotional contagion through tweets
"""

from .data_collection import TwitterDataCollector
from .data_preprocessing import TweetPreprocessor
from .emotion_detection import BERTEmotionClassifier, LeXmoEmotionAnalyzer, EmotionDetector
from .image_processing import ImageEmotionExtractor, ImageTextExtractor, ImageMetadataExtractor
from .contagion_analysis import ContagionAnalyzer

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    'TwitterDataCollector',
    'TweetPreprocessor',
    'BERTEmotionClassifier',
    'LeXmoEmotionAnalyzer',
    'EmotionDetector',
    'ImageEmotionExtractor',
    'ImageTextExtractor',
    'ImageMetadataExtractor',
    'ContagionAnalyzer'
]

