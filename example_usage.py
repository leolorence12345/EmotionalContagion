"""
Example Usage Script
Demonstrates how to use the emotional contagion analysis toolkit
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection import TwitterDataCollector
from data_preprocessing import TweetPreprocessor
from emotion_detection import EmotionDetector, BERTEmotionClassifier, LeXmoEmotionAnalyzer
from image_processing import ImageMetadataExtractor
from contagion_analysis import ContagionAnalyzer
import pandas as pd
import json


def example_emotion_detection():
    """Example: Emotion detection from text"""
    print("\n" + "="*60)
    print("Example 1: Emotion Detection")
    print("="*60)
    
    # Initialize BERT classifier
    print("\n1. Using BERT for emotion detection:")
    bert_classifier = BERTEmotionClassifier()
    
    sample_texts = [
        "I'm so excited about the World Cup final!",
        "This is terrible news, I'm very disappointed.",
        "I'm feeling anxious about the results."
    ]
    
    for text in sample_texts:
        emotion = bert_classifier.predict_emotion(text)
        print(f"  Text: {text}")
        print(f"  Emotion: {emotion}\n")
    
    # Initialize LeXmo analyzer
    print("2. Using LeXmo (NLTK) for emotion detection:")
    lexmo_analyzer = LeXmoEmotionAnalyzer()
    
    for text in sample_texts:
        emotion = lexmo_analyzer.get_dominant_emotion(text)
        detailed = lexmo_analyzer.analyze_emotion(text)
        print(f"  Text: {text}")
        print(f"  Dominant Emotion: {emotion}")
        print(f"  Top 3 Emotions: {sorted(detailed.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)[:3]}\n")


def example_data_preprocessing():
    """Example: Data preprocessing"""
    print("\n" + "="*60)
    print("Example 2: Data Preprocessing")
    print("="*60)
    
    preprocessor = TweetPreprocessor()
    
    # Sample raw tweet text
    raw_text = "@user Check out this link! https://example.com #WorldCup #excited 🎉"
    
    print(f"\nOriginal text: {raw_text}")
    
    # Clean text
    cleaned = preprocessor.clean_text(raw_text)
    print(f"Cleaned text: {cleaned}")
    
    # Translate (if needed)
    # translated = preprocessor.translate_to_english(raw_text)
    # print(f"Translated: {translated}")


def example_contagion_analysis():
    """Example: Contagion analysis"""
    print("\n" + "="*60)
    print("Example 3: Contagion Analysis")
    print("="*60)
    
    # Create sample data
    sample_data = {
        'parent_emotion': ['joy', 'joy', 'anger', 'sadness', 'joy', 'anger'],
        'child_emotion': ['joy', 'joy', 'anger', 'sadness', 'sadness', 'joy']
    }
    df = pd.DataFrame(sample_data)
    
    print("\nSample data:")
    print(df)
    
    # Initialize analyzer
    analyzer = ContagionAnalyzer()
    
    # Create contagion matrix
    matrix = analyzer.create_contagion_matrix(df)
    print("\nContagion Matrix:")
    print(matrix)
    
    # Calculate probabilities
    prob_matrix = analyzer.calculate_contagion_probabilities(matrix)
    print("\nContagion Probabilities:")
    print(prob_matrix)
    
    # Calculate overall strength
    strength = analyzer.calculate_contagion_strength(df)
    print(f"\nOverall Contagion Rate: {strength['contagion_rate']:.2%}")


def example_image_processing():
    """Example: Image processing (if images are available)"""
    print("\n" + "="*60)
    print("Example 4: Image Processing")
    print("="*60)
    
    print("\nNote: This example requires image files and DeepFace/Pytesseract setup")
    print("To use image processing:")
    print("  1. Install Tesseract OCR")
    print("  2. Install DeepFace: pip install deepface")
    print("  3. Provide path to image file")
    
    # Uncomment to use:
    # extractor = ImageMetadataExtractor()
    # metadata = extractor.extract_metadata('path/to/image.jpg')
    # print(f"Emotion: {metadata['emotion']}")
    # print(f"Text: {metadata['extracted_text']}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("Emotional Contagion Analysis - Example Usage")
    print("="*60)
    
    try:
        example_emotion_detection()
        example_data_preprocessing()
        example_contagion_analysis()
        example_image_processing()
        
        print("\n" + "="*60)
        print("Examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()

