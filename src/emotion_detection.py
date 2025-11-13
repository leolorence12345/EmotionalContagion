"""
Emotion Detection Module
Handles emotion classification using BERTweet and NLTK-based LeXmo
"""

import pandas as pd
import requests
from io import StringIO
from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
from transformers import pipeline


class BERTEmotionClassifier:
    """BERT-based emotion classifier using j-hartmann/emotion-english-distilroberta-base"""
    
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        """
        Initialize BERT emotion classifier
        
        Args:
            model_name: HuggingFace model name for emotion classification
        """
        self.classifier = pipeline(
            "text-classification", 
            model=model_name, 
            return_all_scores=True
        )
    
    def predict_emotion(self, text):
        """
        Predict emotion from text
        
        Args:
            text: Input text
            
        Returns:
            Predicted emotion label (anger, disgust, fear, joy, sadness, surprise, neutral)
        """
        if pd.isna(text) or text is None or text == "":
            return None
        
        try:
            # Truncate text if too long (BERT has 512 token limit)
            if len(text) > 500:
                text = text[:500]
            
            results = self.classifier(text)[0]
            # Get emotion with highest score
            emotion = max(results, key=lambda x: x['score'])['label']
            return emotion
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return None
    
    def predict_emotions_batch(self, texts):
        """
        Predict emotions for multiple texts
        
        Args:
            texts: List of texts
            
        Returns:
            List of predicted emotions
        """
        emotions = []
        for text in texts:
            emotions.append(self.predict_emotion(text))
        return emotions


class LeXmoEmotionAnalyzer:
    """NLTK-based emotion analyzer using NRC Emotion Lexicon"""
    
    def __init__(self):
        """Initialize LeXmo analyzer with NRC Emotion Lexicon"""
        self.tweet_tokenizer = TweetTokenizer()
        self.stemmer = SnowballStemmer("english")
        self.emolex_df = None
        self._load_lexicon()
    
    def _load_lexicon(self):
        """Load NRC Emotion Lexicon from GitHub"""
        try:
            response = requests.get(
                'https://raw.github.com/dinbav/LeXmo/master/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
            )
            nrc = StringIO(response.text)
            
            self.emolex_df = pd.read_csv(
                nrc,
                names=["word", "emotion", "association"],
                sep=r'\t',
                engine='python'
            )
            
            self.emolex_words = self.emolex_df.pivot(
                index='word',
                columns='emotion',
                values='association'
            ).reset_index()
            
            self.emotions = self.emolex_words.columns.drop('word')
        except Exception as e:
            print(f"Error loading lexicon: {e}")
            raise
    
    def analyze_emotion(self, text):
        """
        Analyze emotion using NRC Emotion Lexicon (10 emotions)
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with emotion scores for: anger, anticipation, disgust, fear, 
            joy, negative, positive, sadness, surprise, trust
        """
        if pd.isna(text) or text is None or text == "":
            return {emotion: 0.0 for emotion in self.emotions}
        
        LeXmo_dict = {emotion: 0.0 for emotion in self.emotions}
        LeXmo_dict['text'] = text
        
        # Tokenize and stem
        document = self.tweet_tokenizer.tokenize(text)
        word_count = len(document)
        
        if word_count == 0:
            return LeXmo_dict
        
        rows_list = []
        for word in document:
            word = self.stemmer.stem(word.lower())
            emo_score = self.emolex_words[self.emolex_words.word == word]
            if not emo_score.empty:
                rows_list.append(emo_score)
        
        if rows_list:
            df = pd.concat(rows_list)
            df.reset_index(drop=True)
            
            for emotion in self.emotions:
                LeXmo_dict[emotion] = df[emotion].sum() / word_count
        
        return LeXmo_dict
    
    def get_dominant_emotion(self, text):
        """
        Get the dominant emotion from text
        
        Args:
            text: Input text
            
        Returns:
            Dominant emotion label or "Neutral" if all emotions are equal
        """
        emotion_dict = self.analyze_emotion(text)
        emotion_dict.pop('text', None)
        
        # Check if all emotions are equal (neutral)
        values = list(emotion_dict.values())
        if len(set(values)) == 1:
            return "Neutral"
        
        # Return emotion with highest score
        return max(emotion_dict, key=emotion_dict.get)


class EmotionDetector:
    """Combined emotion detection using both BERT and LeXmo"""
    
    def __init__(self, use_bert=True, use_lexmo=True):
        """
        Initialize emotion detector
        
        Args:
            use_bert: Whether to use BERT classifier
            use_lexmo: Whether to use LeXmo analyzer
        """
        self.bert_classifier = None
        self.lexmo_analyzer = None
        
        if use_bert:
            self.bert_classifier = BERTEmotionClassifier()
        
        if use_lexmo:
            self.lexmo_analyzer = LeXmoEmotionAnalyzer()
    
    def detect_emotion(self, text, method='bert'):
        """
        Detect emotion from text
        
        Args:
            text: Input text
            method: 'bert', 'lexmo', or 'both'
            
        Returns:
            Emotion label or dictionary depending on method
        """
        if method == 'bert' and self.bert_classifier:
            return self.bert_classifier.predict_emotion(text)
        elif method == 'lexmo' and self.lexmo_analyzer:
            return self.lexmo_analyzer.get_dominant_emotion(text)
        elif method == 'both':
            result = {}
            if self.bert_classifier:
                result['bert'] = self.bert_classifier.predict_emotion(text)
            if self.lexmo_analyzer:
                result['lexmo'] = self.lexmo_analyzer.get_dominant_emotion(text)
            return result
        else:
            raise ValueError(f"Method {method} not available")


def main():
    """Example usage"""
    # Initialize detector
    detector = EmotionDetector(use_bert=True, use_lexmo=True)
    
    # Example text
    text = "I'm so excited about the World Cup final! This is amazing!"
    
    # Detect emotion with BERT
    bert_emotion = detector.detect_emotion(text, method='bert')
    print(f"BERT emotion: {bert_emotion}")
    
    # Detect emotion with LeXmo
    lexmo_emotion = detector.detect_emotion(text, method='lexmo')
    print(f"LeXmo emotion: {lexmo_emotion}")
    
    # Get detailed LeXmo analysis
    if detector.lexmo_analyzer:
        detailed = detector.lexmo_analyzer.analyze_emotion(text)
        print(f"LeXmo detailed: {detailed}")


if __name__ == "__main__":
    main()

