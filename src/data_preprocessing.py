"""
Data Preprocessing Module
Handles cleaning, translation, and parent-child relationship extraction
"""

import re
import pandas as pd
import json
from googletrans import Translator


class TweetPreprocessor:
    """Preprocesses tweet data"""
    
    def __init__(self):
        self.translator = Translator()
    
    def clean_text(self, text):
        """
        Clean tweet text using Regular Expressions
        
        Args:
            text: Raw tweet text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (keep the text, remove #)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep emojis
        text = re.sub(r'[^\w\s\U0001F300-\U0001F9FF]', '', text)
        
        return text.strip()
    
    def translate_to_english(self, text, max_retries=3):
        """
        Translate text to English
        
        Args:
            text: Text to translate
            max_retries: Maximum number of retry attempts
            
        Returns:
            Translated text or original text if translation fails
        """
        if pd.isna(text) or text is None or text == "":
            return ""
        
        for attempt in range(max_retries):
            try:
                translated = self.translator.translate(text, dest='en')
                return translated.text
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Translation failed after {max_retries} attempts: {e}")
                    return text
                continue
        return text
    
    def extract_parent_child_relationships(self, tweets):
        """
        Extract parent-child tweet relationships
        
        Args:
            tweets: List of tweet dictionaries
            
        Returns:
            DataFrame with parent-child relationships
        """
        # Extract tweet IDs and texts
        id_text = []
        for tweet in tweets:
            if 'id' in tweet and 'text' in tweet:
                id_text.append({'id': tweet['id'], 'text': tweet['text']})
        
        df = pd.DataFrame(id_text)
        
        # Extract referenced tweets
        ref_data = []
        for tweet in tweets:
            if 'referenced_tweets' in tweet and tweet['referenced_tweets']:
                for ref in tweet['referenced_tweets']:
                    if 'id' in ref and 'text' in ref:
                        ref_data.append({
                            'id': tweet.get('id'),
                            'ref_id': ref['id'],
                            'ref_text': ref.get('text', '')
                        })
        
        if not ref_data:
            return pd.DataFrame()
        
        ref_df = pd.DataFrame(ref_data)
        
        # Merge to create parent-child pairs
        merged = pd.merge(
            df,
            ref_df,
            left_on='id',
            right_on='ref_id',
            how='inner'
        )
        
        # Rename columns for clarity
        result = merged[['id', 'text', 'ref_id', 'ref_text']].copy()
        result.columns = ['child_id', 'child_text', 'parent_id', 'parent_text']
        
        return result
    
    def preprocess_dataset(self, tweets, translate=True, clean=True):
        """
        Complete preprocessing pipeline
        
        Args:
            tweets: List of tweet dictionaries
            translate: Whether to translate to English
            clean: Whether to clean text
            
        Returns:
            Preprocessed DataFrame
        """
        # Extract relationships
        df = self.extract_parent_child_relationships(tweets)
        
        if df.empty:
            return df
        
        # Clean text
        if clean:
            df['child_text'] = df['child_text'].apply(self.clean_text)
            df['parent_text'] = df['parent_text'].apply(self.clean_text)
        
        # Translate to English
        if translate:
            print("Translating child texts...")
            df['english_child_text'] = df['child_text'].apply(self.translate_to_english)
            print("Translating parent texts...")
            df['english_parent_text'] = df['parent_text'].apply(self.translate_to_english)
        else:
            df['english_child_text'] = df['child_text']
            df['english_parent_text'] = df['parent_text']
        
        return df


def main():
    """Example usage"""
    preprocessor = TweetPreprocessor()
    
    # Load tweets
    tweets = []
    with open('data/raw/tweets_worldcupfinal.txt', 'r', encoding='utf-8') as f:
        for line in f:
            try:
                tweets.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    # Preprocess
    df = preprocessor.preprocess_dataset(tweets)
    
    # Save
    df.to_csv('data/processed/preprocessed_tweets.csv', index=False)
    print(f"Preprocessed {len(df)} tweet pairs")


if __name__ == "__main__":
    main()

