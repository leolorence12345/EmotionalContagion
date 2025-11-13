"""
Main Pipeline Script
Runs the complete emotional contagion analysis pipeline
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection import TwitterDataCollector
from data_preprocessing import TweetPreprocessor
from emotion_detection import EmotionDetector
from contagion_analysis import ContagionAnalyzer
import pandas as pd


def load_config():
    """Load configuration from config.ini"""
    try:
        import configparser
        config = configparser.ConfigParser()
        config.read('config/config.ini')
        return config
    except Exception as e:
        print(f"Warning: Could not load config.ini: {e}")
        return None


def run_pipeline(config_path=None, query=None, start_time=None, end_time=None, 
                 output_dir='data/processed', use_bert=True, use_lexmo=False):
    """
    Run the complete emotional contagion analysis pipeline
    
    Args:
        config_path: Path to config file
        query: Twitter search query
        start_time: Start datetime for tweet collection
        end_time: End datetime for tweet collection
        output_dir: Output directory for processed data
        use_bert: Whether to use BERT for emotion detection
        use_lexmo: Whether to use LeXmo for emotion detection
    """
    print("=" * 60)
    print("Emotional Contagion Analysis Pipeline")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Step 1: Data Collection
    print("\n[1/5] Collecting tweets...")
    if config and config.has_section('TwitterAPI'):
        collector = TwitterDataCollector(
            consumer_key=config.get('TwitterAPI', 'consumer_key', fallback=None),
            consumer_secret=config.get('TwitterAPI', 'consumer_secret', fallback=None),
            bearer_token=config.get('TwitterAPI', 'bearer_token', fallback=None)
        )
    else:
        print("Warning: No Twitter API config found. Skipping data collection.")
        print("Please ensure tweets are already in data/raw/")
        collector = None
    
    if collector and query and start_time and end_time:
        raw_data_file = os.path.join('data/raw', 'tweets.txt')
        count = collector.collect_tweets(
            query=query,
            start_time=start_time,
            end_time=end_time,
            output_file=raw_data_file
        )
        print(f"Collected {count} tweets")
    else:
        raw_data_file = os.path.join('data/raw', 'tweets_worldcupfinal.txt')
        if not os.path.exists(raw_data_file):
            print(f"Error: {raw_data_file} not found. Please collect tweets first.")
            return
    
    # Step 2: Data Preprocessing
    print("\n[2/5] Preprocessing tweets...")
    preprocessor = TweetPreprocessor()
    
    # Load tweets
    tweets = []
    with open(raw_data_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                tweets.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(tweets)} tweets")
    
    # Preprocess
    df = preprocessor.preprocess_dataset(tweets, translate=True, clean=True)
    preprocessed_file = os.path.join(output_dir, 'preprocessed_tweets.csv')
    df.to_csv(preprocessed_file, index=False)
    print(f"Preprocessed {len(df)} tweet pairs")
    print(f"Saved to {preprocessed_file}")
    
    # Step 3: Emotion Detection
    print("\n[3/5] Detecting emotions...")
    detector = EmotionDetector(use_bert=use_bert, use_lexmo=use_lexmo)
    
    method = 'bert' if use_bert else 'lexmo'
    
    print("Detecting parent emotions...")
    df['parent_emotion'] = df['english_parent_text'].apply(
        lambda x: detector.detect_emotion(x, method=method) if pd.notna(x) else None
    )
    
    print("Detecting child emotions...")
    df['child_emotion'] = df['english_child_text'].apply(
        lambda x: detector.detect_emotion(x, method=method) if pd.notna(x) else None
    )
    
    emotion_file = os.path.join(output_dir, 'emotionalcontagion_results.csv')
    df.to_csv(emotion_file, index=False)
    print(f"Emotion detection complete. Saved to {emotion_file}")
    
    # Step 4: Contagion Analysis
    print("\n[4/5] Analyzing contagion patterns...")
    analyzer = ContagionAnalyzer()
    
    # Create contagion matrix
    matrix = analyzer.create_contagion_matrix(
        df,
        parent_emotion_col='parent_emotion',
        child_emotion_col='child_emotion'
    )
    
    matrix_file = os.path.join(output_dir, 'contagion_matrix.csv')
    matrix.to_csv(matrix_file)
    print(f"Contagion matrix saved to {matrix_file}")
    
    # Calculate probabilities
    prob_matrix = analyzer.calculate_contagion_probabilities(matrix)
    prob_file = os.path.join(output_dir, 'contagion_probabilities.csv')
    prob_matrix.to_csv(prob_file)
    print(f"Contagion probabilities saved to {prob_file}")
    
    # Calculate overall strength
    strength = analyzer.calculate_contagion_strength(
        df,
        parent_emotion_col='parent_emotion',
        child_emotion_col='child_emotion'
    )
    print(f"\nOverall Contagion Rate: {strength['contagion_rate']:.2%}")
    print(f"Total Pairs: {strength['total_pairs']}")
    print(f"Same Emotion: {strength['same_emotion']}")
    print(f"Different Emotion: {strength['different_emotion']}")
    
    # Step 5: Summary
    print("\n[5/5] Pipeline complete!")
    print("=" * 60)
    print("\nOutput files:")
    print(f"  - Preprocessed data: {preprocessed_file}")
    print(f"  - Emotion results: {emotion_file}")
    print(f"  - Contagion matrix: {matrix_file}")
    print(f"  - Contagion probabilities: {prob_file}")
    print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Emotional Contagion Analysis Pipeline'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Twitter search query (e.g., "#WorldCupFinal OR #FIFAWorldCup")'
    )
    parser.add_argument(
        '--start-time',
        type=str,
        help='Start time in format: YYYY-MM-DD HH:MM:SS (UTC)'
    )
    parser.add_argument(
        '--end-time',
        type=str,
        help='End time in format: YYYY-MM-DD HH:MM:SS (UTC)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--use-bert',
        action='store_true',
        default=True,
        help='Use BERT for emotion detection (default: True)'
    )
    parser.add_argument(
        '--use-lexmo',
        action='store_true',
        default=False,
        help='Use LeXmo for emotion detection (default: False)'
    )
    
    args = parser.parse_args()
    
    # Parse times if provided
    start_time = None
    end_time = None
    if args.start_time and args.end_time:
        try:
            start_time = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M:%S')
            start_time = start_time.replace(tzinfo=timezone.utc)
            end_time = datetime.strptime(args.end_time, '%Y-%m-%d %H:%M:%S')
            end_time = end_time.replace(tzinfo=timezone.utc)
        except ValueError:
            print("Error: Invalid time format. Use YYYY-MM-DD HH:MM:SS")
            return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    
    # Run pipeline
    run_pipeline(
        query=args.query,
        start_time=start_time,
        end_time=end_time,
        output_dir=args.output_dir,
        use_bert=args.use_bert,
        use_lexmo=args.use_lexmo
    )


if __name__ == "__main__":
    main()

