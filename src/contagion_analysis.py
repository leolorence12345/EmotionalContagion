"""
Emotional Contagion Analysis Module
Analyzes emotion transfer from parent to child tweets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class ContagionAnalyzer:
    """Analyze emotional contagion patterns in tweet networks"""
    
    def __init__(self):
        """Initialize contagion analyzer"""
        self.emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
    
    def create_contagion_matrix(self, df: pd.DataFrame, 
                                parent_emotion_col: str = 'parent_emotion',
                                child_emotion_col: str = 'child_emotion') -> pd.DataFrame:
        """
        Create contagion matrix showing emotion transfer patterns
        
        Args:
            df: DataFrame with parent and child emotions
            parent_emotion_col: Column name for parent emotion
            child_emotion_col: Column name for child emotion
            
        Returns:
            Contagion matrix DataFrame
        """
        # Initialize matrix
        matrix = pd.DataFrame(
            0,
            index=self.emotion_labels,
            columns=self.emotion_labels
        )
        
        # Count emotion transitions
        for _, row in df.iterrows():
            parent_emotion = str(row[parent_emotion_col]).lower()
            child_emotion = str(row[child_emotion_col]).lower()
            
            if parent_emotion in self.emotion_labels and child_emotion in self.emotion_labels:
                matrix.loc[parent_emotion, child_emotion] += 1
        
        return matrix
    
    def calculate_contagion_probabilities(self, contagion_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate probability of emotion contagion
        
        Args:
            contagion_matrix: Contagion count matrix
            
        Returns:
            Probability matrix (normalized by row)
        """
        # Normalize by row to get probabilities
        prob_matrix = contagion_matrix.div(contagion_matrix.sum(axis=1), axis=0)
        prob_matrix = prob_matrix.fillna(0)
        return prob_matrix
    
    def calculate_contagion_strength(self, df: pd.DataFrame,
                                     parent_emotion_col: str = 'parent_emotion',
                                     child_emotion_col: str = 'child_emotion') -> Dict:
        """
        Calculate overall contagion strength metrics
        
        Args:
            df: DataFrame with parent and child emotions
            parent_emotion_col: Column name for parent emotion
            child_emotion_col: Column name for child emotion
            
        Returns:
            Dictionary with contagion metrics
        """
        # Count same emotion (contagion) vs different emotion (no contagion)
        same_emotion = 0
        different_emotion = 0
        total = 0
        
        for _, row in df.iterrows():
            parent_emotion = str(row[parent_emotion_col]).lower()
            child_emotion = str(row[child_emotion_col]).lower()
            
            if parent_emotion in self.emotion_labels and child_emotion in self.emotion_labels:
                total += 1
                if parent_emotion == child_emotion:
                    same_emotion += 1
                else:
                    different_emotion += 1
        
        contagion_rate = same_emotion / total if total > 0 else 0
        
        return {
            'total_pairs': total,
            'same_emotion': same_emotion,
            'different_emotion': different_emotion,
            'contagion_rate': contagion_rate
        }
    
    def analyze_emotion_by_category(self, df: pd.DataFrame,
                                    parent_emotion_col: str = 'parent_emotion',
                                    child_emotion_col: str = 'child_emotion') -> Dict:
        """
        Analyze contagion patterns for each emotion category
        
        Args:
            df: DataFrame with parent and child emotions
            parent_emotion_col: Column name for parent emotion
            child_emotion_col: Column name for child emotion
            
        Returns:
            Dictionary with analysis for each emotion
        """
        results = {}
        
        for emotion in self.emotion_labels:
            emotion_df = df[df[parent_emotion_col].str.lower() == emotion]
            
            if len(emotion_df) > 0:
                child_emotions = emotion_df[child_emotion_col].value_counts().to_dict()
                total = len(emotion_df)
                
                results[emotion] = {
                    'total_responses': total,
                    'child_emotion_distribution': child_emotions,
                    'contagion_rate': child_emotions.get(emotion, 0) / total if total > 0 else 0
                }
        
        return results
    
    def generate_hourly_contagion_data(self, df: pd.DataFrame,
                                      timestamp_col: str = 'timestamp',
                                      parent_emotion_col: str = 'parent_emotion',
                                      child_emotion_col: str = 'child_emotion') -> pd.DataFrame:
        """
        Generate hourly contagion patterns
        
        Args:
            df: DataFrame with timestamps and emotions
            timestamp_col: Column name for timestamp
            parent_emotion_col: Column name for parent emotion
            child_emotion_col: Column name for child emotion
            
        Returns:
            DataFrame with hourly contagion data
        """
        if timestamp_col not in df.columns:
            print(f"Warning: {timestamp_col} column not found. Using index as timestamp.")
            df['hour'] = df.index // 3600  # Assuming hourly intervals
        else:
            df['hour'] = pd.to_datetime(df[timestamp_col]).dt.hour
        
        hourly_data = []
        
        for hour in range(24):
            hour_df = df[df['hour'] == hour]
            if len(hour_df) > 0:
                contagion_stats = self.calculate_contagion_strength(
                    hour_df, parent_emotion_col, child_emotion_col
                )
                contagion_stats['hour'] = hour
                hourly_data.append(contagion_stats)
        
        return pd.DataFrame(hourly_data)


def main():
    """Example usage"""
    # Load preprocessed data with emotions
    df = pd.read_csv('data/processed/emotionalcontagion_worldcup.csv')
    
    # Initialize analyzer
    analyzer = ContagionAnalyzer()
    
    # Create contagion matrix
    matrix = analyzer.create_contagion_matrix(
        df, 
        parent_emotion_col='rtextem',  # parent emotion
        child_emotion_col='textem'      # child emotion
    )
    
    print("Contagion Matrix:")
    print(matrix)
    
    # Calculate probabilities
    prob_matrix = analyzer.calculate_contagion_probabilities(matrix)
    print("\nContagion Probabilities:")
    print(prob_matrix)
    
    # Calculate overall strength
    strength = analyzer.calculate_contagion_strength(
        df,
        parent_emotion_col='rtextem',
        child_emotion_col='textem'
    )
    print(f"\nOverall Contagion Rate: {strength['contagion_rate']:.2%}")
    
    # Save results
    matrix.to_csv('data/processed/contagion_matrix.csv')
    prob_matrix.to_csv('data/processed/contagion_probabilities.csv')


if __name__ == "__main__":
    main()

