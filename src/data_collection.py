"""
Twitter Data Collection Module
Handles web scraping of tweets with images and text using Twitter API
"""

import json
import datetime
from twarc import Twarc2, expansions
from TwitterAPI import TwitterAPI


class TwitterDataCollector:
    """Collects tweets from Twitter API"""
    
    def __init__(self, consumer_key=None, consumer_secret=None, 
                 access_token_key=None, access_token_secret=None, 
                 bearer_token=None):
        """
        Initialize Twitter API client
        
        Args:
            consumer_key: Twitter API consumer key
            consumer_secret: Twitter API consumer secret
            access_token_key: Twitter API access token key
            access_token_secret: Twitter API access token secret
            bearer_token: Twitter API bearer token (alternative to OAuth)
        """
        if bearer_token:
            self.client = Twarc2(bearer_token=bearer_token)
        elif consumer_key and consumer_secret:
            self.client = Twarc2(consumer_key=consumer_key,
                                consumer_secret=consumer_secret)
        else:
            raise ValueError("Either bearer_token or consumer_key/secret must be provided")
    
    def collect_tweets(self, query, start_time, end_time, 
                      output_file='tweets.txt', max_results=100):
        """
        Collect tweets based on query and time range
        
        Args:
            query: Twitter search query (e.g., "#WorldCupFinal OR #FIFAWorldCup")
            start_time: Start datetime in UTC
            end_time: End datetime in UTC
            output_file: Output file path for saving tweets
            max_results: Maximum number of results per page
            
        Returns:
            Number of tweets collected
        """
        search_results = self.client.search_all(
            query=query, 
            start_time=start_time, 
            end_time=end_time, 
            max_results=max_results
        )
        
        tweet_count = 0
        for page in search_results:
            result = expansions.flatten(page)
            with open(output_file, 'a+', encoding='utf-8') as filehandle:
                for tweet in result:
                    filehandle.write('%s\n' % json.dumps(tweet))
                    tweet_count += 1
        
        return tweet_count
    
    def load_tweets_from_file(self, file_path):
        """
        Load tweets from JSON file
        
        Args:
            file_path: Path to JSON file containing tweets
            
        Returns:
            List of tweet dictionaries
        """
        tweets = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    tweets.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return tweets


def main():
    """Example usage"""
    # Example: Collect tweets for World Cup Final
    collector = TwitterDataCollector(
        consumer_key="your_consumer_key",
        consumer_secret="your_consumer_secret"
    )
    
    start_time = datetime.datetime(2022, 12, 18, 9, 0, 0, 0, datetime.timezone.utc)
    end_time = datetime.datetime(2022, 12, 18, 14, 0, 0, 0, datetime.timezone.utc)
    query = "#WorldCupFinal OR #ArgentinaVsFrance OR #worldcup OR #FIFAWorldCup OR #FIFAWorldCupQatar2022"
    
    count = collector.collect_tweets(
        query=query,
        start_time=start_time,
        end_time=end_time,
        output_file='data/raw/tweets_worldcupfinal.txt'
    )
    print(f"Collected {count} tweets")


if __name__ == "__main__":
    main()

