# Emotional Contagion Analysis through Tweets

A comprehensive machine learning pipeline to measure emotional contagion through tweets by integrating text and image metadata. This project analyzes how emotions spread through social media networks by examining parent-child tweet relationships.

##  Visualization: Emotional Contagion Flow

The following Sankey diagram visualizes the emotional contagion patterns, showing how emotions flow from parent tweets to child tweets:

![Emotional Contagion Sankey Diagram](Screenshot%202025-11-12%20204312.png)

**Key Insights from the Visualization:**
- **Fear** shows the strongest persistence (237,196 → 248,691), indicating high emotional contagion
- **Anger** demonstrates significant transformation, with many instances converting to fear or neutral states
- **Neutral** emotions tend to remain neutral, showing stability in non-emotional content
- Cross-emotion flows reveal complex emotional dynamics, with anger-to-fear being a notable transformation pattern

##  Project Overview

This project develops a model to measure emotional contagion through tweets by:
- **Data Collection**: Web scraping 17,000+ tweets with images and text
- **Data Preprocessing**: Cleaning using Regular Expressions and translation
- **Emotion Detection**: Extracting emotional data from facial expressions (DeepFace) and text (BERTweet, NLTK)
- **Contagion Analysis**: Analyzing emotion transfer patterns from parent to child tweets

### Key Features

-  **Multi-modal Emotion Detection**: Combines text-based (BERT) and image-based (DeepFace) emotion analysis
-  **Text Extraction from Images**: Uses Pytesseract OCR to extract text from tweet images
-  **BERTweet Integration**: Modified BERTweet model predicting 10 emotions with 85% accuracy
-  **NLTK-based Analysis**: LeXmo emotion analyzer using NRC Emotion Lexicon
-  **Contagion Pattern Analysis**: Quantifies emotion transfer in tweet networks

##  Dataset

- **Size**: 17,000+ tweets with images and text
- **Source**: Twitter API v2 (web scraping)
- **Preprocessing**: Regular Expression cleaning, translation to English
- **Structure**: Parent-child tweet relationships for contagion analysis

**Note**: Sample processed datasets are included in the repository:
- `emotionalcontagion_worldcup.csv` - Processed World Cup tweets with emotions
- `fifaworldcup2022.csv` - Raw World Cup tweet pairs
- `bertemotionresult_fifaworldcup.csv` - BERT emotion classification results

##  Project Structure

```
.
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_collection.py        # Twitter API data collection
│   ├── data_preprocessing.py     # Text cleaning and translation
│   ├── emotion_detection.py      # BERT and NLTK emotion detection
│   ├── image_processing.py       # DeepFace and Pytesseract
│   └── contagion_analysis.py    # Contagion pattern analysis
├── notebooks/                    # Jupyter notebooks for exploration
│   ├── extract_tweet.ipynb       # Tweet extraction
│   ├── emotionalcontagion_bert.ipynb  # BERT emotion classification
│   └── tweet_contagion.ipynb     # Contagion analysis
├── data/                         # Data directory
│   ├── raw/                      # Raw scraped data
│   └── processed/                # Processed datasets
├── config/                       # Configuration files
│   └── config.example.ini        # Example config (copy to config.ini)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

##  Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR installed on your system
  - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
  - **Linux**: `sudo apt-get install tesseract-ocr`
  - **Mac**: `brew install tesseract`

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd IIM_FINAL
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

5. **Configure API keys**
   ```bash
   cp config/config.example.ini config/config.ini
   # Edit config/config.ini with your Twitter API credentials
   ```

##  Usage

### 1. Data Collection

Collect tweets from Twitter API:

```python
from src.data_collection import TwitterDataCollector
import datetime

collector = TwitterDataCollector(
    consumer_key="your_key",
    consumer_secret="your_secret"
)

start_time = datetime.datetime(2022, 12, 18, 9, 0, 0, 0, datetime.timezone.utc)
end_time = datetime.datetime(2022, 12, 18, 14, 0, 0, 0, datetime.timezone.utc)
query = "#WorldCupFinal OR #FIFAWorldCup"

count = collector.collect_tweets(
    query=query,
    start_time=start_time,
    end_time=end_time,
    output_file='data/raw/tweets.txt'
)
```

### 2. Data Preprocessing

Clean and translate tweets:

```python
from src.data_preprocessing import TweetPreprocessor
import json

preprocessor = TweetPreprocessor()

# Load tweets
tweets = []
with open('data/raw/tweets.txt', 'r') as f:
    for line in f:
        tweets.append(json.loads(line))

# Preprocess
df = preprocessor.preprocess_dataset(tweets, translate=True, clean=True)
df.to_csv('data/processed/preprocessed_tweets.csv', index=False)
```

### 3. Emotion Detection

#### Using BERT:

```python
from src.emotion_detection import BERTEmotionClassifier

classifier = BERTEmotionClassifier()
emotion = classifier.predict_emotion("I'm so excited about this!")
print(emotion)  # Output: 'joy'
```

#### Using LeXmo (NLTK):

```python
from src.emotion_detection import LeXmoEmotionAnalyzer

analyzer = LeXmoEmotionAnalyzer()
emotion = analyzer.get_dominant_emotion("I'm so excited about this!")
print(emotion)  # Output: 'joy'
```

### 4. Image Processing

Extract emotions and text from images:

```python
from src.image_processing import ImageMetadataExtractor

extractor = ImageMetadataExtractor()
metadata = extractor.extract_metadata('path/to/image.jpg')

print(f"Emotion: {metadata['emotion']}")
print(f"Text: {metadata['extracted_text']}")
```

### 5. Contagion Analysis

Analyze emotion transfer patterns:

```python
from src.contagion_analysis import ContagionAnalyzer
import pandas as pd

# Load data with emotions
df = pd.read_csv('data/processed/emotionalcontagion_worldcup.csv')

analyzer = ContagionAnalyzer()

# Create contagion matrix
matrix = analyzer.create_contagion_matrix(
    df,
    parent_emotion_col='rtextem',
    child_emotion_col='textem'
)

# Calculate contagion probabilities
prob_matrix = analyzer.calculate_contagion_probabilities(matrix)

# Calculate overall strength
strength = analyzer.calculate_contagion_strength(df)
print(f"Contagion Rate: {strength['contagion_rate']:.2%}")
```

### Complete Pipeline

Run the complete pipeline:

```python
# 1. Collect data
from src.data_collection import TwitterDataCollector
collector = TwitterDataCollector(...)
collector.collect_tweets(...)

# 2. Preprocess
from src.data_preprocessing import TweetPreprocessor
preprocessor = TweetPreprocessor()
df = preprocessor.preprocess_dataset(tweets)

# 3. Detect emotions
from src.emotion_detection import EmotionDetector
detector = EmotionDetector(use_bert=True, use_lexmo=True)
df['parent_emotion'] = df['english_parent_text'].apply(
    lambda x: detector.detect_emotion(x, method='bert')
)
df['child_emotion'] = df['english_child_text'].apply(
    lambda x: detector.detect_emotion(x, method='bert')
)

# 4. Analyze contagion
from src.contagion_analysis import ContagionAnalyzer
analyzer = ContagionAnalyzer()
matrix = analyzer.create_contagion_matrix(df)
```

##  Results

### Model Performance

- **BERT Emotion Classification**: 85% accuracy
- **Emotions Detected**: anger, disgust, fear, joy, sadness, surprise, neutral (7 emotions)
- **LeXmo Emotions**: 10 emotions (anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust)

### Contagion Metrics

The analysis provides:
- Contagion matrix showing emotion transfer patterns
- Contagion probabilities for each emotion pair
- Overall contagion rate
- Hourly contagion patterns

##  Configuration

Edit `config/config.ini` to set:
- Twitter API credentials
- Tesseract OCR path
- Model configurations
- Data paths

##  Notebooks

Explore the Jupyter notebooks in the `notebooks/` directory:
- `extract_tweet.ipynb`: Tweet extraction and preprocessing
- `emotionalcontagion_bert.ipynb`: BERT-based emotion classification
- `tweet_contagion.ipynb`: Contagion pattern analysis

##  Technologies Used

- **NLP**: Transformers (BERT), NLTK, NRC Emotion Lexicon
- **Computer Vision**: DeepFace, OpenCV, Pytesseract
- **Data Processing**: Pandas, NumPy
- **API**: Twitter API v2 (Twarc2)
- **Translation**: Googletrans

##  License

[Add your license here]

##  Authors

[Add your name/team here]

##  Acknowledgments

- NRC Emotion Lexicon
- HuggingFace Transformers
- DeepFace library
- Twitter API


##  Quick Start

For a quick start guide, see [QUICKSTART.md](QUICKSTART.md)

## Additional Resources

- [Quick Start Guide](QUICKSTART.md) - Get started in 5 minutes
- [Example Usage](example_usage.py) - Code examples and demonstrations
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute to the project

---

**Note**: Make sure to set up your Twitter API credentials in `config/config.ini` before running the data collection scripts. If you're using existing data files, you can skip the data collection step.

