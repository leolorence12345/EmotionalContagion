# Repository Structure

This document outlines the complete structure of the Emotional Contagion Analysis repository.

## Directory Structure

```
IIM_FINAL/
│
├── src/                          # Main source code modules
│   ├── __init__.py              # Package initialization
│   ├── data_collection.py       # Twitter API data collection
│   ├── data_preprocessing.py    # Text cleaning and translation
│   ├── emotion_detection.py     # BERT and NLTK emotion detection
│   ├── image_processing.py      # DeepFace and Pytesseract
│   └── contagion_analysis.py    # Contagion pattern analysis
│
├── notebooks/                    # Jupyter notebooks for exploration
│   ├── extract_tweet.ipynb      # Tweet extraction workflow
│   ├── emotionalcontagion_bert.ipynb  # BERT emotion classification
│   ├── tweet_contagion.ipynb    # Contagion analysis
│   └── [other notebooks]         # Additional analysis notebooks
│
├── data/                         # Data directory
│   ├── raw/                      # Raw scraped tweet data
│   │   └── tweets_worldcupfinal.txt  # Example raw data
│   └── processed/                # Processed datasets
│       └── [processed CSV files]  # Output from pipeline
│
├── config/                       # Configuration files
│   └── config.example.ini        # Example config template
│
├── main.py                       # Main pipeline script
├── example_usage.py              # Example usage demonstrations
├── setup.py                      # Package setup script
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── .gitattributes               # Git attributes for line endings
│
├── README.md                     # Main project documentation
├── QUICKSTART.md                 # Quick start guide
├── CONTRIBUTING.md               # Contribution guidelines
├── LICENSE                       # MIT License
└── REPOSITORY_STRUCTURE.md       # This file
```

## Module Descriptions

### Source Modules (`src/`)

1. **data_collection.py**
   - `TwitterDataCollector`: Collects tweets from Twitter API v2
   - Handles authentication and rate limiting
   - Saves tweets to JSON format

2. **data_preprocessing.py**
   - `TweetPreprocessor`: Cleans and preprocesses tweet text
   - Regular expression-based cleaning
   - Translation to English using googletrans
   - Parent-child relationship extraction

3. **emotion_detection.py**
   - `BERTEmotionClassifier`: BERT-based emotion classification
   - `LeXmoEmotionAnalyzer`: NLTK-based emotion analysis using NRC Lexicon
   - `EmotionDetector`: Combined emotion detection interface

4. **image_processing.py**
   - `ImageEmotionExtractor`: DeepFace-based facial emotion detection
   - `ImageTextExtractor`: Pytesseract OCR for text extraction
   - `ImageMetadataExtractor`: Combined image analysis

5. **contagion_analysis.py**
   - `ContagionAnalyzer`: Analyzes emotion transfer patterns
   - Creates contagion matrices
   - Calculates contagion probabilities
   - Generates hourly patterns

### Notebooks (`notebooks/`)

- **extract_tweet.ipynb**: Complete workflow for tweet extraction
- **emotionalcontagion_bert.ipynb**: BERT emotion classification pipeline
- **tweet_contagion.ipynb**: Contagion analysis and visualization

### Configuration (`config/`)

- **config.example.ini**: Template for API keys and settings
  - Twitter API credentials
  - Tesseract OCR path
  - Model configurations

### Data Files

The repository includes sample datasets:
- `emotionalcontagion_worldcup.csv`: Processed tweets with emotions
- `fifaworldcup2022.csv`: Raw tweet pairs
- `bertemotionresult_fifaworldcup.csv`: BERT classification results

## Usage Flow

1. **Data Collection** → `src/data_collection.py`
2. **Preprocessing** → `src/data_preprocessing.py`
3. **Emotion Detection** → `src/emotion_detection.py`
4. **Image Processing** (optional) → `src/image_processing.py`
5. **Contagion Analysis** → `src/contagion_analysis.py`

Or use the complete pipeline: `python main.py`

## Key Features

✅ Modular architecture for easy extension
✅ Comprehensive documentation
✅ Example usage scripts
✅ Jupyter notebooks for exploration
✅ Configuration management
✅ Ready for GitHub deployment

## Next Steps

1. Review `README.md` for full documentation
2. Check `QUICKSTART.md` for quick setup
3. Run `example_usage.py` to see demonstrations
4. Explore notebooks for detailed workflows

