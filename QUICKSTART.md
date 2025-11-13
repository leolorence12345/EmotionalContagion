# Quick Start Guide

Get up and running with the Emotional Contagion Analysis toolkit in 5 minutes!

## Prerequisites Check

Before starting, ensure you have:
- ✅ Python 3.8+ installed
- ✅ Twitter API credentials (optional, for data collection)
- ✅ Tesseract OCR installed (for image text extraction)

## Installation Steps

### 1. Clone and Setup

```bash
# Navigate to project directory
cd IIM_FINAL

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download NLTK Data

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 3. Configure API Keys (Optional)

```bash
# Copy example config
cp config/config.example.ini config/config.ini

# Edit config/config.ini with your Twitter API credentials
```

## Quick Examples

### Example 1: Detect Emotion from Text

```python
from src.emotion_detection import BERTEmotionClassifier

classifier = BERTEmotionClassifier()
emotion = classifier.predict_emotion("I'm so excited!")
print(emotion)  # Output: 'joy'
```

### Example 2: Run Complete Pipeline

```python
python main.py --query "#WorldCupFinal" \
               --start-time "2022-12-18 09:00:00" \
               --end-time "2022-12-18 14:00:00" \
               --use-bert
```

### Example 3: Analyze Existing Data

If you already have preprocessed data:

```python
import pandas as pd
from src.contagion_analysis import ContagionAnalyzer

# Load your data
df = pd.read_csv('data/processed/emotionalcontagion_worldcup.csv')

# Analyze
analyzer = ContagionAnalyzer()
matrix = analyzer.create_contagion_matrix(
    df,
    parent_emotion_col='rtextem',
    child_emotion_col='textem'
)

print(matrix)
```

## Common Issues

### Issue: Tesseract not found
**Solution**: Install Tesseract and set path in `config/config.ini`

### Issue: DeepFace import error
**Solution**: `pip install deepface tensorflow`

### Issue: Twitter API rate limits
**Solution**: Use existing data files or wait between requests

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Explore [example_usage.py](example_usage.py) for more examples
3. Check out notebooks in `notebooks/` directory

## Need Help?

- Check the main README.md
- Review example_usage.py
- Examine the notebooks for detailed workflows

