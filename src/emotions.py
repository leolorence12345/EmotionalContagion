import nltk
# nltk.download()
import pandas as pd
from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
import requests
# from LeXmo import LeXmo
from csv import DictWriter
from googletrans import Translator
from io import StringIO

translator = Translator()
tweet_tokenizer = TweetTokenizer()

data='parent1660157493.csv' 
df=pd.read_csv(data)
parent_list = df["parent_text"].tolist()
child_list = df["child_text"].tolist()


def LeXmo(text):
    '''
      Takes text and adds if to a dictionary with 10 Keys  for each of the 10 emotions in the NRC Emotion Lexicon,
      each dictionay contains the value of the text in that emotions divided to the text word count
      INPUT: string
      OUTPUT: dictionary with the text and the value of 10 emotions
      '''
    response = requests.get('https://raw.github.com/dinbav/LeXmo/master/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    nrc = StringIO(response.text)
    LeXmo_dict = {'text': text, 'anger': [], 'anticipation': [], 'disgust': [], 'fear': [], 'joy': [], 'negative': [],
                  'positive': [], 'sadness': [], 'surprise': [], 'trust': []}

    emolex_df = pd.read_csv(nrc,
                            names=["word", "emotion", "association"],
                            sep=r'\t', engine='python')

    emolex_words = emolex_df.pivot(index='word',
                                   columns='emotion',
                                   values='association').reset_index()
    emolex_words.drop(emolex_words.index[0])

    emotions = emolex_words.columns.drop('word')

    stemmer = SnowballStemmer("english")
    
    document = tweet_tokenizer.tokenize(text) 

    word_count = len(document)
    rows_list = []
    for word in document:
        word = stemmer.stem(word.lower())

        emo_score = (emolex_words[emolex_words.word == word])
        rows_list.append(emo_score)

    df = pd.concat(rows_list)
    df.reset_index(drop=True)

    for emotion in list(emotions):
        LeXmo_dict[emotion] = df[emotion].sum() / word_count

    return LeXmo_dict

parent_emotion=[]
child_emotion=[] 
y=-1
for i,j in zip(parent_list,child_list):
            y+=1
            print(y)
            emo_p=LeXmo(i)
            emo_c=LeXmo(j)
            emo_p.pop('text', None)
            emo_c.pop('text', None)
            print(emo_p)
            print(emo_c)
            if(df["relation"].iloc[y] != "retweeted"):
                if(len(list(set(list(emo_p.values())))) != 1):
                    parent_value = max(emo_p, key=emo_p.get)
                    parent_emotion.append(parent_value)
                else:
                    parent_emotion.append("Neutral")
                if(len(list(set(list(emo_c.values())))) != 1):
                    child_value = max(emo_c, key=emo_c.get)
                    child_emotion.append(child_value)
                else:
                    child_emotion.append("Neutral")
            else :
                    if(len(list(set(list(emo_p.values())))) != 1):
                        parent_value = max(emo_p, key=emo_p.get)
                        parent_emotion.append(parent_value)
                        child_emotion.append(parent_value)
                    else:
                        parent_emotion.append("Neutral")
                        child_emotion.append("Neutral")
               
                        
                        
            if(y==200):
                break
    
df["parent_emotion"]= pd.Series(parent_emotion)
df["child_emotion"]= pd.Series(child_emotion)
df2 = df.to_csv("emotion_parent_child.csv")

print(parent_emotion)
print(child_emotion)
# with open('spreadsheet.csv','w',newline='', encoding='utf-8') as outfile:
#     writer = DictWriter(outfile, ('text', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative',
#                   'positive', 'sadness', 'surprise', 'trust'))
#     writer.writeheader()
#     writer.writerows(parent_emotion)