import pandas as pd
import json

# from googletrans import Translator

# translator = Translator()
 
from translate import Translator
translator= Translator(to_lang="en")

df = pd.read_json('tweets.txt',lines=True)
print(df.shape[0]-1)
# for i in range(2000):
#     print(i)
#     df['text'].iloc[i] = translator.translate(df['text'].iloc[i])
#     df['id'].iloc[i]= str(df['id'].iloc[i])+'t'
# df = df.to_csv('translated_tweet2.csv')
  