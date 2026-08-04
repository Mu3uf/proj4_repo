import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nltk.download('stopwords')

df = pd.read_csv('data/clean_raw.csv')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)      
    text = re.sub(r'@\w+', '', text)                 
    text = re.sub(r'#', '', text)                     
    text = re.sub(r'[^a-z\s]', '', text)              
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)
df = df.dropna(subset=['clean_text'])
df = df[df['clean_text'].str.strip() != '']

df.to_csv('data/preprocessed.csv', index=False)
print(df[['text', 'clean_text']].head(10))