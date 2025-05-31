import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

def load_data(filepath):
    """Wczytuje i czyści dane z tweetami"""
    df = pd.read_csv(filepath)
    
    # Czyszczenie tekstu
    df['clean_text'] = df['text'].apply(clean_text)
    
    return df

def clean_text(text):
    """Funkcja czyszcząca pojedynczy tweet"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.lower()
    
    # Usuwanie stop words i lematyzacja
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)
