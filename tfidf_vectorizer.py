import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.preprocess import cleantext
import pickle

def create_model():
    # Load dataset
    dataset = pd.read_csv('../cyberbullying_tweets.csv')

    # Data preprocessing
    dataset.drop_duplicates(subset='tweet_text', keep='first', inplace=True)

    # Clean text
    dataset['tweet_text'] = dataset['tweet_text'].apply(cleantext)

    tfidf = TfidfVectorizer(max_features=5000)
    tfidf.fit_transform(dataset['tweet_text'])

    with open('tfidf_vectorizer.pkl', 'wb') as file:
        pickle.dump(tfidf, file)