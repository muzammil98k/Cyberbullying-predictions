import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.despine()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud, STOPWORDS
import re,string, nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import GridSearchCV
import pickle
import warnings
warnings.filterwarnings(action="ignore")

import nltk
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('stopwords')

def create_model():
    df = pd.read_csv("cyberbullying_tweets.csv")
    df = df.rename(columns={"tweet_text": "text", "cyberbullying_type": "sentiment"})

    clean_tweet(df, "text")

    corpus = []

    df["text_clean"] = df["text"].apply(preprocess_tweet)

    df.drop_duplicates("text_clean", inplace=True)

    # removing other_cyberbullying category as it doesnot contribute much.
    df = df[df["sentiment"] != "other_cyberbullying"]

    # Calculating tweet length
    tweet_len = pd.Series([len(tweet.split()) for tweet in df["text"]])

    labels = {"not_cyberbullying": 0, "gender": 1, "ethnicity": 2, "religion": 3, "age": 4}
    corpus, target_labels, target_names = (
        df['text_clean'], [labels[label] for label in df['sentiment']], df['sentiment'])
    df_new = pd.DataFrame({"text_clean": corpus, "sentiment Label": target_labels, "sentiment names": target_names})

    X_train, X_test, y_train, y_test = train_test_split(np.array(df_new["text_clean"]),
                                                        np.array(df_new["sentiment Label"]), test_size=0.25,
                                                        random_state=0)
    tfidf = TfidfVectorizer(use_idf=True, tokenizer=word_tokenize, min_df=0.00002, max_df=0.70)
    X_train_tf = tfidf.fit_transform(X_train.astype('U'))
    X_test_tf = tfidf.transform(X_test.astype('U'))

    RF = RandomForestClassifier(random_state=42)
    RF.fit(X_train_tf, y_train)
    predict = RF.predict(X_test_tf)
    RF_accuracy = RF.score(X_test_tf, y_test)
    print('RF Accuracy:', RF_accuracy)
    conf_matrix = confusion_matrix(y_test, predict)

    tr_acc=round(RF.score(X_train_tf, y_train) * 100,2)
    ts_acc=round(RF.score(X_test_tf, y_test) * 100,2)

    print(f"Training Accuracy Score: {RF.score(X_train_tf, y_train) * 100:.1f}%")
    print(f"Validation Accuracy Score: {RF.score(X_test_tf, y_test) * 100:.1f}%")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap='YlGnBu', fmt='g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion Matrix', fontsize=20, y=1.1)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.savefig('static/pimg/cmat.jpg', bbox_inches="tight")
    print(classification_report(y_test, predict))

    # save model
    pickle.dump(RF, open('CSM_model.pkl', 'wb'))
    pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
    msg="Model created successfully using Random Forest Classifier and TF-IDF vectorizer"
    return msg, tr_acc, ts_acc



# function for cleaning tweets
def clean_tweet(df,field):
    df[field] = df[field].str.replace(r"http\S+"," ")
    df[field] = df[field].str.replace(r"http"," ")
    df[field] = df[field].str.replace(r"@","at")
    df[field] = df[field].str.replace("#[A-Za-z0-9_]+", ' ')
    df[field] = df[field].str.replace(r"[^A-Za-z(),!?@\'\"_\n]"," ")
    df[field] = df[field].str.lower()
    return df

def preprocess_tweet(tweet):
    # Applying Lemmmatizer to remove tenses from texts.
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")
    STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm',
                      'im', 'll', 'y', 've', 'u', 'ur', 'don',
                      'p', 't', 's', 'aren', 'kp', 'o', 'kat',
                      'de', 're', 'amp', 'will'])

    tweet = re.sub(r"won\'t", "will not", tweet)
    tweet = re.sub(r"can\'t", "can not", tweet)
    tweet = re.sub(r"n\'t", " not", tweet)
    tweet = re.sub(r"\'re", " are", tweet)
    tweet = re.sub(r"\'s", " is", tweet)
    tweet = re.sub(r"\'d", " would",tweet)
    tweet = re.sub(r"\'ll", " will", tweet)
    tweet = re.sub(r"\'t", " not", tweet)
    tweet = re.sub(r"\'ve", " have", tweet)
    tweet = re.sub(r"\'m", " am", tweet)
    tweet = re.sub('[^a-zA-Z]',' ',tweet)
    #tweet = re.sub(emoji.get_emoji_regexp(),"",tweet)
    tweet = re.sub(r'[^\x00-\x7f]','',tweet)
    tweet = " ".join([stemmer.stem(word) for word in tweet.split()])
    tweet = [lemmatizer.lemmatize(word) for word in tweet.split() if not word in set(STOPWORDS)]
    tweet = ' '.join(tweet)
    return tweet


# create_model()