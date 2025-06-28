#from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.despine()
plt.style.use("fivethirtyeight")
sns.set_style("darkgrid")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
import emoji
from wordcloud import WordCloud, STOPWORDS
import re,string, nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.snowball import SnowballStemmer
import warnings
warnings.filterwarnings(action="ignore")


def dataAnalysis():
    df = pd.read_csv("cyberbullying_tweets.csv")
    df = df.rename(columns={"tweet_text": "text", "cyberbullying_type": "sentiment"})
    #print(df.head(5))

    plt.figure(figsize=(14, 5))
    sns.countplot(x=df.sentiment, data=df, palette="mako")
    plt.savefig('static/pimg/tcnt.jpg')
    plt.clf()

    clean_tweet(df, "text")

    corpus = []

    df["text_clean"] = df["text"].apply(preprocess_tweet)

    df.drop_duplicates("text_clean", inplace=True)

    # removing other_cyberbullying category as it doesnot contribute much.
    df = df[df["sentiment"] != "other_cyberbullying"]

    # Calculating tweet length
    tweet_len = pd.Series([len(tweet.split()) for tweet in df["text"]])

    df["Length"] = df.text_clean.str.split().apply(len)
    plt.figure(figsize=(14, 7))
    sns.histplot(df[df["sentiment"] == "not_cyberbullying"]['Length'], color="g")
    plt.title("Distribution of Tweet Length for not_cyberbullying")
    plt.savefig('static/pimg/ncb.jpg')
    plt.clf()

    plt.figure(figsize=(14, 7))
    sns.histplot(df[df["sentiment"] == "gender"]["Length"], color="r")
    plt.title("Distribution of Tweet length for Gender")
    plt.savefig('static/pimg/gen.jpg')
    plt.clf()

    plt.figure(figsize=(14, 7))
    sns.histplot(df[df["sentiment"] == "religion"]["Length"], color="y")
    plt.title("Distribution of Tweet length for Religion")
    plt.savefig('static/pimg/rel.jpg')
    plt.clf()

    plt.figure(figsize=(14, 7))
    sns.histplot(df[df["sentiment"] == "age"]["Length"], color="b")
    plt.title('Distribution of Tweet length for Age')
    plt.savefig('static/pimg/age.jpg')
    plt.clf()

    plt.figure(figsize=(14, 7))
    sns.histplot(df[df["sentiment"] == "ethnicity"]["Length"], color="b")
    plt.title('Distribution of Tweet length for Ethnicity')
    plt.savefig('static/pimg/eth.jpg')
    plt.clf()

    plt.figure(figsize=(20, 20))
    wc = WordCloud(max_words=2000, min_font_size=10, height=800, width=1600,
                   background_color="white").generate(" ".join(df[df["sentiment"] == "not_cyberbullying"].text_clean))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('static/pimg/wncb.jpg')
    plt.clf()

    plt.figure(figsize=(20, 20))
    wc1 = WordCloud(max_words=2000, min_font_size=10, height=800, width=1600,
                    background_color="white").generate(" ".join(df[df["sentiment"] == "gender"].text_clean))
    plt.imshow(wc1, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('static/pimg/wgen.jpg')
    plt.clf()

    plt.figure(figsize=(20, 20))
    wc2 = WordCloud(max_words=2000, min_font_size=10, height=800, width=1600,
                    background_color="white").generate(" ".join(df[df["sentiment"] == "religion"].text_clean))
    plt.imshow(wc2, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('static/pimg/wrel.jpg')
    plt.clf()

    plt.figure(figsize=(20, 20))
    wc3 = WordCloud(max_words=2000, min_font_size=10, height=800, width=1600,
                    background_color="white").generate(" ".join(df[df["sentiment"] == "ethnicity"].text_clean))
    plt.imshow(wc3, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('static/pimg/weth.jpg')
    plt.clf()









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

#dataAnalysis()