import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.despine()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from wordcloud import WordCloud, STOPWORDS
from sklearn.naive_bayes import MultinomialNB
import re,string, nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.snowball import SnowballStemmer
import warnings
warnings.filterwarnings(action="ignore")
nltk.download('punkt')

def compAlg():
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

    X_train, X_test, y_train, y_test = train_test_split(np.array(df_new["text_clean"]),np.array(df_new["sentiment Label"]), test_size=0.25, random_state=0)
    tfidf = TfidfVectorizer(use_idf=True, tokenizer=word_tokenize, min_df=0.00002, max_df=0.70)
    X_train_tf = tfidf.fit_transform(X_train.astype('U'))
    X_test_tf = tfidf.transform(X_test.astype('U'))

    model = []
    accuracy = []
    acc={}

    DT = DecisionTreeClassifier(random_state=42)
    DT.fit(X_train_tf, y_train)
    predict = DT.predict(X_test_tf)
    DT_accuracy = DT.score(X_test_tf, y_test)
    print('DT Accuracy:', DT_accuracy)
    accuracy.append(DT_accuracy)
    model.append('DT')
    acc['DT']=round(DT_accuracy*100,2)
    """
    GB = GradientBoostingClassifier(random_state=42)
    GB.fit(X_train_tf, y_train)
    predict = GB.predict(X_test_tf)
    GB_accuracy = GB.score(X_test_tf, y_test)
    print('GB Accuracy:', GB_accuracy)
    accuracy.append(GB_accuracy)
    model.append('GB')
    """
    AB = AdaBoostClassifier(random_state=42)
    AB.fit(X_train_tf, y_train)
    predict = AB.predict(X_test_tf)
    AB_accuracy = AB.score(X_test_tf, y_test)
    print('AB Accuracy:', AB_accuracy)
    accuracy.append(AB_accuracy)
    model.append('AB')
    acc['AB'] = round(AB_accuracy * 100, 2)

    NB = MultinomialNB()
    NB.fit(X_train_tf, y_train)
    predict = NB.predict(X_test_tf)
    NB_accuracy = NB.score(X_test_tf, y_test)
    print('NB Accuracy:', NB_accuracy)
    accuracy.append(NB_accuracy)
    model.append('NB')
    acc['NB'] = round(NB_accuracy * 100, 2)
    """
    SVM = SVC(random_state=42)
    SVM.fit(X_train_tf, y_train)
    predict = SVM.predict(X_test_tf)
    SVM_accuracy = SVM.score(X_test_tf, y_test)
    print('SVM Accuracy:', SVM_accuracy)
    accuracy.append(SVM_accuracy)
    model.append('SVM')
    """
    XGB = XGBClassifier(random_state=42)
    XGB.fit(X_train_tf, y_train)
    predict = XGB.predict(X_test_tf)
    XGB_accuracy = XGB.score(X_test_tf, y_test)
    print('XGB Accuracy:', XGB_accuracy)
    accuracy.append(XGB_accuracy)
    model.append('XGB')
    acc['XGB'] = round(XGB_accuracy * 100, 2)

    LGB = LGBMClassifier(random_state=42)
    LGB.fit(X_train_tf, y_train)
    predict = LGB.predict(X_test_tf)
    LGB_accuracy = LGB.score(X_test_tf, y_test)
    print('LGB Accuracy:', LGB_accuracy)
    accuracy.append(LGB_accuracy)
    model.append('LGB')
    acc['LGB'] = round(LGB_accuracy * 100, 2)

    RF = RandomForestClassifier(random_state=42)
    RF.fit(X_train_tf, y_train)
    predict = RF.predict(X_test_tf)
    RF_accuracy = RF.score(X_test_tf, y_test)
    print('RF Accuracy:', RF_accuracy)
    accuracy.append(RF_accuracy)
    model.append('RF')
    acc['RF'] = round(RF_accuracy * 100, 2)

    plt.figure(figsize=(16, 5))
    plt.ylabel("Accuracy")
    plt.xlabel("Algorithms")
    ax=sns.barplot(x=model, y=accuracy, palette='Spectral')
    ax.bar_label(ax.containers[0])
    plt.savefig('static/pimg/algcomp.jpg')

    return acc



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

def fit_model(clf,x_train,y_train,x_test, y_test):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    return accuracy

#compAlg()


