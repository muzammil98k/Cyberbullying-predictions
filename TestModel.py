import pickle
import re
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



def preprocess_tweet(tweet):
    # Define global STOPWORDS
    STOPWORDS = set(stopwords.words('english'))
    STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 'im', 'll', 'y', 've', 'u', 'ur', 'don',
                      'p', 't', 's', 'aren', 'kp', 'o', 'kat', 'de', 're', 'amp', 'will'])

    # Label dictionary
    label_map = {0: "not_cyberbullying", 1: "gender", 2: "ethnicity", 3: "religion", 4: "age"}


    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")

    tweet = re.sub(r"won\'t", "will not", tweet)
    tweet = re.sub(r"can\'t", "can not", tweet)
    tweet = re.sub(r"n\'t", " not", tweet)
    tweet = re.sub(r"\'re", " are", tweet)
    tweet = re.sub(r"\'s", " is", tweet)
    tweet = re.sub(r"\'d", " would", tweet)
    tweet = re.sub(r"\'ll", " will", tweet)
    tweet = re.sub(r"\'t", " not", tweet)
    tweet = re.sub(r"\'ve", " have", tweet)
    tweet = re.sub(r"\'m", " am", tweet)
    tweet = re.sub('[^a-zA-Z]', ' ', tweet)
    tweet = re.sub(r'[^\x00-\x7f]', '', tweet)

    tweet = " ".join([stemmer.stem(word) for word in tweet.split()])
    #tweet = [lemmatizer.lemmatize(word) for word in tweet if word not in STOPWORDS]
    return ' '.join(tweet)

def test_model(tweet_text):
    # Load model and vectorizer
    model = pickle.load(open('../CSM_model.pkl', 'rb'))
    vectorizer = pickle.load(open('../tfidf.pkl', 'rb'))

    # Preprocess the tweet
    cleaned = preprocess_tweet(tweet_text)
    print(cleaned)

    # Vectorize
    vectorized = vectorizer.transform([cleaned])

    # Predict
    prediction = model.predict(vectorized)[0]
    label = label_map[prediction]

    return f"Predicted class: {label}"

# Example usage
#print(test_model("Idiot!! Nobody likes you"))
