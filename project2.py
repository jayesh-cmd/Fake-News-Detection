# FAKE NEWS DETECTION MODEL -- 

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix
import joblib

nltk.download('stopwords')
nltk.download('wordnet')

# READING CSV FILE & MAKING IT DATAFRAME
data = pd.read_csv(r"AI ML LEARN\news_data.csv")
df = pd.DataFrame(data)

# WordNetLemmatizer - THIS WILL REDUCE THE WORDS AND KEEP THEM MEANINGFULL , Exe. RUNNING - RUN
Lemmatizer = WordNetLemmatizer()

# IT WILL REMOVE STOPWORDS - common words (like "the," "a," "is")
stop_words = set(stopwords.words('english'))

# FUNCTION TO CLEAN TEXT.
def cleantext(text):
    text = text.lower()
    text = re.sub(r'\d+','',text) # REMOVE EXTRA SPACES
    text = text.translate(str.maketrans('','', string.punctuation)) # REMOVE PUNCTUATIONS
    text = ' '.join([Lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

df['cleantext'] = df['title'].apply(cleantext)

# TfidfVectorizer - THIS WILL CONVERT TEXTS INTO NUMERICAL FORM
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df['cleantext'])
y = df['label']

# TRAIN TEST SPLIT
X_train , X_test , Y_train , Y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

# USING LOGISTIC REGRESSION , BECAUSE WE HAVE TO PREDICT TRUE OR FAKE
model = LogisticRegression()
model.fit(X_train , Y_train)

# ACCURACY SCORE -
print("Model Training Score : " , model.score(X_train , Y_train)) # PRINTING TRAINING SCORE
print("Model Testing Score : " , model.score(X_test , Y_test) , "\n") # PRINTING TESTING SCORE

y_pred = model.predict(X_test)

# SAVING MODEL USING JOBLIB
joblib.dump(model , "fake_news_detector.pkl")
joblib.dump(vectorizer , "tfidf_vectorizer.pkl")
print("Model Saved Successfully !")

# LOAD MODEL 
model = joblib.load("fake_news_detector.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
print("Model Loaded Successfully !")

# FUNCTION TO PREDICT NEWS 
def predict_news(newstext, threshold=0.8):  # Default threshold 0.8
    newsvector = vectorizer.transform([newstext])
    prediction_prob = model.predict_proba(newsvector)  # Get probabilities

    if prediction_prob[0][0] > threshold:
        return "Fake News"
    else:
        return "Real News"


news = "Drinking hot water every morning makes you immune to all diseases."
print(predict_news(news))
