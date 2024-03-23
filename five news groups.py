import pandas as pd

import re
import string
from kcontractions import kkcontractions
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

df=pd.read_csv('mydataset.csv')

df['type_id']=df.type.map({
    'business': 0,
    'entertainment': 1,
    'politics': 2,
    'sport': 3,
    'tech': 4
})

def remove_htmls(text):
    pattern=re.compile('<.*?>')
    return pattern.sub(r'',text)
    
def remove_urls(text):
    pattern=re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'',text)

wlm=WordNetLemmatizer()

def transformed_text(text):
    text=text.lower()
    text=remove_htmls(text)
    text=remove_urls(text)
    text=nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(wlm.lemmatize(i))
    return " ".join(y)
        

df['news']=df['news'].apply(transformed_text)

tfidf=TfidfVectorizer()

X=tfidf.fit_transform(df['news']).toarray()

Y=df['type_id'].values

X_train, X_test, y_train, y_test =train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2022,
    stratify=Y
)

nbCls=MultinomialNB()


nbCls.fit(X_train,y_train)
y_pred1=nbCls.predict(X_test)

print(classification_report(y_test, y_pred1))
print('Confusion Matrix :')
print("Confusion Matrix", confusion_matrix(y_test,y_pred1))

print("Accuracy:", accuracy_score(y_text,y_pred1))