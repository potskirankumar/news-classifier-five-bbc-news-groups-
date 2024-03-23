import streamlit as st
import nltk
import re
from nltk.stem import WordNetLemmatizer
import pickle
import string
from nltk.corpus import stopwords



wlm=WordNetLemmatizer()

def remove_htmls(text):
    pattern=re.compile('<.*?>')
    return pattern.sub(r'',text)

def remove_urls(text):
    pattern=re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'',text)


def transformed_text(text):
    text = text.lower()
    text = remove_htmls(text)
    text = remove_urls(text)
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(wlm.lemmatize(i))
    return " ".join(y)


tfidf = pickle.load(open('Vectorizednb.pkl','rb'))
model = pickle.load(open('modelnb.pkl','rb'))

st.title("News Classifier")

input_text = st.text_input("Enter the news")

if st.button("Predict"):
    transformed_inp = transformed_text(input_text)
    vector_inp = tfidf.transform([transformed_inp])
    result = model.predict(vector_inp)[0]
    if result == 0:
        st.header("Business")
    elif result == 1:
        st.header("Entertainment")
    elif result == 2:
        st.header("Politics")
    elif result == 3:
        st.header("Politics")
    elif result == 4:
        st.header("Tech")