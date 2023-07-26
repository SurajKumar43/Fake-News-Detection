import streamlit as st
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sklearn


st.title('Fake News Detection')
news = pickle.load(open('test.pkl','rb'))
news_list = news['title'].values

selected_news = st.selectbox(
    'Enter the News Title:',
      news_list)

cv = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def fake_news(sample_news):
      sample_news = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_news)
      sample_news = sample_news.lower()
      sample_news_words = sample_news.split()
      sample_news_words = [word for word in sample_news_words if word not in set(stopwords.words('english'))]
      ps = PorterStemmer()
      final_news = [ps.stem(word) for word in sample_news_words]
      final_news = ' '.join(final_news)
      
      temp = cv.transform([final_news]).toarray()
      return model.predict(temp)

if st.button('Predict'):
      result = fake_news(selected_news)
      if result == 1:
            st.header('This is a Fake news!!!')
      else:
            st.header('This is a Real news.')