import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence


model = load_model('imdb_rnn_model.h5')



## streamlit app
st.title("ANN Model Prediction App")
st.write("This app predicts whether a customer will leave the bank based on their information.")
# Input fields
input_text = st.text_area("Enter your text here")

word_index = imdb.get_word_index();
reverse_word_index =  {value: key for (key, value) in word_index.items()}

### function to decode review
def decode_review(text):
    words = text.lower().split()
    text = [int(word) for word in words]
    text = [reverse_word_index.get(i - 3, '?') for i in text]
    return ' '.join(text)

### function to pre-process input
def preprocess_input(text):
    words = text.lower().split()
    encoded_review = [ word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


### prediction function
if st.button('Predict'):
    padded_review = preprocess_input(input_text)
    prediction = model.predict(padded_review)
    score = prediction[0][0] * 100
    sentiment = 'Postive' if prediction[0][0] > 0.7 else 'Average' if prediction[0][0] > 0.3  else 'Negative'
    
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Score: {score:.2f}%")

else:
    st.write("Please enter a review to predict sentiment.")
    



# example_review = "This movie was fantastic! I loved it."
# print(f"Review: {decode_review(example_review)}")
