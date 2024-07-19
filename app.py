import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# Load model and tokenizer once when the script is run
#@st.cache(allow_output_mutation=True)
def load_resources():
    model = load_model('saved_model.keras')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_resources()

st.title("Next Word Prediction")

X = st.text_input("Enter a seed word")

if st.button("Predict"):
    for _ in range(20):
        sequence = tokenizer.texts_to_sequences([X])
        padded_sequence = pad_sequences(sequence, maxlen=23)
        predicted_word_index = np.argmax(model.predict(padded_sequence), axis=-1)
        if predicted_word_index:
            word = tokenizer.index_word.get(predicted_word_index[0], "")
            X = X + " " + word
            st.write(X)
        else:
            st.write("Model did not predict a word.")
        time.sleep(0.9)
