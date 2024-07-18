import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the model and tokenizer
model_path = "saved_model.keras"
tokenizer_path = "tokenizer.pkl"

try:
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Ensure tokenizer loading and validation
try:
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    if not hasattr(tokenizer, 'word_index'):
        raise ValueError("Tokenizer does not have 'word_index' attribute.")
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    st.stop()

# Function to generate the next word
def predict_next_word(seed_text, model, tokenizer, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    predicted = np.argmax(predicted_probs, axis=-1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    return output_word

# Set page title
st.title("Next Word Prediction")

# Input text from user
input_text = st.text_area("Enter seed text:")

# Number of words to predict
num_words = st.number_input("Number of words to predict", min_value=1, max_value=100, value=1)

# Predict next words
if st.button("Predict"):
    max_sequence_len = model.input_shape[1]
    current_text = input_text
    for _ in range(num_words):
        next_word = predict_next_word(current_text, model, tokenizer, max_sequence_len)
        if next_word:
            current_text += " " + next_word
        else:
            break
    st.write("Predicted Text:")
    st.write(current_text)
