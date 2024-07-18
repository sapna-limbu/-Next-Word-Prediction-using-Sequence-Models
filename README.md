# -Next-Word-Prediction-using-Sequence-Models\


Predicting the next word in a sequence. Using a dataset of Spotify reviews, I trained a Bidirectional LSTM model to understand and generate text sequences. This project leverages the power of neural networks and deep learning to predict the next word given a seed text.

📊 Data Preparation:
The dataset was preprocessed and tokenized using the TensorFlow Keras Tokenizer. Each review was converted into sequences of words, which were then used to create input-output pairs for training.

🔧 Model Architecture:
I used a Sequential model with the following layers:

Embedding Layer: To convert word indices to dense vectors.
Bidirectional LSTM Layer: To capture dependencies in both forward and backward directions.
Dense Layer: With a softmax activation to predict the next word.

🌟 Results:
The model achieved impressive accuracy and can generate coherent next words based on the seed text provided. This is just the beginning, and I am excited to explore more advanced models and techniques to further enhance the predictions.

