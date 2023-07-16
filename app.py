from flask import Flask, render_template, request
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from googletrans import Translator

app = Flask(__name__)

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

# Read the CSV file
data = pd.read_csv('cleantrainst.csv')

# Extract the text column from the CSV
texts = data['stop_word'].tolist()

# Replace NaN values with "missing"
texts = data['stop_word'].fillna("missing").tolist()

import os
print(os.getcwd())

# Initialize the tokenizer
tokenizer = Tokenizer()

# Fit the tokenizer on the texts
tokenizer.fit_on_texts(texts)

# Save the tokenizer as a pickle file
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Define the maximum sequence length
max_sequence_length = 30

# Load Word2Vec model
word2vec_model = Word2Vec.load('word2vec_model.bin')

# Load the trained classification model
model = load_model('classification_model.h5')

# Initialize the translator
translator = Translator(service_urls=['translate.google.com'])

def translate_text(text):
    translation = translator.translate(text, src='id', dest='en')
    return translation.text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # Translate the text from Indonesian to English
    translated_text = translate_text(text)

    # Load the tokenizer from the pickle file
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Preprocess the text
    sequences = tokenizer.texts_to_sequences([translated_text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # Perform the prediction
    prediction = model.predict(padded_sequences)
    predicted_class = 'Yes, it contains an occurrence of a Natural Disaster' if prediction[0] >= 0.2 else 'No, it does not contain an occurrence of a Natural Disaster'

    return render_template('result.html', text=translated_text, prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)