from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import pickle
import io
import base64
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load your trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

port_stemmer = PorterStemmer()

@app.route('/')
def index():
    return render_template('index_sms.html', input_sms=None, prediction=None, error=None)

@app.route('/predict_sms', methods=['POST'])
def predict_sms():
    input_sms = request.form['input_sms']

    if input_sms.strip() != '':
        # Preprocess the SMS
        transform_text = clean_text(input_sms)

        # Vectorize the SMS
        vector_input = tfidf.transform([transform_text])

        # Make prediction
        result = model.predict(vector_input)

        # Display result
        if result == 1:
            prediction = "Spam"
        else:
            prediction = "Not Spam"

        return render_template('index_sms.html', input_sms=input_sms, prediction=prediction, error=None)
    else:
        return render_template('index_sms.html', input_sms=None, prediction=None, error='Please enter a message.')

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = word_tokenize(text)  # Create tokens
    text = " ".join(text)  # Join tokens
    text = [char for char in text if char not in string.punctuation]  # Remove punctuations
    text = ''.join(text)  # Join the letters
    text = [char for char in text if char not in re.findall(r"[0-9]", text)]  # Remove Numbers
    text = ''.join(text)  # Join the letters
    text = [word.lower() for word in text.split() if word.lower() not in stop_words]  # Remove common English words
    text = ' '.join(text)  # Join the letters
    text = list(map(lambda x: port_stemmer.stem(x), text.split()))
    return " ".join(text)

if __name__ == '__main__':
    app.run(debug=True)
