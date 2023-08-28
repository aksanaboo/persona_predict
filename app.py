import streamlit as st
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import torch
import numpy as np
from transformers import BertTokenizer



# Initialize NLTK resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load the serialized model
model_filename = 'mbti500_SVCmodel.sav'
with open(model_filename, 'rb') as model_file:
    text_clf = pickle.load(model_file)

# Preprocessing function
def preprocess_text(text):
    # Remove punctuation, URLs, and stopwords
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = ' '.join([word for word in text.split() if word.lower() not in stopwords.words('english')])

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return text

# Create the Streamlit app
def main():
    st.title("MBTI Personality Type Predictor")
    st.write("Enter your text and we'll predict your personality type!")

    # User input
    user_input = st.text_area("Enter your text here:")

    if st.button("Predict"):
        # Preprocess user input
        processed_input = preprocess_text(user_input)

        # Predict using the first model (SVC-based)
        prediction_svc = text_clf.predict([processed_input])[0]
        st.write(f"Predicted Personality Type : {prediction_svc}")

if __name__ == "__main__":
    main()
