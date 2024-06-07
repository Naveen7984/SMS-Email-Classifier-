import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Load stopwords once for efficiency
stop_words = set(stopwords.words("english"))


def transform_text(text):
    # Convert to lower case
    text = text.lower()
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove special characters and stopwords, and perform stemming
    filtered_tokens = []  # Initialize an empty list to store filtered tokens

    # Iterate over each token in the 'tokens' list
    for token in tokens:
        # Check if the token consists of alphanumeric characters only and is not a stop-word
        if token.isalnum() and token not in stop_words:
            # Apply stemming to the token using the PorterStemmer and append it to the filtered_tokens list
            filtered_tokens.append(ps.stem(token))

    # Join the filtered tokens into a single string
    transformed_text = " ".join(filtered_tokens)
    return transformed_text


tfidf = pickle.load(open('Vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("E-Mail/SMS Classifier")

input_sms = st.text_input('Enter the Message')

if st.button('Predict'):
    if input_sms:
        # Preprocess
        transformed_sms = transform_text(input_sms)
        # Vectorized
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]
        # Display result
        if result == 1:
            st.header('Spam')
        else:
            st.header('Not Spam')
    else:
        st.write("Please enter a message for prediction.")
