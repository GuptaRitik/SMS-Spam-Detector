#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize PorterStemmer for stemming words    
ps = PorterStemmer()

def transform_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text into individual words
    text = nltk.word_tokenize(text)

    # Remove any non-alphanumeric characters from the list of words
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove any stopwords (common words that typically don't carry much meaning) and punctuation marks
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stem each word (i.e. convert it to its base form)
    for i in text:
        y.append(ps.stem(i))

    # Convert the list of stemmed words back into a string
    return " ".join(y)

tfidf = pickle.load(open('Pickle Objects/vectorizer.pkl','rb'))
model = pickle.load(open('Pickle Objects/model.pkl','rb'))

# Define the app title
st.title("SMS Spam Detection")

# Add a text input field for the user to enter their message
message = st.text_area("Enter a message:")

# Add a button to submit the message for prediction
if st.button("Predict"):
    # Clean the message using the loaded text cleaning function
    cleaned_message = transform_text(message)

    # Vectorize the cleaned message using the loaded vectorizer
    vectorized_message = tfidf.transform([cleaned_message])

    # Make a prediction using the loaded model
    prediction = model.predict(vectorized_message)[0]

    # Display the prediction result
    if prediction == 1:
        st.warning("This message is spam!")
    else:
        st.success("This message is not spam.")

# Add a section for the user to choose from pre-defined SMS messages
st.sidebar.subheader("Sample Messages")

# Define the pre-defined messages
predefined_messages = [
    "Hi there! How are you?",
    "What are you up to?",
    "Can you call me later?",
    "Click here to claim your prize now!!!",
    "Congratulations! You have been selected for a free luxury vacation to Hawaii. Click here to claim your prize now!",
    "URGENT: Your account is about to expire. Please update your information.",
    "Congratulations! You have been selected to win a brand new iPhone. Click here to claim your prize."
]

# Add a selectbox for the user to choose from the pre-defined messages
selected_message = st.sidebar.selectbox("Select a message:", predefined_messages)

# Display the selected message
st.sidebar.write(">", selected_message)


# In[ ]:




