from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import joblib
import numpy as np
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(layout="wide")
"""
# Welcome to CheapGPT!
"""
col1, col2, col3, col7, col8 = st.columns((1,1,1,1,1))
col4, col5, col6 = st.columns((1,1,1))
dom = "DS"
q = "Data Scientist (n.): Person who is better at statistics than any software engineer and better at software engineering than any statistician."

# Load the data from a CSV file
data = pd.read_csv('quotes_database.csv')

# Create a TF-IDF vectorizer to convert quotes into numerical vectors
tfidf_vectorizer = TfidfVectorizer()

# Convert the quotes into TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Quotes'])

# Compute the cosine similarity between all pairs of quotes
cosine_sim = cosine_similarity(tfidf_matrix)

# Define a function that takes a quote and category as input and returns the top 5 most similar quotes
def recommend_quotes(quote, category):
    # Find the index of the input quote in the data
    quote_index = data[data['Quotes'] == quote].index[0]
    
    # Find all quotes that belong to the same category as the input quote
    same_category_quotes = data[data['Category'] == category]
    
    # Compute the cosine similarity between the input quote and all quotes in the same category
    cosine_scores = cosine_sim[quote_index][same_category_quotes.index]
    
    # Sort the quotes by their cosine similarity to the input quote
    sorted_quotes = same_category_quotes.loc[same_category_quotes.index[cosine_scores.argsort()[::-1]]]
    
    # Return the top 5 most similar quotes
    return sorted_quotes['Quotes'].head(5)


with st.container():
    with col4:
        txt = st.text_area("Enter the Domain", key=1)
        if(st.button("Get Motivated!")):
            dom = txt
            
q_arr = []
for i in range(90):
    if(data["Category"][i] == dom):
        q_arr.append(i)

q_index = random.choice(q_arr)
q = data["Quotes"][q_index]

if dom not in ["DS", "AI", "DA"]:
    dom = "DS"

# Example usage:
recommended_quotes = recommend_quotes(q, dom)
recommended_authors = []
for i in recommended_quotes:
    recommended_authors.append(data.iloc[(data[data["Quotes"] == i].index.values)]["Author"].item())





with st.container():
    with col1:
        st.write(recommended_quotes.values[0])
        st.write(" - ", recommended_authors[0])
    with col2:
        st.write(recommended_quotes.values[1])
        st.write(" - ",recommended_authors[1])
    with col3:
        st.write(recommended_quotes.values[2])
        st.write(" - ",recommended_authors[2])
    with col7:
        st.write(recommended_quotes.values[3])
        st.write(" - ",recommended_authors[3])
    with col8:
        st.write(recommended_quotes.values[4])
        st.write(" - ",recommended_authors[4])


