import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data from a CSV file
data = pd.read_csv('quotes_database.csv')

# Create a TF-IDF vectorizer to convert quotes into numerical vectors
tfidf_vectorizer = TfidfVectorizer()

# Convert the quotes into TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(data['quote'])

# Compute the cosine similarity between all pairs of quotes
cosine_sim = cosine_similarity(tfidf_matrix)

# Define a function that takes a quote and category as input and returns the top 5 most similar quotes
def recommend_quotes(quote, category):
    # Find the index of the input quote in the data
    quote_index = data[data['quote'] == quote].index[0]
    
    # Find all quotes that belong to the same category as the input quote
    same_category_quotes = data[data['category'] == category]
    
    # Compute the cosine similarity between the input quote and all quotes in the same category
    cosine_scores = cosine_sim[quote_index][same_category_quotes.index]
    
    # Sort the quotes by their cosine similarity to the input quote
    sorted_quotes = same_category_quotes.loc[same_category_quotes.index[cosine_scores.argsort()[::-1]]]
    
    # Return the top 5 most similar quotes
    return sorted_quotes['quote'].head(5)

# Example usage:
recommended_quotes = recommend_quotes("Data Scientist (n.): Person who is better at statistics than any software engineer and better at software engineering than any statistician.", "DS")
print(recommended_quotes)