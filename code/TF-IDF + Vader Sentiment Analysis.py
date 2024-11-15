#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:01:40 2024

@author: gloria
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Define a function to write objects to a pickle file
def write_pickle(obj, path, name):
    with open(os.path.join(path, f"{name}.pkl"), 'wb') as f:
        pickle.dump(obj, f)

# Modified xform_fun function
def xform_fun(df_in, m_in, n_in, sw_in, path_in):
    if sw_in == "tf":
        cv = CountVectorizer(ngram_range=(m_in, n_in), max_features=5000)
    else:
        cv = TfidfVectorizer(ngram_range=(m_in, n_in), use_idf=True, max_features=5000)
    
    df_in = df_in.fillna('')
    
    x_f_data_t = pd.DataFrame(cv.fit_transform(df_in).toarray())
    write_pickle(cv, path_in, sw_in)
    
    x_f_data_t.columns = cv.get_feature_names_out()
    return x_f_data_t

# Load the dataset
dataset_path = '/Users/gloria/Desktop/cleaned_sentiment_analysis.csv'

# Check if the file exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"File not found: {dataset_path}")

df = pd.read_csv(dataset_path)

# Apply the xform_fun function for feature extraction
tfidf = xform_fun(df['cleaned_lemmatized_text'], 1, 3, "tfidf", '/Users/gloria/Desktop')

# Perform Vader sentiment analysis
sid = SentimentIntensityAnalyzer()
df['vader_sentiment'] = df['cleaned_lemmatized_text'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Save the DataFrame with sentiment analysis results to a CSV file
output_path = '/Users/gloria/Desktop/vader_sentiment_analysis_results.csv'
df[['cleaned_lemmatized_text', 'vader_sentiment']].to_csv(output_path, index=False)

# Print a sample of the results
print(df[['cleaned_lemmatized_text', 'vader_sentiment']].head())
