#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:51:26 2024

@author: yu
"""

"""
Data Pre-processing

"""
#import kagglehub
import re
import pandas as pd
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Download the latest version of the dataset
#path = kagglehub.dataset_download("suchintikasarkar/sentiment-analysis-for-mental-health")

#print("Path to dataset files:", path)
#path = 'C:/Users/meisn\OneDrive/Documents/GitHub/NLP_Fall2024_DepressionAnxiety/Data/Raw Data'
#dataset_path = os.path.join(path, 'Data.csv')
dataset = pd.read_csv('data.csv')
dataset['statement'] = dataset['statement'].fillna('').astype(str)

lemmatizer = WordNetLemmatizer()

# Modified text cleaning function
def clean_txt(var_in):
    # Remove #NAME? and similar noise
    var_in = re.sub(r'#NAME\?+', '', var_in)  # Remove #NAME? or similar content
    var_in = re.sub(r"[^A-Za-z\s']+", " ", var_in)  # Remove other meaningless characters
    var_in = var_in.strip().lower()

    # Handle empty values
    if var_in == '' or var_in == 'nan' or var_in == 'null':  
        return ''  # Or return 'unknown'

    return var_in

# Function to get wordnet part of speech (POS)
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.ADJ

# Lemmatization and cleaning function
def clean_and_lemmatize(text):
    cleaned_text = clean_txt(text)
    if cleaned_text == '':  # If the cleaned text is empty, return it as empty
        return ''
    
    tokens = word_tokenize(cleaned_text)
    pos_tags = nltk.pos_tag(tokens)
    lemmatized_tokens = []
    for token, tag in pos_tags:
        wordnet_pos = get_wordnet_pos(tag)
        lemmatized_tokens.append(lemmatizer.lemmatize(token, pos=wordnet_pos))
    
    lemmatized_text = " ".join(lemmatized_tokens)
    
    return lemmatized_text

# Process the text column in the dataset
dataset['cleaned_lemmatized_text'] = dataset['statement'].apply(clean_and_lemmatize)

dataset = dataset[dataset['cleaned_lemmatized_text'].str.strip() != '']

# Show the processed data
print(dataset[['statement', 'cleaned_lemmatized_text']].head())
