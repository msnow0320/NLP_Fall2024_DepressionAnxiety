#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:21:45 2024

@author: yanyuge
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 2024

Confusion Matrix Analysis for Anxiety and Depression Classification
"""

# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the dataset
dataset_path = '/Users/yanyuge/Downloads/updated_sentiment_analysis_results.csv'
df = pd.read_csv(dataset_path)

# Filter data for Anxiety and Depression
model_data = df[df['status'].isin(['Depression', 'Anxiety'])]

# Extract TF-IDF features
X_text = model_data['cleaned_lemmatized_text']
X_sentiment = model_data[['vader_sentiment']]
y = model_data['status']

# Extract TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(X_text)

# Combine TF-IDF features with sentiment scores
X_combined = hstack([X_tfidf, X_sentiment])

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict the results
y_pred = svm_model.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['Anxiety', 'Depression'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Anxiety', 'Depression'])

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Anxiety and Depression Classification")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Print
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
