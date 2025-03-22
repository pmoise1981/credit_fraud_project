# eda_and_split.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
data = pd.read_csv("data/credit_card_cleaned.csv")

# General Overview
print("First few rows of data:\n", data.head())
print("\nSummary statistics:\n", data.describe())
print("\nMissing values:\n", data.isnull().sum())
print("\nData types:\n", data.info())

# Visualize class distribution
sns.countplot(x='Class', data=data)
plt.title('Class Distribution')
plt.show()

# Correlation heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()

# Train-test split
X = data.drop(columns=['Class'])
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTrain-test split successful. Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

