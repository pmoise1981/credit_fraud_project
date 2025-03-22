import os
import kaggle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Download dataset
print("📥 Downloading dataset...")
kaggle.api.dataset_download_files("mlg-ulb/creditcardfraud", path="data/", unzip=True)

# Load dataset
print("📊 Loading dataset...")
df = pd.read_csv("data/creditcard.csv")

# Data preprocessing
print("⚙️ Preprocessing dataset...")
df.drop_duplicates(inplace=True)  # Remove duplicates
df.fillna(0, inplace=True)        # Fill missing values

# Normalize 'Amount' column
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

# Save preprocessed data
df.to_csv("data/creditcard_cleaned.csv", index=False)
print("✅ Preprocessing complete! Data saved as 'creditcard_cleaned.csv'.")

