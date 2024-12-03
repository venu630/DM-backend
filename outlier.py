import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import joblib
import os

# Function to detect outliers using Z-score
def detect_outliers(df, attribute, threshold=3.0):
    # Calculate Z-scores for the attribute
    z_scores = stats.zscore(df[attribute])
    
    # Find outliers based on Z-score threshold
    outliers = np.where(np.abs(z_scores) > threshold)
    return outliers, z_scores

# Function to save outlier model
def save_outlier_model(df, filename="outlier_model.pkl"):
    outlier_models = {}
    for column in df.columns:
        if np.issubdtype(df[column].dtype, np.number):  # Only consider numerical columns
            outliers, z_scores = detect_outliers(df, column)
            outlier_models[column] = {
                "outliers": outliers,
                "z_scores": z_scores
            }
    
    # Save the model (dictionary) to a file
    with open(filename, "wb") as f:
        joblib.dump(outlier_models, f)
    print(f"Outlier detection model saved as {filename}")

# Load the dataset (adjust this path accordingly)
file_path = 'salaries.csv'  # Update with actual file path
df = pd.read_csv(file_path)

# Detect and save the outliers for the dataset
save_outlier_model(df)

