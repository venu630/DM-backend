import pandas as pd
import numpy as np
import pickle

# Function to detect outliers using IQR (Interquartile Range)
def detect_outliers_iqr(df, attribute):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(df[attribute], 25)
    Q3 = np.percentile(df[attribute], 75)
    
    # Calculate the IQR (Interquartile Range)
    IQR = Q3 - Q1
    
    # Calculate lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Find outliers based on the bounds
    outliers = df[(df[attribute] < lower_bound) | (df[attribute] > upper_bound)].index
    
    return outliers, (lower_bound, upper_bound)

# Function to save the outlier detection model
def save_outlier_model(df, filename="outlier_model.pkl"):
    outlier_models = {}
    
    for column in df.columns:
        if np.issubdtype(df[column].dtype, np.number):  # Only consider numerical columns
            outliers, bounds = detect_outliers_iqr(df, column)
            outlier_models[column] = {
                "outliers": outliers.tolist(),
                "bounds": bounds
            }
    
    # Save the model (dictionary) to a file
    with open(filename, "wb") as f:
        pickle.dump(outlier_models, f)
    print(f"Outlier detection model saved as {filename}")

# Load the dataset (adjust this path accordingly)
file_path = 'salaries.csv'  # Update with actual file path
df = pd.read_csv(file_path)

# Detect and save the outliers for the dataset
save_outlier_model(df)
