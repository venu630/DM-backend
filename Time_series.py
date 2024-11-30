import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.arima.model import ARIMA
import os
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'salaries.csv'
df = pd.read_csv(file_path)

# Encode the job_title to numerical values
job_title_encoder = LabelEncoder()
df['job_title_encoded'] = job_title_encoder.fit_transform(df['job_title'])

# Group the data by work_year and job_title_encoded, calculating the mean salary_in_usd for each group
grouped_data_v2 = df.groupby(['work_year', 'job_title_encoded'])['salary_in_usd'].mean().reset_index()

# Convert 'work_year' to datetime for compatibility with ARIMA
grouped_data_v2['work_year'] = pd.to_datetime(grouped_data_v2['work_year'], format='%Y')

# Train and save ARIMA models for each job title
unique_job_titles = grouped_data_v2['job_title_encoded'].unique()
for job_title in unique_job_titles:
    # Filter data for the specific job title
    job_data = grouped_data_v2[grouped_data_v2['job_title_encoded'] == job_title]
    job_data.set_index('work_year', inplace=True)
    job_data = job_data.sort_index()

    # Debug: Print the data being used for this job title
    print(f"Processing job title: {job_title_encoder.inverse_transform([job_title])[0]}")
    print(job_data)

    # Skip if the data is insufficient for ARIMA modeling
    if len(job_data) < 3:
        continue

    # Fit the ARIMA model
    try:
        model = ARIMA(job_data['salary_in_usd'], order=(1, 1, 1))  # Order can be tuned
        model_fit = model.fit()
    except Exception as e:
        print(f"Error fitting ARIMA for job title {job_title}: {e}")
        continue

    # Save the model to disk
    model_filename = f'arima_models/arima_model_{job_title}.pkl'
    if not os.path.exists('arima_models'):
        os.makedirs('arima_models')
    joblib.dump(model_fit, model_filename)

    print(f"Model saved for job title: {job_title_encoder.inverse_transform([job_title])[0]}")

# Save the encoder for deployment
joblib.dump(job_title_encoder, 'job_title_encoder.pkl')