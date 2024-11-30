import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.arima.model import ARIMA
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

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

# K-Fold cross-validation settings
n_splits = 5  # Desired number of splits for TimeSeriesSplit

# List to store RMSE scores across all models
all_fold_rmse = []

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
        print(f"Skipping job title {job_title_encoder.inverse_transform([job_title])[0]} due to insufficient data.")
        continue

    # Adjust the number of splits based on the size of the data
    n_splits = min(n_splits, len(job_data) - 1)

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_rmse = []

    for train_index, test_index in tscv.split(job_data):
        train_data, test_data = job_data.iloc[train_index], job_data.iloc[test_index]
        
        try:
            # Fit the ARIMA model (simpler model if data is small)
            if len(train_data) < 5:
                model = ARIMA(train_data['salary_in_usd'], order=(0, 1, 0))  # Use a simpler model for small data
            else:
                model = ARIMA(train_data['salary_in_usd'], order=(1, 1, 1))  # Default ARIMA model

            model_fit = model.fit()

            # Make predictions on the test set
            predictions = model_fit.forecast(steps=len(test_data))

            # Calculate RMSE for this fold
            rmse = np.sqrt(mean_squared_error(test_data['salary_in_usd'], predictions))
            fold_rmse.append(rmse)

            print(f"Fold RMSE: {rmse}")

        except Exception as e:
            print(f"Error during ARIMA fitting for job title {job_title_encoder.inverse_transform([job_title])[0]}: {e}")
            continue

    # Calculate and print average RMSE across all folds for this job title
    if fold_rmse:
        avg_rmse = np.mean(fold_rmse)
        print(f"Average RMSE for {job_title_encoder.inverse_transform([job_title])[0]}: {avg_rmse}")
        all_fold_rmse.extend(fold_rmse)  # Add this job title's RMSE scores to the global list
    else:
        print(f"No valid RMSE values for {job_title_encoder.inverse_transform([job_title])[0]}")

    # Fit the final ARIMA model on the entire dataset for this job title
    try:
        # Use simpler ARIMA models if the dataset is too small
        if len(job_data) < 5:
            model = ARIMA(job_data['salary_in_usd'], order=(0, 1, 0))
        else:
            model = ARIMA(job_data['salary_in_usd'], order=(1, 1, 1))  # Default ARIMA model
        model_fit = model.fit()

        # Save the model to disk
        model_filename = f'arima_models/arima_model_{job_title}.pkl'
        if not os.path.exists('arima_models'):
            os.makedirs('arima_models')
        joblib.dump(model_fit, model_filename)

        print(f"Model saved for job title: {job_title_encoder.inverse_transform([job_title])[0]}")

    except Exception as e:
        print(f"Error fitting ARIMA for job title {job_title_encoder.inverse_transform([job_title])[0]}: {e}")
        continue

# Calculate the overall mean salary_in_usd (mean of actual salaries)
mean_salary = np.mean(grouped_data_v2['salary_in_usd'])

# Calculate the overall average RMSE as a percentage of the mean salary
if all_fold_rmse:
    overall_avg_rmse = np.mean(all_fold_rmse)
    rmse_percentage = (overall_avg_rmse / mean_salary) * 100
    print(f"Overall Average RMSE: {overall_avg_rmse}")
    print(f"Overall RMSE as Percentage of Mean Salary: {rmse_percentage:.2f}%")
else:
    print("No valid RMSE scores across all models.")

    
# Save the encoder for deployment
joblib.dump(job_title_encoder, 'job_title_encoder.pkl')
