import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# Function to perform differencing
def difference(series, lag=1):
    return series[lag:] - series[:-lag]

# Function to calculate AR(p)
def autoregressive_model(series, p):
    X = np.array([series[i:i + p] for i in range(len(series) - p)])
    y = series[p:]
    return X, y

# Function to calculate MA(q)
def moving_average_model(series, q):
    errors = series - np.mean(series)
    X = np.array([errors[i:i + q] for i in range(len(errors) - q)])
    y = errors[q:]
    return X, y

# Function to fit ARIMA model from scratch
def fit_arima(series, p, d, q):
    # Step 1: Differencing
    for _ in range(d):
        series = difference(series)
    
    # Step 2: Fit AR(p) model
    if p > 0:
        X_ar, y_ar = autoregressive_model(series, p)
        ar_params = np.linalg.lstsq(X_ar, y_ar, rcond=None)[0]
    else:
        ar_params = np.zeros(p)
    
    # Step 3: Fit MA(q) model
    if q > 0:
        X_ma, y_ma = moving_average_model(series, q)
        ma_params = np.linalg.lstsq(X_ma, y_ma, rcond=None)[0]
    else:
        ma_params = np.zeros(q)
    
    # Return model parameters
    return ar_params, ma_params, series

# Function to forecast using fitted ARIMA model
def forecast_arima(series, ar_params, ma_params, p, q, steps=1):
    forecast = []
    # Use the most recent values for forecasting
    for _ in range(steps):
        ar_part = np.dot(ar_params, series[-p:][::-1]) if p > 0 else 0
        ma_part = np.dot(ma_params, (series[-q:] - np.mean(series))[-q:][::-1]) if q > 0 else 0
        forecast_value = ar_part + ma_part + np.mean(series)  # Adding the mean of the series as constant term
        forecast.append(forecast_value)
        series = np.append(series, forecast_value)  # Update the series for the next step
    return np.array(forecast)

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
            # Fit ARIMA model from scratch
            ar_params, ma_params, differenced_series = fit_arima(train_data['salary_in_usd'].values, p=1, d=1, q=1)

            # Make predictions on the test set
            forecast = forecast_arima(differenced_series, ar_params, ma_params, p=1, q=1, steps=len(test_data))
            
            # Calculate RMSE for this fold
            rmse = np.sqrt(mean_squared_error(test_data['salary_in_usd'], forecast))
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
        # Fit ARIMA model from scratch
        ar_params, ma_params, differenced_series = fit_arima(job_data['salary_in_usd'].values, p=2, d=1, q=2)

        # Forecast future values
        forecast = forecast_arima(differenced_series, ar_params, ma_params, p=2, q=2, steps=5)  # Forecasting next 5 years

        # Save the model to disk
        model_filename = f'arima_models/arima_model_{job_title}.pkl'
        if not os.path.exists('arima_models'):
            os.makedirs('arima_models')
        joblib.dump((ar_params, ma_params), model_filename)

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
