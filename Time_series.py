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

# Encode the experience level to numerical values
experience_encoder = LabelEncoder()
df['experience_level_encoded'] = experience_encoder.fit_transform(df['experience_level'])

# Group the data by work_year and experience_level_encoded, calculating the mean salary_in_usd for each group
grouped_data_v2 = df.groupby(['work_year', 'experience_level_encoded'])['salary_in_usd'].mean().reset_index()

# Train and save ARIMA models for each experience level
unique_experience_levels = grouped_data_v2['experience_level_encoded'].unique()
for experience_level in unique_experience_levels:
    # Filter data for the specific experience level
    experience_data = grouped_data_v2[grouped_data_v2['experience_level_encoded'] == experience_level]
    experience_data.set_index('work_year', inplace=True)
    experience_data = experience_data.sort_index()

    # Fit the ARIMA model
    model = ARIMA(experience_data['salary_in_usd'], order=(1, 1, 1))  # Order can be tuned
    model_fit = model.fit()

    # Save the model to disk
    model_filename = f'arima_models/arima_model_{experience_level}.pkl'
    if not os.path.exists('arima_models'):
        os.makedirs('arima_models')
    joblib.dump(model_fit, model_filename)

# Save the encoder for deployment
joblib.dump(experience_encoder, 'experience_encoder.pkl')

# Plotting the trends for better visualization
plt.figure(figsize=(12, 6))
for experience_level in unique_experience_levels:
    exp_label = experience_encoder.inverse_transform([experience_level])[0]
    experience_data = grouped_data_v2[grouped_data_v2['experience_level_encoded'] == experience_level]
    plt.plot(experience_data['work_year'], experience_data['salary_in_usd'], marker='o', label=exp_label)

plt.xlabel('Year')
plt.ylabel('Average Salary (USD)')
plt.title('Average Salary Trends by Experience Level Over Time')
plt.legend(title="Experience Level")
plt.grid(True)
plt.savefig('salary_trends.png')