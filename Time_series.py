import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
file_path = 'salaries.csv'
df = pd.read_csv(file_path)

# Encode the experience level to numerical values
experience_encoder = LabelEncoder()
df['experience_level_encoded'] = experience_encoder.fit_transform(df['experience_level'])

# Group the data by work_year and experience_level_encoded, calculating the mean salary_in_usd for each group
grouped_data_v2 = df.groupby(['work_year', 'experience_level_encoded'])['salary_in_usd'].mean().reset_index()

# Set up a time series dataset
# We'll focus on predicting salaries for each experience level over the years
experience_levels = grouped_data_v2['experience_level_encoded'].unique()

# Determine the min and max date ranges for user input
min_year = grouped_data_v2['work_year'].min()
max_year = grouped_data_v2['work_year'].max() + 6  # Including the forecast period

# Initialize a plot for all experience levels
plt.figure(figsize=(12, 8))

# Create a time series model for each experience level and forecast future salaries
for experience_level in experience_levels:
    # Filter data for the current experience level
    experience_data = grouped_data_v2[grouped_data_v2['experience_level_encoded'] == experience_level]
    experience_data.set_index('work_year', inplace=True)
    experience_data = experience_data.sort_index()

    # Fit ARIMA model (p, d, q) parameters can be tuned for better performance
    model = ARIMA(experience_data['salary_in_usd'], order=(1, 1, 1))
    model_fit = model.fit()

    # Forecasting the next 6 years
    forecast_years = [experience_data.index[-1] + i for i in range(1, 7)]
    forecast = model_fit.forecast(steps=6)

    # Display forecasted values
    print(f"Experience Level: {experience_encoder.inverse_transform([experience_level])[0]}")
    for year, value in zip(forecast_years, forecast):
        print(f"Year {year}: Predicted Average Salary = ${value:.2f}")

    # Plotting the original data and forecast in the combined graph
    plt.plot(experience_data.index, experience_data['salary_in_usd'], marker='o', label=f'Historical Salary - {experience_encoder.inverse_transform([experience_level])[0]}')
    plt.plot(forecast_years, forecast, marker='x', linestyle='--', label=f'Forecasted Salary - {experience_encoder.inverse_transform([experience_level])[0]}')

# Finalize the combined plot
plt.xlabel('Year')
plt.ylabel('Average Salary (USD)')
plt.title('Average Salary Trends by Experience Level Over Time')
plt.legend(title="Experience Level")
plt.grid(True)
plt.savefig('combined_salary_trends_forecast.png')
plt.show()

# Save the label encoder for deployment
joblib.dump(experience_encoder, 'experience_encoder.pkl')

# Print min and max year for frontend integration
print(f"Min Year: {min_year}, Max Year: {max_year}")