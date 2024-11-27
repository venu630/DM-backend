import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load dataset
df = pd.read_csv('salaries.csv')

# Manually encode 'experience_level' as numerical values
experience_levels = df['experience_level'].unique()
experience_encoder = {level: idx for idx, level in enumerate(experience_levels)}
experience_decoder = {idx: level for level, idx in experience_encoder.items()}
df['experience_level_encoded'] = df['experience_level'].map(experience_encoder)

# Group data by 'work_year' and 'experience_level_encoded' and calculate average salary
grouped_data = df.groupby(['work_year', 'experience_level_encoded'])['salary_in_usd'].mean().reset_index()

# Get unique experience levels
unique_experience_levels = grouped_data['experience_level_encoded'].unique()

# Create plot settings
plt.figure(figsize=(12, 8))

# Iterate over each experience level to forecast salaries
for level in unique_experience_levels:
    # Filter data for each experience level and sort by year
    level_data = grouped_data[grouped_data['experience_level_encoded'] == level].set_index('work_year')
    level_data = level_data.sort_index()

    # Check if there is enough data to perform a moving average
    if len(level_data) < 3:
        print(f"Not enough data to forecast for experience level {experience_decoder[level]}")
        continue

    # Moving average to smooth out yearly fluctuations
    level_data['moving_avg'] = level_data['salary_in_usd'].rolling(window=3, min_periods=1).mean()

    # Manual Forecast: Extrapolate based on the trend in the last few years
    last_two_years = level_data.tail(2)
    if len(last_two_years) == 2:
        # Calculate average yearly change based on the last two years' data
        year_diff = last_two_years.index[-1] - last_two_years.index[-2]
        salary_diff = last_two_years['moving_avg'].iloc[-1] - last_two_years['moving_avg'].iloc[-2]
        yearly_change = salary_diff / year_diff
    else:
        yearly_change = 0  # No trend available if only one year of data exists

    # Forecast for the next 6 years
    last_year = level_data.index[-1]
    forecast_years = [last_year + i for i in range(1, 7)]
    forecast_salaries = [level_data['moving_avg'].iloc[-1] + (yearly_change * i) for i in range(1, 7)]

    # Print forecasted values for each year
    print(f"Experience Level: {experience_decoder[level]}")
    for year, value in zip(forecast_years, forecast_salaries):
        print(f"Year {year}: Predicted Average Salary = ${value:.2f}")

    # Plot historical and forecasted salaries
    plt.plot(level_data.index, level_data['salary_in_usd'], marker='o', label=f'Historical - {experience_decoder[level]}')
    plt.plot(level_data.index, level_data['moving_avg'], linestyle='-', color='gray', alpha=0.5, label=f'Smoothed (Moving Avg) - {experience_decoder[level]}')
    plt.plot(forecast_years, forecast_salaries, marker='x', linestyle='--', label=f'Forecast - {experience_decoder[level]}')

# Finalize and save the plot
plt.xlabel('Year')
plt.ylabel('Average Salary (USD)')
plt.title('Average Salary Trends by Experience Level Over Time')
plt.legend(title="Experience Level")
plt.grid(True)
plt.savefig('manual_salary_forecast.png')
plt.show()

# Save label encoder for future use
joblib.dump(experience_encoder, 'experience_encoder_manual.pkl')

# Print min and max year for frontend integration
min_year = grouped_data['work_year'].min()
max_year = grouped_data['work_year'].max() + 6
print(f"Min Year: {min_year}, Max Year: {max_year}")
