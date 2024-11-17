import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load the dataset
file_path = 'salaries.csv'
df = pd.read_csv(file_path)

# Encode the experience level to numerical values
experience_encoder = LabelEncoder()
df['experience_level_encoded'] = experience_encoder.fit_transform(df['experience_level'])

# Group the data by work_year and experience_level_encoded, calculating the mean salary_in_usd for each group
grouped_data_v2 = df.groupby(['work_year', 'experience_level_encoded'])['salary_in_usd'].mean().reset_index()

# Prepare the dataset for training
X = grouped_data_v2[['work_year', 'experience_level_encoded']]
y = grouped_data_v2['salary_in_usd']

# Split the data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Gradient Boosting Regressor model
gbr_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
gbr_model.fit(X_train, y_train)

# Make predictions
y_pred = gbr_model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Display the updated metrics
print("Updated Metrics:")
print(f"MAE = {mae:.2f}, RMSE = {rmse:.2f}")

# Save the trained model for deployment
joblib.dump(gbr_model, 'gbr_model.pkl')
joblib.dump(experience_encoder, 'experience_encoder.pkl')

# Plotting the trends for better visualization
plt.figure(figsize=(12, 6))
unique_experience_levels = grouped_data_v2['experience_level_encoded'].unique()
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