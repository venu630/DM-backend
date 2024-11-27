import matplotlib
matplotlib.use('Agg') 

from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.decomposition import PCA
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from flask_cors import CORS
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Load models and encoder
knn_model = joblib.load('knn_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')
onehot_encoder = joblib.load('onehot_encoder.pkl')
experience_encoder = joblib.load('experience_encoder.pkl')

# Load the dataset for time series analysis
file_path = 'salaries.csv'
df = pd.read_csv(file_path)

# Encode the experience level to numerical values
df['experience_level_encoded'] = experience_encoder.fit_transform(df['experience_level'])

# Group the data by work_year and experience_level_encoded, calculating the mean salary_in_usd for each group
grouped_data_v2 = df.groupby(['work_year', 'experience_level_encoded'])['salary_in_usd'].mean().reset_index()

# Endpoint to predict experience level using KNN model
@app.route('/knn_predict', methods=['POST'])
def knn_predict():
    # Get user input and create a DataFrame
    user_data = request.get_json()
    user_input = pd.DataFrame([user_data])

    # Separate categorical and numerical features for preprocessing
    categorical_features = ['job_title', 'employment_type', 'employee_residence', 'company_size']
    numerical_features = ['salary_in_usd', 'remote_ratio']

    # Apply one-hot encoding and scaling to match training data preprocessing
    user_categorical = onehot_encoder.transform(user_input[categorical_features])
    user_numerical = scaler.transform(user_input[numerical_features])

    # Combine transformed features
    user_transformed = np.hstack((user_numerical, user_categorical.toarray()))

    # Predict using the loaded model
    user_prediction = knn_model.predict(user_transformed)
    predicted_level = label_encoder.inverse_transform(user_prediction)[0]
    
    return jsonify({
        'predicted_experience_level': predicted_level,
    })

# Endpoint to predict salary using time series ARIMA model
@app.route('/timeseries_predict', methods=['POST'])
def timeseries_predict():
    # Get user input from request
    user_data = request.get_json()
    start_year = user_data['start_year']
    end_year = user_data['end_year']
    experience_level = user_data['experience_level']
    
    # Encode experience level
    exp_level_encoded = experience_encoder.transform([experience_level])[0]
    
    # Filter data for the specified experience level
    experience_data = grouped_data_v2[grouped_data_v2['experience_level_encoded'] == exp_level_encoded]
    experience_data.set_index('work_year', inplace=True)
    experience_data = experience_data.sort_index()

    # Fit ARIMA model (p, d, q) parameters can be tuned for better performance
    model = ARIMA(experience_data['salary_in_usd'], order=(1, 1, 1))
    model_fit = model.fit()

    # Forecast for the specified range of years
    forecast_years = [int(year) for year in range(int(experience_data.index[-1]) + 1, end_year + 1)]
    steps = len(forecast_years)
    forecast = model_fit.forecast(steps=steps)

    # Combine historical data with forecast data for a continuous plot
    full_years = list(map(int, experience_data.index)) + forecast_years
    full_salaries = list(map(float, experience_data['salary_in_usd'])) + list(map(float, forecast))

    # Prepare the result with both historical and predicted data
    forecast_results = {
        'years': full_years,
        'salaries': full_salaries,
        'forecast_start_year': int(experience_data.index[-1]) + 1
    }
    
    return jsonify(forecast_results)

if __name__ == '__main__':
    app.run(debug=True)
