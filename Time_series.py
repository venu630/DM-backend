import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# -------------------- Load Models and Encoders --------------------

# Load KNN models and encoders
knn_model = joblib.load('knn_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')
onehot_encoder = joblib.load('onehot_encoder.pkl')
experience_encoder = joblib.load('experience_encoder.pkl')

# Load the dataset for time series analysis
file_path = 'salaries.csv'
df = pd.read_csv(file_path)

# Encode the experience level to numerical values
df['experience_level_encoded'] = experience_encoder.transform(df['experience_level'])

# Group the data by work_year and experience_level_encoded, calculating the mean salary_in_usd for each group
grouped_data_v2 = df.groupby(['work_year', 'experience_level_encoded'])['salary_in_usd'].mean().reset_index()

# -------------------- KNN Prediction Endpoint --------------------

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

# -------------------- Time Series Prediction Endpoint --------------------

# Endpoint to predict salary using pre-trained ARIMA models
@app.route('/timeseries_predict', methods=['POST'])
def timeseries_predict():
    try:
        # Get user input from request
        user_data = request.get_json()
        start_year = int(user_data['start_year'])
        end_year = int(user_data['end_year'])
        experience_level = user_data['experience_level']

        # Encode experience level
        exp_level_encoded = experience_encoder.transform([experience_level])[0]

        # Load the pre-trained ARIMA model for the specified experience level
        model_filename = f'arima_models/arima_model_{exp_level_encoded}.pkl'
        if not os.path.exists(model_filename):
            return jsonify({'error': 'Model for the specified experience level not found.'}), 400

        model_fit = joblib.load(model_filename)

        # Filter historical data based on start_year
        experience_data = grouped_data_v2[
            (grouped_data_v2['experience_level_encoded'] == exp_level_encoded) &
            (grouped_data_v2['work_year'] >= start_year)
        ]
        if experience_data.empty:
            return jsonify({'error': 'No historical data available after start year for the specified experience level.'}), 400

        experience_data.set_index('work_year', inplace=True)
        experience_data = experience_data.sort_index()
        experience_data.index = pd.to_datetime(experience_data.index, format='%Y')

        # Forecast for the specified range of years
        last_year = experience_data.index[-1]
        # Calculate steps based on the difference between end_year and the last historical year
        steps = end_year - last_year.year
        if steps > 0:
            forecast = model_fit.forecast(steps=steps)
            forecast = forecast.tolist()
            # Generate forecast years
            forecast_years = pd.date_range(start=last_year + pd.DateOffset(years=1), periods=steps, freq='Y').year.tolist()
        else:
            forecast = []
            forecast_years = []

        # Combine historical data with forecast data
        full_years = experience_data.index.year.tolist() + forecast_years
        full_salaries = experience_data['salary_in_usd'].tolist() + list(map(float, forecast))

        # Prepare the result
        forecast_results = {
            'years': full_years,
            'salaries': full_salaries,
            'forecast_start_year': forecast_years[0] if forecast_years else None
        }

        return jsonify(forecast_results)

    except Exception as e:
        print('Exception in /timeseries_predict:', e)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
