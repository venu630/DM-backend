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

app = Flask(__name__)
CORS(app)

# Load models and encoder
knn_model = joblib.load('knn_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')
onehot_encoder = joblib.load('onehot_encoder.pkl')
model = joblib.load('gbr_model.pkl')
experience_encoder = joblib.load('experience_encoder.pkl')

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

# Endpoint to predict salary using time series Gradient Boosting model
@app.route('/timeseries_predict', methods=['POST'])
def timeseries_predict():
    # Get user input from request
    user_data = request.get_json()
    work_year = user_data['work_year']
    experience_level = user_data['experience_level']
    
    # Encode experience level
    exp_level_encoded = experience_encoder.transform([experience_level])[0]
    
    # Prepare input data for prediction
    X_input = np.array([[work_year, exp_level_encoded]])
    
    # Make prediction
    predicted_salary = model.predict(X_input)[0]
    
    return jsonify({
        'predicted_salary': round(predicted_salary, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)