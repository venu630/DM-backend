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

# Load model, label encoder, scaler, and one-hot encoder
model = joblib.load('knn_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')
onehot_encoder = joblib.load('onehot_encoder.pkl')

# Load the original dataset
data = pd.read_csv('salaries.csv')

@app.route('/knn_predict', methods=['POST'])
def predict():
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
    user_prediction = model.predict(user_transformed)
    predicted_level = label_encoder.inverse_transform(user_prediction)[0]
    
    return jsonify({
        'predicted_experience_level': predicted_level,
    })

if __name__ == '__main__':
    app.run(debug=True)
