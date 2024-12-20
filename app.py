from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib
from sklearn.calibration import LabelEncoder
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

######################## Loading Models and Encoders ##############################

# Loading KNN models and encoders
knn_model = joblib.load('knn_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler_knn = joblib.load('scaler.pkl')  # Use a specific scaler for KNN
onehot_encoder = joblib.load('onehot_encoder.pkl')
job_title_encoder = joblib.load('job_title_encoder.pkl')
# Loading the dataset for time series analysis
file_path = 'salaries.csv'
df = pd.read_csv(file_path)
# Loading the clustering model-related objects
centroids, clusterAssignment_train = joblib.load('kmeans_custom_model.pkl')
scaler_clustering = joblib.load('scaler1.pkl')
labelEncoderJob = joblib.load('label_encoder_job.pkl')
labelEncoderLocation = joblib.load('label_encoder_location.pkl')
labelEncoderExperience = joblib.load('label_encoder_experience.pkl')
X_train_df = joblib.load('X_train_df.pkl')

############################# KNN Prediction Route ######################################

@app.route('/knn_predict', methods=['POST'])
def knn_predict():
    # Getting user input and creating a dataframe object
    user_data = request.get_json()
    user_input = pd.DataFrame([user_data])
    # Separating categorical and numerical features for preprocessing
    categorical_features = ['job_title', 'employment_type', 'employee_residence', 'company_size']
    numerical_features = ['salary_in_usd', 'remote_ratio']
    # Applying one-hot encoding and scaling to match training data preprocessing
    user_categorical = onehot_encoder.transform(user_input[categorical_features])
    user_numerical = scaler_knn.transform(user_input[numerical_features])
    # Combining transformed features
    user_transformed = np.hstack((user_numerical, user_categorical.toarray()))
    # Predicting the result using the loaded model (built from scratch)
    user_prediction = knn_model.predict(user_transformed)
    predicted_level = label_encoder.inverse_transform(user_prediction)[0]
    #returning the output in json format to the front end
    return jsonify({
        'predicted_experience_level': predicted_level,
    })

########################## Time Series Prediction Route ######################################

# Encoding the job_title to numerical values
job_title_encoder = LabelEncoder()
df['job_title_encoded'] = job_title_encoder.fit_transform(df['job_title'])
# Grouping the data by work_year and job_title_encoded
# Calculating the mean salary_in_usd for each group
grouped_data_v2 = df.groupby(['work_year', 'job_title_encoded'])['salary_in_usd'].mean().reset_index()
@app.route('/timeseries_predict', methods=['POST'])
def timeseries_predict():
    try:
        # Getting user input from request
        user_data = request.get_json()
        start_year = int(user_data['start_year'])
        end_year = int(user_data['end_year'])
        job_title = user_data['job_title']
        # Encoding job title
        job_title_encoded = job_title_encoder.transform([job_title])[0]
        # Checking if the job_title_encoded exists in the grouped_data_v2 DataFrame
        if 'job_title_encoded' not in grouped_data_v2.columns:
            return jsonify({'error': "'job_title_encoded' column not found in the dataset."}), 400
        # Loading the pre-trained ARIMA model
        model_filename = f'arima_models/arima_model_{job_title_encoded}.pkl'
        if not os.path.exists(model_filename):
            return jsonify({'error': 'Model for the specified job title not found.'}), 400
        model_fit = joblib.load(model_filename)
        # Filtering historical data based on start_year and job_title_encoded
        job_title_data = grouped_data_v2[
            (grouped_data_v2['job_title_encoded'] == job_title_encoded) &
            (grouped_data_v2['work_year'] >= start_year)
        ]
        if job_title_data.empty:
            return jsonify({'error': 'No historical data available after start year for the specified job title.'}), 400
        job_title_data.set_index('work_year', inplace=True)
        job_title_data = job_title_data.sort_index()
        job_title_data.index = pd.to_datetime(job_title_data.index, format='%Y')
        # Forecasting for the specified range of years
        last_year = job_title_data.index[-1]
        # Calculating steps based on the difference between end_year and the last historical year
        steps = end_year - last_year.year
        if steps > 0:
            # Generating forecast years
            forecast_years = pd.date_range(start=last_year + pd.DateOffset(years=1), periods=steps, freq='Y').year.tolist()
            forecast = model_fit.forecast(steps=steps)
            forecast = forecast.tolist()
        else:
            forecast = []
            forecast_years = []
        # Combining historical data with forecast data
        full_years = job_title_data.index.year.tolist() + forecast_years
        full_salaries = job_title_data['salary_in_usd'].tolist() + list(map(float, forecast))
        # Preparing the result
        forecast_results = {
            'years': full_years,
            'average_salaries': full_salaries,
            'forecast_start_year': forecast_years[0] if forecast_years else None
        }
        # retrning our result in json format
        return jsonify(forecast_results)
    # throwing an exception in case of any issue with the input
    except Exception as e:
        print('Exception in /timeseries_predict:', e)
        return jsonify({'error': str(e)}), 400
    
###################### Clustering Prediction Route #########################################

@app.route('/clustering_predict', methods=['POST'])
def clustering_predict():
    try:
        # Geting user input from the request
        user_data = request.get_json()
        user_salary = float(user_data['salary_in_usd'])
        user_job_title = user_data['job_title']
        user_location = user_data['company_location']
        user_experience_level = user_data['experience_level']
        # Preprocessing the user input data
        job_title_encoded = labelEncoderJob.transform([user_job_title])[0]
        company_location_encoded = labelEncoderLocation.transform([user_location])[0]
        experience_level_encoded = labelEncoderExperience.transform([user_experience_level])[0]
        # Standardizing the user data point using the pre-trained scaler for clustering
        user_data_point = pd.DataFrame([[user_salary, job_title_encoded, company_location_encoded, experience_level_encoded]], columns=['salary_in_usd', 'job_title_encoded', 'company_location_encoded', 'experience_level_encoded'])
        user_data_scaled = scaler_clustering.transform(user_data_point)
        # Predicting the user's cluster based on the pre-trained centroids
        distances = np.linalg.norm(user_data_scaled - centroids, axis=1)
        user_cluster = np.argmin(distances)
        # Plotting the training data clusters and highlight the user's data point
        # plt.figure(figsize=(8, 6))
        scatter = sns.scatterplot(x=X_train_df['PCA1'], y=X_train_df['PCA2'], hue=X_train_df['cluster'], palette='Set2', s=100, legend="full")
        plt.scatter(user_data_scaled[0][0], user_data_scaled[0][1], color='red', s=200, marker='X', label="User Input")
        plt.title(f'K-Means Clustering (k=3) with User Input Highlighted')
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.legend(title='Cluster')
        plt.tight_layout()
        # Converting the plot to base64 to send it to the frontend
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()
        # Preparing the summary for the user's cluster
        cluster_data = X_train_df[X_train_df['cluster'] == user_cluster]
        avg_salary = cluster_data['salary_in_usd'].mean()
        avg_job_title = cluster_data['job_title'].mode()[0]
        avg_location = cluster_data['company_location'].mode()[0]
        avg_experience = cluster_data['experience_level'].mode()[0]
        # Ensuring that all values are of a type that can be serialized by JSON
        cluster_summary = {
            'cluster': int(user_cluster),
            'avg_salary': float(avg_salary),
            'avg_job_title': str(avg_job_title),
            'avg_location': str(avg_location),
            'avg_experience': str(avg_experience)
        }
        # Returning the plot image and cluster summary
        response = {
            'cluster_summary': cluster_summary,
            'plot_image': img_base64
        }
        return jsonify(response)

    except Exception as e:
        print('Exception in /clustering_predict:', e)
        return jsonify({'error': str(e)}), 400


# Loading the outlier detection model
outlier_model_path = 'outlier_model.pkl'
outlier_models = joblib.load(outlier_model_path)
# function used for plotting outlier box plots
def generate_outlier_plot(df, attribute, outliers, z_scores):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Creating a vertical box plot for the attribute
    ax.boxplot(df[attribute], vert=True, patch_artist=True, boxprops=dict(facecolor='skyblue', color='blue'), flierprops=dict(marker='o', color='red', markersize=8))
    # Adding labels and title
    ax.set_ylabel(attribute)
    ax.set_title(f'Outlier Detection for {attribute} (Box Plot)')
    # Converting plot to base64 string
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    img_base64 = base64.b64encode(img_stream.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

###################### Outlier detection rote #########################################
@app.route('/outliers', methods=['GET'])
def get_outliers():
    attribute = request.args.get('attribute')
    if attribute not in outlier_models:
        return jsonify({'error': f"Attribute '{attribute}' not found in the model."}), 400
    # Fetching the dataset and outlier data from the model
    df = pd.read_csv('salaries.csv')
    outliers, z_scores = outlier_models[attribute]["outliers"], outlier_models[attribute]["z_scores"]
    # Getting the outlier job titles and their respective salaries
    outlier_job_titles = df['job_title'].iloc[outliers[0]].tolist()
    outlier_salaries = df['salary_in_usd'].iloc[outliers[0]].tolist()
    # Generating the outlier plot and return as base64
    img_base64 = generate_outlier_plot(df, attribute, outliers[0], z_scores)
    return jsonify({
        'attribute': attribute,
        'outliers': outliers[0].tolist(),
        'outlier_job_titles': outlier_job_titles,
        'outlier_salaries': outlier_salaries,
        'image': img_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
