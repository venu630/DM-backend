import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import joblib
import numpy as np

# Load data
df = pd.read_csv('salaries.csv')

# Prepare features and labels
X = df[['job_title', 'employment_type', 'salary_in_usd', 'employee_residence', 'remote_ratio', 'company_size']]
y = df['experience_level']

# Encode target labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Preprocess categorical and numerical features separately
categorical_features = ['job_title', 'employment_type', 'employee_residence', 'company_size']
numerical_features = ['salary_in_usd', 'remote_ratio']

# Apply OneHotEncoding to categorical features
onehot_encoder = OneHotEncoder(handle_unknown='ignore')
X_categorical = onehot_encoder.fit_transform(X[categorical_features])

# Apply StandardScaler to numerical features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(X[numerical_features])

# Combine transformed numerical and categorical features
X_transformed = np.hstack((X_numerical, X_categorical.toarray()))

# Set up the KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=7)

# Perform k-fold cross-validation using accuracy as the default scoring metric
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(knn_model, X_transformed, y_encoded, cv=kf)
print("Cross-Validation Scores:", cv_scores)

# Calculate and print the average cross-validation score
average_cv_score = np.mean(cv_scores)
print("Average Cross-Validation Score:", average_cv_score)

# Train on full data and save the model and label encoder
knn_model.fit(X_transformed, y_encoded)
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(onehot_encoder, 'onehot_encoder.pkl')
