import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# We are loading the data from the salaries.csv dataset
df = pd.read_csv('salaries.csv')

# We are preparing the features
X = df[['job_title', 'employment_type', 'salary_in_usd', 'employee_residence', 'remote_ratio', 'company_size']]
y = df['experience_level']

# Using the label_encoder, we are encoding the target variables as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Preprocessing categorical features
categorical_features = ['job_title', 'employment_type', 'employee_residence', 'company_size']
# Preprocessing numerical features
numerical_features = ['salary_in_usd', 'remote_ratio']

# We are apply OneHot encoding to the categorical features
onehot_encoder = OneHotEncoder(handle_unknown='ignore')
X_categorical = onehot_encoder.fit_transform(X[categorical_features])

# Scaling the numerical features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(X[numerical_features])

# we use hstack to combine the categorical and numerical features
X_transformed = np.hstack((X_numerical, X_categorical.toarray()))

# function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Writing a scratch algoirhm to implement KNN
class CustomKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # Storing the training data
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # below list is used to store the predictions on each instance of X
        predictions = []
        for x in X:
            # Calculating distances from x to all training samples
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
            # Obtaining indices of k nearest neghbours
            neighbor_indices = np.argsort(distances)[:self.n_neighbors]
            neighbor_labels = [self.y_train[i] for i in neighbor_indices]
            # Using majority voting to decide on the most common label
            predicted_label = max(set(neighbor_labels), key=neighbor_labels.count)
            predictions.append(predicted_label)
        return np.array(predictions)

# Using k-fold cross validation for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_index, test_index in kf.split(X_transformed):
    X_train, X_test = X_transformed[train_index], X_transformed[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    # Create and train a KNN model with k=5 and evaluate its performance using accuracy score
    knn_model = CustomKNN(n_neighbors = 5)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores.append(accuracy)

# Printing accuracy scores and the avg accuracy score
print("Accuracy Scores from Cross-Validation:", cv_scores)
average_accuracy_score = np.mean(cv_scores)
print("Average Accuracy Score:", average_accuracy_score)

# Saving the model to integrate it to the front end
knn_model.fit(X_transformed, y_encoded)
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(onehot_encoder, 'onehot_encoder.pkl')