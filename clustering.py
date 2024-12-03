import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

# Loading the dataset
df = pd.read_csv('salaries.csv')

# Step 1: Preprocessing the data
df['salary_in_usd'] = df['salary_in_usd'].fillna(df['salary_in_usd'].median())
df['job_title'] = df['job_title'].fillna(df['job_title'].mode()[0])
df['company_location'] = df['company_location'].fillna(df['company_location'].mode()[0])
df['experience_level'] = df['experience_level'].fillna(df['experience_level'].mode()[0])

# Step 2: Label Encoding for categorical features
labelEncoderJob = LabelEncoder()
df['job_title_encoded'] = labelEncoderJob.fit_transform(df['job_title'])

labelEncoderLocation = LabelEncoder()
df['company_location_encoded'] = labelEncoderLocation.fit_transform(df['company_location'])

labelEncoderExperience = LabelEncoder()
df['experience_level_encoded'] = labelEncoderExperience.fit_transform(df['experience_level'])

# Step 3: Selecting features for clustering
selectedFeatures = ['salary_in_usd', 'job_title_encoded', 'company_location_encoded', 'experience_level_encoded']
dfSelected = df[selectedFeatures]

# Step 4: Standardizing the features
scaler = StandardScaler()
dfScaled = scaler.fit_transform(dfSelected)

# Step 5: Train-test split (80% training and 20% test data)
X_train, X_test = train_test_split(dfScaled, test_size=0.2, random_state=42)

# Step 6: K-Means from scratch implementation
def kMeans(data, k, maxIters=100):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    for i in range(maxIters):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        clusterAssignment = np.argmin(distances, axis=1)
        
        newCentroids = np.array([data[clusterAssignment == i].mean(axis=0) for i in range(k)])

        if np.all(centroids == newCentroids):
            break

        centroids = newCentroids

    return clusterAssignment, centroids

# Step 7: Elbow Method to determine the optimal k
wss = []
kRange = range(1, 11)
for k in kRange:
    clusterAssignment, centroids = kMeans(X_train, k)
    wss.append(np.sum(np.min(np.linalg.norm(X_train[:, np.newaxis] - centroids, axis=2), axis=1)**2))

# Plot the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(kRange, wss, marker='o', label='WSS')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WSS)')
plt.axvline(x=3, color='r', linestyle='--', label='Optimal k = 3')  # Line showing optimal k value
plt.legend()
plt.show()

# Step 8: Choosing the optimal k based on the Elbow Method
optimalK = 3

# Step 9: Running K-Means with the optimal k on training data
clusterAssignment_train, centroids = kMeans(X_train, optimalK)

# Step 10: Applying PCA for better 2D visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Step 11: Adding PCA components to the training data
X_train_df = pd.DataFrame(X_train_pca, columns=['PCA1', 'PCA2'])
X_train_df['cluster'] = clusterAssignment_train

# Merge the original features into X_train_df
X_train_df['salary_in_usd'] = df['salary_in_usd'].iloc[X_train_df.index]
X_train_df['job_title'] = df['job_title'].iloc[X_train_df.index]
X_train_df['company_location'] = df['company_location'].iloc[X_train_df.index]
X_train_df['experience_level'] = df['experience_level'].iloc[X_train_df.index]

# Step 12: Visualizing the clustering results for the training data using PCA space
plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(x=X_train_df['PCA1'], y=X_train_df['PCA2'], hue=X_train_df['cluster'], palette='Set2', s=100, legend="full")
plt.title(f'K-Means Clustering (k={optimalK}) on Training Data with PCA')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.show()

# Step 13: Silhouette Score for Training Data
silhouetteAvg_train = silhouette_score(X_train, clusterAssignment_train)
print(f"Silhouette Score for Training Data: {silhouetteAvg_train:.4f}")

# Step 14: Running K-Means on Test Data using the same centroids
distances_test = np.linalg.norm(X_test[:, np.newaxis] - centroids, axis=2)
clusterAssignment_test = np.argmin(distances_test, axis=1)

# Step 15: PCA transformation for Test Data
X_test_pca = pca.transform(X_test)

# Step 16: Adding PCA components to the test data
X_test_df = pd.DataFrame(X_test_pca, columns=['PCA1', 'PCA2'])
X_test_df['cluster'] = clusterAssignment_test

# Merge the original features into X_test_df
X_test_df['salary_in_usd'] = df['salary_in_usd'].iloc[X_test_df.index]
X_test_df['job_title'] = df['job_title'].iloc[X_test_df.index]
X_test_df['company_location'] = df['company_location'].iloc[X_test_df.index]
X_test_df['experience_level'] = df['experience_level'].iloc[X_test_df.index]

# Step 17: Visualizing the clustering results for the test data using PCA space
plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(x=X_test_df['PCA1'], y=X_test_df['PCA2'], hue=X_test_df['cluster'], palette='Set2', s=100, legend="full")
plt.title(f'K-Means Clustering (k={optimalK}) on Test Data with PCA')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
# plt.show()

# Step 18: Silhouette Score for Test Data
silhouetteAvg_test = silhouette_score(X_test, clusterAssignment_test)
print(f"Silhouette Score for Test Data: {silhouetteAvg_test:.4f}")

# Step 19: Summarizing each cluster (training data)
for i in range(optimalK):
    clusterData_train = X_train_df[X_train_df['cluster'] == i]
    avgSalary = clusterData_train['salary_in_usd'].mean()
    avgJobTitle = clusterData_train['job_title'].mode()[0]
    avgLocation = clusterData_train['company_location'].mode()[0]
    avgExperience = clusterData_train['experience_level'].mode()[0]
    print(f"\nCluster {i} Summary for Training Data:")
    print(f"Number of data points in cluster: {len(clusterData_train)}")
    print(f"Average Salary: ${avgSalary:.2f}")
    print(f"Typical Job Title: {avgJobTitle}")
    print(f"Typical Location: {avgLocation}")
    print(f"Average Experience Level: {avgExperience}")

# Step 20: Summarizing each cluster (test data)
for i in range(optimalK):
    clusterData_test = X_test_df[X_test_df['cluster'] == i]
    avgSalary = clusterData_test['salary_in_usd'].mean()
    avgJobTitle = clusterData_test['job_title'].mode()[0]
    avgLocation = clusterData_test['company_location'].mode()[0]
    avgExperience = clusterData_test['experience_level'].mode()[0]
    print(f"\nCluster {i} Summary for Test Data:")
    print(f"Number of data points in cluster: {len(clusterData_test)}")
    print(f"Average Salary: ${avgSalary:.2f}")
    print(f"Typical Job Title: {avgJobTitle}")
    print(f"Typical Location: {avgLocation}")
    print(f"Average Experience Level: {avgExperience}")

# Saving the KMeans model (centroids and cluster assignments)
joblib.dump((centroids, clusterAssignment_train), 'kmeans_custom_model.pkl')
# Saving the StandardScaler object
joblib.dump(scaler, 'scaler1.pkl')
# Saving the label encoders
joblib.dump(labelEncoderJob, 'label_encoder_job.pkl')
joblib.dump(labelEncoderLocation, 'label_encoder_location.pkl')
joblib.dump(labelEncoderExperience, 'label_encoder_experience.pkl')
joblib.dump(X_train_df, 'X_train_df.pkl')
