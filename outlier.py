import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class OutlierDetection:
    def __init__(self, train_data, test_data):
        self.data_sets = {"Train": train_data, "Test": test_data}
        self.results = {}

    def calculate_iqr(self, column):
        for set_name, data in self.data_sets.items():
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            self.results[set_name] = {
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "outliers": outliers
            }

    def plot_outliers(self, column):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set colors for training and testing data
        colors = {"Train": "orange", "Test": "lightblue"}
        outlier_colors = {"Train": "red", "Test": "blue"}

        for idx, (set_name, result) in enumerate(self.results.items(), start=1):
            # Box plot for each dataset
            ax.boxplot(self.data_sets[set_name][column], positions=[idx], widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor=colors[set_name]), medianprops=dict(color=outlier_colors[set_name]))
            
            # Outliers as solid dots
            ax.scatter(np.ones(len(result["outliers"])) * idx, result["outliers"][column], 
                       color=outlier_colors[set_name], label=f"{set_name} Outliers", s=10)

        # Labels and title
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Train Data", "Test Data"])
        ax.set_ylabel("Salaries in USD")
        ax.set_title("Outlier Detection for Train and Test Sets")
        ax.legend()

        # Display plot instead of saving it
        plt.show()

    def save_model(self, filename="model/outlier_model.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.results, f)

# Load dataset
data = pd.read_csv("salaries.csv")

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Initialize outlier detection for 'salary_in_usd' column
outlier_detector = OutlierDetection(train_data, test_data)
outlier_detector.calculate_iqr("salary_in_usd")
outlier_detector.plot_outliers("salary_in_usd")
outlier_detector.save_model()
