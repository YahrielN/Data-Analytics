import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import urllib.error

# Load Dataset
# Using a public dataset available on the internet - Telco Customer Churn Dataset from IBM
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

# Function to load data from a URL or local file
def load_data(source):
    try:
        if source.startswith('http'):
            return pd.read_csv(source)
        else:
            return pd.read_csv(source)
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e}. Please provide a local file path instead.")
        exit()
    except FileNotFoundError:
        print("File not found. Please provide a valid local file path.")
        exit()

# Replace 'your_local_file.csv' with the path to your local file if needed
source = url  # Change 'url' to local file path if necessary

# Check if source is a URL or a local file
data = load_data(source)

# Data Exploration (EDA)
print("\nInitial Data Preview:")
print(data.head())
print("\nDataset Information:")
data.info()
print("\nStatistical Summary:")
print(data.describe())

# Data Cleaning
# Removing customerID as it's not useful for prediction
data.drop(['customerID'], axis=1, inplace=True)

# Converting TotalCharges to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Fill missing values with the median
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# Encoding categorical variables
data = pd.get_dummies(data, drop_first=True)

# Data Visualization
sns.countplot(x='Churn_Yes', data=data)
plt.title('Churn Count')
plt.show()

sns.heatmap(data.corr(), annot=False, cmap='viridis')
plt.title('Correlation Heatmap')
plt.show()

# Feature Selection and Splitting Data
X = data.drop('Churn_Yes', axis=1)
y = data['Churn_Yes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building Models
# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Model Evaluation
# Logistic Regression Evaluation
y_pred_lr = lr_model.predict(X_test)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))
print("Accuracy: ", accuracy_score(y_test, y_pred_lr))

# Random Forest Evaluation
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy: ", accuracy_score(y_test, y_pred_rf))

# Save Summary Report and Plots
output_dir = "churn_analysis_report"
os.makedirs(output_dir, exist_ok=True)

# Save classification reports
with open(os.path.join(output_dir, "logistic_regression_report.txt"), "w") as f:
    f.write(classification_report(y_test, y_pred_lr))
    f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred_lr)}\n")

with open(os.path.join(output_dir, "random_forest_report.txt"), "w") as f:
    f.write(classification_report(y_test, y_pred_rf))
    f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred_rf)}\n")

# Save plots
sns.countplot(x='Churn_Yes', data=data)
plt.title('Churn Count')
plt.savefig(os.path.join(output_dir, "churn_count_plot.png"))
plt.clf()

sns.heatmap(data.corr(), annot=False, cmap='viridis')
plt.title('Correlation Heatmap')
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.clf()
