import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import requests
from io import StringIO

# Function to load dataset from URL or local file
# Remove the url if you wish to use a local file
def load_dataset():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
    try:
        # Attempt to download the dataset from the URL
        response = requests.get(url)
        response.raise_for_status()
        print("Dataset successfully loaded from URL.")
        # The dataset is within a zip file; read it directly
        from zipfile import ZipFile
        from io import BytesIO
        with ZipFile(BytesIO(response.content)) as thezip:
            with thezip.open('bank-additional/bank-additional-full.csv') as thefile:
                return pd.read_csv(thefile, sep=';')
    except requests.RequestException:
        print("Failed to download dataset from URL. Please select the dataset file manually.")
        # Use Tkinter to prompt user for file location
        Tk().withdraw()  # Close the root window
        file_path = askopenfilename(title="Select the marketing campaign CSV file", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            raise FileNotFoundError("No file was selected. Please provide the dataset file.")
        return pd.read_csv(file_path, sep=';')

# Load the dataset
df = load_dataset()

# Data Cleaning and Preprocessing
df = df.dropna()  # Drop rows with missing values
df = df[df['age'] > 0]  # Remove rows with invalid age

# Feature Engineering - Creating new metrics
# Assuming 'duration' is the call duration in seconds
df['Total_Spend'] = df['duration']  # Placeholder for actual spending data
df['ResponseRate'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Descriptive Analysis - Summary statistics
print(df.describe())

# Visualization - Conversion Rates by Age Group
age_groups = pd.cut(df['age'], bins=[18, 30, 40, 50, 60, 100], labels=['18-30', '31-40', '41-50', '51-60', '60+'])
df['AgeGroup'] = age_groups
conversion_rate = df.groupby('AgeGroup')['ResponseRate'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(conversion_rate['AgeGroup'], conversion_rate['ResponseRate'], color='skyblue')
plt.xlabel('Age Group')
plt.ylabel('Average Response Rate')
plt.title('Conversion Rate by Age Group')
plt.show()

# Interactive Visualization using Plotly - Spending by Income
# Note: The dataset does not contain 'Income' or 'Total_Spend' columns; this is a placeholder
fig = px.scatter(df, x='age', y='duration', color='AgeGroup',
                 title='Call Duration vs Age by Age Group', labels={'age': 'Age', 'duration': 'Call Duration (seconds)'})
fig.show()

# Clustering to Identify High-Value Segments
# Note: Using 'age' and 'duration' as proxies for clustering due to lack of 'Income' and 'Total_Spend' data
features = df[['age', 'duration']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize Clusters
fig = px.scatter(df, x='age', y='duration', color='Cluster',
                 title='Customer Segments Based on Age and Call Duration', labels={'age': 'Age', 'duration': 'Call Duration (seconds)'})
fig.show()

# Recommendations based on Clustering
for cluster in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster]
    avg_age = cluster_data['age'].mean()
    avg_duration = cluster_data['duration'].mean()
    print(f'Cluster {cluster}: Average Age = {avg_age}, Average Call Duration = {avg_duration}')
    if avg_duration > df['duration'].mean():
        print(f'Recommend targeting Cluster {cluster} with longer call scripts.')
    else:
        print(f'Consider targeting Cluster {cluster} with shorter, more concise calls.')

# Save Summary Report
summary = df.groupby('Cluster').agg({'age': 'mean', 'duration': 'mean', 'ResponseRate': 'mean'}).reset_index()
summary.to_csv('cluster_summary_report.csv', index=False)
print("Summary report saved as 'cluster_summary_report.csv'")
