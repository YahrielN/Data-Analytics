import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset from URL
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
df = pd.read_excel(url)

# Drop rows with missing values
df.dropna(subset=['CustomerID'], inplace=True)

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Create a new feature: Total Purchase Amount (UnitPrice * Quantity)
df['TotalAmount'] = df['UnitPrice'] * df['Quantity']

# Group by CustomerID to calculate Recency, Frequency, and Monetary value (RFM)
rfm_df = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,  # Recency
    'InvoiceNo': 'count',  # Frequency
    'TotalAmount': 'sum'  # Monetary Value
})

rfm_df.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalAmount': 'Monetary'
}, inplace=True)

# Scaling the Data
scaler = StandardScaler()
scaled_rfm = scaler.fit_transform(rfm_df)

# Determine Optimal Number of Clusters Using Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_rfm)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(scaled_rfm)

# Add cluster labels to original RFM data
rfm_df['Cluster'] = kmeans.labels_

# Visualize the Segmentation using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_rfm)

plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b', 'c']
for i in range(4):
    plt.scatter(pca_data[rfm_df['Cluster'] == i, 0], pca_data[rfm_df['Cluster'] == i, 1],
                s=50, c=colors[i], label=f'Cluster {i}')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Customer Segments')
plt.legend()
plt.show()

# Interpret the Results
cluster_summary = rfm_df.groupby('Cluster').mean()
print(cluster_summary)
