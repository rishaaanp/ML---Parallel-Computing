import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("supermarket_sales - Sheet1.csv")

print("Dataset Loaded Successfully!")

print("\nDataset Shape:", df.shape)


# -----------------------------
# Show Dataset Preview
# -----------------------------

print("\nFirst 5 Rows of Dataset:\n")
print(df.head())

print("\nDataset Information:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())


# -----------------------------
# Select Features for Clustering
# -----------------------------

data = df[['Unit price','Quantity','Total','Rating']]


# -----------------------------
# Scale Data
# -----------------------------

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)


# -----------------------------
# Elbow Method
# -----------------------------

wcss = []

for i in range(1,11):

    kmeans = KMeans(n_clusters=i, random_state=42)

    kmeans.fit(scaled_data)

    wcss.append(kmeans.inertia_)


plt.figure(figsize=(6,4))

plt.plot(range(1,11), wcss, marker='o')

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.show()


# -----------------------------
# Apply K-Means
# -----------------------------

k = 3

kmeans = KMeans(n_clusters=k, random_state=42)

clusters = kmeans.fit_predict(scaled_data)

data['Cluster'] = clusters


# -----------------------------
# Cluster Analysis
# -----------------------------

print("\nCluster Characteristics:\n")

print(data.groupby('Cluster').mean())


# -----------------------------
# PCA for Visualization
# -----------------------------

pca = PCA(n_components=2)

reduced = pca.fit_transform(scaled_data)


# -----------------------------
# Scatter Plot Visualization
# -----------------------------

plt.figure(figsize=(7,5))

plt.scatter(reduced[:,0], reduced[:,1], c=clusters, cmap='viridis')

plt.title("Customer Clusters (PCA Visualization)")

plt.xlabel("PCA Component 1")

plt.ylabel("PCA Component 2")

plt.show()