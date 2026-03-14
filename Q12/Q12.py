import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import skfuzzy as fuzz

# Load Dataset
iris = load_iris()
X = iris.data[:, :2]

k = 3

# K-MEANS
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans_labels = kmeans.fit_predict(X)

print("K-Means Centers:\n", kmeans.cluster_centers_)

# K-MEDOIDS (Manual Implementation)
def k_medoids(X, k, max_iter=100):

    m, n = X.shape
    medoids = X[np.random.choice(m, k, replace=False)]

    for _ in range(max_iter):

        distances = np.linalg.norm(X[:, np.newaxis] - medoids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_medoids = np.copy(medoids)

        for i in range(k):

            cluster_points = X[labels == i]

            if len(cluster_points) == 0:
                continue

            costs = np.sum(
                np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2),
                axis=1
            )

            new_medoids[i] = cluster_points[np.argmin(costs)]

        if np.all(medoids == new_medoids):
            break

        medoids = new_medoids

    return medoids, labels


medoids, kmedoids_labels = k_medoids(X, k)

print("K-Medoids Centers:\n", medoids)

# FUZZY C-MEANS
X_t = X.T

cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X_t,
    c=k,
    m=2,
    error=0.005,
    maxiter=1000
)

fcm_labels = np.argmax(u, axis=0)

print("Fuzzy C-Means Centers:\n", cntr)

# Visualization
plt.figure(figsize=(15,5))


# K-Means
plt.subplot(1,3,1)
plt.scatter(X[:,0], X[:,1], c=kmeans_labels)
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            marker='X',
            color='red')
plt.title("K-Means")


# K-Medoids
plt.subplot(1,3,2)
plt.scatter(X[:,0], X[:,1], c=kmedoids_labels)
plt.scatter(medoids[:,0],
            medoids[:,1],
            marker='X',
            color='red')
plt.title("K-Medoids")


# Fuzzy C-Means
plt.subplot(1,3,3)
plt.scatter(X[:,0], X[:,1], c=fcm_labels)
plt.scatter(cntr[:,0],
            cntr[:,1],
            marker='X',
            color='red')
plt.title("Fuzzy C-Means")

plt.show()