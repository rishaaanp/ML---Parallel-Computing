import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Original Dataset Shape:", X.shape)

# Standardize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function for Cross Validation
def evaluate_model(X_new, y):

    model = SVC(kernel='linear')

    scores = cross_val_score(model, X_new, y, cv=5)

    return scores.mean()

# Case 1: Reduce to 2 Features
print("\n===== CASE 1: 2 FEATURES =====")

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_score = evaluate_model(X_pca, y)
print("PCA Accuracy:", pca_score)


# LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

lda_score = evaluate_model(X_lda, y)
print("LDA Accuracy:", lda_score)


# SVD
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X_scaled)

svd_score = evaluate_model(X_svd, y)
print("SVD Accuracy:", svd_score)


# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

tsne_score = evaluate_model(X_tsne, y)
print("TSNE Accuracy:", tsne_score)

# Case 2: Reduce to 3 Features
print("\n===== CASE 2: 3 FEATURES =====")

# PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

pca_score = evaluate_model(X_pca, y)
print("PCA Accuracy:", pca_score)


# LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

lda_score = evaluate_model(X_lda, y)
print("LDA Accuracy:", lda_score)


# SVD
svd = TruncatedSVD(n_components=3)
X_svd = svd.fit_transform(X_scaled)

svd_score = evaluate_model(X_svd, y)
print("SVD Accuracy:", svd_score)


# t-SNE
tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

tsne_score = evaluate_model(X_tsne, y)
print("TSNE Accuracy:", tsne_score)