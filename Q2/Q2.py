import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.decomposition import PCA

# 1. Load MNIST
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Normalize
X_train_full = X_train_full.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Flatten images (28x28 → 784)
X_train_full = X_train_full.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

# 2. Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000)

log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_val)

# 3. Evaluation Metrics
print("\nModel Performance (Before Tuning):")
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred, average="weighted"))
print("Recall:", recall_score(y_val, y_pred, average="weighted"))
print("F1 Score:", f1_score(y_val, y_pred, average="weighted"))

print("\nClassification Report:\n")
print(classification_report(y_val, y_pred))

# 4. Hyperparameter Tuning (GridSearchCV)
param_grid = {
    "C": [0.01, 0.1, 1],
    "solver": ["lbfgs"],
    "max_iter": [1000]
}

grid = GridSearchCV(LogisticRegression(), param_grid, cv=3, scoring="accuracy")
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)

best_model = grid.best_estimator_

y_pred_tuned = best_model.predict(X_val)

print("\nModel Performance (After Tuning):")
print("Accuracy:", accuracy_score(y_val, y_pred_tuned))
print("Precision:", precision_score(y_val, y_pred_tuned, average="weighted"))
print("Recall:", recall_score(y_val, y_pred_tuned, average="weighted"))
print("F1 Score:", f1_score(y_val, y_pred_tuned, average="weighted"))

# 5. Decision Boundary Visualization (PCA → 2D)
print("\nGenerating Decision Boundary Visualization...")

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Train logistic regression on PCA data
log_reg_2d = LogisticRegression(max_iter=1000)
log_reg_2d.fit(X_train_pca, y_train)

# Create mesh grid
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = log_reg_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, s=5)
plt.title("Decision Boundary (PCA 2D Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

print("\nClassification Task Completed Successfully ✅")
