import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report

# (a) Load Iris Dataset
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

print("\nDataset Shape:", X.shape)
print("\nFirst 5 Rows:")
print(X.head())

# (b) Exploratory Data Analysis (EDA)
print("\nDataset Info:")
print(X.describe())

# Pairplot
sns.pairplot(pd.concat([X, pd.Series(y, name="target")], axis=1),
             hue="target")
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")

plt.title("Feature Correlation Heatmap")
plt.show()

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardization (important for SVM)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline Model (Before Feature Selection)
baseline_model = SVC()

baseline_model.fit(X_train_scaled, y_train)

baseline_pred = baseline_model.predict(X_test_scaled)

baseline_accuracy = accuracy_score(y_test, baseline_pred)

print("\nBaseline Accuracy (All Features):", baseline_accuracy)

# (c1) Univariate Feature Selection
selector_uni = SelectKBest(score_func=f_classif, k=2)

X_uni_train = selector_uni.fit_transform(X_train_scaled, y_train)
X_uni_test = selector_uni.transform(X_test_scaled)

print("\nUnivariate Selected Features:")
print(X.columns[selector_uni.get_support()])

model_uni = SVC()

model_uni.fit(X_uni_train, y_train)

pred_uni = model_uni.predict(X_uni_test)

acc_uni = accuracy_score(y_test, pred_uni)

print("Accuracy After Univariate Selection:", acc_uni)

# (c2) Feature Importance Using Random Forest
rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

importances = rf.feature_importances_

importance_df = pd.DataFrame({

    "Feature": X.columns,
    "Importance": importances

}).sort_values(by="Importance", ascending=False)

print("\nRandom Forest Feature Importance:")

print(importance_df)

# Plot Importance
plt.figure(figsize=(7,4))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df
)

plt.title("Random Forest Feature Importance")

plt.show()

# Select top 2 important features
top_features = importance_df["Feature"].iloc[:2]

X_rf_train = X_train[top_features]
X_rf_test = X_test[top_features]

rf_scaler = StandardScaler()

X_rf_train = rf_scaler.fit_transform(X_rf_train)
X_rf_test = rf_scaler.transform(X_rf_test)

model_rf = SVC()

model_rf.fit(X_rf_train, y_train)

pred_rf = model_rf.predict(X_rf_test)

acc_rf = accuracy_score(y_test, pred_rf)

print("\nAccuracy After Random Forest Selection:", acc_rf)

# (c3) Recursive Feature Elimination (RFE) using SVM
svm = SVC(kernel="linear")

rfe = RFE(estimator=svm, n_features_to_select=2)

X_rfe_train = rfe.fit_transform(X_train_scaled, y_train)
X_rfe_test = rfe.transform(X_test_scaled)

print("\nRFE Selected Features:")
print(X.columns[rfe.support_])

model_rfe = SVC()

model_rfe.fit(X_rfe_train, y_train)

pred_rfe = model_rfe.predict(X_rfe_test)

acc_rfe = accuracy_score(y_test, pred_rfe)

print("Accuracy After RFE:", acc_rfe)

# (d,e) Compare Performance
print("\nPerformance Comparison:")

print("Baseline Accuracy :", baseline_accuracy)
print("Univariate Accuracy :", acc_uni)
print("Random Forest Accuracy :", acc_rf)
print("RFE Accuracy :", acc_rfe)

print("\nClassification Report (Best Model Example):")

print(classification_report(y_test, pred_rfe))

print("\nFeature Selection Task Completed Successfully ✅")