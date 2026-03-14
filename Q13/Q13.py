import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.utils import all_estimators

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Test All Classifiers
results = []

classifiers = all_estimators(type_filter='classifier')

for name, Classifier in classifiers:

    try:

        model = Classifier()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)

        rec = recall_score(y_test, y_pred, average='weighted')

        f1 = f1_score(y_test, y_pred, average='weighted')

        results.append([name, acc, prec, rec, f1])

    except:
        pass

# Convert to DataFrame
df = pd.DataFrame(
    results,
    columns=['Classifier','Accuracy','Precision','Recall','F1']
)

df = df.sort_values(by='Accuracy', ascending=False)

df = df.reset_index(drop=True)

print("\nTop Classifiers:\n")

print(df.head(10))

# Select Best 4 Classifiers
top4 = df.head(4)

print("\nBest 4 Classifiers Selected:\n")

print(top4)


best_names = top4['Classifier'].values

# Create Model List for Stacking
stack_models = []

for name, Classifier in classifiers:

    if name in best_names:

        try:
            stack_models.append((name, Classifier()))
        except:
            pass

# Bagging
bagging = BaggingClassifier(estimator=DecisionTreeClassifier())

bagging.fit(X_train, y_train)

y_pred = bagging.predict(X_test)

print("\nBagging Accuracy:",
      accuracy_score(y_test, y_pred))

# Boosting
boosting = AdaBoostClassifier()

boosting.fit(X_train, y_train)

y_pred = boosting.predict(X_test)

print("Boosting Accuracy:",
      accuracy_score(y_test, y_pred))

# Stacking
stacking = StackingClassifier(
    estimators=stack_models,
    final_estimator=LogisticRegression(max_iter=1000)
)

stacking.fit(X_train, y_train)

y_pred = stacking.predict(X_test)

print("Stacking Accuracy:",
      accuracy_score(y_test, y_pred))