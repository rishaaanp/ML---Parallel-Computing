import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

print("TensorFlow Version:", tf.__version__)

# 1. Load MNIST Dataset
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

print("\nOriginal Dataset Shapes:")
print("Train Full:", X_train_full.shape)
print("Test:", X_test.shape)

# 2. Normalize Pixel Values
X_train_full = X_train_full.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# 3. One-Hot Encode Labels
y_train_full = to_categorical(y_train_full, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# 4. Train - Validation Split
X_train, X_val, y_train, y_val = train_test_split(

    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y_train_full, axis=1)

)

# 5. Print Final Shapes
print("\nFinal Dataset Shapes:")

print("Training set :", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)
print("Test set :", X_test.shape, y_test.shape)

# 6. Verify Normalization
print("\nPixel Value Range:")
print("Min:", X_train.min())
print("Max:", X_train.max())

print("\nPreprocessing Completed Successfully ✅")
