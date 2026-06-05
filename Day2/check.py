import numpy as np

X = np.load("../Day1/X_windows.npy")
y = np.load("../Day1/y_labels.npy")

print(X.shape)
print(y.shape)
print(np.bincount(y))