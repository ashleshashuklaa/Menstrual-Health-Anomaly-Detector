import numpy as np
from sklearn.model_selection import train_test_split

X = np.load("X_combined.npy")
y = np.load("y_combined.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape, "| class counts:", np.bincount(y_train))
print("y_test:", y_test.shape,  "| class counts:", np.bincount(y_test))
print("Done. 4 files saved in Day2/")