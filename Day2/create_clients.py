import numpy as np
import os

# Load training data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

num_clients = 50

# Shuffle data
indices = np.random.permutation(len(X_train))
X_train = X_train[indices]
y_train = y_train[indices]

# Create folder
os.makedirs("federated_clients", exist_ok=True)

# Split data
X_splits = np.array_split(X_train, num_clients)
y_splits = np.array_split(y_train, num_clients)

# Save each client
for i in range(num_clients):
    np.savez(
        f"federated_clients/client_{i}.npz",
        X=X_splits[i],
        y=y_splits[i]
    )

print(f"Created {num_clients} clients")

# Quick check
for i in range(3):
    print(
        f"Client {i}:",
        X_splits[i].shape[0],
        "samples"
    )