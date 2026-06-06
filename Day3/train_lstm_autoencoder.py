import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, RepeatVector, TimeDistributed
)
import matplotlib.pyplot as plt

# ── 1. Load data ─────────────────────────────────────────────────
X_train = np.load("../Day2/X_train.npy")
y_train = np.load("../Day2/y_train.npy")
X_test  = np.load("../Day2/X_test.npy")
y_test  = np.load("../Day2/y_test.npy")

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

# ── 2. Train ONLY on normal windows (autoencoder logic) ──────────
X_train_normal = X_train[y_train == 0]
print(f"Training on normal windows only: {X_train_normal.shape}")

# ── 3. Build LSTM Autoencoder ────────────────────────────────────
timesteps = 6
features  = 6

inputs = Input(shape=(timesteps, features))

# Encoder
x = LSTM(64, return_sequences=True)(inputs)
x = LSTM(32)(x)

# Bottleneck
latent = Dense(16, activation="relu")(x)

# Decoder
x = RepeatVector(timesteps)(latent)
x = LSTM(32, return_sequences=True)(x)
x = LSTM(64, return_sequences=True)(x)

outputs = TimeDistributed(Dense(features))(x)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse")
model.summary()

# ── 4. Train ─────────────────────────────────────────────────────
print("\nTraining LSTM Autoencoder...")
history = model.fit(
    X_train_normal, X_train_normal,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    shuffle=True
)

# ── 5. Save model ────────────────────────────────────────────────
model.save("lstm_autoencoder.keras")
print("Model saved: lstm_autoencoder.keras")

# ── 6. Compute reconstruction error on normal train windows ──────
recon_normal = model.predict(X_train_normal)
errors = np.mean(np.square(X_train_normal - recon_normal), axis=(1, 2))

# ── 7. Set threshold ─────────────────────────────────────────────
threshold = errors.mean() + 2 * errors.std()
print(f"\nAnomaly threshold: {threshold:.6f}")
np.save("threshold.npy", threshold)

# ── 8. Evaluate on full test set ─────────────────────────────────
recon_test = model.predict(X_test)
test_errors = np.mean(np.square(X_test - recon_test), axis=(1, 2))
preds = (test_errors > threshold).astype(int)

from sklearn.metrics import classification_report, confusion_matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))
print("\nClassification Report:")
print(classification_report(y_test, preds))

# ── 9. Save reconstruction error plot (paper figure) ─────────────
plt.figure(figsize=(10, 5))
plt.hist(errors, bins=40, alpha=0.7, label="Normal windows")
plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold: {threshold:.4f}")
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Count")
plt.title("LSTM Autoencoder Reconstruction Error Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("reconstruction_error_distribution.png")
print("Plot saved: reconstruction_error_distribution.png")