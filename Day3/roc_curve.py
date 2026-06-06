import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report
from tensorflow.keras.models import load_model

# Load model and data
model = load_model("lstm_autoencoder.keras")
X_test = np.load("../Day2/X_test.npy")
y_test = np.load("../Day2/y_test.npy")
threshold = np.load("threshold.npy")

# Reconstruction errors
recon = model.predict(X_test)
errors = np.mean(np.square(X_test - recon), axis=(1, 2))

# ROC curve
fpr, tpr, _ = roc_curve(y_test, errors)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2,
         label=f"LSTM Autoencoder (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — LSTM Anomaly Detection")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
print(f"AUC-ROC: {roc_auc:.4f}")
print("Saved: roc_curve.png")

# Print mean/std/threshold for paper
X_train_normal = np.load("../Day2/X_train.npy")
y_train = np.load("../Day2/y_train.npy")
X_train_normal = X_train_normal[y_train == 0]
recon_normal = model.predict(X_train_normal)
train_errors = np.mean(np.square(X_train_normal - recon_normal), axis=(1, 2))

print(f"Mean Error : {train_errors.mean():.6f}")
print(f"Std Error  : {train_errors.std():.6f}")
print(f"Threshold  : {threshold:.6f}")