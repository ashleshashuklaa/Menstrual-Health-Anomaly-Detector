# ============================================
# Kajal - Classical Anomaly Detection Models
# Methods  : DTW + One-Class SVM
# Project  : Menstrual Health Anomaly Detector
# Team     : SheCares
# Date     : June 2026
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from tslearn.metrics import dtw
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)

# ============================================
# 1. LOAD DATA
# Load preprocessed sliding window data
# Shape: (samples, 6 cycles, 6 features)
# Features: cycle_length, cramp_score, mood_score,
#           flow_intensity, sleep_quality, stress_score
# ============================================
print("Loading data...")
base_path = r"C:\Users\Lenovo\Documents\manit\Menstrual_project\Menstrual-Health-Anomaly-Detector\Day2"

X_train = np.load(os.path.join(base_path, "X_train.npy"))  # shape: (939, 6, 6)
X_test  = np.load(os.path.join(base_path, "X_test.npy"))   # shape: (235, 6, 6)
y_train = np.load(os.path.join(base_path, "y_train.npy"))  # 0=normal, 1=anomaly
y_test  = np.load(os.path.join(base_path, "y_test.npy"))   # 0=normal, 1=anomaly

print(f"Data loaded! X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Train — Normal: {sum(y_train==0)}, Anomaly: {sum(y_train==1)}")
print(f"Test  — Normal: {sum(y_test==0)},  Anomaly: {sum(y_test==1)}")

# ============================================
# 2. FLATTEN WINDOWS FOR SVM
# Classical models need 2D input
# Flatten (6 cycles x 6 features) = 36 features per sample
# ============================================
X_train_flat = X_train.reshape(len(X_train), -1)  # shape: (939, 36)
X_test_flat  = X_test.reshape(len(X_test), -1)    # shape: (235, 36)
print(f"\nFlattened — Train: {X_train_flat.shape}, Test: {X_test_flat.shape}")

# ============================================
# 3. DTW MODEL
# Dynamic Time Warping — Time Series Anomaly Detection
#
# How it works:
# - Extract cycle_length (feature index 0) from windows
# - Build a "normal template" = average of all normal cycles
# - Calculate DTW distance of each sample from template
# - If distance > threshold → anomaly
#
# Threshold = mean + 2*std of normal training distances
# ============================================
print("\nRunning DTW...")
start_dtw = time.time()

# Step 1: Extract cycle_length feature (index 0)
train_seq = X_train[:, :, 0]  # shape: (939, 6)
test_seq  = X_test[:, :, 0]   # shape: (235, 6)

# Step 2: Build normal template from training data
normal_seq = train_seq[y_train == 0]           # only normal samples
template   = np.mean(normal_seq, axis=0)        # average normal pattern

# Step 3: Calculate DTW distance of each normal sample from template
# Used to determine anomaly threshold
train_dists = []
for seq in normal_seq:
    dist = dtw(seq, template)
    train_dists.append(dist)

# Step 4: Set threshold = mean + 2*std
# Samples beyond this distance are flagged as anomaly
threshold = np.mean(train_dists) + 2 * np.std(train_dists)
print(f"DTW Threshold: {threshold:.4f}")

# Step 5: Predict on test set
dtw_scores      = []
predictions_dtw = []
for seq in test_seq:
    dist = dtw(seq, template)           # distance from normal template
    dtw_scores.append(dist)
    predictions_dtw.append(1 if dist > threshold else 0)  # 1=anomaly

predictions_dtw = np.array(predictions_dtw)
dtw_scores      = np.array(dtw_scores)
elapsed_dtw     = time.time() - start_dtw

# Step 6: Calculate evaluation metrics
dtw_precision = precision_score(y_test, predictions_dtw, zero_division=0)
dtw_recall    = recall_score(y_test, predictions_dtw, zero_division=0)
dtw_f1        = f1_score(y_test, predictions_dtw, zero_division=0)
dtw_auc       = roc_auc_score(y_test, dtw_scores)

print(f"\n===== DTW RESULTS =====")
print(f"Precision : {dtw_precision:.4f}")
print(f"Recall    : {dtw_recall:.4f}")
print(f"F1 Score  : {dtw_f1:.4f}")
print(f"AUC-ROC   : {dtw_auc:.4f}")
print(f"Time      : {elapsed_dtw:.2f} sec")

# ============================================
# 4. ONE-CLASS SVM MODEL
# Unsupervised Anomaly Detection
#
# How it works:
# - Train ONLY on normal samples
# - Learns boundary of "normal" data
# - At test time: outside boundary = anomaly
# - RBF kernel creates non-linear boundary
# - nu=0.05 means ~5% anomaly rate expected
# ============================================
print("\nRunning One-Class SVM...")
start_svm = time.time()

# Step 1: Use only normal training samples
X_normal = X_train_flat[y_train == 0]  # 758 normal samples only

# Step 2: Standardize features (important for SVM)
scaler          = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal)   # fit on normal only
X_test_scaled   = scaler.transform(X_test_flat)    # transform test data

# Step 3: Train One-Class SVM
ocsvm = OneClassSVM(
    kernel='rbf',    # RBF kernel for non-linear boundary
    nu=0.05,         # expected anomaly fraction
    gamma='scale'    # automatic gamma scaling
)
ocsvm.fit(X_normal_scaled)

# Step 4: Predict on test set
# +1 = normal, -1 = anomaly
raw_preds   = ocsvm.predict(X_test_scaled)
ocsvm_preds = (raw_preds == -1).astype(int)  # convert: -1 → 1, +1 → 0

# Decision scores for AUC-ROC (higher = more anomalous)
svm_scores  = -ocsvm.decision_function(X_test_scaled)
elapsed_svm = time.time() - start_svm

# Step 5: Calculate evaluation metrics
svm_precision = precision_score(y_test, ocsvm_preds, zero_division=0)
svm_recall    = recall_score(y_test, ocsvm_preds, zero_division=0)
svm_f1        = f1_score(y_test, ocsvm_preds, zero_division=0)
svm_auc       = roc_auc_score(y_test, svm_scores)

print(f"\n===== ONE-CLASS SVM RESULTS =====")
print(f"Precision : {svm_precision:.4f}")
print(f"Recall    : {svm_recall:.4f}")
print(f"F1 Score  : {svm_f1:.4f}")
print(f"AUC-ROC   : {svm_auc:.4f}")
print(f"Time      : {elapsed_svm:.2f} sec")

# ============================================
# 5. SAVE RESULTS CSV
# Save all metrics for master comparison table
# Format: model | precision | recall | f1 | auc_roc | train_time_sec
# ============================================
results_path = r"C:\Users\Lenovo\Documents\manit\Menstrual_project\Menstrual-Health-Anomaly-Detector\kajal\results"
os.makedirs(results_path, exist_ok=True)

results = pd.DataFrame([
    {
        'model'         : 'DTW',
        'precision'     : round(dtw_precision, 4),
        'recall'        : round(dtw_recall, 4),
        'f1'            : round(dtw_f1, 4),
        'auc_roc'       : round(dtw_auc, 4),
        'train_time_sec': round(elapsed_dtw, 2)
    },
    {
        'model'         : 'OneClassSVM',
        'precision'     : round(svm_precision, 4),
        'recall'        : round(svm_recall, 4),
        'f1'            : round(svm_f1, 4),
        'auc_roc'       : round(svm_auc, 4),
        'train_time_sec': round(elapsed_svm, 2)
    }
])

csv_path = os.path.join(results_path, "classical_results.csv")
results.to_csv(csv_path, index=False)

print(f"\n====== FINAL RESULTS TABLE ======")
print(results.to_string(index=False))
print(f"=================================")
print("Results CSV saved! ✅")

# ============================================
# 6. ROC CURVE PLOT
# Plot both DTW and SVM ROC curves on same graph
# Ashlesha will combine all 6 model curves
# into the paper's headline figure
# ============================================
figures_path = r"C:\Users\Lenovo\Documents\manit\Menstrual_project\Menstrual-Health-Anomaly-Detector\kajal\figures"
os.makedirs(figures_path, exist_ok=True)

# Calculate ROC curve points for both models
fpr_dtw, tpr_dtw, _ = roc_curve(y_test, dtw_scores)
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_scores)

# Plot
plt.figure(figsize=(7, 5))
plt.plot(fpr_dtw, tpr_dtw,
         label=f'DTW (AUC = {dtw_auc:.2f})',
         color='#2196F3', linewidth=2)
plt.plot(fpr_svm, tpr_svm,
         label=f'One-Class SVM (AUC = {svm_auc:.2f})',
         color='#E91E63', linewidth=2)
plt.plot([0,1], [0,1], '--', color='#CCCCCC', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — DTW vs One-Class SVM')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "roc_classical.png"), dpi=150)
plt.show()
print("ROC Curve saved! ✅")
print("\nAll Done! 🎉")