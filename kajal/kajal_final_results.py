# ============================================
# Kajal - Final Results with All Metrics
# DTW + One-Class SVM
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
# ============================================
print("Loading data...")
base_path = r"C:\Users\Lenovo\Documents\manit\Menstrual_project\Menstrual-Health-Anomaly-Detector\Day2"

X_train = np.load(os.path.join(base_path, "X_train.npy"))
X_test  = np.load(os.path.join(base_path, "X_test.npy"))
y_train = np.load(os.path.join(base_path, "y_train.npy"))
y_test  = np.load(os.path.join(base_path, "y_test.npy"))
print(f"Data loaded! X_train: {X_train.shape}, X_test: {X_test.shape}")

# ============================================
# 2. FLATTEN FOR SVM
# ============================================
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat  = X_test.reshape(len(X_test), -1)

# ============================================
# 3. DTW MODEL
# ============================================
print("\nRunning DTW...")
start_dtw = time.time()

# Extract cycle_length (feature 0)
train_seq = X_train[:, :, 0]
test_seq  = X_test[:, :, 0]

# Normal template banao
normal_seq = train_seq[y_train == 0]
template   = np.mean(normal_seq, axis=0)

# Har normal sample ki template se distance nikalo
train_dists = []
for seq in normal_seq:
    dist = dtw(seq, template)
    train_dists.append(dist)

# Threshold set karo
threshold = np.mean(train_dists) + 2 * np.std(train_dists)
print(f"DTW Threshold: {threshold:.4f}")

# Predict on test set
dtw_scores      = []
predictions_dtw = []
for seq in test_seq:
    dist = dtw(seq, template)
    dtw_scores.append(dist)
    predictions_dtw.append(1 if dist > threshold else 0)

predictions_dtw = np.array(predictions_dtw)
dtw_scores      = np.array(dtw_scores)
elapsed_dtw     = time.time() - start_dtw

# DTW Metrics
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
# ============================================
print("\nRunning One-Class SVM...")
start_svm = time.time()

# Scale
X_normal        = X_train_flat[y_train == 0]
scaler          = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal)
X_test_scaled   = scaler.transform(X_test_flat)

# Train only on normal data
ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
ocsvm.fit(X_normal_scaled)

# Predict
raw_preds   = ocsvm.predict(X_test_scaled)
ocsvm_preds = (raw_preds == -1).astype(int)
svm_scores  = -ocsvm.decision_function(X_test_scaled)
elapsed_svm = time.time() - start_svm

# SVM Metrics
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
# ============================================
figures_path = r"C:\Users\Lenovo\Documents\manit\Menstrual_project\Menstrual-Health-Anomaly-Detector\kajal\figures"
os.makedirs(figures_path, exist_ok=True)

fpr_dtw, tpr_dtw, _ = roc_curve(y_test, dtw_scores)
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_scores)

plt.figure(figsize=(7, 5))
plt.plot(fpr_dtw, tpr_dtw, label=f'DTW (AUC = {dtw_auc:.2f})',
         color='#2196F3', linewidth=2)
plt.plot(fpr_svm, tpr_svm, label=f'One-Class SVM (AUC = {svm_auc:.2f})',
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