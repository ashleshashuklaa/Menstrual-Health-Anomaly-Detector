# ============================================
# Kajal - One-Class SVM for Menstrual Anomaly Detection
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import os

# ============================================
# 1. DATA LOAD KARO
# ============================================
print("Loading data...")

base_path = r"C:\Users\Lenovo\Documents\manit\Menstrual_project\Menstrual-Health-Anomaly-Detector\Day2"

X_train = np.load(os.path.join(base_path, "X_train.npy"))
X_test  = np.load(os.path.join(base_path, "X_test.npy"))
y_train = np.load(os.path.join(base_path, "y_train.npy"))
y_test  = np.load(os.path.join(base_path, "y_test.npy"))

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")

# ============================================
# 2. DATA FLATTEN KARO
# ============================================
# One-Class SVM ko 2D data chahiye
# (samples, 6, 6) → (samples, 36)

X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat  = X_test.reshape(len(X_test), -1)

print(f"Flattened X_train: {X_train_flat.shape}")
print(f"Flattened X_test : {X_test_flat.shape}")

# ============================================
# 3. SIRF NORMAL DATA PE TRAIN KARO
# ============================================
X_train_normal = X_train_flat[y_train == 0]
print(f"\nTraining on {len(X_train_normal)} normal samples only...")

# ============================================
# 4. SCALE KARO (Important!)
# ============================================
scaler = StandardScaler()
X_train_normal_scaled = scaler.fit_transform(X_train_normal)
X_test_scaled         = scaler.transform(X_test_flat)

# ============================================
# 5. ONE-CLASS SVM TRAIN KARO
# ============================================
print("Training One-Class SVM with RBF kernel...")

ocsvm = OneClassSVM(
    kernel='rbf',
    nu=0.05,        # 5% anomaly rate expect kar rahe hain
    gamma='scale'
)
ocsvm.fit(X_train_normal_scaled)
print("Training complete! ✅")

# ============================================
# 6. PREDICT KARO
# ============================================
# One-Class SVM: +1 = normal, -1 = anomaly
raw_pred = ocsvm.predict(X_test_scaled)

# Convert karo: -1 → 1 (anomaly), +1 → 0 (normal)
y_pred_svm = (raw_pred == -1).astype(int)

# Score nikalo for AUC-ROC
scores = -ocsvm.decision_function(X_test_scaled)

# ============================================
# 7. RESULTS CALCULATE KARO
# ============================================
f1_svm  = f1_score(y_test, y_pred_svm)
auc_svm = roc_auc_score(y_test, scores)

print(f"\n======= ONE-CLASS SVM RESULTS =======")
print(f"F1 Score : {f1_svm:.4f}")
print(f"AUC-ROC  : {auc_svm:.4f}")
print(f"=====================================")

# ============================================
# 8. CHART BANAO AUR SAVE KARO
# ============================================
figures_path = r"C:\Users\Lenovo\Documents\manit\Menstrual_project\figures"
os.makedirs(figures_path, exist_ok=True)

plt.figure(figsize=(10, 5))
plt.hist(scores[y_test==0], bins=30, alpha=0.7,
         color='green', label='Normal')
plt.hist(scores[y_test==1], bins=30, alpha=0.7,
         color='red', label='Anomaly')
plt.xlabel('Anomaly Score')
plt.ylabel('Count')
plt.title('One-Class SVM - Anomaly Score Distribution')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "ocsvm_scores.png"))
plt.show()
print("Chart saved!")

# ============================================
# 9. RESULTS CSV UPDATE KARO
# ============================================
results_path = r"C:\Users\Lenovo\Documents\manit\Menstrual_project\results"

# Pehle wali DTW results CSV load karo
csv_path = os.path.join(results_path, "kajal_results.csv")
existing = pd.read_csv(csv_path)

# Naya row add karo
new_row = pd.DataFrame([{
    "Model"   : "One-Class SVM",
    "F1_Score": round(f1_svm, 4),
    "AUC_ROC" : round(auc_svm, 4)
}])

# Dono combine karo
final_results = pd.concat([existing, new_row], ignore_index=True)
final_results.to_csv(csv_path, index=False)

print(f"\n====== FINAL RESULTS TABLE ======")
print(final_results.to_string(index=False))
print(f"=================================")
print("\nAll results saved! ✅")