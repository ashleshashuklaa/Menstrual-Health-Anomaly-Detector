# ============================================
# Kajal - DTW Model for Menstrual Anomaly Detection
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.metrics import dtw
from sklearn.metrics import f1_score, roc_auc_score
import os

# ============================================
# 1. DATA LOAD KARO
# ============================================
print("Loading data...")

# Path set karo - Day2 folder se data load hoga
base_path = r"C:\Users\Lenovo\Documents\manit\Menstrual_project\Menstrual-Health-Anomaly-Detector\Day2"

X_train = np.load(os.path.join(base_path, "X_train.npy"))
X_test  = np.load(os.path.join(base_path, "X_test.npy"))
y_train = np.load(os.path.join(base_path, "y_train.npy"))
y_test  = np.load(os.path.join(base_path, "y_test.npy"))

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"Train - Normal: {sum(y_train==0)}, Anomaly: {sum(y_train==1)}")
print(f"Test  - Normal: {sum(y_test==0)}, Anomaly: {sum(y_test==1)}")

# ============================================
# 2. CYCLE_LENGTH FEATURE NIKALO (Feature 0)
# ============================================
# Data shape: (samples, 6 cycles, 6 features)
# Feature 0 = cycle_length

train_cycle = X_train[:, :, 0]  # shape: (939, 6)
test_cycle  = X_test[:, :, 0]   # shape: (235, 6)

# ============================================
# 3. NORMAL TEMPLATE BANAO (DTW Reference)
# ============================================
# Sirf normal training samples se average nikalo
normal_train = train_cycle[y_train == 0]
template = np.mean(normal_train, axis=0)  # shape: (6,)
print(f"\nNormal template (avg cycle pattern): {template}")

# ============================================
# 4. DTW DISTANCE CALCULATE KARO
# ============================================
print("\nCalculating DTW distances...")

dtw_distances = []
for i in range(len(test_cycle)):
    dist = dtw(test_cycle[i], template)
    dtw_distances.append(dist)

dtw_distances = np.array(dtw_distances)

# ============================================
# 5. THRESHOLD SET KARO
# ============================================
# Normal training distances se threshold nikalo
normal_distances = []
for i in range(len(train_cycle)):
    dist = dtw(train_cycle[i], template)
    normal_distances.append(dist)

normal_distances = np.array(normal_distances)
threshold = np.mean(normal_distances) + 2 * np.std(normal_distances)
print(f"DTW Threshold: {threshold:.4f}")

# ============================================
# 6. PREDICT KARO
# ============================================
# Distance > threshold = Anomaly (1), else Normal (0)
y_pred_dtw = (dtw_distances > threshold).astype(int)

# ============================================
# 7. RESULTS CALCULATE KARO
# ============================================
f1_dtw  = f1_score(y_test, y_pred_dtw)
auc_dtw = roc_auc_score(y_test, dtw_distances)

print(f"\n========== DTW RESULTS ==========")
print(f"F1 Score : {f1_dtw:.4f}")
print(f"AUC-ROC  : {auc_dtw:.4f}")
print(f"=================================")

# ============================================
# 8. CHART BANAO AUR SAVE KARO
# ============================================
figures_path = r"C:\Users\Lenovo\Documents\manit\Menstrual_project\figures"
os.makedirs(figures_path, exist_ok=True)

plt.figure(figsize=(10, 5))
plt.hist(dtw_distances[y_test==0], bins=30, alpha=0.7, 
         color='green', label='Normal')
plt.hist(dtw_distances[y_test==1], bins=30, alpha=0.7, 
         color='red', label='Anomaly')
plt.axvline(threshold, color='black', linestyle='--', 
            label=f'Threshold = {threshold:.2f}')
plt.xlabel('DTW Distance')
plt.ylabel('Count')
plt.title('DTW Distance Distribution - Normal vs Anomaly')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "dtw_distances.png"))
plt.show()
print("Chart saved!")

# ============================================
# 9. RESULTS CSV SAVE KARO
# ============================================
results_path = r"C:\Users\Lenovo\Documents\manit\Menstrual_project\results"
os.makedirs(results_path, exist_ok=True)

results_df = pd.DataFrame([{
    "Model"   : "DTW",
    "F1_Score": round(f1_dtw, 4),
    "AUC_ROC" : round(auc_dtw, 4)
}])

results_df.to_csv(os.path.join(results_path, "kajal_results.csv"), index=False)
print(f"\nResults saved to results/kajal_results.csv")
print("DTW Done! ✅")
