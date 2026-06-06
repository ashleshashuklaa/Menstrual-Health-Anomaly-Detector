import numpy as np
import matplotlib.pyplot as plt

# Load both original and combined
X_orig = np.load("../Day1/X_windows.npy")
y_orig = np.load("../Day1/y_labels.npy")
X_combined = np.load("X_combined.npy")
y_combined = np.load("y_combined.npy")

# Original anomalies
X_real_anomaly = X_orig[y_orig == 1]
# Synthetic anomalies only
X_synthetic = X_combined[974:]  # everything after original 974

# Compare mean of each feature across timesteps
real_mean = X_real_anomaly.mean(axis=0)      # shape (6, 6)
synth_mean = X_synthetic.mean(axis=0)        # shape (6, 6)

feature_names = ["cycle_length", "cramp_score", "mood_score",
                 "flow_intensity", "sleep_quality", "stress_score"]

print("Feature mean comparison (real anomaly vs synthetic anomaly):")
print(f"{'Feature':<15} {'Real Mean':>12} {'Synthetic Mean':>16} {'Diff':>8}")
print("-" * 55)
for i, name in enumerate(feature_names):
    r = real_mean[:, i].mean()
    s = synth_mean[:, i].mean()
    print(f"{name:<15} {r:>12.4f} {s:>16.4f} {abs(r-s):>8.4f}")

print("\nValidation passed if Diff values are small (< 1.0).")
print(f"\nFinal class balance: {np.bincount(y_combined)}")
print("Ready to hand X_combined.npy and y_combined.npy to Ashlesha.")