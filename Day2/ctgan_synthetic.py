import numpy as np
import pandas as pd
from ctgan import CTGAN

# ── 1. Load Ashlesha's window data ──────────────────────────────
X = np.load("../Day1/X_windows.npy")   # shape (974, 6, 6)
y = np.load("../Day1/y_labels.npy")    # [948 normal, 26 anomaly]

print(f"Loaded: X={X.shape}, y={y.shape}")
print(f"Class counts: {np.bincount(y)}")

# ── 2. Extract only the anomaly windows ─────────────────────────
X_anomaly = X[y == 1]                  # shape (26, 6, 6)
print(f"Anomaly windows to learn from: {X_anomaly.shape}")

# ── 3. Flatten windows into a table for CTGAN ───────────────────
# CTGAN works on 2D tables, not 3D arrays
# Each window (6 timesteps x 6 features) becomes 1 row of 36 columns
n_samples, timesteps, features = X_anomaly.shape
X_flat = X_anomaly.reshape(n_samples, timesteps * features)

columns = [f"t{t}_f{f}" for t in range(timesteps) for f in range(features)]
df_anomaly = pd.DataFrame(X_flat, columns=columns)
print(f"Table for CTGAN: {df_anomaly.shape}")

# ── 4. Train CTGAN on anomaly samples ───────────────────────────
print("\nTraining CTGAN... (takes 1-2 minutes)")
ctgan = CTGAN(epochs=300, verbose=True)
ctgan.fit(df_anomaly)
print("CTGAN training complete.")

# ── 5. Generate 200 synthetic anomaly windows ───────────────────
N_SYNTHETIC = 200
df_synthetic = ctgan.sample(N_SYNTHETIC)
print(f"Generated synthetic samples: {df_synthetic.shape}")

# ── 6. Reshape back to (N, 6, 6) ────────────────────────────────
X_synthetic = df_synthetic.values.reshape(N_SYNTHETIC, timesteps, features)
y_synthetic = np.ones(N_SYNTHETIC, dtype=int)   # all labelled as anomaly

# ── 7. Combine with original data ───────────────────────────────
X_combined = np.concatenate([X, X_synthetic], axis=0)
y_combined = np.concatenate([y, y_synthetic], axis=0)

print(f"\nFinal combined dataset:")
print(f"  X_combined shape: {X_combined.shape}")
print(f"  y_combined shape: {y_combined.shape}")
print(f"  Class counts: {np.bincount(y_combined)}")

# ── 8. Save for Ashlesha's model pipeline ───────────────────────
np.save("X_combined.npy", X_combined)
np.save("y_combined.npy", y_combined)
print("\nSaved: X_combined.npy and y_combined.npy in Day2/")
print("Done! Hand these two files to Ashlesha.")