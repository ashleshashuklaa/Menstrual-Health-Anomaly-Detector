import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import csv
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from sklearn.metrics import f1_score, precision_score, recall_score

# --------------------------------------------------
# Config
# --------------------------------------------------

FEATURE_NAMES = [
    "cycle_length",
    "cramp_score",
    "mood_score",
    "flow_intensity",
    "sleep_quality",
    "stress_score",
]

EPOCHS     = 50
BATCH_SIZE = 32

# --------------------------------------------------
# Load data
# --------------------------------------------------

X_train = np.load("../Day2/X_train.npy").astype(np.float32)
y_train = np.load("../Day2/y_train.npy")
X_test  = np.load("../Day2/X_test.npy").astype(np.float32)
y_test  = np.load("../Day2/y_test.npy")

print(f"Loaded  X_train {X_train.shape}  X_test {X_test.shape}")

# --------------------------------------------------
# Helper: build autoencoder for n_features
# --------------------------------------------------

def build_autoencoder(n_features):
    inp = tf.keras.Input(shape=(6, n_features))
    # Encoder
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inp)
    x = tf.keras.layers.LSTM(32)(x)
    # Bottleneck
    x = tf.keras.layers.Dense(16)(x)
    # Decoder
    x = tf.keras.layers.RepeatVector(6)(x)
    x = tf.keras.layers.LSTM(32, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model

# --------------------------------------------------
# Helper: train + evaluate one feature set
# --------------------------------------------------

def run_experiment(X_tr, y_tr, X_te, y_te, label):
    n_features = X_tr.shape[2]

    # Train on normal windows only
    X_tr_normal = np.array(X_tr[y_tr == 0], copy=True)

    model = build_autoencoder(n_features)

    model.fit(
        X_tr_normal, X_tr_normal,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    # Compute reconstruction error on training normals to set threshold
    recon_train = model.predict(X_tr_normal, verbose=0)
    errors_train = np.mean(np.square(X_tr_normal - recon_train), axis=(1, 2))
    threshold = np.mean(errors_train) + 2 * np.std(errors_train)

    # Evaluate on test set
    X_te_copy = np.array(X_te, copy=True)
    recon_test = model.predict(X_te_copy, verbose=0)
    errors_test = np.mean(np.square(X_te_copy - recon_test), axis=(1, 2))
    y_pred = (errors_test > threshold).astype(int)

    f1        = f1_score(y_te, y_pred, zero_division=0)
    precision = precision_score(y_te, y_pred, zero_division=0)
    recall    = recall_score(y_te, y_pred, zero_division=0)
    threshold_val = float(threshold)

    print(f"  {label:<25}  F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}  threshold={threshold_val:.6f}")

    return {
        "experiment": label,
        "f1":         round(f1, 4),
        "precision":  round(precision, 4),
        "recall":     round(recall, 4),
        "threshold":  round(threshold_val, 6),
        "n_features": n_features,
    }

# --------------------------------------------------
# Run baseline (all 6 features)
# --------------------------------------------------

print("\n=== Ablation Study ===\n")
results = []

print("Running baseline (all features)...")
baseline = run_experiment(X_train, y_train, X_test, y_test, "baseline_all_features")
results.append(baseline)
baseline_f1 = baseline["f1"]

# --------------------------------------------------
# Remove one feature at a time
# --------------------------------------------------

for i, feat_name in enumerate(FEATURE_NAMES):
    print(f"Removing feature {i}: {feat_name} ...")
    X_tr_reduced = np.delete(X_train, i, axis=2)
    X_te_reduced = np.delete(X_test,  i, axis=2)
    label = f"without_{feat_name}"
    row = run_experiment(X_tr_reduced, y_train, X_te_reduced, y_test, label)
    row["f1_drop"] = round(baseline_f1 - row["f1"], 4)
    results.append(row)

# fill baseline f1_drop
results[0]["f1_drop"] = 0.0

# --------------------------------------------------
# Print summary table
# --------------------------------------------------

print("\n=== Summary ===")
print(f"{'Experiment':<30} {'F1':>6} {'F1 Drop':>8} {'Precision':>10} {'Recall':>8}")
print("-" * 68)
for r in results:
    print(f"{r['experiment']:<30} {r['f1']:>6.4f} {r.get('f1_drop', 0):>8.4f} {r['precision']:>10.4f} {r['recall']:>8.4f}")

# --------------------------------------------------
# Save to CSV
# --------------------------------------------------

out_path = "ablation_results.csv"
fieldnames = ["experiment", "n_features", "f1", "f1_drop", "precision", "recall", "threshold"]

with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to {out_path}")