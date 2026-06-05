import pandas as pd
import numpy as np

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("FedCycleData071012.csv")

# =====================================================
# NUMERIC CONVERSION
# =====================================================

numeric_cols = [
    'LengthofCycle',
    'LengthofLutealPhase',
    'LengthofMenses',
    'TotalMensesScore',
    'MeanBleedingIntensity',
    'EstimatedDayofOvulation'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# =====================================================
# SORT
# =====================================================

df = df.sort_values(
    ['ClientID', 'CycleNumber']
).reset_index(drop=True)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

# 1. Cycle Length
df['cycle_length'] = df['LengthofCycle']

# 2. Cramp Score
df['cramp_score'] = (
    df['TotalMensesScore']
    / df['LengthofMenses'].replace(0, np.nan)
)

df['cramp_score'] = df['cramp_score'].fillna(
    df['TotalMensesScore']
)

# 3. Mood Score (proxy)
df['mood_score'] = df['LengthofLutealPhase']

# 4. Flow Intensity
df['flow_intensity'] = df['MeanBleedingIntensity']

# 5. Sleep Quality (proxy)
client_mean = (
    df.groupby('ClientID')['LengthofCycle']
      .transform('mean')
)

df['sleep_quality'] = (
    df['LengthofCycle'] - client_mean
).abs()

# 6. Stress Score (proxy)
df['stress_score'] = (
    df.groupby('ClientID')['EstimatedDayofOvulation']
      .transform(
          lambda x: x.rolling(
              3,
              min_periods=1
          ).std()
      )
)

FEATURE_COLS = [
    'cycle_length',
    'cramp_score',
    'mood_score',
    'flow_intensity',
    'sleep_quality',
    'stress_score'
]

# =====================================================
# IMPUTATION
# =====================================================

# Forward fill within client
df[FEATURE_COLS] = (
    df.groupby('ClientID')[FEATURE_COLS]
      .transform(lambda g: g.ffill())
)

# Client median fill
def client_median_fill(group):
    return group.fillna(group.median())

df[FEATURE_COLS] = (
    df.groupby('ClientID')[FEATURE_COLS]
      .transform(client_median_fill)
)

# Global median fill
for f in FEATURE_COLS:
    df[f] = df[f].fillna(df[f].median())

print("\nMissing values after imputation:")
print(df[FEATURE_COLS].isna().sum())

# =====================================================
# OUTLIER CLIPPING
# =====================================================

for f in FEATURE_COLS:
    lo = df[f].quantile(0.01)
    hi = df[f].quantile(0.99)

    df[f] = df[f].clip(lo, hi)

# =====================================================
# LABEL CREATION
# =====================================================

df = df[
    df['ReproductiveCategory'] != 9
].copy()

df['anomaly'] = (
    df['ReproductiveCategory'] > 0
).astype(int)

print("\nCycle-level labels:")
print(df['anomaly'].value_counts())

# =====================================================
# MIN-MAX NORMALIZATION
# =====================================================

for f in FEATURE_COLS:

    fmin = df[f].min()
    fmax = df[f].max()

    df[f + "_norm"] = (
        (df[f] - fmin)
        / (fmax - fmin + 1e-8)
    )

NORM_COLS = [
    f + "_norm"
    for f in FEATURE_COLS
]

# =====================================================
# SLIDING WINDOWS
# =====================================================

WINDOW_SIZE = 6

X_list = []
y_list = []
cid_list = []

for client_id, group in df.groupby("ClientID"):

    group = group.sort_values(
        "CycleNumber"
    )

    X_vals = group[NORM_COLS].values
    y_vals = group["anomaly"].values

    if len(X_vals) < WINDOW_SIZE:
        continue

    for start in range(
        len(X_vals) - WINDOW_SIZE + 1
    ):

        window_X = X_vals[
            start:start + WINDOW_SIZE
        ]

        window_y = y_vals[
            start + WINDOW_SIZE - 1
        ]

        X_list.append(window_X)
        y_list.append(window_y)
        cid_list.append(client_id)

# =====================================================
# CONVERT TO ARRAYS
# =====================================================

X = np.array(X_list)
y = np.array(y_list)
cids = np.array(cid_list)

# =====================================================
# SUMMARY
# =====================================================

print("\n" + "="*50)
print("WINDOW SUMMARY")
print("="*50)

print("X shape:", X.shape)
print("y shape:", y.shape)

print(
    "Clients with windows:",
    len(np.unique(cids))
)

print(
    "Anomaly windows:",
    int(y.sum())
)

print(
    f"Anomaly rate: {y.mean()*100:.2f}%"
)

print(
    "Normal windows:",
    int((y == 0).sum())
)

print(
    "Anomaly windows:",
    int((y == 1).sum())
)

anom_clients = np.unique(
    cids[y == 1]
)

print("Women with anomaly windows:")
print(len(anom_clients))

print(anom_clients)

print(np.unique(cids[y == 1]))
print(len(np.unique(cids[y == 1])))

anom_counts = {}

for cid in np.unique(cids[y == 1]):
    anom_counts[cid] = int(
        np.sum(
            (cids == cid) & (y == 1)
        )
    )

print(anom_counts)

print("\nFINAL CHECK")
print("="*50)
print(f"Windows shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Unique clients: {len(np.unique(cids))}")
print(f"Normal windows: {(y==0).sum()}")
print(f"Anomaly windows: {(y==1).sum()}")
print(f"Anomaly rate: {y.mean()*100:.2f}%")
print(f"Anomaly clients: {len(np.unique(cids[y==1]))}")
print("="*50)