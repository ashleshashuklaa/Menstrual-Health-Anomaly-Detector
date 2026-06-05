import pandas as pd
import numpy as np

# load
df = pd.read_csv("FedCycleData071012.csv")

# numeric conversion
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

# sort
df = df.sort_values(['ClientID', 'CycleNumber'])

# feature engineering
df['cycle_length'] = df['LengthofCycle']

df['cramp_score'] = (
    df['TotalMensesScore']
    / df['LengthofMenses'].replace(0, np.nan)
)

df['cramp_score'] = df['cramp_score'].fillna(
    df['TotalMensesScore']
)

df['mood_score'] = df['LengthofLutealPhase']

df['flow_intensity'] = df['MeanBleedingIntensity']

client_mean = (
    df.groupby('ClientID')['LengthofCycle']
      .transform('mean')
)

df['sleep_quality'] = (
    df['LengthofCycle'] - client_mean
).abs()

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

print("Before:")
print(df[FEATURE_COLS].isna().sum())

# Forward fill within each client
df[FEATURE_COLS] = (
    df.groupby('ClientID')[FEATURE_COLS]
      .transform(lambda g: g.ffill())
)

print("\nAfter forward fill:")
print(df[FEATURE_COLS].isna().sum())

# Per-client median fill

def client_median_fill(group):
    return group.fillna(group.median())

df[FEATURE_COLS] = (
    df.groupby('ClientID')[FEATURE_COLS]
      .transform(client_median_fill)
)

print("\nAfter client median fill:")
print(df[FEATURE_COLS].isna().sum())

missing_flow_clients = df[df["flow_intensity"].isna()]["ClientID"].nunique()

print("\nClients with no bleeding intensity data at all:")
print(missing_flow_clients)

for f in FEATURE_COLS:
    global_med = df[f].median()
    df[f] = df[f].fillna(global_med)

print("\nAfter global median fill:")
print(df[FEATURE_COLS].isna().sum())

lo, hi = df[f].quantile(0.01), df[f].quantile(0.99)

print("\nOUTLIER CHECK")

for f in FEATURE_COLS:
    lo = df[f].quantile(0.01)
    hi = df[f].quantile(0.99)

    n_clipped = (
        ((df[f] < lo) | (df[f] > hi))
    ).sum()

    print(
        f"{f:20s}",
        f"low={lo:.2f}",
        f"high={hi:.2f}",
        f"would_clip={n_clipped}"
    )

df[f] = df[f].clip(lo, hi)