import pandas as pd
import numpy as np

df = pd.read_csv("FedCycleData071012.csv")

# Convert numeric columns
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

# Sort
df = df.sort_values(['ClientID', 'CycleNumber']).reset_index(drop=True)

# Feature engineering

df['cycle_length'] = df['LengthofCycle']

df['cramp_score'] = (
    df['TotalMensesScore'] /
    df['LengthofMenses'].replace(0, np.nan)
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

print(df[FEATURE_COLS].describe())

print("\nMissing values:")
print(df[FEATURE_COLS].isna().sum())

print(df["MeanBleedingIntensity"].value_counts(dropna=False).head(20))

print(df["MeanBleedingIntensity"].value_counts(dropna=False))

print(df["LengthofLutealPhase"].value_counts(dropna=False).head(10))

for col in [
    'LengthofCycle',
    'LengthofLutealPhase',
    'TotalMensesScore',
    'MeanBleedingIntensity',
    'EstimatedDayofOvulation'
]:
    print("\n" + "="*50)
    print(col)
    print(df[col].describe())