import pandas as pd

df = pd.read_csv("FedCycleData071012.csv")

FEATURE_COLS = [
    'cycle_length',
    'cramp_score',
    'mood_score',
    'flow_intensity',
    'sleep_quality',
    'stress_score'
]

print(df[FEATURE_COLS].describe())

for feature in FEATURE_COLS:
    print("\n" + "="*50)
    print(feature)

    print("Normal:")
    print(df[df["anomaly"] == 0][feature].describe())

    print("\nAnomaly:")
    print(df[df["anomaly"] == 1][feature].describe())

print(df[FEATURE_COLS].isna().sum())