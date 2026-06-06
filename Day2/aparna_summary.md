# Aparna Day 2 — Work Summary

## What was done
- Loaded X_windows.npy (974 samples, 26 anomalies) from Ashlesha's Day1 pipeline
- Trained CTGAN on 26 real anomaly windows
- Generated 200 synthetic anomaly windows
- Validated synthetic data (all feature diffs < 0.1)
- Combined into balanced dataset: 1174 samples [948 normal, 226 anomaly]
- Created train/test split (80/20, stratified)

## Files produced (all in Day2/)
| File | Shape | Description |
|---|---|---|
| X_combined.npy | (1174, 6, 6) | Full balanced dataset |
| y_combined.npy | (1174,) | Labels |
| X_train.npy | (939, 6, 6) | Training set |
| X_test.npy | (235, 6, 6) | Test set |
| y_train.npy | (939,) | Train labels [758 normal, 181 anomaly] |
| y_test.npy | (235,) | Test labels [190 normal, 45 anomaly] |

## Hand these to teammates 3, 4, 5
X_train.npy, X_test.npy, y_train.npy, y_test.npy