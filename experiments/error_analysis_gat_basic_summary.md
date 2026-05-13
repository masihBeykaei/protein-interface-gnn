# Error Analysis: GAT with Basic 3 Features

This analysis uses the current best strict protocol model:

```text
Model: GAT
Features: basic 3 features
Input dimension: 3
Split: train/validation/test
Threshold selected on validation set
```

## Training Selection

- Best epoch: `56`
- Best validation threshold: `0.50`
- Best validation F1 for class 1: `0.2571`

## Test Metrics

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------------|----------|------|----------|
| 0.1746 | 0.3642 | 0.2361 | 0.9217 |

## Confusion Matrix on Test Set

| True / Pred | Pred 0 | Pred 1 |
|-------------|--------|--------|
| True 0 | 4137 | 260 |
| True 1 | 96 | 55 |

Definitions:

- FP: model predicted interface/contact, but the true label is non-contact
- FN: model missed a true interface/contact pair

## Per-Test-Graph Error Summary

| Case | Nodes | Positive | Negative | TP | TN | FP | FN |
|------|-------|----------|----------|----|----|----|----|
| 1BRS_A_B | 225 | 16 | 209 | 11 | 139 | 70 | 5 |
| 1FSS_A_B | 2013 | 63 | 1950 | 19 | 1862 | 88 | 44 |
| 3HMX_LH_AB | 2310 | 72 | 2238 | 25 | 2136 | 102 | 47 |

## Top False Positives by Predicted Probability

| Case | Node ID | Residue A | Residue B | P(class 1) | CA Distance | Degree A | Degree B |
|------|---------|-----------|-----------|------------|-------------|----------|----------|
| 3HMX_LH_AB | 2131 | 315 | 58 | 0.6730 | -1.5782 | -2.0079 | -1.2972 |
| 3HMX_LH_AB | 2132 | 315 | 59 | 0.6709 | -1.4135 | -2.0079 | -0.5890 |
| 3HMX_LH_AB | 2130 | 315 | 57 | 0.6668 | -1.6825 | -2.0079 | 0.1191 |
| 3HMX_LH_AB | 2217 | 317 | 60 | 0.6435 | -1.5165 | -0.5898 | -1.2972 |
| 1BRS_A_B | 177 | 104 | 107 | 0.6392 | -1.2181 | -0.9443 | -0.5890 |
| 1BRS_A_B | 173 | 104 | 96 | 0.6381 | -0.9545 | -0.9443 | 0.8273 |
| 3HMX_LH_AB | 2214 | 317 | 57 | 0.6379 | -1.6993 | -0.5898 | 0.1191 |
| 3HMX_LH_AB | 2135 | 315 | 62 | 0.6377 | -1.6961 | -2.0079 | 0.8273 |
| 3HMX_LH_AB | 2215 | 317 | 58 | 0.6348 | -1.5650 | -0.5898 | -1.2972 |
| 3HMX_LH_AB | 2088 | 314 | 57 | 0.6337 | -1.4438 | -0.9443 | 0.1191 |

## Top False Negatives by Lowest Predicted Probability

| Case | Node ID | Residue A | Residue B | P(class 1) | CA Distance | Degree A | Degree B |
|------|---------|-----------|-----------|------------|-------------|----------|----------|
| 3HMX_LH_AB | 1680 | 270 | 14 | 0.0368 | -1.0302 | -1.2989 | -0.2349 |
| 1FSS_A_B | 153 | 66 | 32 | 0.0389 | -1.2758 | -0.9443 | -0.5890 |
| 3HMX_LH_AB | 829 | 93 | 58 | 0.0418 | -1.1076 | -0.9443 | -1.2972 |
| 3HMX_LH_AB | 1282 | 245 | 46 | 0.0583 | -1.0487 | 0.8284 | -0.5890 |
| 1FSS_A_B | 284 | 70 | 31 | 0.0661 | -1.4546 | -1.6534 | -0.5890 |
| 3HMX_LH_AB | 358 | 48 | 46 | 0.0706 | -1.1686 | 1.1830 | -0.5890 |
| 1FSS_A_B | 285 | 70 | 32 | 0.0938 | -1.8393 | -1.6534 | -0.5890 |
| 3HMX_LH_AB | 359 | 48 | 47 | 0.1101 | -1.1936 | 1.1830 | -0.5890 |
| 3HMX_LH_AB | 910 | 95 | 55 | 0.1160 | -0.7612 | 0.8284 | -0.2349 |
| 1FSS_A_B | 154 | 66 | 33 | 0.1197 | -1.3071 | -0.9443 | 0.1191 |

## Output Files

- Full error table: `experiments\error_analysis_gat_basic.csv`
- Summary: `experiments\error_analysis_gat_basic_summary.md`
