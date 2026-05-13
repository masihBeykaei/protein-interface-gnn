# GAT Hyperparameter Tuning Results

This experiment tunes GAT hyperparameters using the strict graph-level train/validation/test protocol.

```text
Model: GAT
Feature set: basic 3 features
Input dimension: 3
Negative ratio: 5
Threshold selection: validation set
Early stopping: validation positive-class F1
```

## Search Space

| hidden_channels | heads | dropout |
|-----------------|-------|---------|
| 16 | 4 | 0.2 |
| 32 | 4 | 0.2 |
| 16 | 8 | 0.2 |
| 32 | 8 | 0.2 |
| 16 | 4 | 0.3 |

## Results

| Hidden | Heads | Dropout | Best Epoch | Threshold | Val P1 | Val R1 | Val F1 | Val Acc | Test P1 | Test R1 | Test F1 | Test Acc |
|--------|-------|---------|------------|-----------|--------|--------|--------|---------|---------|---------|---------|----------|
| 16 | 4 | 0.2 | 56 | 0.50 | 0.1589 | 0.6721 | 0.2571 | 0.8475 | 0.1746 | 0.3642 | 0.2361 | 0.9217 |
| 32 | 4 | 0.2 | 26 | 0.50 | 0.1607 | 0.5902 | 0.2526 | 0.8630 | 0.1424 | 0.3245 | 0.1980 | 0.9127 |
| 16 | 8 | 0.2 | 42 | 0.50 | 0.1611 | 0.5902 | 0.2531 | 0.8633 | 0.1491 | 0.3377 | 0.2069 | 0.9140 |
| 32 | 8 | 0.2 | 6 | 0.40 | 0.1557 | 0.5820 | 0.2457 | 0.8598 | 0.1366 | 0.3113 | 0.1899 | 0.9118 |
| 16 | 4 | 0.3 | 20 | 0.50 | 0.1602 | 0.6066 | 0.2534 | 0.8598 | 0.1642 | 0.2980 | 0.2118 | 0.9263 |

## Best Configuration by Validation F1

| Hidden | Heads | Dropout | Best Epoch | Threshold | Val F1 | Test F1 |
|--------|-------|---------|------------|-----------|--------|---------|
| 16 | 4 | 0.2 | 56 | 0.50 | 0.2571 | 0.2361 |

## Best Configuration by Test F1

This is reported for analysis only. Model selection should be based on validation performance.

| Hidden | Heads | Dropout | Best Epoch | Threshold | Val F1 | Test F1 |
|--------|-------|---------|------------|-----------|--------|---------|
| 16 | 4 | 0.2 | 56 | 0.50 | 0.2571 | 0.2361 |

## Interpretation Notes

- The validation set is used for early stopping and threshold selection.
- The test set is used only for final evaluation.
- The current baseline configuration is `hidden_channels=16`, `heads=4`, `dropout=0.2`.
- If a larger configuration improves validation F1 but not test F1, it may indicate overfitting or limited data size.

## Output Files

- CSV results: `experiments\gat_hyperparameter_tuning_results.csv`
- Markdown summary: `experiments\gat_hyperparameter_tuning_results.md`
