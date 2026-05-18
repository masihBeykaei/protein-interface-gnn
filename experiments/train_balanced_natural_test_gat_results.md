# Expanded Balanced GAT Results

Input feature dimension: `3`

## Splits

- Train: 1AHW_AB_C, 1JPS_HL_T, 1MLC_AB_E, 1WEJ_HL_F, 2FD6_HL_U, 2VIS_AB_C, 3MJ9_HL_A
- Validation: 1DQJ_AB_C, 1E6J_HL_P
- Test: 1BRS_A_B, 1FSS_A_B, 3HMX_LH_AB

## Training Selection

| Best Epoch | Best Threshold | Best Val F1 1 |
|------------|----------------|---------------|
| 4 | 0.30 | 0.2513 |

## Test Metrics

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------------|----------|------|----------|
| 0.1338 | 0.3642 | 0.1957 | 0.9006 |

## Confusion Matrix

| True / Pred | Pred 0 | Pred 1 |
|-------------|--------|--------|
| True 0 | 4041 | 356 |
| True 1 | 96 | 55 |

Previous strict baseline: `GAT + basic 3 features`, Test F1 1 = `0.2361`.
