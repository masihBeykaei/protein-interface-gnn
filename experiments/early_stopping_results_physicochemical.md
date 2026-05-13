# Train/Validation/Test Early Stopping Results

This experiment uses a graph-level train/validation/test split.

Input feature dimension: `11`

The validation set is used for:

- early stopping
- selecting the probability threshold for class 1

The test set is used only for final evaluation.

## Split

### Train Graphs

- 1WEJ_HL_F
- 1JPS_HL_T
- 1AHW_AB_C
- 2FD6_HL_U
- 2VIS_AB_C
- 1MLC_AB_E
- 3MJ9_HL_A

### Validation Graphs

- 1DQJ_AB_C
- 1E6J_HL_P

### Test Graphs

- 1BRS_A_B
- 1FSS_A_B
- 3HMX_LH_AB

## Results

| Model | Input Dim | Best Epoch | Best Threshold | Val Precision 1 | Val Recall 1 | Val F1 1 | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|-------|-----------|------------|----------------|-----------------|--------------|----------|------------------|---------------|-----------|---------------|
| GCN | 11 | 24 | 0.60 | 0.2804 | 0.2459 | 0.2620 | 0.2254 | 0.1060 | 0.1441 | 0.9582 |
| GAT | 11 | 22 | 0.50 | 0.1548 | 0.6066 | 0.2467 | 0.1566 | 0.2914 | 0.2037 | 0.9244 |
