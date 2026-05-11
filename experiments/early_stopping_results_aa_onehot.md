# Train/Validation/Test Early Stopping Results

This experiment uses a graph-level train/validation/test split.

Input feature dimension: `43`

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
| GCN | 43 | 11 | 0.40 | 0.1909 | 0.3443 | 0.2456 | 0.1313 | 0.2781 | 0.1783 | 0.9149 |
| GAT | 43 | 34 | 0.40 | 0.1279 | 0.6885 | 0.2157 | 0.1051 | 0.7285 | 0.1836 | 0.7850 |
