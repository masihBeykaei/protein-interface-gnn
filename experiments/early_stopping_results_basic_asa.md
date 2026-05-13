# Train/Validation/Test Early Stopping Results

This experiment uses a graph-level train/validation/test split.

Input feature dimension: `5`

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
| GCN | 5 | 35 | 0.60 | 0.2251 | 0.5000 | 0.3104 | 0.1887 | 0.2649 | 0.2204 | 0.9378 |
| GAT | 5 | 191 | 0.50 | 0.3050 | 0.3525 | 0.3270 | 0.2184 | 0.2517 | 0.2338 | 0.9453 |
