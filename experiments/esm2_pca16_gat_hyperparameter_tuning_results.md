# ESM-2 PCA16 GAT Hyperparameter Tuning Results

This experiment tunes GAT hyperparameters for the combined current + BM5 dataset with ESM-2 PCA16 features.

## Configuration

- Processed directory: `data/processed_combined_current_bm5_esm2_pca16`
- Input dimension: `19`
- Max epochs: `220`
- Patience: `40`
- Eval every: `5` epochs
- Threshold range: `0.05` to `0.6` step `0.01`
- Seed: `42`

## Split

- Train: 1AHW_AB_C, 1JPS_HL_T, 1MLC_AB_E, 1WEJ_HL_F, 2FD6_HL_U, 2VIS_AB_C, 3MJ9_HL_A, BM5_1BJ1_A_B, BM5_1FCC_A_B, BM5_1JWH_A_B, BM5_1ML0_A_B, BM5_1OFU_A_B, BM5_1QFW_A_B, BM5_1RLB_A_B, BM5_1RV6_A_B, BM5_1WDW_A_B, BM5_1XU1_A_B, BM5_2B4J_A_B, BM5_3LVK_A_B, BM5_4FQI_A_B, BM5_4HX3_A_B, BM5_4LW4_A_B
- Validation: 1DQJ_AB_C, 1E6J_HL_P, BM5_1EZU_A_B, BM5_4GXU_A_B
- Test: 1BRS_A_B, 1FSS_A_B, 3HMX_LH_AB, BM5_1A2K_A_B, BM5_3BP8_A_B

## Results

| Hidden | Heads | Dropout | LR | Weight Decay | Best Epoch | Threshold | Val F1 | Test P1 | Test R1 | Test F1 | Test Acc | TN | FP | FN | TP |
|-------:|------:|--------:|---:|-------------:|-----------:|----------:|-------:|--------:|--------:|--------:|---------:|---:|---:|---:|---:|
| 16 | 4 | 0.2 | 0.005 | 0.0005 | 75 | 0.13 | 0.2160 | 0.1249 | 0.4839 | 0.1985 | 0.8786 | 6893 | 841 | 128 | 120 |
| 16 | 4 | 0.3 | 0.005 | 0.0005 | 25 | 0.34 | 0.2265 | 0.1334 | 0.3266 | 0.1895 | 0.9132 | 7208 | 526 | 167 | 81 |
| 16 | 4 | 0.4 | 0.005 | 0.001 | 20 | 0.36 | 0.2210 | 0.1230 | 0.4355 | 0.1918 | 0.8860 | 6964 | 770 | 140 | 108 |
| 8 | 4 | 0.3 | 0.005 | 0.001 | 45 | 0.40 | 0.2068 | 0.1362 | 0.5847 | 0.2209 | 0.8718 | 6814 | 920 | 103 | 145 |
| 8 | 2 | 0.3 | 0.005 | 0.001 | 135 | 0.39 | 0.2241 | 0.1164 | 0.5000 | 0.1889 | 0.8666 | 6793 | 941 | 124 | 124 |
| 16 | 2 | 0.3 | 0.005 | 0.001 | 95 | 0.46 | 0.2401 | 0.1551 | 0.3145 | 0.2077 | 0.9255 | 7309 | 425 | 170 | 78 |
| 32 | 2 | 0.4 | 0.003 | 0.001 | 35 | 0.39 | 0.2085 | 0.1291 | 0.5685 | 0.2104 | 0.8675 | 6783 | 951 | 107 | 141 |
| 32 | 4 | 0.4 | 0.003 | 0.002 | 35 | 0.43 | 0.2194 | 0.1417 | 0.2863 | 0.1896 | 0.9240 | 7304 | 430 | 177 | 71 |

## Best by Validation F1

- hidden=16, heads=2, dropout=0.3, lr=0.005, weight_decay=0.001
- Val F1 1 = `0.2401`
- Test F1 1 = `0.2077`
- Threshold = `0.46`

## Best by Test F1

Reported for analysis only. Model selection should be based on validation F1.

- hidden=8, heads=4, dropout=0.3, lr=0.005, weight_decay=0.001
- Val F1 1 = `0.2068`
- Test F1 1 = `0.2209`
- Threshold = `0.40`
