# Expanded Balanced GAT Results

Input feature dimension: `67`

## Splits

- Train: 1AHW_AB_C, 1JPS_HL_T, 1MLC_AB_E, 1WEJ_HL_F, 2FD6_HL_U, 2VIS_AB_C, 3MJ9_HL_A, BM5_1BJ1_A_B, BM5_1FCC_A_B, BM5_1JWH_A_B, BM5_1ML0_A_B, BM5_1OFU_A_B, BM5_1QFW_A_B, BM5_1RLB_A_B, BM5_1RV6_A_B, BM5_1WDW_A_B, BM5_1XU1_A_B, BM5_2B4J_A_B, BM5_3LVK_A_B, BM5_4FQI_A_B, BM5_4HX3_A_B, BM5_4LW4_A_B
- Validation: 1DQJ_AB_C, 1E6J_HL_P, BM5_1EZU_A_B, BM5_4GXU_A_B
- Test: 1BRS_A_B, 1FSS_A_B, 3HMX_LH_AB, BM5_1A2K_A_B, BM5_3BP8_A_B

## Training Selection

| Best Epoch | Best Threshold | Best Val F1 1 |
|------------|----------------|---------------|
| 39 | 0.10 | 0.2574 |

## Test Metrics

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------------|----------|------|----------|
| 0.1169 | 0.7823 | 0.2034 | 0.8096 |

## Confusion Matrix

| True / Pred | Pred 0 | Pred 1 |
|-------------|--------|--------|
| True 0 | 6268 | 1466 |
| True 1 | 54 | 194 |

Previous strict baseline: `GAT + basic 3 features`, Test F1 1 = `0.2361`.
