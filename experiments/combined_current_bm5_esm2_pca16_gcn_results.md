# GCN Experiment Result

## Dataset

- Train: 1AHW_AB_C, 1JPS_HL_T, 1MLC_AB_E, 1WEJ_HL_F, 2FD6_HL_U, 2VIS_AB_C, 3MJ9_HL_A, BM5_1BJ1_A_B, BM5_1FCC_A_B, BM5_1JWH_A_B, BM5_1ML0_A_B, BM5_1OFU_A_B, BM5_1QFW_A_B, BM5_1RLB_A_B, BM5_1RV6_A_B, BM5_1WDW_A_B, BM5_1XU1_A_B, BM5_2B4J_A_B, BM5_3LVK_A_B, BM5_4FQI_A_B, BM5_4HX3_A_B, BM5_4LW4_A_B
- Validation: 1DQJ_AB_C, 1E6J_HL_P, BM5_1EZU_A_B, BM5_4GXU_A_B
- Test: 1BRS_A_B, 1FSS_A_B, 3HMX_LH_AB, BM5_1A2K_A_B, BM5_3BP8_A_B

## Configuration

- Model: `GCN`
- Input dimension: `19`
- Hidden channels: `32`
- Dropout: `0.3`
- Learning rate: `0.005`
- Weight decay: `0.0005`
- Best epoch: `95`
- Selected threshold: `0.27`

## Validation Metrics

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|------------:|---------:|-----:|---------:|
| 0.1261 | 0.6582 | 0.2117 | 0.8647 |

## Test Metrics

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|------------:|---------:|-----:|---------:|
| 0.1085 | 0.5968 | 0.1836 | 0.8351 |

## Confusion Matrix

| True / Pred | Pred 0 | Pred 1 |
|-------------|-------:|-------:|
| True 0 | 6518 | 1216 |
| True 1 | 100 | 148 |

## Comparison Target

```text
Previous best GAT: F1 = 0.2924
TransformerConv best single run: F1 = 0.4134
TransformerConv mean over 4 seeds: F1 ≈ 0.3310
```
