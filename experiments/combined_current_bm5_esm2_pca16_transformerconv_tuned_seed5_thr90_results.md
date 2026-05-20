# TransformerConv Experiment Result

## Dataset

- Train: 1AHW_AB_C, 1JPS_HL_T, 1MLC_AB_E, 1WEJ_HL_F, 2FD6_HL_U, 2VIS_AB_C, 3MJ9_HL_A, BM5_1BJ1_A_B, BM5_1FCC_A_B, BM5_1JWH_A_B, BM5_1ML0_A_B, BM5_1OFU_A_B, BM5_1QFW_A_B, BM5_1RLB_A_B, BM5_1RV6_A_B, BM5_1WDW_A_B, BM5_1XU1_A_B, BM5_2B4J_A_B, BM5_3LVK_A_B, BM5_4FQI_A_B, BM5_4HX3_A_B, BM5_4LW4_A_B
- Validation: 1DQJ_AB_C, 1E6J_HL_P, BM5_1EZU_A_B, BM5_4GXU_A_B
- Test: 1BRS_A_B, 1FSS_A_B, 3HMX_LH_AB, BM5_1A2K_A_B, BM5_3BP8_A_B

## Configuration

- Model: `TransformerConv`
- Input dimension: `19`
- Hidden channels: `16`
- Heads: `4`
- Dropout: `0.2`
- Beta gate: `True`
- Learning rate: `0.003`
- Weight decay: `0.001`
- Best epoch: `40`
- Selected threshold: `0.86`

## Validation Metrics

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|------------:|---------:|-----:|---------:|
| 0.5718 | 0.7527 | 0.6499 | 0.9776 |

## Test Metrics

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|------------:|---------:|-----:|---------:|
| 0.4363 | 0.7177 | 0.5427 | 0.9624 |

## Confusion Matrix

| True / Pred | Pred 0 | Pred 1 |
|-------------|-------:|-------:|
| True 0 | 7504 | 230 |
| True 1 | 70 | 178 |

## Comparison Target

```text
Current best: Combined Current + BM5 + Full Pair ESM-2 PCA16 + GAT
P1 = 0.2015
R1 = 0.5323
F1 = 0.2924
Acc = 0.9199
```
