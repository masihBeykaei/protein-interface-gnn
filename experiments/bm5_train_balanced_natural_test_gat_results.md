# Expanded Balanced GAT Results

Input feature dimension: `3`

## Splits

- Train: BM5_1BJ1_A_B, BM5_1FCC_A_B, BM5_1JWH_A_B, BM5_1ML0_A_B, BM5_1OFU_A_B, BM5_1QFW_A_B, BM5_1RLB_A_B, BM5_1RV6_A_B, BM5_1WDW_A_B, BM5_1XU1_A_B, BM5_2B4J_A_B, BM5_3LVK_A_B, BM5_4FQI_A_B, BM5_4HX3_A_B, BM5_4LW4_A_B
- Validation: BM5_1EZU_A_B, BM5_4GXU_A_B
- Test: BM5_1A2K_A_B, BM5_3BP8_A_B

## Training Selection

| Best Epoch | Best Threshold | Best Val F1 1 |
|------------|----------------|---------------|
| 131 | 0.15 | 0.3021 |

## Test Metrics

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------------|----------|------|----------|
| 0.1824 | 0.8144 | 0.2981 | 0.8917 |

## Confusion Matrix

| True / Pred | Pred 0 | Pred 1 |
|-------------|--------|--------|
| True 0 | 2983 | 354 |
| True 1 | 18 | 79 |

Previous strict baseline: `GAT + basic 3 features`, Test F1 1 = `0.2361`.
