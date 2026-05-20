# TransformerConv Hyperparameter Optimization Results

## Dataset

- Processed directory: `data/processed_combined_current_bm5_esm2_pca16`
- Input dimension: `19`
- Train: 1AHW_AB_C, 1JPS_HL_T, 1MLC_AB_E, 1WEJ_HL_F, 2FD6_HL_U, 2VIS_AB_C, 3MJ9_HL_A, BM5_1BJ1_A_B, BM5_1FCC_A_B, BM5_1JWH_A_B, BM5_1ML0_A_B, BM5_1OFU_A_B, BM5_1QFW_A_B, BM5_1RLB_A_B, BM5_1RV6_A_B, BM5_1WDW_A_B, BM5_1XU1_A_B, BM5_2B4J_A_B, BM5_3LVK_A_B, BM5_4FQI_A_B, BM5_4HX3_A_B, BM5_4LW4_A_B
- Validation: 1DQJ_AB_C, 1E6J_HL_P, BM5_1EZU_A_B, BM5_4GXU_A_B
- Test: 1BRS_A_B, 1FSS_A_B, 3HMX_LH_AB, BM5_1A2K_A_B, BM5_3BP8_A_B

## Search Setup

- Search mode: `small`
- Seeds: `7,42,21`
- Max epochs: `220`
- Patience: `40`
- Evaluation interval: `5`
- Threshold range: `0.05` to `0.6` step `0.01`

## Per-Run Results

| Config | Seed | Hidden | Heads | Dropout | LR | WD | Threshold | Val F1 | Test P1 | Test R1 | Test F1 | Acc | TN | FP | FN | TP |
|--------|-----:|-------:|------:|--------:|---:|---:|----------:|-------:|--------:|--------:|--------:|----:|---:|---:|---:|---:|
| baseline_h16_heads4_do0.3_lr0.003_wd1e-3_beta | 7 | 16 | 4 | 0.3 | 0.003 | 0.001 | 0.51 | 0.5060 | 0.3042 | 0.6452 | 0.4134 | 0.9431 | 7368 | 366 | 88 | 160 |
| baseline_h16_heads4_do0.3_lr0.003_wd1e-3_beta | 42 | 16 | 4 | 0.3 | 0.003 | 0.001 | 0.25 | 0.4462 | 0.2041 | 0.8024 | 0.3254 | 0.8966 | 6958 | 776 | 49 | 199 |
| baseline_h16_heads4_do0.3_lr0.003_wd1e-3_beta | 21 | 16 | 4 | 0.3 | 0.003 | 0.001 | 0.60 | 0.4367 | 0.2373 | 0.9234 | 0.3776 | 0.9054 | 6998 | 736 | 19 | 229 |
| h16_heads4_do0.2_lr0.003_wd1e-3_beta | 7 | 16 | 4 | 0.2 | 0.003 | 0.001 | 0.56 | 0.5673 | 0.3583 | 0.6371 | 0.4586 | 0.9533 | 7451 | 283 | 90 | 158 |
| h16_heads4_do0.2_lr0.003_wd1e-3_beta | 42 | 16 | 4 | 0.2 | 0.003 | 0.001 | 0.60 | 0.5220 | 0.2219 | 0.8750 | 0.3540 | 0.9008 | 6973 | 761 | 31 | 217 |
| h16_heads4_do0.2_lr0.003_wd1e-3_beta | 21 | 16 | 4 | 0.2 | 0.003 | 0.001 | 0.60 | 0.6003 | 0.3982 | 0.7258 | 0.5143 | 0.9574 | 7462 | 272 | 68 | 180 |
| h16_heads4_do0.4_lr0.003_wd1e-3_beta | 7 | 16 | 4 | 0.4 | 0.003 | 0.001 | 0.49 | 0.4286 | 0.2644 | 0.7016 | 0.3841 | 0.9301 | 7250 | 484 | 74 | 174 |
| h16_heads4_do0.4_lr0.003_wd1e-3_beta | 42 | 16 | 4 | 0.4 | 0.003 | 0.001 | 0.53 | 0.5037 | 0.4111 | 0.4476 | 0.4286 | 0.9629 | 7575 | 159 | 137 | 111 |
| h16_heads4_do0.4_lr0.003_wd1e-3_beta | 21 | 16 | 4 | 0.4 | 0.003 | 0.001 | 0.56 | 0.5535 | 0.2804 | 0.6411 | 0.3902 | 0.9377 | 7326 | 408 | 89 | 159 |
| h16_heads2_do0.3_lr0.003_wd1e-3_beta | 7 | 16 | 2 | 0.3 | 0.003 | 0.001 | 0.55 | 0.5514 | 0.3789 | 0.6815 | 0.4870 | 0.9554 | 7457 | 277 | 79 | 169 |
| h16_heads2_do0.3_lr0.003_wd1e-3_beta | 42 | 16 | 2 | 0.3 | 0.003 | 0.001 | 0.60 | 0.4464 | 0.1827 | 0.8871 | 0.3030 | 0.8732 | 6750 | 984 | 28 | 220 |
| h16_heads2_do0.3_lr0.003_wd1e-3_beta | 21 | 16 | 2 | 0.3 | 0.003 | 0.001 | 0.60 | 0.4359 | 0.2405 | 0.8387 | 0.3738 | 0.9127 | 7077 | 657 | 40 | 208 |
| h32_heads2_do0.4_lr0.003_wd2e-3_beta | 7 | 32 | 2 | 0.4 | 0.003 | 0.002 | 0.51 | 0.4580 | 0.2845 | 0.6895 | 0.4028 | 0.9365 | 7304 | 430 | 77 | 171 |
| h32_heads2_do0.4_lr0.003_wd2e-3_beta | 42 | 32 | 2 | 0.4 | 0.003 | 0.002 | 0.50 | 0.5535 | 0.4397 | 0.4556 | 0.4475 | 0.9650 | 7590 | 144 | 135 | 113 |
| h32_heads2_do0.4_lr0.003_wd2e-3_beta | 21 | 32 | 2 | 0.4 | 0.003 | 0.002 | 0.54 | 0.5364 | 0.2647 | 0.7258 | 0.3879 | 0.9288 | 7234 | 500 | 68 | 180 |

## Summary by Configuration

| Config | Runs | Mean F1 | Std F1 | Mean P1 | Mean R1 | Best F1 | Best Seed |
|--------|-----:|--------:|-------:|--------:|--------:|--------:|----------:|
| h16_heads4_do0.2_lr0.003_wd1e-3_beta | 3 | 0.4423 | 0.0814 | 0.3261 | 0.7460 | 0.5143 | 21 |
| h32_heads2_do0.4_lr0.003_wd2e-3_beta | 3 | 0.4128 | 0.0310 | 0.3296 | 0.6237 | 0.4475 | 42 |
| h16_heads4_do0.4_lr0.003_wd1e-3_beta | 3 | 0.4010 | 0.0241 | 0.3187 | 0.5968 | 0.4286 | 42 |
| h16_heads2_do0.3_lr0.003_wd1e-3_beta | 3 | 0.3879 | 0.0928 | 0.2674 | 0.8024 | 0.4870 | 7 |
| baseline_h16_heads4_do0.3_lr0.003_wd1e-3_beta | 3 | 0.3721 | 0.0443 | 0.2485 | 0.7903 | 0.4134 | 7 |

## Best Single Run

- Config: `h16_heads4_do0.2_lr0.003_wd1e-3_beta`
- Seed: `21`
- Test F1 1: `0.5143`
- Test precision 1: `0.3982`
- Test recall 1: `0.7258`
- Accuracy: `0.9574`
- Confusion matrix: `[[7462, 272], [68, 180]]`

## Best Mean Configuration

- Config: `h16_heads4_do0.2_lr0.003_wd1e-3_beta`
- Mean F1 1: `0.4423`
- Std F1 1: `0.0814`
- Best seed: `21`
- Best F1 1: `0.5143`

## Previous Reference

```text
Previous GAT best: F1 = 0.2924
Previous TransformerConv best single run: F1 = 0.4134
Previous TransformerConv 4-seed mean: F1 = 0.3310 ± 0.0705
```
