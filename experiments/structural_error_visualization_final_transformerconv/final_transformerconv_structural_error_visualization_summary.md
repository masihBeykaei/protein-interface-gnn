# Final Tuned TransformerConv Structural Error Visualization

## Model Setup

```text
Model: Tuned TransformerConv
Processed dir: data/processed_combined_current_bm5_esm2_pca16
hidden_channels: 16
heads: 4
dropout: 0.2
lr: 0.003
weight_decay: 0.001
seed: 1
best_epoch: 135
threshold_max: 0.9
```

## Global Test Metrics

| Threshold | P1 | R1 | F1 | Acc | TN | FP | FN | TP |
|----------:|---:|---:|---:|----:|---:|---:|---:|---:|
| 0.83 | 0.5342 | 0.6613 | 0.5910 | 0.9716 | 7591 | 143 | 84 | 164 |

## Per-Case Test Summary

| Case | Nodes | Positives | Negatives | P1 | R1 | F1 | Acc | TN | FP | FN | TP | PML |
|------|------:|----------:|----------:|---:|---:|---:|----:|---:|---:|---:|---:|-----|
| 1BRS_A_B | 225 | 16 | 209 | 0.5789 | 0.6875 | 0.6286 | 0.9422 | 201 | 8 | 5 | 11 | `1BRS_A_B_final_transformerconv_structural_errors.pml` |
| 1FSS_A_B | 2013 | 63 | 1950 | 0.5811 | 0.6825 | 0.6277 | 0.9747 | 1919 | 31 | 20 | 43 | `1FSS_A_B_final_transformerconv_structural_errors.pml` |
| 3HMX_LH_AB | 2310 | 72 | 2238 | 0.4886 | 0.5972 | 0.5375 | 0.9680 | 2193 | 45 | 29 | 43 | `3HMX_LH_AB_final_transformerconv_structural_errors.pml` |
| BM5_1A2K_A_B | 1610 | 45 | 1565 | 0.5745 | 0.6000 | 0.5870 | 0.9764 | 1545 | 20 | 18 | 27 | `BM5_1A2K_A_B_final_transformerconv_structural_errors.pml` |
| BM5_3BP8_A_B | 1824 | 52 | 1772 | 0.5063 | 0.7692 | 0.6107 | 0.9720 | 1733 | 39 | 12 | 40 | `BM5_3BP8_A_B_final_transformerconv_structural_errors.pml` |

## Color Legend

| Color | Meaning |
|-------|---------|
| Green | True Positive residue-pair examples |
| Red | False Positive residue-pair examples |
| Orange | False Negative residue-pair examples |
| Gray | Full protein complex cartoon |

## Interpretation

These PyMOL files visualize selected TP, FP, and FN residue-pair examples from the final tuned TransformerConv model. They are intended for qualitative structural analysis and presentation. The selected examples are not necessarily all model predictions; by default, the script exports the top-k examples for each class of prediction outcome.
