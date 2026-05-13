# Structural 3D Error Visualization Summary

This experiment generates PyMOL scripts to visualize important TP, FP, and FN residue pairs in 3D structure.

```text
Model: GAT
Feature set: basic 3 features
Input dimension: 3
Visualization target: top TP / FP / FN residue pairs per test complex
```

## Training Selection

- Best epoch: `56`
- Best validation threshold: `0.50`
- Best validation F1 for class 1: `0.2571`

## Test Metrics

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------------|----------|------|----------|
| 0.1746 | 0.3642 | 0.2361 | 0.9217 |

## Confusion Matrix

| True / Pred | Pred 0 | Pred 1 |
|-------------|--------|--------|
| True 0 | 4137 | 260 |
| True 1 | 96 | 55 |

## Per-Test-Graph Available Error Counts

| Case | TP | FP | FN |
|------|----|----|----|
| 1BRS_A_B | 11 | 70 | 5 |
| 1FSS_A_B | 19 | 88 | 44 |
| 3HMX_LH_AB | 25 | 102 | 47 |

## Selected Residue Pairs for Visualization

Top `10` pairs per class were selected when available.

| Case | TP Selected | FP Selected | FN Selected |
|------|-------------|-------------|-------------|
| 1BRS_A_B | 10 | 10 | 5 |
| 1FSS_A_B | 10 | 10 | 10 |
| 3HMX_LH_AB | 10 | 10 | 10 |

## Output PyMOL Scripts

| Case | PyMOL Script |
|------|--------------|
| 1BRS_A_B | `experiments\structural_error_visualization\1BRS_A_B_structural_errors.pml` |
| 1FSS_A_B | `experiments\structural_error_visualization\1FSS_A_B_structural_errors.pml` |
| 3HMX_LH_AB | `experiments\structural_error_visualization\3HMX_LH_AB_structural_errors.pml` |

## Color Legend

| Color | Meaning |
|-------|---------|
| Green | True Positive residue pairs |
| Red | False Positive residue pairs |
| Orange | False Negative residue pairs |

## How to Open in PyMOL

Example:

```bash
pymol experiments/structural_error_visualization/1BRS_A_B_structural_errors.pml
```

## Notes

- TP and FP pairs are ranked by highest predicted probability for class 1.
- FN pairs are ranked by lowest predicted probability for class 1, because these are the most strongly missed contacts.
- The visualization highlights residues and draws CA-to-CA distance objects for selected pairs.
- This is intended for qualitative structural interpretation, not as an additional quantitative metric.

## Output Tables

- Selected residue pairs: `experiments\structural_error_visualization\structural_error_visualization_pairs.csv`
- Summary: `experiments\structural_error_visualization\structural_error_visualization_summary.md`
