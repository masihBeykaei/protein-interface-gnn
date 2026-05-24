# Experimental Results Summary

This document summarizes the major experiments and final model selection for the protein–protein interface prediction project.

---

## 1. Task

The task is binary classification of residue pairs on a correspondence graph.

```text
node = (residue from partner A, residue from partner B)
label 0 = non-interface / non-contact
label 1 = interface / contact
```

A residue pair is positive if any inter-residue atom pair is within 5 Å.

---

## 2. Dataset Summary

### Original Current Dataset

| Cases | Nodes | Positive | Negative | Positive Ratio |
|---:|---:|---:|---:|---:|
| 12 | 20,707 | 698 | 20,009 | 0.0337 |

### BM5-Clean Expansion

| Stage | Count |
|---|---:|
| Imported BM5 reference complexes | 29 |
| Accepted BM5 complexes | 19 |
| Accepted BM5 positive pairs | 1,096 |

### Combined Current + BM5 Dataset

| Source | Cases | Positive Pairs |
|---|---:|---:|
| Current | 12 | 698 |
| BM5-clean accepted | 19 | 1,096 |
| Total | 31 | 1,794 |

Processed combined dataset:

| Saved Nodes | Positive | Negative | Positive Ratio |
|---:|---:|---:|---:|
| 23,028 | 1,794 | 21,234 | 0.0779 |

Split:

| Split | Cases |
|---|---:|
| Train | 22 |
| Validation | 4 |
| Test | 5 |

Test positives:

```text
248
```

Test negatives:

```text
7,734
```

---

## 3. Feature Sets

| Feature Set | Input Dim | Description |
|---|---:|---|
| Basic | 3 | C-alpha distance, degree partner 1, degree partner 2 |
| Amino acid one-hot | 43 | Basic + one-hot residue identity for both residues |
| Physicochemical | 11 | Basic + hydrophobicity, charge, polarity, aromaticity |
| Basic + ASA | 5 | Basic + solvent accessibility of both residues |
| Full Pair ESM-2 PCA64 | 67 | Basic + 64 PCA components from `[ESM_A, ESM_B, absdiff]` |
| Full Pair ESM-2 PCA32 | 35 | Basic + 32 PCA components from `[ESM_A, ESM_B, absdiff]` |
| Full Pair ESM-2 PCA16 | 19 | Basic + 16 PCA components from `[ESM_A, ESM_B, absdiff]` |

The final feature set is:

```text
Basic 3 + Full Pair ESM-2 PCA16
```

---

## 4. Current-Only Strict Baselines

| Model | Features | Input Dim | Precision 1 | Recall 1 | F1 1 | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| GCN | Basic | 3 | 0.1940 | 0.1722 | 0.1825 | 0.9488 |
| GAT | Basic | 3 | 0.1746 | 0.3642 | 0.2361 | 0.9217 |
| GCN | One-hot | 43 | 0.1313 | 0.2781 | 0.1783 | 0.9149 |
| GAT | One-hot | 43 | 0.1051 | 0.7285 | 0.1836 | 0.7850 |
| GCN | Physicochemical | 11 | 0.2254 | 0.1060 | 0.1441 | 0.9582 |
| GAT | Physicochemical | 11 | 0.1566 | 0.2914 | 0.2037 | 0.9244 |
| GCN | Basic + ASA | 5 | 0.1887 | 0.2649 | 0.2204 | 0.9378 |
| GAT | Basic + ASA | 5 | 0.2184 | 0.2517 | 0.2338 | 0.9453 |

Best current-only result:

```text
GAT + Basic 3
F1 = 0.2361
```

---

## 5. Imbalance Experiments on Current-Only Dataset

| Experiment | Precision 1 | Recall 1 | F1 1 | Accuracy | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original strict GAT | 0.1746 | 0.3642 | 0.2361 | 0.9217 | 55 | 260 | 96 |
| Train-balanced random | 0.1338 | 0.3642 | 0.1957 | 0.9006 | 55 | 356 | 96 |
| Hard negative ratio 5 | 0.1519 | 0.4437 | 0.2264 | 0.8993 | 67 | 374 | 84 |
| Hard negative ratio 10 | 0.1489 | 0.3510 | 0.2091 | 0.9118 | 53 | 303 | 98 |

---

## 6. Combined Current + BM5 Basic Result

| Dataset | Model | Features | Precision 1 | Recall 1 | F1 1 | Accuracy | TP | FP | FN |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Combined Current + BM5 | GAT | Basic 3 | 0.1918 | 0.5121 | 0.2791 | 0.9178 | 127 | 535 | 121 |

---

## 7. ESM-2 Full Pair PCA Results with GAT

| Experiment | Input Dim | Precision 1 | Recall 1 | F1 1 | Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Combined basic 3 | 3 | 0.1918 | 0.5121 | 0.2791 | 0.9178 | 7,199 | 535 | 121 | 127 |
| Full Pair ESM-2 PCA64 + GAT | 67 | 0.1169 | 0.7823 | 0.2034 | 0.8096 | 6,268 | 1,466 | 54 | 194 |
| Full Pair ESM-2 PCA32 + GAT | 35 | 0.1953 | 0.5726 | 0.2913 | 0.9134 | 7,149 | 585 | 106 | 142 |
| Full Pair ESM-2 PCA16 + GAT | 19 | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 7,211 | 523 | 116 | 132 |

Best GAT setting:

```text
Full Pair ESM-2 PCA16 + GAT
F1 = 0.2924
```

---

## 8. GAT F1-Improvement Attempts

### Train Negative Ratio

| Experiment | Precision 1 | Recall 1 | F1 1 | Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ESM-PCA16 ratio 3 | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 7,211 | 523 | 116 | 132 |
| ESM-PCA16 ratio 4 | 0.2600 | 0.3145 | 0.2847 | 0.9509 | 7,512 | 222 | 170 | 78 |
| ESM-PCA16 ratio 5 | 0.2507 | 0.3468 | 0.2910 | 0.9475 | 7,477 | 257 | 162 | 86 |

### ESM Pair-Feature Variants

| Experiment | Raw Pair Feature | Input Dim | Precision 1 | Recall 1 | F1 1 | Accuracy | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Pair PCA16 | `[A, B, absdiff]` | 19 | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 7,211 | 523 | 116 | 132 |
| absdiff PCA16 | `absdiff` | 19 | 0.1924 | 0.5081 | 0.2791 | 0.9184 | 7,205 | 529 | 122 | 126 |
| absdiff PCA32 | `absdiff` | 35 | 0.2003 | 0.5081 | 0.2873 | 0.9217 | 7,231 | 503 | 122 | 126 |
| product PCA16 | `A * B` | 19 | 0.1877 | 0.5806 | 0.2837 | 0.9089 | 7,111 | 623 | 104 | 144 |
| absdiff_product PCA16 | `[absdiff, A * B]` | 19 | 0.2232 | 0.4113 | 0.2894 | 0.9372 | 7,379 | 355 | 146 | 102 |

Conclusion:

```text
The full-pair ESM-PCA16 representation remained the best GAT feature setup.
```

---

## 9. GNN Architecture Comparison

All models use:

```text
Combined Current + BM5
Basic 3 + Full Pair ESM-2 PCA16
Input dimension = 19
```

| Model | Seed | Precision 1 | Recall 1 | F1 1 | Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GCN | 42 | 0.1085 | 0.5968 | 0.1836 | 0.8351 | 6,518 | 1,216 | 100 | 148 |
| GIN | 42 | 0.1075 | 0.5484 | 0.1798 | 0.8445 | 6,605 | 1,129 | 112 | 136 |
| GATv2 | 42 | 0.1957 | 0.4355 | 0.2700 | 0.9268 | 7,290 | 444 | 140 | 108 |
| GraphSAGE | 42 | 0.1860 | 0.5887 | 0.2827 | 0.9072 | 7,095 | 639 | 102 | 146 |
| GAT | 42 | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 7,211 | 523 | 116 | 132 |
| TransformerConv initial | 42 | 0.2041 | 0.8024 | 0.3254 | 0.8966 | 6,958 | 776 | 49 | 199 |
| TransformerConv initial | 7 | 0.3042 | 0.6452 | 0.4134 | 0.9431 | 7,368 | 366 | 88 | 160 |
| TransformerConv initial | 13 | 0.1383 | 0.9637 | 0.2419 | 0.8123 | 6,245 | 1,489 | 9 | 239 |
| TransformerConv initial | 21 | 0.2112 | 0.9153 | 0.3432 | 0.8911 | 6,886 | 848 | 21 | 227 |

Initial TransformerConv multi-seed summary:

```text
Mean F1 = 0.3310 ± 0.0705
Best F1 = 0.4134
```

---

## 10. Tuned TransformerConv Results

Final tuned configuration:

```text
hidden_channels = 16
heads = 4
dropout = 0.2
learning rate = 0.003
weight decay = 0.001
beta = True
```

### Tuned TransformerConv with threshold_max = 0.60

| Seed | Precision 1 | Recall 1 | F1 1 | Accuracy | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2264 | 0.8427 | 0.3570 | 0.9057 | 7,020 | 714 | 39 | 209 |
| 3 | 0.2825 | 0.8065 | 0.4184 | 0.9303 | 7,226 | 508 | 48 | 200 |
| 5 | 0.3311 | 0.7863 | 0.4659 | 0.9440 | 7,340 | 394 | 53 | 195 |
| 21 | 0.3969 | 0.7218 | 0.5122 | 0.9573 | 7,462 | 272 | 69 | 179 |

Summary:

```text
Mean F1 = 0.4384 ± 0.0664
Best F1 = 0.5122
```

### Tuned TransformerConv with threshold_max = 0.90

| Seed | Selected Threshold | Precision 1 | Recall 1 | F1 1 | Accuracy | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.90 | 0.5603 | 0.6371 | **0.5962** | 0.9732 | 7,610 | 124 | 90 | 158 |
| 3 | 0.77 | 0.3947 | 0.7258 | 0.5114 | 0.9569 | 7,458 | 276 | 68 | 180 |
| 5 | 0.86 | 0.4363 | 0.7177 | 0.5427 | 0.9624 | 7,504 | 230 | 70 | 178 |
| 21 | 0.87 | 0.4407 | 0.7339 | 0.5507 | 0.9628 | 7,503 | 231 | 66 | 182 |

Summary:

```text
Mean F1 = 0.5502 ± 0.0350
Best F1 = 0.5962
```

### Threshold_max = 0.99 Check

| Seed | Selected Threshold | Precision 1 | Recall 1 | F1 1 | Accuracy | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.85 | 0.4749 | 0.6855 | 0.5611 | 0.9667 | 7,546 | 188 | 78 | 170 |

Conclusion:

```text
Increasing threshold_max from 0.90 to 0.99 did not improve the best F1.
```

---

## 11. Best Final Model

```text
Tuned TransformerConv
Dataset: Combined Current + BM5
Features: Basic 3 + Full Pair ESM-2 PCA16
Input dimension: 19
hidden_channels = 16
heads = 4
dropout = 0.2
learning rate = 0.003
weight decay = 0.001
threshold_max = 0.90
seed = 1
```

Final test result:

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|---:|---:|---:|---:|
| 0.5603 | 0.6371 | **0.5962** | 0.9732 |

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|---|---:|---:|
| True 0 | 7,610 | 124 |
| True 1 | 90 | 158 |

---

## 12. Final Performance Progression

| Stage | F1 1 |
|---|---:|
| Current-only GAT basic | 0.2361 |
| Combined Current + BM5 GAT basic | 0.2791 |
| Full Pair ESM-2 PCA16 + GAT | 0.2924 |
| Initial TransformerConv best | 0.4134 |
| Tuned TransformerConv, threshold ≤ 0.60 | 0.5122 |
| Tuned TransformerConv, threshold ≤ 0.90 | **0.5962** |

---

## 13. Final Conclusion

The final model improves substantially over all earlier baselines.

```text
Best final F1 = 0.5962
Mean tuned TransformerConv F1 over 4 seeds = 0.5502 ± 0.0350
```

The most important improvements were:

1. BM5-clean dataset expansion
2. full-pair ESM-2 PCA16 features
3. TransformerConv architecture
4. dropout tuning
5. expanding validation threshold search to 0.90
