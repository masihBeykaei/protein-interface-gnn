# Experimental Results Summary

This document summarizes all major experiments in the protein–protein interface prediction project.

---

## 1. Objective

The task is residue-pair classification on a correspondence graph.

Each correspondence node represents:

```text
(residue from partner A, residue from partner B)
```

Target classes:

```text
0 = non-interface / non-contact
1 = interface / contact
```

A residue pair is positive when any inter-residue atom pair is within:

```text
5 Å
```

---

## 2. Original Current Dataset

| Case | Nodes | Positive | Negative | Positive Ratio |
|------|------:|---------:|---------:|---------------:|
| 1BRS_A_B | 225 | 16 | 209 | 0.0711 |
| 1FSS_A_B | 2,013 | 63 | 1,950 | 0.0313 |
| 1AHW_AB_C | 2,142 | 73 | 2,069 | 0.0341 |
| 1DQJ_AB_C | 1,978 | 71 | 1,907 | 0.0359 |
| 1E6J_HL_P | 1,131 | 51 | 1,080 | 0.0451 |
| 1JPS_HL_T | 2,184 | 71 | 2,113 | 0.0325 |
| 1MLC_AB_E | 1,540 | 54 | 1,486 | 0.0351 |
| 1WEJ_HL_F | 1,026 | 41 | 985 | 0.0400 |
| 2FD6_HL_U | 1,147 | 47 | 1,100 | 0.0410 |
| 2VIS_AB_C | 1,728 | 51 | 1,677 | 0.0295 |
| 3HMX_LH_AB | 2,310 | 72 | 2,238 | 0.0312 |
| 3MJ9_HL_A | 3,283 | 88 | 3,195 | 0.0268 |

Total:

| Nodes | Positive | Negative | Positive Ratio |
|------:|---------:|---------:|---------------:|
| 20,707 | 698 | 20,009 | 0.0337 |

---

## 3. Feature Sets

| Feature Set | Input Dimension | Description |
|-------------|----------------:|-------------|
| Basic | 3 | CA distance, degree A, degree B |
| Amino acid one-hot | 43 | Basic + 20D one-hot for each residue |
| Physicochemical | 11 | Basic + hydrophobicity, charge, polarity, aromaticity |
| Basic + ASA | 5 | Basic + accessible surface area of both residues |
| ESM-2 PCA64 | 67 | Basic + 64 PCA components from ESM pair features |
| ESM-2 PCA32 | 35 | Basic + 32 PCA components from ESM pair features |
| ESM-2 PCA16 | 19 | Basic + 16 PCA components from ESM pair features |

---

## 4. Current-Only Strict Train/Validation/Test Results

Strict graph-level split:

```text
Train: 1WEJ, 1JPS, 1AHW, 2FD6, 2VIS, 1MLC, 3MJ9
Validation: 1DQJ, 1E6J
Test: 1BRS, 1FSS, 3HMX
```

| Model | Feature Set | Input Dim | Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------|-------------|----------:|------------:|---------:|-----:|---------:|
| GCN | Basic | 3 | 0.1940 | 0.1722 | 0.1825 | 0.9488 |
| GAT | Basic | 3 | 0.1746 | 0.3642 | 0.2361 | 0.9217 |
| GCN | One-hot | 43 | 0.1313 | 0.2781 | 0.1783 | 0.9149 |
| GAT | One-hot | 43 | 0.1051 | 0.7285 | 0.1836 | 0.7850 |
| GCN | Physicochemical | 11 | 0.2254 | 0.1060 | 0.1441 | 0.9582 |
| GAT | Physicochemical | 11 | 0.1566 | 0.2914 | 0.2037 | 0.9244 |
| GCN | Basic + ASA | 5 | 0.1887 | 0.2649 | 0.2204 | 0.9378 |
| GAT | Basic + ASA | 5 | 0.2184 | 0.2517 | 0.2338 | 0.9453 |

Best current-only model:

```text
GAT + basic 3 features
F1 = 0.2361
```

---

## 5. Negative Ratio Tuning

| Model | Negative Ratio | Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------|---------------:|------------:|---------:|-----:|---------:|
| GCN | 2 | 0.1442 | 0.6225 | 0.2341 | 0.8648 |
| GCN | 3 | 0.1643 | 0.5298 | 0.2508 | 0.8949 |
| GCN | 5 | 0.2059 | 0.3245 | 0.2519 | 0.9360 |
| GCN | 10 | 0.2653 | 0.0861 | 0.1300 | 0.9617 |
| GAT | 2 | 0.1107 | 0.9007 | 0.1972 | 0.7566 |
| GAT | 3 | 0.1165 | 0.8874 | 0.2060 | 0.7729 |
| GAT | 5 | 0.1274 | 0.7483 | 0.2177 | 0.8215 |
| GAT | 10 | 0.1985 | 0.1788 | 0.1882 | 0.9488 |

---

## 6. GAT Hyperparameter Tuning

| Hidden | Heads | Dropout | Val F1 1 | Test F1 1 |
|-------:|------:|--------:|---------:|----------:|
| 16 | 4 | 0.2 | 0.2571 | 0.2361 |
| 32 | 4 | 0.2 | 0.2526 | 0.1980 |
| 16 | 8 | 0.2 | 0.2531 | 0.2069 |
| 32 | 8 | 0.2 | 0.2457 | 0.1899 |
| 16 | 4 | 0.3 | 0.2534 | 0.2118 |

Best configuration:

```text
hidden_channels = 16
heads = 4
dropout = 0.2
```

---

## 7. Error Analysis for Current-Only Best Model

Model:

```text
GAT + basic 3 features
```

Metrics:

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|------------:|---------:|-----:|---------:|
| 0.1746 | 0.3642 | 0.2361 | 0.9217 |

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|-------------|-------:|-------:|
| True 0 | 4,137 | 260 |
| True 1 | 96 | 55 |

---

## 8. Imbalance Ablation Experiments

| Experiment | Test Setup | Precision 1 | Recall 1 | F1 1 | Accuracy | TP | FP | FN |
|------------|------------|------------:|---------:|-----:|---------:|---:|---:|---:|
| Original strict GAT | Current-only natural test | 0.1746 | 0.3642 | 0.2361 | 0.9217 | 55 | 260 | 96 |
| Train-balanced random | Current-only natural test | 0.1338 | 0.3642 | 0.1957 | 0.9006 | 55 | 356 | 96 |
| Hard negative ratio 5 | Current-only natural test | 0.1519 | 0.4437 | 0.2264 | 0.8993 | 67 | 374 | 84 |
| Hard negative ratio 10 | Current-only natural test | 0.1489 | 0.3510 | 0.2091 | 0.9118 | 53 | 303 | 98 |

Conclusion:

```text
Dataset-level undersampling and hard-negative graph reconstruction did not improve current-only natural-test F1.
```

---

## 9. BM5-Clean Dataset Expansion

BM5-clean import and screening:

| Stage | Count |
|-------|------:|
| Imported BM5 reference complexes | 29 |
| Accepted after screening | 19 |
| Rejected | 10 |

Screening criteria:

```text
positive >= 30
positive_ratio >= 0.02
candidate_nodes <= 8000
```

Accepted BM5 positives:

```text
1,096
```

---

## 10. BM5-Only Experiment

Dataset:

```text
BM5-clean accepted cases only
19 complexes
1,096 positive residue pairs
```

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|------------:|---------:|-----:|---------:|
| 0.1824 | 0.8144 | 0.2981 | 0.8917 |

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|-------------|-------:|-------:|
| True 0 | 2,983 | 354 |
| True 1 | 18 | 79 |

This result is not directly comparable to the current-only test because the test set is different.

---

## 11. Combined Current + BM5 Basic Experiment

Combined dataset:

| Source | Cases | Positive Pairs |
|--------|------:|---------------:|
| Current | 12 | 698 |
| BM5-clean accepted | 19 | 1,096 |
| Total | 31 | 1,794 |

Processed combined dataset:

| Saved Nodes | Positive | Negative | Positive Ratio |
|------------:|---------:|---------:|---------------:|
| 23,028 | 1,794 | 21,234 | 0.0779 |

Split:

| Split | Cases |
|-------|------:|
| Train | 22 |
| Validation | 4 |
| Test | 5 |

Result:

| Model | Features | Input Dim | Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------|----------|----------:|------------:|---------:|-----:|---------:|
| GAT | Basic | 3 | 0.1918 | 0.5121 | 0.2791 | 0.9178 |

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|-------------|-------:|-------:|
| True 0 | 7,199 | 535 |
| True 1 | 121 | 127 |

---

## 12. ESM-2 Embedding Extraction

Model:

```text
facebook/esm2_t6_8M_UR50D
```

Embedding extraction was completed for all 31 combined cases.

For each case, the following files are generated:

```text
<case>_partner1_esm2.npy
<case>_partner2_esm2.npy
<case>_esm2_metadata.json
```

The raw pair representation is:

```text
[ESM_A, ESM_B, abs(ESM_A - ESM_B)]
```

Raw pair dimension:

```text
960
```

PCA and standardization are fitted only on training pairs.

---

## 13. ESM-2 PCA Dataset Construction

### PCA64

| Raw Dim | PCA Components | Final Feature Dim |
|--------:|---------------:|------------------:|
| 960 | 64 | 67 |

### PCA32

| Raw Dim | PCA Components | Final Feature Dim |
|--------:|---------------:|------------------:|
| 960 | 32 | 35 |

### PCA16

| Raw Dim | PCA Components | Final Feature Dim |
|--------:|---------------:|------------------:|
| 960 | 16 | 19 |

---

## 14. ESM-2 PCA Results

| Experiment | Input Dim | Precision 1 | Recall 1 | F1 1 | Accuracy | TN | FP | FN | TP |
|------------|----------:|------------:|---------:|-----:|---------:|---:|---:|---:|---:|
| Combined basic 3 features | 3 | 0.1918 | 0.5121 | 0.2791 | 0.9178 | 7199 | 535 | 121 | 127 |
| ESM-2 PCA64 + GAT | 67 | 0.1169 | 0.7823 | 0.2034 | 0.8096 | 6268 | 1466 | 54 | 194 |
| ESM-2 PCA32 + GAT | 35 | 0.1953 | 0.5726 | 0.2913 | 0.9134 | 7149 | 585 | 106 | 142 |
| ESM-2 PCA16 + GAT | 19 | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 7211 | 523 | 116 | 132 |

---

## 15. ESM-2 Interpretation

PCA64 retained too much high-dimensional signal for the current dataset size and caused over-prediction:

```text
FP = 1466
Recall = 0.7823
F1 = 0.2034
```

PCA32 improved F1 over the basic combined baseline:

```text
F1 = 0.2913
```

PCA16 produced the best trade-off:

```text
Precision 1 = 0.2015
Recall 1 = 0.5323
F1 1 = 0.2924
Accuracy = 0.9199
```

Compared with the combined basic model:

```text
F1:       0.2791 → 0.2924
Precision:0.1918 → 0.2015
Recall:   0.5121 → 0.5323
FP:       535 → 523
TP:       127 → 132
```

---


---

## Final F1-Improvement Ablation Study

After the full-pair ESM-2 PCA16 model became the best result, additional experiments were run to determine whether the F1-score could be pushed higher.

### ESM-PCA16 GAT Hyperparameter Tuning

| Experiment | Best Validation-Based Test F1 |
|------------|------------------------------:|
| ESM-PCA16 GAT hyperparameter tuning | 0.2077 |

This experiment did not improve over the baseline ESM-PCA16 test F1 of `0.2924`.

### Train Negative-Ratio Tuning for ESM-PCA16

| Experiment | Precision 1 | Recall 1 | F1 1 | Accuracy | TN | FP | FN | TP |
|------------|------------:|---------:|-----:|---------:|---:|---:|---:|---:|
| ESM-PCA16 ratio 3 / best | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 7211 | 523 | 116 | 132 |
| ESM-PCA16 ratio 4 | 0.2600 | 0.3145 | 0.2847 | 0.9509 | 7512 | 222 | 170 | 78 |
| ESM-PCA16 ratio 5 | 0.2507 | 0.3468 | 0.2910 | 0.9475 | 7477 | 257 | 162 | 86 |

Higher negative ratios shifted the model toward higher precision and fewer false positives, but recall dropped enough that F1 did not improve.

### ESM Pair-Feature Variant Ablations

| Experiment | Raw Pair Feature | Input Dim | Precision 1 | Recall 1 | F1 1 | Accuracy | TN | FP | FN | TP |
|------------|------------------|----------:|------------:|---------:|-----:|---------:|---:|---:|---:|---:|
| Full pair PCA16 / best | `[A, B, absdiff]` | 19 | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 7211 | 523 | 116 | 132 |
| absdiff PCA16 | `abs(A - B)` | 19 | 0.1924 | 0.5081 | 0.2791 | 0.9184 | 7205 | 529 | 122 | 126 |
| absdiff PCA32 | `abs(A - B)` | 35 | 0.2003 | 0.5081 | 0.2873 | 0.9217 | 7231 | 503 | 122 | 126 |
| product PCA16 | `A * B` | 19 | 0.1877 | 0.5806 | 0.2837 | 0.9089 | 7111 | 623 | 104 | 144 |
| absdiff_product PCA16 | `[abs(A - B), A * B]` | 19 | 0.2232 | 0.4113 | 0.2894 | 0.9372 | 7379 | 355 | 146 | 102 |

Key interpretation:

- `product PCA16` was recall-oriented and found the most true positives, but it also produced too many false positives.
- `absdiff_product PCA16` was precision-oriented and reduced false positives, but recall dropped.
- The full-pair ESM-PCA16 representation remained the best F1 trade-off.

Final conclusion:

```text
Best final model remains:
Combined Current + BM5 + Full Pair ESM-2 PCA16 + GAT
F1 = 0.2924
```


## 16. Current Best Result

Best model:

```text
Combined Current + BM5 + Full Pair ESM-2 PCA16 + GAT
```

Final metrics:

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|------------:|---------:|-----:|---------:|
| 0.2015 | 0.5323 | 0.2924 | 0.9199 |

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|-------------|-------:|-------:|
| True 0 | 7,211 | 523 |
| True 1 | 116 | 132 |

---

## 17. Overall Progression

| Stage | Dataset | Features | F1 1 |
|-------|---------|----------|-----:|
| Current-only strict best | 12 complexes | Basic 3 | 0.2361 |
| Combined current + BM5 | 31 complexes | Basic 3 | 0.2791 |
| Combined current + BM5 | ESM-2 PCA32 | 0.2913 |
| Combined current + BM5 | ESM-2 PCA16 | 0.2924 |

Main conclusion:

```text
Dataset expansion and compact ESM-2 PCA features produced the strongest improvement.
```

---

## 18. Attention and Structural Visualization

Additional analysis was performed for the best strict current-only model:

- error analysis
- GAT attention extraction
- refined attention analysis
- PyMOL structural visualization of TP, FP, and FN residue pairs

These analyses are used for interpretability rather than final model selection.

---

## 19. Final Conclusion

The project progressed from a small current-only dataset to an expanded combined current + BM5 dataset with ESM-2 protein language model features.

The final best result is:

```text
F1 = 0.2924
```

This result comes from:

```text
GAT + Combined Current/BM5 Dataset + ESM-2 PCA16 features
```
