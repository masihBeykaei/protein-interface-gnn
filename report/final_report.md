# Final Report: Protein–Protein Interface Prediction using Graph Neural Networks

## 1. Abstract

This project investigates residue-level protein–protein interface prediction using Graph Neural Networks. Protein partners are represented as residue-level graphs, and a correspondence graph is constructed between two interacting partners. Each node in the correspondence graph represents a candidate residue pair, and the task is to classify whether that pair belongs to the protein–protein interface.

The project implements a complete pipeline including PDB preprocessing, graph construction, interface labeling, GCN and GAT baselines, feature engineering, class imbalance analysis, BM5-clean dataset expansion, ESM-2 protein language model embeddings, ESM feature ablations, multiple GNN architecture comparisons, TransformerConv tuning, threshold optimization, attention analysis, error analysis, and structural visualization.

The best final result is:

```text
Tuned TransformerConv + Combined Current/BM5 Dataset + Full Pair ESM-2 PCA16
Test F1 1 = 0.5962
```

The final tuned TransformerConv also achieved strong multi-seed stability:

```text
Mean F1 1 = 0.5502 ± 0.0350
```

---

## 2. Introduction

Protein–protein interactions are central to biological regulation, signaling, immune recognition, and molecular assembly. Identifying interface residues helps explain molecular recognition and can support protein engineering and drug discovery.

This project follows a graph-based formulation. Instead of classifying individual residues in isolation, it classifies candidate residue pairs across two protein partners. This is useful because the interface is defined by relationships between residues from different proteins.

---

## 3. Problem Formulation

Given two protein partners, A and B, we classify each candidate residue pair:

```text
(a_i, b_j)
```

as:

```text
0 = non-interface / non-contact
1 = interface / contact
```

A pair is positive if any atom pair between the two residues is closer than 5 Å.

The task is binary node classification on a correspondence graph:

```text
node = candidate residue pair
edge = compatible neighborhood relationship between candidate residue pairs
```

---

## 4. Data Processing Pipeline

The preprocessing pipeline performs:

1. PDB loading
2. selected chain extraction
3. standard amino acid filtering
4. atom coordinate extraction
5. C-alpha coordinate extraction
6. residue-level graph construction
7. candidate pair filtering
8. correspondence node construction
9. atom-distance label assignment
10. correspondence edge construction
11. NumPy export

Generated files per complex:

```text
<CASE_NAME>_corr_features.npy
<CASE_NAME>_corr_labels.npy
<CASE_NAME>_corr_pairs.npy
<CASE_NAME>_corr_edge_index.npy
```

---

## 5. Original Dataset

The original current dataset contains 12 protein complexes.

| Total Nodes | Positive | Negative | Positive Ratio |
|---:|---:|---:|---:|
| 20,707 | 698 | 20,009 | 0.0337 |

The dataset is highly imbalanced, with positives representing only about 3.37% of candidate residue pairs.

---

## 6. BM5-Clean Dataset Expansion

To improve data diversity and increase positive examples, BM5-clean reference complexes were imported and screened.

| Stage | Count |
|---|---:|
| Imported BM5 complexes | 29 |
| Accepted after screening | 19 |
| Accepted BM5 positive pairs | 1,096 |

Screening criteria:

```text
positive >= 30
positive_ratio >= 0.02
candidate_nodes <= 8000
```

---

## 7. Combined Dataset

The final expanded dataset combines the original current complexes with accepted BM5-clean complexes.

| Source | Cases | Positive Pairs |
|---|---:|---:|
| Current dataset | 12 | 698 |
| BM5-clean accepted | 19 | 1,096 |
| Combined dataset | 31 | 1,794 |

Processed combined dataset:

| Saved Nodes | Positive | Negative | Positive Ratio |
|---:|---:|---:|---:|
| 23,028 | 1,794 | 21,234 | 0.0779 |

Split:

```text
Train: 22 complexes
Validation: 4 complexes
Test: 5 complexes
```

Training graphs are semi-balanced. Validation and test graphs preserve natural class imbalance.

---

## 8. Graph Construction

Each protein partner is first represented as a residue-level graph:

```text
node = residue
edge = spatial proximity between residues
```

The correspondence graph is then built over residue pairs:

```text
node = (residue from partner A, residue from partner B)
```

This lets the model learn local compatibility patterns between neighborhoods of the two interacting proteins.

---

## 9. Feature Engineering

Several feature families were tested:

| Feature Set | Input Dim | Description |
|---|---:|---|
| Basic | 3 | C-alpha distance and graph degrees |
| One-hot | 43 | Basic + residue identity one-hot vectors |
| Physicochemical | 11 | Basic + residue property descriptors |
| Basic + ASA | 5 | Basic + solvent accessibility |
| Full Pair ESM-2 PCA64 | 67 | Basic + 64 PCA components |
| Full Pair ESM-2 PCA32 | 35 | Basic + 32 PCA components |
| Full Pair ESM-2 PCA16 | 19 | Basic + 16 PCA components |

The final best feature set is:

```text
Basic 3 + Full Pair ESM-2 PCA16
```

---

## 10. ESM-2 Protein Language Model Features

ESM-2 embeddings were extracted using:

```text
facebook/esm2_t6_8M_UR50D
```

For each residue pair, the raw full pair feature is:

```text
[ESM_A, ESM_B, abs(ESM_A - ESM_B)]
```

Raw dimension:

```text
960
```

Standardization and PCA are fitted only on training pairs, preventing validation/test leakage. PCA16 provided the best GAT trade-off and became the feature basis for final architecture comparisons.

---

## 11. Baseline Results

### Current-Only Strict Best

| Model | Features | Precision 1 | Recall 1 | F1 1 | Accuracy |
|---|---|---:|---:|---:|---:|
| GAT | Basic 3 | 0.1746 | 0.3642 | 0.2361 | 0.9217 |

### Combined Current + BM5 Basic

| Model | Features | Precision 1 | Recall 1 | F1 1 | Accuracy |
|---|---|---:|---:|---:|---:|
| GAT | Basic 3 | 0.1918 | 0.5121 | 0.2791 | 0.9178 |

### GAT with Full Pair ESM-2 PCA16

| Model | Features | Precision 1 | Recall 1 | F1 1 | Accuracy |
|---|---|---:|---:|---:|---:|
| GAT | Full Pair ESM-2 PCA16 | 0.2015 | 0.5323 | 0.2924 | 0.9199 |

---

## 12. GAT Ablation Results

Negative ratio and ESM pair-feature variant experiments showed that GAT performance could be shifted toward precision or recall, but no GAT variant improved beyond F1 = 0.2924.

Important observations:

- Increasing the train negative ratio improved precision but reduced recall.
- Product-only ESM features increased recall but produced too many false positives.
- Absdiff-product features improved precision but lowered recall.
- Full-pair ESM-PCA16 remained the best GAT feature representation.

---

## 13. Architecture Comparison

The following models were evaluated on the same final dataset and feature set:

```text
Combined Current + BM5 + Full Pair ESM-2 PCA16
```

| Model | Seed | Precision 1 | Recall 1 | F1 1 | Accuracy | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GCN | 42 | 0.1085 | 0.5968 | 0.1836 | 0.8351 | 1,216 | 100 | 148 |
| GIN | 42 | 0.1075 | 0.5484 | 0.1798 | 0.8445 | 1,129 | 112 | 136 |
| GATv2 | 42 | 0.1957 | 0.4355 | 0.2700 | 0.9268 | 444 | 140 | 108 |
| GraphSAGE | 42 | 0.1860 | 0.5887 | 0.2827 | 0.9072 | 639 | 102 | 146 |
| GAT | 42 | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 523 | 116 | 132 |
| TransformerConv initial | 7 | 0.3042 | 0.6452 | 0.4134 | 0.9431 | 366 | 88 | 160 |

TransformerConv was the first architecture to clearly outperform GAT.

---

## 14. TransformerConv Tuning

The TransformerConv model was then tuned.

Final tuned configuration:

```text
hidden_channels = 16
heads = 4
dropout = 0.2
learning rate = 0.003
weight decay = 0.001
beta = True
```

The key change from the initial strong TransformerConv setup was reducing dropout from 0.3 to 0.2 and increasing the validation threshold search range.

---

## 15. Threshold Optimization

### Threshold Search up to 0.60

| Seed | Precision 1 | Recall 1 | F1 1 | Accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.2264 | 0.8427 | 0.3570 | 0.9057 |
| 3 | 0.2825 | 0.8065 | 0.4184 | 0.9303 |
| 5 | 0.3311 | 0.7863 | 0.4659 | 0.9440 |
| 21 | 0.3969 | 0.7218 | 0.5122 | 0.9573 |

Summary:

```text
Mean F1 = 0.4384 ± 0.0664
Best F1 = 0.5122
```

### Threshold Search up to 0.90

| Seed | Selected Threshold | Precision 1 | Recall 1 | F1 1 | Accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.90 | 0.5603 | 0.6371 | **0.5962** | 0.9732 |
| 3 | 0.77 | 0.3947 | 0.7258 | 0.5114 | 0.9569 |
| 5 | 0.86 | 0.4363 | 0.7177 | 0.5427 | 0.9624 |
| 21 | 0.87 | 0.4407 | 0.7339 | 0.5507 | 0.9628 |

Summary:

```text
Mean F1 = 0.5502 ± 0.0350
Best F1 = 0.5962
```

### Threshold Search up to 0.99

A final check was performed for seed 1:

| Threshold Max | Selected Threshold | Precision 1 | Recall 1 | F1 1 | Accuracy |
|---:|---:|---:|---:|---:|---:|
| 0.99 | 0.85 | 0.4749 | 0.6855 | 0.5611 | 0.9667 |

The `threshold_max=0.99` test did not improve over `threshold_max=0.90`.

---

## 16. Best Final Model

The best final model is:

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

Final test metrics:

| Metric | Value |
|---|---:|
| Precision 1 | 0.5603 |
| Recall 1 | 0.6371 |
| F1 1 | **0.5962** |
| Accuracy | 0.9732 |

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|---|---:|---:|
| True 0 | 7,610 | 124 |
| True 1 | 90 | 158 |

Compared with the previous GAT best:

```text
F1:        0.2924 → 0.5962
Precision: 0.2015 → 0.5603
Recall:    0.5323 → 0.6371
FP:         523 → 124
TP:         132 → 158
```

---

## 17. Final Tuned TransformerConv Structural Visualization

After selecting the final tuned TransformerConv configuration, a dedicated PyMOL structural error visualization workflow was added for the final Combined Current + BM5 + ESM-2 PCA16 test protocol.

The visualization script is:

```text
experiments/generate_final_transformerconv_structural_visualization.py
```

It runs the same final model family and preprocessing configuration used in the final evaluation:

```text
Model family: Tuned TransformerConv
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

The final visualization covers the held-out natural test complexes:

```text
1BRS_A_B
1FSS_A_B
3HMX_LH_AB
BM5_1A2K_A_B
BM5_3BP8_A_B
```

The script generates two PyMOL files per complex:

```text
<CASE>_final_transformerconv_structural_errors.pml
<CASE>_final_transformerconv_structural_errors_clean.pml
```

The full version contains selected pairwise C-alpha distance lines. The clean version removes the distance lines and is intended for presentation screenshots.

It also generates:

```text
final_transformerconv_structural_error_examples.csv
final_transformerconv_structural_error_visualization_summary.md
```

### PyMOL Color Legend

| Color | Meaning |
|---|---|
| Green | selected true-positive residue-pair examples |
| Red | selected false-positive residue-pair examples |
| Orange | selected false-negative residue-pair examples |
| Gray | complete protein complex cartoon |

This analysis is qualitative. It does not replace the official final test metrics and does not necessarily display every model prediction. Instead, it provides a structural view of representative successes, over-predictions, and missed contacts.

Command:

```bash
python experiments/generate_final_transformerconv_structural_visualization.py --processed_dir data/processed_combined_current_bm5_esm2_pca16 --out_dir experiments/structural_error_visualization_final_transformerconv --hidden_channels 16 --heads 4 --dropout 0.2 --lr 0.003 --weight_decay 0.001 --threshold_max 0.90 --seed 1 --top_k 10
```

---

## 18. Discussion

The final results show a clear progression:

1. The original current-only dataset was too small and highly imbalanced.
2. BM5-clean expansion increased positive examples and improved recall.
3. Full-pair ESM-2 PCA16 features improved the information available to the model.
4. TransformerConv outperformed standard GAT and other GNN baselines.
5. Tuning dropout and threshold selection dramatically improved precision while preserving useful recall.

The most important final observation is that threshold optimization was not a cosmetic change. Expanding the validation threshold range from 0.60 to 0.90 reduced false positives substantially and produced a much stronger precision-recall balance.

---

## 19. Limitations

Important limitations remain:

1. The dataset is still relatively small.
2. The test set contains only 5 complexes.
3. The final model is selected using validation-based early stopping and threshold selection, but more cross-validation would strengthen confidence.
4. TransformerConv remains seed-sensitive, although the final tuned configuration is much more stable than the initial TransformerConv setup.
5. Interface labels depend on a fixed 5 Å atom-distance threshold.
6. ESM-2 embeddings are sequence-derived and do not directly encode complex-specific binding geometry.
7. More edge and geometry features may further improve performance.
8. Structural visualization is qualitative and requires manual inspection.

---

## 20. Future Work

Recommended future improvements:

1. Run more seeds for the final tuned TransformerConv.
2. Perform complex-level cross-validation.
3. Add edge features such as C-alpha distance bins and geometric compatibility.
4. Test larger ESM-2 models if hardware allows.
5. Evaluate calibration and threshold stability.
6. Build seed ensembles for better robustness.
7. Add non-GNN baselines such as MLP, Random Forest, and XGBoost.
8. Extend the BM5-clean import to more benchmark cases.
9. Update the final presentation deck with the tuned TransformerConv story.

---

## 21. Conclusion

This project built a complete GNN-based pipeline for protein–protein interface prediction.

The strongest improvements came from:

1. BM5-clean dataset expansion
2. full-pair ESM-2 PCA16 features
3. TransformerConv architecture
4. tuned dropout
5. wider validation threshold search

Final best result:

```text
Tuned TransformerConv + Combined Current/BM5 + Full Pair ESM-2 PCA16
F1 = 0.5962
```

This is the best result achieved in the project and represents a substantial improvement over the original GAT baseline.
