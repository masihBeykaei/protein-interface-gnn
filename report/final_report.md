# Final Report: Protein–Protein Interface Prediction using Graph Neural Networks

## 1. Abstract

This project investigates residue-level protein–protein interface prediction using Graph Neural Networks. Protein partners are represented as residue-level graphs, and a correspondence graph is constructed between two interacting partners. Each node in the correspondence graph represents a candidate residue pair, and the task is to classify whether that pair belongs to the interface.

The project implements a complete pipeline including PDB preprocessing, graph construction, interface labeling, GCN and GAT models, feature engineering, class imbalance analysis, BM5-clean dataset expansion, ESM-2 protein language model embeddings, attention analysis, error analysis, and structural visualization.

The best final result is achieved by combining the original dataset with BM5-clean and adding PCA-reduced ESM-2 pair features:

```text
GAT + Combined Current/BM5 Dataset + ESM-2 PCA16
Test F1 1 = 0.2924
```

---

## 2. Introduction

Protein–protein interactions are essential to many biological processes. Identifying which residues form an interface is important for understanding molecular recognition, protein engineering, and drug design.

Traditional interface prediction often relies on handcrafted structural and physicochemical features. Graph Neural Networks provide a natural framework because proteins can be represented as graphs of residues, where edges encode spatial proximity.

This project formulates interface prediction as binary node classification on a correspondence graph between two protein partners.

---

## 3. Problem Formulation

For a pair of protein partners A and B, the task is to classify residue pairs:

```text
(a_i, b_j)
```

as either:

```text
0 = non-interface
1 = interface/contact
```

A residue pair is positive if any atom pair between the two residues is closer than 5 Å.

The graph-learning problem is:

```text
Input: correspondence graph with node features
Output: binary label for every correspondence node
```

---

## 4. Dataset Construction

The preprocessing pipeline performs:

1. PDB loading
2. chain extraction
3. residue filtering
4. atom coordinate extraction
5. C-alpha coordinate extraction
6. intra-partner graph construction
7. candidate filtering
8. correspondence node construction
9. atom-distance contact labeling
10. correspondence edge construction
11. NumPy array export

Files generated per complex:

```text
<CASE_NAME>_corr_features.npy
<CASE_NAME>_corr_labels.npy
<CASE_NAME>_corr_pairs.npy
<CASE_NAME>_corr_edge_index.npy
```

---

## 5. Original Dataset

The original current dataset contains 12 complexes and 698 positive residue pairs.

| Total Nodes | Positive | Negative | Positive Ratio |
|------------:|---------:|---------:|---------------:|
| 20,707 | 698 | 20,009 | 0.0337 |

This confirms a strong class imbalance. Only about 3.37% of candidate residue pairs are positives.

---

## 6. BM5-Clean Dataset Expansion

To improve data size and diversity, BM5-clean reference complexes were imported.

Imported BM5 complexes:

```text
29
```

Accepted after screening:

```text
19
```

Accepted BM5 positives:

```text
1,096
```

Screening criteria:

```text
positive >= 30
positive_ratio >= 0.02
candidate_nodes <= 8000
```

---

## 7. Combined Dataset

The current dataset and accepted BM5-clean cases were combined.

| Source | Cases | Positive Pairs |
|--------|------:|---------------:|
| Current dataset | 12 | 698 |
| BM5-clean accepted | 19 | 1,096 |
| Combined dataset | 31 | 1,794 |

Processed combined dataset:

| Saved Nodes | Positive | Negative | Positive Ratio |
|------------:|---------:|---------:|---------------:|
| 23,028 | 1,794 | 21,234 | 0.0779 |

Split:

```text
Train: 22 complexes
Validation: 4 complexes
Test: 5 complexes
```

Training graphs were semi-balanced using all positives and 3x negatives. Validation and test graphs were left natural.

---

## 8. Graph Construction

### Residue-Level Graphs

Each protein partner is represented as a residue graph:

```text
node = residue
edge = C-alpha distance < 8 Å
```

### Correspondence Graph

A correspondence node represents:

```text
(residue from partner A, residue from partner B)
```

A correspondence edge connects two correspondence nodes when the corresponding residues are neighbors in both partner graphs.

This structure allows message passing over compatible local residue neighborhoods.

---

## 9. Feature Engineering

Several feature sets were tested.

| Feature Set | Input Dim | Description |
|-------------|----------:|-------------|
| Basic | 3 | CA distance and residue graph degrees |
| Amino acid one-hot | 43 | Basic + residue identity one-hot vectors |
| Physicochemical | 11 | Basic + residue-level physicochemical features |
| Basic + ASA | 5 | Basic + residue solvent accessibility |
| ESM-2 PCA64 | 67 | Basic + 64 PCA components from ESM pair features |
| ESM-2 PCA32 | 35 | Basic + 32 PCA components from ESM pair features |
| ESM-2 PCA16 | 19 | Basic + 16 PCA components from ESM pair features |

---

## 10. Protein Language Model Features

ESM-2 embeddings were extracted using:

```text
facebook/esm2_t6_8M_UR50D
```

For each residue pair, the raw pair feature was:

```text
[ESM_A, ESM_B, abs(ESM_A - ESM_B)]
```

Raw dimension:

```text
960
```

To reduce dimensionality and avoid overfitting, PCA was applied.

Important methodological detail:

```text
Standardization and PCA were fitted only on training pairs.
```

This avoids validation/test leakage.

---

## 11. Models

### GCN

The GCN baseline uses two graph convolution layers.

### GAT

The GAT model uses two graph attention layers.

Best GAT configuration:

```text
hidden_channels = 16
heads = 4
dropout = 0.2
```

GAT was used as the main model for final experiments because it consistently achieved better positive-class recall and F1 in the stricter experiments.

---

## 12. Evaluation Metrics

Because the dataset is highly imbalanced, accuracy is not sufficient.

Primary metrics:

- Precision for class 1
- Recall for class 1
- F1-score for class 1
- Confusion matrix

Positive-class F1 is the main selection metric.

---

## 13. Current-Only Strict Results

Strict split:

```text
Train: 7 current complexes
Validation: 2 current complexes
Test: 3 current complexes
```

Best current-only result:

| Model | Features | Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------|----------|------------:|---------:|-----:|---------:|
| GAT | Basic 3 | 0.1746 | 0.3642 | 0.2361 | 0.9217 |

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|-------------|-------:|-------:|
| True 0 | 4,137 | 260 |
| True 1 | 96 | 55 |

This was the strongest model before dataset expansion.

---

## 14. Feature Engineering Results

| Feature Set | Model | Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------------|-------|------------:|---------:|-----:|---------:|
| Basic 3 | GCN | 0.1940 | 0.1722 | 0.1825 | 0.9488 |
| Basic 3 | GAT | 0.1746 | 0.3642 | 0.2361 | 0.9217 |
| One-hot | GCN | 0.1313 | 0.2781 | 0.1783 | 0.9149 |
| One-hot | GAT | 0.1051 | 0.7285 | 0.1836 | 0.7850 |
| Physicochemical | GCN | 0.2254 | 0.1060 | 0.1441 | 0.9582 |
| Physicochemical | GAT | 0.1566 | 0.2914 | 0.2037 | 0.9244 |
| Basic + ASA | GCN | 0.1887 | 0.2649 | 0.2204 | 0.9378 |
| Basic + ASA | GAT | 0.2184 | 0.2517 | 0.2338 | 0.9453 |

The best current-only F1 came from the basic GAT model.

---

## 15. Class Imbalance Analysis

Several imbalance strategies were tested.

| Experiment | Precision 1 | Recall 1 | F1 1 | TP | FP | FN |
|------------|------------:|---------:|-----:|---:|---:|---:|
| Original strict GAT | 0.1746 | 0.3642 | 0.2361 | 55 | 260 | 96 |
| Train-balanced random | 0.1338 | 0.3642 | 0.1957 | 55 | 356 | 96 |
| Hard negative ratio 5 | 0.1519 | 0.4437 | 0.2264 | 67 | 374 | 84 |
| Hard negative ratio 10 | 0.1489 | 0.3510 | 0.2091 | 53 | 303 | 98 |

Random balancing and hard negative graph reconstruction did not improve F1 on the current-only natural test set.

The likely reason is that removing negative nodes changes message-passing context and can increase over-prediction.

---

## 16. BM5-Only Result

BM5-only dataset:

```text
19 accepted BM5 complexes
1,096 positive residue pairs
```

Result:

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|------------:|---------:|-----:|---------:|
| 0.1824 | 0.8144 | 0.2981 | 0.8917 |

This shows that dataset expansion provides a strong training signal, especially for recall.

---

## 17. Combined Current + BM5 Basic Result

Combined dataset:

```text
31 complexes
1,794 positive residue pairs
```

Result:

| Features | Precision 1 | Recall 1 | F1 1 | Accuracy |
|----------|------------:|---------:|-----:|---------:|
| Basic 3 | 0.1918 | 0.5121 | 0.2791 | 0.9178 |

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|-------------|-------:|-------:|
| True 0 | 7,199 | 535 |
| True 1 | 121 | 127 |

Dataset expansion improved F1 from 0.2361 to 0.2791, though the test set is larger and different.

---

## 18. ESM-2 PCA Experiments

ESM-2 PCA features were added to the combined current + BM5 dataset.

| Experiment | Input Dim | Precision 1 | Recall 1 | F1 1 | Accuracy | TP | FP | FN |
|------------|----------:|------------:|---------:|-----:|---------:|---:|---:|---:|
| Basic combined | 3 | 0.1918 | 0.5121 | 0.2791 | 0.9178 | 127 | 535 | 121 |
| ESM-2 PCA64 | 67 | 0.1169 | 0.7823 | 0.2034 | 0.8096 | 194 | 1,466 | 54 |
| ESM-2 PCA32 | 35 | 0.1953 | 0.5726 | 0.2913 | 0.9134 | 142 | 585 | 106 |
| ESM-2 PCA16 | 19 | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 132 | 523 | 116 |

---

## 19. Best Final Model

The best final model is:

```text
GAT + Combined Current/BM5 Dataset + ESM-2 PCA16
```

Metrics:

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|------------:|---------:|-----:|---------:|
| 0.2015 | 0.5323 | 0.2924 | 0.9199 |

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|-------------|-------:|-------:|
| True 0 | 7,211 | 523 |
| True 1 | 116 | 132 |

Compared with the basic combined model:

```text
F1: 0.2791 → 0.2924
Precision: 0.1918 → 0.2015
Recall: 0.5121 → 0.5323
```

---

## 20. Interpretation of ESM Results

ESM-2 PCA64 produced too many false positives. This suggests that high-dimensional language model features may over-amplify positive predictions on this dataset.

ESM-2 PCA32 improved recall and F1.

ESM-2 PCA16 gave the best balance between precision and recall. It reduced false positives compared with the basic combined baseline while also increasing true positives.

This shows that compact protein language model features can improve graph-based interface prediction when carefully reduced and combined with geometric features.

---

## 21. Attention Analysis

GAT attention weights were extracted from the first GAT layer of the best strict current-only model.

The attention analysis showed that raw top attention weights are often dominated by self-loops. A refined attention analysis was therefore added to separate:

- non-self attention edges
- predicted-positive context
- true-positive context
- false-positive context
- false-negative context

Attention weights are interpreted as local message-passing importance, not direct biological importance.

---

## 22. Structural 3D Error Visualization

PyMOL scripts were generated to visualize true positives, false positives, and false negatives in 3D structures.

Output examples:

```text
experiments/structural_error_visualization/1BRS_A_B_structural_errors.pml
experiments/structural_error_visualization/1FSS_A_B_structural_errors.pml
experiments/structural_error_visualization/3HMX_LH_AB_structural_errors.pml
```

Color legend:

| Color | Meaning |
|-------|---------|
| Green | True positives |
| Red | False positives |
| Orange | False negatives |

This analysis connects numerical model errors to structural interpretation.

---

## 23. Discussion

The project shows that GNNs can learn meaningful patterns for protein–protein interface prediction, but class imbalance remains a major challenge.

The original current-only dataset was too small and imbalanced for strong generalization. Dataset expansion using BM5-clean provided a major improvement by increasing positive residue pairs from 698 to 1,794.

The ESM-2 experiments show that protein language model embeddings can improve performance, but dimensionality matters. PCA64 caused severe over-prediction, while PCA16 produced the best final result.

This suggests that pretrained protein embeddings are useful, but they must be regularized or compressed for small-to-medium structural datasets.

---

## 24. Limitations

Important limitations:

1. The dataset is still relatively small.
2. The combined test set is larger and different from the original current-only test set.
3. Interface labels use a fixed 5 Å atom-distance threshold.
4. Candidate filtering may remove some useful context.
5. ESM-2 embeddings are sequence-based and do not directly include complex-specific structural context.
6. The GAT model still produces many false positives.
7. Attention weights should not be interpreted as direct biological causality.
8. PyMOL visualization is qualitative and requires manual inspection.

---

## 25. Future Work

Recommended next steps:

1. Evaluate ESM-2 PCA8 or PCA24 for additional dimensionality ablation.
2. Add edge features such as distance bins or relative geometry.
3. Compare with non-GNN baselines such as MLP or Random Forest.
4. Use cross-validation across complexes.
5. Add more BM5-clean cases or additional benchmark datasets.
6. Test larger ESM-2 variants if GPU memory allows.
7. Add calibration or precision-oriented thresholding.
8. Update final presentation deck with the ESM-2 PCA16 result.

---

## 26. Conclusion

This project implemented a full graph-based protein–protein interface prediction pipeline, starting from PDB files and ending with interpretable GNN predictions.

The strongest improvement came from two steps:

1. expanding the dataset with BM5-clean
2. adding compact PCA-reduced ESM-2 protein language model features

Final best result:

```text
Combined Current + BM5 + ESM-2 PCA16 + GAT
Precision 1 = 0.2015
Recall 1 = 0.5323
F1 1 = 0.2924
Accuracy = 0.9199
```

This is the best result achieved in the project so far and represents a meaningful improvement over both the current-only model and the combined basic-feature model.
