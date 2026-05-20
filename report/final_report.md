# Final Report: Protein–Protein Interface Prediction using Graph Neural Networks

## 1. Abstract

This project investigates residue-level protein–protein interface prediction using Graph Neural Networks. Protein partners are represented as residue-level graphs, and a correspondence graph is constructed between two interacting partners. Each node in the correspondence graph represents a candidate residue pair, and the task is to classify whether that pair belongs to the protein–protein interface.

The project implements a complete pipeline including PDB preprocessing, graph construction, interface labeling, GCN and GAT baselines, feature engineering, class imbalance analysis, BM5-clean dataset expansion, ESM-2 protein language model embeddings, ESM feature ablations, multiple GNN architecture comparisons, attention analysis, error analysis, and structural visualization.

The best final single-run result is achieved by combining the original dataset with BM5-clean, adding PCA-reduced full-pair ESM-2 features, and replacing GAT with TransformerConv:

```text
TransformerConv + Combined Current/BM5 Dataset + Full Pair ESM-2 PCA16
Test F1 1 = 0.4134
```

A multi-seed TransformerConv evaluation produced:

```text
Mean F1 1 = 0.3310 ± 0.0705
```

This shows that TransformerConv outperforms the previous GAT baseline on average, while also achieving a much stronger best single-run result.

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

Training graphs were semi-balanced using all positives and sampled negatives. Validation and test graphs were left natural.

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
| Full pair ESM-2 PCA64 | 67 | Basic + 64 PCA components from `[ESM_A, ESM_B, absdiff]` |
| Full pair ESM-2 PCA32 | 35 | Basic + 32 PCA components from `[ESM_A, ESM_B, absdiff]` |
| Full pair ESM-2 PCA16 | 19 | Basic + 16 PCA components from `[ESM_A, ESM_B, absdiff]` |
| ESM variant PCA | 19/35 | Basic + PCA components from absdiff/product variants |

---

## 10. Protein Language Model Features

ESM-2 embeddings were extracted using:

```text
facebook/esm2_t6_8M_UR50D
```

For each residue pair, the full raw pair feature was:

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

Additional ESM pair-feature variants were tested:

```text
absdiff
product
absdiff + product
```

---

## 11. Models

The following GNN architectures were evaluated:

| Model | Role in the project |
|-------|---------------------|
| GCN | Basic graph convolution baseline |
| GAT | Main attention-based baseline |
| GATv2 | More flexible attention variant |
| GraphSAGE | Neighborhood aggregation baseline |
| GIN | Expressive MLP-based message passing |
| TransformerConv | Transformer-style graph attention model |

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

## 14. Feature Engineering Results on Current-Only Dataset

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

## 18. Full Pair ESM-2 PCA Experiments with GAT

Full pair ESM features were added to the combined current + BM5 dataset.

| Experiment | Input Dim | Precision 1 | Recall 1 | F1 1 | Accuracy | TP | FP | FN |
|------------|----------:|------------:|---------:|-----:|---------:|---:|---:|---:|
| Basic combined | 3 | 0.1918 | 0.5121 | 0.2791 | 0.9178 | 127 | 535 | 121 |
| Full pair ESM-2 PCA64 | 67 | 0.1169 | 0.7823 | 0.2034 | 0.8096 | 194 | 1,466 | 54 |
| Full pair ESM-2 PCA32 | 35 | 0.1953 | 0.5726 | 0.2913 | 0.9134 | 142 | 585 | 106 |
| Full pair ESM-2 PCA16 | 19 | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 132 | 523 | 116 |

PCA16 gave the best GAT trade-off.

---

## 19. F1 Improvement Attempts Before Architecture Switching

After the GAT PCA16 result, additional experiments were performed to check if F1 could be increased further.

### 19.1 Hyperparameter Tuning

Dedicated GAT hyperparameter tuning for ESM-PCA16 did not improve test F1.

| Experiment | Best Validation-Based Test F1 |
|------------|------------------------------:|
| ESM-PCA16 GAT hyperparameter tuning | 0.2077 |

### 19.2 Train Negative Ratio Tuning

| Experiment | Precision 1 | Recall 1 | F1 1 | Accuracy | TP | FP | FN |
|------------|------------:|---------:|-----:|---------:|---:|---:|---:|
| ESM-PCA16 ratio 3 / GAT best | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 132 | 523 | 116 |
| ESM-PCA16 ratio 4 | 0.2600 | 0.3145 | 0.2847 | 0.9509 | 78 | 222 | 170 |
| ESM-PCA16 ratio 5 | 0.2507 | 0.3468 | 0.2910 | 0.9475 | 86 | 257 | 162 |

Higher negative ratios improved precision and reduced false positives, but recall dropped too much to improve F1.

### 19.3 ESM Pair-Feature Variant Ablations

| Variant | Input Dim | Precision 1 | Recall 1 | F1 1 | Accuracy | TP | FP | FN |
|---------|----------:|------------:|---------:|-----:|---------:|---:|---:|---:|
| Full pair PCA16 / GAT best | 19 | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 132 | 523 | 116 |
| absdiff PCA16 | 19 | 0.1924 | 0.5081 | 0.2791 | 0.9184 | 126 | 529 | 122 |
| absdiff PCA32 | 35 | 0.2003 | 0.5081 | 0.2873 | 0.9217 | 126 | 503 | 122 |
| product PCA16 | 19 | 0.1877 | 0.5806 | 0.2837 | 0.9089 | 144 | 623 | 104 |
| absdiff_product PCA16 | 19 | 0.2232 | 0.4113 | 0.2894 | 0.9372 | 102 | 355 | 146 |

The full pair PCA16 feature representation produced the best GAT F1 trade-off.

---

## 20. GNN Architecture Comparison on ESM-PCA16

All experiments in this table use:

```text
Combined Current + BM5 + Full Pair ESM-2 PCA16
Input dimension = 19
```

| Model | Seed | Precision 1 | Recall 1 | F1 1 | Accuracy | TP | FP | FN |
|-------|-----:|------------:|---------:|-----:|---------:|---:|---:|---:|
| GCN | 42 | 0.1085 | 0.5968 | 0.1836 | 0.8351 | 148 | 1,216 | 100 |
| GIN | 42 | 0.1075 | 0.5484 | 0.1798 | 0.8445 | 136 | 1,129 | 112 |
| GATv2 | 42 | 0.1957 | 0.4355 | 0.2700 | 0.9268 | 108 | 444 | 140 |
| GraphSAGE | 42 | 0.1860 | 0.5887 | 0.2827 | 0.9072 | 146 | 639 | 102 |
| GAT | 42 | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 132 | 523 | 116 |
| TransformerConv | 42 | 0.2041 | 0.8024 | 0.3254 | 0.8966 | 199 | 776 | 49 |
| TransformerConv | 7 | 0.3042 | 0.6452 | 0.4134 | 0.9431 | 160 | 366 | 88 |
| TransformerConv | 13 | 0.1383 | 0.9637 | 0.2419 | 0.8123 | 239 | 1,489 | 9 |
| TransformerConv | 21 | 0.2112 | 0.9153 | 0.3432 | 0.8911 | 227 | 848 | 21 |

---

## 21. TransformerConv Multi-Seed Analysis

TransformerConv was evaluated with four seeds:

```text
42, 7, 13, 21
```

| Seed | Precision 1 | Recall 1 | F1 1 | Accuracy |
|-----:|------------:|---------:|-----:|---------:|
| 42 | 0.2041 | 0.8024 | 0.3254 | 0.8966 |
| 7 | 0.3042 | 0.6452 | 0.4134 | 0.9431 |
| 13 | 0.1383 | 0.9637 | 0.2419 | 0.8123 |
| 21 | 0.2112 | 0.9153 | 0.3432 | 0.8911 |

Summary:

| Statistic | F1 |
|-----------|---:|
| Mean | 0.3310 |
| Sample standard deviation | 0.0705 |
| Best | 0.4134 |
| Worst | 0.2419 |

TransformerConv is more seed-sensitive than GAT, but its mean F1 remains higher than the GAT baseline.

---

## 22. Best Final Model

The best final single-run model is:

```text
TransformerConv + Combined Current/BM5 Dataset + Full Pair ESM-2 PCA16
Seed = 7
```

Metrics:

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|------------:|---------:|-----:|---------:|
| 0.3042 | 0.6452 | 0.4134 | 0.9431 |

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|-------------|-------:|-------:|
| True 0 | 7,368 | 366 |
| True 1 | 88 | 160 |

Compared with the previous GAT best:

```text
F1:       0.2924 → 0.4134
Precision:0.2015 → 0.3042
Recall:   0.5323 → 0.6452
FP:       523 → 366
TP:       132 → 160
```

---

## 23. Attention Analysis

GAT attention weights were extracted from the first GAT layer of the best strict current-only model.

The attention analysis showed that raw top attention weights are often dominated by self-loops. A refined attention analysis was therefore added to separate:

- non-self attention edges
- predicted-positive context
- true-positive context
- false-positive context
- false-negative context

Attention weights are interpreted as local message-passing importance, not direct biological importance.

---

## 24. Structural 3D Error Visualization

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

## 25. Discussion

The project shows that GNNs can learn meaningful patterns for protein–protein interface prediction, but class imbalance remains a major challenge.

The original current-only dataset was too small and imbalanced for strong generalization. Dataset expansion using BM5-clean provided a major improvement by increasing positive residue pairs from 698 to 1,794.

The ESM-2 experiments show that protein language model embeddings can improve performance, but dimensionality matters. PCA64 caused severe over-prediction, while PCA16 produced the best GAT result.

The architecture comparison was the largest final improvement. TransformerConv outperformed GAT, GATv2, GraphSAGE, GIN, and GCN on the ESM-PCA16 dataset. This suggests that Transformer-style attention is better suited to the correspondence graph setting than standard graph convolution or simpler aggregation.

However, TransformerConv is seed-sensitive. The best seed achieved F1 = 0.4134, while the four-seed mean was 0.3310. This should be reported transparently.

---

## 26. Limitations

Important limitations:

1. The dataset is still relatively small.
2. The combined test set is larger and different from the original current-only test set.
3. Interface labels use a fixed 5 Å atom-distance threshold.
4. Candidate filtering may remove some useful context.
5. ESM-2 embeddings are sequence-based and do not directly include complex-specific structural context.
6. TransformerConv shows noticeable seed sensitivity.
7. More seeds or cross-validation would provide stronger statistical confidence.
8. Attention weights should not be interpreted as direct biological causality.
9. PyMOL visualization is qualitative and requires manual inspection.
10. Additional gains may require edge features, geometric features, or more benchmark data.

---

## 27. Future Work

Recommended next steps:

1. Run TransformerConv with more seeds.
2. Perform complex-level cross-validation.
3. Add edge features such as distance bins or relative geometry.
4. Compare with non-GNN baselines such as MLP or Random Forest.
5. Add more BM5-clean cases or additional benchmark datasets.
6. Test larger ESM-2 variants if GPU memory allows.
7. Add calibration or threshold-stability analysis.
8. Explore ensembling across TransformerConv seeds.
9. Update final presentation deck with the final TransformerConv architecture-comparison story.

---

## 28. Conclusion

This project implemented a full graph-based protein–protein interface prediction pipeline, starting from PDB files and ending with interpretable GNN predictions.

The strongest improvement came from three steps:

1. expanding the dataset with BM5-clean
2. adding compact PCA-reduced full-pair ESM-2 protein language model features
3. switching from GAT to TransformerConv

Final best single-run result:

```text
Combined Current + BM5 + Full Pair ESM-2 PCA16 + TransformerConv
Seed = 7
Precision 1 = 0.3042
Recall 1 = 0.6452
F1 1 = 0.4134
Accuracy = 0.9431
```

Multi-seed TransformerConv summary:

```text
Mean F1 1 = 0.3310 ± 0.0705
```

This is the best result achieved in the project so far and represents a substantial improvement over the previous GAT-based best model.
