# Experimental Results Summary

This document summarizes the current experimental results for protein–protein interface prediction using residue-level correspondence graphs and Graph Neural Networks.

---

## 1. Project Goal

The goal of this project is to predict protein–protein interface/contact residue pairs using graph-based learning.

Each protein partner is represented as a residue-level graph.  
A correspondence graph is then built between two interacting partners, where each correspondence node represents a possible residue pair.

The final task is a node-level binary classification problem:

- `0`: non-interface residue pair
- `1`: interface/contact residue pair

---

## 2. Dataset Construction

For each protein complex:

1. Load the PDB structure.
2. Extract residues from selected chains.
3. Build residue-level graphs using Cα distance.
4. Build a correspondence graph between the two partners.
5. Label residue pairs using atomic distance.
6. Apply candidate filtering to reduce graph size.
7. Generate node features for each correspondence node.

A residue pair is labeled as positive if at least one atom pair between the two residues is closer than:

```text
5Å
```

---

## 3. Node Features

Each correspondence node represents:

```text
(residue_i, residue_j)
```

### Basic 3-Feature Representation

```text
[CA_distance, degree_partner_1, degree_partner_2]
```

Input dimension:

```text
3
```

### Amino Acid One-Hot Representation

```text
[CA_distance, degree_partner_1, degree_partner_2, aa_A_onehot(20), aa_B_onehot(20)]
```

Input dimension:

```text
43
```

### Physicochemical Representation

```text
[
  CA_distance,
  degree_partner_1,
  degree_partner_2,
  hydrophobicity_A,
  hydrophobicity_B,
  charge_A,
  charge_B,
  polarity_A,
  polarity_B,
  aromaticity_A,
  aromaticity_B
]
```

Input dimension:

```text
11
```

### Basic + Accessible Surface Area Representation

```text
[
  CA_distance,
  degree_partner_1,
  degree_partner_2,
  ASA_A,
  ASA_B
]
```

Input dimension:

```text
5
```

---

## 4. Processed Protein Complexes

The current multi-protein dataset includes original examples and DBD-style protein complexes.

| Case | Nodes | Positive | Negative | Positive Ratio | Edges |
|------|-------|----------|----------|----------------|-------|
| 1BRS_A_B | 225 | 16 | 209 | 0.0711 | 13,448 |
| 1FSS_A_B | 2,013 | 63 | 1,950 | 0.0313 | 200,384 |
| 1AHW_AB_C | 2,142 | 73 | 2,069 | 0.0341 | 216,032 |
| 1DQJ_AB_C | 1,978 | 71 | 1,907 | 0.0359 | 202,048 |
| 1E6J_HL_P | 1,131 | 51 | 1,080 | 0.0451 | 99,696 |
| 1JPS_HL_T | 2,184 | 71 | 2,113 | 0.0325 | 214,656 |
| 1MLC_AB_E | 1,540 | 54 | 1,486 | 0.0351 | 155,232 |
| 1WEJ_HL_F | 1,026 | 41 | 985 | 0.0400 | 89,280 |
| 2FD6_HL_U | 1,147 | 47 | 1,100 | 0.0410 | 103,824 |
| 2VIS_AB_C | 1,728 | 51 | 1,677 | 0.0295 | 147,744 |
| 3HMX_LH_AB | 2,310 | 72 | 2,238 | 0.0312 | 208,680 |
| 3MJ9_HL_A | 3,283 | 88 | 3,195 | 0.0268 | 310,000 |

### Total Dataset

| Total Nodes | Total Positive | Total Negative | Positive Ratio |
|------------|----------------|----------------|----------------|
| 20,707 | 698 | 20,009 | 0.0337 |

The dataset is highly imbalanced, which is expected in protein–protein interface prediction.

---

## 5. Initial Graph-Level Train/Test Split

The initial multi-graph experiments used a graph-level train/test split.

### Train Graphs

- 1WEJ_HL_F
- 1JPS_HL_T
- 1AHW_AB_C
- 2FD6_HL_U
- 2VIS_AB_C
- 1MLC_AB_E
- 3MJ9_HL_A
- 1DQJ_AB_C
- 1E6J_HL_P

### Test Graphs

- 1BRS_A_B
- 1FSS_A_B
- 3HMX_LH_AB

---

## 6. Class Imbalance Handling

Protein–protein interface prediction is naturally imbalanced because only a small fraction of residue pairs are true interface/contact pairs.

To handle this, multi-graph training uses a balanced loss mask.

For each training batch:

- all positive nodes are included in the loss
- a random subset of negative nodes is sampled
- the full graph is still used for message passing

The loss is computed using:

```text
all positive nodes + NEGATIVE_RATIO × positive_count negative nodes
```

---

## 7. Initial Multi-Graph GCN Results

Script:

```text
training/train_multi_graph_gcn.py
```

Configuration:

```text
NEGATIVE_RATIO = 5
threshold = 0.50
features = basic 3 features
```

### Train Results

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.9764 | 0.9627 | 0.9695 | 15,612 |
| 1 | 0.2402 | 0.3364 | 0.2803 | 547 |

Accuracy:

```text
0.9415
```

### Test Results

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.9763 | 0.9572 | 0.9667 | 4,397 |
| 1 | 0.2068 | 0.3245 | 0.2526 | 151 |

Accuracy:

```text
0.9362
```

---

## 8. Initial Multi-Graph GAT Results

Script:

```text
training/train_multi_graph_gat.py
```

Configuration:

```text
NEGATIVE_RATIO = 5
threshold = 0.50
features = basic 3 features
```

### Train Results

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.9939 | 0.8071 | 0.8908 | 15,612 |
| 1 | 0.1350 | 0.8592 | 0.2333 | 547 |

Accuracy:

```text
0.8088
```

### Test Results

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.9896 | 0.8240 | 0.8992 | 4,397 |
| 1 | 0.1274 | 0.7483 | 0.2177 | 151 |

Accuracy:

```text
0.8215
```

---

## 9. Initial Model Comparison on Test Set

| Model | Precision 1 | Recall 1 | F1-score 1 | Accuracy |
|-------|-------------|----------|------------|----------|
| Multi-Graph GCN | 0.2068 | 0.3245 | 0.2526 | 0.9362 |
| Multi-Graph GAT | 0.1274 | 0.7483 | 0.2177 | 0.8215 |

### Interpretation

The GCN model is more conservative. It predicts fewer positive nodes, which leads to higher precision and a slightly better positive-class F1-score.

The GAT model is more sensitive to interface/contact nodes. It achieves much higher recall for the positive class, meaning it detects more true interface residue pairs. However, this comes at the cost of more false positives and lower precision.

---

## 10. Negative Ratio Tuning Experiment

Script:

```text
experiments/tune_negative_ratio.py
```

This experiment compares GCN and GAT under different negative sampling ratios.

Results were generated using the basic 3-feature representation.

### Test Set Results

| Model | Negative Ratio | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|-------|----------------|------------------|---------------|-----------|---------------|
| GCN | 2 | 0.1442 | 0.6225 | 0.2341 | 0.8648 |
| GCN | 3 | 0.1643 | 0.5298 | 0.2508 | 0.8949 |
| GCN | 5 | 0.2059 | 0.3245 | 0.2519 | 0.9360 |
| GCN | 10 | 0.2653 | 0.0861 | 0.1300 | 0.9617 |
| GAT | 2 | 0.1107 | 0.9007 | 0.1972 | 0.7566 |
| GAT | 3 | 0.1165 | 0.8874 | 0.2060 | 0.7729 |
| GAT | 5 | 0.1274 | 0.7483 | 0.2177 | 0.8215 |
| GAT | 10 | 0.1985 | 0.1788 | 0.1882 | 0.9488 |

### Best Settings by Positive-Class F1

| Model | Best Negative Ratio | Precision 1 | Recall 1 | F1 1 |
|-------|---------------------|-------------|----------|------|
| GCN | 5 | 0.2059 | 0.3245 | 0.2519 |
| GAT | 5 | 0.1274 | 0.7483 | 0.2177 |

---

## 11. Probability Threshold Tuning Experiment

Script:

```text
experiments/tune_probability_threshold.py
```

A node is predicted as positive if:

```text
P(class 1) >= threshold
```

Results were generated using the basic 3-feature representation.

### Best Thresholds by Positive-Class F1

| Model | Best Threshold | Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------|----------------|-------------|----------|------|----------|
| GCN | 0.40 | 0.1762 | 0.4702 | 0.2563 | 0.9094 |
| GAT | 0.60 | 0.1434 | 0.4967 | 0.2226 | 0.8848 |

Threshold tuning shows that probability calibration can slightly improve both models, especially when the dataset is highly imbalanced.

---

## 12. Train/Validation/Test Early Stopping Experiment

Script:

```text
experiments/train_val_test_early_stopping.py
```

This experiment uses a graph-level train/validation/test split.

The validation set is used for:

- early stopping
- selecting the probability threshold for class 1

The test set is used only for final evaluation.

### Split

#### Train Graphs

- 1WEJ_HL_F
- 1JPS_HL_T
- 1AHW_AB_C
- 2FD6_HL_U
- 2VIS_AB_C
- 1MLC_AB_E
- 3MJ9_HL_A

#### Validation Graphs

- 1DQJ_AB_C
- 1E6J_HL_P

#### Test Graphs

- 1BRS_A_B
- 1FSS_A_B
- 3HMX_LH_AB

### Basic 3-Feature Results

| Model | Input Dim | Best Epoch | Best Threshold | Val Precision 1 | Val Recall 1 | Val F1 1 | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|-------|-----------|------------|----------------|-----------------|--------------|----------|------------------|---------------|-----------|---------------|
| GCN | 3 | 7 | 0.50 | 0.3434 | 0.2787 | 0.3077 | 0.1940 | 0.1722 | 0.1825 | 0.9488 |
| GAT | 3 | 56 | 0.50 | 0.1589 | 0.6721 | 0.2571 | 0.1746 | 0.3642 | 0.2361 | 0.9217 |

### Interpretation

This is the most scientifically reliable experiment for the basic feature representation because the test set is not used for threshold selection or early stopping.

---

## 13. Amino Acid One-Hot Feature Experiment

The amino acid one-hot feature vector is:

```text
[CA_distance, degree_partner_1, degree_partner_2, aa_A_onehot(20), aa_B_onehot(20)]
```

Input dimension:

```text
43
```

### Train/Validation/Test Results

| Model | Input Dim | Best Epoch | Best Threshold | Val Precision 1 | Val Recall 1 | Val F1 1 | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|-------|-----------|------------|----------------|-----------------|--------------|----------|------------------|---------------|-----------|---------------|
| GCN | 43 | 11 | 0.40 | 0.1909 | 0.3443 | 0.2456 | 0.1313 | 0.2781 | 0.1783 | 0.9149 |
| GAT | 43 | 34 | 0.40 | 0.1279 | 0.6885 | 0.2157 | 0.1051 | 0.7285 | 0.1836 | 0.7850 |

### Interpretation

Adding amino acid one-hot features increased recall, especially for GAT, but also increased false positives.

---

## 14. Physicochemical Feature Experiment

The physicochemical feature vector is:

```text
[
  CA_distance,
  degree_partner_1,
  degree_partner_2,
  hydrophobicity_A,
  hydrophobicity_B,
  charge_A,
  charge_B,
  polarity_A,
  polarity_B,
  aromaticity_A,
  aromaticity_B
]
```

Input dimension:

```text
11
```

### Train/Validation/Test Results

| Model | Input Dim | Best Epoch | Best Threshold | Val Precision 1 | Val Recall 1 | Val F1 1 | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|-------|-----------|------------|----------------|-----------------|--------------|----------|------------------|---------------|-----------|---------------|
| GCN | 11 | 24 | 0.60 | 0.2804 | 0.2459 | 0.2620 | 0.2254 | 0.1060 | 0.1441 | 0.9582 |
| GAT | 11 | 22 | 0.50 | 0.1548 | 0.6066 | 0.2467 | 0.1566 | 0.2914 | 0.2037 | 0.9244 |

### Interpretation

Compared with amino acid one-hot features, physicochemical features improved GAT positive-class F1-score, but still did not outperform the basic 3-feature representation.

---

## 15. Accessible Surface Area Feature Experiment

Accessible surface area was added as a structural residue-level feature.

The feature vector is:

```text
[
  CA_distance,
  degree_partner_1,
  degree_partner_2,
  ASA_A,
  ASA_B
]
```

Input dimension:

```text
5
```

### Train/Validation/Test Results

| Model | Input Dim | Best Epoch | Best Threshold | Val Precision 1 | Val Recall 1 | Val F1 1 | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|-------|-----------|------------|----------------|-----------------|--------------|----------|------------------|---------------|-----------|---------------|
| GCN | 5 | 35 | 0.60 | 0.2251 | 0.5000 | 0.3104 | 0.1887 | 0.2649 | 0.2204 | 0.9378 |
| GAT | 5 | 191 | 0.50 | 0.3050 | 0.3525 | 0.3270 | 0.2184 | 0.2517 | 0.2338 | 0.9453 |

### Interpretation

ASA features improved GCN substantially compared with the basic 3-feature GCN.

For GAT, ASA increased precision but reduced recall. The final F1-score is very close to the best basic GAT result, but slightly lower.

---

## 16. Feature Set Comparison

| Feature Set | Input Dim | Model | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|------------|-----------|-------|------------------|---------------|-----------|---------------|
| Basic 3 features | 3 | GCN | 0.1940 | 0.1722 | 0.1825 | 0.9488 |
| Basic 3 features | 3 | GAT | 0.1746 | 0.3642 | 0.2361 | 0.9217 |
| Amino acid one-hot | 43 | GCN | 0.1313 | 0.2781 | 0.1783 | 0.9149 |
| Amino acid one-hot | 43 | GAT | 0.1051 | 0.7285 | 0.1836 | 0.7850 |
| Physicochemical | 11 | GCN | 0.2254 | 0.1060 | 0.1441 | 0.9582 |
| Physicochemical | 11 | GAT | 0.1566 | 0.2914 | 0.2037 | 0.9244 |
| Basic + ASA | 5 | GCN | 0.1887 | 0.2649 | 0.2204 | 0.9378 |
| Basic + ASA | 5 | GAT | 0.2184 | 0.2517 | 0.2338 | 0.9453 |

---

## 17. GAT Hyperparameter Tuning

Script:

```text
experiments/tune_gat_hyperparameters.py
```

The experiment tuned:

- hidden channels
- number of attention heads
- dropout

### Results

| Hidden | Heads | Dropout | Val F1 1 | Test F1 1 |
|--------|-------|---------|----------|-----------|
| 16 | 4 | 0.2 | 0.2571 | 0.2361 |
| 32 | 4 | 0.2 | 0.2526 | 0.1980 |
| 16 | 8 | 0.2 | 0.2531 | 0.2069 |
| 32 | 8 | 0.2 | 0.2457 | 0.1899 |
| 16 | 4 | 0.3 | 0.2534 | 0.2118 |

The best configuration remains:

```text
hidden_channels = 16
heads = 4
dropout = 0.2
```

This suggests that larger GAT models do not improve generalization on the current dataset.

---

## 18. Experiment Figures

Generated plots are stored in:

```text
experiments/figures/
```

Figures include:

| File | Description |
|------|-------------|
| `class_imbalance.png` | Positive vs negative correspondence node counts |
| `feature_set_f1_comparison.png` | Positive-class F1 comparison across feature sets |
| `feature_set_precision_recall_f1.png` | Precision, recall, and F1 comparison across feature sets |
| `best_strict_gcn_vs_gat.png` | GCN vs GAT under the strict train/validation/test protocol |
| `negative_ratio_tuning_gcn.png` | Negative sampling ratio tuning for GCN |
| `negative_ratio_tuning_gat.png` | Negative sampling ratio tuning for GAT |
| `threshold_tuning_gcn.png` | Probability threshold tuning for GCN |
| `threshold_tuning_gat.png` | Probability threshold tuning for GAT |
| `gat_attention_distribution.png` | Distribution of GAT attention weights |
| `gat_attention_distribution_log.png` | Log-scale distribution of GAT attention weights |

---

## 19. Error Analysis

Error analysis was performed for the current best strict protocol model:

```text
Model: GAT
Features: basic 3 features
Input dimension: 3
Split: train/validation/test
Threshold selected on validation set
```

Script:

```text
experiments/analyze_errors.py
```

### Training Selection

| Best Epoch | Best Threshold | Best Validation F1 1 |
|------------|----------------|----------------------|
| 56 | 0.50 | 0.2571 |

### Test Metrics

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------------|----------|------|----------|
| 0.1746 | 0.3642 | 0.2361 | 0.9217 |

### Confusion Matrix on Test Set

| True / Pred | Pred 0 | Pred 1 |
|-------------|--------|--------|
| True 0 | 4137 | 260 |
| True 1 | 96 | 55 |

### Per-Test-Graph Error Summary

| Case | Nodes | Positive | Negative | TP | TN | FP | FN |
|------|-------|----------|----------|----|----|----|----|
| 1BRS_A_B | 225 | 16 | 209 | 11 | 139 | 70 | 5 |
| 1FSS_A_B | 2,013 | 63 | 1,950 | 19 | 1,862 | 88 | 44 |
| 3HMX_LH_AB | 2,310 | 72 | 2,238 | 25 | 2,136 | 102 | 47 |

### Error Analysis Interpretation

The model produces more false positives than false negatives:

```text
False positives: 260
False negatives: 96
```

This suggests that the best GAT model is sensitive to potential interface/contact patterns, but it is not yet highly precise.

---

## 20. GAT Attention Analysis

Attention weights were extracted from the first GATConv layer of the best strict GAT model.

Scripts:

```text
experiments/visualize_gat_attention.py
experiments/refine_gat_attention_analysis.py
```

### Attention Output Files

```text
experiments/gat_attention_summary.md
experiments/gat_attention_refined_summary.md
experiments/gat_attention_top_edges.csv
experiments/gat_attention_top_non_self_edges.csv
experiments/gat_attention_top_predicted_positive_edges.csv
experiments/gat_attention_top_tp_context_edges.csv
experiments/gat_attention_top_fp_context_edges.csv
experiments/gat_attention_top_fn_context_edges.csv
experiments/gat_attention_error_context_edges.csv
```

The full attention table is intentionally not committed because it is large:

```text
experiments/gat_attention_weights.csv
```

### Interpretation

Raw top attention edges were dominated by self-loops. The refined attention analysis removes self-loop dominance and separates attention edges by prediction context.

Attention subsets include:

- non-self edges
- predicted-positive context
- true-positive context
- false-positive context
- false-negative context
- FP/FN error context

GAT attention weights are locally normalized, so they should be interpreted as local message-passing importance rather than global biological importance.

---

## 21. Current Best Scientifically Reliable Result

The most reliable setting is the train/validation/test split with validation-based early stopping.

| Feature Set | Model | Best Epoch | Threshold | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|------------|-------|------------|-----------|------------------|---------------|-----------|---------------|
| Basic 3 features | GCN | 7 | 0.50 | 0.1940 | 0.1722 | 0.1825 | 0.9488 |
| Basic 3 features | GAT | 56 | 0.50 | 0.1746 | 0.3642 | 0.2361 | 0.9217 |
| Amino acid one-hot, 43 features | GCN | 11 | 0.40 | 0.1313 | 0.2781 | 0.1783 | 0.9149 |
| Amino acid one-hot, 43 features | GAT | 34 | 0.40 | 0.1051 | 0.7285 | 0.1836 | 0.7850 |
| Physicochemical, 11 features | GCN | 24 | 0.60 | 0.2254 | 0.1060 | 0.1441 | 0.9582 |
| Physicochemical, 11 features | GAT | 22 | 0.50 | 0.1566 | 0.2914 | 0.2037 | 0.9244 |
| Basic + ASA, 5 features | GCN | 35 | 0.60 | 0.1887 | 0.2649 | 0.2204 | 0.9378 |
| Basic + ASA, 5 features | GAT | 191 | 0.50 | 0.2184 | 0.2517 | 0.2338 | 0.9453 |

Current best model under the strict train/validation/test protocol:

```text
GAT with basic 3-feature representation
```

Best strict positive-class F1-score:

```text
0.2361
```

The Basic + ASA GAT model is very close:

```text
Test F1 1 = 0.2338
```

---

## 22. Current Conclusion

- Multi-graph training works successfully.
- Adding DBD-style complexes increased the number of positive samples to 698.
- The dataset is still imbalanced, but it is large enough for meaningful multi-graph experiments.
- GCN is conservative and often achieves higher accuracy.
- GAT detects more positive interface/contact nodes.
- In the strict train/validation/test setup, GAT outperforms GCN on positive-class F1-score when using the basic 3-feature representation.
- Amino acid one-hot features increase recall, especially for GAT, but also increase false positives.
- Physicochemical features are more compact and improve over one-hot features for GAT, but still do not outperform the basic 3-feature representation.
- ASA features substantially improve GCN and produce a more precision-oriented GAT.
- The best overall strict F1 remains GAT with basic 3 features.
- Larger GAT models did not improve generalization.
- Attention analysis is useful for local explanation, but raw attention is not a direct biological importance score.
- Error analysis shows that the best GAT model produces more false positives than false negatives.

---

## 23. Remaining Next Experiments

Potential next steps:

1. Analyze false positives and false negatives in 3D structural context.
2. Add structural visualization of predicted interface pairs.
3. Add protein language model embeddings.
4. Expand the dataset with more complexes.
5. Prepare final presentation/deck.