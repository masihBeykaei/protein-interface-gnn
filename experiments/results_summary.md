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

Current feature vector:

```text
[CA_distance, degree_partner_1, degree_partner_2]
```

Features:

- Cα distance between the two residues
- Degree of residue `i` in partner 1 graph
- Degree of residue `j` in partner 2 graph

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
However, after adding DBD-style complexes, the number of positive samples increased significantly.

---

## 5. Original Graph-Level Train/Test Split

The initial multi-graph experiments used a graph-level train/test split.

This means the model was evaluated on protein complexes that were not seen during training.

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

### Interpretation

For GCN, `NEGATIVE_RATIO = 5` gives the best positive-class F1-score.  
However, `NEGATIVE_RATIO = 3` produces almost the same F1-score with much higher recall.

For GAT, `NEGATIVE_RATIO = 5` gives the best positive-class F1-score.  
However, `NEGATIVE_RATIO = 2` and `3` achieve much higher recall and may be useful for recall-oriented interface discovery.

---

## 11. Probability Threshold Tuning Experiment

Script:

```text
experiments/tune_probability_threshold.py
```

This experiment evaluates different probability thresholds for predicting the positive class.

A node is predicted as positive if:

```text
P(class 1) >= threshold
```

### Test Set Results

| Model | Threshold | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|-------|-----------|------------------|---------------|-----------|---------------|
| GCN | 0.05 | 0.0865 | 0.9603 | 0.1586 | 0.6618 |
| GCN | 0.10 | 0.1008 | 0.9073 | 0.1815 | 0.7282 |
| GCN | 0.15 | 0.1139 | 0.8411 | 0.2006 | 0.7775 |
| GCN | 0.20 | 0.1247 | 0.7748 | 0.2149 | 0.8120 |
| GCN | 0.25 | 0.1377 | 0.7086 | 0.2306 | 0.8430 |
| GCN | 0.30 | 0.1444 | 0.6159 | 0.2340 | 0.8661 |
| GCN | 0.40 | 0.1762 | 0.4702 | 0.2563 | 0.9094 |
| GCN | 0.50 | 0.2059 | 0.3245 | 0.2519 | 0.9360 |
| GCN | 0.60 | 0.2400 | 0.1589 | 0.1912 | 0.9554 |
| GCN | 0.70 | 0.2250 | 0.0596 | 0.0942 | 0.9620 |
| GCN | 0.80 | 0.0000 | 0.0000 | 0.0000 | 0.9668 |
| GCN | 0.90 | 0.0000 | 0.0000 | 0.9668 |
| GAT | 0.05 | 0.0719 | 0.9868 | 0.1341 | 0.5770 |
| GAT | 0.10 | 0.0828 | 0.9603 | 0.1525 | 0.6456 |
| GAT | 0.15 | 0.0927 | 0.9470 | 0.1688 | 0.6904 |
| GAT | 0.20 | 0.1011 | 0.9404 | 0.1825 | 0.7203 |
| GAT | 0.25 | 0.1073 | 0.9338 | 0.1925 | 0.7399 |
| GAT | 0.30 | 0.1117 | 0.9205 | 0.1993 | 0.7544 |
| GAT | 0.40 | 0.1179 | 0.8477 | 0.2070 | 0.7843 |
| GAT | 0.50 | 0.1274 | 0.7483 | 0.2177 | 0.8215 |
| GAT | 0.60 | 0.1434 | 0.4967 | 0.2226 | 0.8848 |
| GAT | 0.70 | 0.1881 | 0.1258 | 0.1508 | 0.9529 |
| GAT | 0.80 | 0.0000 | 0.0000 | 0.0000 | 0.9668 |
| GAT | 0.90 | 0.0000 | 0.0000 | 0.0000 | 0.9668 |

### Best Thresholds by Positive-Class F1

| Model | Best Threshold | Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------|----------------|-------------|----------|------|----------|
| GCN | 0.40 | 0.1762 | 0.4702 | 0.2563 | 0.9094 |
| GAT | 0.60 | 0.1434 | 0.4967 | 0.2226 | 0.8848 |

### Interpretation

For GCN, lowering the threshold from `0.50` to `0.40` improves the positive-class F1-score.

For GAT, increasing the threshold from `0.50` to `0.60` improves the positive-class F1-score by reducing false positives.

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

### Results

| Model | Best Epoch | Best Threshold | Val Precision 1 | Val Recall 1 | Val F1 1 | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|-------|------------|----------------|-----------------|--------------|----------|------------------|---------------|-----------|---------------|
| GCN | 7 | 0.50 | 0.3434 | 0.2787 | 0.3077 | 0.1940 | 0.1722 | 0.1825 | 0.9488 |
| GAT | 56 | 0.50 | 0.1589 | 0.6721 | 0.2571 | 0.1746 | 0.3642 | 0.2361 | 0.9217 |

### Interpretation

This is the most scientifically reliable experiment so far because the test set is not used for threshold selection or early stopping.

Under this setup:

- GAT achieves higher positive-class F1-score on the test set.
- GAT achieves substantially higher recall than GCN.
- GCN remains more conservative, with higher accuracy but lower recall.
- GAT is more suitable for interface discovery when finding more true interface pairs is important.

---

## 13. Current Best Scientifically Reliable Result

The most reliable setting is the train/validation/test split with validation-based early stopping.

| Model | Best Epoch | Threshold | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|-------|------------|-----------|------------------|---------------|-----------|---------------|
| GCN | 7 | 0.50 | 0.1940 | 0.1722 | 0.1825 | 0.9488 |
| GAT | 56 | 0.50 | 0.1746 | 0.3642 | 0.2361 | 0.9217 |

Current best model under the strict train/validation/test protocol:

```text
GAT
```

---

## 14. Current Conclusion

- Multi-graph training works successfully.
- Adding DBD-style complexes increased the number of positive samples to 698.
- The dataset is still imbalanced, but it is large enough for meaningful multi-graph experiments.
- GCN is conservative and often achieves higher accuracy.
- GAT detects more positive interface/contact nodes.
- In the most reliable train/validation/test setup, GAT outperforms GCN on positive-class F1-score.
- Validation-based early stopping makes the evaluation more scientifically defensible.
- Probability threshold tuning on test data showed useful behavior, but validation-based threshold selection is the preferred protocol.

---

## 15. Next Experiments

Potential next steps:

1. Add richer node features:
   - amino acid type
   - hydrophobicity
   - charge
   - accessible surface area
2. Visualize GAT attention weights.
3. Compare different GAT head counts and hidden dimensions.
4. Analyze false positives and false negatives.
5. Add plots for:
   - class imbalance
   - precision vs recall
   - F1-score comparison
   - negative ratio tuning results
   - threshold tuning results
   - early stopping results
6. Prepare final report and presentation.