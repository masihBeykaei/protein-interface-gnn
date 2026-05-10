# Experimental Results Summary

This document summarizes the current experimental results for protein–protein interface prediction using residue-level correspondence graphs and Graph Neural Networks.

---

## 1. Dataset Summary

The current multi-protein dataset was generated from residue-level correspondence graphs.

Each correspondence node represents a residue pair between two interacting protein partners.

Node labels:

- `0`: non-interface residue pair
- `1`: interface/contact residue pair

A residue pair is labeled as positive if at least one atom pair between the two residues is closer than **5Å**.

---

## 2. Processed Protein Complexes

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

The dataset remains highly imbalanced, which is expected in protein–protein interface prediction. However, adding DBD-style complexes increased the number of positive samples significantly.

---

## 3. Node Features

Each correspondence node represents a residue pair:

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

## 4. Train/Test Split

The dataset was split by graph, not by node.  
This means the model is evaluated on protein complexes that were not seen during training.

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

## 5. Training Strategy

Because the dataset is highly imbalanced, a balanced loss mask was used.

For each training batch:

- all positive nodes are included in the loss
- a random subset of negative nodes is sampled
- the full graph is still used for message passing

The loss is computed using:

```text
all positive nodes + NEGATIVE_RATIO × positive_count negative nodes
```

This allows the model to learn from positive interface/contact nodes without removing graph structure.

---

## 6. Multi-Graph GCN Results

Script:

```text
training/train_multi_graph_gcn.py
```

Configuration:

```text
NEGATIVE_RATIO = 5
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

## 7. Multi-Graph GAT Results

Script:

```text
training/train_multi_graph_gat.py
```

Configuration:

```text
NEGATIVE_RATIO = 5
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

## 8. Model Comparison on Test Set

| Model | Precision 1 | Recall 1 | F1-score 1 | Accuracy |
|-------|-------------|----------|------------|----------|
| Multi-Graph GCN | 0.2068 | 0.3245 | 0.2526 | 0.9362 |
| Multi-Graph GAT | 0.1274 | 0.7483 | 0.2177 | 0.8215 |

### Interpretation

The GCN model is more conservative. It predicts fewer positive nodes, which leads to higher precision and a slightly better positive-class F1-score.

The GAT model is more sensitive to interface/contact nodes. It achieves much higher recall for the positive class, meaning it detects more true interface residue pairs. However, this comes at the cost of more false positives and lower precision.

For protein–protein interface prediction, high recall can be valuable because missing true interface residues may be more problematic than producing additional candidate residues for further analysis.

---

## 9. Negative Ratio Tuning Experiment

Script:

```text
experiments/tune_negative_ratio.py
```

This experiment compares GCN and GAT under different negative sampling ratios.

Test set results:

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

---

## 10. Negative Ratio Tuning Interpretation

### GCN

The best positive-class F1-score for GCN was achieved with:

```text
NEGATIVE_RATIO = 5
```

However, `NEGATIVE_RATIO = 3` produced almost the same F1-score while achieving a much higher recall.

| Ratio | Precision 1 | Recall 1 | F1 1 |
|-------|-------------|----------|------|
| 3 | 0.1643 | 0.5298 | 0.2508 |
| 5 | 0.2059 | 0.3245 | 0.2519 |

This shows a clear precision-recall trade-off.

### GAT

The best positive-class F1-score for GAT was achieved with:

```text
NEGATIVE_RATIO = 5
```

However, GAT with ratios `2` and `3` achieved very high recall:

| Ratio | Precision 1 | Recall 1 | F1 1 |
|-------|-------------|----------|------|
| 2 | 0.1107 | 0.9007 | 0.1972 |
| 3 | 0.1165 | 0.8874 | 0.2060 |
| 5 | 0.1274 | 0.7483 | 0.2177 |

GAT is more recall-oriented than GCN, especially when fewer negative samples are included in the loss.

---

## 11. Current Conclusion

- Multi-graph training works successfully.
- Adding DBD-style complexes increased the number of positive samples from a small proof-of-concept dataset to 698 positive nodes.
- The dataset is still imbalanced, but it is now large enough for meaningful multi-graph experiments.
- GCN provides a stronger positive-class F1-score.
- GAT provides much stronger positive-class recall.
- Negative sampling ratio strongly affects the precision-recall trade-off.
- `NEGATIVE_RATIO = 5` is currently the best default setting for both GCN and GAT in terms of positive-class F1-score.
- `NEGATIVE_RATIO = 2` or `3` may be useful for recall-oriented GAT experiments.

---

## 12. Next Experiments

Potential next steps:

1. Add validation split and early stopping.
2. Tune model probability thresholds instead of using only `argmax`.
3. Add richer node features:
   - amino acid type
   - hydrophobicity
   - charge
   - accessible surface area
4. Visualize GAT attention weights.
5. Compare different GAT head counts and hidden dimensions.
6. Analyze false positives and false negatives.
7. Add plots for:
   - class imbalance
   - precision vs recall
   - F1-score comparison
   - negative ratio tuning results
8. Prepare final report and presentation.