# Experimental Results Summary

This document summarizes the current experimental results for protein–protein interface prediction using residue-level correspondence graphs and Graph Neural Networks.

---

## 1. Project Goal

The goal of this project is to predict protein–protein interface/contact residue pairs using graph-based learning.

Each protein partner is represented as a residue-level graph. A correspondence graph is then built between two interacting partners, where each correspondence node represents a possible residue pair.

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

## 4. Original Current Dataset

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

### Original Current Dataset Total

| Total Nodes | Total Positive | Total Negative | Positive Ratio |
|------------|----------------|----------------|----------------|
| 20,707 | 698 | 20,009 | 0.0337 |

---

## 5. BM5-Clean Dataset Expansion

BM5-clean was used to increase dataset size and diversity.

### BM5 Import and Screening

| Stage | Count |
|-------|------:|
| Imported BM5 reference complexes | 29 |
| Accepted after screening | 19 |
| Rejected after screening | 10 |

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

### Accepted BM5 Cases

| Case | Candidate Nodes | Positive | Positive Ratio | Status |
|------|----------------:|---------:|---------------:|--------|
| BM5_1A2K_A_B | 1,610 | 45 | 0.0280 | accepted |
| BM5_1BJ1_A_B | 1,260 | 67 | 0.0532 | accepted |
| BM5_1EZU_A_B | 4,680 | 100 | 0.0214 | accepted |
| BM5_1FCC_A_B | 1,330 | 43 | 0.0323 | accepted |
| BM5_1JWH_A_B | 1,435 | 41 | 0.0286 | accepted |
| BM5_1ML0_A_B | 2,714 | 73 | 0.0269 | accepted |
| BM5_1OFU_A_B | 1,715 | 50 | 0.0292 | accepted |
| BM5_1QFW_A_B | 1,050 | 75 | 0.0714 | accepted |
| BM5_1RLB_A_B | 1,666 | 48 | 0.0288 | accepted |
| BM5_1RV6_A_B | 1,376 | 45 | 0.0327 | accepted |
| BM5_1WDW_A_B | 4,030 | 111 | 0.0275 | accepted |
| BM5_1XU1_A_B | 1,320 | 55 | 0.0417 | accepted |
| BM5_2B4J_A_B | 936 | 39 | 0.0417 | accepted |
| BM5_3BP8_A_B | 1,824 | 52 | 0.0285 | accepted |
| BM5_3LVK_A_B | 945 | 37 | 0.0392 | accepted |
| BM5_4FQI_A_B | 1,144 | 42 | 0.0367 | accepted |
| BM5_4GXU_A_B | 2,173 | 53 | 0.0244 | accepted |
| BM5_4HX3_A_B | 2,958 | 81 | 0.0274 | accepted |
| BM5_4LW4_A_B | 1,530 | 39 | 0.0255 | accepted |

---

## 6. Combined Current + BM5 Dataset

The original current dataset and accepted BM5 cases were combined.

### Combined Case Counts

| Source | Cases |
|--------|------:|
| Current dataset | 12 |
| BM5-clean accepted | 19 |
| Combined total | 31 |

### Combined Processed Dataset

The combined dataset was built using train-balanced / natural-evaluation construction.

| Total Saved Nodes | Total Positive | Total Negative | Positive Ratio |
|------------------:|---------------:|---------------:|---------------:|
| 23,028 | 1,794 | 21,234 | 0.0779 |

Split counts:

| Split | Cases |
|-------|------:|
| Train | 22 |
| Validation | 4 |
| Test | 5 |

The training split was semi-balanced using:

```text
all positive pairs + 3 × positive_count negative pairs
```

Validation and test splits kept their natural class imbalance.

---

## 7. Class Imbalance Handling

Multi-graph training originally used a balanced loss mask.

For each training batch:

- all positive nodes are included in the loss
- a random subset of negative nodes is sampled
- the full graph is still used for message passing

The loss is computed using:

```text
all positive nodes + NEGATIVE_RATIO × positive_count negative nodes
```

Default setting:

```text
NEGATIVE_RATIO = 5
```

Later experiments also tested dataset-level train balancing and hard negative mining.

---

## 8. Initial Multi-Graph GCN Results

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

## 9. Initial Multi-Graph GAT Results

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

## 10. Negative Ratio Tuning

Script:

```text
experiments/tune_negative_ratio.py
```

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

## 11. Probability Threshold Tuning

Script:

```text
experiments/tune_probability_threshold.py
```

A node is predicted as positive if:

```text
P(class 1) >= threshold
```

Best threshold results:

| Model | Best Threshold | Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------|----------------|-------------|----------|------|----------|
| GCN | 0.40 | 0.1762 | 0.4702 | 0.2563 | 0.9094 |
| GAT | 0.60 | 0.1434 | 0.4967 | 0.2226 | 0.8848 |

This experiment is exploratory because threshold tuning was performed on the test set. The stricter validation-based protocol is preferred.

---

## 12. Strict Current-Only Train/Validation/Test Experiment

Script:

```text
experiments/train_val_test_early_stopping.py
```

The validation set is used for:

- early stopping
- selecting the probability threshold for class 1

The test set is used only for final evaluation.

### Current-Only Split

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

---

## 13. Basic 3-Feature Results

| Model | Input Dim | Best Epoch | Best Threshold | Val Precision 1 | Val Recall 1 | Val F1 1 | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|-------|-----------|------------|----------------|-----------------|--------------|----------|------------------|---------------|-----------|---------------|
| GCN | 3 | 7 | 0.50 | 0.3434 | 0.2787 | 0.3077 | 0.1940 | 0.1722 | 0.1825 | 0.9488 |
| GAT | 3 | 56 | 0.50 | 0.1589 | 0.6721 | 0.2571 | 0.1746 | 0.3642 | 0.2361 | 0.9217 |

This was the strongest current-only result before dataset expansion.

---

## 14. Feature Set Comparison

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

## 15. GAT Hyperparameter Tuning

Script:

```text
experiments/tune_gat_hyperparameters.py
```

| Hidden | Heads | Dropout | Val F1 1 | Test F1 1 |
|--------|-------|---------|----------|-----------|
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

Larger GAT models did not improve generalization.

---

## 16. Current-Only Error Analysis

Error analysis was performed for the previous best strict current-only model:

```text
GAT + basic 3 features
```

Script:

```text
experiments/analyze_errors.py
```

### Test Metrics

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------------|----------|------|----------|
| 0.1746 | 0.3642 | 0.2361 | 0.9217 |

### Confusion Matrix

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

The model produces more false positives than false negatives:

```text
False positives: 260
False negatives: 96
```

This suggests the model is sensitive but not highly precise.

---

## 17. Imbalance Ablation Experiments

Several imbalance strategies were tested while keeping validation/test natural.

| Experiment | Test Setup | Precision 1 | Recall 1 | F1 1 | Accuracy | TN | FP | FN | TP |
|------------|------------|-------------|----------|------|----------|----|----|----|----|
| Original strict GAT | Current-only natural test | 0.1746 | 0.3642 | 0.2361 | 0.9217 | 4137 | 260 | 96 | 55 |
| Train-balanced random | Current-only natural test | 0.1338 | 0.3642 | 0.1957 | 0.9006 | 4041 | 356 | 96 | 55 |
| Hard negative ratio 5 | Current-only natural test | 0.1519 | 0.4437 | 0.2264 | 0.8993 | 4023 | 374 | 84 | 67 |
| Hard negative ratio 10 | Current-only natural test | 0.1489 | 0.3510 | 0.2091 | 0.9118 | 4094 | 303 | 98 | 53 |

Interpretation:

- Random train balancing did not improve F1.
- Hard negative ratio 5 improved recall but increased false positives.
- Hard negative ratio 10 reduced false positives relative to ratio 5 but still did not improve F1.
- Randomly removing negative nodes from graphs can remove useful message-passing context.

---

## 18. BM5-Only Experiment

Dataset:

```text
BM5-clean accepted cases only
19 accepted complexes
1,096 positive residue pairs
```

Build strategy:

```text
Train: semi-balanced
Validation: natural
Test: natural
```

Script:

```text
experiments/train_expanded_gat.py
```

### BM5-Only GAT Results

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------------|----------|------|----------|
| 0.1824 | 0.8144 | 0.2981 | 0.8917 |

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|-------------|--------|--------|
| True 0 | 2983 | 354 |
| True 1 | 18 | 79 |

This result is not directly comparable to the current-only result because the test set is different. However, it shows that the expanded BM5 dataset provides useful training signal and substantially improves recall on BM5 natural test complexes.

---

## 19. Combined Current + BM5 Experiment

Dataset:

```text
Current dataset + BM5 accepted dataset
31 usable complexes
23,028 saved nodes
1,794 positive residue pairs
21,234 negative residue pairs
```

Split:

```text
Train: 22 complexes
Validation: 4 complexes
Test: 5 complexes
```

Build strategy:

```text
Train: semi-balanced using all positives + 3x negatives
Validation: natural
Test: natural
```

Script:

```text
experiments/train_expanded_gat.py
```

### Combined Current + BM5 GAT Results

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------------|----------|------|----------|
| 0.1918 | 0.5121 | 0.2791 | 0.9178 |

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|-------------|--------|--------|
| True 0 | 7199 | 535 |
| True 1 | 121 | 127 |

Comparison:

| Experiment | Test Setting | Precision 1 | Recall 1 | F1 1 |
|------------|--------------|-------------|----------|------|
| Current-only strict GAT | 3 current test complexes | 0.1746 | 0.3642 | 0.2361 |
| BM5-only GAT | 2 BM5 test complexes | 0.1824 | 0.8144 | 0.2981 |
| Combined current + BM5 GAT | 5 combined test complexes | 0.1918 | 0.5121 | 0.2791 |

Interpretation:

- Combined current + BM5 improved F1 from `0.2361` to `0.2791`.
- Recall improved from `0.3642` to `0.5121`.
- The test set changed after dataset expansion, so this is not a perfect one-to-one comparison.
- Still, the result shows that dataset expansion improves GAT behavior under a larger and more diverse natural-test evaluation.

---

## 20. GAT Attention Analysis

Attention weights were extracted from the first GATConv layer of the best strict current-only GAT model.

Scripts:

```text
experiments/visualize_gat_attention.py
experiments/refine_gat_attention_analysis.py
```

Output files:

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

The full table is intentionally not committed:

```text
experiments/gat_attention_weights.csv
```

GAT attention should be interpreted as local message-passing importance rather than direct biological importance.

---

## 21. Structural 3D Error Visualization

Structural visualization files were generated for the best strict current-only model:

```text
GAT + basic 3 features
```

Script:

```text
experiments/generate_structural_error_visualization.py
```

Output directory:

```text
experiments/structural_error_visualization/
```

Generated files:

```text
1BRS_A_B_structural_errors.pml
1FSS_A_B_structural_errors.pml
3HMX_LH_AB_structural_errors.pml
structural_error_visualization_pairs.csv
structural_error_visualization_summary.md
```

### Selected Residue Pairs for Visualization

Top `10` pairs per class were selected when available.

| Case | TP Selected | FP Selected | FN Selected |
|------|-------------|-------------|-------------|
| 1BRS_A_B | 10 | 10 | 5 |
| 1FSS_A_B | 10 | 10 | 10 |
| 3HMX_LH_AB | 10 | 10 | 10 |

### PyMOL Color Legend

| Color | Meaning |
|-------|---------|
| Green | True Positive residue pairs |
| Red | False Positive residue pairs |
| Orange | False Negative residue pairs |

---

## 22. Current Best Result

The current strongest result comes from dataset expansion:

```text
Combined current + BM5 GAT
Test F1 1 = 0.2791
```

Previous best current-only result:

```text
Current-only strict GAT
Test F1 1 = 0.2361
```

Important caveat:

```text
The combined experiment uses a larger and different natural test set, so it is not a direct one-to-one replacement for the current-only result. However, it demonstrates that expanding the dataset improves model behavior under a larger natural-test setting.
```

---

## 23. Current Conclusion

- Multi-graph training works successfully.
- The original current dataset is highly imbalanced.
- GAT generally detects more positive interface/contact nodes than GCN.
- The original best current-only strict model achieved F1 = 0.2361.
- Random train balancing did not improve natural-test F1.
- Hard negative mining increased recall in one setting but did not improve F1 due to increased false positives.
- BM5-clean expansion increased usable complexes from 12 to 31.
- Positive residue pairs increased from 698 to 1,794.
- The combined current + BM5 experiment achieved F1 = 0.2791.
- Dataset expansion is currently the most effective improvement path.
- The next scientifically meaningful step is adding protein language model embeddings on top of the expanded dataset.

---

## 24. Remaining Next Experiments

Potential next steps:

1. Add protein language model embeddings such as ESM-2.
2. Evaluate ESM features on the combined current + BM5 dataset.
3. Compare GAT with non-GNN baselines.
4. Use cross-validation across complexes.
5. Add edge features based on distance or residue geometry.
6. Update the final presentation/deck with the BM5 expansion result.
