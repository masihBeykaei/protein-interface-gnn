# Protein–Protein Interface Prediction using Graph Neural Networks

## 1. Introduction

Protein–protein interactions play a central role in many biological processes, including signal transduction, immune response, enzyme regulation, and molecular recognition. Understanding which residues participate in a protein–protein interface is important for structural biology, drug discovery, and protein engineering.

In this project, the goal is to predict residue-level protein–protein interface/contact pairs using graph-based machine learning. Each protein partner is represented as a residue-level graph, and a correspondence graph is constructed between two interacting partners. Each node in the correspondence graph represents a candidate residue pair, and the learning task is to classify whether that residue pair belongs to the interface.

The project implements and evaluates Graph Neural Network models, specifically Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT), under a multi-protein experimental setting.

The project also includes feature engineering, class imbalance handling, hyperparameter tuning, visualization, GAT attention analysis, and error analysis.

---

## 2. Project Goal

The main goal of this project is to build a reproducible graph-based pipeline for protein–protein interface prediction.

The final task is formulated as a binary node classification problem on a correspondence graph:

```text
0 = non-interface / non-contact residue pair
1 = interface / contact residue pair
```

Each correspondence node represents a pair of residues:

```text
(residue_i from partner A, residue_j from partner B)
```

The model predicts whether the two residues are in contact based on geometric, topological, and biological features.

---

## 3. Related Work

This project is inspired by the paper:

```text
Graph Neural Networks for the Prediction of Protein–Protein Interfaces
```

The core idea is to model proteins as graphs and use graph neural networks to learn interaction patterns. In this project, the original idea is implemented in a simplified and reproducible way using:

- residue-level protein graphs
- correspondence graphs between protein partners
- GCN and GAT models
- multi-protein training and evaluation
- class imbalance handling
- feature engineering experiments
- attention analysis
- error analysis

---

## 4. Dataset

The dataset consists of multiple protein complexes from PDB structures. The project started with small examples such as `1BRS` and `1FSS`, then expanded to include several DBD-style complexes.

The current processed dataset includes 12 protein-complex cases:

| Case | Nodes | Positive | Negative | Positive Ratio |
|------|-------|----------|----------|----------------|
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

### Total Dataset

| Total Nodes | Total Positive | Total Negative | Positive Ratio |
|------------|----------------|----------------|----------------|
| 20,707 | 698 | 20,009 | 0.0337 |

The dataset is highly imbalanced. Only around 3.37% of correspondence nodes are positive interface/contact pairs.

---

## 5. Preprocessing Pipeline

For each protein complex, the preprocessing pipeline performs the following steps:

1. Load the PDB structure.
2. Extract selected chains for partner 1 and partner 2.
3. Extract standard amino acid residues.
4. Store all atom coordinates for each residue.
5. Extract Cα coordinates for residue-level graph construction.
6. Optionally compute residue-level accessible surface area.
7. Build intra-partner residue graphs using Cα distance.
8. Apply candidate filtering to reduce correspondence graph size.
9. Build correspondence nodes.
10. Label correspondence nodes using atom-level distance.
11. Build correspondence graph edges.
12. Save processed graph data as NumPy files.

The processed files saved for each complex are:

```text
<CASE_NAME>_corr_labels.npy
<CASE_NAME>_corr_pairs.npy
<CASE_NAME>_corr_edge_index.npy
<CASE_NAME>_corr_features.npy
```

---

## 6. Interface Labeling

A residue pair is labeled as positive if any atom from residue A is closer than 5Å to any atom from residue B.

The labeling rule is:

```text
label = 1 if min_atom_distance(residue_A, residue_B) < 5Å
label = 0 otherwise
```

This makes the label based on atom-level contact rather than only Cα distance.

---

## 7. Graph Construction

### 7.1 Residue-Level Graphs

Each protein partner is first represented as a residue-level graph.

Nodes:

```text
protein residues
```

Edges:

```text
edge exists if Cα distance between two residues < 8Å
```

This creates one graph for partner A and one graph for partner B.

### 7.2 Correspondence Graph

The correspondence graph is constructed from candidate residue pairs between the two partners.

Each correspondence node represents:

```text
(a_i, b_j)
```

where:

- `a_i` is a residue from partner A
- `b_j` is a residue from partner B

A correspondence edge is created between two correspondence nodes:

```text
(a, b) and (a', b')
```

if:

```text
a is connected to a' in partner A graph
and
b is connected to b' in partner B graph
```

This allows the model to learn local structural compatibility patterns between residue neighborhoods.

---

## 8. Candidate Filtering

Without filtering, the correspondence graph can become very large because it includes all possible residue pairs between the two partners.

To reduce graph size, candidate filtering keeps only residues that are within a certain Cα radius of the opposite partner.

Current setting:

```text
CANDIDATE_RADIUS = 12Å
```

This significantly reduces the number of correspondence nodes while keeping likely interface-region candidates.

---

## 9. Node Features

Four feature representations were tested.

### 9.1 Basic 3-Feature Representation

The basic feature vector is:

```text
[CA_distance, degree_partner_1, degree_partner_2]
```

Input dimension:

```text
3
```

### 9.2 Amino Acid One-Hot Representation

This representation adds amino acid identity for both residues.

Feature vector:

```text
[CA_distance, degree_partner_1, degree_partner_2, aa_A_onehot(20), aa_B_onehot(20)]
```

Input dimension:

```text
43
```

### 9.3 Physicochemical Representation

This representation adds compact biological properties for both residues.

Feature vector:

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

### 9.4 Basic + Accessible Surface Area Representation

This representation adds residue-level accessible surface area.

Feature vector:

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

ASA values are computed using Biopython's Shrake-Rupley implementation.

---

## 10. Models

Two graph neural network models were implemented and evaluated.

### 10.1 Graph Convolutional Network

The GCN model uses two graph convolution layers:

```text
GCNConv → ReLU → GCNConv
```

It serves as the main baseline model.

### 10.2 Graph Attention Network

The GAT model uses two attention-based graph layers:

```text
GATConv → ELU → GATConv
```

The best GAT configuration is:

```text
hidden_channels = 16
heads = 4
dropout = 0.2
```

The motivation for using GAT is that attention can assign different importance to neighboring correspondence nodes.

---

## 11. Class Imbalance Handling

Protein–protein interface prediction is highly imbalanced because only a small fraction of residue pairs are true contacts.

To handle this, a balanced loss mask is used during training.

For each batch:

- all positive nodes are included in the loss
- a random subset of negative nodes is sampled
- the full graph is still used for message passing

The loss is computed on:

```text
all positive nodes + NEGATIVE_RATIO × positive_count negative nodes
```

The default setting is:

```text
NEGATIVE_RATIO = 5
```

This allows the model to learn from positive samples more effectively without discarding graph structure during message passing.

---

## 12. Experimental Protocol

A strict graph-level train/validation/test split was used for the most reliable evaluation.

### Train Graphs

```text
1WEJ_HL_F
1JPS_HL_T
1AHW_AB_C
2FD6_HL_U
2VIS_AB_C
1MLC_AB_E
3MJ9_HL_A
```

### Validation Graphs

```text
1DQJ_AB_C
1E6J_HL_P
```

### Test Graphs

```text
1BRS_A_B
1FSS_A_B
3HMX_LH_AB
```

The validation set is used for:

- early stopping
- selecting probability threshold for class 1

The test set is used only for final evaluation.

This makes the final results more scientifically reliable than tuning hyperparameters directly on the test set.

---

## 13. Single-Graph Experiments

Initial experiments were performed on a single complex, `1BRS`.

| Model | Precision 0 | Recall 0 | F1 0 | Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------|-------------|----------|------|-------------|----------|------|----------|
| GCN | 0.9615 | 0.5981 | 0.7375 | 0.1158 | 0.6875 | 0.1982 | 0.6044 |
| GAT tuned | 1.0000 | 0.2153 | 0.3543 | 0.0889 | 1.0000 | 0.1633 | 0.2711 |

These experiments showed that accuracy is not a reliable metric under severe imbalance. Positive-class recall and F1-score are more informative.

---

## 14. Initial Multi-Graph Results

Using the initial graph-level train/test split and the basic 3-feature representation:

| Model | Precision 1 | Recall 1 | F1-score 1 | Accuracy |
|-------|-------------|----------|------------|----------|
| Multi-Graph GCN | 0.2068 | 0.3245 | 0.2526 | 0.9362 |
| Multi-Graph GAT | 0.1274 | 0.7483 | 0.2177 | 0.8215 |

The GCN model is more conservative and has higher precision.  
The GAT model is more sensitive and achieves much higher recall.

---

## 15. Negative Ratio Tuning

Different negative sampling ratios were tested.

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

For GCN, `NEGATIVE_RATIO = 5` produced the best positive-class F1-score.  
For GAT, `NEGATIVE_RATIO = 5` also produced the best positive-class F1-score, while smaller ratios produced higher recall.

---

## 16. Probability Threshold Tuning

Instead of always using `argmax`, different probability thresholds were tested.

A node is predicted as positive if:

```text
P(class 1) >= threshold
```

Best threshold results:

| Model | Best Threshold | Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------|----------------|-------------|----------|------|----------|
| GCN | 0.40 | 0.1762 | 0.4702 | 0.2563 | 0.9094 |
| GAT | 0.60 | 0.1434 | 0.4967 | 0.2226 | 0.8848 |

This experiment showed that threshold selection can improve positive-class F1-score.

However, threshold tuning on the test set is only exploratory. The stricter validation-based protocol is preferred.

---

## 17. Strict Train/Validation/Test Results

The strict experiment uses validation-based early stopping and validation-based threshold selection.

### Basic 3-Feature Results

| Model | Input Dim | Best Epoch | Best Threshold | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|-------|-----------|------------|----------------|------------------|---------------|-----------|---------------|
| GCN | 3 | 7 | 0.50 | 0.1940 | 0.1722 | 0.1825 | 0.9488 |
| GAT | 3 | 56 | 0.50 | 0.1746 | 0.3642 | 0.2361 | 0.9217 |

Under this strict protocol, GAT achieves the best positive-class F1-score.

---

## 18. Feature Engineering Results

Four feature sets were compared under the strict protocol.

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

### Interpretation

The basic 3-feature representation produced the best strict positive-class F1-score with GAT.

Amino acid one-hot features increased recall, especially for GAT, but also introduced many false positives.

Physicochemical features performed better than amino acid one-hot features for GAT, but still did not outperform the basic representation.

ASA features substantially improved GCN and produced a more precision-oriented GAT model. The ASA-based GAT result is very close to the best model, but slightly lower in F1-score.

This suggests that, under the current dataset size and model settings, simple geometric and topological features still generalize best, while structural features such as ASA are promising.

---

## 19. GAT Hyperparameter Tuning

A small GAT hyperparameter grid was tested under the strict protocol.

| Hidden | Heads | Dropout | Val F1 1 | Test F1 1 |
|--------|-------|---------|----------|-----------|
| 16 | 4 | 0.2 | 0.2571 | 0.2361 |
| 32 | 4 | 0.2 | 0.2526 | 0.1980 |
| 16 | 8 | 0.2 | 0.2531 | 0.2069 |
| 32 | 8 | 0.2 | 0.2457 | 0.1899 |
| 16 | 4 | 0.3 | 0.2534 | 0.2118 |

The best configuration remained:

```text
hidden_channels = 16
heads = 4
dropout = 0.2
```

Larger GAT models did not improve generalization, likely due to limited dataset size and class imbalance.

---

## 20. Current Best Model

The best scientifically reliable result is:

```text
Model: GAT
Feature set: Basic 3 features
Input dimension: 3
Best epoch: 56
Threshold: 0.50
```

Performance:

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------------|----------|------|----------|
| 0.1746 | 0.3642 | 0.2361 | 0.9217 |

This model achieves the highest positive-class F1-score under the strict train/validation/test protocol.

The closest alternative is:

```text
Model: GAT
Feature set: Basic + ASA
Input dimension: 5
Best epoch: 191
Threshold: 0.50
Test F1 1: 0.2338
```

---

## 21. Error Analysis

Error analysis was performed for the best model:

```text
GAT + basic 3 features
```

### Confusion Matrix

| True / Pred | Pred 0 | Pred 1 |
|-------------|--------|--------|
| True 0 | 4137 | 260 |
| True 1 | 96 | 55 |

This corresponds to:

```text
True Positives: 55
True Negatives: 4137
False Positives: 260
False Negatives: 96
```

### Per-Test-Graph Error Summary

| Case | Nodes | Positive | Negative | TP | TN | FP | FN |
|------|-------|----------|----------|----|----|----|----|
| 1BRS_A_B | 225 | 16 | 209 | 11 | 139 | 70 | 5 |
| 1FSS_A_B | 2,013 | 63 | 1,950 | 19 | 1,862 | 88 | 44 |
| 3HMX_LH_AB | 2,310 | 72 | 2,238 | 25 | 2,136 | 102 | 47 |

### Interpretation

The model produces more false positives than false negatives:

```text
False positives: 260
False negatives: 96
```

This suggests that the GAT model is sensitive to potential interface/contact patterns, but it is not highly precise.

The hardest test graph is:

```text
3HMX_LH_AB
```

It has the largest number of both false positives and false negatives.

This indicates that some protein complexes are more difficult to generalize to than others.

---

## 22. GAT Attention Analysis

Attention weights were extracted from the first GATConv layer of the best strict GAT model.

Scripts:

```text
experiments/visualize_gat_attention.py
experiments/refine_gat_attention_analysis.py
```

The refined attention analysis creates several attention subsets:

- top non-self attention edges
- top edges connected to predicted-positive nodes
- top edges connected to true-positive nodes
- top edges connected to false-positive nodes
- top edges connected to false-negative nodes
- top edges connected to FP/FN error nodes

### Interpretation

Raw top attention edges were dominated by self-loops, especially in the hardest test complex.

The refined attention analysis removes self-loop dominance and makes the attention results more interpretable by separating them into prediction and error contexts.

However, GAT attention is normalized over local neighborhoods. Therefore, attention weights should be interpreted as local message-passing importance rather than global biological importance.

---

## 23. Visualization

Several figures were generated for easier interpretation:

```text
experiments/figures/
```

Generated figures include:

- class imbalance chart
- feature-set F1 comparison
- precision/recall/F1 comparison
- GCN vs GAT strict protocol comparison
- negative ratio tuning curves
- threshold tuning curves
- GAT attention distribution
- log-scale GAT attention distribution

These plots help summarize the experimental results visually and are useful for presentations or final reporting.

---

## 24. Discussion

The experiments show that graph neural networks can learn useful patterns for protein–protein interface prediction, but the task remains challenging due to severe class imbalance and limited dataset size.

The GAT model generally detects more true interface pairs than GCN. However, this comes with increased false positives. This behavior is useful in biological discovery settings where missing true interface residues may be more harmful than producing extra candidates, but precision must be improved for practical deployment.

Feature engineering experiments showed that adding biological residue identity or physicochemical properties changes model behavior. Amino acid one-hot features increased recall, while physicochemical features were more stable. ASA features improved GCN and made GAT more precise, but did not surpass the basic GAT F1-score.

This suggests that current model performance is driven strongly by geometric proximity and graph topology. Richer biological or structural features may require larger datasets, better normalization, or more expressive models to improve generalization.

GAT hyperparameter tuning showed that increasing model size did not improve performance. This suggests that the current limitation is more likely data size, class imbalance, or feature representation rather than insufficient model capacity.

Attention analysis provided insight into local message-passing behavior. However, raw attention weights should not be interpreted directly as biological importance scores.

---

## 25. Limitations

Several limitations remain:

1. The dataset is still small for deep learning.
2. The positive class is highly underrepresented.
3. Only a limited number of protein complexes were used.
4. Labels are based on a fixed 5Å atom-distance threshold.
5. Candidate filtering may remove some potentially useful context.
6. Biological features were simple and manually defined.
7. ASA was computed directly from the available complex structures.
8. No pretrained protein language model embeddings were used.
9. The model was evaluated only on selected complexes.
10. Error analysis is not yet linked back to 3D structural visualization.
11. Attention weights provide local message-passing explanations but not direct biological importance.

---

## 26. Future Work

Possible next steps include:

1. Add more protein complexes.
2. Add residue embeddings from protein language models.
3. Analyze false positives and false negatives in 3D structure.
4. Visualize predicted interface pairs on protein structures.
5. Compare with non-GNN baselines.
6. Use cross-validation across protein complexes.
7. Improve negative sampling strategies.
8. Explore edge features based on distance or residue geometry.
9. Add structural features beyond ASA.
10. Prepare a final presentation/deck.

---

## 27. Conclusion

This project implemented a complete graph-based pipeline for residue-level protein–protein interface prediction.

The pipeline includes:

- PDB preprocessing
- residue-level graph construction
- correspondence graph generation
- interface labeling
- GCN and GAT training
- class imbalance handling
- hyperparameter tuning
- feature engineering
- accessible surface area features
- visualization
- error analysis
- GAT attention analysis

The best model under the strict train/validation/test protocol is:

```text
GAT with basic 3-feature representation
```

with:

```text
Test Precision 1 = 0.1746
Test Recall 1    = 0.3642
Test F1 1        = 0.2361
Test Accuracy    = 0.9217
```

The closest feature-engineered alternative is:

```text
GAT with basic + ASA features
Test F1 1 = 0.2338
```

The results show that GAT is more effective than GCN for detecting positive interface/contact residue pairs in this setup. However, false positives remain a major challenge.

Overall, the project demonstrates that graph neural networks are a promising approach for protein–protein interface prediction and provides a strong foundation for future improvements using larger datasets, richer structural features, pretrained protein embeddings, and structural error visualization.