# Protein–Protein Interface Prediction using Graph Neural Networks

## 1. Introduction

Protein–protein interactions play a central role in many biological processes, including signal transduction, immune response, enzyme regulation, and molecular recognition. Understanding which residues participate in a protein–protein interface is important for structural biology, drug discovery, and protein engineering.

In this project, the goal is to predict residue-level protein–protein interface/contact pairs using graph-based machine learning. Each protein partner is represented as a residue-level graph, and a correspondence graph is constructed between two interacting partners. Each node in the correspondence graph represents a candidate residue pair, and the learning task is to classify whether that residue pair belongs to the interface.

The project implements and evaluates Graph Neural Network models, specifically Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT), under a multi-protein experimental setting.

The project also includes feature engineering, class imbalance handling, hyperparameter tuning, visualization, GAT attention analysis, numerical error analysis, structural 3D error visualization, and dataset expansion using BM5-clean.

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

The core idea is to model proteins as graphs and use graph neural networks to learn interaction patterns.

In this project, the original idea is implemented in a simplified and reproducible way using:

- residue-level protein graphs
- correspondence graphs between protein partners
- GCN and GAT models
- multi-protein training and evaluation
- class imbalance handling
- feature engineering experiments
- attention analysis
- error analysis
- structural visualization
- dataset expansion using BM5-clean

---

## 4. Original Dataset

The original dataset consists of multiple protein complexes from PDB structures. The project started with small examples such as `1BRS` and `1FSS`, then expanded to include several DBD-style complexes.

The original processed dataset includes 12 protein-complex cases:

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

### Original Dataset Total

| Total Nodes | Total Positive | Total Negative | Positive Ratio |
|------------|----------------|----------------|----------------|
| 20,707 | 698 | 20,009 | 0.0337 |

The dataset is highly imbalanced. Only around 3.37% of correspondence nodes are positive interface/contact pairs.

---

## 5. BM5-Clean Dataset Expansion

To improve data diversity and increase the number of positive samples, the project was expanded using BM5-clean reference complexes.

A BM5 import pipeline was added to convert BM5 reference complexes into the project case format:

```text
case_name,pdb_id,pdb_file,partner1_chains,partner2_chains,source,split,enabled
```

BM5 structures were imported using chain `A` as partner 1 and chain `B` as partner 2.

### BM5 Screening

Imported BM5 reference complexes:

```text
29
```

Accepted after screening:

```text
19
```

Rejected:

```text
10
```

Screening criteria:

```text
positive >= 30
positive_ratio >= 0.02
candidate_nodes <= 8000
```

Accepted BM5 positive residue pairs:

```text
1,096
```

The BM5 expansion significantly increased the amount of usable training and evaluation data.

---

## 6. Combined Dataset

The original current dataset and accepted BM5-clean dataset were combined.

### Combined Dataset Size

| Source | Cases | Positive Pairs |
|--------|------:|---------------:|
| Current dataset | 12 | 698 |
| BM5-clean accepted | 19 | 1,096 |
| Combined total | 31 | 1,794 |

The combined processed dataset contains:

| Saved Nodes | Positive | Negative | Positive Ratio |
|------------:|---------:|---------:|---------------:|
| 23,028 | 1,794 | 21,234 | 0.0779 |

The combined experiment uses:

```text
Train: 22 complexes
Validation: 4 complexes
Test: 5 complexes
```

Training graphs are semi-balanced, while validation and test graphs preserve their natural class imbalance.

---

## 7. Preprocessing Pipeline

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

## 8. Interface Labeling

A residue pair is labeled as positive if any atom from residue A is closer than 5Å to any atom from residue B.

The labeling rule is:

```text
label = 1 if min_atom_distance(residue_A, residue_B) < 5Å
label = 0 otherwise
```

This makes the label based on atom-level contact rather than only Cα distance.

---

## 9. Graph Construction

### 9.1 Residue-Level Graphs

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

### 9.2 Correspondence Graph

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

## 10. Candidate Filtering

Without filtering, the correspondence graph can become very large because it includes all possible residue pairs between the two partners.

To reduce graph size, candidate filtering keeps only residues that are within a certain Cα radius of the opposite partner.

Current setting:

```text
CANDIDATE_RADIUS = 12Å
```

This significantly reduces the number of correspondence nodes while keeping likely interface-region candidates.

---

## 11. Node Features

Four feature representations were tested.

### 11.1 Basic 3-Feature Representation

```text
[CA_distance, degree_partner_1, degree_partner_2]
```

Input dimension:

```text
3
```

### 11.2 Amino Acid One-Hot Representation

```text
[CA_distance, degree_partner_1, degree_partner_2, aa_A_onehot(20), aa_B_onehot(20)]
```

Input dimension:

```text
43
```

### 11.3 Physicochemical Representation

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

### 11.4 Basic + Accessible Surface Area Representation

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

## 12. Models

Two graph neural network models were implemented and evaluated.

### 12.1 Graph Convolutional Network

The GCN model uses two graph convolution layers:

```text
GCNConv → ReLU → GCNConv
```

It serves as the main baseline model.

### 12.2 Graph Attention Network

The GAT model uses two attention-based graph layers:

```text
GATConv → ELU → GATConv
```

The best GAT configuration from the current-only hyperparameter tuning is:

```text
hidden_channels = 16
heads = 4
dropout = 0.2
```

The motivation for using GAT is that attention can assign different importance to neighboring correspondence nodes.

---

## 13. Class Imbalance Handling

Protein–protein interface prediction is highly imbalanced because only a small fraction of residue pairs are true contacts.

Several imbalance strategies were tested:

1. Balanced loss masking
2. Random train graph undersampling
3. Hard negative mining
4. Dataset expansion using BM5-clean

The original current-only best model used balanced loss masking while preserving full graph structure for message passing.

The expanded current + BM5 experiment uses semi-balanced training graphs while keeping validation and test graphs natural.

---

## 14. Experimental Protocols

### 14.1 Current-Only Strict Protocol

The current-only strict protocol uses a graph-level train/validation/test split.

Train graphs:

```text
1WEJ_HL_F
1JPS_HL_T
1AHW_AB_C
2FD6_HL_U
2VIS_AB_C
1MLC_AB_E
3MJ9_HL_A
```

Validation graphs:

```text
1DQJ_AB_C
1E6J_HL_P
```

Test graphs:

```text
1BRS_A_B
1FSS_A_B
3HMX_LH_AB
```

### 14.2 Combined Current + BM5 Protocol

The combined dataset protocol uses:

```text
Current dataset + BM5 accepted dataset
31 total complexes
22 train complexes
4 validation complexes
5 test complexes
```

Training graphs are semi-balanced using:

```text
all positive pairs + 3 × positive_count negative pairs
```

Validation and test graphs are kept natural.

---

## 15. Single-Graph Experiments

Initial experiments were performed on a single complex, `1BRS`.

| Model | Precision 0 | Recall 0 | F1 0 | Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------|-------------|----------|------|-------------|----------|------|----------|
| GCN | 0.9615 | 0.5981 | 0.7375 | 0.1158 | 0.6875 | 0.1982 | 0.6044 |
| GAT tuned | 1.0000 | 0.2153 | 0.3543 | 0.0889 | 1.0000 | 0.1633 | 0.2711 |

These experiments showed that accuracy is not a reliable metric under severe imbalance. Positive-class recall and F1-score are more informative.

---

## 16. Initial Multi-Graph Results

Using the initial graph-level train/test split and the basic 3-feature representation:

| Model | Precision 1 | Recall 1 | F1-score 1 | Accuracy |
|-------|-------------|----------|------------|----------|
| Multi-Graph GCN | 0.2068 | 0.3245 | 0.2526 | 0.9362 |
| Multi-Graph GAT | 0.1274 | 0.7483 | 0.2177 | 0.8215 |

The GCN model is more conservative and has higher precision. The GAT model is more sensitive and achieves much higher recall.

---

## 17. Negative Ratio and Threshold Tuning

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

Threshold tuning showed that validation-based threshold selection is important for positive-class F1.

---

## 18. Strict Current-Only Results

The strict experiment uses validation-based early stopping and validation-based threshold selection.

### Basic 3-Feature Results

| Model | Input Dim | Best Epoch | Best Threshold | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|-------|-----------|------------|----------------|------------------|---------------|-----------|---------------|
| GCN | 3 | 7 | 0.50 | 0.1940 | 0.1722 | 0.1825 | 0.9488 |
| GAT | 3 | 56 | 0.50 | 0.1746 | 0.3642 | 0.2361 | 0.9217 |

Under this strict current-only protocol, GAT achieves the best positive-class F1-score.

---

## 19. Feature Engineering Results

Four feature sets were compared under the strict current-only protocol.

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

---

## 20. GAT Hyperparameter Tuning

A small GAT hyperparameter grid was tested under the strict current-only protocol.

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

## 21. Current-Only Error Analysis

Error analysis was performed for the best current-only model:

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

The model produces more false positives than false negatives:

```text
False positives: 260
False negatives: 96
```

This suggests that the GAT model is sensitive to potential interface/contact patterns, but it is not highly precise.

---

## 22. Imbalance Ablation Experiments

Several class-imbalance strategies were evaluated before dataset expansion.

| Experiment | Test Setup | Precision 1 | Recall 1 | F1 1 | Accuracy | TP | FP | FN |
|------------|------------|-------------|----------|------|----------|----|----|----|
| Original strict GAT | Current-only natural test | 0.1746 | 0.3642 | 0.2361 | 0.9217 | 55 | 260 | 96 |
| Train-balanced random | Current-only natural test | 0.1338 | 0.3642 | 0.1957 | 0.9006 | 55 | 356 | 96 |
| Hard negative ratio 5 | Current-only natural test | 0.1519 | 0.4437 | 0.2264 | 0.8993 | 67 | 374 | 84 |
| Hard negative ratio 10 | Current-only natural test | 0.1489 | 0.3510 | 0.2091 | 0.9118 | 53 | 303 | 98 |

### Interpretation

Random train balancing did not improve natural-test F1.

Hard negative ratio 5 increased recall and reduced false negatives, but it also increased false positives. Therefore, F1 did not improve over the original strict GAT model.

These experiments suggest that simply removing negative nodes from training graphs can alter message-passing context and may make the model more prone to over-predicting positive pairs.

---

## 23. BM5-Only Experiment

BM5-only training and evaluation was performed after screening 19 acceptable BM5-clean complexes.

Dataset:

```text
BM5-clean accepted cases only
19 accepted complexes
1,096 positive residue pairs
```

Training strategy:

```text
Train: semi-balanced
Validation: natural
Test: natural
```

### BM5-Only GAT Result

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------------|----------|------|----------|
| 0.1824 | 0.8144 | 0.2981 | 0.8917 |

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|-------------|--------|--------|
| True 0 | 2983 | 354 |
| True 1 | 18 | 79 |

### Interpretation

The BM5-only result shows much higher recall than the current-only experiment. This suggests that dataset expansion provides more useful positive examples and improves the model's ability to detect interface/contact pairs.

However, BM5-only uses a different test set, so it should not be treated as a direct replacement for the current-only result.

---

## 24. Combined Current + BM5 Experiment

The most important expanded experiment combines both datasets.

Dataset:

```text
Current dataset + BM5 accepted dataset
31 usable complexes
23,028 saved nodes
1,794 positive residue pairs
21,234 negative residue pairs
```

Training strategy:

```text
Train: semi-balanced
Validation: natural
Test: natural
Input features: basic 3 features
Model: GAT
```

### Combined Current + BM5 GAT Result

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------------|----------|------|----------|
| 0.1918 | 0.5121 | 0.2791 | 0.9178 |

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|-------------|--------|--------|
| True 0 | 7199 | 535 |
| True 1 | 121 | 127 |

### Comparison with Previous Best

| Experiment | Test Setup | Precision 1 | Recall 1 | F1 1 |
|------------|------------|-------------|----------|------|
| Previous current-only strict GAT | 3 current test complexes | 0.1746 | 0.3642 | 0.2361 |
| BM5-only GAT | 2 BM5 test complexes | 0.1824 | 0.8144 | 0.2981 |
| Combined current + BM5 GAT | 5 combined test complexes | 0.1918 | 0.5121 | 0.2791 |

### Interpretation

The combined current + BM5 experiment achieved the strongest overall result in the project:

```text
Test F1 1 = 0.2791
```

This improves over the previous current-only strict result:

```text
Previous F1 1 = 0.2361
```

The test set changed after dataset expansion, so this is not a perfect one-to-one comparison. However, the combined result is more informative because it evaluates the model on a larger and more diverse natural test set.

This suggests that dataset expansion is currently the most effective improvement path.

---

## 25. GAT Attention Analysis

Attention weights were extracted from the first GATConv layer of the best strict current-only GAT model.

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

## 26. Structural 3D Error Visualization

Structural visualization files were generated for the best strict current-only model:

```text
GAT + basic 3 features
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

### Visualization Target

The visualization focuses on:

- TP residue pairs
- FP residue pairs
- FN residue pairs

Top pairs are selected as follows:

- TP and FP: highest predicted probability for class 1
- FN: lowest predicted probability for class 1

### Selected Pairs

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

This step connects numerical error analysis with qualitative 3D structural inspection.

---

## 27. Visualization

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

## 28. Discussion

The experiments show that graph neural networks can learn useful patterns for protein–protein interface prediction, but the task remains challenging due to severe class imbalance and limited dataset size.

The GAT model generally detects more true interface pairs than GCN. However, this comes with increased false positives. This behavior may be acceptable in discovery-oriented biological settings where missing interface residues is costly, but precision must improve for practical deployment.

Feature engineering experiments showed that adding biological residue identity or physicochemical properties changes model behavior. Amino acid one-hot features increased recall, while physicochemical features were more stable. ASA features improved GCN and made GAT more precise, but did not surpass the basic GAT F1-score in the current-only setting.

Class imbalance ablations showed that random train balancing and hard-negative graph reconstruction were not sufficient to improve natural-test F1. These methods can alter graph structure and message-passing context, increasing false positives.

The most important improvement came from dataset expansion. BM5-clean increased the number of usable complexes from 12 to 31 and increased positive residue pairs from 698 to 1,794. Under the combined current + BM5 setting, the GAT model achieved the strongest F1-score of 0.2791.

This suggests that data diversity and positive sample count are more important than simply modifying the sampling distribution on the small current-only dataset.

---

## 29. Limitations

Several limitations remain:

1. The dataset is still small compared with large-scale deep learning datasets.
2. The positive class is still highly underrepresented in natural validation/test graphs.
3. Labels are based on a fixed 5Å atom-distance threshold.
4. Candidate filtering may remove some potentially useful context.
5. Biological features were simple and manually defined.
6. ASA was computed directly from available complex structures.
7. No pretrained protein language model embeddings were used yet.
8. The combined result uses a larger but different test set, so it is not a perfect one-to-one comparison with the original current-only result.
9. Structural visualization is qualitative and requires manual inspection in PyMOL.
10. Attention weights provide local message-passing explanations but not direct biological importance.

---

## 30. Future Work

Possible next steps include:

1. Add residue embeddings from protein language models such as ESM-2.
2. Evaluate ESM features on the combined current + BM5 dataset.
3. Add more BM5-clean or other benchmark complexes.
4. Compare with non-GNN baselines.
5. Use cross-validation across protein complexes.
6. Improve negative sampling without removing important graph context.
7. Explore edge features based on distance or residue geometry.
8. Add structural features beyond ASA.
9. Update the final presentation/deck with the BM5 expansion result.

---

## 31. Conclusion

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
- structural 3D error visualization
- BM5-clean dataset expansion
- combined current + BM5 evaluation

The best current-only model under the strict train/validation/test protocol is:

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

After BM5-clean dataset expansion, the best expanded result is:

```text
Combined current + BM5 GAT
```

with:

```text
Test Precision 1 = 0.1918
Test Recall 1    = 0.5121
Test F1 1        = 0.2791
Test Accuracy    = 0.9178
```

The expanded dataset result is not a perfect direct replacement for the current-only result because the test set changed. However, it is the strongest and most meaningful result so far because it evaluates the model under a larger and more diverse natural-test setting.

Overall, the project demonstrates that graph neural networks are a promising approach for protein–protein interface prediction. The most effective improvement so far is dataset expansion, and the next major step is to incorporate pretrained protein language model embeddings on top of the expanded dataset.
