# Protein–Protein Interface Prediction using Graph Neural Networks

Residue-level protein–protein interface prediction using correspondence graphs and Graph Neural Networks (GCN & GAT).

This project is inspired by the paper **"Graph Neural Networks for the Prediction of Protein–Protein Interfaces"** and extends the idea using a simplified residue-level graph pipeline, Graph Attention Networks, multi-protein experiments, and biological feature engineering.

---

## 🚀 Project Overview

This project implements a graph-based pipeline for detecting protein–protein interaction interfaces.

The main idea is to represent two interacting protein partners as graphs, build a correspondence graph between their residues, and train Graph Neural Networks to classify which residue pairs belong to the protein–protein interface.

Implemented so far:

- Residue-level graph construction from PDB files
- Atomic-distance-based interface labeling
- Correspondence graph generation
- Candidate filtering to reduce graph size
- Basic node feature engineering
- Amino acid one-hot feature engineering
- Physicochemical residue feature engineering
- Single-graph GCN and GAT experiments on 1BRS
- DBD-style multi-chain complex support
- Multi-protein dataset generation from several protein complexes
- Multi-graph GCN and GAT training with balanced loss masking
- Negative sampling ratio tuning
- Probability threshold tuning
- Train/validation/test split with validation-based early stopping
- Comparison between:
  - basic 3-feature representation
  - 43-dimensional amino acid one-hot representation
  - 11-dimensional physicochemical representation

---

## 🧬 Dataset Pipeline

For each protein complex:

1. Load the PDB structure.
2. Extract residues from selected protein chains.
3. Build residue-level graphs using Cα distance.
4. Generate a correspondence graph between two protein partners.
5. Label residue-pair nodes as positive if any atom pair is closer than 5Å.
6. Apply candidate filtering to reduce graph size.
7. Build node features for each correspondence node.

A residue pair is labeled as positive if at least one atom pair between the two residues is closer than:

```text
5Å
```

---

## 🧩 Node Features

Each correspondence node represents a residue pair:

```text
(residue_i, residue_j)
```

### Basic Feature Vector

The initial feature vector was:

```text
[CA_distance, degree_partner_1, degree_partner_2]
```

Features:

- Cα distance between the two residues
- Degree of residue `i` in partner 1 graph
- Degree of residue `j` in partner 2 graph

Input dimension:

```text
3
```

### Amino Acid One-Hot Feature Vector

A biological feature engineering experiment was added by encoding amino acid identity.

Feature vector:

```text
[CA_distance, degree_partner_1, degree_partner_2, aa_A_onehot(20), aa_B_onehot(20)]
```

Input dimension:

```text
43
```

This experiment tests whether residue identity helps the model detect protein–protein interface/contact pairs.

### Physicochemical Feature Vector

A compact biological feature representation was also tested.

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

This representation aims to provide biologically meaningful residue information without using a high-dimensional sparse one-hot encoding.

---

## 📦 Processed Dataset

Raw PDB files are stored in:

```text
data/raw_pdb/
```

Processed NumPy files are generated in:

```text
data/processed/
```

For each complex, the pipeline saves:

```text
<CASE_NAME>_corr_labels.npy
<CASE_NAME>_corr_pairs.npy
<CASE_NAME>_corr_edge_index.npy
<CASE_NAME>_corr_features.npy
```

Example:

```text
1AHW_AB_C_corr_labels.npy
1AHW_AB_C_corr_features.npy
1AHW_AB_C_corr_edge_index.npy
1AHW_AB_C_corr_pairs.npy
```

---

## 🧪 Current Multi-Protein Dataset Summary

The current dataset includes original examples plus several DBD-style protein complexes.

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

The dataset is highly imbalanced, which is expected in protein–protein interface prediction.  
After adding DBD-style complexes, the number of positive samples increased significantly.

---

## 🧠 Models

### GCN Baseline

Implemented in:

```text
training/train_single_graph.py
training/train_multi_graph_gcn.py
```

Current setup:

- 2-layer GCN
- CrossEntropy loss
- Balanced loss mask for multi-graph training
- Dynamic input dimension detection
- Precision, Recall, and F1-score evaluation

### GAT

Implemented in:

```text
training/train_single_graph_gat.py
training/train_single_graph_gat_tuned.py
training/train_multi_graph_gat.py
```

Current setup:

- 2-layer GAT
- 16 hidden units
- 4 attention heads
- Dropout = 0.2
- Balanced loss mask for multi-graph training
- Dynamic input dimension detection
- Designed to improve recall on positive interface nodes

---

## 📊 Single-Graph Results on 1BRS

| Model | Precision 0 | Recall 0 | F1 0 | Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------|-------------|----------|------|-------------|----------|------|----------|
| GCN | 0.9615 | 0.5981 | 0.7375 | 0.1158 | 0.6875 | 0.1982 | 0.6044 |
| GAT tuned | 1.0000 | 0.2153 | 0.3543 | 0.0889 | 1.0000 | 0.1633 | 0.2711 |

> Accuracy is misleading due to strong class imbalance. Recall and F1-score for the positive class are more informative.

---

## 📈 Initial Multi-Graph Experimental Results

Multi-graph training was performed using 12 processed protein complex graphs.

The dataset was split by graph, meaning the model was evaluated on protein complexes that were not used during training.

### Default Test Set Comparison

Default setting:

```text
NEGATIVE_RATIO = 5
threshold = 0.50
features = basic 3 features
```

| Model | Precision 1 | Recall 1 | F1-score 1 | Accuracy |
|-------|-------------|----------|------------|----------|
| Multi-Graph GCN | 0.2068 | 0.3245 | 0.2526 | 0.9362 |
| Multi-Graph GAT | 0.1274 | 0.7483 | 0.2177 | 0.8215 |

The GCN model is more conservative and achieves a higher positive-class F1-score.  
The GAT model is more sensitive and achieves substantially higher recall for positive interface nodes.

---

## ✅ Validation-Based Early Stopping Results

A stricter experiment was added using a graph-level train/validation/test split.

The validation set is used for:

- early stopping
- selecting the probability threshold for class 1

The test set is used only for final evaluation.

Script:

```text
experiments/train_val_test_early_stopping.py
```

### Split

```text
Train:      1WEJ, 1JPS, 1AHW, 2FD6, 2VIS, 1MLC, 3MJ9
Validation: 1DQJ, 1E6J
Test:       1BRS, 1FSS, 3HMX
```

### Basic 3-Feature Results

| Model | Input Dim | Best Epoch | Best Threshold | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|-------|-----------|------------|----------------|------------------|---------------|-----------|---------------|
| GCN | 3 | 7 | 0.50 | 0.1940 | 0.1722 | 0.1825 | 0.9488 |
| GAT | 3 | 56 | 0.50 | 0.1746 | 0.3642 | 0.2361 | 0.9217 |

### Interpretation

This is the most scientifically reliable experiment for the basic feature representation because the test set is not used for early stopping or threshold selection.

Under this stricter setup:

- GAT achieves better positive-class F1-score than GCN.
- GAT achieves higher recall than GCN.
- GCN remains more conservative and has higher accuracy.
- GAT is more suitable for interface discovery when finding more true interface pairs is important.

---

## 🧬 Amino Acid One-Hot Feature Experiment

A biological feature engineering experiment was added by encoding amino acid identity.

Original feature vector:

```text
[CA_distance, degree_partner_1, degree_partner_2]
```

New feature vector:

```text
[CA_distance, degree_partner_1, degree_partner_2, aa_A_onehot(20), aa_B_onehot(20)]
```

This increases the input feature dimension from:

```text
3 → 43
```

### Strict Train/Validation/Test Results

| Feature Set | Model | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|------------|-------|------------------|---------------|-----------|---------------|
| Basic 3 features | GCN | 0.1940 | 0.1722 | 0.1825 | 0.9488 |
| Basic 3 features | GAT | 0.1746 | 0.3642 | 0.2361 | 0.9217 |
| Amino acid one-hot, 43 features | GCN | 0.1313 | 0.2781 | 0.1783 | 0.9149 |
| Amino acid one-hot, 43 features | GAT | 0.1051 | 0.7285 | 0.1836 | 0.7850 |

### Interpretation

Amino acid one-hot features increased recall, especially for GAT, but also increased false positives.

The best strict result is still achieved by the GAT model using the basic 3-feature representation:

```text
GAT, basic 3 features
Test F1 1 = 0.2361
```

The amino acid one-hot experiment is kept as a feature engineering experiment because it shows that biological residue identity affects model behavior and can be useful for recall-oriented interface discovery.

---

## 🧪 Physicochemical Feature Experiment

A compact biological feature representation was added using residue-level physicochemical properties.

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

This gives an input dimension of:

```text
11
```

### Strict Train/Validation/Test Results

| Feature Set | Model | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|------------|-------|------------------|---------------|-----------|---------------|
| Basic 3 features | GCN | 0.1940 | 0.1722 | 0.1825 | 0.9488 |
| Basic 3 features | GAT | 0.1746 | 0.3642 | 0.2361 | 0.9217 |
| Amino acid one-hot, 43 features | GCN | 0.1313 | 0.2781 | 0.1783 | 0.9149 |
| Amino acid one-hot, 43 features | GAT | 0.1051 | 0.7285 | 0.1836 | 0.7850 |
| Physicochemical, 11 features | GCN | 0.2254 | 0.1060 | 0.1441 | 0.9582 |
| Physicochemical, 11 features | GAT | 0.1566 | 0.2914 | 0.2037 | 0.9244 |

### Interpretation

Physicochemical features improved GAT compared with amino acid one-hot features in terms of positive-class F1-score:

```text
GAT one-hot F1:          0.1836
GAT physicochemical F1:  0.2037
```

However, the best strict result is still achieved by the GAT model using the basic 3-feature representation:

```text
GAT, basic 3 features
Test F1 1 = 0.2361
```

This suggests that compact biological features are more stable than sparse one-hot features in this setup, but the current dataset still generalizes best with simple geometric/topological features.

---

## ⚖️ Handling Class Imbalance

Protein–protein interface prediction is naturally imbalanced because only a small fraction of residue pairs are true interface/contact pairs.

To address this issue, multi-graph training uses a balanced loss mask:

- All positive nodes are included in the loss.
- A random subset of negative nodes is sampled.
- The full graph is still used for message passing.

The loss is computed using:

```text
all positive nodes + NEGATIVE_RATIO × positive_count negative nodes
```

---

## 🧪 Negative Ratio Tuning

Different negative sampling ratios were tested for both GCN and GAT.

Script:

```text
experiments/tune_negative_ratio.py
```

Results with basic 3-feature representation:

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

### Current Best Settings by F1

| Model | Best Negative Ratio | Precision 1 | Recall 1 | F1 1 |
|-------|---------------------|-------------|----------|------|
| GCN | 5 | 0.2059 | 0.3245 | 0.2519 |
| GAT | 5 | 0.1274 | 0.7483 | 0.2177 |

Tuning results are saved in:

```text
experiments/negative_ratio_tuning_results.md
experiments/negative_ratio_tuning_results.csv
```

---

## 🎚️ Probability Threshold Tuning

Instead of always using `argmax`, different probability thresholds were tested.

A node is predicted as positive if:

```text
P(class 1) >= threshold
```

Script:

```text
experiments/tune_probability_threshold.py
```

### Best Thresholds by Positive-Class F1

Results with basic 3-feature representation:

| Model | Best Threshold | Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------|----------------|-------------|----------|------|----------|
| GCN | 0.40 | 0.1762 | 0.4702 | 0.2563 | 0.9094 |
| GAT | 0.60 | 0.1434 | 0.4967 | 0.2226 | 0.8848 |

Threshold tuning results are saved in:

```text
experiments/threshold_tuning_results.md
experiments/threshold_tuning_results.csv
```

> Note: This threshold tuning was performed as an exploratory experiment. The stricter validation-based experiment should be considered the more scientifically reliable result.

---

## 🏆 Current Best Scientifically Reliable Result

The most reliable result currently comes from the train/validation/test experiment with validation-based early stopping.

| Feature Set | Model | Best Epoch | Threshold | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|------------|-------|------------|-----------|------------------|---------------|-----------|---------------|
| Basic 3 features | GCN | 7 | 0.50 | 0.1940 | 0.1722 | 0.1825 | 0.9488 |
| Basic 3 features | GAT | 56 | 0.50 | 0.1746 | 0.3642 | 0.2361 | 0.9217 |
| Amino acid one-hot, 43 features | GCN | 11 | 0.40 | 0.1313 | 0.2781 | 0.1783 | 0.9149 |
| Amino acid one-hot, 43 features | GAT | 34 | 0.40 | 0.1051 | 0.7285 | 0.1836 | 0.7850 |
| Physicochemical, 11 features | GCN | 24 | 0.60 | 0.2254 | 0.1060 | 0.1441 | 0.9582 |
| Physicochemical, 11 features | GAT | 22 | 0.50 | 0.1566 | 0.2914 | 0.2037 | 0.9244 |

Current best model under the strict protocol:

```text
GAT with basic 3-feature representation
```

Best strict positive-class F1-score:

```text
0.2361
```

---

## 🖥 Requirements

- Python 3.10
- CUDA-enabled GPU
- PyTorch
- PyTorch Geometric
- Biopython
- NumPy
- scikit-learn

Tested on:

```text
GPU: NVIDIA GeForce RTX 2070
CUDA: available
Python: 3.10
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 1. Build the multi-protein dataset

```bash
python preprocessing/build_multi_protein_dataset.py
```

This generates correspondence graph labels, node features, residue pairs, and edge indices for all configured complexes.

The current preprocessing script supports feature modes:

```text
basic
aa_onehot
physicochemical
```

The feature mode can be changed in:

```text
preprocessing/build_multi_protein_dataset.py
```

### 2. Train single-graph GCN baseline

```bash
python training/train_single_graph.py
```

### 3. Train tuned single-graph GAT

```bash
python training/train_single_graph_gat_tuned.py
```

### 4. Train multi-graph GCN

```bash
python training/train_multi_graph_gcn.py
```

### 5. Train multi-graph GAT

```bash
python training/train_multi_graph_gat.py
```

### 6. Run negative ratio tuning

```bash
python experiments/tune_negative_ratio.py
```

### 7. Run probability threshold tuning

```bash
python experiments/tune_probability_threshold.py
```

### 8. Run train/validation/test early stopping experiment

```bash
python experiments/train_val_test_early_stopping.py
```

---

## 📂 Project Structure

```text
protein-interface-gnn/
│
├── data/
│   ├── raw_pdb/          # Raw PDB files
│   ├── processed/        # Generated graph data (.npy files)
│   └── graphs/
│
├── preprocessing/
│   ├── read_pdb.py
│   ├── build_residue_graph.py
│   ├── build_correspondence_graph.py
│   └── build_multi_protein_dataset.py
│
├── models/
│   ├── gcn_model.py
│   └── gat_model.py
│
├── training/
│   ├── train_single_graph.py
│   ├── train_single_graph_gat.py
│   ├── train_single_graph_gat_tuned.py
│   ├── train_multi_graph_gcn.py
│   └── train_multi_graph_gat.py
│
├── experiments/
│   ├── results_summary.md
│   ├── tune_negative_ratio.py
│   ├── negative_ratio_tuning_results.md
│   ├── negative_ratio_tuning_results.csv
│   ├── tune_probability_threshold.py
│   ├── threshold_tuning_results.md
│   ├── threshold_tuning_results.csv
│   ├── train_val_test_early_stopping.py
│   ├── early_stopping_results.md
│   ├── early_stopping_results.csv
│   ├── early_stopping_results_aa_onehot.md
│   ├── early_stopping_results_aa_onehot.csv
│   ├── early_stopping_results_physicochemical.md
│   └── early_stopping_results_physicochemical.csv
│
├── utils/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔬 Current Interpretation

Current results show different behaviors between feature sets and models:

- GCN is more conservative.
- GAT detects more true positive interface/contact nodes.
- In the strict train/validation/test setup, GAT with the basic 3-feature representation achieves the best positive-class F1-score.
- Amino acid one-hot features increase recall, especially for GAT, but also increase false positives.
- Physicochemical features are more compact and perform better than one-hot features for GAT, but still do not outperform the basic 3-feature representation.
- Under the current dataset and model settings, simple geometric/topological features generalize best in terms of positive-class F1-score.
- Biological features are still valuable because they reveal useful recall-oriented behavior and provide a foundation for future feature engineering.

---

## 🔜 Next Steps

- Add accessible surface area if feasible.
- Visualize GAT attention weights.
- Tune GAT hidden dimensions and attention heads.
- Analyze false positives and false negatives.
- Add plots for:
  - class imbalance
  - precision vs recall
  - F1-score comparison
  - negative ratio tuning results
  - threshold tuning results
  - early stopping results
  - feature engineering comparison
- Prepare final report and presentation.

---

## 📌 Notes

- Raw PDB files and processed `.npy` files should not be committed to GitHub.
- The repository contains the reproducible preprocessing and training pipeline.
- The processed dataset can be regenerated by running the preprocessing scripts.