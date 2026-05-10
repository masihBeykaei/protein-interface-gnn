# Protein–Protein Interface Prediction using Graph Neural Networks

Residue-level protein–protein interface prediction using correspondence graphs and Graph Neural Networks (GCN & GAT).

This project is inspired by the paper **"Graph Neural Networks for the Prediction of Protein–Protein Interfaces"** and extends the idea using a simplified residue-level graph pipeline and Graph Attention Networks.

---

## 🚀 Project Overview

This project implements a graph-based pipeline for detecting protein–protein interaction interfaces.

The main idea is to represent two interacting protein partners as graphs, build a correspondence graph between their residues, and train Graph Neural Networks to classify which residue pairs belong to the protein–protein interface.

Implemented so far:

- Residue-level graph construction from PDB files
- Atomic-distance-based interface labeling
- Correspondence graph generation
- Candidate filtering to reduce graph size
- Node feature engineering
- Single-graph GCN and GAT experiments on 1BRS
- DBD-style multi-chain complex support
- Multi-protein dataset generation from several protein complexes

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

### Node Features

Each correspondence node represents a residue pair `(residue_i, residue_j)`.

Current node features:

- Cα distance between the two residues
- Degree of residue `i` in partner 1 graph
- Degree of residue `j` in partner 2 graph

Feature vector:

```text
[CA_distance, degree_partner_1, degree_partner_2]
```

---

## 📦 Processed Dataset

The processed dataset is generated from PDB files stored in:

```text
data/raw_pdb/
```

The generated NumPy files are stored in:

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

The dataset currently includes original examples plus several DBD-style protein complexes.

| Case | Nodes | Positive | Negative | Positive Ratio |
|------|-------|----------|----------|----------------|
| 1BRS_A_B | 225 | 16 | 209 | 0.0711 |
| 1FSS_A_B | 2013 | 63 | 1950 | 0.0313 |
| 1AHW_AB_C | 2142 | 73 | 2069 | 0.0341 |
| 1DQJ_AB_C | 1978 | 71 | 1907 | 0.0359 |
| 1E6J_HL_P | 1131 | 51 | 1080 | 0.0451 |
| 1JPS_HL_T | 2184 | 71 | 2113 | 0.0325 |
| 1MLC_AB_E | 1540 | 54 | 1486 | 0.0351 |
| 1WEJ_HL_F | 1026 | 41 | 985 | 0.0400 |
| 2FD6_HL_U | 1147 | 47 | 1100 | 0.0410 |
| 2VIS_AB_C | 1728 | 51 | 1677 | 0.0295 |
| 3HMX_LH_AB | 2310 | 72 | 2238 | 0.0312 |
| 3MJ9_HL_A | 3283 | 88 | 3195 | 0.0268 |

### Total

| Total Nodes | Total Positive | Total Negative | Positive Ratio |
|------------|----------------|----------------|----------------|
| 20707 | 698 | 20009 | 0.0337 |

The dataset is still imbalanced, which is expected in protein–protein interface prediction. However, the number of positive samples increased significantly after adding DBD-style complexes.

---

## 🧠 Models

### GCN Baseline

Implemented in:

```text
training/train_single_graph.py
```

Current setup:

- 2-layer GCN
- CrossEntropy loss
- Class weighting for imbalance
- Precision, Recall, and F1-score evaluation

### GAT Tuned

Implemented in:

```text
training/train_single_graph_gat_tuned.py
```

Current setup:

- 2-layer GAT
- 16 hidden units
- 4 attention heads
- Dropout = 0.2
- Class weighting `[1.0, 10.0]`
- Designed to improve recall on positive interface nodes

---

## 📊 Single-Graph Results on 1BRS

| Model | Precision 0 | Recall 0 | F1 0 | Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------|-------------|----------|------|-------------|----------|------|----------|
| GCN | 0.9615 | 0.5981 | 0.7375 | 0.1158 | 0.6875 | 0.1982 | 0.6044 |
| GAT tuned | 1.0000 | 0.2153 | 0.3543 | 0.0889 | 1.0000 | 0.1633 | 0.2711 |

> Accuracy is misleading due to strong class imbalance. Recall and F1-score for the positive class are more informative.

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

### 2. Train single-graph GCN baseline

```bash
python training/train_single_graph.py
```

### 3. Train tuned GAT on single graph

```bash
python training/train_single_graph_gat_tuned.py
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
│   └── train_single_graph_gat_tuned.py
│
├── utils/
│
├── experiments/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔜 Next Steps

- Implement multi-graph GCN training using all processed complexes
- Implement multi-graph GAT training
- Use balanced loss masks or class weighting to handle imbalance
- Compare GCN vs GAT on the full multi-protein dataset
- Add plots for precision, recall, F1-score, and class imbalance
- Analyze false positives and false negatives
- Optionally visualize GAT attention weights
- Improve node features using amino acid type, hydrophobicity, charge, and ASA
- Prepare final project report and presentation

---

## 📌 Notes

- Raw PDB files and processed `.npy` files should not be committed to GitHub.
- The repository contains the reproducible pipeline and training scripts.
- The processed dataset can be regenerated by running the preprocessing scripts.