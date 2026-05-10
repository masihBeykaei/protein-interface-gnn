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
- Multi-graph GCN and GAT training with balanced loss masking
- Negative sampling ratio tuning experiments

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

Each correspondence node represents a residue pair:

```text
(residue_i, residue_j)
```

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
- Designed to improve recall on positive interface nodes

---

## 📊 Single-Graph Results on 1BRS

| Model | Precision 0 | Recall 0 | F1 0 | Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------|-------------|----------|------|-------------|----------|------|----------|
| GCN | 0.9615 | 0.5981 | 0.7375 | 0.1158 | 0.6875 | 0.1982 | 0.6044 |
| GAT tuned | 1.0000 | 0.2153 | 0.3543 | 0.0889 | 1.0000 | 0.1633 | 0.2711 |

> Accuracy is misleading due to strong class imbalance. Recall and F1-score for the positive class are more informative.

---

## 📈 Multi-Graph Experimental Results

Multi-graph training was performed using 12 processed protein complex graphs.

The dataset was split by graph, meaning the model was evaluated on protein complexes that were not used during training.

### Test Set Comparison

| Model | Precision 1 | Recall 1 | F1-score 1 | Accuracy |
|-------|-------------|----------|------------|----------|
| Multi-Graph GCN | 0.2068 | 0.3245 | 0.2526 | 0.9362 |
| Multi-Graph GAT | 0.1274 | 0.7483 | 0.2177 | 0.8215 |

The GCN model is more conservative and achieves a higher positive-class F1-score.  
The GAT model is more sensitive and achieves substantially higher recall for positive interface nodes.

Full experimental details are available in:

```text
experiments/results_summary.md
```

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

Results:

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

### Current Best Settings

| Model | Best Ratio by F1 | Precision 1 | Recall 1 | F1 1 |
|-------|------------------|-------------|----------|------|
| GCN | 5 | 0.2059 | 0.3245 | 0.2519 |
| GAT | 5 | 0.1274 | 0.7483 | 0.2177 |

### Interpretation

- GCN with `NEGATIVE_RATIO = 5` gives the best positive-class F1-score.
- GCN with `NEGATIVE_RATIO = 3` gives a better recall while keeping almost the same F1-score.
- GAT with `NEGATIVE_RATIO = 5` gives the best positive-class F1-score among GAT settings.
- GAT with `NEGATIVE_RATIO = 2` or `3` gives very high recall and may be useful for recall-oriented interface discovery.

Tuning results are saved in:

```text
experiments/negative_ratio_tuning_results.md
experiments/negative_ratio_tuning_results.csv
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
│   └── negative_ratio_tuning_results.csv
│
├── utils/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔬 Current Interpretation

Current results show different behaviors between GCN and GAT:

- GCN is more conservative.
- GCN achieves the best positive-class F1-score on the current test split.
- GAT detects more true positive interface nodes.
- GAT achieves much higher recall but produces more false positives.
- Negative sampling ratio strongly affects the precision-recall trade-off.
- For biological interface discovery, high recall can be useful because missing true interface residues may be more harmful than producing extra candidates.

---

## 🔜 Next Steps

- Add validation split and early stopping.
- Tune model probability thresholds instead of using only `argmax`.
- Improve node features using:
  - amino acid type
  - hydrophobicity
  - charge
  - accessible surface area
- Tune GAT hidden dimensions and attention heads.
- Visualize GAT attention weights.
- Analyze false positives and false negatives.
- Add plots for:
  - class imbalance
  - precision vs recall
  - F1-score comparison
  - negative ratio tuning results
- Prepare final project report and presentation.

---

## 📌 Notes

- Raw PDB files and processed `.npy` files should not be committed to GitHub.
- The repository contains the reproducible preprocessing and training pipeline.
- The processed dataset can be regenerated by running the preprocessing scripts.