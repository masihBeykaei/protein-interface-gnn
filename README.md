# Protein–Protein Interface Prediction using Graph Neural Networks

Residue-level protein–protein interface prediction using correspondence graphs and Graph Neural Networks (GCN & GAT).

---

## 🚀 Project Overview

This project implements a graph-based approach for detecting protein–protein interaction interfaces.

Steps implemented so far:

- Residue-level graph construction from PDB files
- Atomic-distance-based interface labeling
- Correspondence graph generation
- Candidate filtering to reduce graph size
- GPU training with PyTorch
- Baseline GCN model with class weighting

---

## 🧬 Dataset Pipeline

For each protein complex:

1. Extract residues from PDB file
2. Build residue-level graph using Cα distance
3. Generate correspondence graph
4. Label interacting residue pairs using atomic distance (< 5Å)
5. Apply candidate filtering to reduce imbalance

---

## 🧠 Model

Current baseline:

- 2-layer GCN
- CrossEntropy with class weighting
- Evaluated using Precision, Recall, F1-score

---

## 📊 Current Results (1BRS Example)

| Metric | Class 0 | Class 1 |
|--------|---------|---------|
| Precision | 0.95 | 0.11 |
| Recall | 0.66 | 0.56 |
| F1-score | 0.77 | 0.18 |

Shows significant improvement in recall after applying class weighting.

---

## 🖥 Requirements

- Python 3.10
- CUDA-enabled GPU (tested on RTX 2070)
- PyTorch (CUDA 12.1)
- PyTorch Geometric



## 📂 Project Structure
- preprocessing/   # Dataset pipeline
- training/        # Training scripts
- models/          # GNN models
- data/            # PDB and processed data


## 🔜 Next Steps
- Implement Graph Attention Network (GAT)
- Compare GCN vs GAT
- Extend training to multiple protein complexes


Install:

```bash
pip install -r requirements.txt


