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
- Tuned GAT with full recall on 1BRS

---

## 🧬 Dataset Pipeline

For each protein complex:

1. Extract residues from PDB file
2. Build residue-level graph using Cα distance
3. Generate correspondence graph
4. Label interacting residue pairs using atomic distance (< 5Å)
5. Apply candidate filtering to reduce imbalance
6. Build node features: [Cα distance, degree in A, degree in B]

Files saved in `data/processed/`:

- `corr_labels.npy`
- `corr_pairs.npy`
- `corr_edge_index.npy`
- `corr_features.npy`

---

## 🧠 Models

### GCN (baseline)

- 2-layer GCN
- Class weighting to handle imbalance
- Precision/Recall/F1 reported for each class

### GAT (tuned)

- 2-layer GAT with 16 hidden units, 4 attention heads, dropout=0.2
- Class weighting: `[1, 10]` to emphasize positive nodes
- Achieved **full recall (1.0) on positive nodes** for 1BRS

---

## 📊 Example Results (1BRS)

| Model      | Precision 0 | Recall 0 | F1 0 | Precision 1 | Recall 1 | F1 1 | Accuracy |
|------------|-------------|----------|------|-------------|----------|------|----------|
| GCN        | 0.9615      | 0.5981   | 0.7375 | 0.1158      | 0.6875   | 0.1982 | 0.6044 |
| GAT tuned  | 1.0000      | 0.2153   | 0.3543 | 0.0889      | 1.0000   | 0.1633 | 0.2711 |

> Note: Accuracy is misleading due to strong class imbalance; recall/F1 for positive class is more informative.

---

## 🖥 Requirements

- Python 3.10
- CUDA-enabled GPU (tested on RTX 2070)
- PyTorch (CUDA 12.1)
- PyTorch Geometric
- Biopython, numpy, scikit-learn


## 📂 Project Structure
- preprocessing/   # Dataset pipeline
- training/        # Training scripts (GCN and GAT)
- models/          # GNN model definitions
- data/            # PDB raw files and processed dataset (labels, pairs, features, edge_index)


## 🔜 Next Steps
- 1.Run the pipeline on additional proteins (2PTC, 1AK4, 1FSS) to create a multi-protein dataset
- 2.Train and compare GCN vs GAT on the full dataset
- 3.Visualize attention maps for GAT and analyze false positives
- 4.Optimize node features to improve recall/precision
- 5.Prepare final results, plots, and documentation for reporting


Install:

```bash
pip install -r requirements.txt


