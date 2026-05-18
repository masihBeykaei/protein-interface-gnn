# Protein–Protein Interface Prediction using Graph Neural Networks

Residue-level protein–protein interface prediction using correspondence graphs and Graph Neural Networks (GCN & GAT).

This project is inspired by the paper **"Graph Neural Networks for the Prediction of Protein–Protein Interfaces"** and extends the idea using a simplified residue-level graph pipeline, Graph Attention Networks, multi-protein experiments, biological feature engineering, result visualization, attention analysis, hyperparameter tuning, error analysis, and structural 3D visualization.

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
- Accessible surface area feature engineering
- Single-graph GCN and GAT experiments on 1BRS
- DBD-style multi-chain complex support
- Multi-protein dataset generation from several protein complexes
- Multi-graph GCN and GAT training with balanced loss masking
- Negative sampling ratio tuning
- Probability threshold tuning
- Train/validation/test split with validation-based early stopping
- GAT hyperparameter tuning
- GAT attention weight extraction and refinement
- Error analysis for the best strict GAT model
- Structural 3D error visualization using PyMOL scripts
- Experiment result visualization plots
- Final project report

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
8. Save processed graph arrays as NumPy files.

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

```text
[CA_distance, degree_partner_1, degree_partner_2]
```

Input dimension:

```text
3
```

Features:

- Cα distance between the two residues
- Degree of residue `i` in partner 1 graph
- Degree of residue `j` in partner 2 graph

### Amino Acid One-Hot Feature Vector

```text
[CA_distance, degree_partner_1, degree_partner_2, aa_A_onehot(20), aa_B_onehot(20)]
```

Input dimension:

```text
43
```

This experiment tests whether residue identity helps the model detect protein–protein interface/contact pairs.

### Physicochemical Feature Vector

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

This representation provides compact biological residue information without using a sparse one-hot encoding.

### Basic + ASA Feature Vector

Accessible surface area was added as a structural residue-level feature.

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

ASA values are computed at residue level using Biopython's Shrake-Rupley implementation.

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

Current best GAT setup:

```text
hidden_channels = 16
heads = 4
dropout = 0.2
```

The GAT model uses:

- 2 GAT layers
- Balanced loss mask
- Dynamic input dimension detection
- Validation-based early stopping
- Validation-based threshold selection

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

The GCN model is more conservative.  
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

Under this stricter setup:

- GAT achieves better positive-class F1-score than GCN.
- GAT achieves higher recall than GCN.
- GCN remains more conservative and has higher accuracy.

---

## 🧬 Amino Acid One-Hot Feature Experiment

Feature vector:

```text
[CA_distance, degree_partner_1, degree_partner_2, aa_A_onehot(20), aa_B_onehot(20)]
```

Input dimension:

```text
43
```

| Feature Set | Model | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|------------|-------|------------------|---------------|-----------|---------------|
| Amino acid one-hot, 43 features | GCN | 0.1313 | 0.2781 | 0.1783 | 0.9149 |
| Amino acid one-hot, 43 features | GAT | 0.1051 | 0.7285 | 0.1836 | 0.7850 |

Amino acid one-hot features increased recall, especially for GAT, but also increased false positives.

---

## 🧪 Physicochemical Feature Experiment

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

| Feature Set | Model | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|------------|-------|------------------|---------------|-----------|---------------|
| Physicochemical, 11 features | GCN | 0.2254 | 0.1060 | 0.1441 | 0.9582 |
| Physicochemical, 11 features | GAT | 0.1566 | 0.2914 | 0.2037 | 0.9244 |

Physicochemical features improved GAT compared with amino acid one-hot features, but still did not outperform the basic 3-feature representation.

---

## 🧫 Accessible Surface Area Feature Experiment

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

| Feature Set | Model | Best Epoch | Threshold | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|------------|-------|------------|-----------|------------------|---------------|-----------|---------------|
| Basic + ASA, 5 features | GCN | 35 | 0.60 | 0.1887 | 0.2649 | 0.2204 | 0.9378 |
| Basic + ASA, 5 features | GAT | 191 | 0.50 | 0.2184 | 0.2517 | 0.2338 | 0.9453 |

ASA substantially improved GCN compared with the basic 3-feature GCN.

For GAT, ASA increased precision but reduced recall. Its F1-score is very close to the best basic GAT model.

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

---

## 🎚️ Probability Threshold Tuning

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

This threshold tuning was exploratory.  
The stricter validation-based experiment should be considered more scientifically reliable.

---

## 🔧 GAT Hyperparameter Tuning

Script:

```text
experiments/tune_gat_hyperparameters.py
```

Search space:

| Hidden Channels | Heads | Dropout |
|----------------|-------|---------|
| 16 | 4 | 0.2 |
| 32 | 4 | 0.2 |
| 16 | 8 | 0.2 |
| 32 | 8 | 0.2 |
| 16 | 4 | 0.3 |

Results:

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

Larger GAT configurations did not improve generalization.

---

## 📊 Experiment Figures

Generated plots are stored in:

```text
experiments/figures/
```

Figures include:

| File | Description |
|------|-------------|
| `class_imbalance.png` | Positive vs negative correspondence node counts |
| `feature_set_f1_comparison.png` | Positive-class F1 comparison across feature sets |
| `feature_set_precision_recall_f1.png` | Precision, recall, and F1 comparison across feature sets |
| `best_strict_gcn_vs_gat.png` | GCN vs GAT under the strict train/validation/test protocol |
| `negative_ratio_tuning_gcn.png` | Negative sampling ratio tuning for GCN |
| `negative_ratio_tuning_gat.png` | Negative sampling ratio tuning for GAT |
| `threshold_tuning_gcn.png` | Probability threshold tuning for GCN |
| `threshold_tuning_gat.png` | Probability threshold tuning for GAT |
| `gat_attention_distribution.png` | Distribution of GAT attention weights |
| `gat_attention_distribution_log.png` | Log-scale distribution of GAT attention weights |

---

## 🔎 Error Analysis

Error analysis was performed for the current best strict protocol model:

```text
Model: GAT
Features: basic 3 features
Input dimension: 3
Split: train/validation/test
Threshold selected on validation set
```

Script:

```text
experiments/analyze_errors.py
```

Output files:

```text
experiments/error_analysis_gat_basic.csv
experiments/error_analysis_gat_basic_summary.md
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

This indicates that the best GAT model is sensitive but not yet highly precise.

---

## 👁️ GAT Attention Analysis

Attention weights were extracted from the first GATConv layer of the best strict model.

Scripts:

```text
experiments/visualize_gat_attention.py
experiments/refine_gat_attention_analysis.py
```

Output files include:

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

The full attention table is large and intentionally not tracked in Git:

```text
experiments/gat_attention_weights.csv
```

### Interpretation

Raw top attention edges were dominated by self-loops.  
The refined analysis removes self-loop dominance and separates attention edges by prediction context.

GAT attention should be interpreted as local message-passing importance, not as a direct global biological importance score.

---

## 🧬 Structural 3D Error Visualization

Structural visualization files were generated for the best strict model:

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

### Visualization Target

For each test complex, the script selects top residue pairs from:

- TP: true positive pairs
- FP: false positive pairs
- FN: false negative pairs

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

Example command:

```bash
pymol experiments/structural_error_visualization/1BRS_A_B_structural_errors.pml
```

This step enables qualitative inspection of prediction errors in 3D structural context.

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
| Basic + ASA, 5 features | GCN | 35 | 0.60 | 0.1887 | 0.2649 | 0.2204 | 0.9378 |
| Basic + ASA, 5 features | GAT | 191 | 0.50 | 0.2184 | 0.2517 | 0.2338 | 0.9453 |

Current best model:

```text
GAT with basic 3-feature representation
```

Best strict positive-class F1-score:

```text
0.2361
```

The ASA-based GAT model is very close:

```text
GAT with basic + ASA features
Test F1 1 = 0.2338
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
- matplotlib
- PyMOL, optional, only for viewing generated `.pml` files

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

Supported feature modes:

```text
basic
aa_onehot
physicochemical
basic_asa
physicochemical_asa
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

### 9. Tune GAT hyperparameters

```bash
python experiments/tune_gat_hyperparameters.py
```

### 10. Generate experiment plots

```bash
python experiments/plot_results.py
```

### 11. Run error analysis

```bash
python experiments/analyze_errors.py
```

### 12. Run GAT attention analysis

```bash
python experiments/visualize_gat_attention.py
python experiments/refine_gat_attention_analysis.py
```

### 13. Generate structural 3D error visualization files

```bash
python experiments/generate_structural_error_visualization.py
```

Open a generated PyMOL script:

```bash
pymol experiments/structural_error_visualization/1BRS_A_B_structural_errors.pml
```

---

## 📂 Project Structure

```text
protein-interface-gnn/
│
├── data/
│   ├── raw_pdb/
│   ├── processed/
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
│   ├── plot_results.py
│   ├── analyze_errors.py
│   ├── visualize_gat_attention.py
│   ├── refine_gat_attention_analysis.py
│   ├── generate_structural_error_visualization.py
│   ├── tune_gat_hyperparameters.py
│   ├── tune_negative_ratio.py
│   ├── tune_probability_threshold.py
│   ├── train_val_test_early_stopping.py
│   ├── gat_hyperparameter_tuning_results.md
│   ├── gat_attention_summary.md
│   ├── gat_attention_refined_summary.md
│   ├── error_analysis_gat_basic_summary.md
│   ├── early_stopping_results_basic_asa.md
│   ├── structural_error_visualization/
│   │   ├── 1BRS_A_B_structural_errors.pml
│   │   ├── 1FSS_A_B_structural_errors.pml
│   │   ├── 3HMX_LH_AB_structural_errors.pml
│   │   ├── structural_error_visualization_pairs.csv
│   │   └── structural_error_visualization_summary.md
│   └── figures/
│       ├── class_imbalance.png
│       ├── feature_set_f1_comparison.png
│       ├── feature_set_precision_recall_f1.png
│       ├── best_strict_gcn_vs_gat.png
│       ├── negative_ratio_tuning_gcn.png
│       ├── negative_ratio_tuning_gat.png
│       ├── threshold_tuning_gcn.png
│       ├── threshold_tuning_gat.png
│       ├── gat_attention_distribution.png
│       └── gat_attention_distribution_log.png
│
├── report/
│   └── final_report.md
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
- ASA features substantially improve GCN and make GAT more precise.
- Basic + ASA GAT is very close to the best model, but the best F1 remains GAT with basic 3 features.
- Larger GAT configurations do not improve generalization on the current dataset.
- Attention analysis is useful for local message-passing interpretation, but raw attention should not be treated as direct biological importance.
- Error analysis shows that the best GAT model produces more false positives than false negatives.
- Structural visualization files enable qualitative inspection of TP, FP, and FN residue pairs in 3D.

---

## 🔜 Remaining Next Steps

Most planned next steps have been completed. Remaining useful extensions include:

- Try protein language model embeddings.
- Expand the dataset with more protein complexes.
- Prepare a final presentation/deck.
- Optionally inspect generated PyMOL visualizations manually.

---

## 📌 Notes

- Raw PDB files and processed `.npy` files should not be committed to GitHub.
- Large generated attention files should not be committed to GitHub.
- PyMOL is optional and only needed for viewing `.pml` visualization files.
- The repository contains the reproducible preprocessing and training pipeline.
- The processed dataset can be regenerated by running the preprocessing scripts.
