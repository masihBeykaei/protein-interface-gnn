# Protein–Protein Interface Prediction using Graph Neural Networks

Residue-level protein–protein interface prediction using correspondence graphs and Graph Neural Networks (GCN & GAT).

This project is inspired by the paper **"Graph Neural Networks for the Prediction of Protein–Protein Interfaces"** and extends the idea using a simplified residue-level graph pipeline, Graph Attention Networks, multi-protein experiments, biological feature engineering, class-imbalance analysis, attention analysis, structural error visualization, dataset expansion using BM5-clean, and expanded natural-test evaluation.

---

## 🚀 Project Overview

This project implements a graph-based pipeline for detecting protein–protein interaction interfaces.

The main idea is to represent two interacting protein partners as residue-level graphs, build a correspondence graph between their residues, and train Graph Neural Networks to classify which residue pairs belong to the protein–protein interface.

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
- Class-imbalance ablation experiments
- Hard negative mining experiments
- GAT attention weight extraction and refinement
- Error analysis for the best strict GAT model
- Structural 3D error visualization using PyMOL scripts
- Dataset expansion using BM5-clean
- Combined current + BM5 evaluation
- Experiment result visualization plots
- Final project report and presentation deck

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

### Basic + ASA Feature Vector

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

## 📦 Processed Dataset Files

Raw PDB files are stored in:

```text
data/raw_pdb/
```

Expanded BM5 raw PDB files are stored locally in:

```text
data/raw_pdb_expanded_bm5/
```

Processed NumPy files are generated in:

```text
data/processed/
data/processed_* /
```

For each complex, the pipeline saves:

```text
<CASE_NAME>_corr_labels.npy
<CASE_NAME>_corr_pairs.npy
<CASE_NAME>_corr_edge_index.npy
<CASE_NAME>_corr_features.npy
```

---

## 🧪 Original Current Dataset Summary

The original current dataset includes original examples plus several DBD-style protein complexes.

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

### Original Current Dataset Total

| Total Nodes | Total Positive | Total Negative | Positive Ratio |
|------------|----------------|----------------|----------------|
| 20,707 | 698 | 20,009 | 0.0337 |

The dataset is highly imbalanced, which is expected in protein–protein interface prediction.

---

## 🧬 BM5-Clean Dataset Expansion

To increase dataset size and improve generalization, BM5-clean reference complexes were imported and screened.

Imported BM5 reference complexes:

```text
29
```

Accepted after screening:

```text
19
```

Screening filters:

```text
positive >= 30
positive_ratio >= 0.02
candidate_nodes <= 8000
```

BM5 accepted positives:

```text
1,096
```

The combined current + BM5 dataset contains:

```text
31 usable complexes
1,794 positive residue pairs
```

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
experiments/train_expanded_gat.py
```

Current best GAT setup:

```text
hidden_channels = 16
heads = 4
dropout = 0.2
```

The GAT model uses:

- 2 GAT layers
- Balanced loss mask or balanced dataset construction depending on experiment
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

## ✅ Strict Current-Only Train/Validation/Test Results

A stricter experiment was added using a graph-level train/validation/test split.

The validation set is used for:

- early stopping
- selecting the probability threshold for class 1

The test set is used only for final evaluation.

Script:

```text
experiments/train_val_test_early_stopping.py
```

### Current-Only Split

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

Under this stricter current-only setup:

- GAT achieves better positive-class F1-score than GCN.
- GAT achieves higher recall than GCN.
- GCN remains more conservative and has higher accuracy.

---

## 🧬 Feature Engineering Results

| Feature Set | Model | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|------------|-------|------------------|---------------|-----------|---------------|
| Basic 3 features | GCN | 0.1940 | 0.1722 | 0.1825 | 0.9488 |
| Basic 3 features | GAT | 0.1746 | 0.3642 | 0.2361 | 0.9217 |
| Amino acid one-hot, 43 features | GCN | 0.1313 | 0.2781 | 0.1783 | 0.9149 |
| Amino acid one-hot, 43 features | GAT | 0.1051 | 0.7285 | 0.1836 | 0.7850 |
| Physicochemical, 11 features | GCN | 0.2254 | 0.1060 | 0.1441 | 0.9582 |
| Physicochemical, 11 features | GAT | 0.1566 | 0.2914 | 0.2037 | 0.9244 |
| Basic + ASA, 5 features | GCN | 0.1887 | 0.2649 | 0.2204 | 0.9378 |
| Basic + ASA, 5 features | GAT | 0.2184 | 0.2517 | 0.2338 | 0.9453 |

ASA substantially improved GCN compared with the basic 3-feature GCN.  
For GAT, ASA increased precision but reduced recall. Its F1-score is very close to the best basic GAT model.

---

## ⚖️ Handling Class Imbalance

Protein–protein interface prediction is naturally imbalanced because only a small fraction of residue pairs are true interface/contact pairs.

Several imbalance strategies were tested:

1. Balanced loss mask
2. Random train graph undersampling
3. Hard negative mining
4. Dataset expansion using BM5-clean

The original best current-only model used balanced loss masking while keeping the full graph structure for message passing.

---

## 🧪 Imbalance Ablation Experiments

These experiments were designed to test whether changing the training graph distribution improves natural-test generalization.

| Experiment | Test Setup | Precision 1 | Recall 1 | F1 1 | Accuracy | TP | FP | FN |
|------------|------------|-------------|----------|------|----------|----|----|----|
| Original strict GAT | Current-only natural test | 0.1746 | 0.3642 | 0.2361 | 0.9217 | 55 | 260 | 96 |
| Train-balanced random | Current-only natural test | 0.1338 | 0.3642 | 0.1957 | 0.9006 | 55 | 356 | 96 |
| Hard negative ratio 5 | Current-only natural test | 0.1519 | 0.4437 | 0.2264 | 0.8993 | 67 | 374 | 84 |
| Hard negative ratio 10 | Current-only natural test | 0.1489 | 0.3510 | 0.2091 | 0.9118 | 53 | 303 | 98 |

Interpretation:

- Random train balancing did not improve natural-test F1.
- Hard negative ratio 5 increased recall, but also increased false positives.
- Hard negative ratio 10 reduced false positives compared with ratio 5, but did not improve F1.
- Removing negative nodes from the graph can change message-passing context and may increase over-prediction of positive pairs.

---

## 🧬 BM5-Only Expansion Experiment

Script:

```text
experiments/train_expanded_gat.py
```

Dataset:

```text
BM5-clean accepted cases only
19 accepted complexes
1,096 positive residue pairs
```

Build strategy:

```text
Train: balanced/semi-balanced
Validation: natural
Test: natural
```

### BM5-Only GAT Result

| Dataset | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|---------|------------------|---------------|-----------|---------------|
| BM5-only | 0.1824 | 0.8144 | 0.2981 | 0.8917 |

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|-------------|--------|--------|
| True 0 | 2983 | 354 |
| True 1 | 18 | 79 |

This result shows that with more training data from BM5-clean, GAT can achieve much higher recall on a natural test set.

---

## 🏆 Combined Current + BM5 Result

The most important expanded experiment combines the original current dataset with accepted BM5-clean complexes.

Dataset:

```text
Current dataset + BM5 accepted dataset
31 usable complexes
23,028 saved nodes
1,794 positive residue pairs
21,234 negative residue pairs
Natural test set: 5 complexes
Natural test positives: 248
```

Training setup:

```text
Train: balanced/semi-balanced
Validation: natural
Test: natural
Input features: basic 3 features
Model: GAT
```

### Combined Current + BM5 GAT Result

| Dataset | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |
|---------|------------------|---------------|-----------|---------------|
| Current-only strict GAT | 0.1746 | 0.3642 | 0.2361 | 0.9217 |
| BM5-only GAT | 0.1824 | 0.8144 | 0.2981 | 0.8917 |
| Combined Current + BM5 GAT | 0.1918 | 0.5121 | 0.2791 | 0.9178 |

Confusion matrix for combined current + BM5:

| True / Pred | Pred 0 | Pred 1 |
|-------------|--------|--------|
| True 0 | 7199 | 535 |
| True 1 | 121 | 127 |

Interpretation:

- The combined dataset improves positive-class F1 from `0.2361` to `0.2791`.
- Recall improves from `0.3642` to `0.5121`.
- The comparison is not perfectly one-to-one because the test set changed after dataset expansion.
- However, the result shows that dataset expansion improves model behavior under a larger and more diverse natural-test evaluation setup.

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

Larger GAT configurations did not improve generalization on the current-only dataset.

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

Error analysis was performed for the best current-only strict protocol model:

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

### Current-Only Test Metrics

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------------|----------|------|----------|
| 0.1746 | 0.3642 | 0.2361 | 0.9217 |

### Confusion Matrix

| True / Pred | Pred 0 | Pred 1 |
|-------------|--------|--------|
| True 0 | 4137 | 260 |
| True 1 | 96 | 55 |

The model produces more false positives than false negatives, indicating that the best current-only GAT model is sensitive but not yet highly precise.

---

## 👁️ GAT Attention Analysis

Attention weights were extracted from the first GATConv layer of the best strict current-only model.

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

GAT attention should be interpreted as local message-passing importance, not as a direct global biological importance score.

---

## 🧬 Structural 3D Error Visualization

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

### Visualization Target

For each test complex, the script selects top residue pairs from:

- TP: true positive pairs
- FP: false positive pairs
- FN: false negative pairs

Top pairs are selected as follows:

- TP and FP: highest predicted probability for class 1
- FN: lowest predicted probability for class 1

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

---

## 🏆 Current Best Result

The current strongest result comes from dataset expansion using BM5-clean:

```text
Combined Current + BM5 GAT
Input features: basic 3 features
Test F1 1 = 0.2791
```

This improves over the previous current-only strict GAT result:

```text
Previous current-only strict GAT F1 1 = 0.2361
```

Important note:

```text
The expanded dataset uses a larger and different natural test set, so the comparison is not a perfect one-to-one replacement. However, it demonstrates that expanding the dataset improves GAT behavior under a larger natural-test evaluation setup.
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

### 1. Build the original multi-protein dataset

```bash
python preprocessing/build_multi_protein_dataset.py
```

### 2. Train current-only strict model

```bash
python experiments/train_val_test_early_stopping.py
```

### 3. Import BM5-clean reference complexes

```bash
python preprocessing/import_bm5_reference_complexes.py --bm5_haddock_ready_dir <PATH_TO_BM5_CLEAN>/HADDOCK-ready --out_pdb_dir data/raw_pdb_expanded_bm5 --out_cases data/cases/bm5_reference_cases.csv
```

### 4. Screen BM5 candidate complexes

```bash
python preprocessing/screen_candidate_complexes.py --cases data/cases/bm5_reference_cases.csv --auto_from_dir data/no_auto --output data/cases/bm5_screening_results.csv --accepted_output data/cases/bm5_cases_accepted.csv --min_positive 30 --min_positive_ratio 0.02 --max_candidate_nodes 8000
```

### 5. Combine current and BM5 cases

```bash
python combine_current_bm5_cases.py
```

### 6. Build combined train-balanced / natural-test dataset

```bash
python preprocessing/build_train_balanced_natural_eval_dataset.py --cases data/cases/combined_current_bm5_cases.csv --out_dir data/processed_combined_current_bm5_train_balanced_natural_test --train_negative_ratio 3
```

### 7. Train GAT on the combined dataset

```bash
python experiments/train_expanded_gat.py --processed_dir data/processed_combined_current_bm5_train_balanced_natural_test --output_csv experiments/combined_current_bm5_gat_results.csv --output_md experiments/combined_current_bm5_gat_results.md
```

### 8. Run analysis and visualization

```bash
python experiments/analyze_errors.py
python experiments/visualize_gat_attention.py
python experiments/refine_gat_attention_analysis.py
python experiments/generate_structural_error_visualization.py
```

---

## 📂 Project Structure

```text
protein-interface-gnn/
│
├── data/
│   ├── cases/
│   │   ├── expanded_cases.csv
│   │   ├── bm5_reference_cases.csv
│   │   ├── bm5_screening_results.csv
│   │   ├── bm5_cases_accepted.csv
│   │   └── combined_current_bm5_cases.csv
│   ├── raw_pdb/
│   ├── raw_pdb_expanded_bm5/        # ignored by Git
│   ├── processed/
│   └── processed_* /                # ignored by Git
│
├── preprocessing/
│   ├── build_multi_protein_dataset.py
│   ├── screen_candidate_complexes.py
│   ├── import_bm5_reference_complexes.py
│   ├── build_train_balanced_natural_eval_dataset.py
│   └── build_hard_negative_train_dataset.py
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
│   ├── train_val_test_early_stopping.py
│   ├── train_expanded_gat.py
│   ├── combined_current_bm5_gat_results.md
│   ├── bm5_train_balanced_natural_test_gat_results.md
│   ├── hard_negative_r5_gat_results.md
│   ├── hard_negative_r10_gat_results.md
│   ├── train_balanced_natural_test_gat_results.md
│   ├── analyze_errors.py
│   ├── visualize_gat_attention.py
│   ├── refine_gat_attention_analysis.py
│   ├── generate_structural_error_visualization.py
│   └── figures/
│
├── report/
│   └── final_report.md
│
├── combine_current_bm5_cases.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔬 Current Interpretation

Current results show several important behaviors:

- GCN is more conservative.
- GAT detects more true positive interface/contact nodes.
- In the original strict current-only setup, GAT with the basic 3-feature representation achieves the best positive-class F1-score.
- Amino acid one-hot features increase recall, especially for GAT, but also increase false positives.
- Physicochemical features are more compact and perform better than one-hot features for GAT, but still do not outperform the basic 3-feature representation.
- ASA features substantially improve GCN and make GAT more precise.
- Random train balancing and hard negative graph reconstruction did not improve natural-test F1 on the current-only dataset.
- BM5-clean dataset expansion substantially increased the amount of positive training data.
- The combined current + BM5 experiment achieved the best overall positive-class F1-score so far.
- Attention analysis is useful for local message-passing interpretation, but raw attention should not be treated as direct biological importance.
- Structural visualization files enable qualitative inspection of TP, FP, and FN residue pairs in 3D.

---

## 🔜 Remaining Next Steps

Most planned next steps have been completed. Remaining useful extensions include:

- Add protein language model embeddings such as ESM-2.
- Compare GAT with non-GNN baselines on the expanded dataset.
- Run cross-validation across complexes.
- Add edge features based on distance or residue geometry.
- Prepare updated final presentation/deck using the BM5 expansion result.

---

## 📌 Notes

- Raw PDB files and processed `.npy` files should not be committed to GitHub.
- Large generated attention files should not be committed to GitHub.
- PyMOL is optional and only needed for viewing `.pml` visualization files.
- The repository contains the reproducible preprocessing and training pipeline.
- The processed dataset can be regenerated by running the preprocessing scripts.
