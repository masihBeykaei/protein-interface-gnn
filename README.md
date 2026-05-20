# Protein–Protein Interface Prediction using Graph Neural Networks

Residue-level protein–protein interface prediction using correspondence graphs, Graph Neural Networks, BM5-clean dataset expansion, and protein language model embeddings.

This project is inspired by the paper **"Graph Neural Networks for the Prediction of Protein–Protein Interfaces"** and implements a reproducible pipeline for predicting protein–protein interface/contact residue pairs.

---

## Project Overview

The project represents protein partners as residue-level graphs, constructs a correspondence graph between residues of two interacting partners, and trains Graph Neural Networks to classify whether each residue pair belongs to the protein–protein interface.

Implemented components:

- PDB structure parsing with Biopython
- Residue-level graph construction using C-alpha distance
- Correspondence graph construction between two protein partners
- Atom-distance-based interface/contact labeling
- Candidate filtering to reduce graph size
- Multi-chain partner support
- GCN and GAT models
- Multi-graph training
- Validation-based early stopping
- Validation-based probability threshold selection
- Feature engineering experiments
- Accessible surface area features
- Class imbalance ablation experiments
- Hard negative mining experiments
- BM5-clean dataset expansion
- ESM-2 protein language model embeddings
- PCA-reduced ESM pair features
- GAT attention analysis
- Error analysis
- Structural 3D error visualization with PyMOL scripts
- Final documentation and presentation-ready results

---

## Task Definition

Each correspondence node represents a candidate residue pair:

```text
(residue_i from partner A, residue_j from partner B)
```

The binary classification target is:

```text
0 = non-interface / non-contact residue pair
1 = interface / contact residue pair
```

A residue pair is labeled as positive when any atom pair between the two residues is closer than:

```text
5 Å
```

---

## Dataset Construction

For each complex:

1. Load the PDB file.
2. Extract selected chains for partner 1 and partner 2.
3. Keep standard amino acid residues with C-alpha atoms.
4. Build intra-partner residue graphs using C-alpha distance.
5. Apply candidate filtering using a 12 Å C-alpha radius.
6. Generate correspondence nodes between candidate residues.
7. Label correspondence nodes using atom-level distance.
8. Build correspondence graph edges from intra-partner graph topology.
9. Save graph arrays as NumPy files.

Processed files for each case:

```text
<CASE_NAME>_corr_features.npy
<CASE_NAME>_corr_labels.npy
<CASE_NAME>_corr_pairs.npy
<CASE_NAME>_corr_edge_index.npy
```

---

## Original Current Dataset

The original current dataset contains 12 protein complexes.

| Case | Nodes | Positive | Negative | Positive Ratio |
|------|------:|---------:|---------:|---------------:|
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

Total:

| Nodes | Positive | Negative | Positive Ratio |
|------:|---------:|---------:|---------------:|
| 20,707 | 698 | 20,009 | 0.0337 |

The original dataset is highly imbalanced, which is expected for protein–protein interface prediction.

---

## BM5-Clean Dataset Expansion

To increase data diversity and the number of positive examples, BM5-clean reference complexes were imported and screened.

Imported BM5 reference complexes:

```text
29
```

Accepted after screening:

```text
19
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

---

## Combined Current + BM5 Dataset

The current dataset and accepted BM5-clean cases were combined.

| Source | Cases | Positive Pairs |
|--------|------:|---------------:|
| Current dataset | 12 | 698 |
| BM5-clean accepted | 19 | 1,096 |
| Combined total | 31 | 1,794 |

Processed combined dataset:

| Saved Nodes | Positive | Negative | Positive Ratio |
|------------:|---------:|---------:|---------------:|
| 23,028 | 1,794 | 21,234 | 0.0779 |

Split:

| Split | Cases |
|-------|------:|
| Train | 22 |
| Validation | 4 |
| Test | 5 |

Training graphs are semi-balanced using all positive pairs and 3x sampled negatives. Validation and test graphs preserve their natural class imbalance.

---

## Feature Sets

### Basic 3 Features

```text
[CA_distance, degree_partner_1, degree_partner_2]
```

Input dimension:

```text
3
```

### Amino Acid One-Hot Features

```text
[CA_distance, degree_partner_1, degree_partner_2, aa_A_onehot(20), aa_B_onehot(20)]
```

Input dimension:

```text
43
```

### Physicochemical Features

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

### Basic + ASA Features

```text
[CA_distance, degree_partner_1, degree_partner_2, ASA_A, ASA_B]
```

Input dimension:

```text
5
```

### ESM-2 PCA Pair Features

Per-residue ESM-2 embeddings were extracted using:

```text
facebook/esm2_t6_8M_UR50D
```

For each residue pair, the raw ESM pair representation is:

```text
[ESM_A, ESM_B, abs(ESM_A - ESM_B)]
```

The raw pair dimension is:

```text
960
```

PCA is fitted only on training pairs to avoid validation/test leakage.

Final ESM feature vector:

```text
[basic_3_features, PCA(ESM_A, ESM_B, abs(ESM_A - ESM_B))]
```

Tested PCA settings:

| PCA Components | Final Input Dim |
|---------------:|----------------:|
| 64 | 67 |
| 32 | 35 |
| 16 | 19 |

---

## Models

### GCN

A two-layer Graph Convolutional Network:

```text
GCNConv → ReLU → GCNConv
```

### GAT

A two-layer Graph Attention Network:

```text
GATConv → ELU → GATConv
```

Best current GAT configuration:

```text
hidden_channels = 16
heads = 4
dropout = 0.2
```

---

## Main Results

### Strict Current-Only Experiment

| Model | Features | Input Dim | Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------|----------|----------:|------------:|---------:|-----:|---------:|
| GCN | Basic | 3 | 0.1940 | 0.1722 | 0.1825 | 0.9488 |
| GAT | Basic | 3 | 0.1746 | 0.3642 | 0.2361 | 0.9217 |

The best strict current-only model was GAT with basic 3 features.

---

### Combined Current + BM5 Basic Result

| Dataset | Model | Features | Precision 1 | Recall 1 | F1 1 | Accuracy |
|---------|-------|----------|------------:|---------:|-----:|---------:|
| Combined Current + BM5 | GAT | Basic 3 | 0.1918 | 0.5121 | 0.2791 | 0.9178 |

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|-------------|-------:|-------:|
| True 0 | 7,199 | 535 |
| True 1 | 121 | 127 |

BM5-clean expansion improved performance under a larger natural-test setup.

---

### ESM-2 PCA Results

| Experiment | Input Dim | Precision 1 | Recall 1 | F1 1 | Accuracy | TP | FP | FN |
|------------|----------:|------------:|---------:|-----:|---------:|---:|---:|---:|
| Combined basic 3 features | 3 | 0.1918 | 0.5121 | 0.2791 | 0.9178 | 127 | 535 | 121 |
| ESM-2 PCA64 + GAT | 67 | 0.1169 | 0.7823 | 0.2034 | 0.8096 | 194 | 1,466 | 54 |
| ESM-2 PCA32 + GAT | 35 | 0.1953 | 0.5726 | 0.2913 | 0.9134 | 142 | 585 | 106 |
| ESM-2 PCA16 + GAT | 19 | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 132 | 523 | 116 |

The best overall result is:

```text
Combined Current + BM5 + ESM-2 PCA16 + GAT
Test F1 1 = 0.2924
```

---

## Interpretation of ESM-2 Results

ESM-2 PCA64 produced very high recall but too many false positives.

ESM-2 PCA32 improved recall and F1 compared with the basic combined baseline.

ESM-2 PCA16 gave the best trade-off:

- highest F1-score
- highest precision among combined experiments
- lower false positives than the basic combined baseline
- improved recall compared with the basic combined baseline

The final improvement is:

```text
F1: 0.2791 → 0.2924
Precision: 0.1918 → 0.2015
Recall: 0.5121 → 0.5323
```

---

## Imbalance Ablation Experiments

| Experiment | Test Setup | Precision 1 | Recall 1 | F1 1 | Accuracy | TP | FP | FN |
|------------|------------|------------:|---------:|-----:|---------:|---:|---:|---:|
| Original strict GAT | Current-only natural test | 0.1746 | 0.3642 | 0.2361 | 0.9217 | 55 | 260 | 96 |
| Train-balanced random | Current-only natural test | 0.1338 | 0.3642 | 0.1957 | 0.9006 | 55 | 356 | 96 |
| Hard negative ratio 5 | Current-only natural test | 0.1519 | 0.4437 | 0.2264 | 0.8993 | 67 | 374 | 84 |
| Hard negative ratio 10 | Current-only natural test | 0.1489 | 0.3510 | 0.2091 | 0.9118 | 53 | 303 | 98 |

Random train balancing and hard negative graph reconstruction did not improve current-only natural-test F1. Dataset expansion and ESM-2 PCA features were more effective.

---

## GAT Attention and Error Analysis

Error analysis and attention analysis were performed for the best strict current-only GAT model.

Important files:

```text
experiments/error_analysis_gat_basic_summary.md
experiments/gat_attention_summary.md
experiments/gat_attention_refined_summary.md
```

The best strict current-only model produced:

```text
TP = 55
FP = 260
FN = 96
TN = 4137
```

Attention analysis is interpreted as local message-passing importance, not direct biological importance.

---

## Structural 3D Error Visualization

Structural visualization scripts were generated for the best strict current-only GAT model.

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

PyMOL color legend:

| Color | Meaning |
|-------|---------|
| Green | True Positive residue pairs |
| Red | False Positive residue pairs |
| Orange | False Negative residue pairs |

---


---

## Final F1-Improvement Ablation Study

After reaching the best ESM-2 PCA16 result, several additional experiments were performed to check whether the positive-class F1-score could be improved further.

### ESM-PCA16 Hyperparameter Tuning

A dedicated GAT hyperparameter tuning experiment was run for the ESM-PCA16 feature set. The best configuration selected by validation F1 did not improve the test F1 over the existing PCA16 baseline.

| Experiment | Best Validation-Based Test F1 |
|------------|------------------------------:|
| ESM-PCA16 GAT hyperparameter tuning | 0.2077 |

Interpretation:

```text
Changing GAT capacity, heads, dropout, learning rate, and weight decay did not improve the final test F1.
```

### Train Negative-Ratio Tuning for ESM-PCA16

| Experiment | Precision 1 | Recall 1 | F1 1 | Accuracy | FP | TP |
|------------|------------:|---------:|-----:|---------:|---:|---:|
| ESM-PCA16 ratio 3 / best | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 523 | 132 |
| ESM-PCA16 ratio 4 | 0.2600 | 0.3145 | 0.2847 | 0.9509 | 222 | 78 |
| ESM-PCA16 ratio 5 | 0.2507 | 0.3468 | 0.2910 | 0.9475 | 257 | 86 |

Interpretation:

```text
Increasing the negative ratio made the model more precision-oriented and reduced false positives, but recall dropped too much to improve F1.
```

### ESM Pair-Feature Variant Ablations

The original full-pair ESM representation was:

```text
[ESM_A, ESM_B, abs(ESM_A - ESM_B)]
```

Additional variants were tested:

| Experiment | Raw ESM Pair Feature | Input Dim | Precision 1 | Recall 1 | F1 1 | Accuracy | FP | TP |
|------------|----------------------|----------:|------------:|---------:|-----:|---------:|---:|---:|
| Full pair PCA16 / best | `[A, B, absdiff]` | 19 | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 523 | 132 |
| absdiff PCA16 | `abs(A - B)` | 19 | 0.1924 | 0.5081 | 0.2791 | 0.9184 | 529 | 126 |
| absdiff PCA32 | `abs(A - B)` | 35 | 0.2003 | 0.5081 | 0.2873 | 0.9217 | 503 | 126 |
| product PCA16 | `A * B` | 19 | 0.1877 | 0.5806 | 0.2837 | 0.9089 | 623 | 144 |
| absdiff_product PCA16 | `[abs(A - B), A * B]` | 19 | 0.2232 | 0.4113 | 0.2894 | 0.9372 | 355 | 102 |

Interpretation:

- `product PCA16` produced the highest recall and TP count, but false positives increased.
- `absdiff_product PCA16` improved precision and reduced false positives, but recall dropped.
- The full-pair PCA16 representation remained the best overall F1 trade-off.

Final result after all improvement attempts:

```text
Best final model remains:
Combined Current + BM5 + Full Pair ESM-2 PCA16 + GAT
F1 = 0.2924
```


## Current Best Model

```text
Model: GAT
Dataset: Combined Current + BM5
Features: Basic 3 features + ESM-2 PCA16 pair features
Input dimension: 19
Test Precision 1: 0.2015
Test Recall 1: 0.5323
Test F1 1: 0.2924
Test Accuracy: 0.9199
```

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|-------------|-------:|-------:|
| True 0 | 7,211 | 523 |
| True 1 | 116 | 132 |

---

## How to Run

### 1. Build the original dataset

```bash
python preprocessing/build_multi_protein_dataset.py
```

### 2. Import BM5-clean reference complexes

```bash
python preprocessing/import_bm5_reference_complexes.py --bm5_haddock_ready_dir <PATH_TO_BM5_CLEAN>/HADDOCK-ready --out_pdb_dir data/raw_pdb_expanded_bm5 --out_cases data/cases/bm5_reference_cases.csv
```

### 3. Screen BM5 complexes

```bash
python preprocessing/screen_candidate_complexes.py --cases data/cases/bm5_reference_cases.csv --auto_from_dir data/no_auto --output data/cases/bm5_screening_results.csv --accepted_output data/cases/bm5_cases_accepted.csv --min_positive 30 --min_positive_ratio 0.02 --max_candidate_nodes 8000
```

### 4. Combine current and BM5 cases

```bash
python combine_current_bm5_cases.py
```

### 5. Build combined dataset

```bash
python preprocessing/build_train_balanced_natural_eval_dataset.py --cases data/cases/combined_current_bm5_cases.csv --out_dir data/processed_combined_current_bm5_train_balanced_natural_test --train_negative_ratio 3
```

### 6. Train combined basic GAT

```bash
python experiments/train_expanded_gat.py --processed_dir data/processed_combined_current_bm5_train_balanced_natural_test --output_csv experiments/combined_current_bm5_gat_results.csv --output_md experiments/combined_current_bm5_gat_results.md
```

### 7. Extract ESM-2 embeddings

```bash
python preprocessing/extract_esm2_embeddings.py --cases data/cases/combined_current_bm5_cases.csv --out_dir data/esm2_embeddings --model_name facebook/esm2_t6_8M_UR50D
```

### 8. Build ESM-2 PCA16 dataset

```bash
python preprocessing/build_esm2_pca_dataset.py --source_processed_dir data/processed_combined_current_bm5_train_balanced_natural_test --embedding_dir data/esm2_embeddings --out_dir data/processed_combined_current_bm5_esm2_pca16 --pca_components 16
```

### 9. Train ESM-2 PCA16 GAT

```bash
python experiments/train_expanded_gat.py --processed_dir data/processed_combined_current_bm5_esm2_pca16 --output_csv experiments/combined_current_bm5_esm2_pca16_gat_results.csv --output_md experiments/combined_current_bm5_esm2_pca16_gat_results.md
```

---

## Requirements

- Python 3.10
- PyTorch
- PyTorch Geometric
- Biopython
- NumPy
- scikit-learn
- matplotlib
- transformers
- CUDA-enabled GPU recommended
- PyMOL optional for structural visualization

Install dependencies:

```bash
pip install -r requirements.txt
pip install transformers
```

---

## Files and Directories

```text
data/cases/
preprocessing/
training/
models/
experiments/
experiments/figures/
experiments/structural_error_visualization/
report/
```

Large generated directories should not be committed:

```text
data/raw_pdb_expanded_bm5/
data/esm2_embeddings/
data/processed_* /
```

---

## Final Conclusion

The project demonstrates that graph neural networks can learn meaningful patterns for protein–protein interface prediction from correspondence graphs.

The strongest improvements came from:

1. expanding the dataset using BM5-clean
2. adding PCA-reduced ESM-2 protein language model features

The final best model is:

```text
Combined Current + BM5 + Full Pair ESM-2 PCA16 + GAT
F1 = 0.2924
```

Final ablation experiments showed that hyperparameter tuning, higher negative ratios, and ESM feature variants did not improve beyond this score.
