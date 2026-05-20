# Protein–Protein Interface Prediction using Graph Neural Networks

Residue-level protein–protein interface prediction using correspondence graphs, Graph Neural Networks, BM5-clean dataset expansion, ESM-2 protein language model embeddings, and multiple GNN architecture comparisons.

This project is inspired by the paper **"Graph Neural Networks for the Prediction of Protein–Protein Interfaces"** and implements a reproducible pipeline for predicting protein–protein interface/contact residue pairs from protein complex structures.

---

## Project Overview

The project represents two interacting protein partners as residue-level graphs, constructs a correspondence graph between residues of the two partners, and trains Graph Neural Networks to classify whether each residue pair belongs to the protein–protein interface.

Implemented components:

- PDB structure parsing with Biopython
- Residue-level protein graph construction using C-alpha distances
- Correspondence graph construction between two protein partners
- Atom-distance-based interface/contact labeling
- Candidate filtering to reduce graph size
- Multi-chain partner support
- GCN, GAT, GATv2, GraphSAGE, GIN, and TransformerConv experiments
- Multi-graph training
- Validation-based early stopping
- Validation-based probability threshold selection
- Feature engineering experiments
- Accessible surface area feature experiments
- Class imbalance ablation experiments
- Hard negative mining experiments
- BM5-clean dataset expansion
- ESM-2 protein language model embedding extraction
- PCA-reduced ESM pair features
- ESM pair-feature variant ablations
- ESM-PCA16 hyperparameter tuning
- Train negative-ratio tuning for ESM-PCA16
- Multi-seed TransformerConv evaluation
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

This makes the task a highly imbalanced binary node-classification problem on a correspondence graph.

---

## Dataset Construction

For each protein complex:

1. Load the PDB file.
2. Extract selected chains for partner 1 and partner 2.
3. Keep standard amino acid residues with C-alpha atoms.
4. Build residue-level intra-partner graphs using C-alpha distance.
5. Apply candidate filtering using a 12 Å C-alpha radius.
6. Generate correspondence nodes between candidate residues.
7. Label correspondence nodes using atom-level inter-residue distance.
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

Training graphs are semi-balanced using all positive pairs and sampled negatives. Validation and test graphs preserve their natural class imbalance.

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

### ESM-2 Full Pair PCA Features

Per-residue ESM-2 embeddings were extracted using:

```text
facebook/esm2_t6_8M_UR50D
```

For each residue pair, the full raw ESM pair representation is:

```text
[ESM_A, ESM_B, abs(ESM_A - ESM_B)]
```

The raw pair dimension is:

```text
960
```

PCA is fitted only on training pairs to avoid validation/test leakage.

Final full-pair ESM feature vector:

```text
[basic_3_features, PCA(ESM_A, ESM_B, abs(ESM_A - ESM_B))]
```

Tested PCA settings:

| PCA Components | Final Input Dim |
|---------------:|----------------:|
| 64 | 67 |
| 32 | 35 |
| 16 | 19 |

### ESM-2 Variant Features

Additional ESM pair-feature variants were tested:

| Variant | Raw Pair Feature | Raw Dim |
|---------|------------------|--------:|
| absdiff | `abs(ESM_A - ESM_B)` | 320 |
| product | `ESM_A * ESM_B` | 320 |
| absdiff_product | `[abs(ESM_A - ESM_B), ESM_A * ESM_B]` | 640 |

---

## Models

The following GNN architectures were tested on the final ESM-PCA16 dataset:

| Model | Description |
|-------|-------------|
| GCN | Classic graph convolution baseline |
| GAT | Graph attention network |
| GATv2 | More flexible attention mechanism than standard GAT |
| GraphSAGE | Neighborhood aggregation model |
| GIN | Expressive MLP-based message-passing model |
| TransformerConv | Transformer-style graph attention with beta gating |

---

## Main Experimental Progression

### 1. Strict Current-Only Baseline

| Model | Features | Input Dim | Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------|----------|----------:|------------:|---------:|-----:|---------:|
| GCN | Basic | 3 | 0.1940 | 0.1722 | 0.1825 | 0.9488 |
| GAT | Basic | 3 | 0.1746 | 0.3642 | 0.2361 | 0.9217 |

The best strict current-only model was GAT with basic 3 features.

---

### 2. Combined Current + BM5 Basic Result

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

### 3. ESM-2 Full Pair PCA Results with GAT

| Experiment | Input Dim | Precision 1 | Recall 1 | F1 1 | Accuracy | TP | FP | FN |
|------------|----------:|------------:|---------:|-----:|---------:|---:|---:|---:|
| Combined basic 3 features | 3 | 0.1918 | 0.5121 | 0.2791 | 0.9178 | 127 | 535 | 121 |
| ESM-2 full pair PCA64 + GAT | 67 | 0.1169 | 0.7823 | 0.2034 | 0.8096 | 194 | 1,466 | 54 |
| ESM-2 full pair PCA32 + GAT | 35 | 0.1953 | 0.5726 | 0.2913 | 0.9134 | 142 | 585 | 106 |
| ESM-2 full pair PCA16 + GAT | 19 | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 132 | 523 | 116 |

Best GAT result:

```text
Combined Current + BM5 + Full Pair ESM-2 PCA16 + GAT
F1 = 0.2924
```

---

## F1 Improvement Ablations

After obtaining the GAT + ESM-PCA16 result, several additional experiments were performed to check whether F1 could be improved further.

### ESM-PCA16 Hyperparameter Tuning

A dedicated GAT hyperparameter tuning experiment was run for ESM-PCA16. The best model selected by validation F1 did not improve the test F1 over the existing PCA16 baseline.

| Experiment | Best Validation-Based Test F1 |
|------------|------------------------------:|
| ESM-PCA16 GAT hyperparameter tuning | 0.2077 |

Conclusion:

```text
Changing GAT capacity and regularization alone did not improve the final test F1.
```

### Train Negative Ratio Tuning

| Experiment | Precision 1 | Recall 1 | F1 1 | Accuracy | FP | TP |
|------------|------------:|---------:|-----:|---------:|---:|---:|
| ESM-PCA16 ratio 3 / GAT best | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 523 | 132 |
| ESM-PCA16 ratio 4 | 0.2600 | 0.3145 | 0.2847 | 0.9509 | 222 | 78 |
| ESM-PCA16 ratio 5 | 0.2507 | 0.3468 | 0.2910 | 0.9475 | 257 | 86 |

Higher negative ratios improved precision and reduced false positives, but recall dropped too much to improve F1.

### ESM Pair-Feature Variant Ablations

| Experiment | Input Dim | Precision 1 | Recall 1 | F1 1 | Accuracy | FP | TP |
|------------|----------:|------------:|---------:|-----:|---------:|---:|---:|
| Full pair PCA16 / GAT best | 19 | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 523 | 132 |
| absdiff PCA16 | 19 | 0.1924 | 0.5081 | 0.2791 | 0.9184 | 529 | 126 |
| absdiff PCA32 | 35 | 0.2003 | 0.5081 | 0.2873 | 0.9217 | 503 | 126 |
| product PCA16 | 19 | 0.1877 | 0.5806 | 0.2837 | 0.9089 | 623 | 144 |
| absdiff_product PCA16 | 19 | 0.2232 | 0.4113 | 0.2894 | 0.9372 | 355 | 102 |

Interpretation:

- `product PCA16` increased recall and true positives but also increased false positives.
- `absdiff_product PCA16` improved precision and reduced false positives but reduced recall.
- The full pair representation gave the best GAT F1 trade-off.

---

## GNN Architecture Comparison on ESM-PCA16

All architecture comparison experiments below use:

```text
Dataset: Combined Current + BM5
Features: Basic 3 + Full Pair ESM-2 PCA16
Input dimension: 19
```

| Model | Seed | Precision 1 | Recall 1 | F1 1 | Accuracy | FP | TP | FN |
|-------|-----:|------------:|---------:|-----:|---------:|---:|---:|---:|
| GCN | 42 | 0.1085 | 0.5968 | 0.1836 | 0.8351 | 1,216 | 148 | 100 |
| GIN | 42 | 0.1075 | 0.5484 | 0.1798 | 0.8445 | 1,129 | 136 | 112 |
| GATv2 | 42 | 0.1957 | 0.4355 | 0.2700 | 0.9268 | 444 | 108 | 140 |
| GraphSAGE | 42 | 0.1860 | 0.5887 | 0.2827 | 0.9072 | 639 | 146 | 102 |
| GAT | 42 | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 523 | 132 | 116 |
| TransformerConv | 42 | 0.2041 | 0.8024 | 0.3254 | 0.8966 | 776 | 199 | 49 |
| TransformerConv | 7 | 0.3042 | 0.6452 | 0.4134 | 0.9431 | 366 | 160 | 88 |
| TransformerConv | 13 | 0.1383 | 0.9637 | 0.2419 | 0.8123 | 1,489 | 239 | 9 |
| TransformerConv | 21 | 0.2112 | 0.9153 | 0.3432 | 0.8911 | 848 | 227 | 21 |

### TransformerConv Multi-Seed Summary

| Metric | Value |
|--------|------:|
| Seeds | 42, 7, 13, 21 |
| Mean F1 | 0.3310 |
| Sample Std F1 | 0.0705 |
| Best single-run F1 | 0.4134 |
| Best seed | 7 |

---

## Current Best Model

The best single-run model in the project is now:

```text
Model: TransformerConv
Dataset: Combined Current + BM5
Features: Basic 3 features + Full Pair ESM-2 PCA16 features
Input dimension: 19
Seed: 7
Test Precision 1: 0.3042
Test Recall 1: 0.6452
Test F1 1: 0.4134
Test Accuracy: 0.9431
```

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|-------------|-------:|-------:|
| True 0 | 7,368 | 366 |
| True 1 | 88 | 160 |

More robust multi-seed summary:

```text
TransformerConv mean F1 over 4 seeds = 0.3310 ± 0.0705
```

---

## Interpretation of Final Results

The final architecture comparison shows that:

- GCN and GIN over-predict positives and generate too many false positives.
- GATv2 is more conservative than GAT but loses recall.
- GraphSAGE increases recall and true positives, but false positives also increase.
- TransformerConv provides the strongest overall performance.
- TransformerConv seed 7 gives the best precision-recall trade-off.
- TransformerConv seed 13 is very recall-oriented but unstable because false positives become too high.
- Multi-seed evaluation still shows TransformerConv outperforming the previous GAT best result on average.

The key performance progression is:

| Stage | F1 1 |
|-------|-----:|
| Current-only GAT basic | 0.2361 |
| Combined Current + BM5 GAT basic | 0.2791 |
| Combined Current + BM5 + ESM-PCA16 GAT | 0.2924 |
| Combined Current + BM5 + ESM-PCA16 TransformerConv mean | 0.3310 |
| Combined Current + BM5 + ESM-PCA16 TransformerConv best seed | 0.4134 |

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

### 6. Extract ESM-2 embeddings

```bash
python preprocessing/extract_esm2_embeddings.py --cases data/cases/combined_current_bm5_cases.csv --out_dir data/esm2_embeddings --model_name facebook/esm2_t6_8M_UR50D
```

### 7. Build ESM-2 PCA16 dataset

```bash
python preprocessing/build_esm2_pca_dataset.py --source_processed_dir data/processed_combined_current_bm5_train_balanced_natural_test --embedding_dir data/esm2_embeddings --out_dir data/processed_combined_current_bm5_esm2_pca16 --pca_components 16
```

### 8. Train final TransformerConv model

```bash
python experiments/train_expanded_transformerconv.py --processed_dir data/processed_combined_current_bm5_esm2_pca16 --output_csv experiments/combined_current_bm5_esm2_pca16_transformerconv_seed7_results.csv --output_md experiments/combined_current_bm5_esm2_pca16_transformerconv_seed7_results.md --seed 7
```

### 9. Run architecture comparisons

```bash
python experiments/train_expanded_gcn.py --processed_dir data/processed_combined_current_bm5_esm2_pca16
python experiments/train_expanded_gatv2.py --processed_dir data/processed_combined_current_bm5_esm2_pca16
python experiments/train_expanded_graphsage.py --processed_dir data/processed_combined_current_bm5_esm2_pca16
python experiments/train_expanded_gin.py --processed_dir data/processed_combined_current_bm5_esm2_pca16
python experiments/train_expanded_transformerconv.py --processed_dir data/processed_combined_current_bm5_esm2_pca16
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

## Large Generated Directories

Large generated data should not be committed:

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
2. adding compact PCA-reduced full-pair ESM-2 protein language model features
3. replacing GAT with TransformerConv for the final architecture comparison

Final best single-run model:

```text
Combined Current + BM5 + Full Pair ESM-2 PCA16 + TransformerConv
Seed = 7
F1 = 0.4134
```

More robust multi-seed summary:

```text
TransformerConv mean F1 = 0.3310 ± 0.0705
```

Additional model comparisons confirmed that TransformerConv provides the strongest performance among the tested GNN architectures.
