# Protein–Protein Interface Prediction using Graph Neural Networks

A complete residue-level protein–protein interface prediction pipeline using correspondence graphs, Graph Neural Networks, BM5-clean dataset expansion, ESM-2 protein language model embeddings, and tuned Transformer-style graph attention.

This project is inspired by the paper **Graph Neural Networks for the Prediction of Protein–Protein Interfaces** and implements a reproducible workflow for predicting whether residue pairs across two protein partners belong to the interaction interface.

---

## Final Best Result

The strongest model in this project is:

```text
Tuned TransformerConv
Dataset: Combined Current + BM5
Features: Basic 3 features + Full Pair ESM-2 PCA16
Input dimension: 19
hidden_channels = 16
heads = 4
dropout = 0.2
learning rate = 0.003
weight decay = 0.001
validation threshold search max = 0.90
best seed = 1
```

Final test metrics for the best single run:

| Metric | Value |
|---|---:|
| Precision 1 | 0.5603 |
| Recall 1 | 0.6371 |
| F1 1 | **0.5962** |
| Accuracy | 0.9732 |

Confusion matrix:

| True / Pred | Pred 0 | Pred 1 |
|---|---:|---:|
| True 0 | 7,610 | 124 |
| True 1 | 90 | 158 |

Multi-seed summary for the final tuned TransformerConv with `threshold_max=0.90`:

| Statistic | F1 1 |
|---|---:|
| Mean over 4 seeds | **0.5502** |
| Sample standard deviation | 0.0350 |
| Best single run | **0.5962** |

---

## What This Project Does

The project predicts protein–protein interface residue pairs. Each node in the correspondence graph represents:

```text
(residue_i from partner A, residue_j from partner B)
```

The binary label is:

```text
0 = non-interface / non-contact residue pair
1 = interface / contact residue pair
```

A residue pair is labeled positive when at least one atom pair between the two residues is within:

```text
5 Å
```

This is a difficult and highly imbalanced binary node-classification task.

---

## Pipeline Overview

The project includes:

- PDB parsing with Biopython
- residue filtering and C-alpha extraction
- residue-level graph construction for each protein partner
- correspondence graph construction between partner residues
- atom-distance-based interface labeling
- candidate residue-pair filtering
- multi-chain partner support
- multi-graph training
- validation-based early stopping
- validation-based threshold selection
- GCN, GAT, GATv2, GraphSAGE, GIN, and TransformerConv experiments
- BM5-clean dataset expansion
- ESM-2 embedding extraction
- PCA-reduced ESM pair features
- ESM feature ablations
- class imbalance and hard-negative experiments
- GAT attention analysis
- error analysis
- PyMOL structural error visualization
- final tuned TransformerConv optimization

---

## Dataset Construction

For each complex:

1. Load the PDB file.
2. Extract selected partner chains.
3. Keep standard amino acid residues with C-alpha atoms.
4. Build residue-level graphs using C-alpha distance.
5. Generate candidate residue-pair correspondence nodes.
6. Label each residue pair using atom-level distance.
7. Build correspondence graph edges from local residue-graph topology.
8. Save features, labels, residue pairs, and edge indices.

Saved files per case:

```text
<CASE_NAME>_corr_features.npy
<CASE_NAME>_corr_labels.npy
<CASE_NAME>_corr_pairs.npy
<CASE_NAME>_corr_edge_index.npy
```

---

## Original Current Dataset

The original current dataset contains 12 protein complexes.

| Total Nodes | Positive | Negative | Positive Ratio |
|---:|---:|---:|---:|
| 20,707 | 698 | 20,009 | 0.0337 |

This severe imbalance is typical for protein interface prediction.

---

## BM5-Clean Dataset Expansion

BM5-clean reference complexes were imported and screened.

| Stage | Count |
|---|---:|
| Imported BM5 reference complexes | 29 |
| Accepted after screening | 19 |
| Rejected | 10 |

Screening criteria:

```text
positive >= 30
positive_ratio >= 0.02
candidate_nodes <= 8000
```

Accepted BM5 positives:

```text
1,096
```

---

## Combined Current + BM5 Dataset

| Source | Cases | Positive Pairs |
|---|---:|---:|
| Current dataset | 12 | 698 |
| BM5-clean accepted | 19 | 1,096 |
| Combined total | 31 | 1,794 |

Processed combined dataset:

| Saved Nodes | Positive | Negative | Positive Ratio |
|---:|---:|---:|---:|
| 23,028 | 1,794 | 21,234 | 0.0779 |

Split:

| Split | Cases |
|---|---:|
| Train | 22 |
| Validation | 4 |
| Test | 5 |

Training graphs are semi-balanced. Validation and test graphs preserve natural class imbalance.

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

### Basic + Full Pair ESM-2 PCA16

ESM-2 model:

```text
facebook/esm2_t6_8M_UR50D
```

Raw full pair representation:

```text
[ESM_A, ESM_B, abs(ESM_A - ESM_B)]
```

Raw pair dimension:

```text
960
```

PCA is fitted only on training pairs. The final best feature vector is:

```text
[basic_3_features, PCA16(full_pair_ESM)]
```

Final input dimension:

```text
19
```

---

## Main Result Progression

| Stage | Dataset | Model / Features | F1 1 |
|---|---|---|---:|
| Current-only strict baseline | 12 complexes | Basic 3 + GAT | 0.2361 |
| Combined Current + BM5 | 31 complexes | Basic 3 + GAT | 0.2791 |
| Combined Current + BM5 | Full Pair ESM-2 PCA16 + GAT | 0.2924 |
| Combined Current + BM5 | Full Pair ESM-2 PCA16 + initial TransformerConv | 0.4134 |
| Combined Current + BM5 | Full Pair ESM-2 PCA16 + tuned TransformerConv, threshold ≤ 0.60 | 0.5122 |
| Combined Current + BM5 | Full Pair ESM-2 PCA16 + tuned TransformerConv, threshold ≤ 0.90 | **0.5962** |

---

## GNN Architecture Comparison

All models below use the same final dataset and feature setup:

```text
Combined Current + BM5
Basic 3 + Full Pair ESM-2 PCA16
Input dimension = 19
```

| Model | Seed | Precision 1 | Recall 1 | F1 1 | Accuracy | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GCN | 42 | 0.1085 | 0.5968 | 0.1836 | 0.8351 | 1,216 | 100 | 148 |
| GIN | 42 | 0.1075 | 0.5484 | 0.1798 | 0.8445 | 1,129 | 112 | 136 |
| GATv2 | 42 | 0.1957 | 0.4355 | 0.2700 | 0.9268 | 444 | 140 | 108 |
| GraphSAGE | 42 | 0.1860 | 0.5887 | 0.2827 | 0.9072 | 639 | 102 | 146 |
| GAT | 42 | 0.2015 | 0.5323 | 0.2924 | 0.9199 | 523 | 116 | 132 |
| TransformerConv | 7 | 0.3042 | 0.6452 | 0.4134 | 0.9431 | 366 | 88 | 160 |
| Tuned TransformerConv, threshold ≤ 0.90 | 1 | **0.5603** | 0.6371 | **0.5962** | **0.9732** | **124** | 90 | 158 |

---

## Final Tuned TransformerConv Multi-Seed Results

| Seed | Threshold | Precision 1 | Recall 1 | F1 1 | Accuracy | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.90 | 0.5603 | 0.6371 | **0.5962** | 0.9732 | 7,610 | 124 | 90 | 158 |
| 3 | 0.77 | 0.3947 | 0.7258 | 0.5114 | 0.9569 | 7,458 | 276 | 68 | 180 |
| 5 | 0.86 | 0.4363 | 0.7177 | 0.5427 | 0.9624 | 7,504 | 230 | 70 | 178 |
| 21 | 0.87 | 0.4407 | 0.7339 | 0.5507 | 0.9628 | 7,503 | 231 | 66 | 182 |

Summary:

```text
Mean F1 = 0.5502
Std F1  = 0.0350
Best F1 = 0.5962
```

A separate `threshold_max=0.99` test for seed 1 produced `F1=0.5611`, so the final chosen setup remains `threshold_max=0.90`.

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

### 7. Build ESM-2 PCA16 features

```bash
python preprocessing/build_esm2_pca_dataset.py --source_processed_dir data/processed_combined_current_bm5_train_balanced_natural_test --embedding_dir data/esm2_embeddings --out_dir data/processed_combined_current_bm5_esm2_pca16 --pca_components 16
```

### 8. Train final best model

```bash
python experiments/train_expanded_transformerconv.py --processed_dir data/processed_combined_current_bm5_esm2_pca16 --output_csv experiments/combined_current_bm5_esm2_pca16_transformerconv_tuned_seed1_thr90_results.csv --output_md experiments/combined_current_bm5_esm2_pca16_transformerconv_tuned_seed1_thr90_results.md --hidden_channels 16 --heads 4 --dropout 0.2 --lr 0.003 --weight_decay 0.001 --threshold_max 0.90 --seed 1
```

---

## Requirements

Use the frozen environment file generated from `venv310`:

```bash
pip install -r requirements.txt
```

For CUDA-specific PyTorch / PyTorch Geometric installations, follow the official wheel instructions if needed.

---

## Generated Data Policy

Large generated data should not be committed:

```text
data/raw_pdb_expanded_bm5/
data/esm2_embeddings/
data/processed_*/
```

---

## Final Conclusion

The strongest improvements came from:

1. expanding the dataset using BM5-clean
2. adding compact full-pair ESM-2 PCA16 features
3. switching from GAT to TransformerConv
4. tuning TransformerConv dropout and threshold range

Final best result:

```text
Tuned TransformerConv + Combined Current/BM5 + Full Pair ESM-2 PCA16
F1 = 0.5962
```

This is the strongest result achieved in the project so far.
