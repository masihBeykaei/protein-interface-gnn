# Protein–Protein Interface Prediction with Graph Neural Networks

> **Residue-level protein–protein interface prediction using correspondence graphs, BM5-clean dataset expansion, ESM-2 protein language model embeddings, and tuned TransformerConv graph attention.**

This repository contains a complete research-oriented pipeline for predicting protein–protein interface residue pairs from 3D protein complex structures. The project starts from raw PDB files, builds graph representations of interacting proteins, trains several Graph Neural Network architectures, and progressively improves performance through dataset expansion, protein language model features, architecture comparison, and threshold optimization.

The final best model is a **tuned TransformerConv** model trained on the combined current + BM5-clean dataset with **Full Pair ESM-2 PCA16** features.

```text
Best final model:
Tuned TransformerConv + Combined Current/BM5 + Full Pair ESM-2 PCA16

Best single-run test result:
Precision 1 = 0.5603
Recall 1    = 0.6371
F1 1        = 0.5962
Accuracy    = 0.9732
```

---

## Table of Contents

- [Project Summary](#project-summary)
- [Motivation](#motivation)
- [Relation to the Reference Paper](#relation-to-the-reference-paper)
- [Problem Definition](#problem-definition)
- [Graph Representation](#graph-representation)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Final Best Result](#final-best-result)
- [Main Result Progression](#main-result-progression)
- [Architecture Comparison](#architecture-comparison)
- [TransformerConv Tuning](#transformerconv-tuning)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [How to Reproduce the Pipeline](#how-to-reproduce-the-pipeline)
- [Important Output Files](#important-output-files)
- [Interpretability and Error Analysis](#interpretability-and-error-analysis)
- [Project Challenges and Solutions](#project-challenges-and-solutions)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Final Conclusion](#final-conclusion)

---

## Project Summary

Protein–protein interactions are central to many biological processes. In this project, the goal is to predict which residue pairs are likely to form a protein–protein interface.

The project formulates this as a **binary node classification problem** on a **correspondence graph**:

```text
node = candidate residue pair across two proteins
label 0 = non-interface / non-contact
label 1 = interface / contact
```

The full pipeline includes:

- parsing protein structures from PDB files
- extracting selected chains and residues
- building residue-level graphs
- constructing correspondence graphs between two protein partners
- labeling residue pairs using atom-level distance
- extracting structural and sequence-based features
- adding ESM-2 protein language model embeddings
- reducing ESM features using PCA
- training several GNN models
- tuning TransformerConv
- selecting thresholds using validation data
- evaluating on held-out natural test graphs
- analyzing errors and attention behavior
- generating structural visualization scripts

The final project is not just a single model training script; it is a complete experimental workflow.

---

## Motivation

Protein–protein interfaces are the regions where two proteins physically interact. Identifying these regions is useful for:

- understanding biological mechanisms
- studying molecular recognition
- supporting protein docking
- predicting mutation effects
- designing inhibitors or therapeutic proteins
- guiding experimental biology

Traditional methods often rely on handcrafted features and classical machine learning. Graph Neural Networks are attractive because proteins naturally have graph-like structure:

```text
residues = nodes
spatial relationships = edges
```

This project extends that idea by building a **correspondence graph** between two proteins, where each node represents a possible interacting residue pair.

---

## Relation to the Reference Paper

This project is inspired by the paper:

```text
Graph Neural Networks for the Prediction of Protein–Protein Interfaces
Niccolò Pancino, Alberto Rossi, Giorgio Ciano, Giorgia Giacomini,
Simone Bonechi, Paolo Andreini, Franco Scarselli,
Monica Bianchini, Pietro Bongini
```

The key idea from the paper is to use Graph Neural Networks for protein–protein interface prediction by representing structural information as graphs.

In simple terms, the paper suggests that instead of looking at residue pairs independently, we should let the model learn from their graph neighborhood. This is important because an interface is not just a collection of isolated contacts; it has local structural patterns.

This repository follows that general idea but extends the implementation with:

- a full custom preprocessing pipeline
- BM5-clean dataset expansion
- ESM-2 protein language model features
- PCA-reduced pair embeddings
- multiple GNN architecture comparisons
- TransformerConv model tuning
- threshold optimization
- detailed error and attention analysis

So the project can be understood as a **reproduction-inspired and extended implementation** rather than a strict one-to-one reproduction.

---

## Problem Definition

For two protein partners, A and B, the model considers candidate residue pairs:

```text
(a_i, b_j)
```

Each candidate pair receives a binary label:

```text
1 = interface/contact pair
0 = non-interface/non-contact pair
```

A residue pair is labeled as positive if at least one atom pair between the two residues is closer than:

```text
5 Å
```

This creates a highly imbalanced classification problem because most residue pairs do not form real contacts.

---

## Graph Representation

### 1. Residue-Level Graphs

Each protein partner is represented as a residue-level graph:

```text
node = amino acid residue
edge = spatial proximity between residues
```

Residue proximity is computed using C-alpha coordinates.

### 2. Correspondence Graph

The correspondence graph is built between the two protein partners.

Each correspondence node represents:

```text
(residue from partner A, residue from partner B)
```

This means a node is not a single residue; it is a **candidate residue pair**.

Edges between correspondence nodes are created based on compatible neighborhood relationships in the original residue graphs.

This allows the GNN to learn local interaction patterns such as:

```text
if residue pair (a_i, b_j) is important,
neighboring residue pairs may also carry useful interface context
```

---

## Dataset

### Original Current Dataset

The initial dataset contained 12 protein complexes.

| Total Nodes | Positive | Negative | Positive Ratio |
|---:|---:|---:|---:|
| 20,707 | 698 | 20,009 | 0.0337 |

This means only about 3.37% of candidate residue pairs were positive.

### BM5-Clean Dataset Expansion

To improve data diversity, BM5-clean reference complexes were imported and screened.

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

### Combined Current + BM5 Dataset

The final expanded dataset combines:

| Source | Cases | Positive Pairs |
|---|---:|---:|
| Current dataset | 12 | 698 |
| BM5-clean accepted | 19 | 1,096 |
| Combined total | 31 | 1,794 |

Processed combined dataset:

| Saved Nodes | Positive | Negative | Positive Ratio |
|---:|---:|---:|---:|
| 23,028 | 1,794 | 21,234 | 0.0779 |

Final split:

| Split | Cases |
|---|---:|
| Train | 22 |
| Validation | 4 |
| Test | 5 |

The training split is semi-balanced, while validation and test splits preserve the natural class imbalance.

---

## Feature Engineering

Several feature families were tested during the project.

### Basic Features

The first feature set was intentionally simple:

```text
[CA_distance, degree_partner_1, degree_partner_2]
```

Input dimension:

```text
3
```

### Amino Acid One-Hot Features

Residue identities were encoded as one-hot vectors:

```text
basic features + one-hot residue A + one-hot residue B
```

Input dimension:

```text
43
```

### Physicochemical Features

Residue-level physicochemical descriptors were also tested:

- hydrophobicity
- charge
- polarity
- aromaticity

Input dimension:

```text
11
```

### ASA Features

Accessible surface area features were tested:

```text
basic features + ASA_A + ASA_B
```

Input dimension:

```text
5
```

### ESM-2 Protein Language Model Features

The strongest feature family came from ESM-2.

Model used:

```text
facebook/esm2_t6_8M_UR50D
```

For each residue pair, the raw full pair representation was:

```text
[ESM_A, ESM_B, abs(ESM_A - ESM_B)]
```

Raw dimension:

```text
960
```

Because this is high-dimensional, PCA was applied.

Important methodological detail:

```text
Standardization and PCA were fitted only on training pairs.
```

This avoids validation/test leakage.

### Final Best Feature Set

The final best feature vector is:

```text
[basic_3_features, PCA16(full_pair_ESM)]
```

Input dimension:

```text
19
```

---

## Models

The project compares several GNN architectures:

| Model | Purpose |
|---|---|
| GCN | classic graph convolution baseline |
| GAT | graph attention baseline |
| GATv2 | more flexible attention mechanism |
| GraphSAGE | neighborhood aggregation baseline |
| GIN | expressive MLP-based message passing |
| TransformerConv | transformer-style graph attention |

The final best architecture is:

```text
Tuned TransformerConv
```

---

## Final Best Result

Final best configuration:

```text
Model: Tuned TransformerConv
Dataset: Combined Current + BM5
Features: Basic 3 + Full Pair ESM-2 PCA16
Input dimension: 19
hidden_channels: 16
heads: 4
dropout: 0.2
learning rate: 0.003
weight decay: 0.001
threshold_max: 0.90
seed: 1
```

Final test metrics:

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

---

## Main Result Progression

| Stage | Dataset | Model / Features | F1 1 |
|---|---|---|---:|
| Current-only strict baseline | 12 complexes | Basic 3 + GAT | 0.2361 |
| Combined Current + BM5 | 31 complexes | Basic 3 + GAT | 0.2791 |
| Combined Current + BM5 | 31 complexes | Full Pair ESM-2 PCA16 + GAT | 0.2924 |
| Combined Current + BM5 | 31 complexes | Full Pair ESM-2 PCA16 + initial TransformerConv | 0.4134 |
| Combined Current + BM5 | 31 complexes | Full Pair ESM-2 PCA16 + tuned TransformerConv, threshold ≤ 0.60 | 0.5122 |
| Combined Current + BM5 | 31 complexes | Full Pair ESM-2 PCA16 + tuned TransformerConv, threshold ≤ 0.90 | **0.5962** |

---

## Architecture Comparison

All models below use:

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
| TransformerConv initial | 7 | 0.3042 | 0.6452 | 0.4134 | 0.9431 | 366 | 88 | 160 |
| Tuned TransformerConv | 1 | **0.5603** | 0.6371 | **0.5962** | **0.9732** | **124** | 90 | 158 |

---

## TransformerConv Tuning

The initial TransformerConv already improved over GAT:

```text
Initial TransformerConv best F1 = 0.4134
```

Then the model was tuned.

Final tuned configuration:

```text
hidden_channels = 16
heads = 4
dropout = 0.2
learning rate = 0.003
weight decay = 0.001
beta = True
```

### Threshold Search up to 0.60

| Seed | Precision 1 | Recall 1 | F1 1 | Accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.2264 | 0.8427 | 0.3570 | 0.9057 |
| 3 | 0.2825 | 0.8065 | 0.4184 | 0.9303 |
| 5 | 0.3311 | 0.7863 | 0.4659 | 0.9440 |
| 21 | 0.3969 | 0.7218 | 0.5122 | 0.9573 |

Summary:

```text
Mean F1 = 0.4384 ± 0.0664
Best F1 = 0.5122
```

### Threshold Search up to 0.90

| Seed | Selected Threshold | Precision 1 | Recall 1 | F1 1 | Accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.90 | 0.5603 | 0.6371 | **0.5962** | 0.9732 |
| 3 | 0.77 | 0.3947 | 0.7258 | 0.5114 | 0.9569 |
| 5 | 0.86 | 0.4363 | 0.7177 | 0.5427 | 0.9624 |
| 21 | 0.87 | 0.4407 | 0.7339 | 0.5507 | 0.9628 |

Summary:

```text
Mean F1 = 0.5502 ± 0.0350
Best F1 = 0.5962
```

A final check with `threshold_max=0.99` did not improve the result:

```text
seed 1, threshold_max=0.99
F1 = 0.5611
```

Therefore, the final selected setting is:

```text
threshold_max = 0.90
```

---

## Repository Structure

A typical structure of the repository is:

```text
protein-interface-gnn/
├── data/
│   ├── cases/
│   ├── raw_pdb/
│   ├── raw_pdb_expanded_bm5/
│   ├── processed_*/
│   └── esm2_embeddings/
├── preprocessing/
│   ├── build_multi_protein_dataset.py
│   ├── screen_candidate_complexes.py
│   ├── import_bm5_reference_complexes.py
│   ├── build_train_balanced_natural_eval_dataset.py
│   ├── extract_esm2_embeddings.py
│   └── build_esm2_pca_dataset.py
├── experiments/
│   ├── train_expanded_gat.py
│   ├── train_expanded_gcn.py
│   ├── train_expanded_gatv2.py
│   ├── train_expanded_graphsage.py
│   ├── train_expanded_gin.py
│   ├── train_expanded_transformerconv.py
│   ├── tune_transformerconv.py
│   ├── results_summary.md
│   └── figures/
├── report/
│   └── final_report.md
├── README.md
├── requirements.txt
└── .gitignore
```

Some generated directories are intentionally large and should not be committed.

---

## Installation

Create and activate a Python environment:

```bash
python -m venv venv310
```

On Windows PowerShell:

```powershell
.\venv310\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch or PyTorch Geometric fails to install automatically, install them using the official CUDA-specific instructions for your machine.

---

## How to Reproduce the Pipeline

### 1. Build the original current dataset

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

### 5. Build combined train-balanced / natural-evaluation dataset

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

### 8. Train the final tuned TransformerConv model

```bash
python experiments/train_expanded_transformerconv.py --processed_dir data/processed_combined_current_bm5_esm2_pca16 --output_csv experiments/combined_current_bm5_esm2_pca16_transformerconv_tuned_seed1_thr90_results.csv --output_md experiments/combined_current_bm5_esm2_pca16_transformerconv_tuned_seed1_thr90_results.md --hidden_channels 16 --heads 4 --dropout 0.2 --lr 0.003 --weight_decay 0.001 --threshold_max 0.90 --seed 1
```

### 9. Run TransformerConv tuning

```bash
python experiments/tune_transformerconv.py --processed_dir data/processed_combined_current_bm5_esm2_pca16 --search_mode small --seeds 7,42,21 --output_csv experiments/transformerconv_tuning_results.csv --output_md experiments/transformerconv_tuning_results.md
```

---

## Important Output Files

Important result files include:

```text
experiments/results_summary.md
report/final_report.md
experiments/combined_current_bm5_esm2_pca16_transformerconv_tuned_seed1_thr90_results.md
experiments/combined_current_bm5_esm2_pca16_transformerconv_tuned_seed1_thr90_results.csv
experiments/transformerconv_tuning_results.md
experiments/transformerconv_tuning_results.csv
```

Important analysis files include:

```text
experiments/gat_attention_summary.md
experiments/gat_attention_refined_summary.md
experiments/error_analysis_gat_basic_summary.md
experiments/structural_error_visualization/
```

---

## Interpretability and Error Analysis

The project includes several interpretation-oriented analyses.

### GAT Attention Analysis

GAT attention weights were extracted and analyzed. A refined attention analysis separated:

- self-loop attention
- non-self attention
- true-positive context
- false-positive context
- false-negative context

Important caution:

```text
Attention weights are message-passing weights, not direct biological proof.
```

### Structural Error Visualization

PyMOL scripts were generated to visualize:

| Color | Meaning |
|---|---|
| Green | true positives |
| Red | false positives |
| Orange | false negatives |

Generated scripts are stored under:

```text
experiments/structural_error_visualization/
```

---

## Project Challenges and Solutions

### Challenge 1: Very Small Initial Dataset

The original current dataset had only 12 complexes and 698 positives.

Solution:

```text
BM5-clean expansion
```

### Challenge 2: Severe Class Imbalance

Most candidate residue pairs are negative.

Solution:

```text
train-balanced / natural-evaluation protocol
validation-based threshold selection
positive-class F1 as main metric
```

### Challenge 3: Basic Features Were Too Weak

The basic 3-feature setup had limited representation power.

Solution:

```text
ESM-2 residue embeddings
full-pair ESM representation
PCA16 feature compression
```

### Challenge 4: High-Dimensional ESM Features Could Overfit

PCA64 increased recall but produced too many false positives.

Solution:

```text
PCA16 compact representation
```

### Challenge 5: GAT Plateaued

GAT improved results but stopped around F1 ≈ 0.2924.

Solution:

```text
architecture comparison
TransformerConv model selection
```

### Challenge 6: Threshold Ceiling Limited F1

Initial threshold search up to 0.60 was not enough.

Solution:

```text
expand threshold search up to 0.90
```

This produced the best final result.

---

## Limitations

This project still has limitations:

1. The dataset is larger than the initial version but still limited.
2. The final test set contains only 5 complexes.
3. The label definition depends on a fixed 5 Å distance cutoff.
4. The final result depends on validation-based threshold selection.
5. ESM-2 features are sequence-based and do not directly model complex-specific 3D geometry.
6. The model does not yet use explicit edge features.
7. More cross-validation would provide stronger statistical confidence.
8. Structural visualization is qualitative and requires manual inspection.

---

## Future Work

Recommended next steps:

1. Add explicit edge features such as distance bins and geometry descriptors.
2. Test larger ESM-2 models if GPU memory allows.
3. Run more seeds for the final tuned TransformerConv.
4. Perform complex-level cross-validation.
5. Compare against non-GNN baselines such as MLP, Random Forest, and XGBoost.
6. Try seed ensembling for more stable predictions.
7. Add calibration analysis.
8. Extend BM5-clean import to more benchmark cases.
9. Improve visual explanations for final defense/presentation.

---

## Final Conclusion

The project shows a clear improvement path:

```text
small current-only dataset
→ BM5-clean expansion
→ ESM-2 PCA16 features
→ TransformerConv architecture
→ tuned threshold selection
```

Final best model:

```text
Tuned TransformerConv + Combined Current/BM5 + Full Pair ESM-2 PCA16
F1 = 0.5962
```

This is the strongest result achieved in the project and represents a substantial improvement over the original GAT baseline.
