# Protein–Protein Interface Prediction using Graph Neural Networks

This project implements a residue-level correspondence graph approach
for predicting protein–protein interaction interfaces using Graph Neural Networks.

## Features

- Residue-level graph construction from PDB files
- Correspondence graph generation
- Interface labeling using atomic distance threshold
- Candidate filtering to reduce graph size
- Ready for GCN and GAT training

## Structure

- preprocessing/ → data processing
- models/ → GNN models
- training/ → training pipeline
- data/raw_pdb/ → PDB files
- data/processed/ → generated graph data

## Status

Dataset pipeline completed.
Model training coming next.
