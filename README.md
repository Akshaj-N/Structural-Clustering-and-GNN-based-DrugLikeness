# ⚛️ Structural Clustering and GNN‑based Drug‑Likeness Prediction
## Overview
This project builds a machine learning pipeline that processes a BindingDB COVID‑19 dataset of molecules and targets, computes molecular descriptors, performs unsupervised clustering (K‑Means, DBSCAN, hierarchical, GMM), and then trains Graph Neural Networks (GNNs), including GCN, GAT, and GIN, to predict whether molecules satisfy Lipinski's Rule of Five.

Primary goals:

1. Explore molecular clusters in descriptor space.

2. Evaluate which clustering method best separates drug‑like compounds.

3. Build GNN classifiers to predict Lipinski drug‑likeness, comparing models for optimal performance.

## File structure
```
.
├── README.md                       # (you’re reading this)
├── BindingDB_Covid‑19.tsv          # Input raw data file
├── lipinski_balanced_smote.csv     # Balanced descriptor dataset
├── EDA_Clustering.ipynb            # SMILES cleaning & descriptor computation
├── DBSCAN.ipynb                    # Clustering experiments (K‑Means, GMM, DBSCAN, hierarchical)
├── Hierarchical_Clustering.ipynb   # SMOTE balancing routine
├── Kmeans_GMM_clustering.py        # RDKit → PyG `Data` conversion
├── GNN.ipynb                       # graph preprocessing+ GNN model definitions (GCN, GAT, GIN) + training loops + evaluation
```

## Features & Pipeline
1. Data Cleaning & Descriptor Computation
- Load BindingDB_Covid‑19.tsv

- Drop columns with excessive missing values, invalid SMILES

- Deduplicate and keep only SMILES + Ligand Name + Target + sequence

- Compute RDKit descriptors: MolWt, LogP, NumHDonors, NumHAcceptors, TPSA, NumRotatableBonds

- Compute Lipinski pass/fail (binary label)

2. Outlier Removal & Data Balancing
- Detect and drop outliers via IQR per descriptor

- SMOTE used to get a class-balanced dataset (~10k label‑0 and ~10k label‑1)

3. Clustering (Unsupervised Analysis):

- K‑Means and Gaussian Mixture Models (GMM) via silhouette, Calinski‑Harabasz, Davies‑Bouldin metrics with best k=2 in both

- DBSCAN parameter sweep gave the best silhouette of approx. 0.436 with eps≈0.37, min_samples=50, yielding ~40 clusters

- Hierarchical (complete linkage), cut at 70% of max distance of clusters with interpretable size and Lipinski pass rates

- Visualize with PCA and t‑SNE projections

- Evaluate cluster quality and feature importance (Fisher ratio heatmap)

4. Graph Representation & GNN Preparation
- Convert each SMILES into PyTorch Geometric Data objects:

- Node features: atomic number, degree, formal charge, hybridization, aromaticity

- Edge index: undirected bonds

- Build train/validation/test splits (80/10/10)

- Create DataLoader batches

5. GNN Models
- GCN (vanilla)

- GCN_Weighted with class imbalance weighting

- GAT (graph attention) with 2 attention heads

- GIN (graph isomorphism network) for high expressive power

- Training: 20 epochs each, batch size = 32, optimizer = Adam, learning rate = 0.001

6. Evaluation & Results
- Validation metrics monitored: Accuracy, Precision, Recall, F1

- GCN (vanilla): high recall (~1.0) but lower precision (~0.63), F1 ~0.76

- Weighted GCN: better precision (~0.71) but recall drop, F1 ~0.67–0.70

- GAT model: improved balance, F1 ~0.76 overall

- GIN model: best performance with sharper convergence, accuracy up to ~75%, F1 ~0.80–0.82

- GIN outperformed others due to expressive neighborhood learning

