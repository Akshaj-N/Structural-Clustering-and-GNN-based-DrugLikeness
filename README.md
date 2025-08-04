# ⚛️ Structural Clustering and GNN‑based Drug‑Likeness Prediction
## Overview
This project builds a machine learning pipeline that processes a BindingDB COVID‑19 dataset of molecules and targets, computes molecular descriptors, performs unsupervised clustering (K‑Means, DBSCAN, hierarchical, GMM) and then trains Graph Neural Networks (GNNs), including GCN, GAT and GIN to predict whether molecules satisfy Lipinski's Rule of Five.

Primary goals:
1. Explore molecular clusters in descriptor space.  
2. Evaluate which clustering method best separates drug‑like compounds.  
3. Build GNN classifiers to predict Lipinski drug‑likeness, comparing models for optimal performance.

## Datasets

### BindingDB COVID-19 Dataset
- **Description**: COVID-19 related binding data from BindingDB
- **Link**: [https://www.bindingdb.org/rwd/bind/Covid19.jsp](https://www.bindingdb.org/rwd/bind/Covid19.jsp)

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

## Installation

```bash
# Clone the repository
git clone https://github.com/Akshaj-N/Structural-Clustering-and-GNN-based-DrugLikeness.git
cd Structural-Clustering-and-GNN-based-DrugLikeness
```

## Features & Pipeline
### 1. Data Preprocessing & Feature Engineering

#### 1.1 Data Cleaning
- **Input**: `BindingDB_Covid-19.tsv` containing COVID-19 related molecular data
- **Processing Steps**:
  - Remove columns with >50% missing values
  - Validate and filter invalid SMILES strings
  - Deduplicate entries based on molecular structure
  - Retain core features: SMILES, Ligand Name, Target, Sequence

#### 1.2 Molecular Descriptor Calculation
Using RDKit, we compute six key molecular descriptors:
- **MolWt**: Molecular weight
- **LogP**: Lipophilicity coefficient
- **NumHDonors**: Hydrogen bond donor count
- **NumHAcceptors**: Hydrogen bond acceptor count
- **TPSA**: Topological Polar Surface Area
- **NumRotatableBonds**: Molecular flexibility metric

#### 1.3 Target Variable Generation
- Binary classification based on **Lipinski's Rule of Five**
- Label 1: Drug-like (passes all criteria)
- Label 0: Non-drug-like (fails one or more criteria)

### 2. Data Quality Enhancement

#### 2.1 Outlier Detection & Removal
- **Method**: Interquartile Range (IQR) analysis per descriptor
- **Threshold**: Remove points beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR
- **Impact**: Ensures robust model training by eliminating extreme values

#### 2.2 Class Balancing
- **Challenge**: Imbalanced dataset (drug-like molecules underrepresented)
- **Solution**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Result**: Balanced dataset with ~10,000 samples per class

### 3. Unsupervised Clustering Analysis

#### 3.1 Clustering Methods & Results

| Algorithm | Optimal Parameters | Clusters | Best Metric Score |
|-----------|-------------------|----------|-------------------|
| **K-Means** | k=2 | 2 | Silhouette: 0.33 |
| **GMM** | n_components=2 | 2 | BIC-optimized |
| **DBSCAN** | eps=0.37, min_samples=50 | ~40 | Silhouette: 0.436 |
| **Hierarchical** | Complete linkage, 70% cut | 5 | Interpretable sizes |

#### 3.2 Clustering Evaluation
- **Metrics Used**:
  - Silhouette Score (cohesion vs separation)
  - Calinski-Harabasz Index (inter-cluster dispersion)
  - Davies-Bouldin Score (average similarity)
- **Visualization**: PCA and t-SNE projections for 2D/3D cluster analysis
- **Feature Analysis**: Fisher ratio heatmap to identify discriminative descriptors

### 4. Graph Neural Network Pipeline

#### 4.1 Molecular Graph Construction
```python
# SMILES → PyTorch Geometric Data
Node Features (5-dimensional):
- Atomic number (one-hot encoded)
- Degree (number of bonds)
- Formal charge
- Hybridization state (sp, sp2, sp3)
- Is aromatic (boolean)

Edge Features:
- Bond connectivity (undirected graph)
- Edge index representation for PyG
```

#### 4.2 Data Splitting Strategy
- **Training Set**: 80% (16,000 molecules)
- **Validation Set**: 10% (2,000 molecules)
- **Test Set**: 10% (2,000 molecules)
- **Stratified Split**: Maintains class balance across all sets

### 5. GNN Model Architectures

#### 5.1 Model Specifications

| Model | Architecture Details | Key Features |
|-------|---------------------|--------------|
| **GCN** | 2 layers, 64 hidden units | Spectral convolutions, baseline model |
| **GCN-Weighted** | Same as GCN + class weights | Addresses class imbalance |
| **GAT** | 2 attention heads, 3 layers | Self-attention mechanism |
| **GIN** | 3 layers, MLP aggregation | Maximum expressive power |

#### 5.2 Training Configuration
```yaml
Optimizer: Adam
Learning Rate: 0.001
Batch Size: 32
Epochs: 20
Loss Function: Binary Cross-Entropy
```

### 6. Performance Evaluation

#### 6.1 Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **GCN** | 63.4% | 0.65 | 0.92 | 0.75 |
| **GCN-Weighted** | 64.9% | 0.66 | 0.90 | 0.76 |
| **GAT** | 64.2% | 0.69 | 0.91 | 0.76 |
| **GIN** | **65.5%** | **0.67** | **0.92** | **0.78** |

#### 6.2 Key Insights
- **GCN**: Achieves high recall but suffers from low precision
- **Weighted GCN**: Class weighting improves precision at the cost of recall
- **GAT**: Attention mechanism provides balanced performance
- **GIN**: Superior expressiveness captures complex molecular patterns most effectively
ive neighborhood learning

