# Fraud Detection Model Comparison on Tabular Financial Data

## Overview
This project presents a comprehensive comparative study of fraud detection models on structured financial datasets. The study evaluates three major modelling paradigms:

- Classical ensemble methods  
- Transformer-based tabular architectures  
- Graph-based neural networks  

All models are evaluated under a **unified, leakage-aware experimental pipeline** to ensure fair comparison.

The objective is to determine whether advanced AI architectures provide meaningful improvements over traditional machine learning methods in real-world fraud detection scenarios.

---

## Motivation
Financial fraud detection is characterized by:

- Severe class imbalance (very few fraud cases)
- Large-scale transactional data
- Operational constraints (limited investigation capacity)

Most prior research relies heavily on **synthetic oversampling techniques**, which may distort real data distributions and affect model evaluation.

Additionally, many studies:
- Evaluate only a single modelling paradigm
- Use inconsistent preprocessing pipelines
- Focus on threshold-based metrics rather than ranking performance

This project addresses these gaps through a controlled and realistic experimental framework.

---

## Key Contributions
- Unified comparison of ensemble, transformer, and graph-based models  
- Leakage-aware data pipeline with consistent preprocessing  
- Evaluation without relying on synthetic oversampling (baseline setup)  
- Use of **ranking-based metrics (Gain & Lift)** alongside PR-AUC  
- Multi-seed experimentation (10 runs) for robustness  
- Statistical validation using hypothesis testing  

---

## Models Evaluated
### Ensemble Models
- Random Forest (primary baseline)

### Transformer-Based Models
- Tabular Transformer architectures  
- Gated TabTransformer / FT-Transformer variants  

### Graph-Based Models
- Graph Neural Networks (GNN)  
- Heterogeneous Graph Transformer (HGT)  
- Encoder-only graph-based anomaly model  

---

## Datasets
### 1. IEEE-CIS Fraud Detection Dataset
- ~590,540 transactions  
- ~3.5% fraud cases  
- High dimensional, highly imbalanced  

### 2. Bank Marketing Dataset
- Secondary validation dataset  
- Used to test cross-dataset generalization  

---

## Data Pipeline
A unified and leakage-aware preprocessing pipeline is applied:

1. Data assembly and merging  
2. Temporal data splitting (train → validation → test)  
3. Missing value handling (train-fitted imputation)  
4. Feature engineering and selection  
5. Hybrid encoding (categorical handling)  
6. Scaling using training statistics  
7. Graph-based feature augmentation (node-level statistics)  

All transformations are **fit on training data only** and applied consistently to validation/test sets.

---

## Experimental Design
- Unified preprocessing across all models  
- No synthetic oversampling in primary pipeline  
- Stratified handling of imbalance within training data  
- 10 independent seed runs for robustness  
- Evaluation on validation and test splits  

### Additional Experiment
- SMOTE-based oversampling applied **only on training data**  
- Used as a sensitivity analysis (not baseline)

---

## Evaluation Metrics
### Classification Metrics
- PR-AUC (Primary metric)
- ROC-AUC  

### Ranking-Based Metrics
- Cumulative Gain  
- Lift Curve  

These metrics reflect both:
- Predictive performance  
- Operational usefulness in fraud investigation  

---

## Statistical Validation
- Paired comparisons across models using seed-wise results  
- Wilcoxon signed-rank test  
- Significance level: α = 0.05  

This ensures that performance differences are:
- Statistically reliable  
- Not due to random variation  

---

## Key Findings
- **Random Forest consistently achieves the highest performance** across datasets  
- Transformer and graph-based models are competitive but do not outperform ensemble methods  
- Strong feature engineering reduces the advantage of deep architectures  
- Ranking-based evaluation confirms superior early fraud detection by ensemble models  
- Advanced models require higher computational cost without clear performance gains  

---

## Business Implications
- Fraud detection is a **ranking problem**, not just classification  
- Ensemble models provide:
  - Strong early fraud capture  
  - Lower false positives in top-ranked transactions  
  - Better operational efficiency  

- Simpler models can deliver:
  - High performance  
  - Lower infrastructure cost  
  - Easier deployment  

---
## Repository Structure

---

## Future Work
- Hybrid architectures combining ensemble and deep learning models  
- Improved transformer designs for tabular data  
- Selective graph-based modelling based on relational signal strength  
- Standardized benchmarking frameworks for fair model comparison  

---

## Author & Citation

**Author**  
Bhavik Parmar  

**Citation**  
Parmar, B. L. (2026, March 11). *Deep learning techniques for fraud detection* [Essay; Electronic].  
https://ucw.arcabc.ca/mbar-661/deep-learning-techniques-fraud-detection

---

## License
This project is intended for academic and research purposes.
