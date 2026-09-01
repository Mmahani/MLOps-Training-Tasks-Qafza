# Qafza MLOps Scholarship Training

This repository contains my practical work and assignments for the **Qafza MLOps Scholarship Training**. The training follows a 12-week roadmap that progresses from local machine-learning pipelines to a complete production-ready MLOps system.

## Repository

[GitHub Repository](https://github.com/Mmahani/MLOps-Training-Tasks-Qafza)

## Training Roadmap

| Week | Topic | Main Tools | Practical Task |
|---|---|---|---|
| 1 | Leakage-proof ML Pipeline | Python, Scikit-learn | Build a clean pipeline with preprocessing, cross-validation, and model persistence |
| 2 | Deep Learning Pipeline | PyTorch, Hugging Face | Train a deep-learning model and export it to ONNX |
| 3 | Production API | FastAPI, Pydantic | Build a validated REST inference API |
| 4 | Containerization | Docker | Package the API and model in a portable container |
| 5 | ETL Pipeline | Python ETL, Database | Build an automated data-ingestion and transformation pipeline |
| 6 | Data Versioning | DVC | Version datasets and model artifacts |
| 7 | Experiment Tracking | MLflow | Track and compare machine-learning experiments |
| 8 | Distributed Training | Ray Core, Ray Train | Scale model training across available resources |
| 9 | Feature Store | Feast, Ray | Create reusable production features |
| 10 | Monitoring | Prometheus, Grafana | Monitor model and system health |
| 11 | Continuous Retraining | Ray Serve, GitHub Actions | Automate retraining and deployment |
| 12 | Infrastructure as Code | Terraform | Provision infrastructure programmatically |
| Final | End-to-End ML System | Complete Stack | Deliver and present a complete production MLOps system |

## Completed Assignments

### Assignment 1 — PostgreSQL and Data Preparation

The first assignment covers PostgreSQL and pgAdmin4. It includes database creation, table creation, CSV data loading, validation queries, record-count checks, and joins between related tables.

### Assignment 2 — Machine Learning Training Pipeline

The second assignment builds an order-level machine-learning workflow using the Olist dataset stored in PostgreSQL. The workflow includes reading the database tables, creating the `is_late` target, splitting the data, performing EDA, engineering features, training a baseline and Logistic Regression model, tuning hyperparameters, and evaluating the final model.

The Assignment 2 notebooks are:

```text
Assignment-2/
├── 01_read_and_join.ipynb
├── 02_create_labels.ipynb
├── 03_split_data.ipynb
├── 04_eda_train_only.ipynb
├── 05_feature_engineering.ipynb
└── 06_train_tune_evaluate.ipynb
```

## Assignment 2 Workflow

```text
PostgreSQL
    ↓
Read and join tables
    ↓
Create order-level labels
    ↓
Train / validation / test split
    ↓
EDA on training data
    ↓
Feature engineering
    ↓
Model training and evaluation
```

The target variable is defined as follows:

```text
is_late = 1  → delivered after the estimated delivery date
is_late = 0  → delivered on or before the estimated delivery date
```

## Technologies

```text
Python
PostgreSQL
pgAdmin4
Jupyter Notebook
pandas
NumPy
SQLAlchemy
scikit-learn
Matplotlib
Seaborn
Joblib
```

## Future Work

The next stages of the roadmap will extend the current work with deep learning, production APIs, Docker, ETL automation, data and experiment versioning, distributed training, feature stores, monitoring, continuous retraining, infrastructure as code, and an end-to-end capstone project.

## Notes

Run each assignment in the order specified by its notebooks. Do not upload passwords, private database files, or `.ipynb_checkpoints` directories to GitHub.

## Roadmap Reference

The roadmap is based on the Qafza MLOps Scholarship Training plan provided with this project.
