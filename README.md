# Customer Churn Prediction

End-to-end binary classification project for identifying customers at higher risk of churn and comparing interpretable and non-linear models.

## Business problem
Retention teams need a way to prioritize outreach. This project builds a reproducible classification workflow using customer tenure, pricing, support activity, contract type, autopay status, and service type.

## Dataset
A deterministic **5,000-row synthetic customer dataset** is generated locally, so the project is reproducible and contains no private customer information.

## Models compared
- Logistic Regression
- Random Forest

## Model results
| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.745 | 0.694 | 0.605 | 0.704 | 0.651 |
| Random Forest | 0.728 | 0.668 | 0.582 | 0.641 | 0.610 |

## Tools
**Python · pandas · scikit-learn · preprocessing pipelines · classification · model evaluation · joblib · Matplotlib**

## Repository structure
```text
customer-churn-ml/
├── src/
│   ├── generate_data.py
│   └── train.py
├── results/model_metrics.csv
├── requirements.txt
└── README.md
```

Running the training script also generates the dataset, fitted model, ROC curve, and confusion matrix locally.

## Run
```bash
pip install -r requirements.txt
python src/train.py
```

## What this demonstrates
Feature preprocessing, train/test splitting, class-aware modeling, model comparison, ROC-AUC evaluation, reproducible pipelines, model persistence, and business-oriented interpretation of classification metrics.
