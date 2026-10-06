# Customer Churn Prediction

This project is a small model-comparison exercise built around a churn question: **can a simple, interpretable model perform as well as a more flexible tree-based model?**

I used a synthetic dataset so the full workflow can stay public and reproducible. It is useful for demonstrating the modeling process, but I do not treat the results as if they came from a real customer base.

## Setup

The generated dataset contains **5,000 customers** with:

- tenure
- monthly charges
- support-ticket count
- contract type
- autopay status
- internet-service type
- churn flag

The churn probability is generated from those features with controlled randomness, then the model sees only the finished dataset—not the formula used to create the target.

## Models compared

- Logistic Regression
- Random Forest

Both models use the same train/test split and the same preprocessing pipeline for numeric and categorical features.

## Results

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.745 | 0.694 | 0.605 | 0.704 | 0.651 |
| Random Forest | 0.728 | 0.668 | 0.582 | 0.641 | 0.610 |

The simpler Logistic Regression model performed better on this generated test set, including a higher ROC-AUC and recall. For a retention use case, I would pay particular attention to recall and decision-threshold tradeoffs rather than choosing a model from accuracy alone.

## Why I kept Logistic Regression in the comparison

Churn work often gets framed as “use the strongest model,” but an interpretable baseline matters. If two models are close, a simpler model can be easier to explain, monitor, and challenge with business stakeholders.

## Run it

```bash
pip install -r requirements.txt
python src/train.py
```

The script:

1. generates the dataset if it is not already present
2. splits the data with stratification
3. preprocesses numeric and categorical fields
4. trains both models
5. writes model metrics
6. saves the best-performing fitted pipeline locally
7. creates ROC and confusion-matrix plots locally

## Files

```text
customer-churn-ml/
├── src/
│   ├── generate_data.py
│   └── train.py
├── results/
│   └── model_metrics.csv
├── requirements.txt
└── README.md
```

Generated data, model files, and plot images are intentionally ignored from version control because they can be recreated by running the script.

## Stack

Python · pandas · scikit-learn · Matplotlib · joblib

## Limitations and next steps

Because the data is synthetic, the reported metrics are useful for comparing the workflow—not for making a real retention decision.

With real customer data, my next steps would be cross-validation, threshold tuning, calibration, class-cost analysis, feature review for leakage, and an explanation layer for the final model.
