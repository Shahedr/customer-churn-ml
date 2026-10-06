from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/customer_churn.csv")
TARGET = "churn"


def evaluate_model(name, model, x_test, y_test):
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "model": name,
        "roc_auc": roc_auc_score(y_test, probabilities),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }


def main():
    if not DATA_PATH.exists():
        from generate_data import main as generate_data

        generate_data()

    df = pd.read_csv(DATA_PATH)
    x = df.drop(columns=TARGET)
    y = df[TARGET]

    numeric_features = ["tenure_months", "monthly_charges", "support_tickets"]
    categorical_features = ["contract_type", "autopay", "internet_service"]

    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),
    }

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    result_rows = []
    fitted_models = {}

    for name, estimator in models.items():
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        result_rows.append(evaluate_model(name, pipeline, x_test, y_test))
        fitted_models[name] = pipeline

    metrics = pd.DataFrame(result_rows).sort_values("roc_auc", ascending=False)

    Path("results").mkdir(exist_ok=True)
    Path("assets").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    metrics.to_csv("results/model_metrics.csv", index=False)

    best_model_name = metrics.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    joblib.dump(best_model, "models/churn_model.joblib")

    RocCurveDisplay.from_estimator(best_model, x_test, y_test)
    plt.title(f"ROC Curve — {best_model_name}")
    plt.tight_layout()
    plt.savefig("assets/roc_curve.png", dpi=160)
    plt.close()

    ConfusionMatrixDisplay.from_estimator(best_model, x_test, y_test)
    plt.title(f"Confusion Matrix — {best_model_name}")
    plt.tight_layout()
    plt.savefig("assets/confusion_matrix.png", dpi=160)
    plt.close()

    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
