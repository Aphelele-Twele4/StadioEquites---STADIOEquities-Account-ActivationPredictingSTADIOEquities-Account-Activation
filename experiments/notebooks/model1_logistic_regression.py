"""
model1_logistic_regression.py
------------------------------
Part B / Model 1: Logistic Regression baseline.

Rationale: Moro, Cortez & Rita (2014) — the paper behind this dataset —
compared logistic regression, decision trees, a neural network and an
SVM, and logistic regression is the standard, most interpretable
baseline for this type of binary response-prediction problem. It gives
STADIOEquities a transparent, coefficient-level view of which features
push an account towards activation, which matters for a business
audience as much as raw predictive power does.

Hyperparameters:
  - class_weight='balanced'  -> compensates for the ~11:89 class split
    without discarding data (no under/over-sampling needed).
  - max_iter=2000            -> ensures convergence with the expanded
    one-hot feature set.
  - solver='lbfgs'           -> default, suitable for this feature count.
  - C=1.0                    -> default L2 regularisation strength
    (kept at default for this baseline; see Model1.MD for tuning notes).
  - Numeric features are standardised (mean 0, unit variance) via a
    ColumnTransformer inside a Pipeline, since logistic regression is
    scale-sensitive; the one-hot encoded columns are passed through
    unchanged.

Usage:
    python model1_logistic_regression.py
"""
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "Data" / "processed" / "train.csv"
TEST_PATH = BASE / "Data" / "processed" / "test.csv"
MODEL_OUT = BASE / "models" / "model1_logistic_regression.pkl"
METRICS_OUT = BASE / "results" / "metrics" / "model1_logistic_regression_metrics.json"

NUMERIC_COLS = ["age", "campaign", "previous", "pdays_capped",
                "emp_var_rate", "cons_price_idx", "cons_conf_idx",
                "euribor3m", "nr_employed"]


def build_pipeline(numeric_cols, all_cols):
    other_cols = [c for c in all_cols if c not in numeric_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), numeric_cols),
            ("passthrough", "passthrough", other_cols),
        ]
    )
    clf = LogisticRegression(
        class_weight="balanced", max_iter=2000, solver="lbfgs", C=1.0, random_state=42
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    y_train, y_test = train_df["y"], test_df["y"]
    X_train, X_test = train_df.drop(columns=["y"]), test_df.drop(columns=["y"])

    numeric_cols = [c for c in NUMERIC_COLS if c in X_train.columns]
    pipeline = build_pipeline(numeric_cols, X_train.columns.tolist())

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "Logistic Regression",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(pipeline, f)

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {MODEL_OUT}")
    print(f"Saved metrics to {METRICS_OUT}")


if __name__ == "__main__":
    main()
