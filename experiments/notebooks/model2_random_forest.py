"""
model2_random_forest.py
------------------------
Part B / Model 2: Random Forest ensemble.

Rationale: the related-work review (see the Part A report) found that
non-linear / ensemble approaches consistently outperform linear
baselines on this exact dataset — an autoencoder beat PCA-style linear
compression (Liang et al., 2024), and a heterogeneous tree-of-predictors
ensemble beat single global models including XGBoost (Yoon, Zame & van
der Schaar, 2018). A Random Forest is a simpler, well-understood way to
capture that same "different learners for different client sub-groups"
intuition, and it needs no feature scaling or distributional
assumptions, which suits the mix of numeric and one-hot categorical
features produced by feature_engineering.py.

Hyperparameters:
  - n_estimators=300     -> enough trees for a stable AUC estimate
    without excessive training time on this feature count.
  - max_depth=None       -> trees grow until leaves are pure or hit
    min_samples_leaf; regularisation is handled via min_samples_leaf.
  - min_samples_leaf=5   -> light regularisation to reduce overfitting
    on the noisier one-hot columns.
  - class_weight='balanced_subsample' -> re-weights classes within each
    bootstrap sample to counter the ~11:89 imbalance.
  - random_state=42      -> reproducibility.

Usage:
    python model2_random_forest.py
"""
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "Data" / "processed" / "train.csv"
TEST_PATH = BASE / "Data" / "processed" / "test.csv"
MODEL_OUT = BASE / "models" / "model2_random_forest.pkl"
METRICS_OUT = BASE / "results" / "metrics" / "model2_random_forest_metrics.json"
IMPORTANCES_OUT = BASE / "results" / "metrics" / "model2_feature_importances.csv"


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    y_train, y_test = train_df["y"], test_df["y"]
    X_train, X_test = train_df.drop(columns=["y"]), test_df.drop(columns=["y"])

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "Random Forest",
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
        pickle.dump(clf, f)

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    importances = (
        pd.Series(clf.feature_importances_, index=X_train.columns)
        .sort_values(ascending=False)
    )
    importances.to_csv(IMPORTANCES_OUT, header=["importance"])

    print(f"Saved model to {MODEL_OUT}")
    print(f"Saved metrics to {METRICS_OUT}")
    print(f"Saved feature importances to {IMPORTANCES_OUT}")


if __name__ == "__main__":
    main()
