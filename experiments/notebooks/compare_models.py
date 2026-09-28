"""
compare_models.py
------------------
Part C: Loads the saved metrics for Model 1 (Logistic Regression) and
Model 2 (Random Forest) and produces a single side-by-side comparison
table, plus a statistical check (DeLong-style bootstrap CI on the AUC
difference) for whether the difference in ROC AUC is likely to be real
or just noise given the size of the test set.

Usage:
    python compare_models.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import pickle

BASE = Path(__file__).resolve().parents[1]
M1_METRICS = BASE / "results" / "metrics" / "model1_logistic_regression_metrics.json"
M2_METRICS = BASE / "results" / "metrics" / "model2_random_forest_metrics.json"
M1_MODEL = BASE / "models" / "model1_logistic_regression.pkl"
M2_MODEL = BASE / "models" / "model2_random_forest.pkl"
TEST_PATH = BASE / "Data" / "processed" / "test.csv"
OUT_TABLE = BASE / "results" / "reports" / "comparison_table.csv"
OUT_BOOTSTRAP = BASE / "results" / "metrics" / "auc_bootstrap_comparison.json"


def bootstrap_auc_diff(y_true, proba1, proba2, n_boot=2000, seed=42):
    """Paired bootstrap on the test set to estimate a 95% CI for the
    difference in ROC AUC between two models scored on the same rows."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    y_true = np.asarray(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        auc1 = roc_auc_score(yb, proba1[idx])
        auc2 = roc_auc_score(yb, proba2[idx])
        diffs.append(auc2 - auc1)
    diffs = np.array(diffs)
    return {
        "mean_diff_rf_minus_logreg": float(diffs.mean()),
        "ci_lower_2_5pct": float(np.percentile(diffs, 2.5)),
        "ci_upper_97_5pct": float(np.percentile(diffs, 97.5)),
        "n_bootstrap": int(len(diffs)),
        "excludes_zero": bool(np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0),
    }


def main():
    with open(M1_METRICS) as f:
        m1 = json.load(f)
    with open(M2_METRICS) as f:
        m2 = json.load(f)

    rows = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    table = pd.DataFrame({
        "Metric": rows,
        "Model 1 (Logistic Regression)": [m1[r] for r in rows],
        "Model 2 (Random Forest)": [m2[r] for r in rows],
    })
    table["Difference (RF - LogReg)"] = (
        table["Model 2 (Random Forest)"] - table["Model 1 (Logistic Regression)"]
    )
    print(table.to_string(index=False))

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_TABLE, index=False)

    # Statistical comparison of the AUC difference via paired bootstrap
    test_df = pd.read_csv(TEST_PATH)
    y_test = test_df["y"]
    X_test = test_df.drop(columns=["y"])

    with open(M1_MODEL, "rb") as f:
        pipe1 = pickle.load(f)
    with open(M2_MODEL, "rb") as f:
        pipe2 = pickle.load(f)

    proba1 = pipe1.predict_proba(X_test)[:, 1]
    proba2 = pipe2.predict_proba(X_test)[:, 1]

    boot_result = bootstrap_auc_diff(y_test, proba1, proba2)
    print(json.dumps(boot_result, indent=2))

    OUT_BOOTSTRAP.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_BOOTSTRAP, "w") as f:
        json.dump(boot_result, f, indent=2)

    print(f"Saved comparison table to {OUT_TABLE}")
    print(f"Saved bootstrap AUC comparison to {OUT_BOOTSTRAP}")


if __name__ == "__main__":
    main()
