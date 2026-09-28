# Model 1: Logistic Regression

**Script:** [`experiments/model1_logistic_regression.py`](experiments/model1_logistic_regression.py)
**Input:** `Data/processed/train.csv`, `Data/processed/test.csv`
**Output:** `models/model1_logistic_regression.pkl`, `results/metrics/model1_logistic_regression_metrics.json`

## Why this model

Moro, Cortez & Rita (2014) — the paper behind this dataset — used logistic regression as one of four benchmark classifiers for this exact activation/subscription problem. It remains the standard first model for this kind of task because its coefficients are directly interpretable: STADIOEquities' growth team can see, feature by feature, what pushes an account towards or away from activation, not just a single "risk score."

## Preprocessing inside the model pipeline

Logistic regression is scale-sensitive, so a `ColumnTransformer` inside an sklearn `Pipeline`:
- applies `StandardScaler` (mean 0, unit variance) to the numeric columns (`age`, `campaign`, `previous`, `pdays_capped`, and the four socio-economic indicators), and
- passes the already one-hot-encoded categorical columns through unchanged.

The pipeline is fit only on the training split and applied unchanged to the test split, so no information leaks from test to train.

## Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| `class_weight` | `balanced` | Re-weights the loss to counter the ~11:89 class split, without discarding any of the 88.7% majority-class rows. |
| `solver` | `lbfgs` | Default solver, appropriate for this feature count (~70 columns) and dataset size. |
| `C` | `1.0` | Default L2 regularisation strength; used as the Part B baseline (a follow-up grid search over `C` is a natural Part C/D extension). |
| `max_iter` | `2000` | Raised from the sklearn default to guarantee convergence on the expanded one-hot feature set. |
| `random_state` | `42` | Reproducibility. |

## How to use it

```bash
cd experiments
python model1_logistic_regression.py
```

Requires `train.csv`/`test.csv` from `feature_engineering.py` to already exist.

## Result (held-out test set, 8,238 rows)

| Metric | Value |
|---|---|
| Accuracy | 0.834 |
| Precision (activated) | 0.366 |
| Recall (activated) | 0.650 |
| F1-score (activated) | 0.468 |
| ROC AUC | 0.802 |

Full metrics, including the confusion matrix and per-class report, are in `results/metrics/model1_logistic_regression_metrics.json`. See [`Model1Performance.MD`](Model1Performance.MD) (Part C) for discussion.
