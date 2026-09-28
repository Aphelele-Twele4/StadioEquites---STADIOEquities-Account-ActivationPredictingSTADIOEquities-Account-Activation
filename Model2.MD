# Model 2: Random Forest

**Script:** [`experiments/model2_random_forest.py`](experiments/model2_random_forest.py)
**Input:** `Data/processed/train.csv`, `Data/processed/test.csv`
**Output:** `models/model2_random_forest.pkl`, `results/metrics/model2_random_forest_metrics.json`, `results/metrics/model2_feature_importances.csv`

## Why this model

The related-work review (Part A) found that non-linear / ensemble approaches consistently beat linear baselines on this exact dataset: an autoencoder produced a lower reconstruction error than PCA-style linear compression (Liang et al., 2024), and a heterogeneous "tree of predictors" ensemble beat single global models including XGBoost (Yoon, Zame & van der Schaar, 2018). A Random Forest is a simpler, well-understood way to capture that same "different sub-populations of clients behave differently" intuition as a second, contrasting model to the linear baseline in Model 1 — and it needs no feature scaling or distributional assumptions.

## Preprocessing

None beyond what `feature_engineering.py` already produced — tree-based models do not require feature scaling, so the one-hot-encoded, capped/flagged feature set is used directly.

## Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | `300` | Enough trees for a stable ROC AUC estimate without excessive training time on ~70 features / 33k rows. |
| `max_depth` | `None` | Trees grow until leaves are pure or hit `min_samples_leaf`; depth is not hard-capped. |
| `min_samples_leaf` | `5` | Light regularisation to reduce overfitting on the noisier one-hot columns. |
| `class_weight` | `balanced_subsample` | Re-weights classes within each bootstrap sample to counter the ~11:89 imbalance. |
| `random_state` | `42` | Reproducibility. |

## How to use it

```bash
cd experiments
python model2_random_forest.py
```

Requires `train.csv`/`test.csv` from `feature_engineering.py` to already exist.

## Result (held-out test set, 8,238 rows)

| Metric | Value |
|---|---|
| Accuracy | 0.875 |
| Precision (activated) | 0.459 |
| Recall (activated) | 0.612 |
| F1-score (activated) | 0.524 |
| ROC AUC | 0.806 |

Top predictive features by Gini importance: `euribor3m` (0.140), `nr_employed` (0.102), `age` (0.079), `emp_var_rate` (0.074), `campaign` (0.047) — the macro-economic context indicators dominate, echoing Moro et al.'s (2014) finding that socio-economic attributes carry real signal for this kind of response prediction. Full ranking is in `results/metrics/model2_feature_importances.csv`.

See [`Model2Performance.MD`](Model2Performance.MD) (Part C) for discussion.
