# Model Comparison: Logistic Regression vs. Random Forest

**Script:** [`experiments/compare_models.py`](experiments/compare_models.py)
**Inputs:** `results/metrics/model1_logistic_regression_metrics.json`, `results/metrics/model2_random_forest_metrics.json`, both saved model pipelines, and the shared held-out test set.
**Outputs:** `results/reports/comparison_table.csv`, `results/metrics/auc_bootstrap_comparison.json`

## How to reproduce

```bash
cd experiments
python compare_models.py
```

This must be run after both `model1_logistic_regression.py` and `model2_random_forest.py`, since it loads their saved models and metrics.

## Side-by-side results (same held-out test set, 8,238 rows)

| Metric | Model 1 (Logistic Regression) | Model 2 (Random Forest) | Difference (RF − LogReg) |
|---|---|---|---|
| Accuracy | 0.834 | 0.875 | +0.041 |
| Precision (activated) | 0.366 | 0.459 | +0.093 |
| Recall (activated) | 0.650 | 0.612 | −0.038 |
| F1-score (activated) | 0.468 | 0.524 | +0.056 |
| ROC AUC | 0.802 | 0.806 | +0.004 |

## Statistical comparison of the AUC difference

A paired bootstrap (2,000 resamples of the shared test set, evaluating both models on each resample) was used to estimate a 95% confidence interval for the AUC difference, since the raw point-estimate gap (+0.004) is small enough that it could plausibly be noise:

| | Value |
|---|---|
| Mean AUC difference (RF − LogReg) | +0.004 |
| 95% CI | [−0.005, +0.013] |
| CI excludes zero? | **No** |

Because the confidence interval spans zero, the difference in ranking ability (AUC) between the two models on this dataset is **not statistically distinguishable from zero** at the 95% level — i.e. the two models discriminate activators from non-activators about equally well overall.

## Where the models actually differ

The AUC tie masks a real, and business-relevant, trade-off:

- **Random Forest** trades some recall for a large gain in precision (+0.093) and F1 (+0.056) and a meaningfully higher accuracy (+0.041). It is the better choice when the cost of a wasted nudge (a false positive) matters — e.g. limited outreach capacity or a per-contact cost.
- **Logistic Regression** keeps recall higher (0.650 vs. 0.612), i.e. it misses fewer true activators. It is the better choice when missing a genuine activator is costlier than an extra nudge, and its coefficients remain directly explainable to a non-technical stakeholder — useful when the business needs to *justify* why an account was flagged, not just get the flag.

## Recommendation for this proof-of-concept

Given near-identical AUC, the choice between the two should be driven by STADIOEquities' actual cost structure (cost of a wasted nudge vs. cost of a missed activation) rather than by predictive power alone — a point developed further in the Part D recommendations report. Both models comfortably beat a random baseline (AUC 0.5) and are in the same range reported by Moro et al. (2014) for this dataset, which supports the viability of this general modelling approach before it is applied to STADIOEquities' own account data.
