# Model 1 Performance: Logistic Regression

**Script:** [`experiments/model1_logistic_regression.py`](experiments/model1_logistic_regression.py) (metrics are computed and saved at the end of this same script — there is no separate performance script for Model 1).
**Metrics file:** `results/metrics/model1_logistic_regression_metrics.json`

## How to reproduce

```bash
cd experiments
python model1_logistic_regression.py
```

## Results (held-out test set, 8,238 rows, ~11.3% positive)

| Metric | Value | What it means here |
|---|---|---|
| Accuracy | 0.834 | 83.4% of clients correctly classified — but with an 88.7%/11.3% class split, a model that always predicted "no" would already score ~0.887, so accuracy alone is not a useful metric for this problem. |
| Precision (activated) | 0.366 | Of everyone the model flags as likely to activate, 36.6% actually do. |
| Recall (activated) | 0.650 | Of everyone who actually activates, the model correctly flags 65.0% of them. |
| F1-score (activated) | 0.468 | Harmonic mean of precision and recall for the minority (activated) class. |
| ROC AUC | 0.802 | The model ranks a randomly chosen activator above a randomly chosen non-activator 80.2% of the time — the standard metric for this literature (Moro et al., 2014, report AUC in the same range for their logistic regression benchmark). |

Confusion matrix (rows = actual, columns = predicted; 0 = did not activate, 1 = activated):

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 6,266 | 1,044 |
| **Actual 1** | 325 | 603 |

## Interpretation

Because `class_weight='balanced'` was used, the model deliberately trades some precision for higher recall on the minority class — it would rather over-flag borderline clients than miss genuine activators. For a targeted-nudge use case (the STADIOEquities motivation: reach the accounts about to churn or activate) this trade-off is usually the right one, since the cost of an extra nudge is far lower than the cost of a missed activation.

See [`Comparison.MD`](Comparison.MD) for how this stacks up against Model 2.
