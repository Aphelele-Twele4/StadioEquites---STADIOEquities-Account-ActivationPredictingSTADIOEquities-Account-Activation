# Model 2 Performance: Random Forest

**Script:** [`experiments/model2_random_forest.py`](experiments/model2_random_forest.py) (metrics are computed and saved at the end of this same script — there is no separate performance script for Model 2).
**Metrics file:** `results/metrics/model2_random_forest_metrics.json`
**Feature importances:** `results/metrics/model2_feature_importances.csv`

## How to reproduce

```bash
cd experiments
python model2_random_forest.py
```

## Results (held-out test set, 8,238 rows, ~11.3% positive)

| Metric | Value | What it means here |
|---|---|---|
| Accuracy | 0.875 | 87.5% of clients correctly classified. |
| Precision (activated) | 0.459 | Of everyone the model flags as likely to activate, 45.9% actually do. |
| Recall (activated) | 0.612 | Of everyone who actually activates, the model correctly flags 61.2% of them. |
| F1-score (activated) | 0.524 | Harmonic mean of precision and recall for the minority (activated) class. |
| ROC AUC | 0.806 | The model ranks a randomly chosen activator above a randomly chosen non-activator 80.6% of the time. |

Confusion matrix (rows = actual, columns = predicted; 0 = did not activate, 1 = activated):

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 6,640 | 670 |
| **Actual 1** | 360 | 568 |

## Feature importance (top 5, Gini importance)

| Feature | Importance |
|---|---|
| `euribor3m` (Euribor 3-month rate) | 0.140 |
| `nr_employed` (number employed, economy-wide) | 0.102 |
| `age` | 0.079 |
| `emp_var_rate` (employment variation rate) | 0.074 |
| `campaign` (number of contacts this campaign) | 0.047 |

The dominance of macro-economic indicators echoes Moro et al.'s (2014) own finding that social/economic context carries real predictive signal for this problem — for STADIOEquities, the direct analogue would be macro/market-sentiment indicators and the client's acquisition-channel/contact-cadence history.

See [`Comparison.MD`](Comparison.MD) for how this stacks up against Model 1.
