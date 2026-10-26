# SS3 — Model validation and results

**Notebook:** [`SS3_PartB.ipynb`](SS3_PartB.ipynb), Sections 4 and 5
**Outputs:** `results/metrics/ss3_random_forest_metrics.json`, `results/reports/ss3_test_comparison.csv`, `results/reports/ss3_figures/`

## Metrics (threshold 0.548, fixed on validation)

| Metric | Train | Validation | **Test (unseen)** |
|---|---|---|---|
| ROC AUC | 0.873 | 0.832 | **0.820** (95% CI 0.771 – 0.864) |
| PR AUC | 0.739 | 0.665 | **0.534** (base rate 0.230) |
| Recall (activated) | 0.838 | 0.843 | **0.818** |
| Precision (activated) | 0.616 | 0.582 | **0.491** |
| F1 (activated) | 0.710 | 0.688 | **0.614** |
| Accuracy | 0.796 | 0.774 | **0.763** |

**Test confusion matrix:** 81 activators found, 18 missed; 247 non-activators correctly left alone, 84 flagged unnecessarily.

## Against baselines (unseen test set)

| Model | ROC AUC | PR AUC | F1 | Brier |
|---|---|---|---|---|
| No-skill (prior rate) | 0.500 | 0.230 | 0.000 | 0.182 |
| Rule: KYC done in 48 h | 0.767 | 0.428 | 0.598 | 0.244 |
| Logistic Regression (SS2 Model 1) | 0.811 | 0.536 | 0.589 | 0.173 |
| **Random Forest (final)** | **0.820** | 0.534 | **0.614** | 0.173 |
| Random Forest, calibrated | 0.818 | 0.539 | 0.616 | **0.137** |

Paired bootstrap (2,000 resamples) for the RF − LogReg AUC difference: +0.009, 95% CI [−0.010, +0.029]. As in SS2, the two models rank accounts about equally well; the Random Forest is ahead on recall, precision and F1.

## Interpretation

- **It generalises forward in time.** Test AUC (0.820) sits close to validation (0.832) and in line with SS2's results on the public data.
- **Precision falls on test because activation fell.** The newest cohort activates at 23% versus 30% in training.
- **It ranks well enough to target nudges.** The top 30% of scores hold 68% of activators. Deciles 1–2 activate at about 2.5× the average rate.
- **What drives it.** KYC completion within 48 h, and how fast it happened, dominate; then age, channel and campaign. Behavioural features carry little signal because **65% of eligible accounts have no app/web activity in the first 48 h**. A later checkpoint would capture more behaviour at the cost of later intervention, and `CHECKPOINT_DAYS` is a single parameter to change.
- **Calibration helps.** It reduces the Brier score from 0.173 to 0.137 without changing the ranking, so the calibrated scores can drive a cost-based nudge rule.

## Figures (notebook §5, also in `results/reports/ss3_figures/`)

| # | Figure | Phase |
|---|---|---|
| 01 | Positive share per subset; OOB error vs number of trees | Training |
| 02 | Grid-search heatmaps: validation AUC and overfitting gap | Training |
| 03 | Metrics on train vs validation vs test | Training + unseen |
| 04 | ROC and precision–recall curves | Training + unseen |
| 05 | Confusion matrices, train vs test | Training + unseen |
| 06 | Permutation importance | Unseen |
| 07 | Activation rate by score decile; calibration curve | Unseen |
| 08 | Final model vs baselines | Unseen |

## Limitations and next steps

- The test set is small (99 positives), hence the wide AUC interval.
- The data is synthetic, and two README claims (R180 acquisition cost, 41% never funded) don't match it. Confirm both with STADIOEquities.
- Next steps: try a day-5 or day-7 checkpoint, benchmark gradient boosting (Part D improvement 3), and set the threshold from the real costs of a nudge versus a missed activation.
