# SS3 — Model training

**Notebook:** [`SS3_PartB.ipynb`](SS3_PartB.ipynb), Section 3
**Output:** `models/ss3_random_forest.pkl`, `results/metrics/ss3_rf_tuning_grid.csv`

## Procedure

1. **Number of trees.** The out-of-bag (OOB) error on the training set was tracked as the forest grew from 25 to 700 trees. It settles at about 0.207 from 100 trees onward.
2. **Tree shape.** A grid search over 48 combinations: each candidate was trained on the training set and scored on the validation set.
3. **Selection rule.** Highest validation ROC AUC; among candidates within 0.002 of the best, the one with the smallest train-minus-validation AUC gap.
4. **Final fit.** The winner was refit on the training set, and the decision threshold that maximises validation F1 was chosen.
5. **Benchmark and calibration.** Logistic Regression was trained with SS2's settings, and a sigmoid-calibrated copy of the forest was fitted with 5-fold cross-validation inside the training set.

## Final hyperparameters and why

| Hyperparameter | Value | Justification |
|---|---|---|
| `n_estimators` | 500 | Well past the OOB plateau. More trees stabilise the vote without overfitting. |
| `max_depth` | **4** (searched: 4, 8, 12, None) | Best validation AUC (0.832) *and* smallest overfitting gap (0.041). Unlimited-depth forests reached train AUC 1.0 but validated worse. |
| `min_samples_leaf` | **5** (searched: 1, 5, 10, 20) | Within 0.001 of the best validation AUC, with a smaller gap than leaf size 1 |
| `max_features` | **sqrt** (searched: sqrt, 0.3, 0.5) | Decorrelates the trees; best on validation |
| `class_weight` | `balanced_subsample` | 28.8% positives. Re-weighting keeps every row and creates no synthetic accounts. |
| Decision threshold | **0.548** | Maximises F1 on validation (0.688). Balances wasted nudges (precision) against missed activators (recall). |
| `random_state` | 42 | Reproducibility |

**Learning rate, optimiser and epochs** apply to gradient-trained models such as neural networks or boosting. A Random Forest has none: each tree is grown greedily in one pass. The equivalent controls are the number of trees and the capacity / regularisation settings above, and those are what were tuned.

## Benchmark settings (unchanged from SS2 Model 1)

Logistic Regression: `StandardScaler` on numeric columns (fitted on training only), `class_weight="balanced"`, `C=1.0`, `solver="lbfgs"`, `max_iter=2000`. Its threshold (0.425) was also chosen on validation F1.

## Changes from SS2

| | SS2 (public data) | SS3 (client data) |
|---|---|---|
| Hyperparameters | Near-default (300 trees, unlimited depth, leaf 5) | Tuned on a validation set |
| Split | Random 80/20 | Chronological 70/15/15 |
| Threshold | 0.5 | Chosen on validation |
| Calibration | None | Sigmoid, compared on Brier score |
