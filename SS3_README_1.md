# SS3 Part B — Final model applied to STADIOEquities' client data

**Notebook:** [`SS3_PartB.ipynb`](SS3_PartB.ipynb)
**Final method (from SS2 Part D):** Random Forest, with Logistic Regression kept as the interpretable benchmark.

SS2 proved the approach on the public UCI Bank Marketing dataset. SS3 Part B applies the same method to the four extracts STADIOEquities supplied, and also carries out the improvements the Part D report recommended:

| Part D recommendation | Where it is done in SS3 |
|---|---|
| Replace bank-specific features with onboarding / behavioural ones | Notebook §2.4 |
| Redefine the target as funded within 90 days, built only from data available at an early checkpoint | §2.3 (48-hour checkpoint) |
| Out-of-time validation by registration cohort | §2.5 (chronological 70 / 15 / 15 split) |
| Systematic hyperparameter tuning | §3.1–3.2 (OOB curve + 48-combination grid) |
| Probability calibration | §3.4, compared in §4.3 and §5.7 |
| Re-tune imbalance handling and threshold for the client's own class balance | §3.3 and §3.5 |

## Where the marked components are

| Component | Notebook section | Supporting README |
|---|---|---|
| Dataset loading | **§1** | [`SS3_DataLoading_Preprocessing.MD`](SS3_DataLoading_Preprocessing.MD) |
| Dataset preprocessing (incl. train / validation / test split) | **§2** | [`SS3_DataLoading_Preprocessing.MD`](SS3_DataLoading_Preprocessing.MD) |
| Model training (incl. hyperparameter justification) | **§3** | [`SS3_ModelTraining.MD`](SS3_ModelTraining.MD) |
| Model validation | **§4** | [`SS3_Validation_Results.MD`](SS3_Validation_Results.MD) |
| Results visualisation | **§5** | [`SS3_Validation_Results.MD`](SS3_Validation_Results.MD) |

Each component in the notebook begins with a blue banner and a heading of the form `# N. COMPONENT NAME`.

## How to run

1. Put the four client files, unmodified, in `Data/raw/client/`:
   `1_account_registration.csv`, `2_app_web_behavioural.csv`, `3_client_demographics.csv`, `4_marketing_acquisition.csv`
2. Install the dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib jupyter
   ```
3. From the repository root, open and run `SS3_PartB.ipynb` top to bottom (*Kernel → Restart & Run All*). A full run takes about three minutes; most of that is the grid search. All randomness is seeded (`random_state=42`), so results reproduce exactly.

## Outputs the notebook writes

| File | Contents |
|---|---|
| `Data/processed/ss3_model_table.csv` | One row per eligible account: engineered features, label and subset |
| `models/ss3_random_forest.pkl` | Trained Random Forest, decision threshold, feature list, training-set age median |
| `results/metrics/ss3_random_forest_metrics.json` | Train / validation / test metrics, confusion matrix, confidence intervals |
| `results/metrics/ss3_rf_tuning_grid.csv` | All 48 grid-search results |
| `results/metrics/ss3_feature_importances.csv` | Permutation and Gini importances |
| `results/reports/ss3_test_comparison.csv` | Final model vs baselines on the unseen test set |
| `results/reports/ss3_figures/*.png` | The eight result figures |

## Headline result (unseen test set: 430 accounts, Nov 2025 – Jun 2026)

| | Random Forest | Logistic Regression (benchmark) | Rule: "KYC done in 48 h" |
|---|---|---|---|
| ROC AUC | **0.820** | 0.811 | 0.767 |
| Recall (activated) | **0.818** | 0.788 | 0.788 |
| Precision (activated) | **0.491** | 0.470 | 0.481 |
| F1 (activated) | **0.614** | 0.589 | 0.598 |

The top 30% of accounts by score contain 68% of the accounts that went on to activate. See [`SS3_Validation_Results.MD`](SS3_Validation_Results.MD) for the full discussion and limitations.
