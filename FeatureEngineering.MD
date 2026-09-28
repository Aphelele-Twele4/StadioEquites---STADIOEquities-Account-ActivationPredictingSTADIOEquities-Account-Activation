# Feature Engineering

**Script:** [`experiments/feature_engineering.py`](experiments/feature_engineering.py)
**Input:** `Data/processed/bank_marketing_clean.csv`
**Output:** `Data/processed/train.csv`, `Data/processed/test.csv`

## What it does

1. **`was_previously_contacted` flag** — the raw `pdays` column uses `999` as a sentinel meaning "never contacted before this campaign." Left as-is, a model would wrongly read `999` as an enormous numeric gap. This is split into:
   - `was_previously_contacted` (0/1), and
   - `pdays_capped` (the real day-count, or `-1` where not applicable).

   This is the same pattern we'll need for STADIOEquities: any "not applicable" sentinel in the funding/behavioural history (e.g. no prior deposit) must become an explicit flag rather than a misleading number.

2. **Age banding** — `age` is bucketed into `<=25, 26-35, 36-45, 46-55, 56-65, 65+`. This gives the models an easy, human-checkable way to pick up non-linear age effects and produces a feature that will transfer cleanly onto STADIOEquities' own client age data.

3. **One-hot encoding** — every categorical column (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`, plus the new `age_band`) is expanded into binary indicator columns (`drop_first=False`, so every category is explicit and auditable).

4. **Stratified 80/20 train/test split** — `train_test_split(..., stratify=y, random_state=42)` keeps the ~11.3% positive rate consistent across both splits, so neither the training data nor the held-out evaluation set is accidentally easier or harder than the other.

## How to use it

```bash
cd experiments
python feature_engineering.py
```

Run this after `preprocessing.py`. It reads the cleaned CSV and writes the two files both `model1_logistic_regression.py` and `model2_random_forest.py` consume.

## Result

| | Train | Test |
|---|---|---|
| Rows | 32,950 | 8,238 |
| Columns (features + target) | 70 | 70 |
| Positive class rate | 11.27% | 11.26% |

See [`Model1.MD`](Model1.MD) and [`Model2.MD`](Model2.MD) for the modelling step.
