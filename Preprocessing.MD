# Preprocessing

**Script:** [`experiments/preprocessing.py`](experiments/preprocessing.py)
**Input:** `Data/raw/bank-additional-full.csv`
**Output:** `Data/processed/bank_marketing_clean.csv`

## What it does

1. **Loads** the raw UCI Bank Marketing dataset (41,188 rows, 21 columns, `;`-delimited).
2. **Standardises column names** — replaces dots with underscores (`emp.var.rate` → `emp_var_rate`) so every downstream tool can reference columns without quoting.
3. **Drops `duration`** — this is target leakage. Call duration is only known once the call (and therefore the outcome) has already happened, so a model scoring a client *before* contacting them could never have this feature. This mirrors a rule we will carry over to STADIOEquities' data: any feature only populated after the activation event itself must be excluded from the live-scoring feature set.
4. **Encodes the target** — `y` (`"yes"`/`"no"`) is converted to a binary integer (1 = subscribed / activated, 0 = did not).
5. **Strips whitespace** from all text columns.
6. **Leaves `"unknown"` categorical values in place** rather than imputing them, since a client's choice not to disclose (e.g. education = `unknown`) can itself carry signal, and STADIOEquities' own onboarding data will very likely contain the same kind of "not yet provided" categories.
7. **Validates** the result: asserts there are no nulls and that the target is strictly binary before saving.

## How to use it

```bash
cd experiments
python preprocessing.py
```

No arguments are required — paths are resolved relative to the script location. Re-run it any time `Data/raw/bank-additional-full.csv` changes.

## Result

| | Raw | Cleaned |
|---|---|---|
| Rows | 41,188 | 41,188 |
| Columns | 21 | 20 (`duration` dropped) |
| Class balance (target = activated) | 11.27% | 11.27% |

See [`FeatureEngineering.MD`](FeatureEngineering.MD) for the next step.
