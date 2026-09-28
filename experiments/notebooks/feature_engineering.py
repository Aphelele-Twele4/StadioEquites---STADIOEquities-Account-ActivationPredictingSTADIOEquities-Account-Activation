"""
feature_engineering.py
-----------------------
Part B: Feature engineering for the account-activation proof-of-concept.
Reads the cleaned dataset produced by preprocessing.py, engineers
features, and writes a stratified 80/20 train/test split ready for
modelling. See FeatureEngineering.MD for the full write-up.

What this script does:
  1. Loads Data/processed/bank_marketing_clean.csv.
  2. Engineers a 'was_previously_contacted' flag from the 'pdays'
     sentinel value (999 = never contacted before), since 999 is not a
     meaningful numeric distance and would otherwise mislead a model
     that treats pdays as continuous.
  3. Buckets 'age' into age bands (feature that is easy to reason about
     and to reuse later against STADIOEquities' own age data).
  4. One-hot encodes all categorical columns.
  5. Splits into train/test (80/20, stratified on the target to preserve
     the ~11:89 class balance in both splits).
  6. Saves Data/processed/train.csv and Data/processed/test.csv.

Usage:
    python feature_engineering.py
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

IN_PATH = Path(__file__).resolve().parents[1] / "Data" / "processed" / "bank_marketing_clean.csv"
TRAIN_PATH = Path(__file__).resolve().parents[1] / "Data" / "processed" / "train.csv"
TEST_PATH = Path(__file__).resolve().parents[1] / "Data" / "processed" / "test.csv"

CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome", "age_band",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # pdays == 999 is a sentinel for "never contacted before this
    # campaign", not a real distance in days. Turn it into an explicit
    # binary flag and cap the numeric value so it no longer dominates
    # the feature's scale.
    df["was_previously_contacted"] = (df["pdays"] != 999).astype(int)
    df["pdays_capped"] = df["pdays"].replace(999, -1)  # -1 = "not applicable"
    df = df.drop(columns=["pdays"])

    # Simple, interpretable age bands
    df["age_band"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
    ).astype(str)

    # One-hot encode categorical columns (drop_first=False keeps every
    # category explicit and auditable, at the cost of a few extra
    # columns — fine at this dataset size).
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    return df


def main():
    df = pd.read_csv(IN_PATH)
    print(f"Loaded cleaned data: {df.shape}")

    df_fe = engineer_features(df)
    print(f"Engineered feature set: {df_fe.shape}")

    y = df_fe["y"]
    X = df_fe.drop(columns=["y"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = X_train.copy()
    train_df["y"] = y_train.values
    test_df = X_test.copy()
    test_df["y"] = y_test.values

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Train set: {train_df.shape} -> {TRAIN_PATH}")
    print(f"Test set:  {test_df.shape} -> {TEST_PATH}")
    print(f"Train class balance:\n{y_train.value_counts(normalize=True)}")
    print(f"Test class balance:\n{y_test.value_counts(normalize=True)}")


if __name__ == "__main__":
    main()
