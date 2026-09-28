"""
preprocessing.py
-----------------
Part B: Data preprocessing for the STADIOEquities account-activation
proof-of-concept, using the public UCI "Bank Marketing" dataset as a
stand-in for the (not-yet-shared) STADIOEquities account data.

What this script does (see Preprocessing.MD for the full write-up):
  1. Loads the raw dataset (Data/raw/bank-additional-full.csv).
  2. Standardises column names and data types.
  3. Removes the 'duration' column (target leakage: call duration is
     only known *after* the outcome is decided, so a live scoring
     model could never have it — directly analogous to excluding any
     STADIOEquities feature that is only populated after activation).
  4. Leaves 'unknown' categorical values in place as their own
     category rather than imputing, since 'unknown' can itself be
     informative (e.g. a client who declines to state education).
  5. Writes a cleaned dataset to Data/processed/bank_marketing_clean.csv.

Usage:
    python preprocessing.py
"""
import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / "Data" / "raw" / "bank-additional-full.csv"
OUT_PATH = Path(__file__).resolve().parents[1] / "Data" / "processed" / "bank_marketing_clean.csv"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardise column names (dots -> underscores) for easier downstream use
    df.columns = [c.replace(".", "_") for c in df.columns]

    # Drop target-leakage column. 'duration' is the length of the last
    # call in seconds and is only known once the call (and therefore the
    # outcome) is over. Keeping it would make the modelling exercise
    # unrealistically easy and would not transfer to a live scoring
    # scenario. This mirrors excluding any STADIOEquities feature that is
    # only populated after the activation event itself.
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    # Normalise the target to a 0/1 integer for modelling
    df["y"] = (df["y"].str.strip().str.lower() == "yes").astype(int)

    # Strip whitespace from string/categorical columns
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].str.strip()

    # Sanity checks
    assert df.isnull().sum().sum() == 0, "Unexpected nulls after cleaning"
    assert set(df["y"].unique()) == {0, 1}, "Target should be binary 0/1"

    return df


def main():
    df_raw = load_raw()
    print(f"Loaded raw data: {df_raw.shape}")

    df_clean = clean(df_raw)
    print(f"Cleaned data: {df_clean.shape}")
    print(f"Class balance:\n{df_clean['y'].value_counts(normalize=True)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned dataset to {OUT_PATH}")


if __name__ == "__main__":
    main()
