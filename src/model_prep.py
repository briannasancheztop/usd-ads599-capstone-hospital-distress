"""
Feature engineering, step 3: modeling prep.

- Loads the 'features' table built by features.py
- Splits by year: train 2015-2019, validation 2020, test 2021
- Builds a preprocessing pipeline (impute, flag missing, scale, one-hot)
- Runs a quick baseline on validation as a sanity check

Teammates can import get_splits() and build_preprocessor() for their models.
The test year is never touched here.

Run from the project folder after features.py:
    python src/model_prep.py
"""

import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_DIR / "data" / "hospital_cost_reports.db"

TARGET = "distress_next"
TRAIN_YEARS = range(2015, 2020)
VAL_YEAR = 2020
TEST_YEAR = 2021

LAG_COLS = ["total_margin", "operating_margin", "days_cash_on_hand",
            "current_ratio", "debt_ratio"]

NUMERIC = [
    "operating_margin", "total_margin", "current_ratio", "days_cash_on_hand",
    "days_in_ar", "debt_ratio", "equity_ratio", "occupancy",
    "medicare_share", "medicaid_share", "log_beds", "log_revenue",
    "revenue_growth",
] + [f"{c}_lag1" for c in LAG_COLS] + [f"{c}_chg" for c in LAG_COLS]

FLAGS = ["partial_year", "balance_sheet_missing", "medicaid_days_missing",
         "negative_liabilities", "neg_margin_last_year"]

CATEGORICAL = ["ccn_facility_type", "control_group", "rural_urban"]

FEATURES = NUMERIC + FLAGS + CATEGORICAL


def get_splits():
    """Return (X_train, y_train, X_val, y_val, X_test, y_test), labeled rows only."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM features", conn)

    df = df[df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].astype(int)
    df[FLAGS] = df[FLAGS].astype(float)

    train = df[df["file_year"].isin(TRAIN_YEARS)]
    val = df[df["file_year"] == VAL_YEAR]
    test = df[df["file_year"] == TEST_YEAR]

    return (train[FEATURES], train[TARGET],
            val[FEATURES], val[TARGET],
            test[FEATURES], test[TARGET])


def build_preprocessor():
    """Medians and categories are learned from whatever data .fit() sees (train only)."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    flag_pipe = SimpleImputer(strategy="most_frequent")
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("flag", flag_pipe, FLAGS),
        ("cat", cat_pipe, CATEGORICAL),
    ])


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = get_splits()

    print("Split sizes and distress rate:")
    for name, y in [("train 2015-2019", y_train), (f"val {VAL_YEAR}", y_val),
                    (f"test {TEST_YEAR}", y_test)]:
        print(f"  {name:16s} {len(y):>6,} rows   {y.mean():.3f}")

    print("\nMissing % in training features (top 10):")
    print((X_train.isna().mean() * 100).round(1).sort_values(ascending=False)
          .head(10).to_string())

    # Baseline 1: last year's margin alone. Lower margin = higher risk.
    margin = X_val["total_margin"].fillna(X_train["total_margin"].median())
    auc_naive = roc_auc_score(y_val, -margin)

    # Baseline 2: logistic regression on all features.
    model = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    model.fit(X_train, y_train)
    auc_logit = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    print(f"\nValidation AUC ({VAL_YEAR}):")
    print(f"  Current total margin only:  {auc_naive:.3f}")
    print(f"  Logistic regression:        {auc_logit:.3f}")
    print("\nTest year left untouched.")