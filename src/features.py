"""
Feature engineering, step 1: load the clean table, check columns, build core ratios.

Run from the project folder after pipeline.py:
    python src/features.py
"""

import difflib
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_DIR / "data" / "hospital_cost_reports.db"

CCN = "provider_ccn"

# short name -> column name in cost_report_clean (after clean_column_name)
NEEDED = {
    "fy_begin": "fiscal_year_begin_date",
    "fy_end": "fiscal_year_end_date",
    "net_patient_revenue": "net_patient_revenue",
    "operating_income": "net_income_from_service_to_patients",
    "other_income": "total_other_income",
    "net_income": "net_income",
    "total_costs": "total_costs",
    "depreciation": "depreciation_cost",
    "cash": "cash_on_hand_and_in_banks",
    "accounts_receivable": "accounts_receivable",
    "current_assets": "total_current_assets",
    "current_liabilities": "total_current_liabilities",
    "total_assets": "total_assets",
    "total_liabilities": "total_liabilities",
    "fund_balance": "total_fund_balances",
    "beds": "number_of_beds",
    "total_days": "total_days_v_xviii_xix_unknown",
    "medicare_days": "total_days_title_xviii",
    "medicaid_days": "total_days_title_xix",
    "rural_urban": "rural_versus_urban",
    "control_type": "type_of_control",
}


KEEP_TYPES = ["STH", "CAH"]
TRAIN_YEARS = range(2015, 2020)   # 2020 = validation, 2021 = test
MIN_REVENUE = 500_000             # net patient revenue floor for revenue-based ratios

RATIO_COLS = [
    "operating_margin", "total_margin", "current_ratio", "days_cash_on_hand",
    "days_in_ar", "debt_ratio", "equity_ratio", "occupancy",
    "medicare_share", "medicaid_share",
]
LAG_COLS = ["total_margin", "operating_margin", "days_cash_on_hand",
            "current_ratio", "debt_ratio"]

# type_of_control codes grouped the same way as eda_missingness.ipynb
CONTROL_MAP = {1: "Nonprofit", 2: "Nonprofit",
               3: "For-profit", 4: "For-profit", 5: "For-profit", 6: "For-profit",
               7: "Federal",
               8: "Government", 9: "Government", 10: "Government", 11: "Government",
               12: "Government", 13: "Government"}


def load_clean():
    """Load the analytic sample: short-term acute + CAH, no federal hospitals."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM cost_report_clean", conn)
    df[CCN] = df[CCN].astype(str).str.zfill(6)

    df = df[df["ccn_facility_type"].isin(KEEP_TYPES)].copy()
    df["control_group"] = pd.to_numeric(df["type_of_control"], errors="coerce").map(CONTROL_MAP)
    # Federal hospitals are missing cash 98.5% of the time and do not face
    # the same financial pressure, so they are out of scope.
    df = df[df["control_group"] != "Federal"]
    return df.reset_index(drop=True)


def check_columns(df):
    """Stop early if a needed column is missing, and suggest the closest real names."""
    missing = [col for col in NEEDED.values() if col not in df.columns]
    if missing:
        print("Missing columns (closest matches in the table):")
        for col in missing:
            close = difflib.get_close_matches(col, df.columns, n=3, cutoff=0.5)
            print(f"  {col} -> {close}")
        raise SystemExit("Fix the names in NEEDED and rerun.")
    print(f"All {len(NEEDED)} needed columns found.")


def safe_divide(num, den):
    """Divide, returning NaN when the denominator is zero, negative, or missing."""
    den = den.where(den > 0)
    return num / den


def build_ratios(df):
    rename = {v: k for k, v in NEEDED.items()}
    keep = [CCN, "file_year", "ccn_facility_type", "control_group"] + list(NEEDED.values())
    df = df[keep].rename(columns=rename)

    for col in ["fy_begin", "fy_end"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # These can never be negative, so negative values are entry errors.
    # (Cash can go negative from overdrafts, so it is left alone.)
    for col in ["current_assets", "total_assets", "accounts_receivable", "beds"]:
        df[col] = df[col].where(df[col] >= 0)

    # Negative liabilities are real: 2,475 of 2,521 such rows still balance
    # (assets = liabilities + fund balance). Most likely money owed to the
    # hospital by its parent system. Keep the values and flag them.
    df["negative_liabilities"] = (df["total_liabilities"] < 0).astype(int)

    # Revenue near zero blows up the margins (one row has revenue of $1).
    # Treat anything under the floor as unusable for revenue-based ratios.
    npr = df["net_patient_revenue"].where(df["net_patient_revenue"] >= MIN_REVENUE)

    # Some reports cover less (or more) than 12 months. Use the real period
    # length for the "days" ratios and flag the odd ones.
    df["period_days"] = (df["fy_end"] - df["fy_begin"]).dt.days + 1
    df["partial_year"] = ~df["period_days"].between(360, 370)

    cash_expense = df["total_costs"] - df["depreciation"].fillna(0)

    # Profitability
    df["operating_margin"] = safe_divide(df["operating_income"], npr)
    df["total_margin"] = safe_divide(df["net_income"], npr + df["other_income"].fillna(0))

    # Liquidity
    df["current_ratio"] = safe_divide(df["current_assets"], df["current_liabilities"])
    df["days_cash_on_hand"] = safe_divide(df["cash"], cash_expense / df["period_days"])
    df["days_in_ar"] = safe_divide(df["accounts_receivable"], npr / df["period_days"])

    # Leverage
    df["debt_ratio"] = safe_divide(df["total_liabilities"], df["total_assets"])
    df["equity_ratio"] = safe_divide(df["fund_balance"], df["total_assets"])

    # Size, utilization, payer mix
    df["log_beds"] = np.log1p(df["beds"])
    df["log_revenue"] = np.log1p(df["net_patient_revenue"].clip(lower=0))
    df["occupancy"] = safe_divide(df["total_days"], df["beds"] * df["period_days"]).clip(upper=1)
    df["medicare_share"] = safe_divide(df["medicare_days"], df["total_days"])
    df["medicaid_share"] = safe_divide(df["medicaid_days"], df["total_days"])

    # Missing flags. The whole balance sheet tends to go missing at once.
    df["balance_sheet_missing"] = df["total_assets"].isna().astype(int)
    df["medicaid_days_missing"] = df["medicaid_days"].isna().astype(int)

    return df


# ---------------------------------------------------------------------------
# Step 2: target, winsorizing, lag features
# ---------------------------------------------------------------------------
def add_target(df):
    """distress_next = 1 if total margin < 0 next year, or the hospital exits.

    Exit = no later report at all. A gap year (missing t+1 but back later)
    gets no label, since that is a missing report, not an exit.
    The last year has no t+1, so it gets no label either.
    """
    g = df.groupby(CCN)
    next_year = g["file_year"].shift(-1)
    has_next = next_year.eq(df["file_year"] + 1)
    next_margin = g["total_margin"].shift(-1).where(has_next)

    last_year = df["file_year"].max()
    df["exited_next"] = (next_year.isna() & (df["file_year"] < last_year)).astype(int)

    df["distress_next"] = np.where(
        df["exited_next"] == 1, 1.0,
        np.where(next_margin.notna(), (next_margin < 0).astype(float), np.nan),
    )
    return df


def winsorize(df, cols, train_mask, lower=0.01, upper=0.99):
    """Cap each column at percentiles computed on the training years only."""
    bounds = {}
    for col in cols:
        lo, hi = df.loc[train_mask, col].quantile([lower, upper])
        df[col] = df[col].clip(lo, hi)
        bounds[col] = {"lower": lo, "upper": hi}
    return df, pd.DataFrame(bounds).T


def add_lag_features(df):
    """Last year's value and the 1-year change, only when last year is t-1."""
    g = df.groupby(CCN)
    consecutive = g["file_year"].diff().eq(1)

    for col in LAG_COLS:
        prev = g[col].shift(1).where(consecutive)
        df[f"{col}_lag1"] = prev
        df[f"{col}_chg"] = df[col] - prev

    prev_rev = g["net_patient_revenue"].shift(1).where(consecutive)
    df["revenue_growth"] = safe_divide(df["net_patient_revenue"] - prev_rev, prev_rev)
    df["neg_margin_last_year"] = (df["total_margin_lag1"] < 0).astype(float).where(
        df["total_margin_lag1"].notna()
    )
    return df


if __name__ == "__main__":
    df = load_clean()
    print(f"Loaded {len(df):,} hospital-years")
    check_columns(df)
    feats = build_ratios(df)
    feats = feats.sort_values([CCN, "file_year"]).reset_index(drop=True)
    print(f"Partial-year reports: {feats['partial_year'].sum():,}")

    # Target first, from unwinsorized margins (the 1% cap is negative, so
    # winsorizing would not flip any signs anyway).
    feats = add_target(feats)

    train_mask = feats["file_year"].isin(TRAIN_YEARS)
    feats, bounds = winsorize(feats, RATIO_COLS, train_mask)
    feats = add_lag_features(feats)
    feats, growth_bounds = winsorize(feats, ["revenue_growth"], train_mask)
    bounds = pd.concat([bounds, growth_bounds])

    out_dir = PROJECT_DIR / "data" / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds.to_csv(out_dir / "winsor_bounds.csv")

    with sqlite3.connect(DB_PATH) as conn:
        feats.to_sql("features", conn, if_exists="replace", index=False, chunksize=5000)

    labeled = feats["distress_next"].notna()
    print(f"\nLabeled hospital-years: {labeled.sum():,} of {len(feats):,}")
    print(f"Exits counted as distress: {feats['exited_next'].sum():,}")
    print("\nDistress rate by year (share of labeled rows):")
    print(feats[labeled].groupby("file_year")["distress_next"]
          .agg(["size", "mean"]).round(3).to_string())
    print("\nWinsorized ratio summary:")
    print(feats[RATIO_COLS + ["revenue_growth"]]
          .describe(percentiles=[0.01, 0.5, 0.99]).T.round(3).to_string())
    print(f"\nSaved table 'features' to {DB_PATH}")