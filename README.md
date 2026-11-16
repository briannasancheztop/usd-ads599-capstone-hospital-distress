# Predicting Hospital Financial Distress

ADS 599 Capstone, University of San Diego
Team: Brianna Sanchez, Kamran Shirazi, Ramon Gonzalez

This project predicts whether a U.S. hospital will be in financial distress next year using its financial and operating data from this year. It uses the CMS Hospital Provider Cost Report (2015–2022) and compares logistic regression, random forest, and XGBoost, with SHAP values to explain each hospital's risk.

## Data

**Source:** [CMS Hospital Provider Cost Report](https://data.cms.gov/provider-compliance/cost-report/hospital-provider-cost-report), built from the Healthcare Cost Report Information System (HCRIS). Every Medicare-certified hospital files one cost report per fiscal year.

**Unit of analysis:** one hospital per fiscal year (panel data).

**Analytic sample:** short-term acute care hospitals (STH) and critical access hospitals (CAH), excluding federal hospitals.

**Target:** a hospital is labeled as distressed if it has a negative total margin in the following year or stops filing cost reports (closure, merger, or conversion).

Data files are not stored in this repository because of their size. Run the pipeline below to download them.

## Setup

Requires Python 3.9 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

Run the scripts in this order from the project folder:

```bash
python src/pipeline.py      # 1. download, clean, and load the CMS data
python src/features.py      # 2. build ratios, the distress label, and lag features
python src/model_prep.py    # 3. split by year, preprocess, and run a baseline
```

To load the data in a notebook:

```python
import sqlite3, pandas as pd
conn = sqlite3.connect("data/hospital_cost_reports.db")
df = pd.read_sql("SELECT * FROM cost_report_clean", conn)   # or "features"
```

## Repository contents

| File | Description |
|---|---|
| `src/pipeline.py` | Downloads the yearly cost report CSVs from CMS, saves raw copies to `data/raw/`, standardizes column names and types, keeps one report per hospital per year (latest fiscal year end date), loads the `cost_report_raw` and `cost_report_clean` tables into SQLite, and writes data quality checks |
| `src/features.py` | Limits the data to STH and CAH hospitals (excluding federal), builds financial ratios (margins, liquidity, leverage, occupancy, payer mix), creates the next-year distress label, winsorizes ratios at the 1st and 99th percentiles of the training years, adds prior-year and year-over-year change features, and saves the `features` table |
| `src/model_prep.py` | Splits the data by year (train 2015–2019, validation 2020, test 2021), builds the preprocessing pipeline (median imputation with missing indicators, scaling, one-hot encoding) fit on training data only, and compares baseline models on the validation year |
| `notebooks/eda_provider_type.ipynb` | Counts hospitals and hospital-years by facility type and checks the effect of limiting the sample to STH and CAH hospitals |
| `notebooks/eda_missingness.ipynb` | Tests whether missing values are random (MCAR, MAR, or MNAR), including missingness regressions, prior-year margin comparisons, and panel attrition |
| `notebooks/eda_negative_liabilities.ipynb` | Investigates reports with negative total liabilities and checks whether their balance sheets still balance |
| `reports/quality_summary.txt` | Data quality summary from the pipeline (rows by year, duplicates removed, missing values) |
| `reports/rows_by_year.csv` | Raw rows, unique hospitals, duplicates, and clean rows by year |
| `reports/missing_by_column.csv` | Percent missing for every column in the clean table |
| `reports/hospital_years_by_type.png` | Chart of hospital-years by facility type |
| `reports/missingness_heatmap.png` | Chart of how missing values cluster across balance-sheet fields |
| `requirements.txt` | Python packages |

## Data outputs (created locally, not in the repo)

| Path | Description |
|---|---|
| `data/raw/` | Untouched yearly CSV files from CMS |
| `data/hospital_cost_reports.db` | SQLite database with `cost_report_raw`, `cost_report_clean`, and `features` tables |
| `data/quality/` | Quality checks and winsorizing bounds |