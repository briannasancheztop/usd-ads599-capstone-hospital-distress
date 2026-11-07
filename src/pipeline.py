"""
Data pipeline: Predicting Hospital Financial Distress.

Steps
1. Download yearly CMS Hospital Provider Cost Report CSV files.
2. Save the raw files untouched in data/raw/ so every result traces back to the source.
3. Standardize column names, convert numeric columns, and tag each row with its file year.
4. Keep one report per hospital (CCN) per year, using the report with the latest
   fiscal year end date when a hospital filed more than one.
5. Load raw and cleaned tables into a SQLite database (data/hospital_cost_reports.db).
6. Write data quality checks (row counts, duplicates, missing values) to data/quality/.

Source: Centers for Medicare & Medicaid Services, Hospital Provider Cost Report
https://data.cms.gov/provider-compliance/cost-report/hospital-provider-cost-report

Usage (from the project folder):
    python src/pipeline.py                      # default years 2015-2022
    python src/pipeline.py --start 2011 --end 2022
    python src/pipeline.py --skip-download      # use CSVs already in data/raw/
"""

import argparse
import re
import sqlite3
from pathlib import Path

import pandas as pd
import requests

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
QUALITY_DIR = DATA_DIR / "quality"
DB_PATH = DATA_DIR / "hospital_cost_reports.db"

CATALOG_URL = "https://data.cms.gov/data.json"
DATASET_TITLE = "Hospital Provider Cost Report"

# Column names 
CCN_COL = "provider_ccn"
FY_END_COL = "fiscal_year_end_date"
FY_BEGIN_COL = "fiscal_year_begin_date"


# ---------------------------------------------------------------------------
# Step 1-2: download raw files
# ---------------------------------------------------------------------------
def year_from_text(text):
    """Pull a 4-digit year (2000-2099) out of a URL, title, or date range."""
    match = re.search(r"(20\d{2})", text or "")
    return int(match.group(1)) if match else None


def find_download_links():
    """Read the CMS data catalog and return {year: csv_url} for the cost report."""
    catalog = requests.get(CATALOG_URL, timeout=60).json()
    datasets = catalog.get("dataset", [])

    # Match any catalog entry whose title mentions the hospital provider cost report
    # (CMS sometimes lists it as one entry, sometimes one entry per year).
    target = DATASET_TITLE.lower()
    matches = [d for d in datasets if target in d.get("title", "").lower()]
    if not matches:
        similar = [d.get("title") for d in datasets if "cost report" in d.get("title", "").lower()]
        print("Could not find the Hospital Provider Cost Report in the CMS catalog.")
        print("Catalog titles that mention 'cost report':", similar[:20])
        raise SystemExit("Download the CSVs by hand into data/raw/ and rerun with --skip-download.")

    links = {}
    for dataset in matches:
        for dist in dataset.get("distribution", []):
            url = dist.get("downloadURL") or dist.get("accessURL") or ""
            name = Path(url.split("?")[0]).name
            is_csv = dist.get("mediaType") == "text/csv" or name.lower().endswith(".csv")
            if not url or not is_csv:
                continue
            # Prefer the year in the file name (e.g., CostReport_2021_Final.csv)
            year = (year_from_text(name) if "costreport" in name.lower() else None) \
                or year_from_text(dist.get("temporal")) or year_from_text(dist.get("title"))
            if year:
                links.setdefault(year, url)

    if not links:
        print("Found the dataset but no CSV download links. Entries found:",
              [d.get("title") for d in matches])
        raise SystemExit("Download the CSVs by hand into data/raw/ and rerun with --skip-download.")
    return links


def download_raw_files(start, end):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    links = find_download_links()
    print(f"Found {len(links)} yearly files in the CMS catalog: {sorted(links)}")
    for year in range(start, end + 1):
        if year not in links:
            print(f"  {year}: not listed in the catalog, skipping")
            continue
        out_path = RAW_DIR / f"CostReport_{year}_Final.csv"
        if out_path.exists():
            print(f"  {year}: already downloaded")
            continue
        print(f"  {year}: downloading...")
        with requests.get(links[year], stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)


# ---------------------------------------------------------------------------
# Step 3: standardize
# ---------------------------------------------------------------------------
def clean_column_name(name):
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def load_raw_files(start, end):
    frames = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        year = year_from_text(path.name)
        if year is None or not (start <= year <= end):
            continue
        # Read everything as text first so CCNs keep their leading zeros
        df = pd.read_csv(path, dtype=str, low_memory=False)
        df.columns = [clean_column_name(c) for c in df.columns]
        df.insert(0, "file_year", year)
        frames.append(df)
        print(f"  Loaded {path.name}: {len(df):,} rows, {df.shape[1] - 1} columns")
    if not frames:
        raise SystemExit("No CSV files found in data/raw/. Run without --skip-download first.")
    return pd.concat(frames, ignore_index=True, sort=False)


def convert_types(df):
    """Convert date columns to dates and mostly-numeric columns to numbers."""
    df = df.copy()
    for col in df.columns:
        if col in ("file_year", CCN_COL):
            continue
        if col.endswith("_date"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
            continue
        values = df[col].dropna()
        if values.empty:
            continue
        numeric = pd.to_numeric(values.str.replace(",", "", regex=False), errors="coerce")
        # Treat as numeric if at least 95% of non-missing values parse as numbers
        if numeric.notna().mean() >= 0.95:
            df[col] = pd.to_numeric(df[col].str.replace(",", "", regex=False), errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Step 4: one report per hospital per year
# ---------------------------------------------------------------------------
def deduplicate(df):
    if CCN_COL not in df.columns:
        print(f"  WARNING: '{CCN_COL}' column not found; skipping de-duplication.")
        return df, 0
    sort_cols = ["file_year", CCN_COL] + ([FY_END_COL] if FY_END_COL in df.columns else [])
    df = df.sort_values(sort_cols)
    before = len(df)
    df = df.drop_duplicates(subset=["file_year", CCN_COL], keep="last")
    return df.reset_index(drop=True), before - len(df)


# ---------------------------------------------------------------------------
# Step 6: data quality checks
# ---------------------------------------------------------------------------
def quality_checks(raw, clean, removed):
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    by_year = raw.groupby("file_year").size().rename("raw_rows").to_frame()
    if CCN_COL in raw.columns:
        by_year["unique_hospitals"] = raw.groupby("file_year")[CCN_COL].nunique()
        by_year["duplicate_reports"] = by_year["raw_rows"] - by_year["unique_hospitals"]
    by_year["clean_rows"] = clean.groupby("file_year").size()
    by_year.to_csv(QUALITY_DIR / "rows_by_year.csv")

    missing = (clean.isna().mean() * 100).round(1).sort_values(ascending=False)
    missing.rename("pct_missing").to_csv(QUALITY_DIR / "missing_by_column.csv", header=True)

    lines = [
        "DATA QUALITY SUMMARY",
        "====================",
        f"Years loaded: {clean['file_year'].min()}-{clean['file_year'].max()}",
        f"Raw rows: {len(raw):,}",
        f"Duplicate reports removed (same hospital, same year): {removed:,}",
        f"Clean rows (hospital-years): {len(clean):,}",
        f"Unique hospitals (CCNs): {clean[CCN_COL].nunique():,}" if CCN_COL in clean.columns else "",
        f"Columns: {clean.shape[1]}",
        "",
        "Rows by year:",
        by_year.to_string(),
        "",
        "Columns with the most missing values (%):",
        missing.head(20).to_string(),
        "",
        f"Columns more than 50% missing: {(missing > 50).sum()} of {len(missing)}",
    ]
    report = "\n".join(line for line in lines if line is not None)
    (QUALITY_DIR / "quality_summary.txt").write_text(report)
    print("\n" + report)


# ---------------------------------------------------------------------------
# Step 5: load to SQLite
# ---------------------------------------------------------------------------
def load_to_sqlite(raw, clean):
    with sqlite3.connect(DB_PATH) as conn:
        raw.to_sql("cost_report_raw", conn, if_exists="replace", index=False, chunksize=5000)
        clean.to_sql("cost_report_clean", conn, if_exists="replace", index=False, chunksize=5000)
        if CCN_COL in clean.columns:
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_clean_ccn_year ON cost_report_clean ({CCN_COL}, file_year)")
    print(f"\nSaved tables cost_report_raw and cost_report_clean to {DB_PATH}")


def main():
    parser = argparse.ArgumentParser(description="CMS Hospital Provider Cost Report pipeline")
    parser.add_argument("--start", type=int, default=2015, help="first year to include")
    parser.add_argument("--end", type=int, default=2022, help="last year to include")
    parser.add_argument("--skip-download", action="store_true", help="use CSVs already in data/raw/")
    args = parser.parse_args()

    if not args.skip_download:
        print("Step 1: downloading raw files")
        download_raw_files(args.start, args.end)

    print("\nStep 2: loading and standardizing")
    raw = load_raw_files(args.start, args.end)
    raw = convert_types(raw)

    print("\nStep 3: keeping one report per hospital per year")
    clean, removed = deduplicate(raw)

    print("\nStep 4: loading to SQLite")
    load_to_sqlite(raw, clean)

    print("\nStep 5: data quality checks")
    quality_checks(raw, clean, removed)


if __name__ == "__main__":
    main()