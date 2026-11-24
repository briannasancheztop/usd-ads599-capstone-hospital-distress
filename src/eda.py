"""
EDA for the hospital financial distress project.
Run after pipeline.py:  python src/eda.py
Saves figures to reports/figures/ and the numbers to reports/eda_summary.txt.
"""

import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (  # noqa: E402
    CCN, DB_PATH, PROJECT_DIR, RATIO_COLS, TRAIN_YEARS,
    add_lag_features, add_target, build_ratios, load_clean, winsorize,
)

FIG_DIR = PROJECT_DIR / "reports" / "figures"
SUMMARY_PATH = PROJECT_DIR / "reports" / "eda_summary.txt"
STATS_PATH = PROJECT_DIR / "reports" / "eda_stats_by_class.csv"

FULL_W, COL_W = 6.5, 3.25          # inches: full page width, one column
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HEALTHY, DISTRESSED = "#2a78d6", "#eb6834"   # categorical slots 1 and 2
STH_C, CAH_C = "#2a78d6", "#1baf7a"          # categorical slots 1 and 3
RAW_C = "#b9b8b2"                            # neutral gray for "before"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

LABELS = {
    "operating_margin": "Operating margin",
    "total_margin": "Total margin",
    "current_ratio": "Current ratio",
    "days_cash_on_hand": "Days cash on hand",
    "days_in_ar": "Days in accounts receivable",
    "debt_ratio": "Debt ratio",
    "equity_ratio": "Equity ratio",
    "occupancy": "Occupancy rate",
    "medicare_share": "Medicare share",
    "medicaid_share": "Medicaid share",
    "revenue_growth": "Revenue growth",
    "log_beds": "Beds (log)",
    "log_revenue": "Net patient revenue (log)",
    "total_margin_chg": "Change in total margin",
}
PCT_COLS = {"operating_margin", "total_margin", "occupancy", "medicare_share",
            "medicaid_share", "revenue_growth", "total_margin_chg"}

lines = []  # summary text


def say(text=""):
    print(text)
    lines.append(str(text))


def section(title):
    say()
    say(title)
    say("=" * len(title))


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    say(f"  saved {path.relative_to(PROJECT_DIR)}")


def pct_axis(ax, axis="x"):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def load_data():
    clean = load_clean()
    extra = clean[[CCN, "file_year", "state_code"]].copy()

    raw = build_ratios(clean)
    raw = raw.merge(extra, on=[CCN, "file_year"], how="left")
    raw = raw.sort_values([CCN, "file_year"]).reset_index(drop=True)
    raw = add_target(raw)

    train_mask = raw["file_year"].isin(TRAIN_YEARS)
    capped, bounds = winsorize(raw.copy(), RATIO_COLS, train_mask)
    capped = add_lag_features(capped)

    capped["outcome_year"] = capped["file_year"] + 1
    capped["facility"] = capped["ccn_facility_type"].map(
        {"STH": "Short-term acute", "CAH": "Critical access"})
    capped["location"] = capped["rural_urban"].map({"R": "Rural", "U": "Urban"})
    return raw, capped, bounds, train_mask


def sample_and_target(df):
    section("1. SAMPLE AND TARGET")
    lab = df[df["distress_next"].notna()]
    say(f"Hospital-years in analytic sample: {len(df):,}")
    say(f"Unique hospitals: {df[CCN].nunique():,}")
    say(f"Labeled hospital-years (feature years {lab['file_year'].min()}-{lab['file_year'].max()}): {len(lab):,}")
    say(f"Unlabeled: {len(df) - len(lab):,} (last year, gap years, or next-year margin missing)")
    say(f"Distressed share: {pct(lab['distress_next'].mean())} "
        f"({int(lab['distress_next'].sum()):,} distressed, {int((1 - lab['distress_next']).sum()):,} healthy)")
    say(f"Exits counted as distress: {int(lab['exited_next'].sum()):,}")
    for name, mask in [("Train (2015-2019)", lab["file_year"].isin(TRAIN_YEARS)),
                       ("Validation (2020)", lab["file_year"] == 2020),
                       ("Test (2021)", lab["file_year"] == 2021)]:
        part = lab[mask]
        say(f"  {name}: {len(part):,} rows, {pct(part['distress_next'].mean())} distressed")
    return lab


def fig_winsorizing(raw, capped, bounds, train_mask):
    section("2. WINSORIZING (1st/99th percentile of training years)")
    cols = ["total_margin", "operating_margin", "days_cash_on_hand",
            "current_ratio", "debt_ratio", "days_in_ar"]
    say(f"{'Variable':<28}{'Lower cap':>12}{'Upper cap':>12}{'Raw min':>14}{'Raw max':>14}"
        f"{'Rows capped':>13}{'Skew before':>13}{'Skew after':>12}")
    for col in RATIO_COLS:
        r = raw[col].dropna()
        c = capped[col].dropna()
        lo, hi = bounds.loc[col, "lower"], bounds.loc[col, "upper"]
        n_cap = int(((r < lo) | (r > hi)).sum())
        say(f"{LABELS[col]:<28}{lo:>12.3f}{hi:>12.3f}{r.min():>14.2f}{r.max():>14.2f}"
            f"{pct(n_cap / len(r)):>13}{stats.skew(r):>13.1f}{stats.skew(c):>12.1f}")

    fig, axes = plt.subplots(3, 2, figsize=(FULL_W, 6.6))
    for ax, col in zip(axes.flat, cols):
        r = raw[col].dropna()
        c = capped[col].dropna()
        lo, hi = bounds.loc[col, "lower"], bounds.loc[col, "upper"]
        # Show the raw distribution over a window a bit wider than the caps,
        # so the tails that get cut are visible.
        span = hi - lo
        view_lo, view_hi = lo - 0.35 * span, hi + 0.35 * span
        bins = np.linspace(view_lo, view_hi, 60)
        ax.hist(r.clip(view_lo, view_hi), bins=bins, color=RAW_C, label="Before (raw)")
        ax.hist(c, bins=bins, color=HEALTHY, alpha=0.85, label="After (capped)")
        for v in (lo, hi):
            ax.axvline(v, color=INK, lw=0.8, ls="--")
        n_out = int(((r < view_lo) | (r > view_hi)).sum())
        ax.set_title(LABELS[col], loc="left")
        ax.set_xlabel(f"Raw range {r.min():,.2f} to {r.max():,.2f}\n"
                      f"{n_out:,} raw values beyond plot edges", fontsize=7.5, color=MUTED)
        ax.set_ylabel("Hospital-years")
        if col in PCT_COLS:
            pct_axis(ax)
    handles = [matplotlib.patches.Patch(color=RAW_C, label="Before (raw values)"),
               matplotlib.patches.Patch(color=HEALTHY, label="After (capped values)"),
               matplotlib.lines.Line2D([], [], color=INK, lw=0.8, ls="--",
                                       label="Caps: 1st and 99th percentiles of training years")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(h_pad=1.8)
    save(fig, "eda_winsorizing_before_after")


def fig_log_transform(raw):
    section("3. LOG TRANSFORM (beds and net patient revenue)")
    pairs = [("beds", "log_beds", "Number of beds"),
             ("net_patient_revenue", "log_revenue", "Net patient revenue ($)")]
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 4.4))
    for row, (rawcol, logcol, label) in enumerate(pairs):
        r = raw[rawcol].dropna()
        r = r[r > 0]
        lg = raw[logcol].replace([np.inf, -np.inf], np.nan).dropna()
        # trim the extreme 0.1% on each side so a few data-entry errors
        # (e.g., a hospital reporting over a million beds) don't stretch the axis
        lg = lg[(lg > lg.quantile(0.001)) & (lg < lg.quantile(0.999))]
        say(f"{label}: skew before {stats.skew(r):.1f}, after log {stats.skew(lg):.1f}; "
            f"median {r.median():,.0f}, 99th percentile {r.quantile(0.99):,.0f}, max {r.max():,.0f}")
        axes[row, 0].hist(r.clip(upper=r.quantile(0.995)), bins=60, color=RAW_C)
        axes[row, 0].set_title(f"{label}: before", loc="left")
        axes[row, 1].hist(lg, bins=60, color=HEALTHY)
        axes[row, 1].set_title(f"{label}: after log transform", loc="left")
        axes[row, 0].set_ylabel("Hospital-years")
        if rawcol == "net_patient_revenue":
            axes[row, 0].xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1e9:,.1f}B"))
            axes[row, 0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    axes[0, 1].set_xlabel("log(1 + beds)")
    axes[1, 1].set_xlabel("log(1 + revenue)")
    fig.tight_layout(h_pad=1.6)
    save(fig, "eda_log_transform_before_after")


def stats_by_class(lab):
    section("4. FEATURES BY DISTRESS CLASS (labeled rows, capped values)")
    cols = RATIO_COLS + ["revenue_growth", "log_beds", "total_margin_chg"]
    rows = []
    for col in cols:
        h = lab.loc[lab["distress_next"] == 0, col].dropna()
        d = lab.loc[lab["distress_next"] == 1, col].dropna()
        if len(h) == 0 or len(d) == 0:
            continue
        _, p = stats.mannwhitneyu(h, d)
        # rank-biserial effect size: share of pairs where distressed > healthy, rescaled
        u = stats.mannwhitneyu(d, h).statistic
        effect = 2 * u / (len(d) * len(h)) - 1
        rows.append({
            "variable": LABELS[col],
            "healthy_median": h.median(), "healthy_q1": h.quantile(.25), "healthy_q3": h.quantile(.75),
            "distressed_median": d.median(), "distressed_q1": d.quantile(.25), "distressed_q3": d.quantile(.75),
            "mann_whitney_p": p, "rank_biserial": effect,
        })
    out = pd.DataFrame(rows)
    out.to_csv(STATS_PATH, index=False)
    say(f"{'Variable':<28}{'Healthy median [IQR]':>30}{'Distressed median [IQR]':>32}{'p':>10}{'effect':>8}")
    for _, r in out.iterrows():
        say(f"{r['variable']:<28}"
            f"{r['healthy_median']:>10.3f} [{r['healthy_q1']:.3f}, {r['healthy_q3']:.3f}]".ljust(58)
            + f"{r['distressed_median']:>10.3f} [{r['distressed_q1']:.3f}, {r['distressed_q3']:.3f}]".ljust(32)
            + f"{r['mann_whitney_p']:>10.1e}{r['rank_biserial']:>8.2f}")
    say("effect = rank-biserial correlation (distressed vs. healthy); |0.1| small, |0.3| medium, |0.5| large")
    say(f"saved {STATS_PATH.relative_to(PROJECT_DIR)}")
    return out


def fig_by_class(lab):
    cols = ["total_margin", "operating_margin", "days_cash_on_hand", "current_ratio",
            "debt_ratio", "occupancy", "medicare_share", "medicaid_share"]
    fig, axes = plt.subplots(2, 4, figsize=(FULL_W, 4.6))
    for ax, col in zip(axes.flat, cols):
        data = [lab.loc[lab["distress_next"] == k, col].dropna() for k in (0, 1)]
        bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                        medianprops={"color": INK, "lw": 1.2},
                        whiskerprops={"color": MUTED, "lw": 0.8},
                        capprops={"color": MUTED, "lw": 0.8},
                        boxprops={"lw": 0})
        for patch, color in zip(bp["boxes"], (HEALTHY, DISTRESSED)):
            patch.set_facecolor(color)
        ax.set_xticks([])
        ax.set_title(LABELS[col], loc="left", fontsize=8.5)
        if col in PCT_COLS:
            pct_axis(ax, "y")
    handles = [matplotlib.patches.Patch(color=HEALTHY, label="Healthy next year"),
               matplotlib.patches.Patch(color=DISTRESSED, label="Distressed next year")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.0, -0.01, "Boxes show the middle 50% of hospital-years; whiskers extend to 1.5 × IQR; "
             "outliers not drawn.", fontsize=7.5, color=MUTED)
    fig.tight_layout(w_pad=1.4, h_pad=1.6)
    save(fig, "eda_features_by_distress_class")


def group_rates(lab):
    section("5. DISTRESS RATE BY GROUP (labeled rows)")
    groups = [("facility", "Facility type"), ("control_group", "Ownership"),
              ("location", "Location")]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.4), sharex=True)
    overall = lab["distress_next"].mean()
    for ax, (col, title) in zip(axes, groups):
        g = lab.groupby(col)["distress_next"].agg(["mean", "size"]).sort_values("mean")
        table = pd.crosstab(lab[col], lab["distress_next"])
        chi2, p, _, _ = stats.chi2_contingency(table)
        say(f"{title} (chi-square = {chi2:,.1f}, p = {p:.1e}):")
        for name, r in g.iterrows():
            say(f"  {name:<20} {pct(r['mean'])} of {int(r['size']):,} hospital-years")
        ax.barh(g.index, g["mean"], color=DISTRESSED, height=0.6)
        for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
            ax.text(max(m, overall) + 0.03, i, f"{pct(m, 0)}\n(n={n:,})", va="center",
                    fontsize=7, color=INK, linespacing=1.1)
        ax.axvline(overall, color=INK, lw=0.8, ls="--")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        pct_axis(ax)
        ax.set_xlim(0, max(g["mean"].max(), overall) + 0.38)
    axes[1].set_xlabel("Share distressed next year (dashed line = all hospitals)")
    fig.tight_layout(w_pad=0.6)
    save(fig, "eda_distress_rate_by_group")


def fig_trend_by_type(lab):
    section("6. DISTRESS RATE BY OUTCOME YEAR AND FACILITY TYPE")
    t = lab.groupby(["outcome_year", "facility"])["distress_next"].mean().unstack()
    t["All hospitals"] = lab.groupby("outcome_year")["distress_next"].mean()
    say(t.apply(lambda s: s.map(pct)).to_string())

    fig, ax = plt.subplots(figsize=(FULL_W, 2.8))
    years = t.index.values
    for lo, hi, name in [(years.min() - 0.5, 2020.5, "Training"),
                         (2020.5, 2021.5, "Validation"), (2021.5, 2022.5, "Test")]:
        ax.axvspan(lo, hi, color="#f0efec" if name != "Validation" else "#e7f1fb", lw=0)
        ax.text((lo + hi) / 2, 0.02, name, ha="center", fontsize=7.5, color=MUTED,
                transform=ax.get_xaxis_transform())
    for col, color, ls in [("Short-term acute", STH_C, "-"), ("Critical access", CAH_C, "-"),
                           ("All hospitals", INK, "--")]:
        if col in t:
            ax.plot(years, t[col], color=color, lw=2, ls=ls, marker="o", ms=4, label=col)
    ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
    ax.set_xticks(years)
    ax.set_xlabel("Outcome year (year the distress label refers to)")
    ax.set_ylabel("Share distressed")
    pct_axis(ax, "y")
    ax.set_ylim(0, max(0.6, t.max().max() + 0.08))
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    save(fig, "eda_distress_trend_by_facility_type")


def fig_ratios_over_time(df):
    section("7. MEDIAN RATIOS BY YEAR AND FACILITY TYPE (capped values)")
    cols = ["total_margin", "operating_margin", "days_cash_on_hand"]
    med = df.groupby(["file_year", "facility"])[cols].median()
    say(med.round(3).to_string())
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.4))
    for ax, col in zip(axes, cols):
        m = med[col].unstack()
        for fac, color in [("Short-term acute", STH_C), ("Critical access", CAH_C)]:
            if fac in m:
                ax.plot(m.index, m[fac], color=color, lw=2, marker="o", ms=3.5, label=fac)
        ax.axvspan(2019.5, 2021.5, color="#f0efec", lw=0)
        ax.set_title(LABELS[col], loc="left")
        ax.set_xticks(m.index[::2])
        if col in PCT_COLS:
            pct_axis(ax, "y")
            ax.axhline(0, color=MUTED, lw=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.0, -0.03, "Medians by reporting year. Shaded band = 2020-2021 (federal pandemic relief).",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(w_pad=1.2)
    save(fig, "eda_median_ratios_over_time")


def fig_persistence(lab):
    section("8. PERSISTENCE: THIS YEAR'S MARGIN VS. NEXT YEAR'S DISTRESS")
    neg = lab["total_margin"] < 0
    say(f"Distress next year if total margin < 0 this year: {pct(lab.loc[neg, 'distress_next'].mean())} "
        f"(n={int(neg.sum()):,})")
    say(f"Distress next year if total margin >= 0 this year: {pct(lab.loc[~neg & lab['total_margin'].notna(), 'distress_next'].mean())}")
    naive = (lab["total_margin"] < 0).astype(float).where(lab["total_margin"].notna())
    ok = naive.notna()
    tp = ((naive == 1) & (lab["distress_next"] == 1) & ok).sum()
    say(f"Naive rule 'distressed next year if margin < 0 now': recall {pct(tp / lab.loc[ok, 'distress_next'].sum())}, "
        f"precision {pct(tp / (naive == 1).sum())}")

    dec = pd.qcut(lab["total_margin"], 10, labels=False, duplicates="drop")
    by_dec = lab.groupby(dec).agg(rate=("distress_next", "mean"),
                                  lo=("total_margin", "min"), hi=("total_margin", "max"))
    say("Distress rate by decile of this year's total margin:")
    for d, r in by_dec.iterrows():
        say(f"  decile {int(d) + 1:>2} ({pct(r['lo'])} to {pct(r['hi'])}): {pct(r['rate'])}")

    chg = lab["total_margin_chg"]
    buckets = pd.cut(chg, [-np.inf, -0.05, -0.01, 0.01, 0.05, np.inf],
                     labels=["Fell\n>5 pts", "Fell\n1-5 pts", "Stable\n(±1 pt)",
                             "Rose\n1-5 pts", "Rose\n>5 pts"])
    # Compare within the same current margin level to show the trend adds information
    level = np.where(lab["total_margin"] < 0, "Negative margin now", "Positive margin now")
    by_chg = lab.groupby([level, buckets], observed=True)["distress_next"].agg(["mean", "size"])
    say("Distress rate by 1-year change in total margin, within current margin level:")
    say(by_chg.assign(mean=by_chg["mean"].map(pct)).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.7), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    ax.bar(by_dec.index + 1, by_dec["rate"], color=DISTRESSED, width=0.7)
    ax.set_xticks(by_dec.index + 1)
    ax.set_xlabel("Decile of this year's total margin (1 = lowest)")
    ax.set_ylabel("Share distressed next year")
    ax.set_title("By current margin", loc="left")
    pct_axis(ax, "y")

    ax = axes[1]
    order = list(buckets.cat.categories)
    x = np.arange(len(order))
    for i, (lvl, color) in enumerate([("Positive margin now", HEALTHY), ("Negative margin now", DISTRESSED)]):
        vals = [by_chg["mean"].get((lvl, b), np.nan) for b in order]
        ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=color, label=lvl)
    ax.set_xticks(x, order, fontsize=7.5)
    ax.set_title("By 1-year change in margin", loc="left")
    pct_axis(ax, "y")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout(w_pad=1.5)
    save(fig, "eda_persistence_margin_level_and_trend")


def fig_correlations(lab, train_mask_lab):
    section("9. SPEARMAN CORRELATIONS (training years)")
    cols = ["total_margin", "operating_margin", "days_cash_on_hand", "current_ratio",
            "days_in_ar", "debt_ratio", "equity_ratio", "occupancy", "medicare_share",
            "medicaid_share", "log_beds", "revenue_growth", "total_margin_chg"]
    tr = lab.loc[train_mask_lab, cols + ["distress_next"]]
    corr = tr.corr(method="spearman")
    names = [LABELS[c] for c in cols] + ["Distressed next year"]

    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pairs.append((abs(corr.loc[a, b]), corr.loc[a, b], a, b))
    say("Strongest feature pairs:")
    for _, v, a, b in sorted(pairs, reverse=True)[:8]:
        say(f"  {LABELS[a]} / {LABELS[b]}: rho = {v:.2f}")
    say("Correlation with next-year distress:")
    for c in corr["distress_next"].drop("distress_next").sort_values().index:
        say(f"  {LABELS[c]:<28} rho = {corr.loc[c, 'distress_next']:.2f}")

    fig, ax = plt.subplots(figsize=(FULL_W, 5.4))
    sub = corr.iloc[1:, :-1]                     # drop the diagonal of 1.00s
    mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
    data = np.ma.array(sub.values, mask=mask)
    row_names, col_names = names[1:], names[:-1]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "div", ["#2a78d6", "#f0efec", "#eb6834"])
    im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1)
    for i in range(len(row_names)):
        for j in range(i + 1):
            v = sub.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 else INK)
    ax.set_xticks(range(len(col_names)), col_names, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(row_names)), row_names, fontsize=7.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Spearman ρ", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    save(fig, "eda_correlation_heatmap")


def fig_states(lab, min_rows=150):
    section(f"10. DISTRESS RATE BY STATE (states with at least {min_rows} labeled hospital-years)")
    g = lab.groupby("state_code")["distress_next"].agg(["mean", "size"])
    g = g[g["size"] >= min_rows].sort_values("mean")
    say(f"States shown: {len(g)}; range {pct(g['mean'].min())} ({g.index[0]}) to "
        f"{pct(g['mean'].max())} ({g.index[-1]})")
    say("Highest 5: " + ", ".join(f"{s} {pct(m)}" for s, m in g["mean"].tail(5)[::-1].items()))
    say("Lowest 5: " + ", ".join(f"{s} {pct(m)}" for s, m in g["mean"].head(5).items()))

    fig, ax = plt.subplots(figsize=(FULL_W, max(3.0, 0.15 * len(g) + 0.8)))
    ax.barh(g.index, g["mean"], color=DISTRESSED, height=0.7)
    ax.axvline(lab["distress_next"].mean(), color=INK, lw=0.8, ls="--")
    ax.text(lab["distress_next"].mean(), len(g) - 0.3, " national rate", fontsize=7.5, va="bottom")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Share of hospital-years distressed next year")
    pct_axis(ax)
    ax.set_ylim(-0.6, len(g) - 0.4)
    save(fig, "eda_distress_rate_by_state")


if __name__ == "__main__":
    raw, df, bounds, train_mask = load_data()
    lab = sample_and_target(df)
    fig_winsorizing(raw, df, bounds, train_mask)
    fig_log_transform(raw)
    stats_by_class(lab)
    fig_by_class(lab)
    group_rates(lab)
    fig_trend_by_type(lab)
    fig_ratios_over_time(df)
    fig_persistence(lab)
    fig_correlations(lab, lab["file_year"].isin(TRAIN_YEARS))
    fig_states(lab)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nSummary written to {SUMMARY_PATH.relative_to(PROJECT_DIR)}")
