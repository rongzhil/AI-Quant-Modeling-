from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REGRESSION_DIR = Path(r"F:\icarus fund\Quantitative AI financial modeling\regression")

PANEL_PARQUET = REGRESSION_DIR / "sp500_alpha_target_panel.parquet"
PANEL_CSV = REGRESSION_DIR / "sp500_alpha_target_panel.csv"
QUALITY_CSV = REGRESSION_DIR / "panel_data_quality_report.csv"
ALPHA_MISSING_CSV = REGRESSION_DIR / "alpha_missing_summary.csv"
UNMATCHED_CSV = REGRESSION_DIR / "unmatched_alpha_price_rows.csv"
RANK_IC_1D_CSV = REGRESSION_DIR / "alpha_rank_ic_summary_1d.csv"
RANK_IC_5D_CSV = REGRESSION_DIR / "alpha_rank_ic_summary_5d.csv"
REG_1D_CSV = REGRESSION_DIR / "single_alpha_regression_summary_1d.csv"
REG_5D_CSV = REGRESSION_DIR / "single_alpha_regression_summary_5d.csv"

SUMMARY_OUTPUT = REGRESSION_DIR / "panel_generation_diagnostic_summary.csv"
REPORT_OUTPUT = REGRESSION_DIR / "panel_generation_diagnostic_report.txt"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_panel() -> tuple[pd.DataFrame, Path]:
    if PANEL_PARQUET.exists():
        print(f"Loading panel from parquet: {PANEL_PARQUET}")
        return pd.read_parquet(PANEL_PARQUET), PANEL_PARQUET
    require_file(PANEL_CSV)
    print(f"Loading panel from CSV: {PANEL_CSV}")
    return pd.read_csv(PANEL_CSV), PANEL_CSV


def metric_value(quality: pd.DataFrame, metric: str, default: Any = None) -> Any:
    match = quality.loc[quality["metric"].eq(metric), "value"]
    if match.empty:
        return default
    return match.iloc[0]


def to_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def add_summary(rows: list[dict[str, Any]], section: str, item: str, value: Any, status: str, detail: str = "") -> None:
    rows.append(
        {
            "section": section,
            "item": item,
            "value": value,
            "status": status,
            "detail": detail,
        }
    )


def format_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "(none)"
    return df.head(max_rows).to_string(index=False)


def status_from_issues(failures: list[str], warnings: list[str]) -> str:
    if failures:
        return "FAIL"
    if warnings:
        return "WARNING"
    return "PASS"


def main() -> None:
    required = [
        QUALITY_CSV,
        ALPHA_MISSING_CSV,
        UNMATCHED_CSV,
        RANK_IC_1D_CSV,
        RANK_IC_5D_CSV,
        REG_1D_CSV,
        REG_5D_CSV,
    ]
    for path in required:
        require_file(path)
    if not PANEL_PARQUET.exists() and not PANEL_CSV.exists():
        raise FileNotFoundError(f"Neither panel file exists: {PANEL_PARQUET} or {PANEL_CSV}")

    panel, panel_source = load_panel()
    panel.columns = [str(c).strip() for c in panel.columns]
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")

    raw_alpha_cols = [
        c for c in panel.columns if c.startswith("alpha_") and not c.endswith("_z") and not c.endswith("_rank")
    ]
    alpha_z_cols = [c for c in panel.columns if c.startswith("alpha_") and c.endswith("_z")]
    alpha_rank_cols = [c for c in panel.columns if c.startswith("alpha_") and c.endswith("_rank")]

    final_shape = panel.shape
    n_tickers = int(panel["ticker"].nunique())
    n_dates = int(panel["date"].nunique())
    date_min = panel["date"].min()
    date_max = panel["date"].max()
    valid_ret_1d = int(panel["ret_fwd_1d"].notna().sum())
    valid_ret_5d = int(panel["ret_fwd_5d"].notna().sum())

    print("\nPanel overview")
    print(f"Final panel shape: {final_shape}")
    print(f"Number of tickers: {n_tickers}")
    print(f"Number of dates: {n_dates}")
    print(f"Date range: {date_min.date()} to {date_max.date()}")
    print(f"Raw alpha columns: {len(raw_alpha_cols)}")
    print(f"Alpha_z columns: {len(alpha_z_cols)}")
    print(f"Alpha_rank columns: {len(alpha_rank_cols)}")
    print(f"Valid ret_fwd_1d observations: {valid_ret_1d}")
    print(f"Valid ret_fwd_5d observations: {valid_ret_5d}")

    quality = pd.read_csv(QUALITY_CSV)
    alpha_missing = pd.read_csv(ALPHA_MISSING_CSV)
    unmatched = pd.read_csv(UNMATCHED_CSV)
    rank_ic_1d = pd.read_csv(RANK_IC_1D_CSV)
    rank_ic_5d = pd.read_csv(RANK_IC_5D_CSV)
    reg_1d = pd.read_csv(REG_1D_CSV)
    reg_5d = pd.read_csv(REG_5D_CSV)

    warnings: list[str] = []
    failures: list[str] = []
    summary_rows: list[dict[str, Any]] = []

    add_summary(summary_rows, "panel", "panel_source", str(panel_source), "PASS")
    add_summary(summary_rows, "panel", "final_panel_shape", f"{final_shape[0]} x {final_shape[1]}", "PASS")
    add_summary(summary_rows, "panel", "number_of_tickers", n_tickers, "PASS")
    add_summary(summary_rows, "panel", "number_of_dates", n_dates, "PASS")
    add_summary(summary_rows, "panel", "date_range", f"{date_min.date()} to {date_max.date()}", "PASS")
    add_summary(summary_rows, "panel", "raw_alpha_columns", len(raw_alpha_cols), "PASS")
    add_summary(summary_rows, "panel", "alpha_z_columns", len(alpha_z_cols), "PASS")
    add_summary(summary_rows, "panel", "alpha_rank_columns", len(alpha_rank_cols), "PASS")
    add_summary(summary_rows, "panel", "valid_ret_fwd_1d", valid_ret_1d, "PASS")
    add_summary(summary_rows, "panel", "valid_ret_fwd_5d", valid_ret_5d, "PASS")

    quality_warnings: list[str] = []
    duplicate_count = to_int(metric_value(quality, "duplicate_ticker_date_count"))
    unmatched_quality_count = to_int(metric_value(quality, "unmatched_ohlcv_row_count"))
    all_nan_count = to_int(metric_value(quality, "alpha_columns_entirely_nan"))
    high_missing_count = to_int(metric_value(quality, "alpha_columns_more_than_50pct_missing"))

    if duplicate_count > 0:
        failures.append(f"Duplicate ticker-date rows found: {duplicate_count}")
        quality_warnings.append(f"Duplicate ticker-date rows: {duplicate_count}")
    if unmatched_quality_count > 0:
        failures.append(f"Unmatched OHLCV rows found: {unmatched_quality_count}")
        quality_warnings.append(f"Unmatched OHLCV rows: {unmatched_quality_count}")
    if all_nan_count > 0:
        warnings.append(f"{all_nan_count} alpha columns are entirely NaN and must be excluded.")
        quality_warnings.append(f"Entirely NaN alpha columns: {all_nan_count}")
    if high_missing_count > 0:
        warnings.append(f"{high_missing_count} alpha columns have more than 50% missing values.")
        quality_warnings.append(f">50% missing alpha columns: {high_missing_count}")

    for _, row in quality.iterrows():
        metric = str(row["metric"])
        value = row["value"]
        if metric.startswith("missing_ratio_"):
            ratio = to_float(value)
            if np.isnan(ratio):
                if str(value) == "column_not_present" and metric == "missing_ratio_adjusted_close":
                    quality_warnings.append("adjusted_close column not present; close was used for targets.")
                    add_summary(summary_rows, "quality", metric, value, "WARNING", "Not fatal if adjusted close is unavailable.")
                else:
                    add_summary(summary_rows, "quality", metric, value, "WARNING", "Missing ratio unavailable.")
            elif ratio > 0:
                warnings.append(f"{metric} is {ratio:.2%}.")
                quality_warnings.append(f"{metric}: {ratio:.2%}")
                add_summary(summary_rows, "quality", metric, ratio, "WARNING")
            else:
                add_summary(summary_rows, "quality", metric, ratio, "PASS")

    add_summary(summary_rows, "quality", "duplicate_ticker_date_count", duplicate_count, "FAIL" if duplicate_count else "PASS")
    add_summary(summary_rows, "quality", "unmatched_ohlcv_row_count", unmatched_quality_count, "FAIL" if unmatched_quality_count else "PASS")
    add_summary(summary_rows, "quality", "alpha_columns_entirely_nan", all_nan_count, "WARNING" if all_nan_count else "PASS")
    add_summary(summary_rows, "quality", "alpha_columns_more_than_50pct_missing", high_missing_count, "WARNING" if high_missing_count else "PASS")

    all_nan_alphas = alpha_missing.loc[alpha_missing["all_nan_flag"].astype(bool), "alpha_name"].tolist()
    high_missing_alphas = alpha_missing.loc[alpha_missing["high_missing_flag"].astype(bool), "alpha_name"].tolist()
    exclude_alphas = sorted(set(all_nan_alphas) | set(high_missing_alphas))

    print("\nAlpha missingness")
    print(f"Alphas entirely NaN: {len(all_nan_alphas)}")
    print(", ".join(all_nan_alphas) if all_nan_alphas else "(none)")
    print(f"Alphas >50% missing: {len(high_missing_alphas)}")
    print(", ".join(high_missing_alphas) if high_missing_alphas else "(none)")
    print(f"Alphas to exclude from modeling: {len(exclude_alphas)}")
    print(", ".join(exclude_alphas) if exclude_alphas else "(none)")

    add_summary(summary_rows, "alpha_missingness", "alphas_entirely_nan", len(all_nan_alphas), "WARNING" if all_nan_alphas else "PASS", "; ".join(all_nan_alphas))
    add_summary(summary_rows, "alpha_missingness", "alphas_more_than_50pct_missing", len(high_missing_alphas), "WARNING" if high_missing_alphas else "PASS", "; ".join(high_missing_alphas))
    add_summary(summary_rows, "alpha_missingness", "alphas_excluded_from_modeling", len(exclude_alphas), "WARNING" if exclude_alphas else "PASS", "; ".join(exclude_alphas))

    unmatched_count = len(unmatched)
    unmatched_pct = unmatched_count / len(panel) if len(panel) else np.nan
    if unmatched_count:
        unmatched_by_ticker = unmatched.groupby("ticker").size().sort_values(ascending=False).head(20).reset_index(name="unmatched_rows")
        failures.append(f"Unmatched rows file contains {unmatched_count} rows.")
    else:
        unmatched_by_ticker = pd.DataFrame(columns=["ticker", "unmatched_rows"])

    print("\nUnmatched alpha-price rows")
    print(f"Unmatched rows: {unmatched_count}")
    print(f"Unmatched percentage relative to full panel: {unmatched_pct:.4%}")
    print("Tickers with most unmatched rows:")
    print(format_table(unmatched_by_ticker))

    add_summary(summary_rows, "unmatched", "unmatched_rows", unmatched_count, "FAIL" if unmatched_count else "PASS")
    add_summary(summary_rows, "unmatched", "unmatched_percentage", unmatched_pct, "FAIL" if unmatched_count else "PASS")

    rank_ic_5d["abs_t_stat"] = rank_ic_5d["t_stat"].abs()
    reg_5d["abs_beta_t_stat"] = reg_5d["beta_t_stat"].abs()
    top_ic_5d = rank_ic_5d.sort_values("abs_t_stat", ascending=False).head(20)
    top_reg_5d = reg_5d.sort_values("abs_beta_t_stat", ascending=False).head(20)

    print("\nTop 20 alphas by absolute 5-day Rank IC t-stat")
    print(format_table(top_ic_5d[["alpha", "target", "n_days", "mean_ic", "std_ic", "icir", "positive_ic_ratio", "t_stat"]], 20))

    print("\nTop 20 alphas by absolute 5-day single-alpha regression beta t-stat")
    print(format_table(top_reg_5d[["alpha", "n_days", "mean_beta", "std_beta", "beta_t_stat", "positive_beta_ratio", "avg_daily_r2"]], 20))

    direction = rank_ic_5d[["alpha", "mean_ic", "t_stat"]].merge(
        reg_5d[["alpha", "mean_beta", "beta_t_stat"]], on="alpha", how="inner"
    )
    direction = direction.dropna(subset=["mean_ic", "mean_beta"])
    direction = direction[(direction["mean_ic"] != 0) & (direction["mean_beta"] != 0)]
    direction["direction_agrees"] = np.sign(direction["mean_ic"]) == np.sign(direction["mean_beta"])
    agreement_ratio = float(direction["direction_agrees"].mean()) if len(direction) else np.nan
    disagreement = direction.loc[~direction["direction_agrees"]].copy()
    disagreement["combined_abs_t"] = disagreement["t_stat"].abs().fillna(0) + disagreement["beta_t_stat"].abs().fillna(0)
    disagreement = disagreement.sort_values("combined_abs_t", ascending=False).head(20)

    print("\nIC/regression direction agreement")
    print(f"Comparable alphas: {len(direction)}")
    print(f"Agreement ratio: {agreement_ratio:.2%}" if not np.isnan(agreement_ratio) else "Agreement ratio: unavailable")
    print("Largest direction disagreements:")
    print(format_table(disagreement[["alpha", "mean_ic", "t_stat", "mean_beta", "beta_t_stat"]]))

    agreement_status = "PASS"
    if not np.isnan(agreement_ratio) and agreement_ratio < 0.80:
        warnings.append(f"5-day IC and regression beta directions agree for only {agreement_ratio:.2%} of comparable alphas.")
        agreement_status = "WARNING"
    add_summary(summary_rows, "validation", "5d_ic_beta_direction_agreement_ratio", agreement_ratio, agreement_status)

    if len(alpha_z_cols) != len(alpha_rank_cols):
        failures.append("alpha_z and alpha_rank column counts differ.")
    if len(raw_alpha_cols) == 0 or len(alpha_z_cols) == 0:
        failures.append("No usable alpha feature columns were detected.")
    if valid_ret_1d == 0 or valid_ret_5d == 0:
        failures.append("Forward return targets have no valid observations.")

    overall_status = status_from_issues(failures, warnings)
    ready_for_stability_testing = "NO" if failures else "YES_WITH_WARNINGS" if warnings else "YES"
    add_summary(summary_rows, "overall", "status", overall_status, overall_status)
    add_summary(summary_rows, "overall", "ready_for_stability_testing", ready_for_stability_testing, overall_status)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    serious_issues = failures if failures else ["No serious blocking issues found."]
    warning_lines = warnings if warnings else ["No warnings."]
    quality_warning_lines = quality_warnings if quality_warnings else ["No quality-report warnings."]

    agreement_text = f"{agreement_ratio:.2%}" if not np.isnan(agreement_ratio) else "unavailable"

    report = f"""S&P 500 Alpha Panel Generation Diagnostic Report
Generated from: {REGRESSION_DIR}

Overall status: {overall_status}
Ready for stability testing: {ready_for_stability_testing}

Panel Overview
- Source file: {panel_source}
- Final panel shape: {final_shape[0]} rows x {final_shape[1]} columns
- Number of tickers: {n_tickers}
- Number of dates: {n_dates}
- Date range: {date_min.date()} to {date_max.date()}
- Raw alpha columns: {len(raw_alpha_cols)}
- alpha_z columns: {len(alpha_z_cols)}
- alpha_rank columns: {len(alpha_rank_cols)}
- Valid ret_fwd_1d observations: {valid_ret_1d}
- Valid ret_fwd_5d observations: {valid_ret_5d}

Quality Report Warnings
{chr(10).join(f"- {line}" for line in quality_warning_lines)}

Alpha Missingness
- Entirely NaN alphas ({len(all_nan_alphas)}): {", ".join(all_nan_alphas) if all_nan_alphas else "(none)"}
- >50% missing alphas ({len(high_missing_alphas)}): {", ".join(high_missing_alphas) if high_missing_alphas else "(none)"}
- Exclude from modeling ({len(exclude_alphas)}): {", ".join(exclude_alphas) if exclude_alphas else "(none)"}

Unmatched Alpha-Price Rows
- Unmatched rows: {unmatched_count}
- Unmatched percentage relative to full panel: {unmatched_pct:.4%}
- Tickers with most unmatched rows:
{format_table(unmatched_by_ticker)}

Top 20 Alphas by Absolute 5-Day Rank IC t-stat
{format_table(top_ic_5d[["alpha", "target", "n_days", "mean_ic", "std_ic", "icir", "positive_ic_ratio", "t_stat"]], 20)}

Top 20 Alphas by Absolute 5-Day Single-Alpha Regression beta t-stat
{format_table(top_reg_5d[["alpha", "n_days", "mean_beta", "std_beta", "beta_t_stat", "positive_beta_ratio", "avg_daily_r2"]], 20)}

IC and Regression Direction Agreement
- Comparable alphas: {len(direction)}
- Agreement ratio: {agreement_text}
- Largest disagreements:
{format_table(disagreement[["alpha", "mean_ic", "t_stat", "mean_beta", "beta_t_stat"]])}

Serious Issues Before Continuing
{chr(10).join(f"- {line}" for line in serious_issues)}

Warnings Before Continuing
{chr(10).join(f"- {line}" for line in warning_lines)}

Conclusion
The panel is ready for stability testing if blocking failures are absent. Exclude the listed high-missing/all-NaN alpha columns from modeling inputs, and treat missing adjusted_close as a pricing convention note rather than a merge failure.
"""
    REPORT_OUTPUT.write_text(report, encoding="utf-8")

    print("\nDiagnostic outputs saved")
    print(f"Summary CSV: {SUMMARY_OUTPUT}")
    print(f"Text report: {REPORT_OUTPUT}")
    print(f"\nOverall status: {overall_status}")
    print(f"Ready for stability testing: {ready_for_stability_testing}")
    if failures:
        print("\nSerious issues:")
        for issue in failures:
            print(f"- {issue}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
