from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


INPUT_PANEL_CSV = Path(r"F:\icarus fund\Quantitative AI financial modeling\regression\sp500_alpha_target_panel.csv")
OUTPUT_DIR = Path(r"F:\icarus fund\Quantitative AI financial modeling\regression")

DETAIL_1D = OUTPUT_DIR / "daily_single_alpha_regression_detail_1d.csv"
DETAIL_5D = OUTPUT_DIR / "daily_single_alpha_regression_detail_5d.csv"
SUMMARY_1D = OUTPUT_DIR / "single_alpha_regression_stata_style_summary_1d.csv"
SUMMARY_5D = OUTPUT_DIR / "single_alpha_regression_stata_style_summary_5d.csv"
REPORT_5D = OUTPUT_DIR / "single_alpha_regression_stata_style_report_5d.txt"

TARGETS = ["ret_fwd_1d", "ret_fwd_5d"]
MIN_OBS = 50


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def detect_columns(path: Path) -> tuple[list[str], list[str]]:
    header = pd.read_csv(path, nrows=0)
    columns = list(header.columns)
    alpha_z_cols = [c for c in columns if c.startswith("alpha_") and c.endswith("_z")]
    required = ["date", "ticker"] + TARGETS
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")
    if not alpha_z_cols:
        raise ValueError("No alpha_*_z columns were found in the panel.")
    return required, alpha_z_cols


def load_model_panel(path: Path) -> tuple[pd.DataFrame, list[str]]:
    print(f"Inspecting columns from: {path}")
    required, alpha_z_cols = detect_columns(path)
    usecols = required + alpha_z_cols
    print(f"Loading {len(usecols)} columns from panel CSV...")
    panel = pd.read_csv(path, usecols=usecols)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["ticker"] = panel["ticker"].astype(str).str.strip().str.upper()
    panel = panel.dropna(subset=["date", "ticker"]).sort_values(["date", "ticker"]).reset_index(drop=True)
    for col in TARGETS + alpha_z_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    return panel, alpha_z_cols


def ols_one_alpha(y: np.ndarray, x: np.ndarray) -> dict[str, float] | None:
    valid = ~(np.isnan(y) | np.isnan(x))
    yv = y[valid]
    xv = x[valid]
    n = len(yv)
    if n < MIN_OBS:
        return None

    x_mean = float(np.mean(xv))
    y_mean = float(np.mean(yv))
    sxx = float(np.sum((xv - x_mean) ** 2))
    if sxx <= 0:
        return None

    beta = float(np.sum((xv - x_mean) * (yv - y_mean)) / sxx)
    intercept = y_mean - beta * x_mean
    fitted = intercept + beta * xv
    resid = yv - fitted
    sse = float(np.sum(resid**2))
    sst = float(np.sum((yv - y_mean) ** 2))
    df_resid = n - 2
    if df_resid <= 0:
        return None

    residual_std_error = math.sqrt(sse / df_resid)
    std_error = residual_std_error / math.sqrt(sxx)
    t_stat = beta / std_error if std_error > 0 else np.nan
    p_value = float(2 * student_t.sf(abs(t_stat), df=df_resid)) if not np.isnan(t_stat) else np.nan
    r_squared = 1.0 - sse / sst if sst > 0 else np.nan
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / df_resid if not np.isnan(r_squared) else np.nan

    return {
        "n_obs": float(n),
        "intercept": intercept,
        "beta": beta,
        "std_error": std_error,
        "t_stat": t_stat,
        "p_value": p_value,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "residual_std_error": residual_std_error,
    }


def run_daily_regressions(panel: pd.DataFrame, alpha_z_cols: list[str], target: str) -> pd.DataFrame:
    print(f"\nRunning daily cross-sectional regressions for {target}...")
    rows: list[dict[str, Any]] = []
    grouped_indices = list(panel.groupby("date", sort=True).indices.items())
    target_values = panel[target].to_numpy(dtype=float)

    for alpha_idx, alpha in enumerate(alpha_z_cols, start=1):
        alpha_values = panel[alpha].to_numpy(dtype=float)
        for date_value, positions in grouped_indices:
            stats = ols_one_alpha(target_values[positions], alpha_values[positions])
            if stats is None:
                continue
            rows.append(
                {
                    "date": pd.Timestamp(date_value).date().isoformat(),
                    "target": target,
                    "alpha": alpha,
                    "n_obs": int(stats["n_obs"]),
                    "intercept": stats["intercept"],
                    "beta": stats["beta"],
                    "std_error": stats["std_error"],
                    "t_stat": stats["t_stat"],
                    "p_value": stats["p_value"],
                    "r_squared": stats["r_squared"],
                    "adj_r_squared": stats["adj_r_squared"],
                    "residual_std_error": stats["residual_std_error"],
                }
            )
        if alpha_idx % 10 == 0 or alpha_idx == len(alpha_z_cols):
            print(f"  {target}: processed {alpha_idx}/{len(alpha_z_cols)} alpha_z columns")

    detail = pd.DataFrame(rows)
    if not detail.empty:
        detail["date"] = pd.to_datetime(detail["date"])
        detail = detail.sort_values(["alpha", "date"]).reset_index(drop=True)
    print(f"{target}: generated {len(detail)} daily regression rows")
    return detail


def split_dates(all_dates: pd.Series) -> dict[str, set[pd.Timestamp]]:
    dates = pd.Series(pd.to_datetime(all_dates.dropna().unique())).sort_values().reset_index(drop=True)
    n = len(dates)
    train_end = int(math.floor(0.60 * n))
    val_end = int(math.floor(0.80 * n))
    return {
        "train": set(dates.iloc[:train_end]),
        "val": set(dates.iloc[train_end:val_end]),
        "test": set(dates.iloc[val_end:]),
    }


def beta_mean_t_p(values: pd.Series) -> tuple[float, float, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean_beta = float(vals.mean())
    if n < 2:
        return mean_beta, np.nan, np.nan
    std_beta = float(vals.std(ddof=1))
    if std_beta == 0 or np.isnan(std_beta):
        return mean_beta, np.nan, np.nan
    t_stat = mean_beta / (std_beta / math.sqrt(n))
    p_value = float(2 * student_t.sf(abs(t_stat), df=n - 1))
    return mean_beta, t_stat, p_value


def summarize_alpha_regressions(
    detail: pd.DataFrame, alpha_z_cols: list[str], target: str, date_splits: dict[str, set[pd.Timestamp]]
) -> pd.DataFrame:
    print(f"Summarizing alpha-level results for {target}...")
    rows: list[dict[str, Any]] = []
    detail = detail.copy()
    if not detail.empty:
        detail["date"] = pd.to_datetime(detail["date"])

    for alpha in alpha_z_cols:
        g = detail.loc[detail["alpha"].eq(alpha)].copy()
        n_days = len(g)
        mean_beta, beta_t_stat, beta_p_value = beta_mean_t_p(g["beta"] if n_days else pd.Series(dtype=float))
        std_beta = float(g["beta"].std(ddof=1)) if n_days > 1 else np.nan

        split_stats: dict[str, tuple[float, float, float]] = {}
        for split_name, split_date_set in date_splits.items():
            split_g = g.loc[g["date"].isin(split_date_set)]
            split_stats[split_name] = beta_mean_t_p(split_g["beta"])

        train_beta_mean, train_beta_t_stat, train_beta_p_value = split_stats["train"]
        val_beta_mean, val_beta_t_stat, val_beta_p_value = split_stats["val"]
        test_beta_mean, test_beta_t_stat, test_beta_p_value = split_stats["test"]

        val_test_signs = [np.sign(val_beta_mean), np.sign(test_beta_mean)]
        all_signs = [np.sign(train_beta_mean), np.sign(val_beta_mean), np.sign(test_beta_mean)]
        same_direction_val_test = (
            bool(val_test_signs[0] != 0 and val_test_signs[0] == val_test_signs[1])
            if not any(np.isnan(s) for s in val_test_signs)
            else False
        )
        same_direction_all = (
            bool(all_signs[0] != 0 and all_signs[0] == all_signs[1] == all_signs[2])
            if not any(np.isnan(s) for s in all_signs)
            else False
        )

        rows.append(
            {
                "alpha": alpha,
                "target": target,
                "n_days": n_days,
                "mean_beta": mean_beta,
                "std_beta": std_beta,
                "beta_t_stat": beta_t_stat,
                "beta_p_value": beta_p_value,
                "mean_std_error": float(g["std_error"].mean()) if n_days else np.nan,
                "mean_daily_t_stat": float(g["t_stat"].mean()) if n_days else np.nan,
                "median_daily_t_stat": float(g["t_stat"].median()) if n_days else np.nan,
                "mean_daily_p_value": float(g["p_value"].mean()) if n_days else np.nan,
                "significant_days_ratio_5pct": float((g["p_value"] < 0.05).mean()) if n_days else np.nan,
                "positive_beta_ratio": float((g["beta"] > 0).mean()) if n_days else np.nan,
                "negative_beta_ratio": float((g["beta"] < 0).mean()) if n_days else np.nan,
                "avg_daily_r_squared": float(g["r_squared"].mean()) if n_days else np.nan,
                "avg_daily_adj_r_squared": float(g["adj_r_squared"].mean()) if n_days else np.nan,
                "train_beta_mean": train_beta_mean,
                "val_beta_mean": val_beta_mean,
                "test_beta_mean": test_beta_mean,
                "train_beta_t_stat": train_beta_t_stat,
                "val_beta_t_stat": val_beta_t_stat,
                "test_beta_t_stat": test_beta_t_stat,
                "train_beta_p_value": train_beta_p_value,
                "val_beta_p_value": val_beta_p_value,
                "test_beta_p_value": test_beta_p_value,
                "same_direction_val_test": same_direction_val_test,
                "same_direction_all": same_direction_all,
            }
        )

    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "(none)"
    return df.head(max_rows).to_string(index=False)


def save_5d_report(summary_5d: pd.DataFrame) -> None:
    top_abs_t = summary_5d.sort_values("beta_t_stat", key=lambda s: s.abs(), ascending=False).head(20)
    top_low_p = summary_5d.sort_values("beta_p_value", ascending=True, na_position="last").head(20)
    stable_positive = summary_5d.loc[
        summary_5d["same_direction_all"]
        & (summary_5d["train_beta_mean"] > 0)
        & (summary_5d["val_beta_mean"] > 0)
        & (summary_5d["test_beta_mean"] > 0)
    ].sort_values("beta_p_value", ascending=True, na_position="last")
    stable_negative = summary_5d.loc[
        summary_5d["same_direction_all"]
        & (summary_5d["train_beta_mean"] < 0)
        & (summary_5d["val_beta_mean"] < 0)
        & (summary_5d["test_beta_mean"] < 0)
    ].sort_values("beta_p_value", ascending=True, na_position="last")
    unstable = summary_5d.loc[~summary_5d["same_direction_all"]].sort_values(
        "beta_t_stat", key=lambda s: s.abs(), ascending=False
    )
    high_r2 = summary_5d.sort_values("avg_daily_r_squared", ascending=False, na_position="last").head(20)
    exclude = summary_5d.loc[(summary_5d["beta_p_value"].isna()) | (summary_5d["beta_p_value"] > 0.05) | (~summary_5d["same_direction_all"])].copy()
    exclude = exclude.sort_values(["same_direction_all", "beta_p_value"], ascending=[True, True], na_position="last")

    cols_core = [
        "alpha",
        "n_days",
        "mean_beta",
        "std_beta",
        "beta_t_stat",
        "beta_p_value",
        "positive_beta_ratio",
        "avg_daily_r_squared",
        "train_beta_mean",
        "val_beta_mean",
        "test_beta_mean",
        "same_direction_all",
    ]

    report = f"""Stata-Style Single-Alpha Regression Report: 5-Day Forward Return

Method
- Each alpha is tested separately using daily cross-sectional OLS.
- Regression: ret_fwd_5d_i,t = intercept_t + beta_t * alpha_z_i,t + error_i,t.
- Dates with fewer than {MIN_OBS} valid observations are skipped.
- Alpha-level beta t-stat = mean daily beta / (std daily beta / sqrt(n_days)).
- Alpha-level beta p-value is two-sided from Student t with n_days - 1 degrees of freedom.
- Chronological split: first 60% train, next 20% validation, last 20% test.

Top 20 Alphas by Absolute beta_t_stat
{format_table(top_abs_t[cols_core], 20)}

Top 20 Alphas by Lowest beta_p_value
{format_table(top_low_p[cols_core], 20)}

Stable Positive Beta in Train / Validation / Test
{format_table(stable_positive[cols_core], 20)}

Stable Negative Beta in Train / Validation / Test
{format_table(stable_negative[cols_core], 20)}

Unstable Signs
{format_table(unstable[cols_core], 30)}

High Average Daily R-squared
{format_table(high_r2[cols_core], 20)}

Alphas to Exclude Due to Weak p-values or Unstable Beta Direction
Rule used: exclude when beta_p_value > 0.05, beta_p_value is missing, or train/validation/test beta signs are not all aligned.
{format_table(exclude[cols_core], 50)}
"""
    REPORT_5D.write_text(report, encoding="utf-8")


def main() -> None:
    require_file(INPUT_PANEL_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel, alpha_z_cols = load_model_panel(INPUT_PANEL_CSV)
    date_splits = split_dates(panel["date"])

    detail_1d = run_daily_regressions(panel, alpha_z_cols, "ret_fwd_1d")
    detail_5d = run_daily_regressions(panel, alpha_z_cols, "ret_fwd_5d")

    summary_1d = summarize_alpha_regressions(detail_1d, alpha_z_cols, "ret_fwd_1d", date_splits)
    summary_5d = summarize_alpha_regressions(detail_5d, alpha_z_cols, "ret_fwd_5d", date_splits)

    print("\nSaving outputs...")
    detail_1d.to_csv(DETAIL_1D, index=False)
    detail_5d.to_csv(DETAIL_5D, index=False)
    summary_1d.to_csv(SUMMARY_1D, index=False)
    summary_5d.to_csv(SUMMARY_5D, index=False)
    save_5d_report(summary_5d)

    print("\nRun complete.")
    print(f"Number of alpha_z columns tested: {len(alpha_z_cols)}")
    print(f"Number of targets tested: {len(TARGETS)}")
    print(f"Number of regression rows generated: {len(detail_1d) + len(detail_5d)}")
    print("\nTop 20 5-day alpha regression results by absolute beta_t_stat:")
    top_20_5d = summary_5d.sort_values("beta_t_stat", key=lambda s: s.abs(), ascending=False).head(20)
    print(
        format_table(
            top_20_5d[
                [
                    "alpha",
                    "target",
                    "n_days",
                    "mean_beta",
                    "std_beta",
                    "beta_t_stat",
                    "beta_p_value",
                    "positive_beta_ratio",
                    "avg_daily_r_squared",
                    "same_direction_all",
                ]
            ],
            20,
        )
    )
    print("\nOutput files:")
    for path in [DETAIL_1D, DETAIL_5D, SUMMARY_1D, SUMMARY_5D, REPORT_5D]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
