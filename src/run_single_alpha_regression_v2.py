"""Step 7 v2 alpha validation with robust cross-sectional diagnostics.

This rerun intentionally de-emphasizes pooled OLS because stacked panel
observations are not independent. It validates single alphas using:

* Fama-MacBeth cross-sectional regressions by date
* daily Spearman/Pearson rank IC
* decile spreads for continuous features
* binary-group spreads for Alpha 37/38

No XGBoost, portfolio construction, transaction-cost backtest, or execution
backtest is run here.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .io_utils import write_csv


TARGET_LABELS = [
    "forward_5d_excess_return",
    "tradable_forward_5d_return",
    "forward_21d_excess_return",
    "tradable_forward_21d_return",
]

LABEL_COLUMNS = [
    "close_to_close_forward_1d_return",
    "close_to_close_forward_5d_return",
    "close_to_close_forward_21d_return",
    "tradable_forward_1d_return",
    "tradable_forward_5d_return",
    "tradable_forward_21d_return",
    "vwap_forward_1d_return",
    "vwap_forward_5d_return",
    "vwap_forward_21d_return",
    "forward_1d_excess_return",
    "forward_5d_excess_return",
    "forward_21d_excess_return",
]

FEATURE_VARIANTS = {
    "cs_rank": "_cs_rank",
    "cs_zscore": "_cs_zscore",
    "sector_neutral_rank": "_sector_neutral_rank",
}

CONTROL_COLUMNS = {
    60: "alpha_60_market_beta_cs_zscore",
    57: "alpha_57_63_day_realized_volatility_cs_zscore",
    65: "alpha_65_dollar_volume_cs_zscore",
}

BINARY_ALPHA_IDS = {37, 38}
MIN_VALID_DATES = 50
MIN_VALID_OBS = 1000
MIN_CROSS_SECTION_OBS = 20


@dataclass(frozen=True)
class FeatureSpec:
    """One transformed alpha feature to validate."""

    alpha_id: int
    feature_name: str
    feature_column: str
    feature_variant: str
    raw_column: str


@dataclass
class Step7V2QA:
    """QA state for Step 7 v2."""

    failures: list[str] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    tested_count: int = 0
    target_count: int = 0
    feature_count: int = 0

    @property
    def passed(self) -> bool:
        return not self.failures


def parse_bool(value: str | bool) -> bool:
    """Parse flexible true/false CLI values."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def _infer_newey_west_lag(target_label: str) -> int:
    if "21d" in target_label:
        return 21
    if "5d" in target_label:
        return 5
    return 0


def _standard_t_stat(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(series)
    if n < 2:
        return float("nan")
    std = float(np.std(series, ddof=1))
    if std == 0 or not np.isfinite(std):
        return float("nan")
    return float(np.mean(series) / (std / math.sqrt(n)))


def _newey_west_t_stat(values: pd.Series, lag: int) -> float:
    """Newey-West t-stat for the mean of a time series."""
    series = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(series)
    if n < 2:
        return float("nan")
    lag = max(0, min(int(lag), n - 1))
    centered = series - float(np.mean(series))
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_var = gamma0
    for k in range(1, lag + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        weight = 1.0 - k / (lag + 1.0)
        long_run_var += 2.0 * weight * gamma
    if long_run_var <= 0 or not np.isfinite(long_run_var):
        return _standard_t_stat(pd.Series(series))
    standard_error = math.sqrt(long_run_var / n)
    if standard_error == 0 or not np.isfinite(standard_error):
        return float("nan")
    return float(np.mean(series) / standard_error)


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(left, errors="coerce"), "y": pd.to_numeric(right, errors="coerce")})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2 or frame["x"].std(ddof=1) == 0 or frame["y"].std(ddof=1) == 0:
        return float("nan")
    return float(frame["x"].corr(frame["y"]))


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(left, errors="coerce"), "y": pd.to_numeric(right, errors="coerce")})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2:
        return float("nan")
    x_rank = frame["x"].rank(method="average")
    y_rank = frame["y"].rank(method="average")
    if x_rank.std(ddof=1) == 0 or y_rank.std(ddof=1) == 0:
        return float("nan")
    return float(x_rank.corr(y_rank))


def _ols_by_date(group: pd.DataFrame, target_col: str, feature_col: str, controls: list[str]) -> dict[str, float]:
    """Run one date's cross-sectional OLS and return alpha slope plus R2."""
    columns = [target_col, feature_col, *controls]
    frame = group[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < max(MIN_CROSS_SECTION_OBS, len(controls) + 3):
        return {"beta_t": np.nan, "cs_r2": np.nan}
    y = frame[target_col].to_numpy(dtype=float)
    x = frame[[feature_col, *controls]].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(frame)), x])
    try:
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"beta_t": np.nan, "cs_r2": np.nan}
    fitted = design @ beta
    residual = y - fitted
    rss = float(np.dot(residual, residual))
    tss = float(np.dot(y - y.mean(), y - y.mean()))
    return {"beta_t": float(beta[1]), "cs_r2": float(1.0 - rss / tss) if tss > 0 else np.nan}


def _inf_count(frame: pd.DataFrame) -> int:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return int(np.isinf(numeric.to_numpy(dtype=float)).sum())


def load_dataset(path: Path) -> pd.DataFrame:
    """Load model dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Model dataset not found: {path}")
    data = pd.read_parquet(path)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["ticker"] = data["ticker"].astype(str).str.strip().str.upper()
    return data.sort_values(["date", "ticker"]).reset_index(drop=True)


def validate_input(data: pd.DataFrame) -> list[str]:
    """Validate dataset-level prerequisites."""
    failures: list[str] = []
    for column in ["date", "ticker"]:
        if column not in data.columns:
            failures.append(f"missing required column: {column}")
    if {"date", "ticker"}.issubset(data.columns):
        duplicate_count = int(data.duplicated(["date", "ticker"]).sum())
        if duplicate_count:
            failures.append(f"duplicate date-ticker rows: {duplicate_count}")
    missing_labels = [label for label in TARGET_LABELS if label not in data.columns]
    if missing_labels:
        failures.append(f"missing target labels: {missing_labels}")
    return failures


def discover_features(data: pd.DataFrame) -> list[FeatureSpec]:
    """Discover requested transformed alpha features, skipping Alpha 35."""
    label_set = set(LABEL_COLUMNS)
    features: list[FeatureSpec] = []
    for spec in config.ALPHA_SPECS:
        if spec.alpha_id == 35:
            continue
        raw_column = f"{spec.column}_raw"
        for variant, suffix in FEATURE_VARIANTS.items():
            column = f"{spec.column}{suffix}"
            if column in data.columns and column not in label_set:
                features.append(
                    FeatureSpec(
                        alpha_id=spec.alpha_id,
                        feature_name=spec.name,
                        feature_column=column,
                        feature_variant=variant,
                        raw_column=raw_column,
                    )
                )
    return features


def controls_for_feature(data: pd.DataFrame, feature: FeatureSpec, with_controls: bool) -> list[str]:
    """Return optional controls, excluding controls for the same alpha as the tested feature."""
    if not with_controls:
        return []
    controls: list[str] = []
    for alpha_id, column in CONTROL_COLUMNS.items():
        if alpha_id == feature.alpha_id:
            continue
        if column in data.columns and column != feature.feature_column:
            controls.append(column)
    return controls


def prepare_frame(data: pd.DataFrame, feature: FeatureSpec, target_label: str, controls: list[str]) -> pd.DataFrame:
    """Drop only selected feature/label/control NaNs for this regression."""
    columns = ["date", "ticker", feature.feature_column, target_label, *controls]
    if "sector" in data.columns:
        columns.append("sector")
    if feature.alpha_id in BINARY_ALPHA_IDS and feature.raw_column in data.columns:
        columns.append(feature.raw_column)
    frame = data[columns].replace([np.inf, -np.inf], np.nan).copy()
    frame = frame.dropna(subset=[feature.feature_column, target_label, *controls]).reset_index(drop=True)
    return frame


def enough_data(frame: pd.DataFrame) -> tuple[bool, str]:
    n_obs = len(frame)
    n_dates = frame["date"].nunique() if "date" in frame.columns else 0
    if n_obs < MIN_VALID_OBS:
        return False, f"valid observations below {MIN_VALID_OBS}: {n_obs}"
    if n_dates < MIN_VALID_DATES:
        return False, f"valid dates below {MIN_VALID_DATES}: {n_dates}"
    return True, ""


def run_fama_macbeth_and_ic(
    frame: pd.DataFrame,
    feature: FeatureSpec,
    target_label: str,
    controls: list[str],
    nw_lag: int,
) -> dict[str, Any]:
    """Run daily cross-sectional OLS plus daily rank IC and aggregate."""
    daily_rows: list[dict[str, Any]] = []
    for date_value, group in frame.groupby("date", sort=True):
        if len(group) < max(MIN_CROSS_SECTION_OBS, len(controls) + 3):
            continue
        ols = _ols_by_date(group, target_label, feature.feature_column, controls)
        spearman = _safe_spearman(group[feature.feature_column], group[target_label])
        pearson = _safe_corr(group[feature.feature_column], group[target_label])
        daily_rows.append(
            {
                "date": date_value,
                "beta_t": ols["beta_t"],
                "cs_r2": ols["cs_r2"],
                "rank_ic_spearman_t": spearman,
                "rank_ic_pearson_t": pearson,
            }
        )
    daily = pd.DataFrame(daily_rows)
    beta = daily["beta_t"].dropna() if "beta_t" in daily.columns else pd.Series(dtype=float)
    rank_ic = daily["rank_ic_spearman_t"].dropna() if "rank_ic_spearman_t" in daily.columns else pd.Series(dtype=float)
    rank_ic_std = float(rank_ic.std(ddof=1)) if len(rank_ic) > 1 else np.nan
    mean_rank_ic = float(rank_ic.mean()) if not rank_ic.empty else np.nan
    valid_dates = int(len(beta))
    return {
        "target_label": target_label,
        "alpha_id": feature.alpha_id,
        "feature_name": feature.feature_name,
        "feature_column": feature.feature_column,
        "feature_variant": feature.feature_variant,
        "n_obs": int(len(frame)),
        "n_dates": valid_dates,
        "n_tickers": int(frame["ticker"].nunique()),
        "mean_beta": float(beta.mean()) if not beta.empty else np.nan,
        "beta_t_stat_standard": _standard_t_stat(beta),
        "beta_t_stat_newey_west": _newey_west_t_stat(beta, nw_lag),
        "beta_positive_ratio": float((beta > 0).mean()) if not beta.empty else np.nan,
        "number_of_valid_dates": valid_dates,
        "mean_cs_r2": float(daily["cs_r2"].dropna().mean()) if "cs_r2" in daily.columns and daily["cs_r2"].notna().any() else np.nan,
        "mean_rank_ic_spearman": mean_rank_ic,
        "rank_ic_t_stat_standard": _standard_t_stat(rank_ic),
        "rank_ic_t_stat_newey_west": _newey_west_t_stat(rank_ic, nw_lag),
        "ICIR": float(mean_rank_ic / rank_ic_std) if rank_ic_std and np.isfinite(rank_ic_std) and rank_ic_std != 0 else np.nan,
        "positive_ic_ratio": float((rank_ic > 0).mean()) if not rank_ic.empty else np.nan,
        "mean_rank_ic_pearson": float(daily["rank_ic_pearson_t"].dropna().mean()) if "rank_ic_pearson_t" in daily.columns and daily["rank_ic_pearson_t"].notna().any() else np.nan,
        "newey_west_lag": int(nw_lag),
    }


def run_spread_diagnostic(frame: pd.DataFrame, feature: FeatureSpec, target_label: str, nw_lag: int) -> dict[str, Any]:
    """Run decile spread for continuous features or binary-group spread for Alpha 37/38."""
    rows: list[dict[str, Any]] = []
    is_binary = feature.alpha_id in BINARY_ALPHA_IDS and feature.raw_column in frame.columns
    diagnostic_type = "binary_group" if is_binary else "decile"
    for date_value, group in frame.groupby("date", sort=True):
        if len(group) < MIN_CROSS_SECTION_OBS:
            continue
        if is_binary:
            binary = pd.to_numeric(group[feature.raw_column], errors="coerce")
            usable = group.assign(_binary=binary).dropna(subset=["_binary", target_label])
            group_1 = usable.loc[usable["_binary"] == 1, target_label]
            group_0 = usable.loc[usable["_binary"] == 0, target_label]
            if group_1.empty or group_0.empty:
                continue
            high_return = float(group_1.mean())
            low_return = float(group_0.mean())
        else:
            rank = group[feature.feature_column].rank(method="average", pct=True)
            top = group.loc[rank >= 0.9, target_label]
            bottom = group.loc[rank <= 0.1, target_label]
            if top.empty or bottom.empty:
                continue
            high_return = float(top.mean())
            low_return = float(bottom.mean())
        rows.append({"date": date_value, "high_return": high_return, "low_return": low_return, "spread": high_return - low_return})

    daily = pd.DataFrame(rows)
    spread = daily["spread"].dropna() if "spread" in daily.columns else pd.Series(dtype=float)
    base = {
        "target_label": target_label,
        "alpha_id": feature.alpha_id,
        "feature_name": feature.feature_name,
        "feature_column": feature.feature_column,
        "feature_variant": feature.feature_variant,
        "diagnostic_type": diagnostic_type,
        "spread_t_stat_standard": _standard_t_stat(spread),
        "spread_t_stat_newey_west": _newey_west_t_stat(spread, nw_lag),
        "positive_spread_ratio": float((spread > 0).mean()) if not spread.empty else np.nan,
        "n_dates": int(len(spread)),
    }
    if diagnostic_type == "binary_group":
        base.update(
            {
                "mean_group_1_return": float(daily["high_return"].mean()) if "high_return" in daily and daily["high_return"].notna().any() else np.nan,
                "mean_group_0_return": float(daily["low_return"].mean()) if "low_return" in daily and daily["low_return"].notna().any() else np.nan,
                "mean_group_1_minus_group_0": float(spread.mean()) if not spread.empty else np.nan,
            }
        )
    else:
        base.update(
            {
                "mean_top_decile_return": float(daily["high_return"].mean()) if "high_return" in daily and daily["high_return"].notna().any() else np.nan,
                "mean_bottom_decile_return": float(daily["low_return"].mean()) if "low_return" in daily and daily["low_return"].notna().any() else np.nan,
                "mean_top_minus_bottom": float(spread.mean()) if not spread.empty else np.nan,
            }
        )
    return base


def build_combined_summary(fmb: pd.DataFrame, spread: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    """Merge Fama-MacBeth, rank IC, and spread metrics into one clean row per feature/label."""
    key = ["target_label", "alpha_id", "feature_name", "feature_column", "feature_variant"]
    combined = fmb.merge(spread, on=key, how="left", suffixes=("", "_spread"))
    combined["data_coverage"] = combined["n_obs"] / float(total_rows)
    combined["spread_metric"] = np.where(
        combined["diagnostic_type"].eq("binary_group"),
        combined.get("mean_group_1_minus_group_0"),
        combined.get("mean_top_minus_bottom"),
    )
    combined["warning_flags"] = ""
    combined.loc[combined["alpha_id"].isin(BINARY_ALPHA_IDS) & combined["diagnostic_type"].ne("binary_group"), "warning_flags"] += "binary_alpha_spread_not_binary;"
    combined.loc[combined["n_dates"] < 100, "warning_flags"] += "low_valid_dates;"
    return combined


def _sign(value: float) -> int:
    if not np.isfinite(value) or value == 0:
        return 0
    return 1 if value > 0 else -1


def add_stability_flags(combined: pd.DataFrame) -> pd.DataFrame:
    """Flag rows whose signs are stable between 5d and 21d labels of the same family."""
    out = combined.copy()
    out["stable_across_5d_21d"] = False
    out["sign_pair_label"] = ""
    pairs = [
        ("forward_5d_excess_return", "forward_21d_excess_return", "forward_excess"),
        ("tradable_forward_5d_return", "tradable_forward_21d_return", "tradable"),
    ]
    idx_cols = ["alpha_id", "feature_variant"]
    for label_5d, label_21d, pair_name in pairs:
        five = out[out["target_label"] == label_5d].set_index(idx_cols)
        twenty_one = out[out["target_label"] == label_21d].set_index(idx_cols)
        common = five.index.intersection(twenty_one.index)
        for idx in common:
            beta_consistent = _sign(float(five.loc[idx, "mean_beta"])) == _sign(float(twenty_one.loc[idx, "mean_beta"]))
            ic_consistent = _sign(float(five.loc[idx, "mean_rank_ic_spearman"])) == _sign(float(twenty_one.loc[idx, "mean_rank_ic_spearman"]))
            stable = bool(beta_consistent and ic_consistent and _sign(float(five.loc[idx, "mean_beta"])) != 0)
            mask = out["alpha_id"].eq(idx[0]) & out["feature_variant"].eq(idx[1]) & out["target_label"].isin([label_5d, label_21d])
            out.loc[mask, "stable_across_5d_21d"] = stable
            out.loc[mask, "sign_pair_label"] = pair_name
    return out


def build_top_alpha_summary(combined: pd.DataFrame) -> pd.DataFrame:
    """Rank alpha rows with robust primary/secondary/tertiary Newey-West metrics."""
    scored = add_stability_flags(combined)
    scored = scored[scored["n_dates"].ge(100) & scored["stable_across_5d_21d"]].copy()
    if scored.empty:
        return scored
    scored["abs_fmb_nw_t"] = scored["beta_t_stat_newey_west"].abs()
    scored["abs_rank_ic_nw_t"] = scored["rank_ic_t_stat_newey_west"].abs()
    scored["abs_spread_nw_t"] = scored["spread_t_stat_newey_west"].abs()
    scored = scored.sort_values(["abs_fmb_nw_t", "abs_rank_ic_nw_t", "abs_spread_nw_t"], ascending=False)
    keep_cols = [
        "target_label",
        "alpha_id",
        "feature_name",
        "feature_variant",
        "feature_column",
        "n_obs",
        "n_dates",
        "n_tickers",
        "mean_beta",
        "beta_t_stat_newey_west",
        "mean_rank_ic_spearman",
        "rank_ic_t_stat_newey_west",
        "spread_metric",
        "spread_t_stat_newey_west",
        "diagnostic_type",
        "data_coverage",
        "stable_across_5d_21d",
        "sign_pair_label",
        "warning_flags",
    ]
    return scored[keep_cols].head(50).reset_index(drop=True)


def run_validation(data: pd.DataFrame, with_controls: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Step7V2QA]:
    """Run Step 7 v2 for all requested labels and feature variants."""
    qa = Step7V2QA(target_count=len(TARGET_LABELS))
    features = discover_features(data)
    qa.feature_count = len(features)
    if not features:
        qa.failures.append("no valid transformed alpha features discovered")
    if any(feature.alpha_id == 35 for feature in features):
        qa.failures.append("Alpha 35 was not skipped")
    if set(LABEL_COLUMNS) & {feature.feature_column for feature in features}:
        qa.failures.append("label columns were discovered as features")

    fmb_rows: list[dict[str, Any]] = []
    spread_rows: list[dict[str, Any]] = []
    for target_label in TARGET_LABELS:
        nw_lag = _infer_newey_west_lag(target_label)
        for feature in features:
            controls = controls_for_feature(data, feature, with_controls)
            frame = prepare_frame(data, feature, target_label, controls)
            input_cols = [feature.feature_column, target_label, *controls]
            if feature.alpha_id in BINARY_ALPHA_IDS and feature.raw_column in frame.columns:
                input_cols.append(feature.raw_column)
            if _inf_count(frame[input_cols]) != 0:
                qa.skipped.append({"target_label": target_label, "alpha_id": feature.alpha_id, "feature_variant": feature.feature_variant, "reason": "inf/-inf in regression inputs"})
                continue
            enough, reason = enough_data(frame)
            if not enough:
                qa.skipped.append({"target_label": target_label, "alpha_id": feature.alpha_id, "feature_variant": feature.feature_variant, "reason": reason})
                continue
            fmb_rows.append(run_fama_macbeth_and_ic(frame, feature, target_label, controls, nw_lag))
            spread_rows.append(run_spread_diagnostic(frame, feature, target_label, nw_lag))
            qa.tested_count += 1

    fmb = pd.DataFrame(fmb_rows)
    spread = pd.DataFrame(spread_rows)
    combined = build_combined_summary(fmb, spread, total_rows=len(data)) if not fmb.empty else pd.DataFrame()
    top = build_top_alpha_summary(combined) if not combined.empty else pd.DataFrame()

    key = ["target_label", "alpha_id", "feature_variant"]
    for name, frame in [("fama_macbeth_summary_v2", fmb), ("spread_summary_v2", spread), ("combined_alpha_validation_summary_v2", combined)]:
        if not frame.empty and frame.duplicated(key).any():
            qa.failures.append(f"{name} has duplicate target_label + alpha_id + feature_variant rows")
    if fmb.empty or spread.empty or combined.empty:
        qa.failures.append("one or more required output summaries are empty")
    binary_spread = spread[spread["alpha_id"].isin(BINARY_ALPHA_IDS)] if not spread.empty else pd.DataFrame()
    if not binary_spread.empty and not binary_spread["diagnostic_type"].eq("binary_group").all():
        qa.failures.append("Alpha 37/38 did not use binary-group spread")
    if not fmb.empty and not set(fmb["target_label"].unique()) == set(TARGET_LABELS):
        qa.failures.append("not all target labels were tested")
    return fmb, spread, combined, top, qa


def print_top_table(title: str, table: pd.DataFrame, cols: list[str]) -> None:
    print(title)
    if table.empty:
        print("  none")
        return
    for _, row in table[cols].head(10).iterrows():
        print("  " + ", ".join(f"{col}={row[col]}" for col in cols))


def print_terminal_summary(
    fmb: pd.DataFrame,
    spread: pd.DataFrame,
    combined: pd.DataFrame,
    top: pd.DataFrame,
    qa: Step7V2QA,
    output_dir: Path,
) -> None:
    """Print requested Step 7 v2 terminal summary."""
    print("Step 7 v2 alpha validation")
    print(f"target labels tested: {TARGET_LABELS}")
    print(f"number of target labels tested: {qa.target_count}")
    print(f"number of alpha features discovered: {qa.feature_count}")
    print(f"number of alpha feature-label regressions tested: {qa.tested_count}")
    print(f"number skipped: {len(qa.skipped)}")
    print(f"output directory: {output_dir}")

    f5 = combined[combined["target_label"] == "forward_5d_excess_return"] if not combined.empty else pd.DataFrame()
    if not f5.empty:
        print_top_table(
            "top 10 alphas by Newey-West Fama-MacBeth t-stat for forward_5d_excess_return:",
            f5.assign(abs_metric=f5["beta_t_stat_newey_west"].abs()).sort_values("abs_metric", ascending=False),
            ["alpha_id", "feature_name", "feature_variant", "beta_t_stat_newey_west", "mean_beta", "n_dates"],
        )
        print_top_table(
            "top 10 alphas by Newey-West rank IC t-stat for forward_5d_excess_return:",
            f5.assign(abs_metric=f5["rank_ic_t_stat_newey_west"].abs()).sort_values("abs_metric", ascending=False),
            ["alpha_id", "feature_name", "feature_variant", "rank_ic_t_stat_newey_west", "mean_rank_ic_spearman", "n_dates"],
        )
        print_top_table(
            "top 10 alphas by Newey-West spread t-stat for forward_5d_excess_return:",
            f5.assign(abs_metric=f5["spread_t_stat_newey_west"].abs()).sort_values("abs_metric", ascending=False),
            ["alpha_id", "feature_name", "feature_variant", "diagnostic_type", "spread_t_stat_newey_west", "spread_metric", "n_dates"],
        )
    else:
        print("top tables for forward_5d_excess_return: none")

    f21 = combined[combined["target_label"] == "forward_21d_excess_return"] if not combined.empty else pd.DataFrame()
    if not f21.empty:
        print_top_table(
            "top 10 alphas by Newey-West Fama-MacBeth t-stat for forward_21d_excess_return:",
            f21.assign(abs_metric=f21["beta_t_stat_newey_west"].abs()).sort_values("abs_metric", ascending=False),
            ["alpha_id", "feature_name", "feature_variant", "beta_t_stat_newey_west", "mean_beta", "n_dates"],
        )
    else:
        print("top table for forward_21d_excess_return: none")

    print_top_table(
        "top alphas that are stable across 5d and 21d:",
        top,
        ["target_label", "alpha_id", "feature_name", "feature_variant", "beta_t_stat_newey_west", "rank_ic_t_stat_newey_west", "spread_t_stat_newey_west"],
    )
    print("warning: pooled OLS is not used in v2 because pooled panel t-stats can be inflated by non-independent observations.")
    print("Newey-West: manually computed with lag=5 for 5d labels and lag=21 for 21d labels.")
    print("Binary spread handling: Alpha 37 and Alpha 38 use binary group spread, not deciles.")
    print("No XGBoost, portfolio construction, full backtest, or transaction-cost backtest was run.")
    print(f"Final output: Step 7 v2 alpha validation status: {'PASS' if qa.passed else 'FAIL'}")
    if qa.failures:
        print("failed checks:")
        for failure in qa.failures:
            print(f"  {failure}")
    if qa.skipped:
        print("skipped examples:")
        for item in qa.skipped[:30]:
            print(f"  {item}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Step 7 v2 robust single-alpha validation.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--with-controls", type=parse_bool, default=False)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    data = load_dataset(args.input)
    failures = validate_input(data)
    if failures:
        print("Input validation failed:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)

    fmb, spread, combined, top, qa = run_validation(data, with_controls=args.with_controls)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(fmb, args.output_dir / "fama_macbeth_summary_v2.csv")
    write_csv(spread, args.output_dir / "spread_summary_v2.csv")
    write_csv(combined, args.output_dir / "combined_alpha_validation_summary_v2.csv")
    write_csv(top, args.output_dir / "top_alpha_summary_v2.csv")
    print_terminal_summary(fmb, spread, combined, top, qa, args.output_dir)
    if not qa.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
