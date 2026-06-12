"""Terminal-only QA checks for batch and merged alpha panels."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .alpha_features import compute_alpha_features


FORMULA_ALPHA_IDS = [32, 37, 38, 46, 55, 62, 67, 72, 73]
LEAKAGE_ALPHA_IDS = [26, 32, 37, 46, 55, 62, 67, 72, 73]


@dataclass
class BatchQAResult:
    """QA result for one processed folder."""

    passed: bool
    failures: list[str] = field(default_factory=list)
    duplicate_date_ticker_count: int = 0
    alpha_column_count: int = 0
    inf_count: int = 0
    missing_rates: dict[str, float] = field(default_factory=dict)
    formula_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    formula_pass: bool = False
    leakage_pass: bool = False


@dataclass
class MergeQAResult:
    """QA result for final merged panel."""

    passed: bool
    failures: list[str] = field(default_factory=list)
    duplicate_date_ticker_count: int = 0
    alpha_column_count: int = 0
    inf_count: int = 0
    missing_rates: dict[str, float] = field(default_factory=dict)
    lowest_coverage_tickers: pd.DataFrame = field(default_factory=pd.DataFrame)


def _alpha_columns(panel: pd.DataFrame) -> list[str]:
    return [col for col in panel.columns if col.startswith("alpha_")]


def _inf_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    numeric = df.apply(pd.to_numeric, errors="coerce")
    return int(np.isinf(numeric.to_numpy(dtype=float)).sum())


def _missing_rates(panel: pd.DataFrame) -> dict[str, float]:
    return {col: float(panel[col].isna().mean()) for col in config.ALPHA_COLUMNS if col in panel.columns}


def _expected_alpha_series(daily: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(daily["close"], errors="coerce").reset_index(drop=True)
    high = pd.to_numeric(daily["high"], errors="coerce").reset_index(drop=True)
    low = pd.to_numeric(daily["low"], errors="coerce").reset_index(drop=True)
    volume = pd.to_numeric(daily["volume"], errors="coerce").reset_index(drop=True)
    daily_vwap = pd.to_numeric(daily["daily_vwap"], errors="coerce").reset_index(drop=True)
    dates = daily["date"].astype(str).reset_index(drop=True)
    r1 = np.log(close / close.shift(1))

    expected = pd.DataFrame({"date": dates})
    expected[config.ALPHA_COLUMN_BY_ID[32]] = np.log(close.shift(21) / close.shift(252))
    prev_20_max = close.shift(1).rolling(20, min_periods=20).max()
    prev_20_min = close.shift(1).rolling(20, min_periods=20).min()
    expected[config.ALPHA_COLUMN_BY_ID[37]] = (close > prev_20_max).astype(float).where(prev_20_max.notna())
    expected[config.ALPHA_COLUMN_BY_ID[38]] = (close < prev_20_min).astype(float).where(prev_20_min.notna())
    expected[config.ALPHA_COLUMN_BY_ID[46]] = -(close - daily_vwap) / daily_vwap.replace(0, np.nan)
    true_range = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(
        axis=1
    )
    expected[config.ALPHA_COLUMN_BY_ID[55]] = true_range.rolling(14, min_periods=14).mean()
    expected[config.ALPHA_COLUMN_BY_ID[62]] = r1.rolling(63, min_periods=63).kurt()
    dollar_volume = close * volume
    expected[config.ALPHA_COLUMN_BY_ID[67]] = (r1.abs() / dollar_volume.replace(0, np.nan)).rolling(
        21, min_periods=21
    ).mean()
    return expected


def _compare_series(left: pd.Series, right: pd.Series, tolerance: float = 1e-10) -> tuple[float, int]:
    diff = (pd.to_numeric(left, errors="coerce") - pd.to_numeric(right, errors="coerce")).replace([np.inf, -np.inf], np.nan)
    max_abs = float(diff.abs().max(skipna=True)) if diff.notna().any() else 0.0
    mismatches = int((diff.abs() > tolerance).fillna(False).sum())
    return max_abs, mismatches


def run_formula_spot_checks(panel: pd.DataFrame, daily_by_ticker: dict[str, pd.DataFrame]) -> tuple[bool, dict[str, dict[str, Any]]]:
    """Recompute selected formulas from daily OHLCV and compare with panel."""
    results: dict[str, dict[str, Any]] = {}
    formula_ok = True
    for ticker, daily in daily_by_ticker.items():
        ticker_panel = panel[panel["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        expected = _expected_alpha_series(daily.sort_values("date").reset_index(drop=True))
        merged = ticker_panel[["date", *[config.ALPHA_COLUMN_BY_ID[i] for i in [32, 37, 38, 46, 55, 62, 67]]]].merge(
            expected, on="date", how="left", suffixes=("_panel", "_expected")
        )
        for alpha_id in [32, 37, 38, 46, 55, 62, 67]:
            col = config.ALPHA_COLUMN_BY_ID[alpha_id]
            max_abs, mismatches = _compare_series(merged[f"{col}_panel"], merged[f"{col}_expected"])
            key = f"alpha_{alpha_id}"
            current = results.setdefault(key, {"max_abs_diff": 0.0, "mismatched_rows": 0, "pass": True})
            current["max_abs_diff"] = max(current["max_abs_diff"], max_abs)
            current["mismatched_rows"] += mismatches
            if mismatches:
                current["pass"] = False
                formula_ok = False

    # Range checks requested as formula QA items.
    mfi_col = config.ALPHA_COLUMN_BY_ID[72]
    cmf_col = config.ALPHA_COLUMN_BY_ID[73]
    amihud_col = config.ALPHA_COLUMN_BY_ID[67]
    alpha67 = pd.to_numeric(panel[amihud_col], errors="coerce")
    alpha72 = pd.to_numeric(panel[mfi_col], errors="coerce")
    alpha73 = pd.to_numeric(panel[cmf_col], errors="coerce")
    alpha67_ok = not np.isinf(alpha67.to_numpy(dtype=float)).any() and bool((alpha67.dropna() >= -1e-18).all())
    alpha72_ok = bool(alpha72.dropna().between(0, 100).all())
    alpha73_ok = bool(alpha73.dropna().between(-1, 1).all())
    results["alpha_67_no_inf_nonnegative"] = {"pass": alpha67_ok}
    results["alpha_72_mfi_range"] = {"pass": alpha72_ok}
    results["alpha_73_cmf_range"] = {"pass": alpha73_ok, "extreme_rows": int((~alpha73.dropna().between(-1, 1)).sum())}
    formula_ok = formula_ok and alpha67_ok and alpha72_ok and alpha73_ok
    return formula_ok, results


def run_leakage_spot_checks(panel: pd.DataFrame, daily_by_ticker: dict[str, pd.DataFrame]) -> tuple[bool, list[str]]:
    """Check selected alphas by recomputing from truncated histories only."""
    rng = np.random.default_rng(42)
    failures: list[str] = []
    tickers = sorted(daily_by_ticker)[: min(5, len(daily_by_ticker))]
    for ticker in tickers:
        daily = daily_by_ticker[ticker].sort_values("date").reset_index(drop=True)
        if len(daily) < config.MIN_DAILY_ROWS:
            continue
        candidate_indices = np.arange(252, len(daily))
        if candidate_indices.size == 0:
            continue
        sample_size = min(3, candidate_indices.size)
        sampled_indices = sorted(rng.choice(candidate_indices, size=sample_size, replace=False).tolist())
        full_ticker_panel = panel[panel["ticker"] == ticker].set_index("date")
        for idx in sampled_indices:
            truncated = daily.iloc[: idx + 1].copy()
            recomputed = compute_alpha_features(truncated).iloc[-1]
            date = str(daily.loc[idx, "date"])
            if date not in full_ticker_panel.index:
                failures.append(f"{ticker} {date}: missing date in full panel")
                continue
            full_row = full_ticker_panel.loc[date]
            for alpha_id in LEAKAGE_ALPHA_IDS:
                col = config.ALPHA_COLUMN_BY_ID[alpha_id]
                full_value = full_row[col]
                truncated_value = recomputed[col]
                if pd.isna(full_value) and pd.isna(truncated_value):
                    continue
                if pd.isna(full_value) != pd.isna(truncated_value):
                    failures.append(f"{ticker} {date} alpha {alpha_id}: NaN mismatch")
                    continue
                if abs(float(full_value) - float(truncated_value)) > 1e-10:
                    failures.append(f"{ticker} {date} alpha {alpha_id}: future-leakage spot check mismatch")
    return len(failures) == 0, failures


def run_batch_qa(panel: pd.DataFrame, daily_by_ticker: dict[str, pd.DataFrame]) -> BatchQAResult:
    """Run required QA checks before writing a batch panel."""
    failures: list[str] = []
    alpha_cols = _alpha_columns(panel)
    missing_alpha_cols = [col for col in config.ALPHA_COLUMNS if col not in panel.columns]
    extra_alpha_cols = [col for col in alpha_cols if col not in config.ALPHA_COLUMNS]
    duplicate_count = int(panel.duplicated(["date", "ticker"]).sum()) if not panel.empty else 0
    inf_count = _inf_count(panel[[col for col in config.ALPHA_COLUMNS if col in panel.columns]]) if not panel.empty else 0

    if missing_alpha_cols:
        failures.append(f"missing alpha columns: {missing_alpha_cols}")
    if extra_alpha_cols:
        failures.append(f"unexpected alpha columns: {extra_alpha_cols}")
    if duplicate_count:
        failures.append(f"duplicate date-ticker rows: {duplicate_count}")
    if inf_count:
        failures.append(f"inf/-inf alpha values: {inf_count}")

    formula_pass, formula_results = run_formula_spot_checks(panel, daily_by_ticker) if not panel.empty else (False, {})
    leakage_pass, leakage_failures = run_leakage_spot_checks(panel, daily_by_ticker) if not panel.empty else (False, [])
    if not formula_pass:
        failures.append("formula spot checks failed")
    if not leakage_pass:
        failures.extend(leakage_failures or ["leakage spot checks failed"])

    passed = len(failures) == 0
    return BatchQAResult(
        passed=passed,
        failures=failures,
        duplicate_date_ticker_count=duplicate_count,
        alpha_column_count=len(alpha_cols),
        inf_count=inf_count,
        missing_rates=_missing_rates(panel),
        formula_results=formula_results,
        formula_pass=formula_pass,
        leakage_pass=leakage_pass,
    )


def run_merge_qa(panel: pd.DataFrame) -> MergeQAResult:
    """Run QA checks for the final merged panel."""
    failures: list[str] = []
    alpha_cols = _alpha_columns(panel)
    missing_cols = [col for col in config.OUTPUT_COLUMNS if col not in panel.columns]
    duplicate_count = int(panel.duplicated(["date", "ticker"]).sum()) if not panel.empty else 0
    inf_count = _inf_count(panel[[col for col in config.ALPHA_COLUMNS if col in panel.columns]]) if not panel.empty else 0
    if missing_cols:
        failures.append(f"missing expected columns: {missing_cols}")
    if duplicate_count:
        failures.append(f"duplicate date-ticker rows: {duplicate_count}")
    if len(alpha_cols) != 50 or set(alpha_cols) != set(config.ALPHA_COLUMNS):
        failures.append("alpha IDs 26-75 are not present exactly once")
    if inf_count:
        failures.append(f"inf/-inf alpha values: {inf_count}")

    lowest = pd.DataFrame()
    if not panel.empty and all(col in panel.columns for col in config.ALPHA_COLUMNS):
        coverage = panel.groupby("ticker")[config.ALPHA_COLUMNS].apply(lambda frame: frame.notna().mean().mean())
        lowest = coverage.sort_values().head(10).reset_index(name="mean_alpha_coverage")

    return MergeQAResult(
        passed=len(failures) == 0,
        failures=failures,
        duplicate_date_ticker_count=duplicate_count,
        alpha_column_count=len(alpha_cols),
        inf_count=inf_count,
        missing_rates=_missing_rates(panel),
        lowest_coverage_tickers=lowest,
    )


def print_missing_rates(missing_rates: dict[str, float]) -> None:
    for col in config.ALPHA_COLUMNS:
        value = missing_rates.get(col, float("nan"))
        print(f"  {col}: {value:.4f}")


def print_batch_qa(
    batch_id: str,
    input_dir: Path,
    files_found: int,
    tickers_processed: int,
    skipped: dict[str, str],
    ticker_input_stats: dict[str, dict[str, Any]],
    output_path: Path,
    panel: pd.DataFrame,
    qa: BatchQAResult,
    wrote_output: bool,
) -> None:
    """Print exactly the compact batch QA requested by the user."""
    print(f"batch_id: {batch_id}")
    print(f"input_dir: {input_dir}")
    print(f"number of files found: {files_found}")
    print(f"number of tickers processed: {tickers_processed}")
    print(f"number of tickers skipped: {len(skipped)}")
    print("skipped ticker reasons:")
    if skipped:
        for ticker, reason in skipped.items():
            print(f"  {ticker}: {reason}")
    else:
        print("  none")
    print("ticker input summary:")
    if ticker_input_stats:
        for ticker in sorted(ticker_input_stats):
            stats = ticker_input_stats[ticker]
            duplicate_timestamp_count = stats.get("duplicate_timestamp_count")
            duplicate_timestamp_text = "N/A" if duplicate_timestamp_count is None else str(duplicate_timestamp_count)
            print(
                f"  {ticker}: input_type={stats.get('input_type')} "
                f"files={stats.get('file_count')} "
                f"duplicate_timestamp_count={duplicate_timestamp_text} "
                f"duplicate_daily_date_count={stats.get('duplicate_daily_date_count')} "
                f"daily_row_count={stats.get('daily_row_count')}"
            )
    else:
        print("  none")
    print(f"output path: {output_path if wrote_output else str(output_path) + ' (not written)'}")
    print(f"output shape: {panel.shape}")
    if not panel.empty:
        print(f"date range: {panel['date'].min()} to {panel['date'].max()}")
    else:
        print("date range: N/A")
    print(f"duplicate date-ticker row count: {qa.duplicate_date_ticker_count}")
    print(f"alpha column count: {qa.alpha_column_count}")
    print(f"inf / -inf count: {qa.inf_count}")
    print("missing rate by alpha:")
    print_missing_rates(qa.missing_rates)
    print(f"formula spot-check PASS/FAIL: {'PASS' if qa.formula_pass else 'FAIL'}")
    if qa.formula_results:
        for name, detail in qa.formula_results.items():
            print(f"  {name}: {detail}")
    print(f"leakage / lag spot-check PASS/FAIL: {'PASS' if qa.leakage_pass else 'FAIL'}")
    print(f"final batch status PASS/FAIL: {'PASS' if qa.passed else 'FAIL'}")
    if qa.failures:
        print("failed checks:")
        for failure in qa.failures:
            print(f"  {failure}")


def print_merge_qa(
    batch_files_merged: int,
    output_path: Path,
    panel: pd.DataFrame,
    qa: MergeQAResult,
    wrote_output: bool,
    csv_output_path: Path | None = None,
) -> None:
    """Print compact merge QA requested by the user."""
    print(f"number of batch files merged: {batch_files_merged}")
    print(f"number of unique tickers: {panel['ticker'].nunique() if not panel.empty else 0}")
    print(f"final row count: {len(panel)}")
    if not panel.empty:
        print(f"date range: {panel['date'].min()} to {panel['date'].max()}")
    else:
        print("date range: N/A")
    print(f"duplicate date-ticker row count: {qa.duplicate_date_ticker_count}")
    print(f"alpha column count: {qa.alpha_column_count}")
    print(f"inf / -inf count: {qa.inf_count}")
    print("missing rate by alpha:")
    print_missing_rates(qa.missing_rates)
    print("tickers with lowest coverage:")
    if qa.lowest_coverage_tickers.empty:
        print("  none")
    else:
        for _, row in qa.lowest_coverage_tickers.iterrows():
            print(f"  {row['ticker']}: {row['mean_alpha_coverage']:.4f}")
    print(f"output path: {output_path if wrote_output else str(output_path) + ' (not written)'}")
    if csv_output_path is not None:
        print(f"csv output path: {csv_output_path if wrote_output else str(csv_output_path) + ' (not written)'}")
    print(f"final merge status PASS/FAIL: {'PASS' if qa.passed else 'FAIL'}")
    if qa.failures:
        print("failed checks:")
        for failure in qa.failures:
            print(f"  {failure}")
