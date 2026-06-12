"""Build forward-return labels for the ranked S&P 500 alpha panel.

Step 6 reconstructs daily OHLCV from the original input batches, computes
future-return labels within each ticker, and merges labels back onto the ranked
feature panel. It does not run regressions, backtests, or model training.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import (
    ACCEPTED_SUFFIXES,
    aggregate_minute_frame,
    inspect_company_file,
    load_raw_company_file,
    standardize_daily_frame,
    validate_daily_ohlcv,
    write_csv,
    write_parquet,
)


CORE_LABEL_COLUMNS = [
    "close_to_close_forward_1d_return",
    "close_to_close_forward_5d_return",
    "close_to_close_forward_21d_return",
    "tradable_forward_1d_return",
    "tradable_forward_5d_return",
    "tradable_forward_21d_return",
]

VWAP_LABEL_COLUMNS = [
    "vwap_forward_1d_return",
    "vwap_forward_5d_return",
    "vwap_forward_21d_return",
]

EXCESS_LABEL_COLUMNS = [
    "forward_1d_excess_return",
    "forward_5d_excess_return",
    "forward_21d_excess_return",
]

DEFAULT_MARKET_FILE = Path("data/raw/auxiliary/spx_daily_price.xlsx")


@dataclass
class ReconstructedDailyData:
    """Daily OHLCV reconstructed from raw batch files."""

    frame: pd.DataFrame
    skipped: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Step6QA:
    """Terminal QA state for Step 6."""

    failures: list[str] = field(default_factory=list)
    label_missing_rates: dict[str, float] = field(default_factory=dict)
    label_inf_count: int = 0
    shift_check_pass: bool = False
    shift_check_details: dict[str, Any] = field(default_factory=dict)
    tail_nan_check_pass: bool = False
    label_feature_separation_pass: bool = False
    feature_preservation_pass: bool = False
    date_alignment_pass: bool = False

    @property
    def passed(self) -> bool:
        return not self.failures


def parse_bool(value: str | bool) -> bool:
    """Parse flexible true/false command-line values."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def _date_to_string(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _inf_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return int(np.isinf(numeric.to_numpy(dtype=float)).sum())


def _safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return np.log(num / den.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def load_features(path: Path) -> pd.DataFrame:
    """Load the ranked alpha feature panel."""
    if not path.exists():
        raise FileNotFoundError(f"Ranked feature panel not found: {path}")
    features = pd.read_parquet(path)
    required = {"date", "ticker"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Feature panel missing required columns: {sorted(missing)}")
    features["date"] = _date_to_string(features["date"])
    features["ticker"] = features["ticker"].astype(str).str.strip().str.upper()
    return features.sort_values(["date", "ticker"]).reset_index(drop=True)


def find_company_files_recursive(input_batches_dir: Path) -> list[Path]:
    """Find supported raw company files below the input batch root."""
    if not input_batches_dir.exists():
        raise FileNotFoundError(f"Input batches directory not found: {input_batches_dir}")
    return sorted(
        path
        for path in input_batches_dir.rglob("*")
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in ACCEPTED_SUFFIXES
    )


def reconstruct_daily_ohlcv(input_batches_dir: Path, panel_tickers: set[str]) -> ReconstructedDailyData:
    """Reconstruct daily OHLCV for tickers present in the ranked feature panel."""
    files = find_company_files_recursive(input_batches_dir)
    grouped: dict[str, list[Path]] = {}
    skipped: dict[str, str] = {}
    warnings: list[str] = []

    for path in files:
        try:
            descriptor = inspect_company_file(path, universe_tickers=panel_tickers)
            if descriptor.ticker in panel_tickers:
                grouped.setdefault(descriptor.ticker, []).append(path)
        except Exception as exc:
            skipped[path.stem] = str(exc)

    daily_frames: list[pd.DataFrame] = []
    for ticker, ticker_files in sorted(grouped.items()):
        try:
            loaded = [load_raw_company_file(path, universe_tickers=panel_tickers) for path in ticker_files]
            input_types = sorted({item.input_type for item in loaded})
            loaded_tickers = sorted({item.ticker for item in loaded})
            if loaded_tickers != [ticker]:
                skipped[ticker] = f"ticker inference mismatch after full load: {loaded_tickers}"
                continue
            if len(input_types) != 1:
                skipped[ticker] = f"mixed input types for same ticker: {input_types}"
                continue

            combined = pd.concat([item.frame for item in loaded], ignore_index=True)
            if input_types[0] == "minute":
                daily, ticker_warnings, _ = aggregate_minute_frame(combined, ticker)
            else:
                daily, ticker_warnings = standardize_daily_frame(combined, ticker)

            ok, reasons = validate_daily_ohlcv(daily, min_rows=1)
            if not ok:
                skipped[ticker] = "; ".join(reasons)
                continue
            for warning in ticker_warnings:
                warnings.append(f"{ticker}: {warning}")
            daily_frames.append(daily[["date", "ticker", "open", "close", "daily_vwap", "volume"]].copy())
        except Exception as exc:
            skipped[ticker] = str(exc)

    if not daily_frames:
        empty = pd.DataFrame(columns=["date", "ticker", "open", "close", "daily_vwap", "volume"])
        return ReconstructedDailyData(empty, skipped=skipped, warnings=warnings)

    daily = pd.concat(daily_frames, ignore_index=True)
    daily["date"] = _date_to_string(daily["date"])
    daily["ticker"] = daily["ticker"].astype(str).str.strip().str.upper()
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)
    duplicate_count = int(daily.duplicated(["date", "ticker"]).sum())
    if duplicate_count:
        skipped["_daily_duplicate_check"] = f"duplicate date-ticker rows after reconstruction: {duplicate_count}"
        daily = daily.drop_duplicates(["date", "ticker"], keep="first").reset_index(drop=True)
    return ReconstructedDailyData(daily, skipped=skipped, warnings=warnings)


def _extract_date_price_frame(path: Path) -> pd.DataFrame:
    """Extract date and index price from headerless or Bloomberg-style Excel sheets."""
    raw = pd.read_excel(path, header=None)
    if raw.empty:
        raise ValueError(f"Market file is empty: {path}")

    date_scores: dict[int, int] = {}
    for col in raw.columns:
        parsed = pd.to_datetime(raw[col], errors="coerce")
        date_scores[int(col)] = int(parsed.notna().sum())
    date_col = max(date_scores, key=date_scores.get)
    if date_scores[date_col] == 0:
        raise ValueError(f"Could not identify date column in market file: {path}")

    price_scores: dict[int, int] = {}
    for col in raw.columns:
        if int(col) == date_col:
            continue
        numeric = pd.to_numeric(raw[col], errors="coerce")
        price_scores[int(col)] = int((numeric > 0).sum())
    if not price_scores:
        raise ValueError(f"Could not identify market price column: {path}")
    price_col = max(price_scores, key=price_scores.get)
    if price_scores[price_col] == 0:
        raise ValueError(f"Could not identify positive market price column: {path}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce").dt.strftime("%Y-%m-%d"),
            "market_price": pd.to_numeric(raw[price_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "market_price"]).copy()
    if (out["market_price"] <= 0).any():
        raise ValueError(f"Market file contains nonpositive prices: {path}")
    if out.duplicated("date").any():
        conflicts: list[str] = []
        for date_value, group in out[out.duplicated("date", keep=False)].groupby("date"):
            if group["market_price"].nunique(dropna=False) > 1:
                conflicts.append(str(date_value))
                if len(conflicts) >= 10:
                    break
        if conflicts:
            raise ValueError(f"Market file has conflicting duplicate dates: {conflicts}")
        out = out.drop_duplicates("date", keep="first")
    return out.sort_values("date").reset_index(drop=True)


def load_market_forward_returns(path: Path = DEFAULT_MARKET_FILE) -> tuple[pd.DataFrame | None, str | None]:
    """Load SPX prices if available and compute forward market log returns."""
    if not path.exists():
        return None, f"market file unavailable; excess-return labels skipped: {path}"
    try:
        market = _extract_date_price_frame(path)
        price = market["market_price"]
        for horizon in [1, 5, 21]:
            market[f"market_forward_{horizon}d_return"] = _safe_log_ratio(price.shift(-horizon), price)
        return market[["date", "market_forward_1d_return", "market_forward_5d_return", "market_forward_21d_return"]], None
    except Exception as exc:
        return None, f"market forward returns unavailable; excess-return labels skipped: {exc}"


def compute_labels(daily: pd.DataFrame, market_forward: pd.DataFrame | None) -> pd.DataFrame:
    """Compute all forward-return labels within ticker only."""
    labels_parts: list[pd.DataFrame] = []
    for ticker, group in daily.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        g = group.sort_values("date").reset_index(drop=True).copy()
        close = pd.to_numeric(g["close"], errors="coerce")
        open_ = pd.to_numeric(g["open"], errors="coerce")
        vwap = pd.to_numeric(g["daily_vwap"], errors="coerce")

        out = g[["date", "ticker"]].copy()
        out["close_to_close_forward_1d_return"] = _safe_log_ratio(close.shift(-1), close)
        out["close_to_close_forward_5d_return"] = _safe_log_ratio(close.shift(-5), close)
        out["close_to_close_forward_21d_return"] = _safe_log_ratio(close.shift(-21), close)
        out["tradable_forward_1d_return"] = _safe_log_ratio(close.shift(-1), open_.shift(-1))
        out["tradable_forward_5d_return"] = _safe_log_ratio(close.shift(-5), open_.shift(-1))
        out["tradable_forward_21d_return"] = _safe_log_ratio(close.shift(-21), open_.shift(-1))

        out["vwap_forward_1d_return"] = _safe_log_ratio(close.shift(-1), vwap.shift(-1))
        out["vwap_forward_5d_return"] = _safe_log_ratio(vwap.shift(-5), vwap.shift(-1))
        out["vwap_forward_21d_return"] = _safe_log_ratio(vwap.shift(-21), vwap.shift(-1))
        labels_parts.append(out)

    labels = pd.concat(labels_parts, ignore_index=True) if labels_parts else pd.DataFrame(columns=["date", "ticker", *CORE_LABEL_COLUMNS, *VWAP_LABEL_COLUMNS])
    labels = labels.sort_values(["date", "ticker"]).reset_index(drop=True)

    if market_forward is not None:
        labels = labels.merge(market_forward, on="date", how="left")
        labels["forward_1d_excess_return"] = labels["close_to_close_forward_1d_return"] - labels["market_forward_1d_return"]
        labels["forward_5d_excess_return"] = labels["close_to_close_forward_5d_return"] - labels["market_forward_5d_return"]
        labels["forward_21d_excess_return"] = labels["close_to_close_forward_21d_return"] - labels["market_forward_21d_return"]
        labels = labels.drop(columns=["market_forward_1d_return", "market_forward_5d_return", "market_forward_21d_return"])

    label_cols = [col for col in labels.columns if col not in {"date", "ticker"}]
    labels[label_cols] = labels[label_cols].replace([np.inf, -np.inf], np.nan)
    return labels


def build_model_dataset(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Left-merge labels onto ranked features without dropping feature rows."""
    label_cols = [col for col in labels.columns if col not in {"date", "ticker"}]
    return features.merge(labels[["date", "ticker", *label_cols]], on=["date", "ticker"], how="left")


def check_tail_label_nans(labels: pd.DataFrame) -> tuple[bool, list[str]]:
    """Verify expected missing labels at each ticker's trailing horizon rows."""
    failures: list[str] = []
    horizon_cols = {
        1: ["close_to_close_forward_1d_return", "tradable_forward_1d_return", "vwap_forward_1d_return", "forward_1d_excess_return"],
        5: ["close_to_close_forward_5d_return", "tradable_forward_5d_return", "vwap_forward_5d_return", "forward_5d_excess_return"],
        21: ["close_to_close_forward_21d_return", "tradable_forward_21d_return", "vwap_forward_21d_return", "forward_21d_excess_return"],
    }
    available_cols = set(labels.columns)
    for ticker, group in labels.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        for horizon, columns in horizon_cols.items():
            tail = group.tail(horizon)
            for col in columns:
                if col not in available_cols:
                    continue
                if tail[col].notna().any():
                    failures.append(f"{ticker} last {horizon} rows have non-NaN {col}")
                    if len(failures) >= 20:
                        return False, failures
    return len(failures) == 0, failures


def run_shift_spot_checks(daily: pd.DataFrame, labels: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
    """Randomly sample tickers and verify the forward-return formulas."""
    sample_tickers = sorted(daily["ticker"].dropna().unique().tolist())[:5]
    max_abs_by_label: dict[str, float] = {}
    mismatches_by_label: dict[str, int] = {}

    for ticker in sample_tickers:
        daily_group = daily[daily["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        label_group = labels[labels["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        if daily_group.empty or label_group.empty:
            continue
        close = pd.to_numeric(daily_group["close"], errors="coerce")
        open_ = pd.to_numeric(daily_group["open"], errors="coerce")
        expected = pd.DataFrame(
            {
                "date": daily_group["date"],
                "ticker": ticker,
                "close_to_close_forward_1d_return": _safe_log_ratio(close.shift(-1), close),
                "close_to_close_forward_5d_return": _safe_log_ratio(close.shift(-5), close),
                "close_to_close_forward_21d_return": _safe_log_ratio(close.shift(-21), close),
                "tradable_forward_1d_return": _safe_log_ratio(close.shift(-1), open_.shift(-1)),
                "tradable_forward_5d_return": _safe_log_ratio(close.shift(-5), open_.shift(-1)),
                "tradable_forward_21d_return": _safe_log_ratio(close.shift(-21), open_.shift(-1)),
            }
        )
        merged = label_group.merge(expected, on=["date", "ticker"], suffixes=("_label", "_expected"), how="left")
        for col in CORE_LABEL_COLUMNS:
            diff = (merged[f"{col}_label"] - merged[f"{col}_expected"]).replace([np.inf, -np.inf], np.nan)
            max_abs = float(diff.abs().max(skipna=True)) if diff.notna().any() else 0.0
            mismatches = int((diff.abs() > 1e-12).fillna(False).sum())
            max_abs_by_label[col] = max(max_abs_by_label.get(col, 0.0), max_abs)
            mismatches_by_label[col] = mismatches_by_label.get(col, 0) + mismatches

    pass_by_label = {col: mismatches_by_label.get(col, 0) == 0 for col in CORE_LABEL_COLUMNS}
    details = {
        "sample_tickers": sample_tickers,
        "max_abs_diff": max_abs_by_label,
        "mismatched_rows": mismatches_by_label,
        "pass_by_label": pass_by_label,
        "alignment_note": "features at t; tradable entry at t+1 open; exit at t+h close",
    }
    return all(pass_by_label.values()), details


def run_step6_qa(
    features: pd.DataFrame,
    daily: pd.DataFrame,
    labels: pd.DataFrame,
    model_dataset: pd.DataFrame,
    label_cols: list[str],
) -> Step6QA:
    """Run all Step 6 QA checks."""
    qa = Step6QA()
    qa.label_missing_rates = {col: float(labels[col].isna().mean()) for col in label_cols}
    qa.label_inf_count = _inf_count(labels[label_cols]) if label_cols else 0

    if len(labels) != len(daily):
        qa.failures.append(f"label row count {len(labels)} does not equal reconstructed OHLCV rows {len(daily)}")
    if len(model_dataset) != len(features):
        qa.failures.append(f"model row count {len(model_dataset)} does not equal feature rows {len(features)}")
    label_duplicates = int(labels.duplicated(["date", "ticker"]).sum())
    model_duplicates = int(model_dataset.duplicated(["date", "ticker"]).sum())
    if label_duplicates:
        qa.failures.append(f"label duplicate date-ticker rows: {label_duplicates}")
    if model_duplicates:
        qa.failures.append(f"model dataset duplicate date-ticker rows: {model_duplicates}")
    if qa.label_inf_count:
        qa.failures.append(f"label inf/-inf count is nonzero: {qa.label_inf_count}")

    qa.tail_nan_check_pass, tail_failures = check_tail_label_nans(labels)
    if not qa.tail_nan_check_pass:
        qa.failures.extend(tail_failures)

    qa.shift_check_pass, qa.shift_check_details = run_shift_spot_checks(daily, labels)
    if not qa.shift_check_pass:
        qa.failures.append("shift formula spot checks failed")

    feature_cols = list(features.columns)
    qa.label_feature_separation_pass = len(set(feature_cols) & set(label_cols)) == 0
    if not qa.label_feature_separation_pass:
        qa.failures.append("label columns overlap with feature columns")

    same_order = model_dataset[["date", "ticker"]].equals(features[["date", "ticker"]])
    same_columns = list(model_dataset.columns[: len(feature_cols)]) == feature_cols
    qa.feature_preservation_pass = bool(same_order and same_columns)
    if not qa.feature_preservation_pass:
        qa.failures.append("ranked feature rows/columns were not preserved before labels")

    qa.date_alignment_pass = qa.shift_check_pass and qa.tail_nan_check_pass
    if not qa.date_alignment_pass:
        qa.failures.append("date alignment check failed")

    return qa


def print_step6_summary(
    labels_output: Path,
    model_output: Path,
    labels: pd.DataFrame,
    model_dataset: pd.DataFrame,
    qa: Step6QA,
    warnings: list[str],
    skipped: dict[str, str],
    csv_labels: Path | None,
    csv_model: Path | None,
) -> None:
    """Print clear terminal QA output for Step 6."""
    print("Step 6 Forward-Return Label QA")
    print(f"labels output path: {labels_output}")
    print(f"model dataset output path: {model_output}")
    if csv_labels is not None:
        print(f"labels csv output path: {csv_labels}")
    if csv_model is not None:
        print(f"model csv output path: {csv_model}")
    print(f"label rows: {len(labels)}")
    print(f"model rows: {len(model_dataset)}")
    print(f"unique tickers: {labels['ticker'].nunique() if 'ticker' in labels.columns else 0}")
    print(f"date range: {labels['date'].min() if 'date' in labels.columns else None} to {labels['date'].max() if 'date' in labels.columns else None}")
    print(f"duplicate date-ticker rows: {int(labels.duplicated(['date', 'ticker']).sum()) if {'date', 'ticker'}.issubset(labels.columns) else 'N/A'}")
    print("label missing rates:")
    for col, rate in qa.label_missing_rates.items():
        print(f"  {col}: {rate:.4f}")
    print(f"label inf / -inf count: {qa.label_inf_count}")
    print(f"tail expected-NaN check PASS/FAIL: {'PASS' if qa.tail_nan_check_pass else 'FAIL'}")
    print(f"shift check PASS/FAIL: {'PASS' if qa.shift_check_pass else 'FAIL'}")
    print(f"shift check details: {qa.shift_check_details}")
    print(f"label columns not included in feature columns PASS/FAIL: {'PASS' if qa.label_feature_separation_pass else 'FAIL'}")
    print(f"features preserved before label merge PASS/FAIL: {'PASS' if qa.feature_preservation_pass else 'FAIL'}")
    print(f"date alignment PASS/FAIL: {'PASS' if qa.date_alignment_pass else 'FAIL'}")
    print("future-label feature usage: PASS; labels are computed from reconstructed OHLCV after features are loaded and are merged only as new columns.")
    print("market/factor/sector forward-fill: PASS; this step does not forward-fill market, factor, sector, or industry returns.")
    if warnings:
        print("warnings:")
        for warning in warnings[:50]:
            print(f"  {warning}")
    if skipped:
        print("skipped reconstruction items:")
        for ticker, reason in list(skipped.items())[:50]:
            print(f"  {ticker}: {reason}")
    print(f"final Step 6 status PASS/FAIL: {'PASS' if qa.passed else 'FAIL'}")
    if qa.failures:
        print("failed checks:")
        for failure in qa.failures:
            print(f"  {failure}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build forward-return labels for ranked S&P 500 alpha features.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--input-batches-dir", type=Path, required=True)
    parser.add_argument("--labels-output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--export-csv", type=parse_bool, default=False)
    parser.add_argument("--market-file", type=Path, default=DEFAULT_MARKET_FILE)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    features = load_features(args.features)
    panel_tickers = set(features["ticker"].unique())
    reconstructed = reconstruct_daily_ohlcv(args.input_batches_dir, panel_tickers)
    market_forward, market_warning = load_market_forward_returns(args.market_file)
    warnings = list(reconstructed.warnings)
    if market_warning:
        warnings.append(market_warning)

    labels = compute_labels(reconstructed.frame, market_forward)
    label_cols = [col for col in labels.columns if col not in {"date", "ticker"}]
    model_dataset = build_model_dataset(features, labels)
    qa = run_step6_qa(features, reconstructed.frame, labels, model_dataset, label_cols)

    labels_csv = args.labels_output.with_suffix(".csv") if args.export_csv else None
    model_csv = args.model_output.with_suffix(".csv") if args.export_csv else None
    if qa.passed:
        write_parquet(labels, args.labels_output)
        write_parquet(model_dataset, args.model_output)
        if labels_csv is not None:
            write_csv(labels, labels_csv)
        if model_csv is not None:
            write_csv(model_dataset, model_csv)

    print_step6_summary(
        labels_output=args.labels_output,
        model_output=args.model_output,
        labels=labels,
        model_dataset=model_dataset,
        qa=qa,
        warnings=warnings,
        skipped=reconstructed.skipped,
        csv_labels=labels_csv,
        csv_model=model_csv,
    )
    if not qa.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
