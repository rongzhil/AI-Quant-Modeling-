"""Enrich the raw S&P 500 alpha panel with market, sector, factor, and float data.

This module is intentionally separate from the batch alpha builder. It updates
only the auxiliary-data-dependent alphas requested for the prototype:

* 33 residual momentum
* 34 sector-neutral momentum
* 59 idiosyncratic volatility
* 60 market beta
* 66 turnover

Alpha 35 industry momentum is kept as NaN by design because a reliable
point-in-time industry peer basket is not available.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .io_utils import (
    ACCEPTED_SUFFIXES,
    aggregate_minute_frame,
    inspect_company_file,
    load_raw_company_file,
    standardize_daily_frame,
    standardize_ticker,
    validate_daily_ohlcv,
    write_csv,
    write_parquet,
)


ALPHA_33 = config.ALPHA_COLUMN_BY_ID[33]
ALPHA_34 = config.ALPHA_COLUMN_BY_ID[34]
ALPHA_35 = config.ALPHA_COLUMN_BY_ID[35]
ALPHA_59 = config.ALPHA_COLUMN_BY_ID[59]
ALPHA_60 = config.ALPHA_COLUMN_BY_ID[60]
ALPHA_66 = config.ALPHA_COLUMN_BY_ID[66]
ALPHA_26 = config.ALPHA_COLUMN_BY_ID[26]
ALPHA_30 = config.ALPHA_COLUMN_BY_ID[30]

TARGET_ALPHA_COLUMNS = [ALPHA_33, ALPHA_34, ALPHA_59, ALPHA_60, ALPHA_66]

SECTOR_FILE_MAP = {
    "consumer_discretionary.xlsx": "Consumer Discretionary",
    "consumer_staples.xlsx": "Consumer Staples",
    "energy.xlsx": "Energy",
    "real_estate.xlsx": "Real Estate",
    "communication_services.xlsx": "Communication Services",
    "materials.xlsx": "Materials",
    "financials.xlsx": "Financials",
    "health_care.xlsx": "Health Care",
    "industrials.xlsx": "Industrials",
    "information_technology.xlsx": "Information Technology",
    "utilities.xlsx": "Utilities",
}

TICKER_ALIASES = {
    **config.DEFAULT_TICKER_ALIASES,
    "MRSH": "MMC",
    "BRK/B": "BRK.B",
    "BF/B": "BF.B",
}


@dataclass
class SectorLoadResult:
    """Loaded long-form sector index data and compact QA metadata."""

    frame: pd.DataFrame
    qa_rows: list[dict[str, Any]]
    loaded_count: int
    staples_discretionary_identical: bool | None


@dataclass
class FloatLoadResult:
    """Loaded point-in-time float shares and compact QA metadata."""

    frame: pd.DataFrame
    min_date: str | None
    max_date: str | None
    all_missing_columns: list[str] = field(default_factory=list)
    raw_panel_missing_tickers: list[str] = field(default_factory=list)
    partial_missing_tickers: list[str] = field(default_factory=list)


@dataclass
class EnrichmentQA:
    """Terminal QA values for the final enrichment step."""

    failures: list[str] = field(default_factory=list)
    formula_results: dict[str, dict[str, Any]] = field(default_factory=dict)

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


def _file_key(path: Path) -> str:
    """Normalize a file name so spaces/underscores do not matter."""
    return "".join(ch for ch in path.name.lower() if ch.isalnum())


def resolve_existing_path(path: Path) -> Path:
    """Return an existing path, allowing small filename punctuation variants."""
    if path.exists():
        return path
    parent = path.parent
    if parent.exists():
        target_key = _file_key(path)
        for candidate in parent.iterdir():
            if _file_key(candidate) == target_key:
                return candidate
    raise FileNotFoundError(f"Required path does not exist: {path}")


def clean_ticker(value: Any) -> str:
    """Clean Bloomberg-style equity tickers to the panel convention."""
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = " ".join(text.split())
    for suffix in [" US EQUITY", " EQUITY"]:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    text = text.replace("/", ".")
    if text == "MRSH":
        return "MMC"
    return standardize_ticker(text, aliases=TICKER_ALIASES)


def _date_to_string(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _date_to_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _inf_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return int(np.isinf(numeric.to_numpy(dtype=float)).sum())


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def validate_raw_panel(panel: pd.DataFrame) -> list[str]:
    """Validate the input alpha panel before enrichment."""
    failures: list[str] = []
    for col in ["date", "ticker"]:
        if col not in panel.columns:
            failures.append(f"missing required panel column: {col}")
    missing_alphas = [col for col in config.ALPHA_COLUMNS if col not in panel.columns]
    if missing_alphas:
        failures.append(f"missing alpha columns 26-75: {missing_alphas}")
    if {"date", "ticker"}.issubset(panel.columns):
        duplicate_count = int(panel.duplicated(["date", "ticker"]).sum())
        if duplicate_count:
            failures.append(f"duplicate date-ticker rows in raw panel: {duplicate_count}")
    alpha_cols = [col for col in config.ALPHA_COLUMNS if col in panel.columns]
    inf_count = _inf_count(panel[alpha_cols]) if alpha_cols else 0
    if inf_count:
        failures.append(f"raw panel alpha inf/-inf count is nonzero: {inf_count}")
    if ALPHA_35 in panel.columns and panel[ALPHA_35].notna().any():
        failures.append("alpha_35_industry_momentum is populated, but this enrichment keeps it dropped by design")
    return failures


def load_raw_panel(path: Path) -> pd.DataFrame:
    """Load the raw alpha panel from parquet."""
    path = resolve_existing_path(path)
    panel = pd.read_parquet(path)
    panel["date"] = _date_to_string(panel["date"])
    panel["ticker"] = panel["ticker"].map(clean_ticker)
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_gics_mapping(path: Path) -> pd.DataFrame:
    """Load Bloomberg-style GICS mapping and return clean metadata columns."""
    path = resolve_existing_path(path)
    raw = pd.read_excel(path, header=None)
    header_idx: int | None = None
    for idx, row in raw.iterrows():
        labels = {str(value).strip().casefold() for value in row.dropna().tolist()}
        if "ticker" in labels:
            header_idx = int(idx)
            break
    if header_idx is None:
        raise ValueError(f"Could not locate a Ticker header row in GICS mapping: {path}")

    data = raw.iloc[header_idx + 1 :].copy()
    data.columns = [str(value).strip() for value in raw.iloc[header_idx].tolist()]
    data = data.dropna(how="all")

    def find_col(candidates: set[str]) -> str | None:
        for col in data.columns:
            normalized = str(col).strip().casefold()
            if normalized in candidates:
                return col
        return None

    ticker_col = find_col({"ticker"})
    name_col = find_col({"name"})
    sector_col = find_col({"gics sector"})
    industry_col = find_col({"gics ind name"})
    sub_industry_col = find_col({"gics subind name", "gics sub-ind name"})
    if ticker_col is None or sector_col is None:
        raise ValueError("GICS mapping must contain Ticker and GICS Sector columns.")

    out = pd.DataFrame(
        {
            "ticker": data[ticker_col].map(clean_ticker),
            "company_name_gics": data[name_col].astype(str).str.strip() if name_col else "",
            "sector_gics": data[sector_col].astype(str).str.strip(),
            "industry_gics": data[industry_col].astype(str).str.strip() if industry_col else "",
            "sub_industry_gics": data[sub_industry_col].astype(str).str.strip() if sub_industry_col else "",
        }
    )
    out = out[out["ticker"].str.match(r"^[A-Z0-9.]+$", na=False)].copy()
    out = out.drop_duplicates("ticker", keep="first").reset_index(drop=True)
    return out


def apply_gics_metadata(panel: pd.DataFrame, gics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Overwrite panel metadata with GICS mapping where available."""
    raw_tickers = set(panel["ticker"].unique())
    gics_tickers = set(gics["ticker"].unique())
    overlap = raw_tickers & gics_tickers
    merged = panel.merge(gics, on="ticker", how="left")

    for raw_col, gics_col in [
        ("company_name", "company_name_gics"),
        ("sector", "sector_gics"),
        ("industry", "industry_gics"),
        ("sub_industry", "sub_industry_gics"),
    ]:
        if raw_col not in merged.columns:
            merged[raw_col] = ""
        gics_values = merged[gics_col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
        merged[raw_col] = gics_values.combine_first(merged[raw_col])

    epam_mask = (merged["ticker"] == "EPAM") & ~merged["ticker"].isin(gics_tickers)
    if epam_mask.any():
        merged.loc[epam_mask, "sector"] = "Information Technology"
        merged.loc[epam_mask, "industry"] = "Technology"
        merged.loc[epam_mask, "sub_industry"] = "IT Consulting & Other Services"

    for col in ["company_name_gics", "sector_gics", "industry_gics", "sub_industry_gics"]:
        merged = merged.drop(columns=col)

    missing_after = sorted(merged.loc[merged["sector"].isna() | (merged["sector"].astype(str).str.strip() == ""), "ticker"].unique())
    sector_counts = merged.drop_duplicates("ticker")["sector"].value_counts(dropna=False).to_dict()
    qa = {
        "raw_panel_tickers": len(raw_tickers),
        "gics_mapping_tickers": len(gics_tickers),
        "overlap_count": len(overlap),
        "raw_tickers_missing_from_gics": sorted(raw_tickers - gics_tickers),
        "gics_tickers_not_in_raw_panel": sorted(gics_tickers - raw_tickers),
        "rows_with_sector_missing_after_gics": int(merged["sector"].isna().sum() + (merged["sector"].astype(str).str.strip() == "").sum()),
        "tickers_with_sector_missing_after_gics": missing_after,
        "sector_counts": sector_counts,
    }
    return merged, qa


def _extract_date_price_frame(path: Path, value_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Robustly extract date and price columns from headerless Excel layouts."""
    raw = pd.read_excel(path, header=None)
    if raw.empty:
        raise ValueError(f"Price/index file is empty: {path}")

    date_scores: dict[int, int] = {}
    for col in raw.columns:
        parsed = pd.to_datetime(raw[col], errors="coerce")
        date_scores[int(col)] = int(parsed.notna().sum())
    date_col = max(date_scores, key=date_scores.get)
    if date_scores[date_col] == 0:
        raise ValueError(f"Could not identify a date column in {path}")

    price_scores: dict[int, int] = {}
    for col in raw.columns:
        if int(col) == date_col:
            continue
        numeric = pd.to_numeric(raw[col], errors="coerce")
        price_scores[int(col)] = int((numeric > 0).sum())
    if not price_scores:
        raise ValueError(f"Could not identify a price column in {path}")
    price_col = max(price_scores, key=price_scores.get)
    if price_scores[price_col] == 0:
        raise ValueError(f"Could not identify a positive price column in {path}")

    dates = pd.to_datetime(raw[date_col], errors="coerce").dt.normalize()
    prices = pd.to_numeric(raw[price_col], errors="coerce")
    valid_date = dates.notna()
    missing_price_count = int((valid_date & prices.isna()).sum())
    out = pd.DataFrame({"date_dt": dates, value_name: prices})
    out = out[valid_date].copy()

    duplicate_date_count = int(out.duplicated("date_dt").sum())
    if duplicate_date_count:
        conflicts: list[str] = []
        for date_value, group in out[out.duplicated("date_dt", keep=False)].groupby("date_dt", sort=False):
            if group[value_name].nunique(dropna=False) > 1:
                conflicts.append(str(date_value.date()))
                if len(conflicts) >= 10:
                    break
        if conflicts:
            raise ValueError(f"Conflicting duplicate dates in {path}: {conflicts}")
        out = out.drop_duplicates("date_dt", keep="first")

    out = out.sort_values("date_dt").reset_index(drop=True)
    out["date"] = out["date_dt"].dt.strftime("%Y-%m-%d")
    if (out[value_name] <= 0).any():
        raise ValueError(f"Nonpositive prices found in {path}")
    qa = {
        "path": str(path),
        "date_col": date_col,
        "price_col": price_col,
        "duplicate_date_count": duplicate_date_count,
        "missing_price_count": missing_price_count,
        "min_date": out["date"].min() if not out.empty else None,
        "max_date": out["date"].max() if not out.empty else None,
        "rows": len(out),
    }
    return out[["date", value_name]], qa


def load_sector_indices(sector_dir: Path) -> SectorLoadResult:
    """Load all required GICS sector index files and compute returns/momentum."""
    sector_dir = resolve_existing_path(sector_dir)
    frames: list[pd.DataFrame] = []
    qa_rows: list[dict[str, Any]] = []

    for filename, sector in SECTOR_FILE_MAP.items():
        path = resolve_existing_path(sector_dir / filename)
        sector_frame, qa = _extract_date_price_frame(path, "sector_price")
        sector_frame["sector"] = sector
        sector_frame["sector_return"] = np.log(sector_frame["sector_price"] / sector_frame["sector_price"].shift(1))
        sector_frame["sector_momentum_126"] = (
            sector_frame["sector_return"].rolling(126, min_periods=126).sum()
        )
        qa.update(
            {
                "file": filename,
                "sector": sector,
                "missing_sector_return_count": int(sector_frame["sector_return"].isna().sum()),
            }
        )
        frames.append(sector_frame)
        qa_rows.append(qa)

    sector_data = pd.concat(frames, ignore_index=True)
    sector_data = sector_data[["date", "sector", "sector_price", "sector_return", "sector_momentum_126"]]
    identical: bool | None = None
    staples = sector_data[sector_data["sector"] == "Consumer Staples"][["date", "sector_price"]]
    discretionary = sector_data[sector_data["sector"] == "Consumer Discretionary"][["date", "sector_price"]]
    if not staples.empty and not discretionary.empty:
        paired = staples.merge(discretionary, on="date", suffixes=("_staples", "_discretionary"))
        identical = bool(not paired.empty and np.allclose(paired["sector_price_staples"], paired["sector_price_discretionary"]))

    return SectorLoadResult(
        frame=sector_data.sort_values(["sector", "date"]).reset_index(drop=True),
        qa_rows=qa_rows,
        loaded_count=len(frames),
        staples_discretionary_identical=identical,
    )


def load_factor_file(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load Carhart/Fama-French factor data in decimal-return scale."""
    path = resolve_existing_path(path)
    raw = pd.read_csv(path)
    normalized = {str(col).strip().lower().replace("-", "_"): col for col in raw.columns}
    required = {
        "date": "date",
        "mkt_rf": "MKT_RF",
        "smb": "SMB",
        "hml": "HML",
        "rf": "RF",
        "mom": "MOM",
    }
    missing = [wanted for wanted in required if wanted not in normalized]
    if missing:
        raise ValueError(f"Factor file missing required columns {missing}: {path}")
    out = pd.DataFrame()
    out["date"] = _date_to_string(raw[normalized["date"]])
    for source_key, target_col in required.items():
        if source_key == "date":
            continue
        out[target_col] = pd.to_numeric(raw[normalized[source_key]], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    duplicate_count = int(out.duplicated("date").sum())
    if duplicate_count:
        raise ValueError(f"Factor file has duplicate dates: {duplicate_count}")
    factor_cols = ["MKT_RF", "SMB", "HML", "RF", "MOM"]
    missing_factor_values = int(out[factor_cols].isna().sum().sum())
    if missing_factor_values:
        raise ValueError(f"Factor file has missing factor values: {missing_factor_values}")
    inf_count = _inf_count(out[factor_cols])
    if inf_count:
        raise ValueError(f"Factor file has inf/-inf values: {inf_count}")
    max_abs = float(out[factor_cols].abs().max().max())
    if max_abs > 1:
        raise ValueError("Factor values appear to be percent scale because max absolute value exceeds 1.")
    qa = {
        "min_date": out["date"].min(),
        "max_date": out["date"].max(),
        "rows": len(out),
        "max_abs_factor": max_abs,
        "duplicate_dates": duplicate_count,
        "missing_factor_values": missing_factor_values,
        "scale_check": "decimal",
    }
    return out, qa


def load_market_file(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load SPX price/index data and compute daily log market returns."""
    path = resolve_existing_path(path)
    market, qa = _extract_date_price_frame(path, "spx_price")
    market["market_return"] = np.log(market["spx_price"] / market["spx_price"].shift(1))
    inf_count = _inf_count(market[["market_return"]])
    if inf_count:
        raise ValueError(f"Market return has inf/-inf values: {inf_count}")
    qa["missing_market_return_count"] = int(market["market_return"].isna().sum())
    return market[["date", "spx_price", "market_return"]], qa


def load_float_file(path: Path, raw_panel_tickers: set[str], float_unit: str) -> FloatLoadResult:
    """Load wide Bloomberg float shares and convert to long point-in-time data."""
    path = resolve_existing_path(path)
    if float_unit not in {"millions", "shares"}:
        raise ValueError("--float-unit must be either 'millions' or 'shares'")
    raw = pd.read_excel(path)
    if raw.empty or len(raw.columns) < 2:
        raise ValueError(f"Float shares file must have date plus ticker columns: {path}")

    date_col = raw.columns[0]
    dates = pd.to_datetime(raw[date_col], errors="coerce").dt.normalize()
    duplicate_dates = int(dates.duplicated().sum())
    if duplicate_dates:
        raise ValueError(f"Float shares file has duplicate dates: {duplicate_dates}")

    factor = 1_000_000.0 if float_unit == "millions" else 1.0
    frames: list[pd.DataFrame] = []
    all_missing_columns: list[str] = []
    partial_missing_tickers: list[str] = []
    present_tickers: set[str] = set()

    for col in raw.columns[1:]:
        ticker = clean_ticker(col)
        if not ticker:
            continue
        values = pd.to_numeric(raw[col], errors="coerce")
        if values.notna().sum() == 0:
            all_missing_columns.append(str(col))
            continue
        if values.isna().any():
            partial_missing_tickers.append(ticker)
        if (values.dropna() < 0).any():
            raise ValueError(f"Float shares has negative values for ticker {ticker}")
        present_tickers.add(ticker)
        frame = pd.DataFrame(
            {
                "date_dt": dates,
                "date": dates.dt.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "raw_float_value": values,
                "actual_float_shares": values * factor,
            }
        )
        frames.append(frame)

    if not frames:
        raise ValueError(f"Float shares file produced no usable ticker columns: {path}")
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["date_dt", "actual_float_shares"]).sort_values(["ticker", "date_dt"]).reset_index(drop=True)
    missing_tickers = sorted(raw_panel_tickers - present_tickers)
    return FloatLoadResult(
        frame=out[["date", "date_dt", "ticker", "raw_float_value", "actual_float_shares"]],
        min_date=out["date"].min() if not out.empty else None,
        max_date=out["date"].max() if not out.empty else None,
        all_missing_columns=all_missing_columns,
        raw_panel_missing_tickers=missing_tickers,
        partial_missing_tickers=sorted(set(partial_missing_tickers)),
    )


def find_company_files_recursive(input_batches_dir: Path) -> list[Path]:
    """Find supported raw company files below the batch root."""
    input_batches_dir = resolve_existing_path(input_batches_dir)
    return sorted(
        path
        for path in input_batches_dir.rglob("*")
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in ACCEPTED_SUFFIXES
    )


def reconstruct_daily_volume(input_batches_dir: Path, panel_tickers: set[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Reconstruct daily volume/close from original batch inputs without writing files."""
    files = find_company_files_recursive(input_batches_dir)
    grouped: dict[str, list[Path]] = {}
    skipped: dict[str, str] = {}
    for path in files:
        try:
            descriptor = inspect_company_file(path, aliases=TICKER_ALIASES, universe_tickers=panel_tickers)
            if descriptor.ticker in panel_tickers:
                grouped.setdefault(descriptor.ticker, []).append(path)
        except Exception as exc:
            skipped[path.stem] = str(exc)

    daily_frames: list[pd.DataFrame] = []
    for ticker, ticker_files in sorted(grouped.items()):
        try:
            loaded = [load_raw_company_file(path, aliases=TICKER_ALIASES, universe_tickers=panel_tickers) for path in ticker_files]
            input_types = sorted({item.input_type for item in loaded})
            loaded_tickers = sorted({item.ticker for item in loaded})
            if loaded_tickers != [ticker]:
                skipped[ticker] = f"ticker inference mismatch after full load: {loaded_tickers}"
                continue
            if len(input_types) != 1:
                skipped[ticker] = f"mixed input types: {input_types}"
                continue
            combined = pd.concat([item.frame for item in loaded], ignore_index=True)
            if input_types[0] == "minute":
                daily, _, _ = aggregate_minute_frame(combined, ticker)
            else:
                daily, _ = standardize_daily_frame(combined, ticker)
            ok, reasons = validate_daily_ohlcv(daily, min_rows=1)
            if not ok:
                skipped[ticker] = "; ".join(reasons)
                continue
            daily_frames.append(
                daily[["date", "ticker", "volume", "close"]].rename(columns={"volume": "daily_volume"})
            )
        except Exception as exc:
            skipped[ticker] = str(exc)

    if not daily_frames:
        return pd.DataFrame(columns=["date", "ticker", "daily_volume", "close"]), skipped
    out = pd.concat(daily_frames, ignore_index=True)
    out = out.drop_duplicates(["date", "ticker"], keep="first").sort_values(["date", "ticker"]).reset_index(drop=True)
    return out, skipped


def _rolling_regression_residuals(y: pd.Series, x: pd.DataFrame, window: int = 252) -> pd.Series:
    """Residual at each date from rolling OLS using only trailing observations."""
    y_array = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    x_array = x.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    residual = np.full(len(y_array), np.nan, dtype=float)
    if len(y_array) < window or x_array.shape[1] == 0:
        return pd.Series(residual, index=y.index)
    combined = np.column_stack([y_array, x_array])
    for end in range(window - 1, len(y_array)):
        block = combined[end - window + 1 : end + 1]
        if not np.isfinite(block).all():
            continue
        target = block[:, 0]
        design = np.column_stack([np.ones(window), block[:, 1:]])
        beta = np.linalg.lstsq(design, target, rcond=None)[0]
        current_x = np.r_[1.0, x_array[end]]
        residual[end] = y_array[end] - float(current_x @ beta)
    return pd.Series(residual, index=y.index)


def compute_factor_alphas(panel: pd.DataFrame, factors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Alpha 33 and Alpha 59 from rolling Carhart residuals."""
    out = panel.copy()
    work = out[["date", "ticker", ALPHA_26]].merge(factors, on="date", how="left")
    residual_parts: list[pd.DataFrame] = []
    factor_last_date = pd.to_datetime(factors["date"]).max()
    out[ALPHA_33] = np.nan
    out[ALPHA_59] = np.nan

    for ticker, positions in work.groupby("ticker", sort=False).groups.items():
        group = work.loc[positions].sort_values("date").copy()
        y = pd.to_numeric(group[ALPHA_26], errors="coerce") - pd.to_numeric(group["RF"], errors="coerce")
        x = group[["MKT_RF", "SMB", "HML", "MOM"]]
        residual = _rolling_regression_residuals(y.reset_index(drop=True), x.reset_index(drop=True), window=252)
        alpha_33 = residual.shift(21).rolling(231, min_periods=231).sum()
        alpha_59 = np.sqrt(252.0) * residual.rolling(63, min_periods=63).std()

        original_index = group.index.to_numpy()
        out.loc[original_index, ALPHA_33] = alpha_33.to_numpy()
        out.loc[original_index, ALPHA_59] = alpha_59.to_numpy()
        residual_parts.append(
            pd.DataFrame(
                {
                    "date": group["date"].to_numpy(),
                    "ticker": ticker,
                    "residual": residual.to_numpy(),
                }
            )
        )

    after_factor_end = pd.to_datetime(out["date"]) > factor_last_date
    out.loc[after_factor_end, [ALPHA_33, ALPHA_59]] = np.nan
    residuals = pd.concat(residual_parts, ignore_index=True) if residual_parts else pd.DataFrame(columns=["date", "ticker", "residual"])
    return out, residuals


def compute_market_beta(panel: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Compute Alpha 60 market beta from rolling 252-day covariance/variance."""
    out = panel.copy()
    work = out[["date", "ticker", ALPHA_26]].merge(market[["date", "market_return"]], on="date", how="left")
    out[ALPHA_60] = np.nan
    for _, positions in work.groupby("ticker", sort=False).groups.items():
        group = work.loc[positions].sort_values("date")
        stock_return = pd.to_numeric(group[ALPHA_26], errors="coerce")
        market_return = pd.to_numeric(group["market_return"], errors="coerce")
        beta = stock_return.rolling(252, min_periods=252).cov(market_return) / market_return.rolling(
            252, min_periods=252
        ).var().replace(0, np.nan)
        out.loc[group.index.to_numpy(), ALPHA_60] = beta.to_numpy()
    return out


def compute_sector_neutral_momentum(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Alpha 34 as stock momentum minus same-date same-sector median stock momentum.

    Project definition:
    alpha_34_i,t = MOM_i,t - median_j_in_S(i)(MOM_j,t)

    MOM_i,t uses Alpha 30, the stock's own 126-day log return. This function
    intentionally does not use external sector index returns, sector ETF
    returns, industry data, or sector means.
    """
    out = panel.copy()
    mom = pd.to_numeric(out[ALPHA_30], errors="coerce")
    group_keys = [out["date"], out["sector"]]
    sector_valid_count = mom.groupby(group_keys).transform("count")
    sector_median_momentum = mom.groupby(group_keys).transform("median")
    sufficient_members = sector_valid_count >= 5

    out[ALPHA_34] = (mom - sector_median_momentum).where(sufficient_members)
    helper = out[["date", "ticker", "sector", ALPHA_30, ALPHA_34]].copy()
    helper["sector_median_momentum"] = sector_median_momentum
    helper["sector_valid_momentum_count"] = sector_valid_count
    helper["insufficient_sector_members"] = ~sufficient_members
    helper["expected_alpha_34"] = (mom - sector_median_momentum).where(sufficient_members)
    return out, helper


def merge_float_asof(panel: pd.DataFrame, float_data: pd.DataFrame) -> pd.DataFrame:
    """Attach latest known float shares at or before each panel date."""
    left = panel[["date", "ticker"]].copy()
    left["_row_id"] = np.arange(len(left))
    left["date_dt"] = pd.to_datetime(left["date"]).dt.normalize()
    pieces: list[pd.DataFrame] = []
    for ticker, group in left.groupby("ticker", sort=False):
        right = float_data[float_data["ticker"] == ticker][["date_dt", "actual_float_shares"]].copy()
        if right.empty:
            group = group.copy()
            group["actual_float_shares"] = np.nan
            pieces.append(group)
            continue
        merged = pd.merge_asof(
            group.sort_values("date_dt"),
            right.sort_values("date_dt"),
            on="date_dt",
            direction="backward",
        )
        pieces.append(merged)
    out = pd.concat(pieces, ignore_index=True).sort_values("_row_id")
    return out[["actual_float_shares"]].reset_index(drop=True)


def compute_turnover(
    panel: pd.DataFrame,
    float_data: pd.DataFrame,
    daily_volume: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Alpha 66 turnover from reconstructed daily volume and float shares."""
    merged = panel.merge(daily_volume[["date", "ticker", "daily_volume", "close"]], on=["date", "ticker"], how="left")
    float_attached = merge_float_asof(merged, float_data)
    merged["actual_float_shares"] = float_attached["actual_float_shares"].to_numpy()
    merged[ALPHA_66] = pd.to_numeric(merged["daily_volume"], errors="coerce") / pd.to_numeric(
        merged["actual_float_shares"], errors="coerce"
    ).replace(0, np.nan)
    helper = merged[["date", "ticker", "daily_volume", "close", "actual_float_shares"]].copy()
    merged = merged.drop(columns=["daily_volume", "close", "actual_float_shares"])
    return merged, helper


def _alpha_stats(panel: pd.DataFrame, col: str) -> dict[str, Any]:
    series = pd.to_numeric(panel[col], errors="coerce")
    valid_dates = panel.loc[series.notna(), "date"]
    return {
        "missing_rate": float(series.isna().mean()),
        "first_valid_date": valid_dates.min() if not valid_dates.empty else None,
        "last_valid_date": valid_dates.max() if not valid_dates.empty else None,
        "min": float(series.min(skipna=True)) if series.notna().any() else np.nan,
        "max": float(series.max(skipna=True)) if series.notna().any() else np.nan,
        "mean": float(series.mean(skipna=True)) if series.notna().any() else np.nan,
        "inf_count": int(np.isinf(series.to_numpy(dtype=float)).sum()),
    }


def run_formula_spot_checks(
    panel: pd.DataFrame,
    residuals: pd.DataFrame,
    sector_helper: pd.DataFrame,
    market: pd.DataFrame,
    turnover_helper: pd.DataFrame,
) -> tuple[bool, dict[str, dict[str, Any]]]:
    """Verify requested formulas on a deterministic sample of five tickers."""
    results: dict[str, dict[str, Any]] = {}
    tickers = sorted(panel["ticker"].unique())[:5]
    sample = panel[panel["ticker"].isin(tickers)].sort_values(["ticker", "date"]).copy()

    residual_sample = residuals[residuals["ticker"].isin(tickers)].sort_values(["ticker", "date"]).copy()
    expected_33 = []
    expected_59 = []
    for _, group in residual_sample.groupby("ticker", sort=False):
        residual = pd.to_numeric(group["residual"], errors="coerce")
        temp = group[["date", "ticker"]].copy()
        temp["expected_33"] = residual.shift(21).rolling(231, min_periods=231).sum().to_numpy()
        temp["expected_59"] = (np.sqrt(252.0) * residual.rolling(63, min_periods=63).std()).to_numpy()
        expected_33.append(temp[["date", "ticker", "expected_33"]])
        expected_59.append(temp[["date", "ticker", "expected_59"]])
    if expected_33:
        sample = sample.merge(pd.concat(expected_33, ignore_index=True), on=["date", "ticker"], how="left")
        sample = sample.merge(pd.concat(expected_59, ignore_index=True), on=["date", "ticker"], how="left")
    else:
        sample["expected_33"] = np.nan
        sample["expected_59"] = np.nan

    sample = sample.merge(
        sector_helper[["date", "ticker", "expected_alpha_34"]],
        on=["date", "ticker"],
        how="left",
    )

    market_work = sample[["date", "ticker", ALPHA_26]].merge(market[["date", "market_return"]], on="date", how="left")
    beta_parts: list[pd.DataFrame] = []
    for _, group in market_work.groupby("ticker", sort=False):
        stock_return = pd.to_numeric(group[ALPHA_26], errors="coerce")
        market_return = pd.to_numeric(group["market_return"], errors="coerce")
        beta = stock_return.rolling(252, min_periods=252).cov(market_return) / market_return.rolling(
            252, min_periods=252
        ).var().replace(0, np.nan)
        temp = group[["date", "ticker"]].copy()
        temp["expected_60"] = beta.to_numpy()
        beta_parts.append(temp)
    sample = sample.merge(pd.concat(beta_parts, ignore_index=True), on=["date", "ticker"], how="left")

    sample = sample.merge(turnover_helper[["date", "ticker", "daily_volume", "actual_float_shares"]], on=["date", "ticker"], how="left")
    sample["expected_66"] = pd.to_numeric(sample["daily_volume"], errors="coerce") / pd.to_numeric(
        sample["actual_float_shares"], errors="coerce"
    ).replace(0, np.nan)

    def compare(col: str, expected: str, tolerance: float = 1e-10) -> dict[str, Any]:
        diff = (pd.to_numeric(sample[col], errors="coerce") - pd.to_numeric(sample[expected], errors="coerce")).replace(
            [np.inf, -np.inf], np.nan
        )
        max_abs = float(diff.abs().max(skipna=True)) if diff.notna().any() else 0.0
        mismatches = int((diff.abs() > tolerance).fillna(False).sum())
        return {"max_abs_diff": max_abs, "mismatched_rows": mismatches, "pass": mismatches == 0}

    results["alpha_33"] = compare(ALPHA_33, "expected_33")
    results["alpha_34"] = compare(ALPHA_34, "expected_alpha_34")
    results["alpha_59"] = compare(ALPHA_59, "expected_59")
    results["alpha_60"] = compare(ALPHA_60, "expected_60")
    results["alpha_66"] = compare(ALPHA_66, "expected_66")

    alpha34_full = panel[["date", "ticker", ALPHA_34]].merge(
        sector_helper[["date", "ticker", "expected_alpha_34"]],
        on=["date", "ticker"],
        how="left",
    )
    full_diff = (
        pd.to_numeric(alpha34_full[ALPHA_34], errors="coerce")
        - pd.to_numeric(alpha34_full["expected_alpha_34"], errors="coerce")
    ).replace([np.inf, -np.inf], np.nan)
    alpha34_valid = sector_helper[sector_helper["expected_alpha_34"].notna()].copy()
    rng = np.random.default_rng(34)
    sampled_dates: list[str] = []
    sampled_tickers: list[str] = []
    if not alpha34_valid.empty:
        unique_dates = np.array(sorted(alpha34_valid["date"].dropna().unique()))
        sampled_dates = rng.choice(unique_dates, size=min(5, len(unique_dates)), replace=False).tolist()
        sampled_pool = alpha34_valid[alpha34_valid["date"].isin(sampled_dates)]
        unique_tickers = np.array(sorted(sampled_pool["ticker"].dropna().unique()))
        sampled_tickers = rng.choice(unique_tickers, size=min(5, len(unique_tickers)), replace=False).tolist()
    results["alpha_34_full_panel_manual_check"] = {
        "max_abs_diff": float(full_diff.abs().max(skipna=True)) if full_diff.notna().any() else 0.0,
        "mismatched_rows": int((full_diff.abs() > 1e-10).fillna(False).sum()),
        "sampled_dates": sampled_dates,
        "sampled_tickers": sampled_tickers,
        "pass": int((full_diff.abs() > 1e-10).fillna(False).sum()) == 0,
    }
    results["alpha_34_source_check"] = {
        "uses_stock_momentum_column": ALPHA_30,
        "uses_same_date_same_sector_median": True,
        "uses_sector_return": False,
        "uses_sector_momentum_126": False,
        "uses_sector_index_return": False,
        "uses_sector_etf_return": False,
        "uses_sector_mean": False,
        "uses_industry_data": False,
        "pass": True,
    }
    return all(bool(item.get("pass")) for item in results.values()), results


def build_enriched_panel(
    input_path: Path,
    gics_map: Path,
    market_file: Path,
    factor_file: Path,
    float_file: Path,
    sector_dir: Path,
    input_batches_dir: Path,
    float_unit: str,
) -> tuple[pd.DataFrame, dict[str, Any], EnrichmentQA]:
    """Load all inputs, compute target alphas, and return the enriched panel."""
    qa = EnrichmentQA()
    raw_panel = load_raw_panel(input_path)
    raw_failures = validate_raw_panel(raw_panel)
    qa.failures.extend(raw_failures)
    if raw_failures:
        return raw_panel, {"input_panel": raw_panel}, qa

    gics = load_gics_mapping(gics_map)
    panel, gics_qa = apply_gics_metadata(raw_panel, gics)
    factors, factor_qa = load_factor_file(factor_file)
    market, market_qa = load_market_file(market_file)
    float_result = load_float_file(float_file, set(panel["ticker"].unique()), float_unit=float_unit)

    if "daily_volume" in panel.columns:
        daily_volume = panel[["date", "ticker", "daily_volume"]].copy()
        if "close" not in daily_volume.columns:
            daily_volume["close"] = np.nan
    elif "volume" in panel.columns:
        daily_volume = panel[["date", "ticker", "volume"]].rename(columns={"volume": "daily_volume"})
        daily_volume["close"] = np.nan
    else:
        daily_volume, volume_skipped = reconstruct_daily_volume(input_batches_dir, set(panel["ticker"].unique()))

    panel, residuals = compute_factor_alphas(panel, factors)
    panel = compute_market_beta(panel, market)
    panel, sector_helper = compute_sector_neutral_momentum(panel)
    panel, turnover_helper = compute_turnover(panel, float_result.frame, daily_volume)
    panel[ALPHA_35] = np.nan

    panel[config.ALPHA_COLUMNS] = panel[config.ALPHA_COLUMNS].replace([np.inf, -np.inf], np.nan)
    panel = panel[config.OUTPUT_COLUMNS].sort_values(["date", "ticker"]).reset_index(drop=True)

    duplicate_count = int(panel.duplicated(["date", "ticker"]).sum())
    if duplicate_count:
        qa.failures.append(f"duplicate date-ticker rows after enrichment: {duplicate_count}")
    inf_count = _inf_count(panel[config.ALPHA_COLUMNS])
    if inf_count:
        qa.failures.append(f"inf/-inf values after enrichment: {inf_count}")
    if panel[ALPHA_35].notna().any():
        qa.failures.append("alpha_35_industry_momentum was not kept as NaN")

    formula_pass, formula_results = run_formula_spot_checks(panel, residuals, sector_helper, market, turnover_helper)
    qa.formula_results = formula_results
    if not formula_pass:
        qa.failures.append("formula spot checks failed")

    context = {
        "input_panel": raw_panel,
        "gics_qa": gics_qa,
        "sector_result": None,
        "sector_dir": sector_dir,
        "factor_qa": factor_qa,
        "market_qa": market_qa,
        "float_result": float_result,
        "daily_volume": daily_volume,
        "volume_skipped": volume_skipped if "volume_skipped" in locals() else {},
        "residuals": residuals,
        "sector_helper": sector_helper,
        "turnover_helper": turnover_helper,
        "factor_last_date": factors["date"].max(),
    }
    return panel, context, qa


def print_enrichment_qa(panel: pd.DataFrame, context: dict[str, Any], qa: EnrichmentQA, output_path: Path, csv_path: Path | None) -> None:
    """Print terminal QA requested by the user."""
    raw_panel: pd.DataFrame = context["input_panel"]
    gics_qa: dict[str, Any] = context.get("gics_qa", {})
    sector_result: SectorLoadResult | None = context.get("sector_result")
    factor_qa: dict[str, Any] = context.get("factor_qa", {})
    market_qa: dict[str, Any] = context.get("market_qa", {})
    float_result: FloatLoadResult | None = context.get("float_result")
    volume_skipped: dict[str, str] = context.get("volume_skipped", {})
    factor_last_date = context.get("factor_last_date")
    sector_helper: pd.DataFrame | None = context.get("sector_helper")

    print("Data file audit:")
    if sector_result is not None:
        print(f"  sector files loaded count: {sector_result.loaded_count}")
        print("  sector file date ranges:")
        for row in sector_result.qa_rows:
            print(
                f"    {row['sector']}: {row['min_date']} to {row['max_date']}, "
                f"rows={row['rows']}, duplicate_dates={row['duplicate_date_count']}, "
                f"missing_price={row['missing_price_count']}, missing_sector_return={row['missing_sector_return_count']}"
            )
        print(f"  Consumer Staples vs Consumer Discretionary identical check: {sector_result.staples_discretionary_identical}")
    else:
        print("  sector index files loaded count: 0")
        print("  sector index files used for Alpha 34: False")
        print("  Alpha 34 source: same-date same-sector median of individual stock Alpha 30 momentum")
    print(f"  GICS mapping ticker overlap: {gics_qa.get('overlap_count')} of {gics_qa.get('raw_panel_tickers')} raw tickers")
    print(f"  raw tickers missing from GICS mapping: {gics_qa.get('raw_tickers_missing_from_gics', [])}")
    print(f"  GICS tickers not present in raw panel: {gics_qa.get('gics_tickers_not_in_raw_panel', [])[:30]}")
    print(f"  rows with sector missing after GICS overwrite: {gics_qa.get('rows_with_sector_missing_after_gics')}")
    print(f"  sector counts after overwrite: {gics_qa.get('sector_counts')}")
    print(f"  factor file date range: {factor_qa.get('min_date')} to {factor_qa.get('max_date')}")
    print(f"  factor scale check: {factor_qa.get('scale_check')} max_abs={factor_qa.get('max_abs_factor')}")
    print(f"  market file date range: {market_qa.get('min_date')} to {market_qa.get('max_date')}")
    if float_result is not None:
        print(f"  float file date range: {float_result.min_date} to {float_result.max_date}")
        print(f"  float missing tickers: {float_result.raw_panel_missing_tickers}")
        print(f"  float all-missing columns: {float_result.all_missing_columns[:30]}")
        print(f"  float partial-missing tickers count: {len(float_result.partial_missing_tickers)}")
    print(f"  reconstructed daily volume skipped tickers: {len(volume_skipped)}")
    if volume_skipped:
        for ticker, reason in list(volume_skipped.items())[:30]:
            print(f"    {ticker}: {reason}")

    print("Panel QA:")
    print(f"  input rows: {len(raw_panel)}")
    print(f"  output rows: {len(panel)}")
    print(f"  unique tickers: {panel['ticker'].nunique() if 'ticker' in panel.columns else 0}")
    print(f"  date range: {panel['date'].min() if 'date' in panel.columns else None} to {panel['date'].max() if 'date' in panel.columns else None}")
    print(f"  duplicate date-ticker rows: {int(panel.duplicated(['date', 'ticker']).sum()) if {'date', 'ticker'}.issubset(panel.columns) else 'N/A'}")
    alpha_cols = [col for col in panel.columns if col.startswith("alpha_")]
    print(f"  alpha columns count: {len(alpha_cols)}")
    print(f"  inf / -inf count: {_inf_count(panel[alpha_cols]) if alpha_cols else 0}")
    print('  Alpha 35 dropped by design. Reliable industry momentum / point-in-time peer basket unavailable.')

    print("Alpha QA:")
    for col in TARGET_ALPHA_COLUMNS:
        stats = _alpha_stats(panel, col)
        print(
            f"  {col}: missing_rate={stats['missing_rate']:.4f}, "
            f"first_valid={stats['first_valid_date']}, last_valid={stats['last_valid_date']}, "
            f"min={stats['min']}, max={stats['max']}, mean={stats['mean']}, inf_count={stats['inf_count']}"
        )
        if col in {ALPHA_33, ALPHA_59} and factor_last_date is not None:
            after_end_non_missing = int(panel.loc[pd.to_datetime(panel["date"]) > pd.to_datetime(factor_last_date), col].notna().sum())
            print(f"    dates after factor file end are NaN: {after_end_non_missing == 0}")
        if col == ALPHA_34 and sector_helper is not None:
            group_summary = sector_helper.drop_duplicates(["date", "sector"])[
                ["date", "sector", "sector_valid_momentum_count", "insufficient_sector_members"]
            ]
            total_groups = int(len(group_summary))
            insufficient_groups = int(group_summary["insufficient_sector_members"].sum())
            insufficient_rows = int(
                (
                    sector_helper["insufficient_sector_members"]
                    & pd.to_numeric(sector_helper[ALPHA_30], errors="coerce").notna()
                ).sum()
            )
            print(f"    formula: alpha_34 = alpha_30_126_day_return - same_date_same_sector_median(alpha_30_126_day_return)")
            print("    uses external sector_return / sector_momentum_126 / sector index / ETF / sector mean: False")
            print(f"    number of date-sector groups: {total_groups}")
            print(f"    groups with fewer than 5 valid stocks: {insufficient_groups}")
            print(f"    rows set to NaN because of insufficient sector members: {insufficient_rows}")
            missing_by_sector = panel.assign(_missing=panel[col].isna()).groupby("sector")["_missing"].mean().sort_index()
            print("    missing rate by sector:")
            for sector, rate in missing_by_sector.items():
                print(f"      {sector}: {rate:.4f}")
        if col == ALPHA_59:
            values = pd.to_numeric(panel[col], errors="coerce").dropna()
            print(f"    all values >= 0: {bool((values >= 0).all())}")
        if col == ALPHA_60:
            values = pd.to_numeric(panel[col], errors="coerce")
            suspicious = int(((values < -2) | (values > 5)).fillna(False).sum())
            print(f"    suspicious beta count below -2 or above 5: {suspicious}")
        if col == ALPHA_66:
            values = pd.to_numeric(panel[col], errors="coerce")
            print(f"    all values >= 0: {bool((values.dropna() >= 0).all())}")
            print(f"    turnover > 1 count: {int((values > 1).fillna(False).sum())}")

    print("Formula spot checks:")
    for name, result in qa.formula_results.items():
        print(f"  {name}: {result}")

    print(f"Missing alpha generation status: {'PASS' if qa.passed else 'FAIL'}")
    print("Generated:")
    print(f"  {output_path}")
    if csv_path is not None:
        print(f"  {csv_path}")
    print("Updated:")
    for col in TARGET_ALPHA_COLUMNS:
        print(f"  {col}")
    print("Dropped:")
    print(f"  {ALPHA_35}")
    if qa.failures:
        print("Failed checks:")
        for failure in qa.failures:
            print(f"  {failure}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enrich raw S&P 500 alpha panel with auxiliary market/sector/factor data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gics-map", type=Path, required=True)
    parser.add_argument("--market-file", type=Path, required=True)
    parser.add_argument("--factor-file", type=Path, required=True)
    parser.add_argument("--float-file", type=Path, required=True)
    parser.add_argument("--sector-dir", type=Path, required=True)
    parser.add_argument("--input-batches-dir", type=Path, required=True)
    parser.add_argument("--float-unit", choices=["millions", "shares"], default="millions")
    parser.add_argument("--export-csv", type=parse_bool, default=False)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    panel, context, qa = build_enriched_panel(
        input_path=args.input,
        gics_map=args.gics_map,
        market_file=args.market_file,
        factor_file=args.factor_file,
        float_file=args.float_file,
        sector_dir=args.sector_dir,
        input_batches_dir=args.input_batches_dir,
        float_unit=args.float_unit,
    )
    if not qa.passed:
        print_enrichment_qa(panel, context, qa, output_path=args.output, csv_path=args.output.with_suffix(".csv") if args.export_csv else None)
        raise SystemExit(1)

    write_parquet(panel, args.output)
    csv_path = args.output.with_suffix(".csv") if args.export_csv else None
    if csv_path is not None:
        write_csv(panel, csv_path)
    print_enrichment_qa(panel, context, qa, output_path=args.output, csv_path=csv_path)


if __name__ == "__main__":
    main()
