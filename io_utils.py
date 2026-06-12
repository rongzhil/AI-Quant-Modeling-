"""Input/output utilities for ticker files, universe metadata, and auxiliary data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config


ACCEPTED_SUFFIXES = {".csv", ".parquet"}
RETURN_COLUMN_CANDIDATES = ["return", "ret", "daily_return", "market_return", "spy_return", "factor_return"]


@dataclass
class StandardizedTickerData:
    """Daily OHLCV data prepared for one ticker."""

    ticker: str
    daily: pd.DataFrame
    source_paths: list[Path]
    input_type: str
    warnings: list[str]
    duplicate_timestamp_count: int | None = None
    duplicate_daily_date_count: int = 0


@dataclass
class LoadedCompanyFile:
    """Raw normalized input loaded from one company file."""

    ticker: str
    input_type: str
    frame: pd.DataFrame
    source_path: Path


@dataclass(frozen=True)
class InputFileDescriptor:
    """Lightweight file descriptor inferred from file headers."""

    ticker: str
    input_type: str
    source_path: Path


def standardize_ticker(ticker: Any, aliases: dict[str, str] | None = None) -> str:
    """Uppercase a ticker and apply configurable share-class aliases."""
    aliases = aliases or config.DEFAULT_TICKER_ALIASES
    text = str(ticker).strip().upper()
    text = " ".join(text.split())
    return aliases.get(text, text)


def normalize_column_name(column: str) -> str:
    """Make column matching robust to spaces, case, and punctuation variants."""
    text = str(column).strip().lower()
    text = text.replace(" ", "_").replace("-", "_")
    text = text.replace(".", "_").replace("/", "_")
    return "_".join(part for part in text.split("_") if part)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_column_name(col) for col in out.columns]
    return out


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or parquet file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type {path.suffix!r}; expected .csv or .parquet: {path}")


def read_table_preview(path: str | Path, nrows: int = 5) -> pd.DataFrame:
    """Read a small preview for ticker/type inference without loading a full CSV."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type {path.suffix!r}; expected .csv or .parquet: {path}")


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def find_input_files(input_dir: str | Path) -> list[Path]:
    """Return supported ticker data files in a directory."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    files = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in ACCEPTED_SUFFIXES
    ]
    return files


def load_universe(path: str | Path, aliases: dict[str, str] | None = None) -> pd.DataFrame:
    """Load S&P 500 universe metadata with required ticker column."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")
    universe = normalize_columns(read_table(path))
    if "ticker" not in universe.columns:
        raise ValueError(f"Universe file must contain a ticker column: {path}")

    universe["ticker"] = universe["ticker"].map(lambda x: standardize_ticker(x, aliases))
    for col in config.METADATA_COLUMNS:
        if col not in universe.columns:
            universe[col] = ""
    universe = universe[["ticker", *config.METADATA_COLUMNS]].drop_duplicates("ticker", keep="first")
    return universe.reset_index(drop=True)


def infer_ticker_from_file(
    path: str | Path,
    aliases: dict[str, str] | None = None,
    universe_tickers: set[str] | None = None,
) -> str:
    """Infer ticker from filename, with support for monthly-file suffixes."""
    stem = standardize_ticker(Path(path).stem, aliases)
    if not universe_tickers or stem in universe_tickers:
        return stem
    for separator in ["_", " "]:
        prefix = standardize_ticker(stem.split(separator)[0], aliases)
        if prefix in universe_tickers:
            return prefix
    return stem


def infer_ticker_from_frame(df: pd.DataFrame, fallback: str, aliases: dict[str, str] | None = None) -> str:
    if "ticker" not in df.columns:
        return fallback
    tickers = df["ticker"].dropna().map(lambda x: standardize_ticker(x, aliases)).unique().tolist()
    if len(tickers) == 0:
        return fallback
    if len(tickers) > 1:
        raise ValueError(f"File contains multiple ticker values: {tickers[:10]}")
    return tickers[0]


def detect_input_type(df: pd.DataFrame) -> str:
    if "timestamp" in df.columns or ("date" in df.columns and "time" in df.columns):
        return "minute"
    if "date" in df.columns:
        return "daily"
    raise ValueError("Input file must contain timestamp, date+time, or date columns.")


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _coalesce_columns(df: pd.DataFrame, candidates: list[str], required_name: str) -> pd.Series:
    for candidate in candidates:
        if candidate in df.columns:
            return _numeric_series(df, candidate)
    raise ValueError(f"Missing required column for {required_name}; tried {candidates}")


def standardize_daily_frame(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, list[str]]:
    """Standardize a daily input frame to date/ticker/open/high/low/close/volume/daily_vwap."""
    warnings: list[str] = []
    out = pd.DataFrame()
    dates = pd.to_datetime(df["date"], errors="coerce")
    out["date"] = dates.dt.strftime("%Y-%m-%d")
    out["ticker"] = ticker
    out["open"] = _coalesce_columns(df, ["open", "daily_open"], "open")
    out["high"] = _coalesce_columns(df, ["high", "daily_high"], "high")
    out["low"] = _coalesce_columns(df, ["low", "daily_low"], "low")
    out["close"] = _coalesce_columns(df, ["close", "daily_close"], "close")
    out["volume"] = _coalesce_columns(df, ["volume", "daily_volume"], "volume")

    if "daily_vwap" in df.columns:
        out["daily_vwap"] = _numeric_series(df, "daily_vwap")
    elif "vwap" in df.columns:
        out["daily_vwap"] = _numeric_series(df, "vwap")
    else:
        out["daily_vwap"] = out["close"]
        warnings.append("daily_vwap unavailable; using close proxy for VWAP-dependent feature.")

    if "adjusted_close" in df.columns:
        out["adjusted_close"] = _numeric_series(df, "adjusted_close")

    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    return out, warnings


def raw_timestamp_series(df: pd.DataFrame) -> pd.Series:
    """Return a timestamp-like series from timestamp or date+time columns."""
    if "timestamp" in df.columns:
        return df["timestamp"]
    if "date" in df.columns and "time" in df.columns:
        return df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
    raise ValueError("Minute input must contain timestamp or date+time columns.")


def _localize_or_convert(parsed: pd.Series, timezone: str) -> pd.Series:
    if getattr(parsed.dt, "tz", None) is None:
        return parsed.dt.tz_localize(timezone, nonexistent="shift_forward", ambiguous="NaT")
    return parsed.dt.tz_convert(timezone)


def parse_timestamps(timestamp: pd.Series, timezone: str = config.DEFAULT_TIMEZONE) -> pd.Series:
    """Parse timestamps and convert/localize them to the configured market timezone."""
    parsed = pd.to_datetime(timestamp, errors="coerce")
    return _localize_or_convert(parsed, timezone)


def parse_minute_timestamps(df: pd.DataFrame, timezone: str = config.DEFAULT_TIMEZONE) -> pd.Series:
    """Fast timestamp parser for either timestamp or date+time minute inputs."""
    if "timestamp" in df.columns:
        return parse_timestamps(df["timestamp"], timezone)

    if "date" in df.columns and "time" in df.columns:
        date_text = df["date"].astype(str).str.strip()
        time_text = df["time"].astype(str).str.strip()
        unique_dates = pd.Series(date_text.dropna().unique())
        parsed_dates = pd.to_datetime(unique_dates, errors="coerce", format="%m/%d/%Y")
        if parsed_dates.isna().mean() > 0.50:
            parsed_dates = pd.to_datetime(unique_dates, errors="coerce", format="%Y-%m-%d")
        date_map = dict(zip(unique_dates, parsed_dates))

        unique_times = pd.Series(time_text.dropna().unique())
        parsed_times = pd.to_timedelta(unique_times, errors="coerce")
        time_map = dict(zip(unique_times, parsed_times))

        date_part = date_text.map(date_map)
        time_part = time_text.map(time_map)
        parsed = date_part + time_part
        if parsed.isna().mean() > 0.50:
            combined = date_text + " " + time_text
            parsed = pd.to_datetime(combined, errors="coerce")
        return parsed

    raise ValueError("Minute input must contain timestamp or date+time columns.")




def standardize_minute_frame(
    df: pd.DataFrame,
    ticker: str,
    timezone: str = config.DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Standardize raw minute rows before duplicate timestamp handling."""
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Minute input missing required columns: {missing}")
    return pd.DataFrame(
        {
            "timestamp": parse_minute_timestamps(df, timezone),
            "ticker": ticker,
            "open": _numeric_series(df, "open"),
            "high": _numeric_series(df, "high"),
            "low": _numeric_series(df, "low"),
            "close": _numeric_series(df, "close"),
            "volume": _numeric_series(df, "volume"),
        }
    ).sort_values("timestamp").reset_index(drop=True)


def drop_exact_duplicate_timestamps_or_raise(minute: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate timestamp rows, but reject conflicting duplicates."""
    duplicate_count = int(minute.duplicated("timestamp").sum())
    if duplicate_count == 0:
        return minute, 0

    conflict_timestamps: list[str] = []
    duplicate_rows = minute[minute.duplicated("timestamp", keep=False)]
    value_columns = ["open", "high", "low", "close", "volume"]
    for timestamp, group in duplicate_rows.groupby("timestamp", sort=False):
        if len(group[value_columns].drop_duplicates()) > 1:
            conflict_timestamps.append(str(timestamp))
            if len(conflict_timestamps) >= 10:
                break
    if conflict_timestamps:
        raise ValueError(
            "conflicting duplicate timestamps after combining files: "
            + ", ".join(conflict_timestamps)
        )
    deduped = minute.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    return deduped, duplicate_count


def validate_minute_ohlcv(minute: pd.DataFrame) -> tuple[bool, list[str]]:
    """Critical validation for minute-level rows before daily aggregation."""
    reasons: list[str] = []
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in minute.columns]
    if missing:
        reasons.append(f"missing minute columns: {missing}")
        return False, reasons
    if minute["timestamp"].isna().any():
        reasons.append(f"invalid or missing timestamps: {int(minute['timestamp'].isna().sum())}")
    if minute["timestamp"].duplicated().any():
        reasons.append(f"duplicate timestamps after exact duplicate cleanup: {int(minute['timestamp'].duplicated().sum())}")
    if not minute["timestamp"].is_monotonic_increasing:
        reasons.append("timestamps are not sorted ascending")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    numeric = minute[numeric_cols].apply(pd.to_numeric, errors="coerce")
    inf_count = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
    if inf_count:
        reasons.append(f"inf/-inf minute OHLCV values: {inf_count}")
    if numeric[["open", "high", "low", "close"]].isna().any().any():
        reasons.append("missing or nonnumeric minute OHLC values")
    if numeric["volume"].isna().any():
        reasons.append("missing or nonnumeric minute volume values")

    invalid_high = ~((numeric["high"] >= numeric["open"]) & (numeric["high"] >= numeric["close"]) & (numeric["high"] >= numeric["low"]))
    invalid_low = ~((numeric["low"] <= numeric["open"]) & (numeric["low"] <= numeric["close"]) & (numeric["low"] <= numeric["high"]))
    nonpositive_prices = ~(numeric[["open", "high", "low", "close"]] > 0).all(axis=1)
    negative_volume = numeric["volume"] < 0
    if invalid_high.any():
        reasons.append(f"minute high is below open/close/low on {int(invalid_high.sum())} rows")
    if invalid_low.any():
        reasons.append(f"minute low is above open/close/high on {int(invalid_low.sum())} rows")
    if nonpositive_prices.any():
        reasons.append(f"minute nonpositive OHLC prices on {int(nonpositive_prices.sum())} rows")
    if negative_volume.any():
        reasons.append(f"minute negative volume on {int(negative_volume.sum())} rows")
    return len(reasons) == 0, reasons


def aggregate_standardized_minute_frame(
    minute: pd.DataFrame,
    ticker: str,
    regular_open: str = config.REGULAR_SESSION_OPEN,
    regular_close: str = config.REGULAR_SESSION_CLOSE,
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate standardized regular-session minute OHLCV bars to daily OHLCV."""
    warnings: list[str] = []
    minute = minute.sort_values("timestamp")
    open_hour, open_minute = map(int, regular_open.split(":"))
    close_hour, close_minute = map(int, regular_close.split(":"))
    start_minute = open_hour * 60 + open_minute
    end_minute = close_hour * 60 + close_minute
    minute_of_day = minute["timestamp"].dt.hour * 60 + minute["timestamp"].dt.minute
    minute = minute[(minute_of_day >= start_minute) & (minute_of_day <= end_minute)].copy()
    if minute.empty:
        raise ValueError("No regular-session minute rows remain after 09:30-16:00 filtering.")

    minute["date"] = minute["timestamp"].dt.strftime("%Y-%m-%d")
    minute["dollar_volume"] = minute["close"] * minute["volume"]
    grouped = minute.groupby("date", sort=True)
    daily = pd.DataFrame(
        {
            "date": grouped["date"].first(),
            "ticker": ticker,
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
            "daily_dollar_volume": grouped["dollar_volume"].sum(),
        }
    ).reset_index(drop=True)
    daily["daily_vwap"] = daily["daily_dollar_volume"] / daily["volume"].replace(0, np.nan)
    daily = daily.drop(columns=["daily_dollar_volume"])
    if daily["daily_vwap"].isna().any():
        warnings.append("Some daily_vwap values are NaN because aggregated daily volume is zero.")
    return daily, warnings


def aggregate_minute_frame(
    df: pd.DataFrame,
    ticker: str,
    timezone: str = config.DEFAULT_TIMEZONE,
    regular_open: str = config.REGULAR_SESSION_OPEN,
    regular_close: str = config.REGULAR_SESSION_CLOSE,
) -> tuple[pd.DataFrame, list[str], int]:
    """Validate, de-duplicate, and aggregate raw minute rows to daily OHLCV."""
    minute = standardize_minute_frame(df, ticker, timezone=timezone)
    minute, duplicate_timestamp_count = drop_exact_duplicate_timestamps_or_raise(minute)
    ok, reasons = validate_minute_ohlcv(minute)
    if not ok:
        raise ValueError("; ".join(reasons))
    daily, warnings = aggregate_standardized_minute_frame(
        minute=minute,
        ticker=ticker,
        regular_open=regular_open,
        regular_close=regular_close,
    )
    return daily, warnings, duplicate_timestamp_count


def load_raw_company_file(
    path: str | Path,
    aliases: dict[str, str] | None = None,
    universe_tickers: set[str] | None = None,
) -> LoadedCompanyFile:
    """Load one file, normalize columns, infer ticker, and detect daily/minute type."""
    path = Path(path)
    raw = normalize_columns(read_table(path))
    inferred_ticker = infer_ticker_from_file(path, aliases, universe_tickers=universe_tickers)
    ticker = infer_ticker_from_frame(raw, inferred_ticker, aliases)
    input_type = detect_input_type(raw)
    return LoadedCompanyFile(ticker=ticker, input_type=input_type, frame=raw, source_path=path)


def inspect_company_file(
    path: str | Path,
    aliases: dict[str, str] | None = None,
    universe_tickers: set[str] | None = None,
) -> InputFileDescriptor:
    """Infer ticker and input type without loading a full CSV file."""
    path = Path(path)
    preview = normalize_columns(read_table_preview(path))
    inferred_ticker = infer_ticker_from_file(path, aliases, universe_tickers=universe_tickers)
    ticker = infer_ticker_from_frame(preview, inferred_ticker, aliases)
    input_type = detect_input_type(preview)
    return InputFileDescriptor(ticker=ticker, input_type=input_type, source_path=path)


def load_and_standardize_company_file(
    path: str | Path,
    aliases: dict[str, str] | None = None,
    timezone: str = config.DEFAULT_TIMEZONE,
) -> StandardizedTickerData:
    """Load one company file and return standardized daily OHLCV rows."""
    loaded = load_raw_company_file(path, aliases=aliases)
    if loaded.input_type == "minute":
        daily, warnings, duplicate_timestamp_count = aggregate_minute_frame(loaded.frame, loaded.ticker, timezone=timezone)
    else:
        daily, warnings = standardize_daily_frame(loaded.frame, loaded.ticker)
        duplicate_timestamp_count = None
    return StandardizedTickerData(
        ticker=loaded.ticker,
        daily=daily,
        source_paths=[loaded.source_path],
        input_type=loaded.input_type,
        warnings=warnings,
        duplicate_timestamp_count=duplicate_timestamp_count,
        duplicate_daily_date_count=int(daily["date"].duplicated().sum()) if "date" in daily.columns else 0,
    )


def validate_daily_ohlcv(daily: pd.DataFrame, min_rows: int = config.MIN_DAILY_ROWS) -> tuple[bool, list[str]]:
    """Run critical single-ticker OHLCV validation before alpha generation."""
    reasons: list[str] = []
    required = ["date", "ticker", "open", "high", "low", "close", "volume", "daily_vwap"]
    missing = [col for col in required if col not in daily.columns]
    if missing:
        reasons.append(f"missing standardized columns: {missing}")
        return False, reasons

    if daily["date"].isna().any():
        reasons.append("one or more rows have invalid or missing dates")
    if daily["date"].duplicated().any():
        reasons.append(f"duplicate date rows: {int(daily['date'].duplicated().sum())}")
    if not daily["date"].is_monotonic_increasing:
        reasons.append("dates are not sorted ascending")

    numeric_cols = ["open", "high", "low", "close", "volume", "daily_vwap"]
    numeric = daily[numeric_cols].apply(pd.to_numeric, errors="coerce")
    inf_count = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
    if inf_count:
        reasons.append(f"inf/-inf values in OHLCV columns: {inf_count}")
    if numeric[["open", "high", "low", "close"]].isna().any().any():
        reasons.append("missing or nonnumeric OHLC price values")
    if numeric["volume"].isna().any():
        reasons.append("missing or nonnumeric volume values")

    invalid_high = ~((numeric["high"] >= numeric["open"]) & (numeric["high"] >= numeric["close"]) & (numeric["high"] >= numeric["low"]))
    invalid_low = ~((numeric["low"] <= numeric["open"]) & (numeric["low"] <= numeric["close"]) & (numeric["low"] <= numeric["high"]))
    nonpositive_prices = ~(numeric[["open", "high", "low", "close"]] > 0).all(axis=1)
    negative_volume = numeric["volume"] < 0
    if invalid_high.any():
        reasons.append(f"high is below open/close/low on {int(invalid_high.sum())} rows")
    if invalid_low.any():
        reasons.append(f"low is above open/close/high on {int(invalid_low.sum())} rows")
    if nonpositive_prices.any():
        reasons.append(f"nonpositive OHLC prices on {int(nonpositive_prices.sum())} rows")
    if negative_volume.any():
        reasons.append(f"negative volume on {int(negative_volume.sum())} rows")
    if len(daily) < min_rows:
        reasons.append(f"fewer than {min_rows} valid daily rows: {len(daily)}")

    return len(reasons) == 0, reasons


def _coerce_return_frame(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Return a date/value frame from return or close columns."""
    out = normalize_columns(df)
    if "date" not in out.columns:
        raise ValueError("Auxiliary return file must contain a date column.")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in RETURN_COLUMN_CANDIDATES:
        if col in out.columns:
            return pd.DataFrame({"date": out["date"], value_name: pd.to_numeric(out[col], errors="coerce")})
    for col in ["adjusted_close", "daily_close", "close"]:
        if col in out.columns:
            close = pd.to_numeric(out[col], errors="coerce")
            return pd.DataFrame({"date": out["date"], value_name: np.log(close / close.shift(1))})
    raise ValueError("Auxiliary return file needs a return column or close/adjusted_close price column.")


def _load_factor_returns(path: Path) -> pd.DataFrame:
    factors = normalize_columns(read_table(path))
    if "date" not in factors.columns:
        raise ValueError(f"Factor file must contain date: {path}")
    factors["date"] = pd.to_datetime(factors["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    factor_cols: list[str] = []
    for col in factors.columns:
        if col == "date":
            continue
        factors[col] = pd.to_numeric(factors[col], errors="coerce")
        if factors[col].notna().any():
            factor_cols.append(col)
    if not factor_cols:
        raise ValueError(f"Factor file has no numeric factor columns: {path}")
    return factors[["date", *factor_cols]].sort_values("date").reset_index(drop=True)


def _load_group_returns(path: Path, group_column: str, value_name: str) -> pd.DataFrame:
    raw = normalize_columns(read_table(path))
    if "date" not in raw.columns:
        raise ValueError(f"Group return file must contain date: {path}")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return_col = next((col for col in RETURN_COLUMN_CANDIDATES if col in raw.columns), None)
    if group_column in raw.columns and return_col:
        out = raw[["date", group_column, return_col]].copy()
        out = out.rename(columns={group_column: "group", return_col: value_name})
        out["group"] = out["group"].astype(str).str.strip()
        out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
        return out.sort_values(["group", "date"]).reset_index(drop=True)

    # Wide format: date plus one column per sector or industry name.
    value_columns = [col for col in raw.columns if col != "date"]
    melted = raw.melt(id_vars="date", value_vars=value_columns, var_name="group", value_name=value_name)
    melted["group"] = melted["group"].astype(str).str.strip()
    melted[value_name] = pd.to_numeric(melted[value_name], errors="coerce")
    return melted.sort_values(["group", "date"]).reset_index(drop=True)


def _load_float_shares(path: Path, aliases: dict[str, str] | None = None) -> pd.DataFrame:
    raw = normalize_columns(read_table(path))
    if "float_shares" not in raw.columns and "shares_outstanding" not in raw.columns:
        raise ValueError(f"Float file must contain float_shares or shares_outstanding: {path}")
    share_col = "float_shares" if "float_shares" in raw.columns else "shares_outstanding"
    out = pd.DataFrame({"float_shares": pd.to_numeric(raw[share_col], errors="coerce")})
    if "ticker" in raw.columns:
        out["ticker"] = raw["ticker"].map(lambda x: standardize_ticker(x, aliases))
    if "date" in raw.columns:
        out["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def load_auxiliary_data(auxiliary_dir: str | Path, aliases: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """Load optional auxiliary files if present. Missing files are simply absent."""
    auxiliary_dir = Path(auxiliary_dir)
    aux: dict[str, pd.DataFrame] = {}
    if not auxiliary_dir.exists():
        return aux

    files = {
        "factors": auxiliary_dir / "ff4_factors_daily.csv",
        "market": auxiliary_dir / "spy_market_return.csv",
        "sector": auxiliary_dir / "sector_daily_returns.csv",
        "industry": auxiliary_dir / "industry_daily_returns.csv",
        "float": auxiliary_dir / "float_shares.csv",
    }
    if files["factors"].exists():
        aux["factor_returns"] = _load_factor_returns(files["factors"])
    if files["market"].exists():
        aux["market_return"] = _coerce_return_frame(read_table(files["market"]), "market_return")
    if files["sector"].exists():
        aux["sector_returns"] = _load_group_returns(files["sector"], "sector", "sector_return")
    if files["industry"].exists():
        aux["industry_returns"] = _load_group_returns(files["industry"], "industry", "industry_return")
    if files["float"].exists():
        aux["float_shares"] = _load_float_shares(files["float"], aliases)
    return aux
