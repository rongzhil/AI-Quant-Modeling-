"""Data loading and standardization for the S&P 500 alpha pipeline.

This module turns the raw inputs on disk into clean, aligned tables the rest of
the pipeline can consume:

  * minute OHLCV CSVs (one file per ticker, split across batch folders) are
    aggregated to a daily panel, computing a *real* volume-weighted average
    price (VWAP) from the intraday bars;
  * the GICS mapping spreadsheet is parsed past its Bloomberg metadata rows
    into a ticker -> sector / industry table;
  * Carhart factors, the SPX index, and float shares are loaded and
    standardized for the alphas that need them.

Ticker convention: every source is reduced to a bare symbol (e.g. "AAPL",
"WBD"), stripping Bloomberg's " US Equity" suffix and applying the share-class
aliases in ``config.TICKER_ALIASES`` so the three sources align.

No alpha math lives here. This is the data floor only.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Ticker / column helpers
# --------------------------------------------------------------------------- #
_BLOOMBERG_SUFFIX = re.compile(r"\s+US\s+Equity\s*$", flags=re.IGNORECASE)


def standardize_ticker(raw: str) -> str:
    """Reduce any source's symbol to a bare, aliased ticker.

    Handles Bloomberg's ``"WBD US Equity"`` suffix and share-class variants
    (``BRK/B`` -> ``BRK.B``). Idempotent on already-clean symbols like ``AAPL``.
    """
    text = str(raw).strip()
    text = _BLOOMBERG_SUFFIX.sub("", text).strip()
    text = text.upper()
    return config.TICKER_ALIASES.get(text, text)


def _normalize_columns(columns) -> list[str]:
    """Lowercase column names and collapse separators to underscores."""
    out = []
    for col in columns:
        name = str(col).strip().lower()
        name = re.sub(r"[\s\-/.]+", "_", name)
        out.append(name)
    return out


# --------------------------------------------------------------------------- #
# Minute -> daily aggregation
# --------------------------------------------------------------------------- #
def load_minute_file(path: Path) -> pd.DataFrame:
    """Read one ticker's minute CSV into a typed frame.

    Expected columns: Date, Time, open, high, low, close, volume. Returns a
    frame with a tz-naive ``timestamp`` plus numeric OHLCV.
    """
    raw = pd.read_csv(path)
    raw.columns = _normalize_columns(raw.columns)

    missing = [c for c in ("date", "time", "open", "high", "low", "close", "volume") if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing expected columns {missing}")

    timestamp = pd.to_datetime(
        raw["date"].astype(str).str.strip() + " " + raw["time"].astype(str).str.strip(),
        errors="coerce",
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "volume": pd.to_numeric(raw["volume"], errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return frame


def aggregate_to_daily(minute: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one ticker's minute bars to a daily OHLCV + real VWAP table.

    Only regular-session bars (09:30-16:00 inclusive) are used. VWAP is the
    true volume-weighted average price: sum(close * volume) / sum(volume) over
    the day's bars. Days with zero traded volume get VWAP = NaN.
    """
    if minute.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "daily_vwap"])

    session = minute.set_index("timestamp").between_time(
        config.REGULAR_SESSION_OPEN, config.REGULAR_SESSION_CLOSE
    )
    if session.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "daily_vwap"])

    session = session.copy()
    session["dollar_volume"] = session["close"] * session["volume"]
    session["date"] = session.index.normalize()

    grouped = session.groupby("date", sort=True)
    daily = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        dollar_volume=("dollar_volume", "sum"),
    ).reset_index()

    daily["daily_vwap"] = np.where(
        daily["volume"] > 0,
        daily["dollar_volume"] / daily["volume"].replace(0, np.nan),
        np.nan,
    )
    daily = daily.drop(columns=["dollar_volume"])
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    return daily


def _validate_daily(daily: pd.DataFrame) -> list[str]:
    """Return a list of data-quality problems; empty means clean."""
    problems: list[str] = []
    if daily.empty:
        return ["no regular-session rows"]
    if daily["date"].duplicated().any():
        problems.append("duplicate dates after aggregation")
    if not daily["date"].is_monotonic_increasing:
        problems.append("dates not sorted")
    for col in ("open", "high", "low", "close"):
        if (daily[col] <= 0).any():
            problems.append(f"non-positive {col}")
    if (daily["volume"] < 0).any():
        problems.append("negative volume")
    # OHLC logical bounds (allow tiny float noise).
    eps = 1e-6
    if ((daily["high"] + eps) < daily[["open", "close", "low"]].max(axis=1)).any():
        problems.append("high below open/close/low")
    if ((daily["low"] - eps) > daily[["open", "close", "high"]].min(axis=1)).any():
        problems.append("low above open/close/high")
    return problems


def _read_minute_batches(verbose: bool = True) -> pd.DataFrame:
    """Aggregate per-ticker minute CSVs in the batch dirs into a daily panel.

    Original price path: walks INPUT_BATCH_DIRS, aggregates each minute file to
    daily bars (computing a real intraday VWAP), validates, and concatenates.
    Returns a long panel: date, ticker, open, high, low, close, volume, daily_vwap.
    """
    frames: list[pd.DataFrame] = []
    skipped: dict[str, str] = {}
    files_found = 0

    for batch_dir in config.INPUT_BATCH_DIRS:
        if not batch_dir.exists():
            if verbose:
                print(f"WARNING: batch dir not found: {batch_dir}")
            continue
        for path in sorted(batch_dir.glob("*.csv")):
            files_found += 1
            ticker = standardize_ticker(path.stem)
            try:
                minute = load_minute_file(path)
                daily = aggregate_to_daily(minute)
                problems = _validate_daily(daily)
                if problems:
                    skipped[ticker] = "; ".join(problems)
                    continue
                daily.insert(1, "ticker", ticker)
                frames.append(daily)
            except Exception as exc:  # noqa: BLE001 - report and continue
                skipped[ticker] = f"load error: {exc}"

    if not frames:
        raise RuntimeError("No ticker files produced a valid daily panel.")

    panel = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"Minute-aggregated panel: {panel['ticker'].nunique()} tickers, "
              f"files found {files_found}, skipped {len(skipped)}")
        for tkr, reason in list(skipped.items())[:20]:
            print(f"  skipped {tkr}: {reason}")
    return panel


def _read_daily_file(verbose: bool = True) -> pd.DataFrame:
    """Read the pre-aggregated daily long file (reshaped from Bloomberg).

    Expects a long CSV with columns:
        date, ticker, open, high, low, close, volume, daily_vwap
    where date is M/D/YYYY and ticker is already standardized (no " US Equity").
    This is the output of the reshape_ohlcv_vwap.py one-off script.

    Validation, warmup cropping, sorting and caching are handled by the shared
    code in load_all_prices(), so this function only reads and types the data.
    """
    path = config.DAILY_PRICE_SOURCE_PATH
    if not path.exists():
        raise FileNotFoundError(f"daily price file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = ["date", "ticker", "open", "high", "low", "close", "volume", "daily_vwap"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"daily file missing columns {missing}; found {list(df.columns)}")

    # date may be ISO (from reshape script) or M/D/YYYY; let pandas infer, then
    # fall back to explicit US format if needed.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().mean() > 0.5:
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["date"])

    df["ticker"] = df["ticker"].astype(str).map(standardize_ticker)
    for c in ["open", "high", "low", "close", "volume", "daily_vwap"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if verbose:
        print(f"Daily file read: {df['ticker'].nunique()} tickers, {len(df)} rows "
              f"({df['date'].min().date()} .. {df['date'].max().date()})")
    return df


def load_all_prices(use_cache: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Load the all-market daily price panel from the configured source.

    Dispatches on config.PRICE_SOURCE:
      * "minute_batches" -> aggregate minute files (original path)
      * "daily_file"     -> read a pre-aggregated Bloomberg daily pull

    Whichever source is used, the same post-processing applies: validate columns,
    sort, crop to [WARMUP_START_DATE, END_DATE], and cache to parquet. The output
    schema is identical across sources, so nothing downstream needs to change.
    """
    if use_cache and config.DAILY_PRICE_CACHE_PATH.exists():
        if verbose:
            print(f"Loading cached daily prices: {config.DAILY_PRICE_CACHE_PATH}")
        return pd.read_parquet(config.DAILY_PRICE_CACHE_PATH)

    if config.PRICE_SOURCE == "minute_batches":
        panel = _read_minute_batches(verbose=verbose)
    elif config.PRICE_SOURCE == "daily_file":
        panel = _read_daily_file(verbose=verbose)
    else:
        raise ValueError(f"unknown PRICE_SOURCE: {config.PRICE_SOURCE!r}")

    # --- shared post-processing (source-independent) ---
    required = ["date", "ticker", "open", "high", "low", "close", "volume", "daily_vwap"]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"price panel missing required columns: {missing}")

    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Restrict to the warmup window so long-lookback alphas can use any history
    # before START_DATE. Change WARMUP_START_DATE in config to admit more history.
    panel = panel[
        (panel["date"] >= pd.Timestamp(config.WARMUP_START_DATE))
        & (panel["date"] <= pd.Timestamp(config.END_DATE))
    ].reset_index(drop=True)

    if verbose:
        print(f"Daily panel ready: {panel['ticker'].nunique()} tickers, "
              f"{len(panel)} rows, source={config.PRICE_SOURCE}")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(config.DAILY_PRICE_CACHE_PATH, index=False)
    return panel


# --------------------------------------------------------------------------- #
# GICS sector / industry mapping
# --------------------------------------------------------------------------- #
def load_gics_mapping() -> pd.DataFrame:
    """Parse the GICS mapping spreadsheet into a ticker -> metadata table.

    The sheet has two Bloomberg metadata rows, then a header row
    (Ticker / Name / GICS Sector / GICS Ind Name / ...), then a junk
    "None (503 securities)" row, then data. We locate the header row by finding
    "Ticker", drop the junk row, clean tickers, and return:
    ticker, company_name, sector, industry, sub_industry.
    """
    raw = pd.read_excel(config.GICS_MAPPING_PATH, header=None)

    # Find the header row: the row whose first cell is "Ticker".
    header_row = None
    for i in range(min(10, len(raw))):
        if str(raw.iloc[i, 0]).strip().lower() == "ticker":
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not find the 'Ticker' header row in GICS mapping.")

    header = [str(c).strip() for c in raw.iloc[header_row].tolist()]
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = header

    # Map the Bloomberg column labels to our standard names.
    colmap = {
        "Ticker": "ticker",
        "Name": "company_name",
        "GICS Sector": "sector",
        "GICS Ind Name": "industry",
        "GICS SubInd Name": "sub_industry",
    }
    present = {src: dst for src, dst in colmap.items() if src in data.columns}
    data = data[list(present.keys())].rename(columns=present)

    # Drop the "None (503 securities)" junk row and any blank tickers.
    data = data[data["ticker"].notna()]
    data = data[~data["ticker"].astype(str).str.contains("securities", case=False, na=False)]
    data["ticker"] = data["ticker"].map(standardize_ticker)
    data = data[data["ticker"].astype(bool)]

    for col in ("company_name", "sector", "industry", "sub_industry"):
        if col not in data.columns:
            data[col] = ""
        data[col] = data[col].fillna("").astype(str).str.strip()

    data = data.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return data[["ticker", "company_name", "sector", "industry", "sub_industry"]]


# --------------------------------------------------------------------------- #
# Carhart factors
# --------------------------------------------------------------------------- #
def load_factors() -> pd.DataFrame:
    """Load the Carhart 4-factor daily table in decimal scale.

    Columns standardized to: date, mkt_rf, smb, hml, rf, mom.
    """
    raw = pd.read_csv(config.FACTOR_PATH)
    raw.columns = _normalize_columns(raw.columns)  # Mkt-RF -> mkt_rf, Mom -> mom

    if "date" not in raw.columns:
        raise ValueError(f"Factor file has no 'date' column; got {list(raw.columns)}")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()

    expected = ["mkt_rf", "smb", "hml", "rf", "mom"]
    missing = [c for c in expected if c not in raw.columns]
    if missing:
        raise ValueError(f"Factor file missing columns {missing}; got {list(raw.columns)}")
    for col in expected:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Sanity: values must be in decimal scale (|.| < 1), not percent.
    if raw[expected].abs().to_numpy(dtype=float).max() >= 1.0:
        raise ValueError("Factor values look like percent (>=1); expected decimal scale.")

    raw = raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return raw[["date", *expected]]


# --------------------------------------------------------------------------- #
# SPX market return (for alpha 60 market beta)
# --------------------------------------------------------------------------- #
def load_spx_market_return() -> pd.DataFrame:
    """Load SPX daily close and derive market log-returns.

    The file has a Bloomberg metadata row first ("SPX Index" in col B), then the
    real header on the second row: Dates | PX_HIGH | PX_LOW | PX_LAST. We read
    with header=1 so the real header is used, then take PX_LAST as the close.

    Returns columns: date, market_return (log return of SPX close).
    """
    raw = pd.read_excel(config.SPX_PATH, header=1)
    raw.columns = _normalize_columns(raw.columns)

    date_col = next((c for c in raw.columns if "date" in c), None)
    close_col = next(
        (c for c in raw.columns if c in ("px_last", "last", "close", "spx_index", "spx") or "close" in c),
        None,
    )
    if date_col is None or close_col is None:
        raise ValueError(f"SPX file: could not find date/close columns in {list(raw.columns)}")

    spx = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce").dt.normalize(),
            "spx_close": pd.to_numeric(raw[close_col], errors="coerce"),
        }
    )
    spx = spx.dropna().sort_values("date").reset_index(drop=True)
    spx["market_return"] = np.log(spx["spx_close"] / spx["spx_close"].shift(1))
    return spx[["date", "market_return"]]


# --------------------------------------------------------------------------- #
# Float shares (wide -> long, millions -> shares)
# --------------------------------------------------------------------------- #
def load_float_shares() -> pd.DataFrame:
    """Load the float-shares wide sheet into a long, point-in-time table.

    The sheet is dates down column A and one ticker per remaining column, with
    the ticker symbols ("WBD US Equity", ...) on row 1 and values in millions
    of shares. Returns long columns: date, ticker, float_shares (absolute).
    """
    raw = pd.read_excel(config.FLOAT_PATH, header=0)
    raw.columns = [str(c).strip() for c in raw.columns]

    date_col = raw.columns[0]  # first column holds the dates ("Dates")
    long = raw.melt(id_vars=[date_col], var_name="ticker", value_name="float_shares")
    long = long.rename(columns={date_col: "date"})

    long["date"] = pd.to_datetime(long["date"], errors="coerce").dt.normalize()
    long["ticker"] = long["ticker"].map(standardize_ticker)
    long["float_shares"] = pd.to_numeric(long["float_shares"], errors="coerce") * config.FLOAT_UNIT_MULTIPLIER

    long = long.dropna(subset=["date", "ticker", "float_shares"])
    long = long[long["float_shares"] > 0]
    long = long.sort_values(["ticker", "date"]).reset_index(drop=True)
    return long[["date", "ticker", "float_shares"]]