from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PRICE_VOLUME_DIR = Path(r"F:\icarus fund\Quantitative AI financial modeling\数据集\sp500\Archive")
ALPHA_PANEL_PATH = Path(r"F:\icarus fund\Quantitative AI financial modeling\regression\sp500_alpha_panel_raw.csv")
OUTPUT_DIR = Path(r"F:\icarus fund\Quantitative AI financial modeling\regression")


OUTPUT_PANEL_CSV = OUTPUT_DIR / "sp500_alpha_target_panel.csv"
OUTPUT_PANEL_PARQUET = OUTPUT_DIR / "sp500_alpha_target_panel.parquet"
OUTPUT_ALPHA_MISSING = OUTPUT_DIR / "alpha_missing_summary.csv"
OUTPUT_QUALITY = OUTPUT_DIR / "panel_data_quality_report.csv"
OUTPUT_UNMATCHED = OUTPUT_DIR / "unmatched_alpha_price_rows.csv"
OUTPUT_FEATURES = OUTPUT_DIR / "feature_column_list.txt"
OUTPUT_TARGETS = OUTPUT_DIR / "target_column_list.txt"
OUTPUT_RANK_IC_1D = OUTPUT_DIR / "alpha_rank_ic_summary_1d.csv"
OUTPUT_RANK_IC_5D = OUTPUT_DIR / "alpha_rank_ic_summary_5d.csv"
OUTPUT_REG_1D = OUTPUT_DIR / "single_alpha_regression_summary_1d.csv"
OUTPUT_REG_5D = OUTPUT_DIR / "single_alpha_regression_summary_5d.csv"


SUPPORTED_EXTENSIONS = {".csv", ".txt", ".tsv", ".xlsx", ".xls"}
OHLCV_FIELDS = ["open", "high", "low", "close", "volume", "adjusted_close"]
TARGET_COLUMNS = ["ret_fwd_1d", "ret_fwd_5d"]


def standardize_column_name(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[\s\-/]+", "_", text)
    text = re.sub(r"[^0-9a-zA-Z_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    aliases = {
        "symbol": "ticker",
        "tic": "ticker",
        "security": "ticker",
        "datetime": "date",
        "timestamp": "date",
        "trade_date": "date",
        "adj_close": "adjusted_close",
        "adjusted_close": "adjusted_close",
        "adjustedclose": "adjusted_close",
        "adjclose": "adjusted_close",
        "close_adjusted": "adjusted_close",
        "open_price": "open",
        "high_price": "high",
        "low_price": "low",
        "close_price": "close",
        "last": "close",
        "vol": "volume",
    }
    return aliases.get(text, text)


def ensure_paths() -> None:
    if not ALPHA_PANEL_PATH.exists():
        raise FileNotFoundError(f"Alpha panel not found: {ALPHA_PANEL_PATH}")
    if not PRICE_VOLUME_DIR.exists():
        raise FileNotFoundError(f"Price-volume folder not found: {PRICE_VOLUME_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_alpha_panel(path: Path = ALPHA_PANEL_PATH) -> tuple[pd.DataFrame, list[str]]:
    print(f"\n[1/10] Loading alpha panel: {path}")
    df = pd.read_csv(path)
    df.columns = [standardize_column_name(c) for c in df.columns]

    missing_keys = {"date", "ticker"} - set(df.columns)
    if missing_keys:
        raise ValueError(f"Alpha panel is missing required key columns: {sorted(missing_keys)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["date", "ticker"])
    df = df[df["ticker"] != ""]
    before = len(df)
    df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    duplicate_count = before - len(df)

    alpha_cols = [c for c in df.columns if c.startswith("alpha_")]
    print(f"Alpha panel shape: {df.shape}")
    print(f"Alpha duplicate ticker-date rows removed: {duplicate_count}")
    print(f"Alpha unique tickers: {df['ticker'].nunique()}")
    print(f"Alpha unique dates: {df['date'].nunique()}")
    print(f"Alpha date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Alpha column count: {len(alpha_cols)}")
    return df, alpha_cols


def _read_price_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def _numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _normalize_price_file(raw: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [standardize_column_name(c) for c in df.columns]

    if "date" not in df.columns:
        raise ValueError("No date-like column detected after standardization")

    if "ticker" not in df.columns:
        df["ticker"] = path.stem.upper()
    else:
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    _numeric_columns(df, OHLCV_FIELDS)

    required = ["date", "ticker", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required standardized columns: {missing}")

    keep_cols = ["date", "ticker"] + [c for c in OHLCV_FIELDS if c in df.columns]
    df = df[keep_cols]
    df = df.dropna(subset=["date", "ticker", "close"])
    df = df[df["ticker"] != ""]
    df = df[df["close"] > 0]
    if "adjusted_close" in df.columns:
        df = df[df["adjusted_close"].isna() | (df["adjusted_close"] > 0)]

    if df.empty:
        return df

    aggregation = {}
    if "open" in df.columns:
        aggregation["open"] = "first"
    if "high" in df.columns:
        aggregation["high"] = "max"
    if "low" in df.columns:
        aggregation["low"] = "min"
    if "close" in df.columns:
        aggregation["close"] = "last"
    if "volume" in df.columns:
        aggregation["volume"] = "sum"
    if "adjusted_close" in df.columns:
        aggregation["adjusted_close"] = "last"

    return (
        df.sort_values(["ticker", "date"])
        .groupby(["ticker", "date"], as_index=False, sort=False)
        .agg(aggregation)
    )


def load_price_volume_panel(folder: Path = PRICE_VOLUME_DIR) -> pd.DataFrame:
    print(f"\n[2/10] Loading price-volume files: {folder}")
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No supported price-volume files found in {folder}")

    panels: list[pd.DataFrame] = []
    skipped: list[tuple[str, str]] = []
    for idx, path in enumerate(files, start=1):
        try:
            raw = _read_price_file(path)
            panel = _normalize_price_file(raw, path)
            if not panel.empty:
                panels.append(panel)
        except Exception as exc:
            skipped.append((path.name, str(exc)))
        if idx % 50 == 0 or idx == len(files):
            print(f"  Processed {idx}/{len(files)} files; usable panels so far: {len(panels)}")

    if not panels:
        raise ValueError("No usable price-volume files were loaded.")
    if skipped:
        print(f"Skipped {len(skipped)} unsupported or invalid price files.")
        for name, reason in skipped[:10]:
            print(f"  skipped {name}: {reason}")
        if len(skipped) > 10:
            print("  ... additional skipped files omitted from console output")

    df = pd.concat(panels, ignore_index=True)
    before = len(df)
    df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    df = df.dropna(subset=["date", "ticker", "close"])
    df = df[df["close"] > 0]
    if "adjusted_close" in df.columns:
        df = df[df["adjusted_close"].isna() | (df["adjusted_close"] > 0)]

    print(f"Price duplicate ticker-date rows removed after combine: {before - len(df)}")
    print(f"Price panel shape: {df.shape}")
    print(f"Price unique tickers: {df['ticker'].nunique()}")
    print(f"Price unique dates: {df['date'].nunique()}")
    print(f"Price date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Price columns: {list(df.columns)}")
    return df


def merge_alpha_price(alpha_df: pd.DataFrame, price_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    print("\n[3/10] Merging alpha panel with OHLCV panel")
    price_cols = ["date", "ticker"] + [c for c in OHLCV_FIELDS if c in price_df.columns]
    merged = alpha_df.merge(price_df[price_cols], on=["date", "ticker"], how="left", indicator="_price_merge")
    matched_mask = merged["_price_merge"].eq("both")
    matched_count = int(matched_mask.sum())
    unmatched = merged.loc[~matched_mask, alpha_df.columns].copy()
    unmatched_count = len(unmatched)
    unmatched_pct = unmatched_count / len(merged) if len(merged) else np.nan
    merged = merged.drop(columns=["_price_merge"])

    print(f"Merged panel shape: {merged.shape}")
    print(f"Matched OHLCV row count: {matched_count}")
    print(f"Unmatched OHLCV row count: {unmatched_count}")
    print(f"Unmatched OHLCV percentage: {unmatched_pct:.2%}")
    return merged, unmatched, unmatched_count


def add_forward_returns(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    print("\n[4/10] Constructing forward return targets")
    out = df.sort_values(["ticker", "date"]).copy()
    price_col = "adjusted_close" if "adjusted_close" in out.columns and out["adjusted_close"].notna().any() else "close"
    if price_col not in out.columns:
        raise ValueError("Neither adjusted_close nor close exists for target construction.")

    price = pd.to_numeric(out[price_col], errors="coerce")
    out[price_col] = price
    next_1 = out.groupby("ticker", sort=False)[price_col].shift(-1)
    next_5 = out.groupby("ticker", sort=False)[price_col].shift(-5)
    valid_price = price.gt(0)
    out["ret_fwd_1d"] = np.where(valid_price & next_1.gt(0), np.log(next_1 / price), np.nan)
    out["ret_fwd_5d"] = np.where(valid_price & next_5.gt(0), np.log(next_5 / price), np.nan)
    print(f"Price column used for returns: {price_col}")
    print(f"Valid ret_fwd_1d rows: {out['ret_fwd_1d'].notna().sum()}")
    print(f"Valid ret_fwd_5d rows: {out['ret_fwd_5d'].notna().sum()}")
    return out, price_col


def summarize_alpha_missing(df: pd.DataFrame, alpha_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in alpha_cols:
        valid = int(df[col].notna().sum()) if col in df.columns else 0
        missing_ratio = float(df[col].isna().mean()) if col in df.columns and len(df) else 1.0
        rows.append(
            {
                "alpha_name": col,
                "missing_ratio": missing_ratio,
                "valid_observations": valid,
                "all_nan_flag": valid == 0,
                "high_missing_flag": missing_ratio > 0.50,
            }
        )
    return pd.DataFrame(rows)


def _process_cross_section(group: pd.Series) -> pd.DataFrame:
    valid = group.dropna()
    if len(valid) < 50:
        return pd.DataFrame({"z": np.nan, "rank": np.nan}, index=group.index)

    p01 = valid.quantile(0.01)
    p99 = valid.quantile(0.99)
    clipped = group.clip(lower=p01, upper=p99)
    mean = clipped.mean(skipna=True)
    std = clipped.std(skipna=True, ddof=0)
    rank = clipped.rank(method="average", pct=True)
    if pd.isna(std) or std == 0:
        z = pd.Series(np.nan, index=group.index)
    else:
        z = (clipped - mean) / std
    return pd.DataFrame({"z": z, "rank": rank}, index=group.index)


def cross_sectional_winsorize_zscore_rank(
    df: pd.DataFrame, alpha_cols: list[str]
) -> tuple[pd.DataFrame, list[str], list[str], pd.DataFrame]:
    print("\n[5/10] Summarizing alpha missingness")
    out = df.copy()
    missing_summary = summarize_alpha_missing(out, alpha_cols)
    print(f"Alpha columns entirely NaN: {int(missing_summary['all_nan_flag'].sum())}")
    print(f"Alpha columns >50% missing: {int(missing_summary['high_missing_flag'].sum())}")

    print("\n[6/10] Cross-sectional winsorization, z-score, and percentile rank by date")
    z_cols: list[str] = []
    rank_cols: list[str] = []
    date_indices = list(out.groupby("date", sort=False).indices.values())
    for idx, col in enumerate(alpha_cols, start=1):
        out[col] = pd.to_numeric(out[col], errors="coerce")
        z_col = f"{col}_z"
        rank_col = f"{col}_rank"
        values = out[col].to_numpy(dtype=float)
        z_values = np.full(len(out), np.nan, dtype=float)
        rank_values = np.full(len(out), np.nan, dtype=float)

        for positions in date_indices:
            vals = values[positions]
            valid_mask = ~np.isnan(vals)
            if int(valid_mask.sum()) < 50:
                continue
            valid_vals = vals[valid_mask]
            p01, p99 = np.nanpercentile(valid_vals, [1, 99])
            clipped = vals.copy()
            clipped[valid_mask] = np.clip(valid_vals, p01, p99)
            clipped_valid = clipped[valid_mask]
            std = float(np.nanstd(clipped_valid, ddof=0))
            if std > 0 and not np.isnan(std):
                z_values[positions[valid_mask]] = (clipped_valid - float(np.nanmean(clipped_valid))) / std
            rank_values[positions[valid_mask]] = (
                pd.Series(clipped_valid).rank(method="average", pct=True).to_numpy(dtype=float)
            )

        out[z_col] = z_values
        out[rank_col] = rank_values
        z_cols.append(z_col)
        rank_cols.append(rank_col)
        if idx % 10 == 0 or idx == len(alpha_cols):
            print(f"  Processed {idx}/{len(alpha_cols)} alpha columns")

    return out, z_cols, rank_cols, missing_summary


def generate_quality_reports(
    panel: pd.DataFrame,
    alpha_cols: list[str],
    z_cols: list[str],
    rank_cols: list[str],
    missing_summary: pd.DataFrame,
    unmatched_count: int,
) -> pd.DataFrame:
    print("\n[7/10] Generating data quality report")
    duplicate_count = int(panel.duplicated(["ticker", "date"]).sum())
    metrics: list[dict[str, object]] = [
        {"metric": "final_row_count", "value": len(panel)},
        {"metric": "number_of_tickers", "value": panel["ticker"].nunique()},
        {"metric": "number_of_dates", "value": panel["date"].nunique()},
        {"metric": "date_min", "value": panel["date"].min()},
        {"metric": "date_max", "value": panel["date"].max()},
        {"metric": "number_of_raw_alpha_columns", "value": len(alpha_cols)},
        {"metric": "number_of_z_score_alpha_columns", "value": len(z_cols)},
        {"metric": "number_of_rank_alpha_columns", "value": len(rank_cols)},
        {"metric": "valid_ret_fwd_1d_rows", "value": int(panel["ret_fwd_1d"].notna().sum())},
        {"metric": "valid_ret_fwd_5d_rows", "value": int(panel["ret_fwd_5d"].notna().sum())},
        {"metric": "duplicate_ticker_date_count", "value": duplicate_count},
        {"metric": "unmatched_ohlcv_row_count", "value": unmatched_count},
        {"metric": "alpha_columns_entirely_nan", "value": int(missing_summary["all_nan_flag"].sum())},
        {"metric": "alpha_columns_more_than_50pct_missing", "value": int(missing_summary["high_missing_flag"].sum())},
    ]

    for col in OHLCV_FIELDS:
        if col in panel.columns:
            metrics.append({"metric": f"missing_ratio_{col}", "value": float(panel[col].isna().mean())})
        else:
            metrics.append({"metric": f"missing_ratio_{col}", "value": "column_not_present"})

    return pd.DataFrame(metrics)


def run_rank_ic_validation(panel: pd.DataFrame, z_cols: list[str], target_col: str) -> pd.DataFrame:
    print(f"\n[9/10] Running daily cross-sectional Rank IC validation for {target_col}")
    rows: list[dict[str, object]] = []
    date_indices = list(panel.groupby("date", sort=False).indices.values())
    target_values = panel[target_col].to_numpy(dtype=float)
    for idx, alpha_col in enumerate(z_cols, start=1):
        alpha_values = panel[alpha_col].to_numpy(dtype=float)
        ic_values: list[float] = []
        for positions in date_indices:
            x = alpha_values[positions]
            y = target_values[positions]
            valid = ~(np.isnan(x) | np.isnan(y))
            if int(valid.sum()) < 50:
                continue
            x_valid = x[valid]
            y_valid = y[valid]
            if len(np.unique(x_valid)) < 2 or len(np.unique(y_valid)) < 2:
                continue
            ic = spearmanr(x_valid, y_valid).correlation
            if not np.isnan(ic):
                ic_values.append(float(ic))
        ic_array = np.array(ic_values, dtype=float)
        n_days = int(len(ic_array))
        mean_ic = float(np.mean(ic_array)) if n_days else np.nan
        std_ic = float(np.std(ic_array, ddof=1)) if n_days > 1 else np.nan
        rows.append(
            {
                "alpha": alpha_col,
                "target": target_col,
                "n_days": n_days,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "icir": mean_ic / std_ic if std_ic and not pd.isna(std_ic) else np.nan,
                "positive_ic_ratio": float(np.mean(ic_array > 0)) if n_days else np.nan,
                "t_stat": mean_ic / (std_ic / math.sqrt(n_days)) if std_ic and n_days > 1 else np.nan,
            }
        )
        if idx % 10 == 0 or idx == len(z_cols):
            print(f"  Rank IC processed {idx}/{len(z_cols)} alpha columns for {target_col}")
    return pd.DataFrame(rows)


def run_single_alpha_regression(panel: pd.DataFrame, z_cols: list[str], target_col: str) -> pd.DataFrame:
    print(f"\n[10/10] Running single-alpha daily cross-sectional OLS for {target_col}")
    rows: list[dict[str, object]] = []
    date_indices = list(panel.groupby("date", sort=False).indices.values())
    target_values = panel[target_col].to_numpy(dtype=float)
    for idx, alpha_col in enumerate(z_cols, start=1):
        alpha_values = panel[alpha_col].to_numpy(dtype=float)
        betas: list[float] = []
        r2_values: list[float] = []
        for positions in date_indices:
            x = alpha_values[positions]
            y = target_values[positions]
            valid = ~(np.isnan(x) | np.isnan(y))
            if int(valid.sum()) < 50:
                continue
            x_valid = x[valid]
            y_valid = y[valid]
            x_var = float(np.var(x_valid))
            if x_var <= 0 or len(np.unique(x_valid)) < 2:
                continue
            beta_value = float(np.cov(x_valid, y_valid, ddof=0)[0, 1] / x_var)
            intercept = float(np.mean(y_valid) - beta_value * np.mean(x_valid))
            fitted = intercept + beta_value * x_valid
            sst = float(np.sum((y_valid - np.mean(y_valid)) ** 2))
            sse = float(np.sum((y_valid - fitted) ** 2))
            betas.append(beta_value)
            if sst > 0:
                r2_values.append(1.0 - sse / sst)

        beta = np.array(betas, dtype=float)
        r2 = np.array(r2_values, dtype=float)
        n_days = int(len(beta))
        mean_beta = float(np.mean(beta)) if n_days else np.nan
        std_beta = float(np.std(beta, ddof=1)) if n_days > 1 else np.nan
        rows.append(
            {
                "alpha": alpha_col,
                "n_days": n_days,
                "mean_beta": mean_beta,
                "std_beta": std_beta,
                "beta_t_stat": mean_beta / (std_beta / math.sqrt(n_days)) if std_beta and n_days > 1 else np.nan,
                "positive_beta_ratio": float(np.mean(beta > 0)) if n_days else np.nan,
                "avg_daily_r2": float(np.mean(r2)) if len(r2) else np.nan,
            }
        )
        if idx % 10 == 0 or idx == len(z_cols):
            print(f"  Regression processed {idx}/{len(z_cols)} alpha columns for {target_col}")
    return pd.DataFrame(rows)


def save_text_list(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def save_outputs(
    panel: pd.DataFrame,
    unmatched: pd.DataFrame,
    missing_summary: pd.DataFrame,
    quality_report: pd.DataFrame,
    feature_cols: list[str],
    rank_ic_1d: pd.DataFrame,
    rank_ic_5d: pd.DataFrame,
    reg_1d: pd.DataFrame,
    reg_5d: pd.DataFrame,
) -> None:
    print("\n[8/10] Saving panel and report outputs")
    panel.to_csv(OUTPUT_PANEL_CSV, index=False)
    try:
        panel.to_parquet(OUTPUT_PANEL_PARQUET, index=False)
    except Exception as exc:
        raise RuntimeError(
            "Unable to save parquet output. Install pyarrow or fastparquet, then rerun this script. "
            f"CSV output was written before this failure. Details: {exc}"
        ) from exc

    unmatched.to_csv(OUTPUT_UNMATCHED, index=False)
    missing_summary.to_csv(OUTPUT_ALPHA_MISSING, index=False)
    quality_report.to_csv(OUTPUT_QUALITY, index=False)
    save_text_list(OUTPUT_FEATURES, feature_cols)
    save_text_list(OUTPUT_TARGETS, TARGET_COLUMNS)
    rank_ic_1d.to_csv(OUTPUT_RANK_IC_1D, index=False)
    rank_ic_5d.to_csv(OUTPUT_RANK_IC_5D, index=False)
    reg_1d.to_csv(OUTPUT_REG_1D, index=False)
    reg_5d.to_csv(OUTPUT_REG_5D, index=False)


def print_final_summary(panel: pd.DataFrame, alpha_cols: list[str], feature_cols: list[str]) -> None:
    print("\nFinished building S&P 500 alpha-target panel.")
    print(f"Final panel shape: {panel.shape}")
    print(f"Number of tickers: {panel['ticker'].nunique()}")
    print(f"Number of dates: {panel['date'].nunique()}")
    print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"Number of alpha columns: {len(alpha_cols)}")
    print(f"Number of feature columns: {len(feature_cols)}")
    print(f"Valid ret_fwd_1d rows: {panel['ret_fwd_1d'].notna().sum()}")
    print(f"Valid ret_fwd_5d rows: {panel['ret_fwd_5d'].notna().sum()}")
    print("Output file paths:")
    for path in [
        OUTPUT_PANEL_CSV,
        OUTPUT_PANEL_PARQUET,
        OUTPUT_ALPHA_MISSING,
        OUTPUT_QUALITY,
        OUTPUT_UNMATCHED,
        OUTPUT_FEATURES,
        OUTPUT_TARGETS,
        OUTPUT_RANK_IC_1D,
        OUTPUT_RANK_IC_5D,
        OUTPUT_REG_1D,
        OUTPUT_REG_5D,
    ]:
        print(f"  {path}")


def main() -> None:
    ensure_paths()
    alpha_df, alpha_cols = load_alpha_panel()
    price_df = load_price_volume_panel()
    merged, unmatched, unmatched_count = merge_alpha_price(alpha_df, price_df)
    merged, _price_col = add_forward_returns(merged)
    panel, z_cols, rank_cols, missing_summary = cross_sectional_winsorize_zscore_rank(merged, alpha_cols)
    feature_cols = z_cols + rank_cols
    quality_report = generate_quality_reports(panel, alpha_cols, z_cols, rank_cols, missing_summary, unmatched_count)
    rank_ic_1d = run_rank_ic_validation(panel, z_cols, "ret_fwd_1d")
    rank_ic_5d = run_rank_ic_validation(panel, z_cols, "ret_fwd_5d")
    reg_1d = run_single_alpha_regression(panel, z_cols, "ret_fwd_1d")
    reg_5d = run_single_alpha_regression(panel, z_cols, "ret_fwd_5d")
    save_outputs(panel, unmatched, missing_summary, quality_report, feature_cols, rank_ic_1d, rank_ic_5d, reg_1d, reg_5d)
    print_final_summary(panel, alpha_cols, feature_cols)


if __name__ == "__main__":
    main()
