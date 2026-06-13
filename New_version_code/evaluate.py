"""Single-feature Rank-IC evaluation for feature selection.

For each raw alpha, this measures predictive power against the 5-day forward
return using the Rank Information Coefficient (Rank IC):

  * build ret_fwd_5d per ticker (log return over the next 5 trading days);
  * for each trading date, compute the Spearman correlation between the alpha's
    cross-section and that day's forward returns (this is the day's Rank IC);
  * summarize each alpha's daily IC series into mean_ic, std_ic, icir, t_stat.

Critically, the IC is computed on the TRAIN segment only (dates on or before
``config.TRAIN_END_DATE``). Validation and test dates never enter feature
selection, so the test set stays a clean out-of-sample.

Why Spearman on the raw alpha: Spearman(x, y) = Pearson(rank(x), rank(y)), so
feeding raw alpha values is identical to ranking first. The cross_section rank
columns are for the ranker; evaluation does its own ranking via Spearman and
does not depend on them.

t_stat = icir * sqrt(n_days). Selection later keeps |t_stat| >= threshold, so
both positive (continuation) and negative (reversal) alphas survive if they are
significant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, io_data


def add_forward_return(panel: pd.DataFrame, price_panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the 5-day forward EXCESS return target to the panel.

    Matches the team's definition (Step 6 build_forward_return_labels):

        forward_5d_excess_return(t) = log(close(t+H) / close(t))
                                      - log(SPX(t+H) / SPX(t))

    i.e. the stock's own H-day forward log return minus the market's (SPX) H-day
    forward log return over the same window. The stock leg is computed within
    each ticker (no cross-ticker contamination); the market leg is a single SPX
    series broadcast by date. The last H rows per ticker are NaN (no future), as
    are dates whose t+H SPX value is unavailable.

    The output column name is ``config.TARGET_COLUMN``
    (``forward_5d_excess_return``).
    """
    h = config.FORWARD_RETURN_HORIZON

    # --- stock leg: log(close(t+H)/close(t)) within each ticker ---
    prices = price_panel[["date", "ticker", "close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices = prices.sort_values(["ticker", "date"])
    close = pd.to_numeric(prices["close"], errors="coerce")
    stock_fwd = np.log(prices.groupby("ticker")["close"].shift(-h) / close)
    prices["_stock_fwd"] = stock_fwd.to_numpy()

    # --- market leg: log(SPX(t+H)/SPX(t)), one series by date ---
    spx = io_data.load_spx_close()
    spx["date"] = pd.to_datetime(spx["date"]).dt.normalize()
    spx = spx.sort_values("date").reset_index(drop=True)
    spx_close = pd.to_numeric(spx["spx_close"], errors="coerce")
    spx["_mkt_fwd"] = np.log(spx_close.shift(-h) / spx_close).to_numpy()
    mkt_fwd = spx[["date", "_mkt_fwd"]]

    # --- excess = stock_fwd - mkt_fwd ---
    prices = prices.merge(mkt_fwd, on="date", how="left")
    prices[config.TARGET_COLUMN] = prices["_stock_fwd"] - prices["_mkt_fwd"]

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.merge(
        prices[["date", "ticker", config.TARGET_COLUMN]],
        on=["date", "ticker"],
        how="left",
    )
    return out


def _train_mask(dates: pd.Series) -> pd.Series:
    """Boolean mask for dates on or before the train cutoff."""
    d = pd.to_datetime(dates).dt.normalize()
    return d <= pd.Timestamp(config.TRAIN_END_DATE)


def _daily_rank_ic(group: pd.DataFrame, alpha_col: str, target_col: str) -> float:
    """Spearman corr of one alpha vs forward return for a single date.

    Returns NaN if fewer than MIN_CROSS_SECTIONAL_COUNT valid pairs.
    """
    pair = group[[alpha_col, target_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < config.MIN_CROSS_SECTIONAL_COUNT:
        return np.nan
    if pair[alpha_col].nunique() < 2 or pair[target_col].nunique() < 2:
        return np.nan
    return pair[alpha_col].corr(pair[target_col], method="spearman")


def compute_rank_ic(panel: pd.DataFrame, train_only: bool = True) -> pd.DataFrame:
    """Compute the Rank-IC summary table, one row per alpha.

    Expects ``panel`` to already contain ``config.TARGET_COLUMN`` (call
    ``add_forward_return`` first) and the raw alpha columns. By default only the
    train segment is used.

    Returns columns: alpha, mean_ic, std_ic, icir, t_stat, n_days.
    """
    if config.TARGET_COLUMN not in panel.columns:
        raise ValueError(f"panel is missing target column {config.TARGET_COLUMN!r}; call add_forward_return first.")

    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    if train_only:
        work = work[_train_mask(work["date"])].copy()

    alpha_cols = [c for c in config.ALPHA_COLUMNS if c in work.columns]
    target = config.TARGET_COLUMN

    # Daily IC per alpha: group once by date, correlate each alpha column.
    rows: list[dict[str, float]] = []
    grouped = list(work.groupby("date", sort=True))

    for alpha_col in alpha_cols:
        daily_ics: list[float] = []
        for _, group in grouped:
            ic = _daily_rank_ic(group, alpha_col, target)
            if not np.isnan(ic):
                daily_ics.append(ic)

        ic_series = pd.Series(daily_ics, dtype=float)
        n_days = int(ic_series.notna().sum())
        if n_days == 0:
            rows.append({"alpha": alpha_col, "mean_ic": np.nan, "std_ic": np.nan,
                         "icir": np.nan, "t_stat": np.nan, "n_days": 0})
            continue

        mean_ic = float(ic_series.mean())
        std_ic = float(ic_series.std(ddof=1)) if n_days > 1 else np.nan
        if std_ic is None or not np.isfinite(std_ic) or std_ic == 0:
            icir = np.nan
            t_stat = np.nan
        else:
            icir = mean_ic / std_ic
            t_stat = icir * np.sqrt(n_days)

        rows.append({
            "alpha": alpha_col,
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "icir": icir,
            "t_stat": t_stat,
            "n_days": n_days,
        })

    summary = pd.DataFrame(rows)
    # Sort by absolute t-stat descending so the strongest alphas are on top.
    summary = summary.reindex(
        summary["t_stat"].abs().sort_values(ascending=False, na_position="last").index
    ).reset_index(drop=True)
    return summary