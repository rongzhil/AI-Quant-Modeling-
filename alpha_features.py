"""Raw price/volume alpha IDs 26-75 for one ticker at a time.

The public entry point is ``compute_alpha_features``. It expects one
standardized daily OHLCV table for one ticker and returns the 50 raw alpha
columns. All rolling windows are computed inside that one ticker table, which
prevents cross-ticker leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import config


def _col(alpha_id: int) -> str:
    return config.ALPHA_COLUMN_BY_ID[alpha_id]


def _rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return (series - mean) / std.replace(0, np.nan)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denom = float((x_centered**2).sum())

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y = values.astype(float)
        return float(np.dot(x_centered, y - y.mean()) / denom)

    return series.rolling(window, min_periods=window).apply(slope, raw=True)


def _rolling_regression_residuals(y: pd.Series, x: pd.DataFrame, window: int = 252) -> pd.Series:
    """Residual at each date from rolling OLS of y on one or more factors."""
    residual = pd.Series(np.nan, index=y.index, dtype=float)
    if x.empty:
        return residual
    for end in range(window - 1, len(y)):
        start = end - window + 1
        frame = pd.concat([y.iloc[start : end + 1].rename("y"), x.iloc[start : end + 1]], axis=1).dropna()
        if len(frame) < window or frame.index[-1] != y.index[end]:
            continue
        design = np.column_stack([np.ones(len(frame)), frame[x.columns].to_numpy(dtype=float)])
        target = frame["y"].to_numpy(dtype=float)
        beta = np.linalg.lstsq(design, target, rcond=None)[0]
        current_x = np.r_[1.0, frame[x.columns].iloc[-1].to_numpy(dtype=float)]
        residual.iloc[end] = target[-1] - float(current_x @ beta)
    return residual


def _casefold_text(value: Any) -> str:
    return str(value).strip().casefold()


def _aligned_market_return(daily: pd.DataFrame, aux_data: dict[str, pd.DataFrame]) -> pd.Series | None:
    market = aux_data.get("market_return")
    if market is None:
        return None
    merged = daily[["date"]].merge(market[["date", "market_return"]], on="date", how="left")
    return pd.to_numeric(merged["market_return"], errors="coerce").reset_index(drop=True)


def _factor_matrix(daily: pd.DataFrame, aux_data: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    frames: list[pd.DataFrame | pd.Series] = []
    market = _aligned_market_return(daily, aux_data)
    if market is not None:
        frames.append(market.rename("market_return"))

    factors = aux_data.get("factor_returns")
    if factors is not None:
        factor_merged = daily[["date"]].merge(factors, on="date", how="left")
        frames.append(factor_merged.drop(columns=["date"]).reset_index(drop=True))

    if not frames:
        return None
    matrix = pd.concat(frames, axis=1)
    numeric_cols = []
    for col in matrix.columns:
        matrix[col] = pd.to_numeric(matrix[col], errors="coerce")
        if matrix[col].notna().any():
            numeric_cols.append(col)
    return matrix[numeric_cols] if numeric_cols else None


def _group_return_series(
    daily: pd.DataFrame,
    aux_data: dict[str, pd.DataFrame],
    aux_key: str,
    group_name: str,
    value_column: str,
) -> pd.Series | None:
    group_returns = aux_data.get(aux_key)
    if group_returns is None or not group_name:
        return None
    group_key = _casefold_text(group_name)
    filtered = group_returns[group_returns["group"].map(_casefold_text) == group_key]
    if filtered.empty:
        return None
    merged = daily[["date"]].merge(filtered[["date", value_column]], on="date", how="left")
    return pd.to_numeric(merged[value_column], errors="coerce").reset_index(drop=True)


def _float_shares_series(daily: pd.DataFrame, ticker: str, aux_data: dict[str, pd.DataFrame]) -> pd.Series | None:
    shares = aux_data.get("float_shares")
    if shares is None or shares.empty:
        return None
    data = shares.copy()
    if "ticker" in data.columns:
        data = data[data["ticker"].astype(str).str.upper() == ticker.upper()]
        if data.empty:
            return None
    if "date" not in data.columns:
        value = pd.to_numeric(data["float_shares"], errors="coerce").dropna()
        if value.empty:
            return None
        return pd.Series(float(value.iloc[0]), index=daily.index)

    left = pd.DataFrame({"date": daily["date"].astype(str), "_order": np.arange(len(daily))})
    left["_merge_date"] = pd.to_datetime(left["date"], errors="coerce")
    right = data[["date", "float_shares"]].copy()
    right["_merge_date"] = pd.to_datetime(right["date"], errors="coerce")
    right = right.dropna(subset=["_merge_date"]).sort_values("_merge_date")
    if right.empty:
        return None
    merged = pd.merge_asof(
        left.sort_values("_merge_date"),
        right[["_merge_date", "float_shares"]],
        on="_merge_date",
        direction="backward",
    )
    # Keep dates after the auxiliary file ends as NaN, as requested.
    last_aux_date = right["_merge_date"].max()
    merged.loc[merged["_merge_date"] > last_aux_date, "float_shares"] = np.nan
    return pd.to_numeric(merged.sort_values("_order")["float_shares"], errors="coerce").reset_index(drop=True)


def compute_alpha_features(
    daily: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    aux_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Compute raw alpha IDs 26-75 for one standardized daily ticker table.

    Parameters
    ----------
    daily:
        One ticker's standardized daily table with date, ticker, open, high,
        low, close, volume, and daily_vwap.
    metadata:
        Optional universe metadata. The sector and industry fields are used
        only when matching sector/industry auxiliary return files.
    aux_data:
        Optional auxiliary data dictionary loaded by ``io_utils``. Missing
        auxiliary data leaves alpha IDs 33, 34, 35, 59, 60, and 66 as NaN.
    """
    metadata = metadata or {}
    aux_data = aux_data or {}
    base = daily.sort_values("date").reset_index(drop=True).copy()

    out = pd.DataFrame({"date": base["date"].astype(str)})
    ticker = str(base["ticker"].iloc[0])
    close = pd.to_numeric(base["close"], errors="coerce")
    open_ = pd.to_numeric(base["open"], errors="coerce")
    high = pd.to_numeric(base["high"], errors="coerce")
    low = pd.to_numeric(base["low"], errors="coerce")
    volume = pd.to_numeric(base["volume"], errors="coerce")
    daily_vwap = pd.to_numeric(base["daily_vwap"], errors="coerce")

    # Returns and momentum.
    for alpha_id, lookback in [(26, 1), (27, 5), (28, 21), (29, 63), (30, 126), (31, 252)]:
        out[_col(alpha_id)] = np.log(close / close.shift(lookback))
    r1 = out[_col(26)]
    r5 = out[_col(27)]
    r21 = out[_col(28)]
    r63 = out[_col(29)]
    r126 = out[_col(30)]
    out[_col(32)] = np.log(close.shift(21) / close.shift(252))

    factor_matrix = _factor_matrix(out.assign(date=base["date"]), aux_data)
    residual = _rolling_regression_residuals(r1.reset_index(drop=True), factor_matrix) if factor_matrix is not None else None
    out[_col(33)] = residual.shift(21).rolling(232, min_periods=232).sum() if residual is not None else np.nan

    sector_return = _group_return_series(base, aux_data, "sector_returns", str(metadata.get("sector", "")), "sector_return")
    out[_col(34)] = r63 - sector_return.rolling(63, min_periods=63).sum() if sector_return is not None else np.nan

    industry_return = _group_return_series(
        base, aux_data, "industry_returns", str(metadata.get("industry", "")), "industry_return"
    )
    out[_col(35)] = industry_return.rolling(63, min_periods=63).sum() if industry_return is not None else np.nan

    out[_col(36)] = close / close.rolling(252, min_periods=252).max() - 1
    prev_20_max = close.shift(1).rolling(20, min_periods=20).max()
    prev_20_min = close.shift(1).rolling(20, min_periods=20).min()
    out[_col(37)] = (close > prev_20_max).astype(float).where(prev_20_max.notna())
    out[_col(38)] = (close < prev_20_min).astype(float).where(prev_20_min.notna())

    # Reversal and intraday/gap features.
    out[_col(39)] = -r1
    out[_col(40)] = -r5
    out[_col(41)] = -r21
    out[_col(42)] = np.log(open_ / close.shift(1))
    out[_col(43)] = np.log(close / open_)
    out[_col(44)] = -out[_col(42)]
    out[_col(45)] = -(open_ - close.shift(1)) / close.shift(1)
    out[_col(46)] = -(close - daily_vwap) / daily_vwap.replace(0, np.nan)

    # Moving-average and oscillator features.
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    sd20 = close.rolling(20, min_periods=20).std()
    out[_col(47)] = close / ma20 - 1
    out[_col(48)] = ma20 / ma60 - 1
    out[_col(49)] = ma20 / ma20.shift(5) - 1

    delta = close.diff()
    avg_gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50)
    out[_col(50)] = 50 - rsi

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out[_col(51)] = (ema12 - ema26) / close
    out[_col(52)] = (close - ma20) / sd20.replace(0, np.nan)
    out[_col(53)] = -out[_col(52)]

    # Volatility and distribution shape.
    out[_col(54)] = (high - low) / close
    true_range = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(
        axis=1
    )
    out[_col(55)] = true_range.rolling(14, min_periods=14).mean()
    out[_col(56)] = np.sqrt(252) * r1.rolling(21, min_periods=21).std()
    out[_col(57)] = np.sqrt(252) * r1.rolling(63, min_periods=63).std()
    out[_col(58)] = np.sqrt(252) * r1.clip(upper=0).rolling(63, min_periods=63).std()
    out[_col(59)] = residual.rolling(63, min_periods=63).std() if residual is not None else np.nan

    market_return = _aligned_market_return(base, aux_data)
    if market_return is not None:
        out[_col(60)] = r1.rolling(252, min_periods=252).cov(market_return) / market_return.rolling(
            252, min_periods=252
        ).var()
    else:
        out[_col(60)] = np.nan

    out[_col(61)] = r1.rolling(63, min_periods=63).skew()
    out[_col(62)] = r1.rolling(63, min_periods=63).kurt()
    drawdown_63 = close.rolling(63, min_periods=1).apply(
        lambda values: np.min(values / np.maximum.accumulate(values) - 1), raw=True
    )
    out[_col(63)] = drawdown_63.where(close.rolling(63, min_periods=63).count() >= 63)

    # Liquidity and price-volume interaction.
    dollar_volume = close * volume
    out[_col(64)] = volume / volume.rolling(20, min_periods=20).mean() - 1
    out[_col(65)] = dollar_volume
    float_shares = _float_shares_series(base, ticker, aux_data)
    out[_col(66)] = volume / float_shares.replace(0, np.nan) if float_shares is not None else np.nan
    out[_col(67)] = (r1.abs() / dollar_volume.replace(0, np.nan)).rolling(21, min_periods=21).mean()
    out[_col(68)] = (
        dollar_volume.rolling(5, min_periods=5).mean() / dollar_volume.rolling(60, min_periods=60).mean() - 1
    )
    out[_col(69)] = r1.rolling(20, min_periods=20).corr(volume)
    obv = (np.sign(r1.fillna(0)) * volume).cumsum()
    out[_col(70)] = obv - obv.shift(20)

    range_ = high - low
    money_flow_multiplier = (((close - low) - (high - close)) / range_.replace(0, np.nan)).fillna(0.0)
    money_flow_volume = money_flow_multiplier * volume
    out[_col(71)] = _rolling_slope(money_flow_volume, 20)

    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    tp_change = typical_price.diff()
    positive_flow = raw_money_flow.where(tp_change > 0, 0.0).rolling(14, min_periods=14).sum()
    negative_flow = raw_money_flow.where(tp_change < 0, 0.0).rolling(14, min_periods=14).sum()
    mfi = 100 - 100 / (1 + positive_flow / negative_flow.replace(0, np.nan))
    mfi = mfi.where(~((negative_flow == 0) & (positive_flow > 0)), 100.0)
    mfi = mfi.where(~((positive_flow == 0) & (negative_flow > 0)), 0.0)
    mfi = mfi.clip(lower=0, upper=100)
    out[_col(72)] = mfi

    out[_col(73)] = money_flow_volume.rolling(20, min_periods=20).sum() / volume.rolling(
        20, min_periods=20
    ).sum().replace(0, np.nan)
    volume_ratio = volume / volume.rolling(20, min_periods=20).mean()
    out[_col(74)] = r21 * _rolling_zscore(volume_ratio, 252)
    out[_col(75)] = r126 / r1.rolling(63, min_periods=63).std().replace(0, np.nan)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out[["date", *config.ALPHA_COLUMNS]]

