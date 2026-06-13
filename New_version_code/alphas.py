"""Alpha computation for the S&P 500 price-volume pipeline.

Two entry points, matching the two-stage design:

  * ``compute_single_ticker_alphas(daily, aux)`` computes every alpha that can
    be derived from one stock's own history (48 of the 49), including the
    factor-residual alphas (33, 59, 60) and float turnover (66), which need
    auxiliary series aligned to that stock's dates.

  * ``add_cross_sectional_alphas(panel)`` adds alpha 34 (sector-neutral
    momentum), which is inherently cross-sectional: each stock's 126-day
    momentum (alpha 30) minus the median 126-day momentum of its GICS-sector
    peers on the same date.

All single-ticker rolling windows operate within one ticker, so there is no
cross-ticker leakage. Formula choices that were corrected against the team's
definition document:
  * alpha 33: residual.shift(21).rolling(231).sum()   (231 = 252 - 21)
  * alpha 59: rolling(63).std() of residuals, NOT annualized
  * alpha 34: minus the sector-peer *median* of alpha-30 momentum (see above)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _col(alpha_id: int) -> str:
    return config.ALPHA_COLUMN_BY_ID[alpha_id]


# --------------------------------------------------------------------------- #
# Rolling helpers
# --------------------------------------------------------------------------- #
def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std, full-window."""
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return (series - mean) / std.replace(0, np.nan)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Slope of an OLS fit of the series on 0..window-1 within each window."""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_dev = x - x_mean
    denom = float((x_dev**2).sum())

    def _slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y_mean = values.mean()
        return float((x_dev * (values - y_mean)).sum() / denom)

    return series.rolling(window, min_periods=window).apply(_slope, raw=True)


def _rolling_residuals(stock_excess: pd.Series, factors: pd.DataFrame, window: int = 252) -> pd.Series:
    """Rolling OLS residual of stock excess return on the factor matrix.

    For each day t with a full ``window`` of history, regress the stock's
    excess return on the factors over [t-window+1, t] and take the residual at
    t. Used by the factor-neutral alphas (33, 59) and as a building block.

    Both inputs are aligned positionally (same index, same length).
    """
    n = len(stock_excess)
    y_all = stock_excess.to_numpy(dtype=float)
    # Design matrix with intercept.
    f_all = factors.to_numpy(dtype=float)
    X_all = np.column_stack([np.ones(n), f_all])
    resid = np.full(n, np.nan)

    for t in range(window - 1, n):
        sl = slice(t - window + 1, t + 1)
        y = y_all[sl]
        X = X_all[sl]
        if np.isnan(y[-1]) or np.isnan(X[-1]).any():
                continue
        valid = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        if valid.sum() < window // 2:
            continue
        # Least squares; residual at the last row of the window (day t).
        beta, *_ = np.linalg.lstsq(X[valid], y[valid], rcond=None)
        resid[t] = y[-1] - X[-1] @ beta
    return pd.Series(resid, index=stock_excess.index)


# --------------------------------------------------------------------------- #
# Stage 1: single-ticker alphas
# --------------------------------------------------------------------------- #
def compute_single_ticker_alphas(
    daily: pd.DataFrame,
    factors: pd.DataFrame | None = None,
    market_return: pd.DataFrame | None = None,
    float_shares: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute all single-ticker alphas for one stock's daily table.

    Parameters
    ----------
    daily : one ticker's table with date, ticker, open, high, low, close,
        volume, daily_vwap (sorted or not).
    factors : Carhart factor table (date, mkt_rf, smb, hml, rf, mom). Needed
        for alphas 33 and 59. If None, those are NaN.
    market_return : SPX market return table (date, market_return). Needed for
        alpha 60 beta. If None, alpha 60 is NaN.
    float_shares : long float table (date, ticker, float_shares). Needed for
        alpha 66 turnover. If None, alpha 66 is NaN.

    Returns a frame with date, ticker, and the single-ticker alpha columns
    (everything except alpha 34, which is added later cross-sectionally).
    """
    base = daily.sort_values("date").reset_index(drop=True).copy()
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    ticker = str(base["ticker"].iloc[0])

    close = pd.to_numeric(base["close"], errors="coerce")
    open_ = pd.to_numeric(base["open"], errors="coerce")
    high = pd.to_numeric(base["high"], errors="coerce")
    low = pd.to_numeric(base["low"], errors="coerce")
    volume = pd.to_numeric(base["volume"], errors="coerce")
    vwap = pd.to_numeric(base["daily_vwap"], errors="coerce")

    out = pd.DataFrame({"date": base["date"], "ticker": ticker})

    # --- Returns and momentum (26-32) ---
    for alpha_id, lb in [(26, 1), (27, 5), (28, 21), (29, 63), (30, 126), (31, 252)]:
        out[_col(alpha_id)] = np.log(close / close.shift(lb))
    r1 = out[_col(26)]
    r21 = out[_col(28)]
    r63 = out[_col(29)]
    r126 = out[_col(30)]
    # 12-1 momentum: skip the most recent month.
    out[_col(32)] = np.log(close.shift(21) / close.shift(252))

    # --- Factor residuals for 33 (residual momentum) and 59 (idio vol) ---
    residual = None
    if factors is not None:
        fac = factors.copy()
        fac["date"] = pd.to_datetime(fac["date"]).dt.normalize()
        merged = out[["date"]].merge(fac, on="date", how="left")
        assert len(merged) == len(out), "factor merge changed row count (duplicate dates?)"
        excess = r1 - merged["rf"].to_numpy()
        factor_cols = merged[["mkt_rf", "smb", "hml", "mom"]]
        residual = _rolling_residuals(excess.reset_index(drop=True), factor_cols, window=252)
        residual.index = out.index

    # alpha 33: cumulative residual from t-252 to t-21 = shift(21).rolling(231).
    if residual is not None:
        out[_col(33)] = residual.shift(21).rolling(231, min_periods=200).sum()
    else:
        out[_col(33)] = np.nan

    # --- 36-38 high/low position ---
    high_252 = close.rolling(252, min_periods=252).max()
    out[_col(36)] = close / high_252 - 1.0
    prev_20_max = close.shift(1).rolling(20, min_periods=20).max()
    prev_20_min = close.shift(1).rolling(20, min_periods=20).min()
    out[_col(37)] = (close > prev_20_max).astype(float).where(prev_20_max.notna())
    out[_col(38)] = (close < prev_20_min).astype(float).where(prev_20_min.notna())

    # --- 39-46 reversal / overnight / intraday / gap / vwap ---
    out[_col(39)] = -r1
    out[_col(40)] = -out[_col(27)]
    out[_col(41)] = -r21
    out[_col(42)] = np.log(open_ / close.shift(1))            # overnight
    out[_col(43)] = np.log(close / open_)                      # intraday
    out[_col(44)] = -out[_col(42)]                             # overnight reversal
    out[_col(45)] = -(open_ - close.shift(1)) / close.shift(1) # gap-fill
    out[_col(46)] = -(close - vwap) / vwap.replace(0, np.nan)  # vwap reversion

    # --- 47-49 moving averages ---
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    out[_col(47)] = close / ma20 - 1.0
    out[_col(48)] = ma20 / ma60 - 1.0
    # alpha 49 (doc): MA20_t / MA20_{t-5} - 1  (5-day change of the MA20 line)
    out[_col(49)] = ma20 / ma20.shift(5) - 1.0

    # --- 50 RSI reversion ---
    delta = close.diff()
    avg_gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50)
    out[_col(50)] = 50 - rsi

    # --- 51 MACD normalized (doc): (EMA12(C) - EMA26(C)) / C ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out[_col(51)] = (ema12 - ema26) / close.replace(0, np.nan)

    # --- 52-53 Bollinger ---
    bb_mean = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    out[_col(52)] = (close - bb_mean) / bb_std.replace(0, np.nan)
    out[_col(53)] = -out[_col(52)]

    # --- 54-58 range and volatility ---
    out[_col(54)] = (high - low) / close
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    out[_col(55)] = true_range.rolling(14, min_periods=14).mean()
    out[_col(56)] = r1.rolling(21, min_periods=21).std()
    out[_col(57)] = r1.rolling(63, min_periods=63).std()
    # alpha 58 (doc): sqrt(252) * SD63(min{r,0}) -- downside-only, annualized
    downside = r1.where(r1 < 0, 0.0)
    out[_col(58)] = np.sqrt(252.0) * downside.rolling(63, min_periods=63).std()

    # alpha 59: idiosyncratic volatility = 63-day std of residuals (NOT annualized)
    if residual is not None:
        out[_col(59)] = residual.rolling(63, min_periods=55).std()
    else:
        out[_col(59)] = np.nan

    # --- 60 market beta ---
    if market_return is not None:
        mkt = market_return.copy()
        mkt["date"] = pd.to_datetime(mkt["date"]).dt.normalize()
        merged_m = out[["date"]].merge(mkt, on="date", how="left")
        assert len(merged_m) == len(out), "market merge changed row count (duplicate dates?)"
        mret = merged_m["market_return"].reset_index(drop=True)
        r1r = r1.reset_index(drop=True)
        cov = r1r.rolling(252, min_periods=252).cov(mret)
        var = mret.rolling(252, min_periods=252).var()
        beta = cov / var.replace(0, np.nan)
        beta.index = out.index
        out[_col(60)] = beta
    else:
        out[_col(60)] = np.nan

    # --- 61-63 distribution and drawdown ---
    out[_col(61)] = r1.rolling(63, min_periods=63).skew()
    out[_col(62)] = r1.rolling(63, min_periods=63).kurt()

    def _max_drawdown(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        running_max = np.maximum.accumulate(values)
        return float(np.min(values / running_max - 1.0))

    out[_col(63)] = close.rolling(63, min_periods=63).apply(_max_drawdown, raw=True)

    # --- 64-65 volume ---
    adv20 = volume.rolling(20, min_periods=20).mean()
    out[_col(64)] = volume / adv20.replace(0, np.nan) - 1.0
    dollar_volume = close * volume
    out[_col(65)] = np.log(dollar_volume.replace(0, np.nan))

    # --- 66 turnover (needs float) ---
    if float_shares is not None:
        fl = float_shares[float_shares["ticker"] == ticker][["date", "float_shares"]].copy()
        if not fl.empty:
            fl["date"] = pd.to_datetime(fl["date"]).dt.normalize()
            fl = fl.sort_values("date")
            # Point-in-time: as-of-backward join so each trading day uses the
            # most recent known float, never a future revision.
            aligned = pd.merge_asof(
                out[["date"]].sort_values("date"),
                fl,
                on="date",
                direction="backward",
            )
            aligned.index = out.index
            out[_col(66)] = volume / aligned["float_shares"].replace(0, np.nan)
        else:
            out[_col(66)] = np.nan
    else:
        out[_col(66)] = np.nan

    # --- 67 Amihud illiquidity ---
    out[_col(67)] = (r1.abs() / dollar_volume.replace(0, np.nan)).rolling(21, min_periods=21).mean()

    # --- 68 ADV trend (doc): MA5(DollarVol) / MA60(DollarVol) - 1 ---
    dv5 = dollar_volume.rolling(5, min_periods=5).mean()
    dv60 = dollar_volume.rolling(60, min_periods=60).mean()
    out[_col(68)] = dv5 / dv60.replace(0, np.nan) - 1.0

    # --- 69 price-volume correlation (doc): Corr20(r, Vol) ---
    out[_col(69)] = r1.rolling(20, min_periods=20).corr(volume)

    # --- 70 OBV change (doc): OBV20_t - OBV20_{t-1}, where
    #     OBV20_t = sum_{s=t-19..t} sign(r_s) * Vol_s  (20-day rolling, not cumulative) ---
    signed_volume = np.sign(delta).fillna(0.0) * volume
    obv20 = signed_volume.rolling(20, min_periods=20).sum()
    out[_col(70)] = obv20.diff(1)

    # --- Close-location value (CLV) and money-flow volume, shared by 71 & 73 ---
    # CLV = ((C-L) - (H-C)) / (H-L); flat days (H==L) -> 0 (neutral flow).
    price_range = high - low
    clv = (((close - low) - (high - close)) / price_range.replace(0, np.nan)).fillna(0.0)
    money_flow_volume = clv * volume

    # --- 71 accumulation/distribution slope (doc): slope_20(CLV * Vol) ---
    out[_col(71)] = _rolling_slope(money_flow_volume, 20)

    # --- 72 money flow index ---
    typical = (high + low + close) / 3.0
    raw_money = typical * volume
    pos_flow = raw_money.where(typical > typical.shift(1), 0.0)
    neg_flow = raw_money.where(typical < typical.shift(1), 0.0)
    pos_sum = pos_flow.rolling(14, min_periods=14).sum()
    neg_sum = neg_flow.rolling(14, min_periods=14).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    out[_col(72)] = 100 - 100 / (1 + mfr)

    # --- 73 Chaikin money flow: sum_20(CLV*Vol) / sum_20(Vol) ---
    out[_col(73)] = (
        money_flow_volume.rolling(20, min_periods=20).sum()
        / volume.rolling(20, min_periods=20).sum().replace(0, np.nan)
    )

    # --- 74 volume-confirmed momentum: r21 * z(Vol / ADV20), rolling z over 252 ---
    volume_ratio = volume / adv20.replace(0, np.nan)
    out[_col(74)] = r21 * _rolling_zscore(volume_ratio, 252)

    # --- 75 low-volatility momentum ---
    vol63 = r1.rolling(63, min_periods=63).std()
    out[_col(75)] = r126 / vol63.replace(0, np.nan)

    return out


# --------------------------------------------------------------------------- #
# Stage 2: cross-sectional alpha 34
# --------------------------------------------------------------------------- #
def add_cross_sectional_alphas(panel: pd.DataFrame) -> pd.DataFrame:
    """Add alpha 34 (sector-neutral momentum) to the all-market panel.

    alpha 34 = alpha_30 (126-day momentum) minus the median alpha_30 across the
    stock's GICS-sector peers on the same date. Dates/sectors with fewer than
    ``config.MIN_SECTOR_COUNT`` valid peers get NaN to avoid unstable medians.
    """
    panel = panel.copy()
    mom_col = _col(30)
    sector_col = "sector"
    if sector_col not in panel.columns:
        raise ValueError("panel must carry a 'sector' column for alpha 34.")

    mom = pd.to_numeric(panel[mom_col], errors="coerce")
    grp = panel.assign(_mom=mom).groupby(["date", sector_col])["_mom"]
    sector_median = grp.transform("median")
    sector_count = grp.transform("count")

    alpha34 = mom - sector_median
    alpha34 = alpha34.where(sector_count >= config.MIN_SECTOR_COUNT)
    panel[_col(34)] = alpha34
    return panel