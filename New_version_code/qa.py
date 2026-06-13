"""Repeatable QA suite for the alpha pipeline.

Each check returns a CheckResult with a status:
  * FAIL  - a correctness violation that invalidates results (e.g. look-ahead
            leakage). The pipeline should stop.
  * WARN  - something to look at but not necessarily fatal (e.g. a date with few
            names, an alpha with many NaNs). The pipeline may continue.
  * PASS  - the check passed.

Severity policy (used by run.py):
  * any FAIL  -> stop the run (leakage / structural errors are fatal).
  * WARN only -> continue, but surface the warnings.

Checks are grouped:
  1. Look-ahead / leakage  (FAIL on violation)  -- the most important.
  2. Range / distribution   (WARN on violation).
  3. NaN / completeness     (WARN).
  4. Alignment              (WARN/FAIL).
  5. Structural consistency (FAIL on violation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, cross_section, evaluate, io_data


PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class CheckResult:
    check: str
    status: str
    detail: str = ""


@dataclass
class QAReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, check: str, status: str, detail: str = "") -> None:
        self.results.append(CheckResult(check, status, detail))

    @property
    def n_fail(self) -> int:
        return sum(r.status == FAIL for r in self.results)

    @property
    def n_warn(self) -> int:
        return sum(r.status == WARN for r in self.results)

    @property
    def ok(self) -> bool:
        """True if no FAILs (WARNs allowed)."""
        return self.n_fail == 0

    def text(self) -> str:
        lines = ["=" * 72, "QA REPORT", "=" * 72]
        order = {FAIL: 0, WARN: 1, PASS: 2}
        for r in sorted(self.results, key=lambda x: order[x.status]):
            lines.append(f"[{r.status:4s}] {r.check}")
            if r.detail:
                lines.append(f"        {r.detail}")
        lines.append("-" * 72)
        lines.append(f"PASS={sum(r.status==PASS for r in self.results)} "
                     f"WARN={self.n_warn} FAIL={self.n_fail} "
                     f"-> {'OK' if self.ok else 'STOP (fatal failures)'}")
        lines.append("=" * 72)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 1. Look-ahead / leakage checks (FAIL on violation)
# --------------------------------------------------------------------------- #
def check_alpha_no_lookahead(
    daily_one_ticker: pd.DataFrame,
    alpha_ids: list[int] | None = None,
    factors: pd.DataFrame | None = None,
    market_return: pd.DataFrame | None = None,
    float_shares: pd.DataFrame | None = None,
    tol: float = 1e-8,
) -> CheckResult:
    """Truncated-recompute equals full-panel for the same early dates.

    A single ticker's alphas computed on the full series must match those
    computed on a truncated prefix, at the dates present in both. Any difference
    means a rolling window reached into the future.
    """
    from . import alphas

    full = alphas.compute_single_ticker_alphas(
        daily_one_ticker, factors=factors, market_return=market_return, float_shares=float_shares
    )
    cut = int(len(daily_one_ticker) * 0.7)
    if cut < 30:
        return CheckResult("alpha_no_lookahead", WARN, "ticker too short to test")
    trunc = alphas.compute_single_ticker_alphas(
        daily_one_ticker.iloc[:cut], factors=factors, market_return=market_return, float_shares=float_shares
    )

    cols = config.ALPHA_COLUMNS if alpha_ids is None else [config.ALPHA_COLUMN_BY_ID[i] for i in alpha_ids]
    worst = 0.0
    worst_col = None
    for col in cols:
        if col not in full.columns or col not in trunc.columns:
            continue
        a = full[col].iloc[:cut].to_numpy(dtype=float)
        b = trunc[col].to_numpy(dtype=float)
        both = ~(np.isnan(a) | np.isnan(b))
        if both.sum() == 0:
            continue
        d = np.max(np.abs(a[both] - b[both]))
        if d > worst:
            worst, worst_col = d, col

    if worst > tol:
        return CheckResult("alpha_no_lookahead", FAIL,
                           f"truncated-recompute differs by {worst:.2e} at {worst_col} (look-ahead!)")
    return CheckResult("alpha_no_lookahead", PASS, f"max diff {worst:.1e} across {len(cols)} alphas")


def check_forward_return_no_cross_ticker(price_panel: pd.DataFrame) -> CheckResult:
    """The excess target's stock leg must use only that ticker's own future prices.

    Target is forward_5d_excess_return = log(close(t+H)/close(t)) - log(SPX(t+H)/SPX(t)).
    We recompute it independently (stock leg per-ticker, market leg by date) and
    confirm it matches, which proves the stock leg has no cross-ticker leakage.
    """
    panel = price_panel[["date", "ticker"]].copy()
    res = evaluate.add_forward_return(panel, price_panel)
    h = config.FORWARD_RETURN_HORIZON

    # Stock leg, recomputed independently per ticker.
    p = price_panel[["date", "ticker", "close"]].copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    p = p.sort_values(["ticker", "date"])
    p["_stock"] = np.log(p.groupby("ticker")["close"].shift(-h) / p["close"])

    # Market leg, recomputed independently by date.
    spx = io_data.load_spx_close()
    spx["date"] = pd.to_datetime(spx["date"]).dt.normalize()
    spx = spx.sort_values("date").reset_index(drop=True)
    spx_close = pd.to_numeric(spx["spx_close"], errors="coerce")
    spx["_mkt"] = np.log(spx_close.shift(-h) / spx_close).to_numpy()
    p = p.merge(spx[["date", "_mkt"]], on="date", how="left")
    p["indep"] = p["_stock"] - p["_mkt"]

    merged = res.merge(p[["date", "ticker", "indep"]], on=["date", "ticker"], how="left")
    a = merged[config.TARGET_COLUMN].to_numpy(dtype=float)
    b = merged["indep"].to_numpy(dtype=float)
    both = ~(np.isnan(a) | np.isnan(b))
    if both.sum() == 0:
        return CheckResult("forward_return_no_cross_ticker", WARN, "no overlapping non-NaN values")
    d = np.max(np.abs(a[both] - b[both]))
    if d > 1e-9:
        return CheckResult("forward_return_no_cross_ticker", FAIL,
                           f"excess forward return differs from independent recompute by {d:.2e}")
    return CheckResult("forward_return_no_cross_ticker", PASS, f"max diff {d:.1e}")


def check_cross_section_no_leakage(panel: pd.DataFrame, n_dates: int = 3) -> CheckResult:
    """Corrupting one date's raw alphas must not change another date's transforms."""
    work = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    dates = list(pd.to_datetime(work["date"]).dt.normalize().unique())
    if len(dates) < 2:
        return CheckResult("cross_section_no_leakage", WARN, "need >=2 dates to test")

    base = cross_section.add_cross_sectional_transforms(work)
    target_date = dates[0]
    corrupt_date = dates[-1]

    mod = work.copy()
    mod["date"] = pd.to_datetime(mod["date"]).dt.normalize()
    acol = config.ALPHA_COLUMNS[0]
    mod.loc[mod["date"] == corrupt_date, acol] = -999.0
    modres = cross_section.add_cross_sectional_transforms(mod)

    zcol = acol + cross_section.Z_SUFFIX
    b0 = base.copy()
    b0["date"] = pd.to_datetime(b0["date"]).dt.normalize()
    m0 = modres.copy()
    m0["date"] = pd.to_datetime(m0["date"]).dt.normalize()
    bv = b0[b0["date"] == target_date][zcol].to_numpy(dtype=float)
    mv = m0[m0["date"] == target_date][zcol].to_numpy(dtype=float)
    if not np.allclose(bv, mv, equal_nan=True):
        return CheckResult("cross_section_no_leakage", FAIL,
                           "corrupting a later date changed an earlier date's transform")
    return CheckResult("cross_section_no_leakage", PASS, "earlier date unchanged after corrupting later date")


def check_train_test_isolation(panel_with_target: pd.DataFrame) -> CheckResult:
    """Rank IC must only use dates on or before TRAIN_END_DATE."""
    summary = evaluate.compute_rank_ic(panel_with_target, train_only=True)
    dates = pd.to_datetime(panel_with_target["date"]).dt.normalize()
    n_train_dates = (dates.unique() <= pd.Timestamp(config.TRAIN_END_DATE)).sum()
    max_days = int(summary["n_days"].max()) if len(summary) else 0
    if max_days > n_train_dates:
        return CheckResult("train_test_isolation", FAIL,
                           f"IC used {max_days} days but only {n_train_dates} are in train")
    return CheckResult("train_test_isolation", PASS,
                       f"IC used <= {n_train_dates} train dates (test excluded)")


# --------------------------------------------------------------------------- #
# 2. Range / distribution checks (WARN)
# --------------------------------------------------------------------------- #
def check_transform_ranges(transformed: pd.DataFrame, sample_alpha: str | None = None) -> list[CheckResult]:
    """z ~ mean0/std1 per date; rank and sector_rank within [0,1]."""
    out: list[CheckResult] = []
    acol = sample_alpha or config.ALPHA_COLUMNS[0]
    zc = acol + cross_section.Z_SUFFIX
    rc = acol + cross_section.RANK_SUFFIX
    sc = acol + cross_section.SECTOR_RANK_SUFFIX

    # rank ranges
    for col, name in [(rc, "rank"), (sc, "sector_neutral_rank")]:
        if col in transformed.columns:
            v = pd.to_numeric(transformed[col], errors="coerce").dropna()
            if len(v) and (v.min() < -1e-9 or v.max() > 1 + 1e-9):
                out.append(CheckResult(f"range_{name}", WARN,
                                       f"{col} outside [0,1]: min={v.min():.3f} max={v.max():.3f}"))
            else:
                out.append(CheckResult(f"range_{name}", PASS, f"{col} within [0,1]"))

    # z per-date mean/std (sample a few dates). cross_section._zscore uses the
    # pandas default std (ddof=1), so check with the same ddof to avoid a
    # spurious off-by-(n/(n-1)) mismatch.
    if zc in transformed.columns:
        bad = 0
        checked = 0
        for _, g in transformed.groupby("date"):
            v = pd.to_numeric(g[zc], errors="coerce").dropna()
            if len(v) >= config.MIN_CROSS_SECTIONAL_COUNT:
                checked += 1
                if abs(v.mean()) > 1e-6 or abs(v.std(ddof=1) - 1.0) > 1e-3:
                    bad += 1
        if checked and bad == 0:
            out.append(CheckResult("range_zscore", PASS, f"{zc} mean~0/std~1 on {checked} dates"))
        elif checked:
            out.append(CheckResult("range_zscore", WARN, f"{zc} off on {bad}/{checked} dates"))
    return out


def check_bounded_alphas(panel: pd.DataFrame) -> list[CheckResult]:
    """Alphas with a known theoretical range stay inside it (CMF, RSI, MFI)."""
    out: list[CheckResult] = []
    bounded = {
        config.ALPHA_COLUMN_BY_ID.get(73): (-1.0, 1.0, "CMF"),
        config.ALPHA_COLUMN_BY_ID.get(50): (-50.0, 50.0, "RSI-reversion (50-RSI)"),
        config.ALPHA_COLUMN_BY_ID.get(72): (0.0, 100.0, "MFI"),
    }
    for col, (lo, hi, name) in bounded.items():
        if col and col in panel.columns:
            v = pd.to_numeric(panel[col], errors="coerce").dropna()
            if len(v) and (v.min() < lo - 1e-6 or v.max() > hi + 1e-6):
                out.append(CheckResult(f"bounded_{name}", WARN,
                                       f"{col} outside [{lo},{hi}]: min={v.min():.3f} max={v.max():.3f}"))
            elif len(v):
                out.append(CheckResult(f"bounded_{name}", PASS, f"{col} within [{lo},{hi}]"))
    return out


# --------------------------------------------------------------------------- #
# 3. NaN / completeness checks (WARN)
# --------------------------------------------------------------------------- #
def check_alpha_nan_rates(panel: pd.DataFrame, warn_threshold: float = 0.60) -> list[CheckResult]:
    """Flag alphas that are entirely NaN or mostly NaN.

    A fully-NaN column is a WARN, not a FAIL: it is usually a legitimate
    consequence of insufficient warmup (e.g. alpha 33 needs ~503 days) or of an
    auxiliary input not being supplied (e.g. alpha 66 with no float data), not a
    computation bug. A column that is entirely *missing* from the panel is still
    a FAIL, since that means the alpha was never produced.
    """
    out: list[CheckResult] = []
    n = len(panel)
    if n == 0:
        return [CheckResult("alpha_nan_rates", WARN, "empty panel")]
    for col in config.ALPHA_COLUMNS:
        if col not in panel.columns:
            out.append(CheckResult("alpha_missing_column", FAIL, f"{col} not in panel"))
            continue
        frac = panel[col].isna().mean()
        if frac >= 1.0 - 1e-9:
            out.append(CheckResult("alpha_all_nan", WARN,
                                   f"{col} is entirely NaN (warmup too short or auxiliary input missing)"))
        elif frac >= warn_threshold:
            out.append(CheckResult("alpha_high_nan", WARN, f"{col} is {frac:.0%} NaN (warmup?)"))
    if not out:
        out.append(CheckResult("alpha_nan_rates", PASS, "no all-NaN or high-NaN alphas"))
    return out


def check_cross_section_counts(panel: pd.DataFrame) -> CheckResult:
    """Warn if many dates have fewer than MIN_CROSS_SECTIONAL_COUNT names."""
    counts = panel.groupby("date")["ticker"].nunique()
    thin = (counts < config.MIN_CROSS_SECTIONAL_COUNT).sum()
    if thin > 0:
        return CheckResult("cross_section_counts", WARN,
                           f"{thin} dates have < {config.MIN_CROSS_SECTIONAL_COUNT} names "
                           f"(min={counts.min()})")
    return CheckResult("cross_section_counts", PASS,
                       f"all dates have >= {config.MIN_CROSS_SECTIONAL_COUNT} names (min={counts.min()})")


# --------------------------------------------------------------------------- #
# 4. Alignment checks (WARN / FAIL)
# --------------------------------------------------------------------------- #
def check_ticker_alignment(price_panel: pd.DataFrame, sector_map: pd.DataFrame | None) -> CheckResult:
    """Tickers in the price panel should have GICS sectors; warn on gaps."""
    if sector_map is None or "ticker" not in getattr(sector_map, "columns", []):
        return CheckResult("ticker_alignment", WARN, "no sector map provided")
    price_tk = set(price_panel["ticker"].unique())
    sect_tk = set(sector_map["ticker"].unique())
    missing = price_tk - sect_tk
    if missing:
        sample = ", ".join(sorted(missing)[:8])
        return CheckResult("ticker_alignment", WARN,
                           f"{len(missing)} priced tickers lack a sector: {sample}")
    return CheckResult("ticker_alignment", PASS, f"all {len(price_tk)} priced tickers have a sector")


def check_date_continuity(price_panel: pd.DataFrame, max_gap_days: int = 5) -> CheckResult:
    """Warn on suspiciously large gaps between consecutive trading dates."""
    dates = pd.to_datetime(pd.Series(price_panel["date"].unique())).sort_values()
    if len(dates) < 2:
        return CheckResult("date_continuity", WARN, "too few dates")
    gaps = dates.diff().dt.days.dropna()
    big = gaps[gaps > max_gap_days]
    if len(big):
        return CheckResult("date_continuity", WARN,
                           f"{len(big)} gaps > {max_gap_days} calendar days (max {int(gaps.max())}d)")
    return CheckResult("date_continuity", PASS, f"no gaps > {max_gap_days} days")


# --------------------------------------------------------------------------- #
# 5. Structural consistency checks (FAIL)
# --------------------------------------------------------------------------- #
def check_alpha_count(panel: pd.DataFrame) -> CheckResult:
    """All 49 raw alpha columns must be present (alpha 35 intentionally absent)."""
    missing = [c for c in config.ALPHA_COLUMNS if c not in panel.columns]
    if missing:
        return CheckResult("alpha_count", FAIL, f"missing {len(missing)} alpha columns: {missing[:5]}")
    if 35 in config.ALPHA_IDS:
        return CheckResult("alpha_count", FAIL, "alpha 35 should be removed but is present")
    return CheckResult("alpha_count", PASS, f"all {len(config.ALPHA_COLUMNS)} alphas present, 35 absent")


def check_transform_count(transformed: pd.DataFrame) -> CheckResult:
    """Every alpha must have its three transform columns."""
    expected = cross_section.transform_column_names()
    missing = [c for c in expected if c not in transformed.columns]
    if missing:
        return CheckResult("transform_count", FAIL,
                           f"missing {len(missing)}/{len(expected)} transform cols")
    return CheckResult("transform_count", PASS, f"all {len(expected)} transform columns present")


def check_selected_subset(selected: list[str]) -> CheckResult:
    """Selected features must be a subset of the known alpha columns."""
    bad = [s for s in selected if s not in config.ALPHA_COLUMNS]
    if bad:
        return CheckResult("selected_subset", FAIL, f"selected non-alpha columns: {bad[:5]}")
    return CheckResult("selected_subset", PASS, f"{len(selected)} selected, all valid alphas")


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #
def run_all_qa(
    *,
    daily_one_ticker: pd.DataFrame | None = None,
    price_panel: pd.DataFrame | None = None,
    raw_alpha_panel: pd.DataFrame | None = None,
    transformed_panel: pd.DataFrame | None = None,
    panel_with_target: pd.DataFrame | None = None,
    sector_map: pd.DataFrame | None = None,
    selected: list[str] | None = None,
    factors: pd.DataFrame | None = None,
    market_return: pd.DataFrame | None = None,
    float_shares: pd.DataFrame | None = None,
) -> QAReport:
    """Run every applicable check given whichever artifacts are provided.

    Each argument is optional; a check runs only if its inputs are present. This
    lets run.py call run_all_qa at different stages with different artifacts.
    """
    report = QAReport()

    # 1. leakage
    if daily_one_ticker is not None:
        report.results.append(check_alpha_no_lookahead(
            daily_one_ticker, factors=factors, market_return=market_return, float_shares=float_shares))
    if price_panel is not None:
        report.results.append(check_forward_return_no_cross_ticker(price_panel))
    if raw_alpha_panel is not None:
        report.results.append(check_cross_section_no_leakage(raw_alpha_panel))
    if panel_with_target is not None:
        report.results.append(check_train_test_isolation(panel_with_target))

    # 2. ranges
    if transformed_panel is not None:
        report.results.extend(check_transform_ranges(transformed_panel))
    if raw_alpha_panel is not None:
        report.results.extend(check_bounded_alphas(raw_alpha_panel))

    # 3. NaN / completeness
    if raw_alpha_panel is not None:
        report.results.extend(check_alpha_nan_rates(raw_alpha_panel))
        report.results.append(check_cross_section_counts(raw_alpha_panel))

    # 4. alignment
    if price_panel is not None:
        report.results.append(check_ticker_alignment(price_panel, sector_map))
        report.results.append(check_date_continuity(price_panel))

    # 5. structure
    if raw_alpha_panel is not None:
        report.results.append(check_alpha_count(raw_alpha_panel))
    if transformed_panel is not None:
        report.results.append(check_transform_count(transformed_panel))
    if selected is not None:
        report.results.append(check_selected_subset(selected))

    return report