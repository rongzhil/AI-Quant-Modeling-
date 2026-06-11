"""End-to-end pipeline orchestration.

Stages:
  1. load daily prices + auxiliary data (GICS, factors, market, float)
  2. compute 49 single-ticker alphas per stock, concat, add the cross-sectional
     alpha (34)
  3. attach sector metadata
  4. add cross-sectional transforms (z / rank / sector_neutral_rank -> 147 cols)
  5. add 5-day forward return target
  6. compute per-alpha Rank IC on the train segment
  7. two-step feature selection

QA gates run at the key nodes. A FAIL (look-ahead / structural error) stops the
run; WARNs (warmup NaNs, thin dates) are printed and the run continues.

Intermediate artifacts are cached as parquet so re-runs can skip finished work:
  daily_prices.parquet -> alpha_panel_raw.parquet -> alpha_panel_final.parquet
plus rank_ic_summary.csv, selected_features.csv, feature_selection_report.txt.

This module is import-safe; run it as a script (``python -m quant_pipeline.run``)
or call ``run_pipeline()``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from . import (
    config,
    io_data,
    alphas,
    cross_section,
    evaluate,
    select_features,
    qa,
)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _qa_gate(report: qa.QAReport, stage: str, stop_on_fail: bool = True) -> None:
    """Print a QA report for a stage; raise if it has fatal failures."""
    _log(f"QA @ {stage}: PASS={sum(r.status==qa.PASS for r in report.results)} "
         f"WARN={report.n_warn} FAIL={report.n_fail}")
    for r in report.results:
        if r.status in (qa.WARN, qa.FAIL):
            _log(f"    [{r.status}] {r.check}: {r.detail}")
    if stop_on_fail and not report.ok:
        raise RuntimeError(
            f"QA FAILED at stage '{stage}' with {report.n_fail} fatal issue(s); "
            f"see report above. Pipeline stopped."
        )


# --------------------------------------------------------------------------- #
# Stage 1: load
# --------------------------------------------------------------------------- #
def load_inputs(use_cache: bool = True, verbose: bool = True) -> dict:
    """Load prices and all auxiliary data. Returns a dict of DataFrames."""
    _log("Stage 1: loading daily prices ...")
    daily = io_data.load_all_prices(use_cache=use_cache, verbose=verbose)
    _log(f"  daily prices: {daily['ticker'].nunique()} tickers, {len(daily)} rows")

    _log("Stage 1: loading auxiliary data (GICS, factors, market, float) ...")
    gics = io_data.load_gics_mapping()
    factors = io_data.load_factors()
    market = io_data.load_spx_market_return()
    try:
        float_shares = io_data.load_float_shares()
    except Exception as exc:  # float is optional; alpha 66 becomes NaN without it
        _log(f"  WARNING: float shares unavailable ({exc}); alpha 66 will be NaN")
        float_shares = None

    return {
        "daily": daily,
        "gics": gics,
        "factors": factors,
        "market": market,
        "float_shares": float_shares,
    }


# --------------------------------------------------------------------------- #
# Stage 2: alphas
# --------------------------------------------------------------------------- #
def compute_alpha_panel(inputs: dict, use_cache: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Compute 49 raw alphas per ticker, concat, add cross-sectional alpha 34."""
    cache = config.ALPHA_PANEL_RAW_PATH
    if use_cache and Path(cache).exists():
        _log(f"Stage 2: loading cached raw alpha panel from {cache.name}")
        return pd.read_parquet(cache)

    daily = inputs["daily"]
    factors = inputs["factors"]
    market = inputs["market"]
    float_shares = inputs["float_shares"]

    tickers = sorted(daily["ticker"].unique())
    _log(f"Stage 2: computing alphas for {len(tickers)} tickers ...")

    parts = []
    t0 = time.time()
    for i, tk in enumerate(tickers, 1):
        sub = daily[daily["ticker"] == tk].sort_values("date").reset_index(drop=True)
        a = alphas.compute_single_ticker_alphas(
            sub, factors=factors, market_return=market, float_shares=float_shares
        )
        parts.append(a)
        if verbose and (i % 50 == 0 or i == len(tickers)):
            _log(f"  {i}/{len(tickers)} tickers done ({time.time()-t0:.0f}s)")

    panel = pd.concat(parts, ignore_index=True)

    # QA gate: per-ticker leakage check on one representative ticker.
    rep = qa.QAReport()
    sample = daily[daily["ticker"] == tickers[0]].sort_values("date").reset_index(drop=True)
    rep.results.append(qa.check_alpha_no_lookahead(
        sample, factors=factors, market_return=market, float_shares=float_shares))
    _qa_gate(rep, "alpha look-ahead (single ticker)")

    # alpha 34 (sector-neutral momentum) needs the sector column, so attach
    # GICS metadata before computing it.
    _log("Stage 2: attaching sector metadata ...")
    meta_cols = ["ticker"] + [c for c in config.METADATA_COLUMNS if c in inputs["gics"].columns]
    meta = inputs["gics"][meta_cols].drop_duplicates("ticker")
    panel = panel.merge(meta, on="ticker", how="left")

    _log("Stage 2: adding cross-sectional alpha 34 ...")
    panel = alphas.add_cross_sectional_alphas(panel)

    if use_cache:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache, index=False)
        _log(f"  cached raw alpha panel -> {cache.name}")
    return panel


# --------------------------------------------------------------------------- #
# Stage 3-5: metadata, transforms, target
# --------------------------------------------------------------------------- #
def build_feature_panel(
    raw_panel: pd.DataFrame,
    inputs: dict,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Attach sector metadata, cross-sectional transforms, and forward return."""
    cache = config.ALPHA_PANEL_FINAL_PATH
    if use_cache and Path(cache).exists():
        _log(f"Stage 3-5: loading cached final panel from {cache.name}")
        return pd.read_parquet(cache)

    _log("Stage 3: validating raw alpha panel (sector already attached) ...")
    panel = raw_panel

    # QA: cross-section leakage + alpha completeness + NaN rates on raw alphas
    rep = qa.run_all_qa(
        raw_alpha_panel=panel,
        price_panel=inputs["daily"][["date", "ticker", "close"]],
        sector_map=inputs["gics"],
        factors=inputs["factors"],
        market_return=inputs["market"],
        float_shares=inputs["float_shares"],
    )
    _qa_gate(rep, "raw alpha panel")

    _log("Stage 4: adding cross-sectional transforms (147 cols) ...")
    panel = cross_section.add_cross_sectional_transforms(panel)

    _log("Stage 5: adding 5-day forward return ...")
    price_panel = inputs["daily"][["date", "ticker", "close"]]
    panel = evaluate.add_forward_return(panel, price_panel)

    # QA: transforms + target (ranges, train/test isolation, forward-return leak)
    rep = qa.run_all_qa(
        transformed_panel=panel,
        panel_with_target=panel,
    )
    # forward-return cross-ticker leakage needs the price panel; check explicitly
    rep.results.append(qa.check_forward_return_no_cross_ticker(price_panel))
    _qa_gate(rep, "feature panel (transforms + target)")

    if use_cache:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache, index=False)
        _log(f"  cached final panel -> {cache.name}")
    return panel


# --------------------------------------------------------------------------- #
# Stage 6-7: Rank IC + feature selection
# --------------------------------------------------------------------------- #
def evaluate_and_select(panel: pd.DataFrame) -> dict:
    """Compute train-only Rank IC and run two-step feature selection."""
    _log("Stage 6: computing per-alpha Rank IC (train only) ...")
    ic_summary = evaluate.compute_rank_ic(panel, train_only=True)
    ic_summary.to_csv(config.RANK_IC_REPORT_PATH, index=False)
    _log(f"  Rank IC summary -> {config.RANK_IC_REPORT_PATH.name}")

    _log("Stage 7: feature selection (significance + redundancy) ...")
    result = select_features.run_feature_selection(panel, ic_summary, write=True)
    _log(f"  survivors (step1): {len(result['survivors_step1'])}, "
         f"selected (final): {len(result['selected'])}")
    _log(f"  selected -> {config.SELECTED_FEATURES_PATH.name}, "
         f"report -> {config.SELECTION_REPORT_PATH.name}")

    # QA: selected features are a valid subset
    rep = qa.run_all_qa(selected=result["selected"])
    _qa_gate(rep, "feature selection", stop_on_fail=False)

    return {"ic_summary": ic_summary, "selection": result}


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #
def run_pipeline(use_cache: bool = True, verbose: bool = True) -> dict:
    """Run the whole pipeline end to end. Returns the key artifacts."""
    t0 = time.time()
    _log("=== PIPELINE START ===")

    inputs = load_inputs(use_cache=use_cache, verbose=verbose)
    raw_panel = compute_alpha_panel(inputs, use_cache=use_cache, verbose=verbose)
    feature_panel = build_feature_panel(raw_panel, inputs, use_cache=use_cache)
    outputs = evaluate_and_select(feature_panel)

    _log(f"=== PIPELINE DONE in {time.time()-t0:.0f}s ===")
    _log(f"Selected {len(outputs['selection']['selected'])} features. "
         f"Artifacts in {config.OUTPUT_DIR}")

    return {
        "inputs": inputs,
        "raw_panel": raw_panel,
        "feature_panel": feature_panel,
        "ic_summary": outputs["ic_summary"],
        "selection": outputs["selection"],
    }


if __name__ == "__main__":
    run_pipeline(use_cache=True, verbose=True)