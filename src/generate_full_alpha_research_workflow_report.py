"""Generate the final readable S&P 500 alpha research workflow report.

This is documentation only. It reads existing outputs and creates Markdown,
TXT, CSV, and PDF report artifacts without rerunning models or modifying
upstream model outputs.
"""

from __future__ import annotations

import argparse
import math
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPORT_TITLE = "S&P 500 Quant Alpha Research Pipeline Report: From Raw Data to Final Candidate Alpha Pool"
OUTPUT_DIR = Path("data/processed/final_readable_project_report")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final readable S&P 500 alpha workflow report.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def safe_read_csv(path: Path, missing: list[str]) -> pd.DataFrame:
    if not path.exists():
        missing.append(str(path))
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_read_text(path: Path, missing: list[str]) -> str:
    if not path.exists():
        missing.append(str(path))
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def parquet_summary(path: Path, missing: list[str]) -> dict[str, Any]:
    """Read cheap metadata and date/ticker coverage from parquet files."""
    if not path.exists():
        missing.append(str(path))
        return {"path": str(path), "exists": False}
    meta = pq.ParquetFile(path)
    columns = list(meta.schema.names)
    out: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "rows": int(meta.metadata.num_rows),
        "columns": len(columns),
        "alpha_columns": len([col for col in columns if col.startswith("alpha_")]),
        "has_date": "date" in columns,
        "has_ticker": "ticker" in columns,
    }
    read_cols = [col for col in ["date", "ticker"] if col in columns]
    if read_cols:
        df = pd.read_parquet(path, columns=read_cols)
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce")
            out["date_min"] = str(dates.min().date()) if dates.notna().any() else ""
            out["date_max"] = str(dates.max().date()) if dates.notna().any() else ""
            out["unique_dates"] = int(dates.dt.strftime("%Y-%m-%d").nunique())
        if "ticker" in df.columns:
            out["unique_tickers"] = int(df["ticker"].astype(str).nunique())
        if {"date", "ticker"}.issubset(df.columns):
            out["duplicate_date_ticker_rows"] = int(df.duplicated(["date", "ticker"]).sum())
    return out


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "n/a"
        return f"{float(value):,.{digits}f}"
    return str(value)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def summarize_inputs(missing: list[str]) -> dict[str, Any]:
    processed = Path("data/processed")
    summaries = {
        "raw_panel": parquet_summary(processed / "sp500_alpha_panel_raw.parquet", missing),
        "enriched_panel": parquet_summary(processed / "sp500_alpha_panel_enriched.parquet", missing),
        "ranked_panel": parquet_summary(processed / "sp500_alpha_panel_ranked.parquet", missing),
        "labels": parquet_summary(processed / "sp500_forward_return_labels.parquet", missing),
        "model_dataset": parquet_summary(processed / "sp500_model_dataset.parquet", missing),
    }
    return summaries


def alpha_group_counts(final_summary: pd.DataFrame) -> dict[str, int]:
    if final_summary.empty or "final_group" not in final_summary.columns:
        return {}
    return final_summary["final_group"].value_counts().to_dict()


def top_rows(df: pd.DataFrame, sort_col: str, columns: list[str], n: int = 10, ascending: bool = False) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=columns)
    frame = df.copy()
    if sort_col in frame.columns:
        frame[sort_col] = pd.to_numeric(frame[sort_col], errors="coerce")
        frame = frame.sort_values(sort_col, ascending=ascending)
    keep = [col for col in columns if col in frame.columns]
    return frame[keep].head(n)


def build_final_alpha_pool_table(final_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "alpha_id",
        "alpha_name",
        "final_group",
        "category",
        "recommended_usage",
        "plain_english_reason",
        "risk_warning",
    ]
    out = final_summary[[col for col in columns if col in final_summary.columns]].copy()
    out = out.rename(columns={"plain_english_reason": "key_supporting_evidence"})
    if "next_test" not in out.columns:
        out["next_test"] = out["final_group"].map(
            {
                "final_core_candidates": "Model A clean core and Model B core plus interaction candidates",
                "final_interaction_candidates": "Model B core plus interaction candidates",
                "final_watchlist_candidates": "Robustness review after first compact tests",
                "final_excluded_for_now": "Do not include in first compact model",
            }
        )
    return out


def build_workflow_index(status: dict[str, str]) -> pd.DataFrame:
    rows = [
        (1, "Raw alpha panel construction", "Convert raw S&P 500 files to a daily alpha panel.", "data/raw/input_batches", "data/processed/sp500_alpha_panel_raw.parquet", "src/sp500_batch_alpha_builder.py", status.get("raw_panel", "UNKNOWN"), "Alpha IDs 26-75 computed where data allowed."),
        (2, "Missing alpha enrichment", "Add market, factor, sector, and float-share dependent alphas.", "sp500_alpha_panel_raw.parquet + auxiliary data", "sp500_alpha_panel_enriched.parquet", "src/merge_market_sector_factor_data.py", status.get("enriched_panel", "UNKNOWN"), "Alpha 34 corrected to same-date same-sector median stock momentum."),
        (3, "Cross-sectional normalization", "Winsorize, z-score, rank, and sector-neutralize by date.", "sp500_alpha_panel_enriched.parquet", "sp500_alpha_panel_ranked.parquet", "src/build_cross_sectional_alpha_panel.py", status.get("ranked_panel", "UNKNOWN"), "Point-in-time alignment passed previously."),
        (4, "Forward return labels", "Create future return labels for 1d, 5d, and 21d horizons.", "ranked panel + OHLCV reconstruction", "sp500_forward_return_labels.parquet and sp500_model_dataset.parquet", "src/build_forward_return_labels.py", status.get("model_dataset", "UNKNOWN"), "Labels are never used as features."),
        (5, "Single-alpha validation", "Run Fama-MacBeth, Rank IC, and spread diagnostics.", "sp500_model_dataset.parquet", "regression_results_v2/*.csv", "src/run_single_alpha_regression_v2.py", status.get("linear", "UNKNOWN"), "Used 5d and 21d labels."),
        (6, "Lasso / ElasticNet validation", "Estimate marginal contribution after correlated alphas.", "sp500_model_dataset.parquet", "lasso_elasticnet/*.csv", "src/run_lasso_elasticnet_multifactor_model_v2.py", status.get("lasso", "UNKNOWN"), "Train-only preprocessing and chronological split."),
        (7, "True XGBoost Ranker", "Train grouped learning-to-rank models by trading date.", "sp500_model_dataset.parquet", "xgboost_ranker/*.csv", "src/run_xgboost_ranker_alpha_model_v2.py", status.get("ranker", "UNKNOWN"), "True xgboost_ranker backend, no fallback."),
        (8, "Multi-method selection", "Combine linear, regularized, and ranker evidence.", "regression + lasso + ranker outputs", "final_with_ranker/*.csv", "src/combine_multimethod_factor_selection_with_ranker.py", status.get("final_with_ranker", "UNKNOWN"), "49 rows, Alpha 35 skipped if available."),
        (9, "Diagnostic ablation", "Test force/exclude decisions from teammate diagnostics.", "sp500_model_dataset.parquet", "ranker_diagnostic_ablation/*.csv", "src/run_ranker_diagnostic_ablation.py", status.get("ablation", "UNKNOWN"), "Factor-selection validation only."),
        (10, "Final candidate alpha pool", "Select core, interaction, watchlist, and excluded alphas.", "all evidence sources", "final_candidate_alphas/*.csv", "src/select_final_candidate_alphas.py", status.get("final_candidates", "UNKNOWN"), "Current handoff for next mega-alpha tests."),
    ]
    return pd.DataFrame(rows, columns=["step_number", "step_name", "purpose", "main_input", "main_output", "script", "QA_status", "notes"])


def table_markdown(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 20) -> str:
    if df.empty:
        return "_No data available._"
    frame = df.copy()
    if columns:
        frame = frame[[col for col in columns if col in frame.columns]]
    frame = frame.head(max_rows)
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("\n", " ")[:180] for col in headers) + " |")
    return "\n".join(lines)


def build_markdown(context: dict[str, Any]) -> str:
    missing = context["missing"]
    panels = context["panels"]
    final_summary = context["final_summary"]
    core = context["core"]
    interaction = context["interaction"]
    watchlist = context["watchlist"]
    excluded = context["excluded"]
    clusters = context["clusters"]
    ranker_perf = context["ranker_perf"]
    ranker_summary = context["ranker_summary"]
    ablation = context["ablation"]
    workflow = context["workflow"]
    pool = context["pool"]
    linear_top = context["linear_top"]
    lasso = context["lasso"]
    sections: list[str] = []

    def add(title: str, body: str) -> None:
        sections.append(f"## {title}\n\n{body.strip()}\n")

    executive = f"""
This report documents the S&P 500 quantitative alpha research pipeline from raw data processing through the current final candidate alpha pool. The project began with a single-stock AAPL prototype and moved to an S&P 500 cross-sectional panel because stock-selection alphas are naturally evaluated by comparing many securities on the same date. The completed pipeline constructs alpha features, normalizes them cross-sectionally, creates forward-return labels, validates predictive structure through several methods, and selects alphas for future mega-alpha and portfolio-construction tests.

The current stage is factor research and alpha selection. It is not portfolio construction, not a transaction-cost backtest, and not production trading. The results identify candidates that deserve further testing; they do not establish live-trading profitability.

Current final pool counts:
- Core candidates: {len(core)}
- Interaction candidates: {len(interaction)}
- Watchlist candidates: {len(watchlist)}
- Excluded for now: {len(excluded)}
"""
    add("1. Executive Summary", executive)

    add(
        "2. Project Objective",
        """
The objective is to build a cross-sectional quantitative alpha research pipeline for S&P 500 stocks. The pipeline computes daily price/volume/factor-based alphas, normalizes them across stocks on each date, builds forward-return labels, validates alpha predictive power using multiple statistical and machine learning methods, and selects final candidate alphas for future mega-alpha modeling and portfolio construction.
""",
    )

    panel_rows = []
    for name, info in panels.items():
        panel_rows.append(
            {
                "panel": name,
                "rows": info.get("rows", "missing"),
                "columns": info.get("columns", "missing"),
                "tickers": info.get("unique_tickers", "n/a"),
                "date_range": f"{info.get('date_min', '')} to {info.get('date_max', '')}",
                "duplicate_date_ticker": info.get("duplicate_date_ticker_rows", "n/a"),
            }
        )
    add(
        "3. Data Universe and Raw Inputs",
        f"""
The universe is an S&P 500 stock panel assembled from input batches and auxiliary data. The project used equity OHLCV data, market data, factor returns, float shares, GICS sector mapping, and sector index data. Auxiliary data was needed for residual momentum, idiosyncratic volatility, market beta, turnover, sector metadata, and sector-neutral features.

{table_markdown(pd.DataFrame(panel_rows), max_rows=20)}

Data quality checks from prior steps included duplicate date-ticker validation, no inf/-inf checks, date alignment checks, and point-in-time feature construction. Missing alpha values were kept as NaN where data was unavailable rather than filled with artificial values.
""",
    )

    raw_info = panels["raw_panel"]
    add(
        "4. Step 1: Raw Alpha Panel Construction",
        f"""
Raw input files were processed ticker by ticker. Minute files were first detected as minute-level data, validated by timestamp, aggregated to daily OHLCV, and only then checked for duplicate daily dates. Daily inputs were standardized directly. The pipeline computed Alpha IDs 26-75, giving 50 price/volume and auxiliary-dependent alphas in the raw panel. Alpha 35 industry momentum was kept as unavailable/dropped by design because the required peer/industry basket was not built as a standalone signal.

Output path: `data/processed/sp500_alpha_panel_raw.parquet`

Key raw panel summary:
- Rows: {fmt(raw_info.get('rows'))}
- Columns: {fmt(raw_info.get('columns'))}
- Unique tickers: {fmt(raw_info.get('unique_tickers'))}
- Date range: {raw_info.get('date_min', 'n/a')} to {raw_info.get('date_max', 'n/a')}
- Alpha-related columns: {fmt(raw_info.get('alpha_columns'))}
- Duplicate date-ticker rows: {fmt(raw_info.get('duplicate_date_ticker_rows'))}
""",
    )

    add(
        "5. Step 2: Missing Alpha Enrichment",
        """
Several alphas required auxiliary datasets. Alpha 33 residual momentum uses factor residual returns. Alpha 59 idiosyncratic volatility uses factor residual volatility. Alpha 60 market beta uses market return covariance/variance. Alpha 66 turnover uses float shares and trading volume. Alpha 35 industry momentum remains dropped by design.

Alpha 34 sector-neutral momentum was corrected during the project. The correct formula is:

`alpha_34_i,t = stock_momentum_i,t - median(stock_momentum_j,t for stocks j in the same sector as i on date t)`

This formula uses same-date same-sector median stock momentum. It does not use sector mean, sector index momentum, sector ETF return, or future data. After this fix, the enriched panel, ranked panel, and model dataset were rebuilt; formula spot checks passed with zero mismatch in the validation run.
""",
    )

    ranked = panels["ranked_panel"]
    add(
        "6. Step 3: Cross-Sectional Normalization and Ranking",
        f"""
Raw alphas have different units and scales. The ranked panel therefore keeps the original raw value and adds same-date transformations: cross-sectional winsorized value, cross-sectional z-score, cross-sectional percentile rank, sector-neutral value, and sector-neutral rank. Transformations are by date only, so they do not use future information. Feature-specific NaNs remain NaN and rows are not globally dropped.

Output path: `data/processed/sp500_alpha_panel_ranked.parquet`

Summary:
- Rows: {fmt(ranked.get('rows'))}
- Columns: {fmt(ranked.get('columns'))}
- Unique tickers: {fmt(ranked.get('unique_tickers'))}
- Date range: {ranked.get('date_min', 'n/a')} to {ranked.get('date_max', 'n/a')}
- Duplicate date-ticker rows: {fmt(ranked.get('duplicate_date_ticker_rows'))}
""",
    )

    labels = panels["labels"]
    model = panels["model_dataset"]
    add(
        "7. Step 4: Forward Return Label Construction",
        f"""
Forward labels were built within ticker only. Close-to-close labels measure future close-to-close log returns over 1, 5, and 21 trading days. Tradable labels use next-day open as the entry assumption and future close as the exit, which is closer to realistic signal usage after the current-day close. VWAP labels use VWAP-based entry/exit where available. Excess-return labels subtract SPX market forward returns and are useful for stock-selection validation.

The current alpha validation focused on 5-day and 21-day horizons because these are more compatible with medium-frequency cross-sectional alpha research than 1-day trading. Labels are never used as features.

Label panel rows: {fmt(labels.get('rows'))}; model dataset rows: {fmt(model.get('rows'))}; model dataset date range: {model.get('date_min', 'n/a')} to {model.get('date_max', 'n/a')}.
""",
    )

    add(
        "8. Step 5: Single-Alpha Linear / Fama-MacBeth Validation",
        f"""
Single-alpha tests estimate whether each alpha has standalone predictive relationship with future returns. Pooled OLS can overstate significance because it ignores cross-sectional and time dependence. Fama-MacBeth is more appropriate for cross-sectional alpha validation because it runs date-level cross-sectional regressions and evaluates the time series of estimated coefficients. Newey-West t-stats are used because 5-day and 21-day labels overlap.

Rank IC measures same-date rank correlation between the alpha and future return. Decile spread measures the average future return difference between top-ranked and bottom-ranked stocks. Targets tested were forward_5d_excess_return, tradable_forward_5d_return, forward_21d_excess_return, and tradable_forward_21d_return.

Top linear evidence examples:

{table_markdown(linear_top, ['alpha_id','feature_name','target_label','feature_variant','linear_score','beta_t_stat_newey_west','rank_ic_t_stat_newey_west','spread_t_stat_newey_west'], 12)}

Alpha 34 results from before the formula fix should be treated as obsolete; the relevant downstream artifacts were rebuilt after the correction.
""",
    )

    add(
        "9. Step 6: Lasso / ElasticNet Multi-Factor Regression",
        f"""
Lasso and ElasticNet were used to test marginal contribution after accounting for correlated alphas. Lasso can force coefficients to zero, while ElasticNet combines L1 and L2 penalties and is generally more stable when features are correlated. Chronological train/validation/test splitting is required; imputation and scaling must be fit on the training block only to avoid leakage.

Readable Lasso/ElasticNet summary:

{table_markdown(lasso, ['alpha_id','alpha_name','lasso_selected_any','elasticnet_selected_any','lasso_selection_count','elasticnet_selection_count','coefficient_direction','interpretation'], 12)}

Zero-selection alphas should not automatically be deleted. In correlated alpha libraries, a zero coefficient may mean the alpha overlaps with another stronger feature, not that the economic idea is useless.
""",
    )

    best_ranker = pd.DataFrame()
    if not ranker_perf.empty and "validation_mean_daily_rank_ic" in ranker_perf.columns:
        best_ranker = ranker_perf.sort_values("validation_mean_daily_rank_ic", ascending=False).head(1)
    add(
        "10. Step 7: True XGBoost Ranker",
        f"""
The true XGBoost Ranker was used because the project is fundamentally about ranking stocks cross-sectionally, not predicting raw price levels. In learning-to-rank, each trading date is treated as one query/group. Training labels are within-date non-negative relevance grades derived from future-return ranks; validation and test evaluation still use real continuous future returns.

The earlier regressor fallback was replaced with true `xgboost_ranker` evidence. The confirmed best ranker result was:
- Target: tradable_forward_21d_return
- Feature set: cs_rank
- Objective: rank:pairwise
- Validation Rank IC: 0.172193
- Test Rank IC: 0.080886
- Test top-minus-bottom spread: 0.029017

Top ranker alphas by readable summary:

{table_markdown(ranker_summary, ['alpha_id','alpha_name','best_feature_variant','best_target_label','best_objective','average_gain_importance','best_importance_rank'], 12)}
""",
    )

    final_with_ranker = context["final_with_ranker"]
    add(
        "11. Step 8: Multi-Method Factor Selection",
        f"""
The multi-method selector combined linear/Fama-MacBeth evidence, Lasso/ElasticNet evidence, and true XGBoost Ranker evidence. It grouped alphas into core_alpha, linear_only_alpha, nonlinear_alpha, redundant_alpha, watchlist_alpha, and rejected_alpha.

The final-with-ranker summary had {len(final_with_ranker) if not final_with_ranker.empty else 'n/a'} rows, skipped Alpha 35, and confirmed the true ranker backend. Initially, Alpha 55 ATR emerged as a core multi-method candidate. Later diagnostic review and the teammate IC/correlation report refined the final interpretation into the current core/interaction/watchlist structure.
""",
    )

    add(
        "12. Step 9: Teammate Diagnostic Report and Ablation",
        f"""
The teammate diagnostic report raised concerns that Alpha 55 ATR and Alpha 68 ADV trend could be noisy, suggested Alpha 32 momentum might be underused, and identified Alpha 54 and Alpha 65 as cleaner features. We ran diagnostic ablations without changing production outputs.

Ablation summary:

{table_markdown(ablation, ['experiment_name','best_target_label','best_objective','validation_rank_ic','test_rank_ic','test_top_minus_bottom_spread','recommendation'], 10)}

Conclusions: Alpha 32 was already included, so force-including it changed nothing. Removing Alpha 55 hurt primary test Rank IC and spread. Removing Alpha 68 also hurt primary test Rank IC and spread. Alpha 54 and Alpha 65 remained stable backbone features. The recommended current ranker configuration remained baseline_current / tradable_forward_21d_return / cs_rank / rank:pairwise.
""",
    )

    add(
        "13. Step 10: Rank IC + Correlation Redundancy Report",
        f"""
The teammate IC/correlation report used a significance threshold of |t_stat| >= 2.0 and a redundancy threshold of |rank-corr| >= 0.8. It reviewed 49 alphas, 14 survived the significance filter, and 10 were final selected after redundancy clustering.

Final selected IC/correlation alpha set:
- Alpha 54 Daily high-low range
- Alpha 32 12-1 month momentum
- Alpha 59 Idiosyncratic volatility
- Alpha 56 21-day realized volatility
- Alpha 60 Market beta
- Alpha 58 Downside volatility
- Alpha 65 Dollar volume
- Alpha 53 Bollinger reversion
- Alpha 55 Average true range
- Alpha 66 Turnover

Redundancy clusters:

{table_markdown(clusters, ['cluster_id','kept_alpha','dropped_alphas','reason','rank_corr_to_head'], 12)}
""",
    )

    add(
        "14. Final Candidate Alpha Pool",
        f"""
The final candidate pool is the current handoff for the next mega-alpha and portfolio-construction stage. Core candidates are the cleanest compact candidates. Interaction candidates are not clean standalone alphas but may help nonlinear ranker behavior. Watchlist candidates have mixed evidence or redundancy. Excluded-for-now candidates are not permanently deleted.

Core candidates:

{table_markdown(core, ['alpha_id','alpha_name','category','recommended_usage','risk_warning'], 20)}

Interaction candidates:

{table_markdown(interaction, ['alpha_id','alpha_name','category','recommended_usage','risk_warning'], 20)}

Counts: core {len(core)}, interaction {len(interaction)}, watchlist {len(watchlist)}, excluded for now {len(excluded)}.
""",
    )

    add(
        "15. Current Recommended Next Steps",
        """
Recommended model tests:
- Model A: clean core only.
- Model B: clean core + interaction candidates.
- Model C: current baseline ranker.
- Model D: clean core excluding high-risk volatility features.
- Model E: clean core + risk/liquidity features with exposure controls.

Recommended portfolio construction tests:
- Long top decile and short bottom decile.
- Market-neutral, beta-neutral, and sector-neutral portfolio options.
- Rebalance-frequency analysis.

Recommended transaction-cost backtest:
- Turnover, spread/slippage, capacity, rebalance frequency, and gross versus net return.

Recommended risk analysis:
- Beta, sector, volatility, liquidity, and drawdown exposure.

Production considerations:
- Point-in-time data, survivorship bias, vendor quality, execution assumptions, and monitoring.
""",
    )

    add(
        "16. Limitations and Warnings",
        """
This report does not establish live-trading profitability. A current S&P 500 universe may contain survivorship bias. Some sector mapping may not be fully point-in-time. Alpha 35 was dropped by design. Factor data date limits affect some auxiliary-dependent alphas. The XGBoost Ranker result is promising but must be portfolio-tested. Transaction costs may reduce or eliminate apparent alpha. Volatility/liquidity features may represent risk premia rather than pure stock-selection alpha. IC and ranker performance do not guarantee PnL. Additional out-of-sample and walk-forward robustness is required.
""",
    )

    appendix = f"""
### Key Output File Index

{table_markdown(workflow, ['step_number','step_name','main_output','script','QA_status'], 20)}

### Definitions

- Alpha: a predictive signal intended to rank or forecast future returns.
- Cross-sectional rank: ranking stocks against each other on the same date.
- Rank IC: Spearman rank correlation between alpha scores and future returns.
- ICIR: mean IC divided by IC standard deviation.
- Fama-MacBeth regression: repeated cross-sectional regression by date, with coefficient time-series analysis.
- Newey-West t-stat: t-stat adjusted for autocorrelation, useful for overlapping labels.
- Lasso: L1-regularized regression that can select sparse features.
- ElasticNet: regression combining L1 and L2 regularization.
- XGBoost Ranker: gradient-boosted tree ranking model using grouped queries.
- Ablation: controlled inclusion/exclusion experiment.
- Forward return label: future return target built after the feature date.
- Tradable return: return based on a practical next-open entry assumption.
- Excess return: stock forward return minus market forward return.

### Final Alpha Pool Table

{table_markdown(pool, ['alpha_id','alpha_name','final_group','category','recommended_usage','risk_warning','next_test'], 60)}
"""
    add("17. Appendix", appendix)

    missing_section = "\n".join(f"- {item}" for item in sorted(set(missing))) if missing else "- None."
    md = f"# {REPORT_TITLE}\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n## Table of Contents\n\n"
    for section in sections:
        title = section.split("\n", 1)[0].replace("## ", "")
        md += f"- {title}\n"
    md += f"\n## Missing Referenced Files\n\n{missing_section}\n\n"
    md += "\n".join(sections)
    return md


def markdown_to_text(markdown: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", markdown)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = text.replace("|", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def make_paragraph(text: Any, style: Any) -> Any:
    from reportlab.platypus import Paragraph

    escaped = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    escaped = escaped.replace("\n", "<br/>")
    return Paragraph(escaped, style)


def split_markdown_blocks(markdown: str) -> list[tuple[str, Any]]:
    blocks: list[tuple[str, Any]] = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("# "):
            blocks.append(("title", line[2:].strip()))
            i += 1
        elif line.startswith("## "):
            blocks.append(("h2", line[3:].strip()))
            i += 1
        elif line.startswith("### "):
            blocks.append(("h3", line[4:].strip()))
            i += 1
        elif line.startswith("| "):
            table_lines = []
            while i < len(lines) and lines[i].startswith("| "):
                table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= 2:
                blocks.append(("table", table_lines))
        elif line.startswith("- "):
            bullets = []
            while i < len(lines) and lines[i].startswith("- "):
                bullets.append(lines[i][2:].strip())
                i += 1
            blocks.append(("bullets", bullets))
        else:
            paras = [line.strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "| ", "- ")):
                paras.append(lines[i].strip())
                i += 1
            blocks.append(("para", " ".join(paras)))
    return blocks


def parse_markdown_table(table_lines: list[str]) -> list[list[str]]:
    rows = []
    for idx, line in enumerate(table_lines):
        if idx == 1 and set(line.replace("|", "").replace(" ", "")) <= {"-"}:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def build_pdf(markdown: str, pdf_path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import LongTable, PageBreak, SimpleDocTemplate, Spacer, TableStyle

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], alignment=TA_CENTER, fontSize=22, leading=28, spaceAfter=18)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], alignment=TA_CENTER, fontSize=11, leading=15, textColor=colors.HexColor("#444444"))
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, leading=20, spaceBefore=12, spaceAfter=8, textColor=colors.HexColor("#12355B"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, leading=16, spaceBefore=8, spaceAfter=6, textColor=colors.HexColor("#1D4E89"))
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=8.5, leading=11.5, spaceAfter=6)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=14, firstLineIndent=-7)
    small = ParagraphStyle("Small", parent=body, fontSize=7, leading=9)
    table_header = ParagraphStyle("TableHeader", parent=small, fontName="Helvetica-Bold", textColor=colors.white)
    table_cell = ParagraphStyle("TableCell", parent=small)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.55 * inch,
        title=REPORT_TITLE,
        author="Icarus Fund, LLC Quant Research",
    )

    story: list[Any] = []
    story.append(Spacer(1, 1.7 * inch))
    story.append(make_paragraph(REPORT_TITLE, title_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(make_paragraph("Prepared for quant research, portfolio management, and stakeholder review.", subtitle_style))
    story.append(make_paragraph(f"Generated {datetime.now().strftime('%B %d, %Y')}", subtitle_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(make_paragraph("Important: this is a factor research and alpha selection report. It is not live-trading evidence.", subtitle_style))
    story.append(PageBreak())

    for kind, content in split_markdown_blocks(markdown):
        if kind == "title":
            continue
        if kind == "h2":
            story.append(make_paragraph(content, h1))
        elif kind == "h3":
            story.append(make_paragraph(content, h2))
        elif kind == "para":
            story.append(make_paragraph(content, body))
        elif kind == "bullets":
            for item in content:
                story.append(make_paragraph(f"- {item}", bullet))
        elif kind == "table":
            rows = parse_markdown_table(content)
            if not rows:
                continue
            max_cols = len(rows[0])
            usable_width = letter[0] - doc.leftMargin - doc.rightMargin
            col_width = usable_width / max_cols
            table_data = []
            for r_idx, row in enumerate(rows):
                style = table_header if r_idx == 0 else table_cell
                table_data.append([make_paragraph(cell, style) for cell in row])
            table = LongTable(table_data, colWidths=[col_width] * max_cols, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D4E89")),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D3DF")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 8))

    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawRightString(letter[0] - 0.55 * inch, 0.3 * inch, f"Page {document.page}")
        canvas.drawString(0.55 * inch, 0.3 * inch, "S&P 500 Quant Alpha Research Pipeline Report")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def gather_context(missing: list[str]) -> dict[str, Any]:
    processed = Path("data/processed")
    panels = summarize_inputs(missing)
    linear = safe_read_csv(processed / "regression_results_v2/combined_alpha_validation_summary_v2.csv", missing)
    lasso = safe_read_csv(processed / "factor_selection_multimethod_v2/lasso_elasticnet/lasso_elasticnet_readable_summary_by_alpha.csv", missing)
    ranker_summary = safe_read_csv(processed / "factor_selection_multimethod_v2/xgboost_ranker/xgboost_ranker_readable_summary_by_alpha.csv", missing)
    ranker_perf = safe_read_csv(processed / "factor_selection_multimethod_v2/xgboost_ranker/xgboost_ranker_performance_v2.csv", missing)
    final_with_ranker = safe_read_csv(processed / "factor_selection_multimethod_v2/final_with_ranker/combined_readable_summary_by_alpha_with_ranker.csv", missing)
    ablation = safe_read_csv(processed / "ranker_diagnostic_ablation/ranker_ablation_readable_summary.csv", missing)
    clusters = safe_read_csv(processed / "final_candidate_alphas/final_redundancy_clusters.csv", missing)
    final_summary = safe_read_csv(processed / "final_candidate_alphas/final_candidate_alpha_summary.csv", missing)
    core = safe_read_csv(processed / "final_candidate_alphas/final_core_candidates.csv", missing)
    interaction = safe_read_csv(processed / "final_candidate_alphas/final_interaction_candidates.csv", missing)
    watchlist = safe_read_csv(processed / "final_candidate_alphas/final_watchlist_candidates.csv", missing)
    excluded = safe_read_csv(processed / "final_candidate_alphas/final_excluded_for_now.csv", missing)
    safe_read_text(processed / "factor_selection_multimethod_v2/feature_selection_report.txt", missing)
    safe_read_text(processed / "ranker_diagnostic_ablation/ranker_ablation_report.md", missing)

    linear_top = pd.DataFrame()
    if not linear.empty:
        data = linear.copy()
        for col in ["beta_t_stat_newey_west", "rank_ic_t_stat_newey_west", "spread_t_stat_newey_west"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data["linear_score"] = (
            0.40 * (data["beta_t_stat_newey_west"].abs() / 3).clip(0, 1)
            + 0.35 * (data["rank_ic_t_stat_newey_west"].abs() / 3).clip(0, 1)
            + 0.25 * (data["spread_t_stat_newey_west"].abs() / 3).clip(0, 1)
        )
        linear_top = data.sort_values("linear_score", ascending=False).drop_duplicates("alpha_id").head(12)

    status = {
        "raw_panel": "PASS" if panels["raw_panel"].get("exists") else "MISSING",
        "enriched_panel": "PASS" if panels["enriched_panel"].get("exists") else "MISSING",
        "ranked_panel": "PASS" if panels["ranked_panel"].get("exists") else "MISSING",
        "model_dataset": "PASS" if panels["model_dataset"].get("exists") and panels["labels"].get("exists") else "MISSING",
        "linear": "PASS" if not linear.empty else "MISSING",
        "lasso": "PASS" if not lasso.empty else "MISSING",
        "ranker": "PASS" if not ranker_summary.empty and not ranker_perf.empty else "MISSING",
        "final_with_ranker": "PASS" if not final_with_ranker.empty else "MISSING",
        "ablation": "PASS" if not ablation.empty else "MISSING",
        "final_candidates": "PASS" if not final_summary.empty else "MISSING",
    }
    workflow = build_workflow_index(status)
    pool = build_final_alpha_pool_table(final_summary) if not final_summary.empty else pd.DataFrame()

    return {
        "missing": missing,
        "panels": panels,
        "linear_top": linear_top,
        "lasso": lasso,
        "ranker_summary": ranker_summary,
        "ranker_perf": ranker_perf,
        "final_with_ranker": final_with_ranker,
        "ablation": ablation,
        "clusters": clusters,
        "final_summary": final_summary,
        "core": core,
        "interaction": interaction,
        "watchlist": watchlist,
        "excluded": excluded,
        "workflow": workflow,
        "pool": pool,
    }


def qa_pdf(pdf_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": pdf_path.exists(),
        "size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "page_count": 0,
        "readable_pages": 0,
        "status": "FAIL",
    }
    if not pdf_path.exists() or result["size_bytes"] <= 0:
        return result
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    result["page_count"] = len(reader.pages)
    readable = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        if len(text.strip()) > 40:
            readable += 1
    result["readable_pages"] = readable
    result["status"] = "PASS" if result["page_count"] > 0 and readable > 0 else "FAIL"
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    context = gather_context(missing)
    md = build_markdown(context)
    txt = markdown_to_text(md)

    md_path = args.output_dir / "full_alpha_research_workflow_report.md"
    txt_path = args.output_dir / "full_alpha_research_workflow_report.txt"
    pdf_path = args.output_dir / "full_alpha_research_workflow_report.pdf"
    pool_path = args.output_dir / "final_alpha_pool_summary_table.csv"
    workflow_path = args.output_dir / "project_workflow_index.csv"

    md_path.write_text(md, encoding="utf-8")
    txt_path.write_text(txt, encoding="utf-8")
    context["pool"].to_csv(pool_path, index=False)
    context["workflow"].to_csv(workflow_path, index=False)

    pdf_status = "FAIL"
    pdf_error = ""
    try:
        build_pdf(md, pdf_path)
        pdf_qa = qa_pdf(pdf_path)
        pdf_status = pdf_qa["status"]
    except Exception as exc:
        pdf_error = f"{type(exc).__name__}: {exc}"
        pdf_qa = {"status": "FAIL", "exists": pdf_path.exists(), "size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0, "page_count": 0, "readable_pages": 0}

    report_status = "PASS" if md_path.exists() and txt_path.exists() and pool_path.exists() and workflow_path.exists() and pdf_status == "PASS" else "FAIL"

    print(f"PDF report output path: {pdf_path}")
    print(f"Markdown backup path: {md_path}")
    print(f"TXT backup path: {txt_path}")
    print(f"final_alpha_pool_summary_table.csv path: {pool_path}")
    print(f"project_workflow_index.csv path: {workflow_path}")
    print(f"number of final core alphas: {len(context['core'])}")
    print(f"number of interaction alphas: {len(context['interaction'])}")
    print(f"number of watchlist alphas: {len(context['watchlist'])}")
    print(f"number excluded for now: {len(context['excluded'])}")
    print("missing referenced files:")
    if missing:
        for item in sorted(set(missing)):
            print(f"  - {item}")
    else:
        print("  - None")
    print(f"PDF QA: exists={pdf_qa['exists']} size_bytes={pdf_qa['size_bytes']} pages={pdf_qa['page_count']} readable_pages={pdf_qa['readable_pages']}")
    if pdf_error:
        print(f"PDF generation error: {pdf_error}")
    print(f"final readable PDF report generation status: {report_status}")
    if report_status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
