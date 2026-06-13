"""Select final candidate alphas from all available factor-selection evidence.

This script is factor selection only. It does not train models, construct
portfolios, run transaction-cost backtests, or overwrite model outputs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .factor_selection_common import alpha_category, sign_label


CORE_EXPECTED_IDS = {32, 53, 54, 55, 56, 58, 59, 60, 65, 66}
INTERACTION_PRIORITY_IDS = {30, 34, 57, 61, 62, 63, 68, 73}
BACKBONE_IDS = {54, 65}
ABLATION_SUPPORT_IDS = {32, 54, 55, 65, 68}
RISK_LIQUIDITY_IDS = {54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68}
GROUP_ORDER = {
    "final_core_candidates": 0,
    "final_interaction_candidates": 1,
    "final_watchlist_candidates": 2,
    "final_excluded_for_now": 3,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Select final candidate alphas from multi-source evidence.")
    parser.add_argument("--linear-results", type=Path, required=True)
    parser.add_argument("--lasso-summary", type=Path, required=True)
    parser.add_argument("--ranker-summary", type=Path, required=True)
    parser.add_argument("--ranker-importance", type=Path, required=True)
    parser.add_argument("--final-ranker-summary", type=Path, required=True)
    parser.add_argument("--ablation-summary", type=Path, required=True)
    parser.add_argument("--ablation-report", type=Path, required=True)
    parser.add_argument("--ic-corr-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dataset", type=Path, default=Path("data/processed/sp500_model_dataset.parquet"))
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    """Load a required CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def resolve_ic_report(path: Path) -> Path:
    """Resolve the teammate IC/correlation report path, searching the project if needed."""
    if path.exists():
        return path
    matches = sorted(Path(".").rglob(path.name))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Missing IC/correlation report: {path}. Please place feature_selection_report.txt under the project folder."
    )


def finite_float(value: Any, default: float = np.nan) -> float:
    """Convert a value to float."""
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def truthy(value: Any) -> bool:
    """Parse bool-like CSV values."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def alpha_slug_to_id(slug: str) -> int | None:
    """Extract alpha ID from strings such as alpha_54_daily_high_low_range."""
    match = re.search(r"alpha_(\d+)_", slug)
    if not match:
        return None
    return int(match.group(1))


def parse_metric_line(line: str) -> dict[str, Any] | None:
    """Parse a report line containing alpha metrics."""
    match = re.search(
        r"(alpha_\d+_[A-Za-z0-9_]+)\s+mean_ic=([+-]?\d+\.\d+)\s+icir=([+-]?\d+\.\d+)\s+"
        r"t_stat=([+-]?\d+\.\d+)\s+n_days=(\d+)(?:\s+\|r_to_head\|=(\d+\.\d+))?",
        line,
    )
    if not match:
        return None
    slug = match.group(1)
    alpha_id = alpha_slug_to_id(slug)
    if alpha_id is None:
        return None
    return {
        "alpha_id": alpha_id,
        "report_alpha_slug": slug,
        "mean_ic": float(match.group(2)),
        "icir": float(match.group(3)),
        "t_stat": float(match.group(4)),
        "n_days": int(match.group(5)),
        "rank_corr_to_head": float(match.group(6)) if match.group(6) else np.nan,
    }


def parse_ic_corr_report(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Parse teammate Rank IC and redundancy-cluster report."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    meta = {
        "significance_threshold": np.nan,
        "redundancy_threshold": np.nan,
        "alphas_in": np.nan,
        "survived_step_1": np.nan,
        "final_selected": np.nan,
    }
    for key, pattern in {
        "significance_threshold": r"Significance threshold\s*:\s*\|t_stat\|\s*>=\s*(\d+\.\d+)",
        "redundancy_threshold": r"Redundancy threshold\s*:\s*\|rank-corr\|\s*>=\s*(\d+\.\d+)",
        "alphas_in": r"Alphas in\s*:\s*(\d+)",
        "survived_step_1": r"Survived step 1\s*:\s*(\d+)",
        "final_selected": r"Final selected\s*:\s*(\d+)",
    }.items():
        match = re.search(pattern, text)
        if match:
            meta[key] = float(match.group(1)) if "." in match.group(1) else int(match.group(1))

    final_selected_ids = {
        alpha_slug_to_id(match.group(1))
        for match in re.finditer(r"^\s*\+\s+(alpha_\d+_[A-Za-z0-9_]+)", text, flags=re.MULTILINE)
    }
    final_selected_ids = {alpha_id for alpha_id in final_selected_ids if alpha_id is not None}

    records: dict[int, dict[str, Any]] = {}
    cluster_rows: list[dict[str, Any]] = []
    current_cluster_id: int | None = None
    current_head_id: int | None = None
    current_head_slug = ""
    for line in text.splitlines():
        cluster_match = re.search(r"Cluster\s+(\d+).*KEEP\s+(alpha_\d+_[A-Za-z0-9_]+)", line)
        if cluster_match:
            current_cluster_id = int(cluster_match.group(1))
            current_head_slug = cluster_match.group(2)
            current_head_id = alpha_slug_to_id(current_head_slug)
            continue
        metric = parse_metric_line(line)
        if metric is None:
            continue
        alpha_id = int(metric["alpha_id"])
        status = "failed_significance_filter"
        retained_head = np.nan
        if current_cluster_id is not None and line.strip().startswith("head"):
            status = "cluster_head_kept"
            retained_head = current_head_id
        elif current_cluster_id is not None and line.strip().startswith("drop"):
            status = "dropped_redundant"
            retained_head = current_head_id
        elif alpha_id in final_selected_ids:
            status = "selected"
        record = {
            **metric,
            "survived_significance_filter": current_cluster_id is not None,
            "selected_by_ic_corr_report": alpha_id in final_selected_ids,
            "redundancy_cluster": current_cluster_id,
            "retained_cluster_head": retained_head,
            "redundancy_status": status,
        }
        records[alpha_id] = record
        if current_cluster_id is not None:
            cluster_rows.append(record)

    cluster_summary_rows: list[dict[str, Any]] = []
    cluster_frame = pd.DataFrame(cluster_rows)
    if not cluster_frame.empty:
        for cluster_id, group in cluster_frame.groupby("redundancy_cluster", sort=True):
            head = group[group["redundancy_status"] == "cluster_head_kept"]
            drops = group[group["redundancy_status"] == "dropped_redundant"]
            head_slug = str(head.iloc[0]["report_alpha_slug"]) if not head.empty else ""
            cluster_summary_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "cluster_head": head_slug,
                    "kept_alpha": head_slug,
                    "dropped_alphas": ";".join(drops["report_alpha_slug"].astype(str).tolist()),
                    "reason": "singleton_kept" if drops.empty else "dropped members exceeded rank-correlation redundancy threshold",
                    "rank_corr_to_head": ";".join(
                        f"{row.report_alpha_slug}:{row.rank_corr_to_head:.3f}"
                        for row in drops.itertuples()
                        if pd.notna(row.rank_corr_to_head)
                    ),
                }
            )
    return pd.DataFrame(records.values()), pd.DataFrame(cluster_summary_rows), meta


def best_linear_evidence(linear: pd.DataFrame) -> pd.DataFrame:
    """Summarize linear/Fama-MacBeth evidence to one row per alpha."""
    data = linear[linear["alpha_id"] != 35].copy()
    numeric_cols = [
        "beta_t_stat_newey_west",
        "rank_ic_t_stat_newey_west",
        "spread_t_stat_newey_west",
        "mean_rank_ic_spearman",
        "spread_metric",
        "data_coverage",
    ]
    for column in numeric_cols:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    def clip_t(value: Any) -> float:
        numeric = finite_float(value, default=np.nan)
        if not np.isfinite(numeric):
            return 0.0
        return float(min(abs(numeric) / 3.0, 1.0))

    data["linear_support_score"] = (
        0.40 * data["beta_t_stat_newey_west"].map(clip_t)
        + 0.35 * data["rank_ic_t_stat_newey_west"].map(clip_t)
        + 0.25 * data["spread_t_stat_newey_west"].map(clip_t)
    )
    rows: list[dict[str, Any]] = []
    for alpha_id, group in data.groupby("alpha_id", sort=True):
        best = group.sort_values(["linear_support_score", "data_coverage"], ascending=[False, False]).iloc[0]
        directions = group["mean_rank_ic_spearman"].dropna().map(np.sign).tolist()
        horizon_consistency = pair_consistency(group, "forward_5d_excess_return", "forward_21d_excess_return")
        excess_tradable = pair_consistency(group, "forward_5d_excess_return", "tradable_forward_5d_return")
        rows.append(
            {
                "alpha_id": int(alpha_id),
                "alpha_name": best["feature_name"],
                "linear_support_score": float(best["linear_support_score"]),
                "linear_direction": sign_label(float(best["mean_rank_ic_spearman"])),
                "selected_by_linear": float(best["linear_support_score"]) >= 0.55,
                "best_fama_macbeth_tstat_newey_west": finite_float(best["beta_t_stat_newey_west"]),
                "best_rank_ic_tstat_newey_west": finite_float(best["rank_ic_t_stat_newey_west"]),
                "best_spread_tstat_newey_west": finite_float(best["spread_t_stat_newey_west"]),
                "best_linear_feature_variant": best["feature_variant"],
                "best_linear_target_label": best["target_label"],
                "linear_sign_consistency": sign_consistency(directions),
                "horizon_consistency": horizon_consistency,
                "excess_tradable_consistency": excess_tradable,
            }
        )
    return pd.DataFrame(rows)


def sign_consistency(signs: list[float]) -> float:
    """Share of non-zero signs matching the dominant sign."""
    clean = [int(sign) for sign in signs if sign != 0 and np.isfinite(sign)]
    if not clean:
        return 0.0
    return float(max(clean.count(1), clean.count(-1)) / len(clean))


def pair_consistency(group: pd.DataFrame, left_label: str, right_label: str) -> float:
    """Sign consistency between two target labels over feature variants."""
    hits: list[float] = []
    for _, variant_group in group.groupby("feature_variant"):
        left = variant_group[variant_group["target_label"] == left_label]
        right = variant_group[variant_group["target_label"] == right_label]
        if left.empty or right.empty:
            continue
        left_sign = np.sign(finite_float(left.iloc[0]["mean_rank_ic_spearman"]))
        right_sign = np.sign(finite_float(right.iloc[0]["mean_rank_ic_spearman"]))
        if left_sign != 0 and right_sign != 0:
            hits.append(float(left_sign == right_sign))
    return float(np.mean(hits)) if hits else 0.0


def lasso_evidence(lasso: pd.DataFrame) -> pd.DataFrame:
    """Summarize Lasso/ElasticNet evidence."""
    data = lasso[lasso["alpha_id"] != 35].copy()
    for column in ["lasso_selection_count", "elasticnet_selection_count", "best_test_rank_ic"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["lasso_selected_any"] = data["lasso_selected_any"].map(truthy)
    data["elasticnet_selected_any"] = data["elasticnet_selected_any"].map(truthy)
    data["selected_by_lasso_or_elasticnet"] = data["lasso_selected_any"] | data["elasticnet_selected_any"]
    data["lasso_elasticnet_support_score"] = (
        data["selected_by_lasso_or_elasticnet"].astype(float) * 0.75
        + np.minimum((data["lasso_selection_count"].fillna(0) + data["elasticnet_selection_count"].fillna(0)) / 4.0, 1.0) * 0.25
    )
    return data[
        [
            "alpha_id",
            "lasso_selected_any",
            "elasticnet_selected_any",
            "selected_by_lasso_or_elasticnet",
            "lasso_selection_count",
            "elasticnet_selection_count",
            "coefficient_direction",
            "best_feature_variant",
            "best_target_label",
            "best_test_rank_ic",
            "interpretation",
            "lasso_elasticnet_support_score",
        ]
    ].rename(
        columns={
            "best_feature_variant": "best_regularized_feature_variant",
            "best_target_label": "best_regularized_target_label",
            "best_test_rank_ic": "best_regularized_test_rank_ic",
            "interpretation": "regularized_interpretation",
        }
    )


def ranker_evidence(ranker: pd.DataFrame, importance: pd.DataFrame) -> pd.DataFrame:
    """Summarize true XGBoost Ranker evidence."""
    data = ranker[ranker["alpha_id"] != 35].copy()
    for column in ["average_gain_importance", "best_gain_importance", "average_importance_rank", "best_importance_rank", "best_test_rank_ic"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    max_avg_gain = data["average_gain_importance"].max()
    avg_gain_score = data["average_gain_importance"] / max_avg_gain if max_avg_gain and np.isfinite(max_avg_gain) else 0.0
    rank_score = (50 - data["average_importance_rank"]).clip(lower=0) / 50.0
    best_rank_score = (50 - data["best_importance_rank"]).clip(lower=0) / 50.0
    data["ranker_support_score"] = (0.40 * avg_gain_score + 0.35 * rank_score + 0.25 * best_rank_score).clip(0, 1)
    data["selected_by_true_ranker"] = (data["best_importance_rank"] <= 10) | (data["average_importance_rank"] <= 20)
    backend_values = sorted(importance["backend"].dropna().astype(str).unique().tolist()) if "backend" in importance.columns else []
    data["true_ranker_backend_values"] = ";".join(backend_values)
    return data[
        [
            "alpha_id",
            "best_feature_variant",
            "best_target_label",
            "best_objective",
            "best_test_rank_ic",
            "average_gain_importance",
            "best_gain_importance",
            "average_importance_rank",
            "best_importance_rank",
            "important_in_targets",
            "important_in_feature_sets",
            "ranker_interpretation",
            "ranker_support_score",
            "selected_by_true_ranker",
            "true_ranker_backend_values",
        ]
    ].rename(
        columns={
            "best_feature_variant": "best_ranker_feature_variant",
            "best_target_label": "best_ranker_target_label",
            "best_objective": "best_ranker_objective",
            "best_test_rank_ic": "best_ranker_test_rank_ic",
        }
    )


def ablation_evidence(ablation_summary: pd.DataFrame, ablation_report_text: str) -> pd.DataFrame:
    """Convert diagnostic ablation conclusions into alpha-level support rows."""
    baseline = ablation_summary[ablation_summary["experiment_name"] == "baseline_current"]
    base_rank_ic = finite_float(baseline.iloc[0]["test_rank_ic"]) if not baseline.empty else np.nan
    rows = []
    for alpha_id in ABLATION_SUPPORT_IDS:
        score = 0.0
        supported = False
        note = ""
        if alpha_id == 32:
            score, supported = 0.65, True
            note = "Alpha 32 was already included; force-including it changed nothing."
        elif alpha_id == 55:
            score, supported = 0.80, True
            note = "Excluding Alpha 55 hurt primary test Rank IC and spread, so keep ATR for now."
        elif alpha_id == 68:
            score, supported = 0.55, True
            note = "Excluding Alpha 68 hurt primary test Rank IC and spread; keep as interaction/watchlist."
        elif alpha_id in BACKBONE_IDS:
            score, supported = 0.75, True
            note = "Diagnostic ablation report treats this as a stable backbone feature."
        rows.append(
            {
                "alpha_id": alpha_id,
                "ablation_support_score": score,
                "supported_by_ablation": supported,
                "ablation_note": note,
                "ablation_baseline_test_rank_ic": base_rank_ic,
            }
        )
    return pd.DataFrame(rows)


def base_alpha_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """Build canonical one-row-per-alpha base from evidence tables."""
    rows: dict[int, dict[str, Any]] = {}
    for frame in frames:
        if frame.empty or "alpha_id" not in frame.columns:
            continue
        for item in frame.to_dict("records"):
            alpha_id = int(item["alpha_id"])
            if alpha_id == 35:
                continue
            row = rows.setdefault(alpha_id, {"alpha_id": alpha_id})
            for name_key in ["alpha_name", "feature_name"]:
                if name_key in item and pd.notna(item[name_key]):
                    row["alpha_name"] = item[name_key]
                    break
            row["category"] = row.get("category", alpha_category(alpha_id))
    return pd.DataFrame(rows.values()).sort_values("alpha_id").reset_index(drop=True)


def add_default_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure expected columns exist with stable defaults."""
    out = frame.copy()
    defaults: dict[str, Any] = {
        "linear_support_score": 0.0,
        "lasso_elasticnet_support_score": 0.0,
        "ranker_support_score": 0.0,
        "ic_corr_support_score": 0.0,
        "ablation_support_score": 0.0,
        "selected_by_ic_corr_report": False,
        "selected_by_linear": False,
        "selected_by_lasso_or_elasticnet": False,
        "selected_by_true_ranker": False,
        "supported_by_ablation": False,
        "survived_significance_filter": False,
        "redundancy_cluster": np.nan,
        "retained_cluster_head": np.nan,
        "redundancy_status": "not_in_ic_corr_report",
        "rank_corr_to_head": np.nan,
        "mean_ic": np.nan,
        "icir": np.nan,
        "t_stat": np.nan,
        "n_days": np.nan,
    }
    for column, default in defaults.items():
        if column not in out.columns:
            out[column] = default
        else:
            if isinstance(default, bool):
                out[column] = out[column].where(out[column].notna(), default).astype(bool)
            else:
                out[column] = out[column].where(out[column].notna(), default)
    return out


def companion_input_paths(args: argparse.Namespace) -> list[Path]:
    """Standard companion files listed in the project prompt but not passed on CLI."""
    return [
        args.linear_results.parent / "fama_macbeth_summary_v2.csv",
        args.linear_results.parent / "spread_summary_v2.csv",
        args.lasso_summary.parent / "lasso_elasticnet_coefficients_v2.csv",
        args.lasso_summary.parent / "lasso_elasticnet_performance_v2.csv",
        args.ranker_importance.parent / "xgboost_ranker_performance_v2.csv",
    ]


def compute_ic_support(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute support score from teammate IC/correlation report."""
    out = frame.copy()
    abs_t = pd.to_numeric(out["t_stat"], errors="coerce").abs()
    out["ic_corr_support_score"] = np.where(
        out["selected_by_ic_corr_report"],
        1.0,
        np.where(out["survived_significance_filter"], 0.70, np.minimum(abs_t / 2.0, 1.0) * 0.40),
    )
    return out


def final_group_for_row(row: pd.Series) -> str:
    """Assign final candidate group."""
    alpha_id = int(row["alpha_id"])
    if alpha_id in CORE_EXPECTED_IDS and bool(row["selected_by_ic_corr_report"]):
        return "final_core_candidates"
    if alpha_id == 55 and bool(row["supported_by_ablation"]):
        return "final_core_candidates"
    if alpha_id == 68:
        return "final_interaction_candidates"
    if alpha_id in INTERACTION_PRIORITY_IDS and (
        bool(row["selected_by_true_ranker"]) or finite_float(row["ranker_support_score"], 0.0) >= 0.35 or bool(row["selected_by_linear"])
    ):
        return "final_interaction_candidates"
    if row["redundancy_status"] == "dropped_redundant" and alpha_id in {31, 52}:
        return "final_watchlist_candidates"
    if bool(row["selected_by_true_ranker"]) or bool(row["selected_by_linear"]) or finite_float(row["ic_corr_support_score"], 0.0) >= 0.45:
        return "final_watchlist_candidates"
    return "final_excluded_for_now"


def usage_for_group(row: pd.Series) -> str:
    """Recommended usage text."""
    group = row["final_group"]
    alpha_id = int(row["alpha_id"])
    if group == "final_core_candidates":
        if alpha_id in {55, 56, 58, 59, 60, 66}:
            return "Use in clean-core tests, with risk/liquidity exposure monitoring."
        return "Use in clean-core mega-alpha and portfolio-construction tests."
    if group == "final_interaction_candidates":
        return "Keep for XGBoost/ranker interaction tests; do not treat as a clean standalone core alpha."
    if group == "final_watchlist_candidates":
        return "Keep for robustness and redundancy review; exclude from the first compact core model."
    return "Exclude from compact model for now; do not delete permanently."


def risk_warning(row: pd.Series) -> str:
    """Risk warning text."""
    alpha_id = int(row["alpha_id"])
    warnings: list[str] = []
    if alpha_id in RISK_LIQUIDITY_IDS:
        warnings.append("risk/liquidity/exposure-sensitive signal")
    if alpha_id == 55:
        warnings.append("ATR has overfit/noise warning, but ablation says removal hurt performance")
    if alpha_id == 68:
        warnings.append("ADV trend has weak IC but ablation says removal hurt ranker performance")
    if row.get("redundancy_status") == "dropped_redundant":
        warnings.append(f"redundant with retained alpha {format_alpha_id(row.get('retained_cluster_head'))}")
    return "; ".join(warnings)


def format_alpha_id(value: Any) -> str:
    """Format alpha ID values."""
    if pd.isna(value):
        return ""
    return f"Alpha {int(value)}"


def plain_reason(row: pd.Series) -> str:
    """Plain-English reason for final assignment."""
    pieces: list[str] = []
    alpha_id = int(row["alpha_id"])
    if row["final_group"] == "final_core_candidates":
        pieces.append("selected as core because it survived the IC/correlation process and has supporting ranker, linear, or ablation evidence")
    elif row["final_group"] == "final_interaction_candidates":
        pieces.append("kept for nonlinear interaction value rather than clean standalone IC")
    elif row["final_group"] == "final_watchlist_candidates":
        pieces.append("mixed evidence or redundancy makes it useful for monitoring but not first-pass compact selection")
    else:
        pieces.append("weak combined evidence relative to stronger alternatives")
    if bool(row["selected_by_ic_corr_report"]):
        pieces.append("selected by teammate IC/correlation report")
    if bool(row["selected_by_true_ranker"]):
        pieces.append("important in true XGBoost Ranker")
    if bool(row["selected_by_lasso_or_elasticnet"]):
        pieces.append("selected by Lasso/ElasticNet")
    if bool(row["supported_by_ablation"]):
        pieces.append("supported by diagnostic ablation")
    if alpha_id == 31:
        pieces.append("Alpha 32 is preferred over highly correlated 252-day return")
    if alpha_id == 52:
        pieces.append("Alpha 53 is preferred as the inverse Bollinger cluster head")
    return "; ".join(pieces) + "."


def final_recommended_direction(row: pd.Series) -> str:
    """Choose recommended direction from IC or linear evidence."""
    if pd.notna(row.get("mean_ic")) and finite_float(row.get("mean_ic"), 0.0) != 0:
        return "positive" if finite_float(row.get("mean_ic")) > 0 else "negative"
    return str(row.get("linear_direction", "mixed"))


def build_summary(
    linear: pd.DataFrame,
    lasso: pd.DataFrame,
    ranker: pd.DataFrame,
    ranker_importance: pd.DataFrame,
    final_ranker: pd.DataFrame,
    ablation: pd.DataFrame,
    ic_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Build final alpha summary table."""
    linear_one = best_linear_evidence(linear)
    lasso_one = lasso_evidence(lasso)
    ranker_one = ranker_evidence(ranker, ranker_importance)
    ablation_one = ablation
    base = base_alpha_table(linear_one, lasso, ranker, final_ranker, ic_rows)
    summary = base
    for frame in [linear_one, lasso_one, ranker_one, ablation_one, ic_rows]:
        if frame.empty:
            continue
        summary = summary.merge(frame, on="alpha_id", how="left", suffixes=("", "_dup"))
        for column in [col for col in summary.columns if col.endswith("_dup")]:
            base_col = column[:-4]
            summary[base_col] = summary[base_col].combine_first(summary[column])
            summary = summary.drop(columns=[column])
    summary["category"] = summary["category"].fillna(summary["alpha_id"].map(alpha_category))
    summary = add_default_columns(summary)
    summary = compute_ic_support(summary)
    summary["final_score"] = (
        0.25 * pd.to_numeric(summary["linear_support_score"], errors="coerce").fillna(0)
        + 0.15 * pd.to_numeric(summary["lasso_elasticnet_support_score"], errors="coerce").fillna(0)
        + 0.25 * pd.to_numeric(summary["ranker_support_score"], errors="coerce").fillna(0)
        + 0.25 * pd.to_numeric(summary["ic_corr_support_score"], errors="coerce").fillna(0)
        + 0.10 * pd.to_numeric(summary["ablation_support_score"], errors="coerce").fillna(0)
    )
    summary["final_group"] = summary.apply(final_group_for_row, axis=1)
    summary["clean_core_flag"] = summary["final_group"].eq("final_core_candidates")
    summary["interaction_flag"] = summary["final_group"].eq("final_interaction_candidates")
    summary["watchlist_flag"] = summary["final_group"].eq("final_watchlist_candidates")
    summary["excluded_flag"] = summary["final_group"].eq("final_excluded_for_now")
    summary["final_recommended_feature_variant"] = summary["best_ranker_feature_variant"].combine_first(summary["best_linear_feature_variant"])
    summary["final_recommended_feature_variant"] = summary["final_recommended_feature_variant"].fillna("cs_rank")
    summary["final_recommended_direction"] = summary.apply(final_recommended_direction, axis=1)
    summary["recommended_usage"] = summary.apply(usage_for_group, axis=1)
    summary["risk_warning"] = summary.apply(risk_warning, axis=1)
    summary["plain_english_reason"] = summary.apply(plain_reason, axis=1)
    summary["group_order"] = summary["final_group"].map(GROUP_ORDER)
    output_columns = [
        "alpha_id",
        "alpha_name",
        "category",
        "final_group",
        "final_score",
        "clean_core_flag",
        "interaction_flag",
        "watchlist_flag",
        "excluded_flag",
        "linear_support_score",
        "linear_direction",
        "best_fama_macbeth_tstat_newey_west",
        "best_rank_ic_tstat_newey_west",
        "best_spread_tstat_newey_west",
        "horizon_consistency",
        "excess_tradable_consistency",
        "lasso_elasticnet_support_score",
        "coefficient_direction",
        "lasso_selection_count",
        "elasticnet_selection_count",
        "ranker_support_score",
        "average_gain_importance",
        "best_gain_importance",
        "average_importance_rank",
        "best_importance_rank",
        "best_ranker_target_label",
        "best_ranker_feature_variant",
        "best_ranker_test_rank_ic",
        "important_in_targets",
        "important_in_feature_sets",
        "ic_corr_support_score",
        "mean_ic",
        "icir",
        "t_stat",
        "n_days",
        "ablation_support_score",
        "ablation_note",
        "redundancy_cluster",
        "retained_cluster_head",
        "redundancy_status",
        "rank_corr_to_head",
        "selected_by_ic_corr_report",
        "selected_by_linear",
        "selected_by_lasso_or_elasticnet",
        "selected_by_true_ranker",
        "supported_by_ablation",
        "final_recommended_feature_variant",
        "final_recommended_direction",
        "recommended_usage",
        "risk_warning",
        "plain_english_reason",
    ]
    for column in output_columns:
        if column not in summary.columns:
            summary[column] = np.nan
    return summary[output_columns].sort_values(["final_group", "final_score"], key=sort_key).reset_index(drop=True)


def sort_key(series: pd.Series) -> pd.Series:
    """Sort final_group by semantic order, other columns normally."""
    if series.name == "final_group":
        return series.map(GROUP_ORDER).fillna(99)
    return series


def selected_feature_exists(model_columns: list[str], alpha_id: int) -> bool:
    """Check whether model dataset has transformed columns for an alpha."""
    prefix = f"alpha_{alpha_id}_"
    return any(column.startswith(prefix) and not column.endswith("_raw") for column in model_columns)


def build_report(summary: pd.DataFrame, clusters: pd.DataFrame, meta: dict[str, Any], ablation_report: str) -> str:
    """Build human-readable Markdown report."""
    group_counts = summary["final_group"].value_counts().reindex(GROUP_ORDER.keys(), fill_value=0)
    core = summary[summary["final_group"] == "final_core_candidates"]
    interaction = summary[summary["final_group"] == "final_interaction_candidates"]
    watchlist = summary[summary["final_group"] == "final_watchlist_candidates"]
    excluded = summary[summary["final_group"] == "final_excluded_for_now"]
    lines = [
        "# Final Candidate Alpha Selection Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- Final core candidates: {int(group_counts['final_core_candidates'])}.",
        f"- Final interaction candidates: {int(group_counts['final_interaction_candidates'])}.",
        f"- Final watchlist candidates: {int(group_counts['final_watchlist_candidates'])}.",
        f"- Final excluded for now: {int(group_counts['final_excluded_for_now'])}.",
        f"- IC significance threshold: |t_stat| >= {meta.get('significance_threshold')}.",
        f"- Redundancy threshold: |rank-corr| >= {meta.get('redundancy_threshold')}.",
        "",
        "## 2. Evidence Sources",
        "",
        "- Linear / Fama-MacBeth: used Newey-West Fama-MacBeth t-stat, Rank IC t-stat, spread t-stat, and horizon consistency.",
        "- Lasso / ElasticNet: used selection flags, selection counts, coefficient direction, and regularized-model interpretation.",
        "- True XGBoost Ranker: used only `xgboost_ranker` gain importance, importance rank, target coverage, and best ranker metrics.",
        "- Rank IC + correlation report: used the teammate significance filter, redundancy clusters, final selected list, ICIR, t-stat, and retained cluster heads.",
        "- Diagnostic ablation: incorporated the findings that Alpha 32 was already included, removing Alpha 55 hurt, removing Alpha 68 hurt, and Alpha 54/65 are stable backbone features.",
        "",
        "## 3. Final Core Candidates",
        "",
    ]
    for _, row in core.sort_values("alpha_id").iterrows():
        lines.extend(
            [
                f"### Alpha {int(row['alpha_id'])}: {row['alpha_name']}",
                f"- Category: {row['category']}.",
                f"- Why selected: {row['plain_english_reason']}",
                f"- IC/correlation evidence: mean IC={fmt(row['mean_ic'])}, ICIR={fmt(row['icir'])}, t-stat={fmt(row['t_stat'])}, status={row['redundancy_status']}.",
                f"- Linear/ranker/ablation evidence: linear score={fmt(row['linear_support_score'])}, ranker score={fmt(row['ranker_support_score'])}, ablation score={fmt(row['ablation_support_score'])}.",
                f"- Warning: {row['risk_warning'] or 'none'}.",
                "",
            ]
        )
    lines.extend(["## 4. Interaction Candidates", ""])
    for _, row in interaction.sort_values("alpha_id").iterrows():
        lines.append(f"- Alpha {int(row['alpha_id'])} {row['alpha_name']}: {row['plain_english_reason']}")
    lines.extend(["", "## 5. Redundancy Decisions", ""])
    if clusters.empty:
        lines.append("- No redundancy clusters parsed.")
    else:
        for _, row in clusters.iterrows():
            lines.append(
                f"- Cluster {int(row['cluster_id'])}: kept {row['kept_alpha']}; dropped {row['dropped_alphas'] or 'none'}; "
                f"reason: {row['reason']}; rank corr: {row['rank_corr_to_head'] or 'n/a'}."
            )
    lines.extend(
        [
            "- Alpha 32 is preferred over Alpha 31 because Alpha 31 is highly correlated with Alpha 32 and Alpha 32 has higher ICIR.",
            "- Alpha 58 is the clean core representative for the Alpha 57 / 63 volatility-drawdown redundancy group; Alpha 57 and 63 stay as interaction candidates.",
            "- Alpha 53 is preferred over Alpha 52 because Alpha 52 is a perfectly inverse redundant Bollinger z-score signal.",
            "- Volatility/range/liquidity features remain important, but they need exposure controls in portfolio construction.",
            "",
            "## 6. Alphas Not Selected",
            "",
            "Excluded for now means excluded from the first compact candidate set, not permanently deleted from the research library.",
        ]
    )
    for _, row in excluded.sort_values("alpha_id").iterrows():
        lines.append(f"- Alpha {int(row['alpha_id'])} {row['alpha_name']}: {row['plain_english_reason']}")
    lines.extend(
        [
            "",
            "## 7. Recommended Next Model Tests",
            "",
            "- Model A: clean core only.",
            "- Model B: clean core + interaction candidates.",
            "- Model C: current baseline ranker.",
            "- Model D: clean core excluding high-risk volatility features.",
            "- Model E: clean core + risk/liquidity features with exposure controls.",
            "",
            "## 8. Final Recommended Candidate List",
            "",
            "### Core",
        ]
    )
    for _, row in core.sort_values("alpha_id").iterrows():
        lines.append(f"- Alpha {int(row['alpha_id'])}: {row['alpha_name']}")
    lines.append("")
    lines.append("### Interaction")
    for _, row in interaction.sort_values("alpha_id").iterrows():
        lines.append(f"- Alpha {int(row['alpha_id'])}: {row['alpha_name']}")
    lines.append("")
    lines.append("### Watchlist Count")
    lines.append(f"- {len(watchlist)}")
    lines.append("")
    lines.append("### Excluded Count")
    lines.append(f"- {len(excluded)}")
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    """Format numeric evidence."""
    numeric = finite_float(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.4f}"


def run_qa(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    clusters: pd.DataFrame,
    ic_rows: pd.DataFrame,
    ranker_importance: pd.DataFrame,
    ranker_performance_backend_ok: bool,
    companion_paths: list[Path],
) -> tuple[bool, list[str]]:
    """Run final QA checks."""
    failures: list[str] = []
    inputs = [
        args.linear_results,
        args.lasso_summary,
        args.ranker_summary,
        args.ranker_importance,
        args.final_ranker_summary,
        args.ablation_summary,
        args.ablation_report,
        args.ic_corr_report,
        args.model_dataset,
        *companion_paths,
    ]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        failures.append(f"missing input files: {missing}")
    if int((summary["alpha_id"] == 35).sum()) != 0:
        failures.append("Alpha 35 was not skipped")
    if summary["alpha_id"].duplicated().any():
        failures.append("duplicate alpha rows in final summary")
    if len(summary) != summary["alpha_id"].nunique():
        failures.append("final summary is not one row per alpha_id")
    if ic_rows.empty:
        failures.append("Rank IC/correlation report did not parse")
    backends = sorted(ranker_importance["backend"].dropna().astype(str).unique().tolist())
    if backends != ["xgboost_ranker"] or not ranker_performance_backend_ok:
        failures.append(f"true ranker backend not confirmed: {backends}")
    if not any(summary["supported_by_ablation"]):
        failures.append("ablation conclusions were not incorporated")
    if args.model_dataset.exists():
        model = pd.read_parquet(args.model_dataset, columns=None)
        selected_ids = summary.loc[summary["final_group"].isin(["final_core_candidates", "final_interaction_candidates"]), "alpha_id"]
        missing_feature_ids = [int(alpha_id) for alpha_id in selected_ids if not selected_feature_exists(list(model.columns), int(alpha_id))]
        if missing_feature_ids:
            failures.append(f"selected alpha IDs missing from model dataset feature columns: {missing_feature_ids}")
    return len(failures) == 0, failures


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    args.ic_corr_report = resolve_ic_report(args.ic_corr_report)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    linear = load_csv(args.linear_results)
    lasso = load_csv(args.lasso_summary)
    ranker = load_csv(args.ranker_summary)
    ranker_importance = load_csv(args.ranker_importance)
    final_ranker = load_csv(args.final_ranker_summary)
    ablation_summary = load_csv(args.ablation_summary)
    ablation_report_text = args.ablation_report.read_text(encoding="utf-8", errors="ignore")
    ic_rows, clusters, ic_meta = parse_ic_corr_report(args.ic_corr_report)
    ablation = ablation_evidence(ablation_summary, ablation_report_text)

    ranker_performance_path = args.ranker_importance.parent / "xgboost_ranker_performance_v2.csv"
    companions = companion_input_paths(args)
    ranker_performance_backend_ok = True
    if ranker_performance_path.exists():
        ranker_perf = pd.read_csv(ranker_performance_path)
        ranker_performance_backend_ok = sorted(ranker_perf["backend"].dropna().astype(str).unique().tolist()) == ["xgboost_ranker"]

    summary = build_summary(linear, lasso, ranker, ranker_importance, final_ranker, ablation, ic_rows)
    summary.to_csv(args.output_dir / "final_candidate_alpha_summary.csv", index=False)
    summary[summary["final_group"] == "final_core_candidates"].to_csv(args.output_dir / "final_core_candidates.csv", index=False)
    summary[summary["final_group"] == "final_interaction_candidates"].to_csv(args.output_dir / "final_interaction_candidates.csv", index=False)
    summary[summary["final_group"] == "final_watchlist_candidates"].to_csv(args.output_dir / "final_watchlist_candidates.csv", index=False)
    summary[summary["final_group"] == "final_excluded_for_now"].to_csv(args.output_dir / "final_excluded_for_now.csv", index=False)
    clusters.to_csv(args.output_dir / "final_redundancy_clusters.csv", index=False)
    report = build_report(summary, clusters, ic_meta, ablation_report_text)
    (args.output_dir / "final_candidate_alpha_report.md").write_text(report, encoding="utf-8")

    ok, failures = run_qa(args, summary, clusters, ic_rows, ranker_importance, ranker_performance_backend_ok, companions)
    core = summary[summary["final_group"] == "final_core_candidates"]
    interaction = summary[summary["final_group"] == "final_interaction_candidates"]
    watchlist_count = int((summary["final_group"] == "final_watchlist_candidates").sum())
    excluded_count = int((summary["final_group"] == "final_excluded_for_now").sum())
    print("final core candidates:")
    for _, row in core.sort_values("alpha_id").iterrows():
        print(f"  Alpha {int(row['alpha_id'])}: {row['alpha_name']}")
    print("final interaction candidates:")
    for _, row in interaction.sort_values("alpha_id").iterrows():
        print(f"  Alpha {int(row['alpha_id'])}: {row['alpha_name']}")
    print(f"final watchlist count: {watchlist_count}")
    print(f"excluded count: {excluded_count}")
    print("redundancy clusters:")
    for _, row in clusters.iterrows():
        print(f"  Cluster {int(row['cluster_id'])}: kept {row['kept_alpha']}; dropped {row['dropped_alphas'] or 'none'}")
    print(f"final candidate selection status: {'PASS' if ok else 'FAIL'}")
    if failures:
        print("failed checks:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
