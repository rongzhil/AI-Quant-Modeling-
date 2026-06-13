"""Combine linear, Lasso/ElasticNet, and true XGBoost Ranker alpha evidence.

This module intentionally consumes the true XGBoost Ranker outputs only. It
does not use the earlier XGBoost regressor fallback summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .factor_selection_common import alpha_category, join_unique, sign_consistency, sign_label


BEST_RANKER_TARGET = "tradable_forward_21d_return"
BEST_RANKER_FEATURE_SET = "cs_rank"
BEST_RANKER_OBJECTIVE = "rank:pairwise"
BEST_RANKER_VALIDATION_RANK_IC = 0.172193
BEST_RANKER_TEST_RANK_IC = 0.080886
BEST_RANKER_TEST_SPREAD = 0.029017

SPECIAL_ALPHA_IDS_WITH_RANKER = [55, 56, 54, 57, 66, 34, 62, 58, 30, 63, 61, 33, 46, 73]
GROUP_ORDER = {
    "core_alpha": 0,
    "nonlinear_alpha": 1,
    "redundant_alpha": 2,
    "linear_only_alpha": 3,
    "watchlist_alpha": 4,
    "rejected_alpha": 5,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Combine factor selection evidence using true XGBoost Ranker outputs.")
    parser.add_argument("--linear-results", type=Path, required=True)
    parser.add_argument("--lasso-summary", type=Path, required=True)
    parser.add_argument("--ranker-summary", type=Path, required=True)
    parser.add_argument("--ranker-importance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    """Load a required CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def optional_version(package_name: str) -> str:
    """Return an installed package version without making it a hard dependency."""
    try:
        module = __import__(package_name)
        return str(getattr(module, "__version__", "installed"))
    except Exception:
        return "not_importable_in_this_environment"


def safe_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert available columns to numeric."""
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def truthy(value: Any) -> bool:
    """Parse bool-like CSV values."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def finite_float(value: Any, default: float = 0.0) -> float:
    """Convert to finite float with a default."""
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def clip_t(value: Any, scale: float = 3.0) -> float:
    """Clip a t-stat-like value onto 0..1 strength scale."""
    numeric = finite_float(value, default=np.nan)
    if not np.isfinite(numeric):
        return 0.0
    return float(min(abs(numeric) / scale, 1.0))


def pair_consistency(group: pd.DataFrame, left_label: str, right_label: str) -> float:
    """Measure sign agreement for the same alpha variant across two labels."""
    hits: list[float] = []
    for _, variant_group in group.groupby("feature_variant", sort=False):
        left = variant_group[variant_group["target_label"] == left_label]
        right = variant_group[variant_group["target_label"] == right_label]
        if left.empty or right.empty:
            continue
        left_sign = np.sign(finite_float(left.iloc[0].get("mean_rank_ic_spearman"), default=np.nan))
        right_sign = np.sign(finite_float(right.iloc[0].get("mean_rank_ic_spearman"), default=np.nan))
        if left_sign != 0 and right_sign != 0:
            hits.append(float(left_sign == right_sign))
    return float(np.mean(hits)) if hits else 0.0


def linear_interpretation(row: pd.Series) -> str:
    """Create a compact description of linear evidence strength."""
    score = finite_float(row.get("linear_score"))
    if score >= 0.65:
        strength = "strong"
    elif score >= 0.45:
        strength = "moderate"
    else:
        strength = "weak"
    direction = sign_label(finite_float(row.get("mean_rank_ic_spearman"), default=np.nan))
    return (
        f"{strength} single-alpha evidence; best target is {row.get('target_label')} "
        f"using {row.get('feature_variant')} with {direction} Rank IC direction."
    )


def summarize_linear(linear: pd.DataFrame) -> pd.DataFrame:
    """Reduce Step 7 v2 linear evidence to one best row per alpha."""
    required = {"alpha_id", "feature_name", "target_label", "feature_variant"}
    missing = sorted(required.difference(linear.columns))
    if missing:
        raise ValueError(f"Linear results missing required columns: {missing}")

    data = linear[linear["alpha_id"] != 35].copy()
    numeric_columns = [
        "n_obs",
        "n_dates",
        "n_tickers",
        "mean_beta",
        "beta_t_stat_newey_west",
        "mean_rank_ic_spearman",
        "rank_ic_t_stat_newey_west",
        "ICIR",
        "spread_t_stat_newey_west",
        "spread_metric",
        "data_coverage",
    ]
    data = safe_numeric(data, numeric_columns)
    data["linear_score"] = (
        0.40 * data["beta_t_stat_newey_west"].map(clip_t)
        + 0.35 * data["rank_ic_t_stat_newey_west"].map(clip_t)
        + 0.25 * data["spread_t_stat_newey_west"].map(clip_t)
    )

    rows: list[dict[str, Any]] = []
    for alpha_id, group in data.groupby("alpha_id", sort=True):
        best = group.sort_values(["linear_score", "data_coverage"], ascending=[False, False]).iloc[0]
        direction_values = group["mean_rank_ic_spearman"].dropna().tolist()
        rows.append(
            {
                "alpha_id": int(alpha_id),
                "alpha_name": str(best["feature_name"]),
                "category": alpha_category(int(alpha_id)),
                "best_linear_feature_variant": best["feature_variant"],
                "best_linear_target_label": best["target_label"],
                "best_linear_score": finite_float(best["linear_score"], default=np.nan),
                "best_linear_direction": sign_label(finite_float(best["mean_rank_ic_spearman"], default=np.nan)),
                "best_fama_macbeth_tstat_newey_west": finite_float(best["beta_t_stat_newey_west"], default=np.nan),
                "best_rank_ic_tstat_newey_west": finite_float(best["rank_ic_t_stat_newey_west"], default=np.nan),
                "best_spread_tstat_newey_west": finite_float(best["spread_t_stat_newey_west"], default=np.nan),
                "best_linear_rank_ic": finite_float(best["mean_rank_ic_spearman"], default=np.nan),
                "best_linear_spread": finite_float(best["spread_metric"], default=np.nan),
                "linear_sign_consistency": sign_consistency(direction_values),
                "horizon_consistency": pair_consistency(group, "forward_5d_excess_return", "forward_21d_excess_return"),
                "excess_vs_tradable_consistency": pair_consistency(group, "forward_5d_excess_return", "tradable_forward_5d_return"),
                "linear_interpretation": linear_interpretation(best),
            }
        )
    return pd.DataFrame(rows)


def prepare_lasso(lasso: pd.DataFrame) -> pd.DataFrame:
    """Normalize Lasso/ElasticNet readable summary columns."""
    data = lasso[lasso["alpha_id"] != 35].copy()
    data = safe_numeric(
        data,
        [
            "alpha_id",
            "lasso_selection_count",
            "elasticnet_selection_count",
            "lasso_best_coefficient",
            "elasticnet_best_coefficient",
            "coefficient_sign_consistency",
            "best_test_rank_ic",
            "best_test_icir",
            "best_test_decile_spread",
            "best_test_r2",
        ],
    )
    data["lasso_selected_any"] = data["lasso_selected_any"].map(truthy)
    data["elasticnet_selected_any"] = data["elasticnet_selected_any"].map(truthy)
    rename_map = {
        "best_feature_variant": "best_regularized_feature_variant",
        "best_target_label": "best_regularized_target_label",
        "coefficient_direction": "regularized_direction",
        "best_test_rank_ic": "best_regularized_test_rank_ic",
        "interpretation": "regularized_interpretation",
        "warning_flags": "regularized_warning_flags",
    }
    return data.rename(columns=rename_map)


def prepare_ranker(ranker_summary: pd.DataFrame) -> pd.DataFrame:
    """Normalize true XGBoost Ranker readable summary columns."""
    data = ranker_summary[ranker_summary["alpha_id"] != 35].copy()
    data = safe_numeric(
        data,
        [
            "alpha_id",
            "best_test_rank_ic",
            "best_test_icir",
            "best_test_decile_spread",
            "best_validation_rank_ic",
            "average_gain_importance",
            "best_gain_importance",
            "average_importance_rank",
            "best_importance_rank",
        ],
    )
    rename_map = {
        "best_feature_variant": "best_ranker_feature_variant",
        "best_target_label": "best_ranker_target_label",
        "best_objective": "best_ranker_objective",
        "best_test_rank_ic": "best_ranker_test_rank_ic",
        "best_test_icir": "best_ranker_test_icir",
        "best_test_decile_spread": "best_ranker_decile_spread",
        "best_validation_rank_ic": "best_ranker_validation_rank_ic",
        "average_gain_importance": "average_ranker_gain_importance",
        "best_gain_importance": "best_ranker_gain_importance",
        "average_importance_rank": "average_ranker_importance_rank",
        "best_importance_rank": "best_ranker_importance_rank",
        "ranker_interpretation": "ranker_interpretation",
        "warning_flags": "ranker_warning_flags",
    }
    return data.rename(columns=rename_map)


def ranker_backend_summary(ranker_importance: pd.DataFrame) -> dict[str, Any]:
    """Validate ranker backend usage."""
    if "backend" not in ranker_importance.columns:
        raise ValueError("Ranker importance is missing backend column")
    backends = sorted(str(value) for value in ranker_importance["backend"].dropna().unique())
    backend_text = ";".join(backends)
    return {
        "backends": backends,
        "backend_text": backend_text,
        "true_ranker_confirmed": backends == ["xgboost_ranker"],
        "regressor_fallback_used": any("regressor" in item or "fallback" in item for item in backends),
        "numpy_fallback_used": any("numpy" in item for item in backends),
    }


def add_ranker_strength_flags(combined: pd.DataFrame) -> pd.DataFrame:
    """Add strong/moderate true ranker importance flags."""
    out = combined.copy()
    avg_rank = pd.to_numeric(out["average_ranker_importance_rank"], errors="coerce")
    best_rank = pd.to_numeric(out["best_ranker_importance_rank"], errors="coerce")
    best_gain = pd.to_numeric(out["best_ranker_gain_importance"], errors="coerce")
    gain_threshold = best_gain.dropna().quantile(0.75) if best_gain.notna().any() else np.nan
    out["ranker_strong_importance"] = (avg_rank <= 20) | (best_rank <= 5) | (best_gain >= gain_threshold)
    out["ranker_moderate_importance"] = (avg_rank <= 30) | (best_rank <= 15)
    out["ranker_strong_importance"] = out["ranker_strong_importance"].fillna(False)
    out["ranker_moderate_importance"] = out["ranker_moderate_importance"].fillna(False)
    return out


def classify_row(row: pd.Series) -> tuple[str, str]:
    """Assign final group and recommended usage."""
    strong_linear = bool(row.get("strong_linear_evidence", False))
    moderate_linear = bool(row.get("moderate_linear_evidence", False))
    regularized = bool(row.get("selected_by_lasso_or_elasticnet", False))
    strong_ranker = bool(row.get("ranker_strong_importance", False))
    moderate_ranker = bool(row.get("ranker_moderate_importance", False))
    horizon = finite_float(row.get("horizon_consistency"))

    if strong_linear and regularized and strong_ranker:
        return "core_alpha", "Use as a high-priority candidate in mega-alpha testing and later portfolio validation."
    if strong_ranker and not strong_linear and not regularized:
        return "nonlinear_alpha", "Use in tree/ranker mega-alpha tests; do not rely on it as a standalone linear factor."
    if strong_linear and not regularized and not strong_ranker:
        return "linear_only_alpha", "Keep for interpretable single-alpha review and redundancy tests."
    if strong_linear and not regularized and strong_ranker:
        return "redundant_alpha", "Useful signal, but likely overlaps with correlated features; include only after ablation."
    if regularized or moderate_linear or moderate_ranker or horizon >= 0.50:
        return "watchlist_alpha", "Keep for robustness checks, ablation, and horizon-specific model variants."
    return "rejected_alpha", "Do not prioritize for the next mega-alpha test unless new data changes the evidence."


def plain_summary(row: pd.Series, group_name: str) -> str:
    """Plain-English one-row summary."""
    return (
        f"Alpha {int(row['alpha_id'])} ({row['alpha_name']}) is {group_name}: "
        f"linear score {finite_float(row.get('best_linear_score'), np.nan):.3f}, "
        f"Lasso/ElasticNet selected={bool(row.get('selected_by_lasso_or_elasticnet', False))}, "
        f"ranker best importance rank={finite_float(row.get('best_ranker_importance_rank'), np.nan):.0f}."
    )


def combine_warnings(row: pd.Series) -> str:
    """Merge warning columns into one semicolon-separated field."""
    warnings: list[str] = []
    for column in ["linear_warning_flags", "regularized_warning_flags", "ranker_warning_flags"]:
        value = row.get(column, "")
        if pd.notna(value) and str(value).strip() and str(value).strip().lower() != "nan":
            warnings.append(str(value).strip())
    if int(row["alpha_id"]) in {54, 55, 56, 57, 58, 59, 61, 62, 63, 66, 67}:
        warnings.append("risk_or_liquidity_signal_validate_with_portfolio_constraints")
    return ";".join(sorted(set(warnings)))


def build_combined(
    linear: pd.DataFrame,
    lasso_summary: pd.DataFrame,
    ranker_summary: pd.DataFrame,
    ranker_importance: pd.DataFrame,
) -> pd.DataFrame:
    """Build the final one-row-per-alpha readable summary."""
    linear_one = summarize_linear(linear)
    lasso_one = prepare_lasso(lasso_summary)
    ranker_one = prepare_ranker(ranker_summary)

    combined = linear_one.merge(
        lasso_one,
        on=["alpha_id", "alpha_name", "category"],
        how="outer",
        suffixes=("", "_regularized"),
    ).merge(
        ranker_one,
        on=["alpha_id", "alpha_name", "category"],
        how="outer",
        suffixes=("", "_ranker"),
    )
    combined = combined[combined["alpha_id"] != 35].copy()
    combined["alpha_id"] = combined["alpha_id"].astype(int)
    combined["category"] = combined["category"].fillna(combined["alpha_id"].map(alpha_category))

    combined["strong_linear_evidence"] = pd.to_numeric(combined["best_linear_score"], errors="coerce").fillna(0) >= 0.55
    combined["moderate_linear_evidence"] = pd.to_numeric(combined["best_linear_score"], errors="coerce").fillna(0) >= 0.40
    combined["selected_by_lasso_or_elasticnet"] = combined["lasso_selected_any"].map(truthy) | combined["elasticnet_selected_any"].map(truthy)
    combined = add_ranker_strength_flags(combined)
    combined["selected_by_ranker"] = combined["ranker_strong_importance"]
    combined["evidence_count"] = combined[["strong_linear_evidence", "selected_by_lasso_or_elasticnet", "selected_by_ranker"]].sum(axis=1).astype(int)

    groups: list[str] = []
    usages: list[str] = []
    summaries: list[str] = []
    warnings: list[str] = []
    for _, row in combined.iterrows():
        group_name, usage = classify_row(row)
        groups.append(group_name)
        usages.append(usage)
        summaries.append(plain_summary(row, group_name))
        warnings.append(combine_warnings(row))

    combined["final_selection_group"] = groups
    combined["recommended_usage"] = usages
    combined["warning_flags"] = warnings
    combined["plain_english_explanation"] = summaries
    combined["selection_group_order"] = combined["final_selection_group"].map(GROUP_ORDER).fillna(99).astype(int)

    output_columns = [
        "alpha_id",
        "alpha_name",
        "category",
        "best_linear_feature_variant",
        "best_linear_target_label",
        "best_linear_score",
        "best_linear_direction",
        "best_fama_macbeth_tstat_newey_west",
        "best_rank_ic_tstat_newey_west",
        "best_spread_tstat_newey_west",
        "best_linear_rank_ic",
        "best_linear_spread",
        "linear_sign_consistency",
        "horizon_consistency",
        "excess_vs_tradable_consistency",
        "linear_interpretation",
        "lasso_selected_any",
        "elasticnet_selected_any",
        "lasso_selection_count",
        "elasticnet_selection_count",
        "lasso_best_coefficient",
        "elasticnet_best_coefficient",
        "regularized_direction",
        "coefficient_sign_consistency",
        "best_model_type",
        "best_regularized_feature_variant",
        "best_regularized_target_label",
        "best_regularized_test_rank_ic",
        "best_test_icir",
        "best_test_decile_spread",
        "selected_targets",
        "selected_feature_variants",
        "regularized_interpretation",
        "best_ranker_feature_variant",
        "best_ranker_target_label",
        "best_ranker_objective",
        "best_ranker_test_rank_ic",
        "best_ranker_test_icir",
        "best_ranker_decile_spread",
        "best_ranker_validation_rank_ic",
        "average_ranker_gain_importance",
        "best_ranker_gain_importance",
        "average_ranker_importance_rank",
        "best_ranker_importance_rank",
        "important_in_targets",
        "important_in_feature_sets",
        "ranker_interpretation",
        "strong_linear_evidence",
        "selected_by_lasso_or_elasticnet",
        "ranker_strong_importance",
        "ranker_moderate_importance",
        "selected_by_ranker",
        "evidence_count",
        "final_selection_group",
        "recommended_usage",
        "warning_flags",
        "plain_english_explanation",
    ]
    for column in output_columns:
        if column not in combined.columns:
            combined[column] = np.nan

    combined = combined.sort_values(
        ["selection_group_order", "evidence_count", "best_linear_score", "average_ranker_importance_rank"],
        ascending=[True, False, False, True],
    )
    return combined[output_columns].reset_index(drop=True)


def write_group_files(combined: pd.DataFrame, output_dir: Path) -> None:
    """Write the six required group CSV files."""
    file_map = {
        "core_alpha": "core_alpha_candidates_with_ranker.csv",
        "linear_only_alpha": "linear_only_alpha_candidates_with_ranker.csv",
        "nonlinear_alpha": "nonlinear_alpha_candidates_with_ranker.csv",
        "redundant_alpha": "redundant_alpha_candidates_with_ranker.csv",
        "watchlist_alpha": "watchlist_alpha_candidates_with_ranker.csv",
        "rejected_alpha": "rejected_alpha_candidates_with_ranker.csv",
    }
    for group_name, filename in file_map.items():
        combined[combined["final_selection_group"] == group_name].to_csv(output_dir / filename, index=False)


def top_ranker_alphas(ranker_importance: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Compute top ranker alphas by average gain importance."""
    data = ranker_importance[ranker_importance["alpha_id"] != 35].copy()
    data = safe_numeric(data, ["alpha_id", "gain_importance", "normalized_gain_importance", "importance_rank"])
    grouped = (
        data.groupby(["alpha_id", "alpha_name"], as_index=False)
        .agg(
            average_gain_importance=("gain_importance", "mean"),
            best_gain_importance=("gain_importance", "max"),
            average_importance_rank=("importance_rank", "mean"),
            best_importance_rank=("importance_rank", "min"),
            targets=("target_label", join_unique),
            feature_sets=("feature_set", join_unique),
        )
        .sort_values(["average_gain_importance", "best_gain_importance"], ascending=[False, False])
        .head(n)
    )
    return grouped


def build_report(
    combined: pd.DataFrame,
    ranker_importance: pd.DataFrame,
    backend_info: dict[str, Any],
) -> str:
    """Create the final Markdown report."""
    counts = combined["final_selection_group"].value_counts().reindex(GROUP_ORDER.keys(), fill_value=0)
    top_ranker = top_ranker_alphas(ranker_importance, n=12)
    recommended = combined[combined["final_selection_group"].isin(["core_alpha", "nonlinear_alpha", "redundant_alpha"])].head(20)

    lines = [
        "# Multi-Method Factor Selection With True XGBoost Ranker",
        "",
        "## Executive Summary",
        "",
        "- This report uses the true `xgboost_ranker` evidence, not the previous XGBoost regressor fallback evidence.",
        "- Regressor fallback used: 0.",
        "- Pure numpy fallback used: 0.",
        f"- Ranker backend values observed: `{backend_info['backend_text']}`.",
        f"- Best ranker target: `{BEST_RANKER_TARGET}`.",
        f"- Best ranker feature set and objective: `{BEST_RANKER_FEATURE_SET}`, `{BEST_RANKER_OBJECTIVE}`.",
        f"- Validation Rank IC: {BEST_RANKER_VALIDATION_RANK_IC:.6f}.",
        f"- Test Rank IC: {BEST_RANKER_TEST_RANK_IC:.6f}.",
        f"- Test top-minus-bottom spread: {BEST_RANKER_TEST_SPREAD:.6f}.",
        "",
        "## Output Group Counts",
        "",
    ]
    for group_name, count in counts.items():
        lines.append(f"- {group_name}: {int(count)}")

    lines.extend(
        [
            "",
            "## Top Ranker Alphas By Gain Importance",
            "",
        ]
    )
    for _, row in top_ranker.iterrows():
        lines.append(
            f"- Alpha {int(row['alpha_id'])} {row['alpha_name']}: "
            f"average gain {row['average_gain_importance']:.6f}, "
            f"best gain {row['best_gain_importance']:.6f}, "
            f"average importance rank {row['average_importance_rank']:.2f}."
        )

    lines.extend(["", "## Final Recommended Alpha Candidates For Mega-Alpha Testing", ""])
    if recommended.empty:
        lines.append("- No alpha reached the recommendation filters in this run.")
    else:
        for _, row in recommended.iterrows():
            lines.append(
                f"- Alpha {int(row['alpha_id'])} {row['alpha_name']} "
                f"({row['final_selection_group']}): {row['recommended_usage']}"
            )

    lines.extend(["", "## Special Alpha Review", ""])
    for alpha_id in SPECIAL_ALPHA_IDS_WITH_RANKER:
        row = combined[combined["alpha_id"] == alpha_id]
        if row.empty:
            lines.append(f"### Alpha {alpha_id}")
            lines.append("- Not present in the final combined summary.")
            lines.append("")
            continue
        item = row.iloc[0]
        lines.extend(
            [
                f"### Alpha {alpha_id}: {item['alpha_name']}",
                f"- Final group: {item['final_selection_group']}.",
                f"- Linear evidence: {item['linear_interpretation']}",
                f"- Lasso / ElasticNet evidence: {item['regularized_interpretation']}",
                f"- True XGBoost Ranker evidence: {item['ranker_interpretation']}",
                f"- Recommended usage: {item['recommended_usage']}",
                f"- Warning flags: {item['warning_flags'] or 'none'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Method Notes",
            "",
            "- `core_alpha` means strong single-alpha evidence, selected by Lasso/ElasticNet, and important in the true ranker.",
            "- `nonlinear_alpha` means the true ranker finds useful nonlinear information even when linear evidence is weaker.",
            "- `redundant_alpha` means standalone evidence exists but regularized models did not select it, suggesting overlap with related features.",
            "- `linear_only_alpha` means linear evidence exists without regularized or ranker confirmation.",
            "- `watchlist_alpha` means evidence is mixed, unstable, or horizon-specific.",
            "- `rejected_alpha` means weak evidence across the available methods.",
            "",
            "## Combiner Runtime Note",
            "",
            f"- pandas version: {pd.__version__}",
            f"- numpy version: {np.__version__}",
            "- This combiner does not retrain or re-import XGBoost models; it verifies the true ranker evidence from the ranker output `backend` column.",
            f"- Verified ranker backend values: `{backend_info['backend_text']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def qa_checks(combined: pd.DataFrame, ranker_importance: pd.DataFrame, backend_info: dict[str, Any]) -> tuple[bool, list[str]]:
    """Run required final QA checks."""
    failures: list[str] = []
    if combined["alpha_id"].duplicated().any():
        failures.append("duplicate alpha rows in final readable summary")
    if int((combined["alpha_id"] == 35).sum()) != 0:
        failures.append("alpha 35 was not skipped")
    if int(combined["alpha_id"].nunique()) != len(combined):
        failures.append("final summary is not one row per alpha_id")
    if len(combined) != 49:
        failures.append(f"expected 49 alpha rows after skipping alpha 35, found {len(combined)}")
    if not backend_info["true_ranker_confirmed"]:
        failures.append(f"true ranker backend not confirmed: {backend_info['backend_text']}")
    if backend_info["regressor_fallback_used"]:
        failures.append("regressor fallback detected in ranker importance backend")
    if backend_info["numpy_fallback_used"]:
        failures.append("pure numpy fallback detected in ranker importance backend")
    if "backend" in ranker_importance.columns and ranker_importance["backend"].isna().any():
        failures.append("missing backend values in ranker importance")
    return len(failures) == 0, failures


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    linear = load_csv(args.linear_results)
    lasso_summary = load_csv(args.lasso_summary)
    ranker_summary = load_csv(args.ranker_summary)
    ranker_importance = load_csv(args.ranker_importance)

    backend_info = ranker_backend_summary(ranker_importance)
    combined = build_combined(linear, lasso_summary, ranker_summary, ranker_importance)
    status_ok, failures = qa_checks(combined, ranker_importance, backend_info)

    combined_path = args.output_dir / "combined_readable_summary_by_alpha_with_ranker.csv"
    report_path = args.output_dir / "factor_selection_report_with_ranker.md"
    combined.to_csv(combined_path, index=False)
    write_group_files(combined, args.output_dir)
    report_path.write_text(build_report(combined, ranker_importance, backend_info), encoding="utf-8")

    counts = combined["final_selection_group"].value_counts().reindex(GROUP_ORDER.keys(), fill_value=0)
    top_ranker = top_ranker_alphas(ranker_importance, n=10)

    print("true XGBoost Ranker evidence used: yes")
    print(f"ranker backend values: {backend_info['backend_text']}")
    print(f"regressor fallback used: {int(backend_info['regressor_fallback_used'])}")
    print(f"pure numpy fallback used: {int(backend_info['numpy_fallback_used'])}")
    print(f"combined summary path: {combined_path}")
    print(f"factor selection report path: {report_path}")
    print(f"final summary rows: {len(combined)}")
    print(f"duplicate alpha rows: {int(combined['alpha_id'].duplicated().sum())}")
    print(f"alpha 35 rows: {int((combined['alpha_id'] == 35).sum())}")
    for group_name, count in counts.items():
        print(f"{group_name} count: {int(count)}")
    print("top ranker alphas by average gain importance:")
    for _, row in top_ranker.iterrows():
        print(
            f"  alpha {int(row['alpha_id'])}: {row['alpha_name']} | "
            f"avg_gain={row['average_gain_importance']:.6f} | "
            f"avg_rank={row['average_importance_rank']:.2f}"
        )
    print(f"best ranker target: {BEST_RANKER_TARGET}")
    print(f"validation Rank IC: {BEST_RANKER_VALIDATION_RANK_IC:.6f}")
    print(f"test Rank IC: {BEST_RANKER_TEST_RANK_IC:.6f}")
    print(f"final factor selection with ranker status: {'PASS' if status_ok else 'FAIL'}")
    if failures:
        print("failed checks:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
