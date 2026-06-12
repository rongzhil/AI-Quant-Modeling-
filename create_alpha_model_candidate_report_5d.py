from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


INPUT_DIR = Path(r"F:\icarus fund\Quantitative AI financial modeling\regression")
PANEL_CSV = INPUT_DIR / "sp500_alpha_target_panel.csv"
IC_5D_CSV = INPUT_DIR / "alpha_rank_ic_summary_5d.csv"
REG_5D_CSV = INPUT_DIR / "single_alpha_regression_stata_style_summary_5d.csv"
MISSING_CSV = INPUT_DIR / "alpha_missing_summary.csv"

SUMMARY_OUT = INPUT_DIR / "alpha_model_candidate_summary_5d.csv"
REPORT_OUT = INPUT_DIR / "alpha_model_candidate_report_5d.txt"

CORR_THRESHOLD = 0.85
ZERO_EPS = 1e-12


SUMMARY_COLUMNS = [
    "alpha",
    "missing_ratio",
    "ic_n_days",
    "mean_ic",
    "ic_t_stat",
    "ic_p_value",
    "icir",
    "positive_ic_ratio",
    "mean_beta",
    "beta_t_stat",
    "beta_p_value",
    "positive_beta_ratio",
    "avg_daily_r_squared",
    "train_beta_mean",
    "val_beta_mean",
    "test_beta_mean",
    "same_direction_all",
    "ic_direction",
    "beta_direction",
    "direction_agreement",
    "final_direction",
    "alpha_class",
    "aligned_action",
    "aligned_feature_name",
    "max_abs_corr_with_other_candidate",
    "most_correlated_alpha",
    "redundancy_group_id",
    "redundancy_decision",
    "redundancy_reason",
    "final_model_recommendation",
    "final_model_feature",
    "reason",
]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def resolve_input_path(filename: str) -> Path:
    root_path = INPUT_DIR / filename
    if root_path.exists():
        return root_path
    matches = sorted(INPUT_DIR.rglob(filename))
    if matches:
        print(f"Using fallback input location for {filename}: {matches[0]}")
        return matches[0]
    raise FileNotFoundError(f"Required input file not found at root or below input folder: {filename}")


def detect_alpha_z_columns(path: Path) -> list[str]:
    header = pd.read_csv(path, nrows=0)
    alpha_z = [c for c in header.columns if c.startswith("alpha_") and c.endswith("_z")]
    if not alpha_z:
        raise ValueError("No alpha_*_z columns found in panel.")
    return alpha_z


def raw_alpha_name(alpha_z: str) -> str:
    return alpha_z[:-2] if alpha_z.endswith("_z") else alpha_z


def direction(value: Any) -> str:
    if pd.isna(value) or abs(float(value)) <= ZERO_EPS:
        return "Neutral"
    return "Positive" if float(value) > 0 else "Reverse"


def is_sig(value: Any, threshold: float) -> bool:
    return pd.notna(value) and float(value) <= threshold


def abs_ge(value: Any, threshold: float) -> bool:
    return pd.notna(value) and abs(float(value)) >= threshold


def ic_p_value(ic_t_stat: Any, n_days: Any) -> float:
    if pd.isna(ic_t_stat) or pd.isna(n_days):
        return np.nan
    df = int(n_days) - 1
    if df <= 0:
        return np.nan
    return float(2 * student_t.sf(abs(float(ic_t_stat)), df=df))


def val_test_acceptable(row: pd.Series, final_direction: str) -> bool:
    sign = 1 if final_direction == "Positive" else -1 if final_direction == "Reverse" else 0
    if sign == 0:
        return False
    vals = [row.get("val_beta_mean"), row.get("test_beta_mean")]
    usable = [float(v) for v in vals if pd.notna(v) and abs(float(v)) > ZERO_EPS]
    if not usable:
        return False
    return all(np.sign(v) == sign for v in usable)


def classify_alpha(row: pd.Series) -> tuple[str, str, str, str, str]:
    raw_missing = bool(row.get("all_nan_flag", False)) or bool(row.get("high_missing_flag", False))
    invalid = raw_missing or int(row.get("ic_n_days", 0) or 0) == 0 or int(row.get("n_days", 0) or 0) == 0 or pd.isna(row.get("beta_p_value"))
    if invalid:
        return "Missing / Invalid", "exclude", "", "Missing / Invalid", "Missing or invalid alpha: all-NaN/high-missing, zero valid days, or missing beta p-value."

    ic_dir = row["ic_direction"]
    beta_dir = row["beta_direction"]
    beta_sig_05 = is_sig(row.get("beta_p_value"), 0.05)
    beta_sig_10 = is_sig(row.get("beta_p_value"), 0.10)
    beta_meaningful = beta_sig_05 or abs_ge(row.get("beta_t_stat"), 2)
    ic_meaningful = is_sig(row.get("ic_p_value"), 0.05) or abs_ge(row.get("ic_t_stat"), 2)
    beta_weak = (not abs_ge(row.get("beta_t_stat"), 2)) and (pd.isna(row.get("beta_p_value")) or float(row.get("beta_p_value")) > 0.05)
    ic_weak = not ic_meaningful
    same_all = bool(row.get("same_direction_all", False))

    directions_conflict = ic_dir in {"Positive", "Reverse"} and beta_dir in {"Positive", "Reverse"} and ic_dir != beta_dir
    if directions_conflict and (beta_meaningful or ic_meaningful):
        return "Needs Review", "exclude", "", "Needs Review", "IC and regression beta directions conflict with statistical evidence."

    if beta_weak and ic_weak:
        return "Weak", "exclude", "", "Weak", "Both beta and Rank IC are statistically weak."

    if ic_dir == beta_dir and ic_dir in {"Positive", "Reverse"}:
        final_direction = ic_dir
    elif beta_dir in {"Positive", "Reverse"} and beta_meaningful and not directions_conflict:
        final_direction = beta_dir
    elif ic_dir in {"Positive", "Reverse"} and ic_meaningful and not directions_conflict:
        final_direction = ic_dir
    else:
        final_direction = "Weak"

    if final_direction == "Positive" and beta_sig_05 and abs_ge(row.get("beta_t_stat"), 2) and same_all:
        return "Core Positive", "keep_as_is", f"{row['alpha']}_aligned", "Positive", "Positive beta is significant and train/validation/test beta signs are aligned."
    if final_direction == "Reverse" and beta_sig_05 and abs_ge(row.get("beta_t_stat"), 2) and same_all:
        return "Core Reverse", "multiply_by_minus_one", f"{row['alpha']}_aligned", "Reverse", "Reverse beta is significant and train/validation/test beta signs are aligned."

    if beta_meaningful and not same_all:
        return "Unstable", "exclude", "", final_direction if final_direction in {"Positive", "Reverse"} else "Needs Review", "Beta is meaningful, but train/validation/test signs are not all aligned."

    if final_direction == "Positive" and (beta_sig_10 or abs_ge(row.get("ic_t_stat"), 2)) and val_test_acceptable(row, "Positive"):
        return "Potential Positive", "keep_as_is", f"{row['alpha']}_aligned", "Positive", "Positive evidence is promising but not strong enough for Core classification."
    if final_direction == "Reverse" and (beta_sig_10 or abs_ge(row.get("ic_t_stat"), 2)) and val_test_acceptable(row, "Reverse"):
        return "Potential Reverse", "multiply_by_minus_one", f"{row['alpha']}_aligned", "Reverse", "Reverse evidence is promising but not strong enough for Core classification."

    if final_direction in {"Positive", "Reverse"} and (beta_sig_10 or ic_meaningful):
        return "Needs Review", "exclude", "", final_direction, "Evidence exists, but validation/test direction or method agreement is not clean."

    return "Weak", "exclude", "", "Weak", "Signal is too weak for the baseline multi-factor model."


def class_rank(alpha_class: str) -> int:
    if alpha_class in {"Core Positive", "Core Reverse"}:
        return 3
    if alpha_class in {"Potential Positive", "Potential Reverse"}:
        return 2
    return 1


def economic_clarity_score(alpha: str) -> int:
    score = 0
    clear_terms = ["momentum", "return", "volatility", "volume", "liquidity", "trend", "range", "drawdown", "macd", "rsi"]
    for term in clear_terms:
        if term in alpha:
            score += 1
    if "reversal" in alpha:
        score -= 1
    return score


def choose_redundancy_keeper(group: list[str], summary: pd.DataFrame) -> str:
    candidates = summary.set_index("alpha").loc[group].copy()
    candidates["_class_rank"] = candidates["alpha_class"].map(class_rank)
    candidates["_beta_p_sort"] = candidates["beta_p_value"].fillna(np.inf)
    candidates["_abs_beta_t"] = candidates["beta_t_stat"].abs().fillna(-np.inf)
    candidates["_abs_ic_t"] = candidates["ic_t_stat"].abs().fillna(-np.inf)
    candidates["_r2"] = candidates["avg_daily_r_squared"].fillna(-np.inf)
    candidates["_missing_sort"] = candidates["missing_ratio"].fillna(np.inf)
    candidates["_clarity"] = [economic_clarity_score(a) for a in candidates.index]
    candidates = candidates.sort_values(
        ["_class_rank", "_beta_p_sort", "_abs_beta_t", "_abs_ic_t", "_r2", "_missing_sort", "_clarity"],
        ascending=[False, True, False, False, False, True, False],
    )
    return str(candidates.index[0])


def connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    seen: set[str] = set()
    groups: list[list[str]] = []
    for node in nodes:
        if node in seen:
            continue
        queue = deque([node])
        seen.add(node)
        comp: list[str] = []
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in graph[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        if len(comp) > 1:
            groups.append(sorted(comp))
    return groups


def build_base_summary(alpha_z_cols: list[str], ic_path: Path, reg_path: Path, missing_path: Path) -> pd.DataFrame:
    ic = pd.read_csv(ic_path)
    reg = pd.read_csv(reg_path)
    missing = pd.read_csv(missing_path)

    ic = ic.rename(columns={"n_days": "ic_n_days", "t_stat": "ic_t_stat", "p_value": "ic_p_value"})
    if "ic_p_value" not in ic.columns:
        ic["ic_p_value"] = np.nan
    ic["ic_p_value"] = ic.apply(
        lambda r: ic_p_value(r.get("ic_t_stat"), r.get("ic_n_days")) if pd.isna(r.get("ic_p_value")) else r.get("ic_p_value"),
        axis=1,
    )

    missing["alpha"] = missing["alpha_name"].map(lambda x: f"{x}_z")
    base = pd.DataFrame({"alpha": alpha_z_cols})
    base["raw_alpha"] = base["alpha"].map(raw_alpha_name)
    base = base.merge(
        missing[["alpha", "missing_ratio", "all_nan_flag", "high_missing_flag"]],
        on="alpha",
        how="left",
    )
    base = base.merge(
        ic[["alpha", "ic_n_days", "mean_ic", "ic_t_stat", "ic_p_value", "icir", "positive_ic_ratio"]],
        on="alpha",
        how="left",
    )
    base = base.merge(
        reg[
            [
                "alpha",
                "n_days",
                "mean_beta",
                "std_beta",
                "beta_t_stat",
                "beta_p_value",
                "positive_beta_ratio",
                "avg_daily_r_squared",
                "train_beta_mean",
                "val_beta_mean",
                "test_beta_mean",
                "same_direction_all",
            ]
        ],
        on="alpha",
        how="left",
    )

    base["missing_ratio"] = base["missing_ratio"].fillna(1.0)
    base["all_nan_flag"] = base["all_nan_flag"].fillna(False).astype(bool)
    base["high_missing_flag"] = base["high_missing_flag"].fillna(False).astype(bool)
    base["same_direction_all"] = base["same_direction_all"].fillna(False).astype(bool)
    base["ic_direction"] = base["mean_ic"].map(direction)
    base["beta_direction"] = base["mean_beta"].map(direction)

    def agreement(row: pd.Series) -> str:
        if row["ic_direction"] == "Neutral" or row["beta_direction"] == "Neutral":
            return "Weak/NA"
        return "True" if row["ic_direction"] == row["beta_direction"] else "False"

    base["direction_agreement"] = base.apply(agreement, axis=1)
    classified = base.apply(classify_alpha, axis=1, result_type="expand")
    classified.columns = ["alpha_class", "aligned_action", "aligned_feature_name", "final_direction", "reason"]
    base = pd.concat([base, classified], axis=1)
    return base


def add_redundancy(summary: pd.DataFrame, alpha_z_cols: list[str], panel_path: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    candidate_mask = summary["aligned_action"].isin(["keep_as_is", "multiply_by_minus_one"])
    candidate_alphas = summary.loc[candidate_mask, "alpha"].tolist()

    summary["max_abs_corr_with_other_candidate"] = np.nan
    summary["most_correlated_alpha"] = ""
    summary["redundancy_group_id"] = ""
    summary["redundancy_decision"] = np.where(candidate_mask, "keep", "not_candidate")
    summary["redundancy_reason"] = np.where(candidate_mask, "No candidate correlation above redundancy threshold.", "Not a model candidate.")

    if len(candidate_alphas) < 2:
        return summary, []

    print(f"Loading {len(candidate_alphas)} candidate alpha columns for redundancy analysis...")
    panel = pd.read_csv(panel_path, usecols=candidate_alphas)
    aligned = pd.DataFrame(index=panel.index)
    action_map = summary.set_index("alpha")["aligned_action"].to_dict()
    for alpha in candidate_alphas:
        values = pd.to_numeric(panel[alpha], errors="coerce")
        aligned[alpha] = -values if action_map[alpha] == "multiply_by_minus_one" else values

    corr = aligned.corr(method="pearson", min_periods=50)
    edges: list[tuple[str, str]] = []
    pair_corr: dict[tuple[str, str], float] = {}
    for i, a in enumerate(candidate_alphas):
        others = corr.loc[a, candidate_alphas].drop(labels=[a])
        if not others.empty:
            abs_others = others.abs()
            max_alpha = str(abs_others.idxmax())
            summary.loc[summary["alpha"].eq(a), "max_abs_corr_with_other_candidate"] = float(abs_others.loc[max_alpha])
            summary.loc[summary["alpha"].eq(a), "most_correlated_alpha"] = max_alpha
        for b in candidate_alphas[i + 1 :]:
            value = corr.loc[a, b]
            if pd.notna(value):
                pair_corr[(a, b)] = float(value)
                if abs(float(value)) >= CORR_THRESHOLD:
                    edges.append((a, b))

    groups = connected_components(candidate_alphas, edges)
    group_summaries: list[dict[str, Any]] = []
    for idx, group in enumerate(groups, start=1):
        group_id = f"RG{idx:02d}"
        keeper = choose_redundancy_keeper(group, summary)
        dropped = [a for a in group if a != keeper]
        for alpha in group:
            summary.loc[summary["alpha"].eq(alpha), "redundancy_group_id"] = group_id
            if alpha == keeper:
                summary.loc[summary["alpha"].eq(alpha), "redundancy_decision"] = "keep"
                summary.loc[summary["alpha"].eq(alpha), "redundancy_reason"] = (
                    f"Kept within {group_id} by class strength, p-value, t-stats, R-squared, missingness, and economic clarity."
                )
            else:
                summary.loc[summary["alpha"].eq(alpha), "redundancy_decision"] = "drop_redundant"
                summary.loc[summary["alpha"].eq(alpha), "redundancy_reason"] = (
                    f"Dropped as redundant with {keeper} in {group_id}; absolute aligned correlation >= {CORR_THRESHOLD:.2f}."
                )

        dropped_corrs = []
        for alpha in dropped:
            key = tuple(sorted([keeper, alpha]))
            corr_value = pair_corr.get(key, corr.loc[keeper, alpha])
            dropped_corrs.append((alpha, float(corr_value)))
        group_summaries.append(
            {
                "group_id": group_id,
                "kept_alpha": keeper,
                "dropped": dropped_corrs,
                "members": group,
                "reason": "Selected one representative using Core/Potential class, beta p-value, beta and IC t-stats, R-squared, missingness, and economic clarity.",
            }
        )

    return summary, group_summaries


def add_final_recommendations(summary: pd.DataFrame) -> pd.DataFrame:
    recommendations = []
    final_features = []
    for _, row in summary.iterrows():
        if row["redundancy_decision"] == "drop_redundant":
            recommendations.append("Drop")
            final_features.append("")
        elif row["alpha_class"] in {"Core Positive", "Core Reverse"} and row["redundancy_decision"] == "keep":
            recommendations.append("Strong Keep")
            final_features.append(row["aligned_feature_name"])
        elif row["alpha_class"] in {"Potential Positive", "Potential Reverse"} and row["redundancy_decision"] == "keep":
            recommendations.append("Optional Keep")
            final_features.append(row["aligned_feature_name"])
        elif row["alpha_class"] in {"Needs Review", "Unstable"}:
            recommendations.append("Review")
            final_features.append("")
        else:
            recommendations.append("Drop")
            final_features.append("")
    summary["final_model_recommendation"] = recommendations
    summary["final_model_feature"] = final_features
    return summary


def format_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 50) -> str:
    if columns is not None:
        df = df[columns]
    if df.empty:
        return "(none)"
    return df.head(max_rows).to_string(index=False)


def build_report(summary: pd.DataFrame, group_summaries: list[dict[str, Any]], total_alpha_z: int) -> str:
    counts = summary["alpha_class"].value_counts().to_dict()
    rec_counts = summary["final_model_recommendation"].value_counts().to_dict()
    valid_count = int((summary["alpha_class"] != "Missing / Invalid").sum())
    missing_invalid = int((summary["alpha_class"] == "Missing / Invalid").sum())
    strong = summary.loc[summary["final_model_recommendation"].eq("Strong Keep")].sort_values(
        "beta_t_stat", key=lambda s: s.abs(), ascending=False
    )
    optional = summary.loc[summary["final_model_recommendation"].eq("Optional Keep")].sort_values(
        "beta_t_stat", key=lambda s: s.abs(), ascending=False
    )
    needs_review = summary.loc[summary["final_model_recommendation"].eq("Review")].sort_values(
        "beta_t_stat", key=lambda s: s.abs(), ascending=False
    )

    report_cols = [
        "alpha",
        "final_direction",
        "beta_t_stat",
        "beta_p_value",
        "mean_ic",
        "ic_t_stat",
        "avg_daily_r_squared",
        "redundancy_decision",
        "reason",
    ]

    drop_missing = summary.loc[summary["alpha_class"].eq("Missing / Invalid")]
    drop_weak = summary.loc[summary["alpha_class"].eq("Weak")]
    drop_unstable = summary.loc[summary["alpha_class"].eq("Unstable")]
    drop_redundant = summary.loc[summary["redundancy_decision"].eq("drop_redundant")]

    redundancy_lines = []
    if group_summaries:
        for group in group_summaries:
            dropped_text = ", ".join(f"{alpha} (corr={corr:.3f})" for alpha, corr in group["dropped"])
            redundancy_lines.append(
                f"- {group['group_id']}: kept {group['kept_alpha']}; dropped {dropped_text}. {group['reason']}"
            )
    else:
        redundancy_lines.append("- No aligned candidate pairs had absolute Pearson correlation >= 0.85.")

    feature_list = summary.loc[summary["final_model_feature"].ne(""), "final_model_feature"].tolist()
    feature_lines = "\n".join(f"- {feature}" for feature in feature_list) if feature_list else "(none)"

    return f"""Alpha Model Candidate Report: 5-Day Forward Return

Executive Summary
- Total alpha_z columns: {total_alpha_z}
- Valid alphas: {valid_count}
- Missing / Invalid alphas: {missing_invalid}
- Core Positive: {counts.get("Core Positive", 0)}
- Core Reverse: {counts.get("Core Reverse", 0)}
- Potential Positive: {counts.get("Potential Positive", 0)}
- Potential Reverse: {counts.get("Potential Reverse", 0)}
- Weak: {counts.get("Weak", 0)}
- Unstable: {counts.get("Unstable", 0)}
- Needs Review: {counts.get("Needs Review", 0)}
- Strong Keep: {rec_counts.get("Strong Keep", 0)}
- Optional Keep: {rec_counts.get("Optional Keep", 0)}
- Drop: {rec_counts.get("Drop", 0)}
- Review: {rec_counts.get("Review", 0)}

Strong Keep Factors for Multi-Factor Model
{format_table(strong, report_cols, 100)}

Optional Keep Factors
{format_table(optional, report_cols, 100)}

Drop Factors by Reason

Missing / Invalid
{format_table(drop_missing, ["alpha", "alpha_class", "reason"], 100)}

Weak Predictive Power
{format_table(drop_weak, ["alpha", "alpha_class", "beta_t_stat", "beta_p_value", "ic_t_stat", "ic_p_value", "reason"], 100)}

Unstable Direction
{format_table(drop_unstable, ["alpha", "alpha_class", "train_beta_mean", "val_beta_mean", "test_beta_mean", "reason"], 100)}

Redundant Duplicate
{format_table(drop_redundant, ["alpha", "redundancy_group_id", "most_correlated_alpha", "max_abs_corr_with_other_candidate", "redundancy_reason"], 100)}

Needs Review Factors
{format_table(needs_review, ["alpha", "alpha_class", "ic_direction", "beta_direction", "direction_agreement", "train_beta_mean", "val_beta_mean", "test_beta_mean", "reason"], 100)}

Redundancy Summary
{chr(10).join(redundancy_lines)}

Final Recommended Features for Multi-Factor Regression
{feature_lines}

Modeling Instruction
Use only final_model_feature columns for the next multi-factor regression. Positive alphas are kept as-is. Reverse alphas should be multiplied by -1 before regression. Weak, unstable, missing, and redundant alphas should not be used in the baseline multi-factor model.
"""


def main() -> None:
    panel_path = resolve_input_path("sp500_alpha_target_panel.csv")
    ic_path = resolve_input_path("alpha_rank_ic_summary_5d.csv")
    reg_path = resolve_input_path("single_alpha_regression_stata_style_summary_5d.csv")
    missing_path = resolve_input_path("alpha_missing_summary.csv")

    print("Detecting alpha_z columns...")
    alpha_z_cols = detect_alpha_z_columns(panel_path)
    print(f"Total alpha_z columns: {len(alpha_z_cols)}")

    print("Building merged IC/regression/missingness summary...")
    summary = build_base_summary(alpha_z_cols, ic_path, reg_path, missing_path)
    summary, group_summaries = add_redundancy(summary, alpha_z_cols, panel_path)
    summary = add_final_recommendations(summary)
    summary = summary[SUMMARY_COLUMNS].sort_values(["final_model_recommendation", "alpha"]).reset_index(drop=True)

    print("Saving final candidate summary and report...")
    summary.to_csv(SUMMARY_OUT, index=False)
    report = build_report(summary, group_summaries, len(alpha_z_cols))
    REPORT_OUT.write_text(report, encoding="utf-8")

    strong_count = int(summary["final_model_recommendation"].eq("Strong Keep").sum())
    optional_count = int(summary["final_model_recommendation"].eq("Optional Keep").sum())
    drop_count = int(summary["final_model_recommendation"].eq("Drop").sum())
    review_count = int(summary["final_model_recommendation"].eq("Review").sum())
    final_features = summary.loc[summary["final_model_feature"].ne(""), "final_model_feature"].tolist()

    print("\nFinal alpha model candidate outputs:")
    print(f"  {SUMMARY_OUT}")
    print(f"  {REPORT_OUT}")
    print(f"Strong Keep count: {strong_count}")
    print(f"Optional Keep count: {optional_count}")
    print(f"Drop count: {drop_count}")
    print(f"Review count: {review_count}")
    print("Final recommended feature list:")
    if final_features:
        for feature in final_features:
            print(f"  {feature}")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
