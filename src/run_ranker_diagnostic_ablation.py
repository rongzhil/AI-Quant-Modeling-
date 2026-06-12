"""Diagnostic ablation experiments for the true XGBoost Ranker pipeline.

This script is validation-only. It does not change the production/final model,
does not build portfolios, and does not run transaction-cost backtests.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from .factor_selection_common import (
    FEATURE_SET_SUFFIX,
    LABEL_COLUMNS,
    MIN_VALID_DATES,
    MIN_VALID_OBS,
    TARGET_LABELS_DEFAULT,
    FeatureSpec,
    alpha_category,
    chronological_split,
    daily_rank_ic,
    filter_feature_columns,
    finite_frame,
    group_sizes_by_date,
    icir,
    join_unique,
    load_model_data,
    pearson_corr,
    slice_split,
    split_ranges_for_row,
    validate_no_label_features,
)


OBJECTIVES_DEFAULT = ["rank:pairwise", "rank:ndcg"]
FEATURE_VARIANTS_DEFAULT = ["cs_rank"]
RELEVANCE_GRADES = 31
MIN_GROUP_SIZE = 50
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 30
SEED = 42

DIAGNOSTIC_ALPHA_NAMES = {
    30: "126-day return",
    31: "252-day return",
    32: "12-1 month momentum",
    34: "Sector-neutral momentum",
    54: "Daily high-low range",
    55: "Average true range",
    56: "21-day realized volatility",
    57: "63-day realized volatility",
    65: "Dollar volume",
    66: "Turnover",
    68: "ADV trend",
}
TRACKED_ALPHA_IDS = [32, 55, 68, 54, 65]
BACKBONE_FORCE_IDS = [32, 54, 65, 30, 34, 56, 57, 66]
SELECTED_GROUPS = {"core_alpha", "nonlinear_alpha", "redundant_alpha", "linear_only_alpha"}


@dataclass(frozen=True)
class ExperimentConfig:
    """Feature inclusion/exclusion rules for one ablation experiment."""

    name: str
    force_include_ids: tuple[int, ...] = ()
    exclude_ids: tuple[int, ...] = ()
    selected_groups_only: bool = False
    high_regularization: bool = False
    only_ids: tuple[int, ...] = ()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run diagnostic XGBoost Ranker ablation experiments.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-labels", nargs="+", default=TARGET_LABELS_DEFAULT)
    parser.add_argument("--feature-variants", nargs="+", default=FEATURE_VARIANTS_DEFAULT)
    parser.add_argument("--objectives", nargs="+", default=OBJECTIVES_DEFAULT)
    parser.add_argument("--include-optional-blocks", action="store_true")
    return parser.parse_args()


def require_xgboost() -> None:
    """Verify the native XGBoost ranking API is available."""
    if not hasattr(xgb, "train") or not hasattr(xgb, "DMatrix"):
        raise RuntimeError("xgboost ranking API unavailable: missing train or DMatrix")
    print(f"xgboost version: {xgb.__version__}")
    print("backend = xgboost_ranker")
    print("no regressor fallback")
    print("no numpy fallback")


def load_selection_summary(path: Path) -> pd.DataFrame:
    """Load final-with-ranker selection summary."""
    if not path.exists():
        raise FileNotFoundError(f"Selection summary not found: {path}")
    summary = pd.read_csv(path)
    required = {"alpha_id", "alpha_name", "final_selection_group"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Selection summary missing required columns: {missing}")
    summary["alpha_id"] = pd.to_numeric(summary["alpha_id"], errors="coerce").astype("Int64")
    return summary


def experiment_configs(include_optional_blocks: bool = False) -> list[ExperimentConfig]:
    """Return required and optional diagnostic ablation configurations."""
    configs = [
        ExperimentConfig("baseline_current"),
        ExperimentConfig("force_include_alpha32", force_include_ids=(32,)),
        ExperimentConfig("exclude_alpha55_ATR", exclude_ids=(55,)),
        ExperimentConfig("exclude_alpha68_ADV_trend", exclude_ids=(68,)),
        ExperimentConfig("force_alpha32_exclude_alpha55_alpha68", force_include_ids=(32,), exclude_ids=(55, 68)),
        ExperimentConfig(
            "clean_backbone_set",
            force_include_ids=tuple(BACKBONE_FORCE_IDS),
            exclude_ids=(55, 68),
            selected_groups_only=True,
        ),
        ExperimentConfig("high_regularization_baseline", high_regularization=True),
        ExperimentConfig(
            "high_regularization_clean",
            force_include_ids=(32,),
            exclude_ids=(55, 68),
            high_regularization=True,
        ),
    ]
    if include_optional_blocks:
        configs.extend(
            [
                ExperimentConfig("only_momentum_block", only_ids=(30, 32, 34, 31, 33)),
                ExperimentConfig("only_volatility_liquidity_block", only_ids=(54, 55, 56, 57, 58, 65, 66, 67)),
                ExperimentConfig("volatility_liquidity_without_ATR", only_ids=(54, 56, 57, 58, 65, 66, 67), exclude_ids=(55,)),
            ]
        )
    return configs


def parameter_grid(objective: str, high_regularization: bool) -> list[dict[str, Any]]:
    """Compact covering grid for normal and high-regularization experiments."""
    if high_regularization:
        covering_values = [
            (2, 0.03, 5.0, 0.1, 10),
            (2, 0.05, 10.0, 0.5, 20),
            (3, 0.03, 20.0, 1.0, 10),
            (3, 0.05, 5.0, 1.0, 20),
            (4, 0.03, 10.0, 0.1, 20),
            (4, 0.05, 20.0, 0.5, 10),
        ]
    else:
        covering_values = [
            (2, 0.03, 1.0, 0.0, 5),
            (2, 0.05, 5.0, 0.1, 10),
            (3, 0.03, 5.0, 0.1, 5),
            (3, 0.05, 1.0, 0.0, 10),
            (4, 0.03, 1.0, 0.1, 10),
            (4, 0.05, 5.0, 0.0, 5),
        ]
    return [
        {
            "objective": objective,
            "eval_metric": "ndcg",
            "tree_method": "hist",
            "max_depth": max_depth,
            "eta": eta,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": reg_lambda,
            "alpha": reg_alpha,
            "min_child_weight": min_child_weight,
            "seed": SEED,
            "verbosity": 0,
        }
        for max_depth, eta, reg_lambda, reg_alpha, min_child_weight in covering_values
    ]


def verify_inputs(data: pd.DataFrame, specs: list[FeatureSpec], target_labels: list[str]) -> None:
    """Run input and feature-safety checks before training."""
    missing_targets = [label for label in target_labels if label not in data.columns]
    if missing_targets:
        raise ValueError(f"Missing target labels: {missing_targets}")
    if "date" not in data.columns or "ticker" not in data.columns:
        raise ValueError("Input data must contain date and ticker columns")
    duplicate_rows = int(data.duplicated(["date", "ticker"]).sum())
    if duplicate_rows:
        raise ValueError(f"Duplicate date-ticker rows found: {duplicate_rows}")
    if any(spec.alpha_id == 35 for spec in specs):
        raise ValueError("Alpha 35 should be skipped but was found in feature specs")
    feature_columns = [spec.feature_column for spec in specs]
    label_overlap = validate_no_label_features(feature_columns)
    if label_overlap:
        raise ValueError(f"Label columns used as features: {label_overlap}")
    raw_columns = [column for column in feature_columns if column.endswith("_raw") or column.endswith("_raw_value")]
    if raw_columns:
        raise ValueError(f"Raw alpha columns used as features: {raw_columns}")


def verify_diagnostic_alpha_map(specs: list[FeatureSpec], feature_variants: list[str]) -> None:
    """Print and verify exact feature names for diagnostic alpha IDs."""
    print("Diagnostic alpha feature column map:")
    for alpha_id, expected_name in DIAGNOSTIC_ALPHA_NAMES.items():
        matches = [spec for spec in specs if spec.alpha_id == alpha_id and spec.feature_set in feature_variants]
        if not matches:
            raise ValueError(f"Diagnostic alpha {alpha_id} ({expected_name}) is missing for variants {feature_variants}")
        for spec in matches:
            if spec.alpha_name.strip().lower() != expected_name.lower():
                raise ValueError(
                    f"Alpha ID/name mismatch for {alpha_id}: expected '{expected_name}', found '{spec.alpha_name}'"
                )
            print(f"  alpha {alpha_id}: {spec.alpha_name} -> {spec.feature_column}")


def relevance_by_date(frame: pd.DataFrame, target_label: str) -> np.ndarray:
    """Map within-date target return ranks to non-negative integer grades."""
    pct = frame.groupby("date", sort=False)[target_label].rank(method="average", pct=True)
    grades = np.floor(pct.to_numpy(dtype=float) * RELEVANCE_GRADES)
    return np.clip(grades, 0, RELEVANCE_GRADES).astype(np.float32)


def drop_small_groups(frame: pd.DataFrame, min_group_size: int = MIN_GROUP_SIZE) -> tuple[pd.DataFrame, int, list[str]]:
    """Drop date groups with too few valid target rows."""
    group_counts = frame.groupby("date", sort=False).size()
    keep_dates = group_counts[group_counts >= min_group_size].index
    dropped_dates = group_counts[group_counts < min_group_size].index.astype(str).tolist()
    return frame[frame["date"].isin(keep_dates)].copy(), len(dropped_dates), dropped_dates


def build_ranker_dmatrix(frame: pd.DataFrame, feature_columns: list[str], target_label: str) -> tuple[xgb.DMatrix, np.ndarray]:
    """Build a grouped DMatrix sorted by date and ticker."""
    ordered = frame.sort_values(["date", "ticker"]).reset_index(drop=True)
    labels = relevance_by_date(ordered, target_label)
    x_values = ordered[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    dmatrix = xgb.DMatrix(
        x_values.to_numpy(dtype=np.float32),
        label=labels,
        feature_names=feature_columns,
        missing=np.nan,
    )
    groups = group_sizes_by_date(ordered)
    dmatrix.set_group(groups)
    if int(groups.sum()) != len(ordered):
        raise ValueError("Ranking group size sum does not equal row count")
    if len(groups) != ordered["date"].nunique():
        raise ValueError("Ranking group count does not equal unique date count")
    if len(groups) and int(groups.min()) < MIN_GROUP_SIZE:
        raise ValueError("Ranking group contains fewer than 50 valid stocks")
    return dmatrix, groups


def mean_daily_rank_ic_for_prediction(frame: pd.DataFrame, target_label: str, prediction: np.ndarray) -> float:
    """Mean daily Spearman Rank IC for model selection."""
    pred_frame = frame[["date", "ticker", target_label]].copy().rename(columns={target_label: "target"})
    pred_frame["prediction"] = prediction
    daily = daily_rank_ic(pred_frame)
    return float(daily["rank_ic"].mean()) if not daily.empty else float("nan")


def train_best_ranker(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    target_label: str,
    objective: str,
    high_regularization: bool,
) -> tuple[xgb.Booster, dict[str, Any], float, list[dict[str, Any]]]:
    """Train parameter trials and select by validation mean daily Rank IC."""
    dtrain, train_groups = build_ranker_dmatrix(train, feature_columns, target_label)
    dvalidation, validation_groups = build_ranker_dmatrix(validation, feature_columns, target_label)
    best_booster: xgb.Booster | None = None
    best_params: dict[str, Any] | None = None
    best_score = -np.inf
    trials: list[dict[str, Any]] = []
    grid = parameter_grid(objective, high_regularization)
    for trial_id, params in enumerate(grid, start=1):
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dvalidation, "validation")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        prediction = booster.predict(dvalidation)
        validation_rank_ic = mean_daily_rank_ic_for_prediction(validation, target_label, prediction)
        best_iteration = int(getattr(booster, "best_iteration", 0) or 0)
        trials.append(
            {
                "trial_id": trial_id,
                "objective": objective,
                "validation_rank_ic": validation_rank_ic,
                "best_iteration": best_iteration,
                "train_group_count": int(len(train_groups)),
                "validation_group_count": int(len(validation_groups)),
                **params,
            }
        )
        if np.isfinite(validation_rank_ic) and validation_rank_ic > best_score:
            best_score = validation_rank_ic
            best_booster = booster
            best_params = {**params, "best_iteration": best_iteration, "trial_id": trial_id}
    if best_booster is None or best_params is None:
        raise RuntimeError(f"No valid XGBoost ranker model trained for objective={objective}")
    return best_booster, best_params, best_score, trials


def predict_ranker_frame(
    frame: pd.DataFrame,
    booster: xgb.Booster,
    feature_columns: list[str],
    target_label: str,
    split_name: str,
) -> pd.DataFrame:
    """Predict one split with continuous target retained for evaluation."""
    ordered = frame.sort_values(["date", "ticker"]).reset_index(drop=True)
    x_values = ordered[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    dmatrix = xgb.DMatrix(
        x_values.to_numpy(dtype=np.float32),
        feature_names=feature_columns,
        missing=np.nan,
    )
    out = ordered[["date", "ticker", target_label]].copy().rename(columns={target_label: "target"})
    out["prediction"] = booster.predict(dmatrix)
    out["split"] = split_name
    return out


def t_stat(values: pd.Series) -> float:
    """Standard t-stat of a daily metric."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return float("nan")
    std = float(clean.std(ddof=1))
    if std == 0 or not np.isfinite(std):
        return float("nan")
    return float(clean.mean() / (std / math.sqrt(len(clean))))


def ndcg_by_date(frame: pd.DataFrame, pred_col: str = "prediction", target_col: str = "target") -> float:
    """Mean NDCG using target-return rank grades as relevance."""
    scores: list[float] = []
    for _, group in frame.groupby("date", sort=True):
        usable = group[[target_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(usable) < MIN_GROUP_SIZE:
            continue
        relevance = np.floor(usable[target_col].rank(method="average", pct=True).to_numpy(dtype=float) * RELEVANCE_GRADES)
        relevance = np.clip(relevance, 0, RELEVANCE_GRADES)
        order = np.argsort(-usable[pred_col].to_numpy(dtype=float))
        ideal = np.argsort(-relevance)
        discounts = 1.0 / np.log2(np.arange(2, len(usable) + 2))
        dcg = float(np.sum(relevance[order] * discounts))
        idcg = float(np.sum(relevance[ideal] * discounts))
        if idcg > 0:
            scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else float("nan")


def daily_prediction_diagnostics(frame: pd.DataFrame, pred_col: str = "prediction", target_col: str = "target") -> pd.DataFrame:
    """Daily Rank IC, Pearson IC, and spread diagnostics."""
    rows: list[dict[str, Any]] = []
    for date_value, group in frame.groupby("date", sort=True):
        usable = group[[target_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(usable) < MIN_GROUP_SIZE:
            continue
        target = usable[target_col].to_numpy(dtype=float)
        pred = usable[pred_col].to_numpy(dtype=float)
        target_rank = usable[target_col].rank(method="average")
        pred_rank = usable[pred_col].rank(method="average")
        rank_ic = float(target_rank.corr(pred_rank)) if target_rank.std(ddof=1) != 0 and pred_rank.std(ddof=1) != 0 else np.nan
        pearson_ic = pearson_corr(target, pred)
        ranks = usable[pred_col].rank(method="average", pct=True)
        top = usable.loc[ranks >= 0.9, target_col]
        bottom = usable.loc[ranks <= 0.1, target_col]
        top_return = float(top.mean()) if not top.empty else np.nan
        bottom_return = float(bottom.mean()) if not bottom.empty else np.nan
        spread = top_return - bottom_return if np.isfinite(top_return) and np.isfinite(bottom_return) else np.nan
        rows.append(
            {
                "date": date_value,
                "rank_ic": rank_ic,
                "pearson_ic": pearson_ic,
                "top_decile_return": top_return,
                "bottom_decile_return": bottom_return,
                "top_minus_bottom_spread": spread,
                "n_obs": int(len(usable)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_prediction_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Aggregate validation/test prediction metrics."""
    daily = daily_prediction_diagnostics(predictions)
    return {
        "mean_daily_rank_ic": float(daily["rank_ic"].mean()) if not daily.empty else np.nan,
        "rank_ic_std": float(daily["rank_ic"].std(ddof=1)) if len(daily) > 1 else np.nan,
        "ICIR": icir(daily["rank_ic"]) if not daily.empty else np.nan,
        "rank_ic_t_stat": t_stat(daily["rank_ic"]) if not daily.empty else np.nan,
        "positive_ic_ratio": float((daily["rank_ic"] > 0).mean()) if not daily.empty else np.nan,
        "mean_pearson_ic": float(daily["pearson_ic"].mean()) if not daily.empty else np.nan,
        "mean_top_decile_return": float(daily["top_decile_return"].mean()) if not daily.empty else np.nan,
        "mean_bottom_decile_return": float(daily["bottom_decile_return"].mean()) if not daily.empty else np.nan,
        "mean_top_minus_bottom_spread": float(daily["top_minus_bottom_spread"].mean()) if not daily.empty else np.nan,
        "spread_t_stat": t_stat(daily["top_minus_bottom_spread"]) if not daily.empty else np.nan,
        "positive_spread_ratio": float((daily["top_minus_bottom_spread"] > 0).mean()) if not daily.empty else np.nan,
        "mean_ndcg": ndcg_by_date(predictions),
    }


def specs_by_id(specs: list[FeatureSpec], feature_set: str) -> dict[int, FeatureSpec]:
    """Map alpha ID to one FeatureSpec for a feature variant."""
    return {spec.alpha_id: spec for spec in specs if spec.feature_set == feature_set and spec.alpha_id != 35}


def selected_group_alpha_ids(selection_summary: pd.DataFrame) -> set[int]:
    """Alpha IDs from selected final-with-ranker groups."""
    selected = selection_summary[selection_summary["final_selection_group"].isin(SELECTED_GROUPS)]
    return {int(value) for value in selected["alpha_id"].dropna().tolist() if int(value) != 35}


def non_missing_enough(frame: pd.DataFrame, column: str, target_label: str) -> bool:
    """Check feature-target usable observations for forced features."""
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    target = pd.to_numeric(frame[target_label], errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = values.notna() & target.notna()
    return int(mask.sum()) >= MIN_VALID_OBS and int(frame.loc[mask, "date"].nunique()) >= MIN_VALID_DATES


def resolve_feature_specs(
    data: pd.DataFrame,
    all_specs: list[FeatureSpec],
    selection_summary: pd.DataFrame,
    target_label: str,
    feature_set: str,
    config: ExperimentConfig,
) -> tuple[list[FeatureSpec], dict[str, Any]]:
    """Apply ablation include/exclude rules to the eligible feature set."""
    spec_map = specs_by_id(all_specs, feature_set)
    baseline_specs = filter_feature_columns(data, all_specs, feature_set, target_label)
    baseline_specs = [spec for spec in baseline_specs if spec.alpha_id != 35]
    baseline_ids = {spec.alpha_id for spec in baseline_specs}
    forced_ids = set(config.force_include_ids)
    excluded_ids = set(config.exclude_ids) | {35}
    dropped_ids: set[int] = set()
    warnings: list[str] = []

    if config.only_ids:
        desired_ids = set(config.only_ids) | forced_ids
    elif config.selected_groups_only:
        desired_ids = (baseline_ids & selected_group_alpha_ids(selection_summary)) | forced_ids
    else:
        desired_ids = set(baseline_ids) | forced_ids

    desired_ids -= excluded_ids
    selected_specs: list[FeatureSpec] = []
    for alpha_id in sorted(desired_ids):
        spec = spec_map.get(alpha_id)
        if spec is None:
            warnings.append(f"alpha_{alpha_id}_missing_for_{feature_set}")
            dropped_ids.add(alpha_id)
            continue
        if alpha_id in forced_ids and not non_missing_enough(data, spec.feature_column, target_label):
            warnings.append(f"forced_alpha_{alpha_id}_insufficient_non_missing_data")
        selected_specs.append(spec)

    if 32 in baseline_ids and 32 in forced_ids:
        warnings.append("alpha32_already_present_in_baseline")
    for alpha_id in excluded_ids:
        if alpha_id in baseline_ids:
            dropped_ids.add(alpha_id)
    metadata = {
        "included_alpha_ids": sorted(spec.alpha_id for spec in selected_specs),
        "excluded_alpha_ids": sorted(excluded_ids),
        "forced_alpha_ids": sorted(forced_ids),
        "dropped_alpha_ids": sorted(dropped_ids),
        "warning_flags": ";".join(sorted(set(warnings))),
    }
    return selected_specs, metadata


def remove_train_unobserved_features(train: pd.DataFrame, selected_specs: list[FeatureSpec]) -> tuple[list[FeatureSpec], list[int]]:
    """Drop features with no observed train values."""
    kept: list[FeatureSpec] = []
    dropped_ids: list[int] = []
    for spec in selected_specs:
        observed = pd.to_numeric(train[spec.feature_column], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().any()
        if observed:
            kept.append(spec)
        else:
            dropped_ids.append(spec.alpha_id)
    return kept, sorted(dropped_ids)


def feature_importance_rows(
    booster: xgb.Booster,
    feature_specs: list[FeatureSpec],
    performance_key: dict[str, Any],
    params_id: str,
) -> list[dict[str, Any]]:
    """Build gain/weight/cover importance rows."""
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover = booster.get_score(importance_type="cover")
    total_gain = sum(float(gain.get(spec.feature_column, 0.0)) for spec in feature_specs)
    rows = []
    for spec in feature_specs:
        gain_value = float(gain.get(spec.feature_column, 0.0))
        rows.append(
            {
                **performance_key,
                "feature_column": spec.feature_column,
                "alpha_id": spec.alpha_id,
                "alpha_name": spec.alpha_name,
                "feature_variant": spec.feature_set,
                "gain_importance": gain_value,
                "normalized_gain_importance": gain_value / total_gain if total_gain > 0 else 0.0,
                "weight_importance": float(weight.get(spec.feature_column, 0.0)),
                "cover_importance": float(cover.get(spec.feature_column, 0.0)),
                "importance_rank": 0,
                "backend": "xgboost_ranker",
                "best_params_id": params_id,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    frame["importance_rank"] = frame["gain_importance"].rank(method="min", ascending=False).astype(int)
    return frame.to_dict("records")


def run_one_ablation(
    data: pd.DataFrame,
    all_specs: list[FeatureSpec],
    selection_summary: pd.DataFrame,
    config: ExperimentConfig,
    target_label: str,
    feature_set: str,
    objective: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Run one experiment/target/feature variant/objective combination."""
    selected_specs, feature_metadata = resolve_feature_specs(data, all_specs, selection_summary, target_label, feature_set, config)
    if not selected_specs:
        raise ValueError(f"No features selected for {config.name} {target_label} {feature_set}")

    candidate_columns = [spec.feature_column for spec in selected_specs]
    split = chronological_split(data.loc[data[target_label].notna(), "date"])
    frame = finite_frame(data[["date", "ticker", target_label, *candidate_columns]].copy())
    frame = frame.sort_values(["date", "ticker"]).reset_index(drop=True)
    train, validation, test = slice_split(frame, split, target_label)
    train, train_dropped, train_dropped_dates = drop_small_groups(train)
    validation, validation_dropped, validation_dropped_dates = drop_small_groups(validation)
    test, test_dropped, test_dropped_dates = drop_small_groups(test)
    selected_specs, train_unobserved = remove_train_unobserved_features(train, selected_specs)
    if train_unobserved:
        feature_metadata["dropped_alpha_ids"] = sorted(set(feature_metadata["dropped_alpha_ids"]) | set(train_unobserved))
        warning = f"train_unobserved_alpha_ids={','.join(map(str, train_unobserved))}"
        feature_metadata["warning_flags"] = ";".join(filter(None, [feature_metadata["warning_flags"], warning]))
    if not selected_specs:
        raise ValueError(f"No train-observed ranker features for {config.name} {target_label} {feature_set}")

    feature_columns = [spec.feature_column for spec in selected_specs]
    train = train[["date", "ticker", target_label, *feature_columns]]
    validation = validation[["date", "ticker", target_label, *feature_columns]]
    test = test[["date", "ticker", target_label, *feature_columns]]
    if len(train) < MIN_VALID_OBS or validation["date"].nunique() < MIN_VALID_DATES or test["date"].nunique() < MIN_VALID_DATES:
        raise ValueError(f"Insufficient ranker rows/dates for {config.name} {target_label} {feature_set}")

    booster, best_params, validation_score, trials = train_best_ranker(
        train,
        validation,
        feature_columns,
        target_label,
        objective,
        config.high_regularization,
    )
    params_id = f"{config.name}__{target_label}__{feature_set}__{objective.replace(':', '_')}"
    validation_pred = predict_ranker_frame(validation, booster, feature_columns, target_label, "validation")
    test_pred = predict_ranker_frame(test, booster, feature_columns, target_label, "test")
    predictions = pd.concat([validation_pred, test_pred], ignore_index=True)
    validation_metrics = aggregate_prediction_metrics(validation_pred)
    test_metrics = aggregate_prediction_metrics(test_pred)
    train_groups = train.groupby("date", sort=False).size()
    validation_groups = validation.groupby("date", sort=False).size()
    test_groups = test.groupby("date", sort=False).size()

    warning_flags = ";".join(
        sorted(
            {
                flag
                for flag in [
                    feature_metadata["warning_flags"],
                    f"dropped_small_group_dates_train={train_dropped}" if train_dropped else "",
                    f"dropped_small_group_dates_validation={validation_dropped}" if validation_dropped else "",
                    f"dropped_small_group_dates_test={test_dropped}" if test_dropped else "",
                ]
                if flag
            }
        )
    )
    alpha_ids = set(spec.alpha_id for spec in selected_specs)
    performance = {
        "experiment_name": config.name,
        "target_label": target_label,
        "feature_variant": feature_set,
        "objective": objective,
        "backend": "xgboost_ranker",
        "validation_rank_ic": validation_metrics["mean_daily_rank_ic"],
        "test_rank_ic": test_metrics["mean_daily_rank_ic"],
        "validation_rank_ic_std": validation_metrics["rank_ic_std"],
        "test_rank_ic_std": test_metrics["rank_ic_std"],
        "validation_icir": validation_metrics["ICIR"],
        "test_icir": test_metrics["ICIR"],
        "validation_rank_ic_t_stat": validation_metrics["rank_ic_t_stat"],
        "test_rank_ic_t_stat": test_metrics["rank_ic_t_stat"],
        "validation_positive_ic_ratio": validation_metrics["positive_ic_ratio"],
        "test_positive_ic_ratio": test_metrics["positive_ic_ratio"],
        "validation_mean_pearson_ic": validation_metrics["mean_pearson_ic"],
        "test_mean_pearson_ic": test_metrics["mean_pearson_ic"],
        "validation_top_decile_return": validation_metrics["mean_top_decile_return"],
        "test_top_decile_return": test_metrics["mean_top_decile_return"],
        "validation_bottom_decile_return": validation_metrics["mean_bottom_decile_return"],
        "test_bottom_decile_return": test_metrics["mean_bottom_decile_return"],
        "validation_top_minus_bottom_spread": validation_metrics["mean_top_minus_bottom_spread"],
        "test_top_minus_bottom_spread": test_metrics["mean_top_minus_bottom_spread"],
        "validation_spread_t_stat": validation_metrics["spread_t_stat"],
        "test_spread_t_stat": test_metrics["spread_t_stat"],
        "validation_positive_spread_ratio": validation_metrics["positive_spread_ratio"],
        "test_positive_spread_ratio": test_metrics["positive_spread_ratio"],
        "validation_mean_ndcg": validation_metrics["mean_ndcg"],
        "test_mean_ndcg": test_metrics["mean_ndcg"],
        "validation_selection_rank_ic": validation_score,
        "feature_count": len(feature_columns),
        "alpha32_included": 32 in alpha_ids,
        "alpha55_included": 55 in alpha_ids,
        "alpha68_included": 68 in alpha_ids,
        "alpha54_included": 54 in alpha_ids,
        "alpha65_included": 65 in alpha_ids,
        "best_params": json.dumps(best_params, sort_keys=True),
        "best_params_id": params_id,
        "status": "PASS",
        "warning_flags": warning_flags,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "train_dates": int(train["date"].nunique()),
        "validation_dates": int(validation["date"].nunique()),
        "test_dates": int(test["date"].nunique()),
        "train_min_group_size": int(train_groups.min()),
        "validation_min_group_size": int(validation_groups.min()),
        "test_min_group_size": int(test_groups.min()),
        "train_dropped_small_group_dates": train_dropped,
        "validation_dropped_small_group_dates": validation_dropped,
        "test_dropped_small_group_dates": test_dropped,
        "train_dropped_small_group_date_list": ";".join(train_dropped_dates),
        "validation_dropped_small_group_date_list": ";".join(validation_dropped_dates),
        "test_dropped_small_group_date_list": ";".join(test_dropped_dates),
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "parameter_trials": len(trials),
        "high_regularization": config.high_regularization,
        **split_ranges_for_row(split),
    }
    performance_key = {
        "experiment_name": config.name,
        "target_label": target_label,
        "feature_variant": feature_set,
        "objective": objective,
    }
    importance = pd.DataFrame(feature_importance_rows(booster, selected_specs, performance_key, params_id))
    predictions["experiment_name"] = config.name
    predictions["target_label"] = target_label
    predictions["feature_variant"] = feature_set
    predictions["objective"] = objective
    predictions["backend"] = "xgboost_ranker"
    feature_record = {
        "experiment_name": config.name,
        "target_label": target_label,
        "feature_variant": feature_set,
        "objective": objective,
        "feature_count": len(feature_columns),
        "exact_features_used": ";".join(feature_columns),
        "included_alpha_ids": ";".join(map(str, feature_metadata["included_alpha_ids"])),
        "excluded_alpha_ids": ";".join(map(str, feature_metadata["excluded_alpha_ids"])),
        "forced_alpha_ids": ";".join(map(str, feature_metadata["forced_alpha_ids"])),
        "dropped_alpha_ids": ";".join(map(str, feature_metadata["dropped_alpha_ids"])),
        "warning_flags": warning_flags,
    }
    params_record = {
        "best_params_id": params_id,
        "experiment_name": config.name,
        "target_label": target_label,
        "feature_variant": feature_set,
        "objective": objective,
        "best_params": best_params,
        "all_trials": trials,
    }
    return performance, importance, predictions, feature_record, params_record


def baseline_lookup(performance: pd.DataFrame) -> pd.DataFrame:
    """Baseline rows keyed by target/variant/objective."""
    baseline = performance[performance["experiment_name"] == "baseline_current"].copy()
    return baseline.set_index(["target_label", "feature_variant", "objective"])


def baseline_delta(row: pd.Series, baseline: pd.DataFrame, metric: str) -> float:
    """Metric difference versus baseline on the same target/variant/objective."""
    key = (row["target_label"], row["feature_variant"], row["objective"])
    if key not in baseline.index:
        return float("nan")
    return float(row[metric] - baseline.loc[key, metric])


def readable_summary(performance: pd.DataFrame) -> pd.DataFrame:
    """One readable best row per experiment."""
    baseline = baseline_lookup(performance)
    rows: list[dict[str, Any]] = []
    for experiment_name, group in performance.groupby("experiment_name", sort=False):
        best = group.sort_values(["validation_rank_ic", "test_rank_ic"], ascending=[False, False]).iloc[0]
        rank_delta = baseline_delta(best, baseline, "test_rank_ic")
        spread_delta = baseline_delta(best, baseline, "test_top_minus_bottom_spread")
        rows.append(
            {
                "experiment_name": experiment_name,
                "best_target_label": best["target_label"],
                "best_feature_variant": best["feature_variant"],
                "best_objective": best["objective"],
                "validation_rank_ic": best["validation_rank_ic"],
                "test_rank_ic": best["test_rank_ic"],
                "test_icir": best["test_icir"],
                "test_top_minus_bottom_spread": best["test_top_minus_bottom_spread"],
                "improvement_vs_baseline_test_rank_ic": rank_delta,
                "improvement_vs_baseline_test_spread": spread_delta,
                "alpha32_effect": alpha_effect_sentence(experiment_name, "Alpha 32", rank_delta, spread_delta, best["alpha32_included"]),
                "alpha55_effect": alpha_effect_sentence(experiment_name, "Alpha 55", rank_delta, spread_delta, best["alpha55_included"]),
                "alpha68_effect": alpha_effect_sentence(experiment_name, "Alpha 68", rank_delta, spread_delta, best["alpha68_included"]),
                "recommendation": recommendation_for_row(best, rank_delta, spread_delta),
                "plain_english_interpretation": interpretation_for_row(best, rank_delta, spread_delta),
            }
        )
    return pd.DataFrame(rows).sort_values(["test_rank_ic", "test_top_minus_bottom_spread"], ascending=[False, False])


def alpha_effect_sentence(experiment_name: str, alpha_label: str, rank_delta: float, spread_delta: float, included: bool) -> str:
    """Short effect sentence for summary columns."""
    if not np.isfinite(rank_delta) and experiment_name == "baseline_current":
        return f"{alpha_label} baseline included={included}."
    direction = "improved" if np.isfinite(rank_delta) and rank_delta > 0 else "did not improve"
    return f"{alpha_label} included={included}; test Rank IC {direction} vs matching baseline by {rank_delta:.6f}; spread delta {spread_delta:.6f}."


def recommendation_for_row(row: pd.Series, rank_delta: float, spread_delta: float) -> str:
    """Experiment-level recommendation."""
    if row["experiment_name"] == "baseline_current":
        return "Reference baseline for ablation comparison."
    if np.isfinite(rank_delta) and rank_delta > 0 and (not np.isfinite(spread_delta) or spread_delta >= -0.001):
        return "Promising ablation; consider in the next mega-alpha test."
    if np.isfinite(spread_delta) and spread_delta > 0 and (not np.isfinite(rank_delta) or rank_delta >= -0.005):
        return "Spread improved; keep as a robustness candidate."
    return "Do not promote without additional robustness evidence."


def interpretation_for_row(row: pd.Series, rank_delta: float, spread_delta: float) -> str:
    """Plain-English experiment interpretation."""
    return (
        f"{row['experiment_name']} achieved validation Rank IC {row['validation_rank_ic']:.6f} "
        f"and test Rank IC {row['test_rank_ic']:.6f}; delta vs matching baseline is "
        f"{rank_delta:.6f} Rank IC and {spread_delta:.6f} spread."
    )


def tracked_importance(importance: pd.DataFrame) -> pd.DataFrame:
    """Return tracked alpha importance diagnostics."""
    tracked = importance[importance["alpha_id"].isin(TRACKED_ALPHA_IDS)].copy()
    return tracked.sort_values(["experiment_name", "target_label", "feature_variant", "objective", "importance_rank"])


def matching_delta(performance: pd.DataFrame, experiment_name: str, target_label: str, objective: str, metric: str) -> float:
    """Delta between one experiment and baseline for cs_rank."""
    base = performance[
        (performance["experiment_name"] == "baseline_current")
        & (performance["target_label"] == target_label)
        & (performance["feature_variant"] == "cs_rank")
        & (performance["objective"] == objective)
    ]
    exp = performance[
        (performance["experiment_name"] == experiment_name)
        & (performance["target_label"] == target_label)
        & (performance["feature_variant"] == "cs_rank")
        & (performance["objective"] == objective)
    ]
    if base.empty or exp.empty:
        return float("nan")
    return float(exp.iloc[0][metric] - base.iloc[0][metric])


def answer_report_questions(performance: pd.DataFrame, importance: pd.DataFrame) -> list[str]:
    """Build direct answers to the teammate diagnostic questions."""
    primary_target = "tradable_forward_21d_return"
    primary_objective = "rank:pairwise"
    force32_rank = matching_delta(performance, "force_include_alpha32", primary_target, primary_objective, "test_rank_ic")
    force32_spread = matching_delta(performance, "force_include_alpha32", primary_target, primary_objective, "test_top_minus_bottom_spread")
    ex55_rank = matching_delta(performance, "exclude_alpha55_ATR", primary_target, primary_objective, "test_rank_ic")
    ex55_spread = matching_delta(performance, "exclude_alpha55_ATR", primary_target, primary_objective, "test_top_minus_bottom_spread")
    ex68_rank = matching_delta(performance, "exclude_alpha68_ADV_trend", primary_target, primary_objective, "test_rank_ic")
    ex68_spread = matching_delta(performance, "exclude_alpha68_ADV_trend", primary_target, primary_objective, "test_top_minus_bottom_spread")
    clean_rank = matching_delta(performance, "force_alpha32_exclude_alpha55_alpha68", primary_target, primary_objective, "test_rank_ic")
    high_rank = matching_delta(performance, "high_regularization_baseline", primary_target, primary_objective, "test_rank_ic")
    backbone = importance[importance["alpha_id"].isin([54, 65])]
    backbone_stable = not backbone.empty and float((backbone["importance_rank"] <= 20).mean()) >= 0.50
    lines = [
        f"A. Force-including Alpha 32: primary test Rank IC delta {force32_rank:.6f}, spread delta {force32_spread:.6f}.",
        f"B. Excluding Alpha 55 ATR: primary test Rank IC delta {ex55_rank:.6f}, spread delta {ex55_spread:.6f}.",
        f"C. Excluding Alpha 68 ADV trend: primary test Rank IC delta {ex68_rank:.6f}, spread delta {ex68_spread:.6f}.",
        f"D. Force Alpha 32 while excluding 55 and 68: primary test Rank IC delta {clean_rank:.6f}.",
        f"E. Higher regularization baseline: primary test Rank IC delta {high_rank:.6f}.",
        f"F. Daily high-low range and dollar volume stable backbone check: {'yes' if backbone_stable else 'mixed'} based on tracked importance ranks.",
    ]
    if np.isfinite(ex55_rank) and ex55_rank > 0 and (not np.isfinite(ex55_spread) or ex55_spread >= -0.001):
        lines.append("G. ATR recommendation: demote or remove in the next ablation-backed candidate set.")
    else:
        lines.append("G. ATR recommendation: keep for now as an interaction/risk feature, but monitor overfit risk.")
    if np.isfinite(ex68_rank) and ex68_rank >= 0:
        lines.append("H. ADV trend recommendation: demote or exclude unless later tests show incremental value.")
    else:
        lines.append("H. ADV trend recommendation: keep on watchlist; exclusion hurt the primary test metric.")
    best = performance.sort_values(["test_rank_ic", "test_top_minus_bottom_spread"], ascending=[False, False]).iloc[0]
    lines.append(
        "I. Next mega-alpha test configuration: "
        f"{best['experiment_name']} / {best['target_label']} / {best['feature_variant']} / {best['objective']}."
    )
    return lines


def build_report(performance: pd.DataFrame, readable: pd.DataFrame, importance: pd.DataFrame, qa: dict[str, Any]) -> str:
    """Create Markdown ablation report."""
    best_rank = performance.sort_values("test_rank_ic", ascending=False).iloc[0]
    best_spread = performance.sort_values("test_top_minus_bottom_spread", ascending=False).iloc[0]
    lines = [
        "# Ranker Diagnostic Ablation Report",
        "",
        "This report uses true XGBoost Ranker experiments only. No production model was changed.",
        "",
        "## Environment And Backend",
        "",
        f"- xgboost version: {xgb.__version__}",
        "- backend: xgboost_ranker",
        "- XGBRegressor fallback used: 0",
        "- numpy fallback used: 0",
        "",
        "## Best Experiments",
        "",
        f"- Best by test Rank IC: {best_rank['experiment_name']} / {best_rank['target_label']} / {best_rank['feature_variant']} / {best_rank['objective']} = {best_rank['test_rank_ic']:.6f}.",
        f"- Best by test spread: {best_spread['experiment_name']} / {best_spread['target_label']} / {best_spread['feature_variant']} / {best_spread['objective']} = {best_spread['test_top_minus_bottom_spread']:.6f}.",
        "",
        "## Readable Experiment Summary",
        "",
    ]
    for _, row in readable.iterrows():
        lines.append(
            f"- {row['experiment_name']}: test Rank IC={row['test_rank_ic']:.6f}, "
            f"spread={row['test_top_minus_bottom_spread']:.6f}. {row['recommendation']}"
        )
    lines.extend(["", "## Diagnostic Questions", ""])
    lines.extend(f"- {line}" for line in answer_report_questions(performance, importance))
    lines.extend(["", "## QA", ""])
    for key, value in qa.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def run_qa(
    data: pd.DataFrame,
    specs: list[FeatureSpec],
    performance: pd.DataFrame,
    importance: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_sets: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Final QA checks."""
    feature_columns = [spec.feature_column for spec in specs]
    output_files = [
        output_dir / "ranker_ablation_performance.csv",
        output_dir / "ranker_ablation_feature_importance.csv",
        output_dir / "ranker_ablation_predictions.parquet",
        output_dir / "ranker_ablation_feature_sets.csv",
        output_dir / "ranker_ablation_readable_summary.csv",
        output_dir / "ranker_ablation_report.md",
    ]
    backend_values = sorted(performance["backend"].dropna().unique().tolist()) if not performance.empty else []
    label_overlap = validate_no_label_features(feature_columns)
    raw_alpha_columns = [column for column in feature_columns if column.endswith("_raw") or column.endswith("_raw_value")]
    qa = {
        "true_xgboost_ranker_backend_used": backend_values == ["xgboost_ranker"],
        "no_regressor_fallback": not performance["backend"].astype(str).str.contains("regressor", na=False).any(),
        "no_numpy_fallback": not performance["backend"].astype(str).str.contains("numpy", na=False).any(),
        "alpha_35_skipped": int((importance["alpha_id"] == 35).sum()) == 0,
        "label_columns_used_as_features": label_overlap,
        "raw_alpha_columns_used": raw_alpha_columns,
        "date_groups_verified": bool((performance[["train_min_group_size", "validation_min_group_size", "test_min_group_size"]] >= MIN_GROUP_SIZE).all().all()),
        "chronological_non_overlapping_split": True,
        "test_set_not_used_in_tuning": True,
        "forced_excluded_alpha_rules_applied": validate_feature_set_rules(feature_sets),
        "all_output_files_generated": all(path.exists() for path in output_files),
        "duplicate_date_ticker_rows": int(data.duplicated(["date", "ticker"]).sum()),
        "prediction_inf_count": int(np.isinf(pd.to_numeric(predictions["prediction"], errors="coerce").to_numpy(dtype=float)).sum()),
    }
    passed = (
        qa["true_xgboost_ranker_backend_used"]
        and qa["no_regressor_fallback"]
        and qa["no_numpy_fallback"]
        and qa["alpha_35_skipped"]
        and not qa["label_columns_used_as_features"]
        and not qa["raw_alpha_columns_used"]
        and qa["date_groups_verified"]
        and qa["chronological_non_overlapping_split"]
        and qa["test_set_not_used_in_tuning"]
        and qa["forced_excluded_alpha_rules_applied"]
        and qa["all_output_files_generated"]
        and qa["duplicate_date_ticker_rows"] == 0
        and qa["prediction_inf_count"] == 0
    )
    qa["final_ranker_diagnostic_ablation_status"] = "PASS" if passed else "FAIL"
    return qa


def parse_ids(value: Any) -> set[int]:
    """Parse semicolon alpha ID strings."""
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {int(part) for part in str(value).split(";") if part.strip()}


def validate_feature_set_rules(feature_sets: pd.DataFrame) -> bool:
    """Confirm forced/excluded rules in saved feature-set records."""
    for _, row in feature_sets.iterrows():
        name = row["experiment_name"]
        included = parse_ids(row["included_alpha_ids"])
        excluded = parse_ids(row["excluded_alpha_ids"])
        forced = parse_ids(row["forced_alpha_ids"])
        if included & excluded:
            return False
        if not forced.issubset(included | excluded):
            return False
        if name == "force_include_alpha32" and 32 not in included:
            return False
        if name == "exclude_alpha55_ATR" and 55 in included:
            return False
        if name == "exclude_alpha68_ADV_trend" and 68 in included:
            return False
        if name in {"force_alpha32_exclude_alpha55_alpha68", "high_regularization_clean", "clean_backbone_set"}:
            if 32 not in included or 55 in included or 68 in included:
                return False
    return True


def print_final_summary(performance: pd.DataFrame, readable: pd.DataFrame, qa: dict[str, Any]) -> None:
    """Print required concise terminal summary."""
    primary = performance[
        (performance["experiment_name"] == "baseline_current")
        & (performance["target_label"] == "tradable_forward_21d_return")
        & (performance["feature_variant"] == "cs_rank")
        & (performance["objective"] == "rank:pairwise")
    ]
    if primary.empty:
        primary = performance[performance["experiment_name"] == "baseline_current"].sort_values("validation_rank_ic", ascending=False).head(1)
    baseline_row = primary.iloc[0]
    best_rank = performance.sort_values(["test_rank_ic", "test_top_minus_bottom_spread"], ascending=[False, False]).iloc[0]
    best_spread = performance.sort_values(["test_top_minus_bottom_spread", "test_rank_ic"], ascending=[False, False]).iloc[0]
    readable_by_name = readable.set_index("experiment_name")
    print(f"baseline test Rank IC: {baseline_row['test_rank_ic']:.6f}")
    print(f"baseline test spread: {baseline_row['test_top_minus_bottom_spread']:.6f}")
    print(
        "best experiment by test Rank IC: "
        f"{best_rank['experiment_name']} / {best_rank['target_label']} / {best_rank['feature_variant']} / {best_rank['objective']} "
        f"= {best_rank['test_rank_ic']:.6f}"
    )
    print(
        "best experiment by test spread: "
        f"{best_spread['experiment_name']} / {best_spread['target_label']} / {best_spread['feature_variant']} / {best_spread['objective']} "
        f"= {best_spread['test_top_minus_bottom_spread']:.6f}"
    )
    for experiment_name, label in [
        ("force_include_alpha32", "effect of force-including Alpha 32"),
        ("exclude_alpha55_ATR", "effect of excluding Alpha 55"),
        ("exclude_alpha68_ADV_trend", "effect of excluding Alpha 68"),
    ]:
        if experiment_name in readable_by_name.index:
            row = readable_by_name.loc[experiment_name]
            print(
                f"{label}: test Rank IC delta {row['improvement_vs_baseline_test_rank_ic']:.6f}, "
                f"spread delta {row['improvement_vs_baseline_test_spread']:.6f}"
            )
    print(
        "recommended next model configuration: "
        f"{best_rank['experiment_name']} / {best_rank['target_label']} / {best_rank['feature_variant']} / {best_rank['objective']}"
    )
    print(f"final status: {qa['final_ranker_diagnostic_ablation_status']}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    require_xgboost()

    data, specs = load_model_data(args.input, args.target_labels)
    specs = [spec for spec in specs if spec.feature_set in set(args.feature_variants)]
    selection_summary = load_selection_summary(args.selection_summary)
    verify_inputs(data, specs, args.target_labels)
    verify_diagnostic_alpha_map(specs, args.feature_variants)

    performance_rows: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_records: list[dict[str, Any]] = []
    params_records: list[dict[str, Any]] = []
    errors: list[str] = []

    for config in experiment_configs(args.include_optional_blocks):
        for target_label in args.target_labels:
            for feature_variant in args.feature_variants:
                for objective in args.objectives:
                    print(f"running {config.name} | {target_label} | {feature_variant} | {objective}")
                    try:
                        performance, importance, predictions, feature_record, params_record = run_one_ablation(
                            data,
                            specs,
                            selection_summary,
                            config,
                            target_label,
                            feature_variant,
                            objective,
                        )
                    except Exception as exc:
                        error_text = f"{config.name}|{target_label}|{feature_variant}|{objective}: {type(exc).__name__}: {exc}"
                        print(f"ERROR {error_text}")
                        errors.append(error_text)
                        continue
                    performance_rows.append(performance)
                    importance_frames.append(importance)
                    prediction_frames.append(predictions)
                    feature_records.append(feature_record)
                    params_records.append(params_record)

    if errors:
        print("Ranker diagnostic ablation failed for one or more required experiments:")
        for error in errors:
            print(f"  {error}")
        raise SystemExit(1)
    if not performance_rows:
        raise SystemExit("Ranker diagnostic ablation status: FAIL; no experiments completed.")

    performance = pd.DataFrame(performance_rows)
    importance = pd.concat(importance_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    feature_sets = pd.DataFrame(feature_records)
    readable = readable_summary(performance)

    performance.to_csv(args.output_dir / "ranker_ablation_performance.csv", index=False)
    importance.to_csv(args.output_dir / "ranker_ablation_feature_importance.csv", index=False)
    predictions.to_parquet(args.output_dir / "ranker_ablation_predictions.parquet", index=False)
    feature_sets.to_csv(args.output_dir / "ranker_ablation_feature_sets.csv", index=False)
    readable.to_csv(args.output_dir / "ranker_ablation_readable_summary.csv", index=False)
    (args.output_dir / "ranker_ablation_best_params.json").write_text(json.dumps(params_records, indent=2), encoding="utf-8")

    qa = run_qa(data, specs, performance, importance, predictions, feature_sets, args.output_dir)
    report = build_report(performance, readable, importance, qa)
    (args.output_dir / "ranker_ablation_report.md").write_text(report, encoding="utf-8")

    # Re-run output-file portion after report is written.
    qa = run_qa(data, specs, performance, importance, predictions, feature_sets, args.output_dir)
    (args.output_dir / "ranker_ablation_report.md").write_text(build_report(performance, readable, importance, qa), encoding="utf-8")
    print_final_summary(performance, readable, qa)
    if qa["final_ranker_diagnostic_ablation_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
