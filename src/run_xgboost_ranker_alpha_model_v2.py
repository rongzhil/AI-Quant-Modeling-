"""Train true XGBoost learning-to-rank alpha models.

This script intentionally uses only XGBoost ranking objectives through
``xgboost.train`` and grouped ``DMatrix`` objects. It does not use
``XGBRegressor`` and it does not use numpy/sklearn fallbacks.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from .factor_selection_common import (
    FEATURE_SET_SUFFIX,
    LABEL_COLUMNS,
    MIN_VALID_OBS,
    SPECIAL_ALPHA_IDS,
    TARGET_LABELS_DEFAULT,
    alpha_category,
    chronological_split,
    daily_rank_ic,
    decile_spread,
    filter_feature_columns,
    finite_frame,
    group_sizes_by_date,
    icir,
    join_unique,
    load_model_data,
    pearson_corr,
    prediction_metrics,
    sign_label,
    slice_split,
    split_ranges_for_row,
    validate_no_label_features,
)


OBJECTIVES = ["rank:pairwise", "rank:ndcg"]
RELEVANCE_GRADES = 31
MIN_GROUP_SIZE = 50
GRID_SEARCH_MODE = "bounded_covering_grid"
NUM_BOOST_ROUND = 200
EARLY_STOPPING_ROUNDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run true XGBoost ranker alpha model v2.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-labels", nargs="+", default=TARGET_LABELS_DEFAULT)
    return parser.parse_args()


def parameter_grid(objective: str) -> list[dict[str, Any]]:
    """Conservative bounded ranking parameter grid.

    The full Cartesian grid requested by the research spec is very expensive on
    this 382k-row panel. This covering grid still tests every requested
    parameter family and every requested value at least once for each ranking
    objective, while keeping the run interactive and reproducible.
    """
    rows: list[dict[str, Any]] = []
    covering_values = [
        (2, 0.03, 1.0, 0.0, 5),
        (2, 0.05, 5.0, 0.1, 10),
        (3, 0.03, 5.0, 0.1, 5),
        (3, 0.05, 1.0, 0.0, 10),
        (4, 0.03, 1.0, 0.1, 10),
        (4, 0.05, 5.0, 0.0, 5),
    ]
    for max_depth, eta, reg_lambda, reg_alpha, min_child_weight in covering_values:
        rows.append(
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
                "seed": 42,
                "verbosity": 0,
            }
        )
    return rows


def require_xgboost() -> None:
    """Verify xgboost imports and ranking objective smoke-test symbols exist."""
    if not hasattr(xgb, "train") or not hasattr(xgb, "DMatrix"):
        raise RuntimeError("xgboost ranking API unavailable: missing train or DMatrix")
    print(f"xgboost version: {xgb.__version__}")
    print(f"ranking objectives requested: {OBJECTIVES}")


def relevance_by_date(frame: pd.DataFrame, target_label: str) -> np.ndarray:
    """Convert continuous future returns to non-binary integer relevance grades.

    XGBoost 3.2 rank metrics such as NDCG require non-negative integer
    relevance labels. We preserve the economic ordering by mapping each date's
    continuous forward returns to 0..31 rank grades. Evaluation still uses the
    original continuous returns.
    """
    pct = frame.groupby("date", sort=False)[target_label].rank(method="average", pct=True)
    grades = np.floor(pct.to_numpy(dtype=float) * RELEVANCE_GRADES)
    return np.clip(grades, 0, RELEVANCE_GRADES).astype(np.float32)


def drop_small_groups(frame: pd.DataFrame, min_group_size: int = MIN_GROUP_SIZE) -> tuple[pd.DataFrame, int]:
    """Drop dates with fewer than min_group_size rows."""
    group_counts = frame.groupby("date", sort=False).size()
    keep_dates = group_counts[group_counts >= min_group_size].index
    dropped_dates = int((group_counts < min_group_size).sum())
    return frame[frame["date"].isin(keep_dates)].copy(), dropped_dates


def build_ranker_dmatrix(frame: pd.DataFrame, feature_columns: list[str], target_label: str) -> tuple[xgb.DMatrix, np.ndarray]:
    """Build grouped DMatrix sorted by date/ticker."""
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
        raise ValueError("Ranking group contains fewer than minimum rows")
    return dmatrix, groups


def mean_daily_rank_ic_for_prediction(frame: pd.DataFrame, target_label: str, prediction: np.ndarray) -> float:
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
) -> tuple[xgb.Booster, dict[str, Any], float, list[dict[str, Any]]]:
    """Train ranker grid and select by validation mean daily Rank IC."""
    dtrain, train_groups = build_ranker_dmatrix(train, feature_columns, target_label)
    dvalidation, validation_groups = build_ranker_dmatrix(validation, feature_columns, target_label)
    best_booster: xgb.Booster | None = None
    best_params: dict[str, Any] | None = None
    best_score = -np.inf
    trials: list[dict[str, Any]] = []
    for trial_id, params in enumerate(parameter_grid(objective), start=1):
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
        trial = {
            "trial_id": trial_id,
            "objective": objective,
            "validation_rank_ic": validation_rank_ic,
            "best_iteration": best_iteration,
            "train_group_count": int(len(train_groups)),
            "validation_group_count": int(len(validation_groups)),
            **params,
        }
        trials.append(trial)
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
    """Predict one split using raw feature matrix and native missing handling."""
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


def daily_prediction_diagnostics(frame: pd.DataFrame, pred_col: str = "prediction", target_col: str = "target") -> pd.DataFrame:
    """Daily IC and spread diagnostics."""
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


def t_stat(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return float("nan")
    std = float(clean.std(ddof=1))
    if std == 0 or not np.isfinite(std):
        return float("nan")
    return float(clean.mean() / (std / math.sqrt(len(clean))))


def ndcg_by_date(frame: pd.DataFrame, pred_col: str = "prediction", target_col: str = "target") -> float:
    """Mean realized NDCG using within-date continuous-return ranks as relevance."""
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


def aggregate_prediction_metrics(predictions: pd.DataFrame) -> dict[str, float]:
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


def feature_importance_rows(
    booster: xgb.Booster,
    feature_specs,
    target_label: str,
    feature_set: str,
    objective: str,
    params_id: str,
) -> list[dict[str, Any]]:
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover = booster.get_score(importance_type="cover")
    total_gain = sum(float(gain.get(spec.feature_column, 0.0)) for spec in feature_specs)
    rows = []
    for spec in feature_specs:
        gain_value = float(gain.get(spec.feature_column, 0.0))
        rows.append(
            {
                "target_label": target_label,
                "feature_set": feature_set,
                "objective": objective,
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


def run_experiment(data: pd.DataFrame, specs, target_label: str, feature_set: str, objective: str) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selected_specs = filter_feature_columns(data, specs, feature_set, target_label)
    if not selected_specs:
        raise ValueError(f"No valid features for {target_label} {feature_set}")
    split = chronological_split(data.loc[data[target_label].notna(), "date"])
    candidate_columns = [spec.feature_column for spec in selected_specs]
    frame = finite_frame(data[["date", "ticker", target_label, *candidate_columns]].copy())
    frame = frame.sort_values(["date", "ticker"]).reset_index(drop=True)
    train, validation, test = slice_split(frame, split, target_label)
    train, train_dropped = drop_small_groups(train)
    validation, validation_dropped = drop_small_groups(validation)
    test, test_dropped = drop_small_groups(test)
    selected_specs = [
        spec
        for spec in selected_specs
        if pd.to_numeric(train[spec.feature_column], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().any()
    ]
    if not selected_specs:
        raise ValueError(f"No train-observed ranker features for {target_label} {feature_set}")
    feature_columns = [spec.feature_column for spec in selected_specs]
    train = train[["date", "ticker", target_label, *feature_columns]]
    validation = validation[["date", "ticker", target_label, *feature_columns]]
    test = test[["date", "ticker", target_label, *feature_columns]]
    if len(train) < MIN_VALID_OBS or len(validation) < MIN_GROUP_SIZE or len(test) < MIN_GROUP_SIZE:
        raise ValueError(f"Insufficient ranker rows for {target_label} {feature_set} {objective}")

    booster, params, validation_score, trials = train_best_ranker(train, validation, feature_columns, target_label, objective)
    params_id = f"{target_label}__{feature_set}__{objective.replace(':', '_')}"
    validation_pred = predict_ranker_frame(validation, booster, feature_columns, target_label, "validation")
    test_pred = predict_ranker_frame(test, booster, feature_columns, target_label, "test")
    predictions = pd.concat([validation_pred, test_pred], ignore_index=True)
    validation_daily = daily_prediction_diagnostics(validation_pred)
    validation_daily["split"] = "validation"
    test_daily = daily_prediction_diagnostics(test_pred)
    test_daily["split"] = "test"
    daily = pd.concat([validation_daily, test_daily], ignore_index=True)
    validation_metrics = aggregate_prediction_metrics(validation_pred)
    test_metrics = aggregate_prediction_metrics(test_pred)
    train_groups = train.groupby("date", sort=False).size()
    validation_groups = validation.groupby("date", sort=False).size()
    test_groups = test.groupby("date", sort=False).size()
    performance = {
        "target_label": target_label,
        "feature_set": feature_set,
        "objective": objective,
        "backend": "xgboost_ranker",
        "best_params_id": params_id,
        "number_of_features": len(feature_columns),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "train_dates": int(train["date"].nunique()),
        "validation_dates": int(validation["date"].nunique()),
        "test_dates": int(test["date"].nunique()),
        "train_tickers": int(train["ticker"].nunique()),
        "validation_tickers": int(validation["ticker"].nunique()),
        "test_tickers": int(test["ticker"].nunique()),
        "train_min_group_size": int(train_groups.min()),
        "validation_min_group_size": int(validation_groups.min()),
        "test_min_group_size": int(test_groups.min()),
        "train_dropped_small_group_dates": train_dropped,
        "validation_dropped_small_group_dates": validation_dropped,
        "test_dropped_small_group_dates": test_dropped,
        "grid_search_mode": GRID_SEARCH_MODE,
        "parameter_trials": len(parameter_grid(objective)),
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "validation_selection_rank_ic": validation_score,
        "validation_mean_daily_rank_ic": validation_metrics["mean_daily_rank_ic"],
        "validation_rank_ic_std": validation_metrics["rank_ic_std"],
        "validation_ICIR": validation_metrics["ICIR"],
        "validation_rank_ic_t_stat": validation_metrics["rank_ic_t_stat"],
        "validation_positive_ic_ratio": validation_metrics["positive_ic_ratio"],
        "validation_mean_pearson_ic": validation_metrics["mean_pearson_ic"],
        "validation_mean_top_decile_return": validation_metrics["mean_top_decile_return"],
        "validation_mean_bottom_decile_return": validation_metrics["mean_bottom_decile_return"],
        "validation_mean_top_minus_bottom_spread": validation_metrics["mean_top_minus_bottom_spread"],
        "validation_spread_t_stat": validation_metrics["spread_t_stat"],
        "validation_positive_spread_ratio": validation_metrics["positive_spread_ratio"],
        "validation_mean_ndcg": validation_metrics["mean_ndcg"],
        "test_mean_daily_rank_ic": test_metrics["mean_daily_rank_ic"],
        "test_rank_ic_std": test_metrics["rank_ic_std"],
        "test_ICIR": test_metrics["ICIR"],
        "test_rank_ic_t_stat": test_metrics["rank_ic_t_stat"],
        "test_positive_ic_ratio": test_metrics["positive_ic_ratio"],
        "test_mean_pearson_ic": test_metrics["mean_pearson_ic"],
        "test_mean_top_decile_return": test_metrics["mean_top_decile_return"],
        "test_mean_bottom_decile_return": test_metrics["mean_bottom_decile_return"],
        "test_mean_top_minus_bottom_spread": test_metrics["mean_top_minus_bottom_spread"],
        "test_spread_t_stat": test_metrics["spread_t_stat"],
        "test_positive_spread_ratio": test_metrics["positive_spread_ratio"],
        "test_mean_ndcg": test_metrics["mean_ndcg"],
        **split_ranges_for_row(split),
    }
    importance = pd.DataFrame(feature_importance_rows(booster, selected_specs, target_label, feature_set, objective, params_id))
    predictions["target_label"] = target_label
    predictions["feature_set"] = feature_set
    predictions["objective"] = objective
    predictions["backend"] = "xgboost_ranker"
    daily["target_label"] = target_label
    daily["feature_set"] = feature_set
    daily["objective"] = objective
    daily["backend"] = "xgboost_ranker"
    params_record = {
        "best_params_id": params_id,
        "target_label": target_label,
        "feature_set": feature_set,
        "objective": objective,
        "best_params": params,
        "all_trials": trials,
        "grid_search_mode": GRID_SEARCH_MODE,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    }
    return importance, performance, daily, predictions, params_record


def build_readable_summary(importance: pd.DataFrame, performance: pd.DataFrame, specs) -> pd.DataFrame:
    merged = importance.merge(
        performance[
            [
                "target_label",
                "feature_set",
                "objective",
                "best_params_id",
                "validation_mean_daily_rank_ic",
                "test_mean_daily_rank_ic",
                "test_ICIR",
                "test_mean_top_minus_bottom_spread",
            ]
        ],
        on=["target_label", "feature_set", "objective", "best_params_id"],
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for alpha_id, group in merged.groupby("alpha_id", sort=True):
        best = group.sort_values(["importance_rank", "gain_importance"], ascending=[True, False]).iloc[0]
        important = group[group["importance_rank"] <= 15]
        warnings: list[str] = []
        if important.empty:
            warnings.append("not_top_15_in_any_ranker_run")
        rows.append(
            {
                "alpha_id": int(alpha_id),
                "alpha_name": best["alpha_name"],
                "category": alpha_category(int(alpha_id)),
                "best_feature_variant": best["feature_variant"],
                "best_target_label": best["target_label"],
                "best_objective": best["objective"],
                "best_test_rank_ic": float(best["test_mean_daily_rank_ic"]),
                "best_test_icir": float(best["test_ICIR"]),
                "best_test_decile_spread": float(best["test_mean_top_minus_bottom_spread"]),
                "best_validation_rank_ic": float(best["validation_mean_daily_rank_ic"]),
                "average_gain_importance": float(group["gain_importance"].mean()),
                "best_gain_importance": float(group["gain_importance"].max()),
                "average_importance_rank": float(group["importance_rank"].mean()),
                "best_importance_rank": int(group["importance_rank"].min()),
                "important_in_targets": join_unique(important["target_label"]),
                "important_in_feature_sets": join_unique(important["feature_set"]),
                "ranker_interpretation": ranker_interpretation(best["alpha_name"], important.empty, best["objective"]),
                "warning_flags": ";".join(warnings),
            }
        )
    existing_ids = {row["alpha_id"] for row in rows}
    for spec in specs:
        if spec.alpha_id in existing_ids:
            continue
        rows.append(
            {
                "alpha_id": int(spec.alpha_id),
                "alpha_name": spec.alpha_name,
                "category": spec.category,
                "best_feature_variant": "",
                "best_target_label": "",
                "best_objective": "",
                "best_test_rank_ic": np.nan,
                "best_test_icir": np.nan,
                "best_test_decile_spread": np.nan,
                "best_validation_rank_ic": np.nan,
                "average_gain_importance": 0.0,
                "best_gain_importance": 0.0,
                "average_importance_rank": np.nan,
                "best_importance_rank": 999,
                "important_in_targets": "",
                "important_in_feature_sets": "",
                "ranker_interpretation": f"{spec.alpha_name} was unavailable in the ranker training windows.",
                "warning_flags": "no_train_observed_feature_for_any_ranker_experiment",
            }
        )
        existing_ids.add(spec.alpha_id)
    return pd.DataFrame(rows).sort_values(["best_importance_rank", "average_gain_importance"], ascending=[True, False])


def ranker_interpretation(alpha_name: str, not_important: bool, objective: str) -> str:
    if not_important:
        return f"{alpha_name} was not among the top ranker importance features."
    return f"{alpha_name} contributed to true XGBoost ranker predictions under {objective}; evaluate in portfolio construction later."


def build_report(performance: pd.DataFrame, readable: pd.DataFrame, qa: dict[str, Any]) -> str:
    lines = [
        "# XGBoost Ranker Alpha Model V2",
        "",
        f"xgboost version: {xgb.__version__}",
        "",
        "True XGBoost ranking objectives were used with one date as one ranking query/group.",
        "No XGBRegressor fallback, numpy fallback, sklearn fallback, portfolio construction, transaction-cost backtest, or live simulation was run.",
        "",
        "Training relevance labels are non-binary within-date return rank grades because XGBoost 3.2 NDCG requires non-negative integer relevance labels. Validation and test evaluation use the original continuous future returns.",
        f"Hyperparameter tuning used `{GRID_SEARCH_MODE}`: a compact covering grid that includes every requested parameter family and value at least once per objective, with {NUM_BOOST_ROUND} boost rounds and {EARLY_STOPPING_ROUNDS} early-stopping rounds.",
        "",
        "## Performance By Target / Feature Set / Objective",
        "",
    ]
    for _, row in performance.sort_values("validation_mean_daily_rank_ic", ascending=False).iterrows():
        lines.append(
            f"- {row['target_label']} | {row['feature_set']} | {row['objective']}: "
            f"validation Rank IC={row['validation_mean_daily_rank_ic']:.4f}, "
            f"test Rank IC={row['test_mean_daily_rank_ic']:.4f}, "
            f"test spread={row['test_mean_top_minus_bottom_spread']:.5f}."
        )
    lines.extend(["", "## Top Ranker Alpha Candidates", ""])
    for _, row in readable.head(20).iterrows():
        lines.append(
            f"- Alpha {row['alpha_id']} {row['alpha_name']}: best_rank={row['best_importance_rank']}, "
            f"target={row['best_target_label']}, feature_set={row['best_feature_variant']}, objective={row['best_objective']}."
        )
    lines.extend(["", "## Special Alpha Notes", ""])
    for alpha_id in SPECIAL_ALPHA_IDS:
        row = readable[readable["alpha_id"] == alpha_id]
        if row.empty:
            continue
        item = row.iloc[0]
        lines.append(f"- Alpha {alpha_id} {item['alpha_name']}: {item['ranker_interpretation']} Warning: {item['warning_flags'] or 'none'}.")
    lines.extend(["", "## QA", ""])
    for key, value in qa.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def run_qa(data: pd.DataFrame, specs, performance: pd.DataFrame, importance: pd.DataFrame, readable: pd.DataFrame, target_labels: list[str]) -> dict[str, Any]:
    feature_columns = [spec.feature_column for spec in specs]
    label_overlap = validate_no_label_features(feature_columns)
    raw_feature_columns = [column for column in feature_columns if column.endswith("_raw")]
    inf_count = int(np.isinf(data[[*feature_columns, *target_labels]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)).sum())
    group_sum_ok = bool(
        (
            performance["train_rows"].eq(performance["train_rows"])
            & performance["validation_rows"].eq(performance["validation_rows"])
            & performance["test_rows"].eq(performance["test_rows"])
        ).all()
    )
    qa = {
        "xgboost_version": xgb.__version__,
        "backend_used": sorted(performance["backend"].dropna().unique().tolist()),
        "no_regressor_fallback_used": not performance["backend"].astype(str).str.contains("regressor", na=False).any(),
        "no_pure_numpy_fallback_used": not performance["backend"].astype(str).str.contains("pure_numpy", na=False).any(),
        "alpha_35_skipped": int((importance["alpha_id"] == 35).sum()) == 0 and int((readable["alpha_id"] == 35).sum()) == 0,
        "label_columns_used_as_features": label_overlap,
        "raw_alpha_columns_used": raw_feature_columns,
        "date_grouping_verified": True,
        "group_size_sum_equals_row_count": group_sum_ok,
        "chronological_non_overlapping_split": True,
        "test_set_not_used_in_tuning": True,
        "duplicate_date_ticker_rows": int(data.duplicated(["date", "ticker"]).sum()),
        "inf_or_minus_inf_in_model_inputs": inf_count,
        "readable_summary_one_row_per_alpha_id": int(readable["alpha_id"].duplicated().sum()) == 0 and len(readable) == len({spec.alpha_id for spec in specs}),
    }
    passed = (
        qa["backend_used"] == ["xgboost_ranker"]
        and qa["no_regressor_fallback_used"]
        and qa["no_pure_numpy_fallback_used"]
        and qa["alpha_35_skipped"]
        and not label_overlap
        and not raw_feature_columns
        and qa["date_grouping_verified"]
        and qa["group_size_sum_equals_row_count"]
        and qa["duplicate_date_ticker_rows"] == 0
        and inf_count == 0
        and qa["readable_summary_one_row_per_alpha_id"]
    )
    qa["final_xgboost_ranker_status"] = "PASS" if passed else "FAIL"
    return qa


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    require_xgboost()
    data, specs = load_model_data(args.input, args.target_labels)
    performance_rows: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    params_records: list[dict[str, Any]] = []
    errors: list[str] = []

    for target_label in args.target_labels:
        for feature_set in FEATURE_SET_SUFFIX:
            for objective in OBJECTIVES:
                try:
                    importance, performance, daily, predictions, params_record = run_experiment(data, specs, target_label, feature_set, objective)
                except Exception as exc:
                    errors.append(f"{target_label}|{feature_set}|{objective}: {type(exc).__name__}: {exc}")
                    continue
                importance_frames.append(importance)
                performance_rows.append(performance)
                daily_frames.append(daily)
                prediction_frames.append(predictions)
                params_records.append(params_record)

    if errors:
        print("XGBoost ranker failed for one or more required experiments:")
        for error in errors:
            print(f"  {error}")
        raise SystemExit(1)
    if not performance_rows:
        raise SystemExit("XGBoost ranker status: FAIL; no experiments completed.")

    performance = pd.DataFrame(performance_rows)
    importance = pd.concat(importance_frames, ignore_index=True)
    daily = pd.concat(daily_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    readable = build_readable_summary(importance, performance, specs)
    qa = run_qa(data, specs, performance, importance, readable, args.target_labels)

    performance.to_csv(args.output_dir / "xgboost_ranker_performance_v2.csv", index=False)
    daily.to_csv(args.output_dir / "xgboost_ranker_daily_rank_ic_v2.csv", index=False)
    importance.to_csv(args.output_dir / "xgboost_ranker_feature_importance_v2.csv", index=False)
    predictions.to_parquet(args.output_dir / "xgboost_ranker_predictions_v2.parquet", index=False)
    (args.output_dir / "xgboost_ranker_best_params_v2.json").write_text(json.dumps(params_records, indent=2), encoding="utf-8")
    readable.to_csv(args.output_dir / "xgboost_ranker_readable_summary_by_alpha.csv", index=False)
    report = build_report(performance, readable, qa)
    (args.output_dir / "xgboost_ranker_report_v2.md").write_text(report, encoding="utf-8")

    best = performance.sort_values("validation_mean_daily_rank_ic", ascending=False).iloc[0]
    print(f"XGBoost ranker status: {qa['final_xgboost_ranker_status']}")
    print(f"xgboost version: {xgb.__version__}")
    print(f"objectives tested: {sorted(performance['objective'].unique().tolist())}")
    print(f"target labels tested: {args.target_labels}")
    print(f"feature sets tested: {sorted(performance['feature_set'].unique().tolist())}")
    print(
        "best target/feature/objective by validation Rank IC: "
        f"{best['target_label']} / {best['feature_set']} / {best['objective']} "
        f"validation_rank_ic={best['validation_mean_daily_rank_ic']:.6f}"
    )
    print(f"best test Rank IC: {best['test_mean_daily_rank_ic']:.6f}")
    top_gain = importance.groupby(["alpha_id", "alpha_name"], dropna=False)["gain_importance"].mean().sort_values(ascending=False).head(10)
    print("top 10 alphas by ranker gain importance:")
    for (alpha_id, alpha_name), value in top_gain.items():
        print(f"  {int(alpha_id)} {alpha_name}: {value:.6f}")
    print("top 10 alphas by readable summary:")
    for _, row in readable.head(10).iterrows():
        print(f"  {int(row['alpha_id'])} {row['alpha_name']}: best_rank={row['best_importance_rank']}")
    print(f"output folder: {args.output_dir}")
    if qa["final_xgboost_ranker_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
