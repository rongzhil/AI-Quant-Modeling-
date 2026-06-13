"""Stage 8: train an XGBoost learning-to-rank model over the selected alphas.

This ports the team's ``xgb_ranker`` methodology into this repo's architecture:

  * True XGBoost ranking via ``xgboost.train`` with grouped ``DMatrix`` (one
    trading date = one ranking group). No XGBRegressor / numpy / sklearn
    fallback.
  * Relevance labels are within-date return rank grades 0..31 (XGBoost NDCG
    needs non-negative integer relevance). Evaluation uses the original
    continuous target.
  * Objectives ``rank:pairwise`` and ``rank:ndcg`` both tried.
  * A compact covering hyperparameter grid; the best config is chosen by
    VALIDATION mean daily Rank IC (test never enters tuning).

Differences from the team's version (this repo's choices):
  * Single target ``config.TARGET_COLUMN`` (forward_5d_excess_return), not 4.
  * Fixed split dates from ``config`` (TRAIN/VAL/TEST), not a re-derived 60/20/20.
  * FEATURES: the union alpha list is fed with ALL THREE cross-sectional
    transforms together (``_z`` + ``_rank`` + ``_sector_neutral_rank``) in ONE
    ranker, letting the GBDT pick. (The team trained one ranker per transform
    and compared; here we feed them jointly.) Because a single alpha's three
    transforms are correlated, gain importance is also AGGREGATED BY ALPHA so a
    feature's total contribution is read at the alpha level, not split across
    its three columns.

Inputs:
  * the final alpha panel (date, ticker, transform columns, target);
  * a list of alpha ids to use (the union of the Rank-IC and multivariate
    selections). Defaults to every alpha present.

Outputs (written under config.OUTPUT_DIR):
  * ranker_performance.csv          : val/test metrics per objective.
  * ranker_feature_importance.csv   : gain importance per feature column.
  * ranker_alpha_importance.csv     : gain importance aggregated per alpha.
  * ranker_best_params.json         : chosen hyperparameters + all trials.
  * ranker_report.txt               : human-readable summary.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config


# Cross-sectional transform suffixes produced by cross_section.py.
TRANSFORM_SUFFIXES = ["_z", "_rank", "_sector_neutral_rank"]

OBJECTIVES = ["rank:pairwise", "rank:ndcg"]
RELEVANCE_GRADES = 31
MIN_GROUP_SIZE = 50
NUM_BOOST_ROUND = 200
EARLY_STOPPING_ROUNDS = 30

# Compact covering grid (matches the team's: every value of each family appears
# at least once), tuple = (max_depth, eta, lambda, alpha, min_child_weight).
COVERING_GRID = [
    (2, 0.03, 1.0, 0.0, 5),
    (2, 0.05, 5.0, 0.1, 10),
    (3, 0.03, 5.0, 0.1, 5),
    (3, 0.05, 1.0, 0.0, 10),
    (4, 0.03, 1.0, 0.1, 10),
    (4, 0.05, 5.0, 0.0, 5),
]


# --------------------------------------------------------------------------- #
# Feature columns
# --------------------------------------------------------------------------- #
def _alpha_id_from_column(col: str) -> int | None:
    """Parse the alpha id out of a column name like 'alpha_65_dollar_volume'."""
    parts = str(col).split("_")
    if len(parts) >= 2 and parts[0] == "alpha":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def load_union_alpha_ids(
    selected_features_path: Path | None = None,
    multivariate_selected_path: Path | None = None,
    verbose: bool = True,
) -> list[int]:
    """Read the two selection outputs and return the UNION of their alpha ids.

    Reads (live, from disk):
      * selected_features.csv      -> Rank-IC list (column ``alpha`` holds the
        full alpha column name, e.g. ``alpha_65_dollar_volume``).
      * multivariate_selected.csv  -> ElasticNet/Lasso list (column ``alpha_id``).

    Either file may be absent; whichever is present contributes its ids. The
    union is always recomputed from the current files, so it never goes stale.
    """
    a_path = Path(selected_features_path) if selected_features_path else config.SELECTED_FEATURES_PATH
    b_path = (Path(multivariate_selected_path) if multivariate_selected_path
              else config.OUTPUT_DIR / "multivariate_selected.csv")

    a_ids: set[int] = set()
    if a_path.exists():
        df = pd.read_csv(a_path)
        col = "alpha" if "alpha" in df.columns else df.columns[0]
        for v in df[col]:
            aid = _alpha_id_from_column(v)
            if aid is not None:
                a_ids.add(aid)
    elif verbose:
        print(f"  [union] Rank-IC list not found at {a_path}")

    b_ids: set[int] = set()
    if b_path.exists():
        df = pd.read_csv(b_path)
        if "alpha_id" in df.columns:
            b_ids = {int(x) for x in df["alpha_id"].dropna()}
    elif verbose:
        print(f"  [union] multivariate list not found at {b_path}")

    union = sorted(a_ids | b_ids)
    if verbose:
        print(f"  [union] Rank-IC: {len(a_ids)} | multivariate: {len(b_ids)} | "
              f"union: {len(union)} -> {union}")
    return union


# --------------------------------------------------------------------------- #
# Feature columns (transforms)
# --------------------------------------------------------------------------- #
def feature_columns_for_alphas(panel_columns, alpha_ids: list[int]) -> list[tuple[int, str]]:
    """Return (alpha_id, feature_column) for every transform of every alpha id.

    Feeds all three transforms (_z / _rank / _sector_neutral_rank) for each
    selected alpha that exists in the panel.
    """
    col_set = set(panel_columns)
    pairs: list[tuple[int, str]] = []
    for aid in alpha_ids:
        base = config.ALPHA_COLUMN_BY_ID.get(aid)
        if base is None:
            continue
        for suf in TRANSFORM_SUFFIXES:
            col = f"{base}{suf}"
            if col in col_set:
                pairs.append((aid, col))
    return pairs


# --------------------------------------------------------------------------- #
# Splits (fixed config dates)
# --------------------------------------------------------------------------- #
def _slice(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    d = pd.to_datetime(panel["date"]).dt.normalize()
    mask = (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))
    return panel[mask].copy()


def make_splits(panel: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train / val / test frames using the fixed dates in config."""
    target = config.TARGET_COLUMN
    keep = ["date", "ticker", target, *feature_cols]
    base = panel[keep].copy()
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    base = base.dropna(subset=[target])  # can't rank without a target
    train = _slice(base, config.TRAIN_START_DATE, config.TRAIN_END_DATE)
    val = _slice(base, config.VAL_START_DATE, config.VAL_END_DATE)
    test = _slice(base, config.TEST_START_DATE, config.TEST_END_DATE)
    return train, val, test


def drop_small_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop dates with fewer than MIN_GROUP_SIZE rows (ranking needs a crowd)."""
    counts = frame.groupby("date", sort=False).size()
    keep = counts[counts >= MIN_GROUP_SIZE].index
    return frame[frame["date"].isin(keep)].copy()


# --------------------------------------------------------------------------- #
# DMatrix
# --------------------------------------------------------------------------- #
def relevance_by_date(frame: pd.DataFrame) -> np.ndarray:
    """Within-date continuous target -> integer 0..31 relevance grades."""
    target = config.TARGET_COLUMN
    pct = frame.groupby("date", sort=False)[target].rank(method="average", pct=True)
    grades = np.floor(pct.to_numpy(dtype=float) * RELEVANCE_GRADES)
    return np.clip(grades, 0, RELEVANCE_GRADES).astype(np.float32)


def group_sizes(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby("date", sort=False).size().to_numpy(dtype=np.uint32)


def build_dmatrix(frame: pd.DataFrame, feature_cols: list[str], with_label: bool = True) -> xgb.DMatrix:
    """Grouped DMatrix sorted by date/ticker; native NaN handling."""
    ordered = frame.sort_values(["date", "ticker"]).reset_index(drop=True)
    x = ordered[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if with_label:
        labels = relevance_by_date(ordered)
        dmat = xgb.DMatrix(x.to_numpy(dtype=np.float32), label=labels,
                           feature_names=feature_cols, missing=np.nan)
    else:
        dmat = xgb.DMatrix(x.to_numpy(dtype=np.float32),
                           feature_names=feature_cols, missing=np.nan)
    groups = group_sizes(ordered)
    dmat.set_group(groups)
    if int(groups.sum()) != len(ordered):
        raise ValueError("group size sum != row count")
    return dmat


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _daily_rank_ic(pred_frame: pd.DataFrame) -> pd.DataFrame:
    """Per-date Spearman(target, prediction) and top-bottom decile spread."""
    rows: list[dict[str, Any]] = []
    for date_value, g in pred_frame.groupby("date", sort=True):
        u = g[["target", "prediction"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(u) < MIN_GROUP_SIZE:
            continue
        tr = u["target"].rank(method="average")
        pr = u["prediction"].rank(method="average")
        ic = float(tr.corr(pr)) if tr.std(ddof=1) and pr.std(ddof=1) else np.nan
        ranks = u["prediction"].rank(method="average", pct=True)
        top = u.loc[ranks >= 0.9, "target"]
        bottom = u.loc[ranks <= 0.1, "target"]
        spread = (float(top.mean()) - float(bottom.mean())) if len(top) and len(bottom) else np.nan
        rows.append({"date": date_value, "rank_ic": ic, "spread": spread, "n": len(u)})
    return pd.DataFrame(rows)


def _ndcg_by_date(pred_frame: pd.DataFrame) -> float:
    scores: list[float] = []
    for _, g in pred_frame.groupby("date", sort=True):
        u = g[["target", "prediction"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(u) < MIN_GROUP_SIZE:
            continue
        rel = np.floor(u["target"].rank(method="average", pct=True).to_numpy(float) * RELEVANCE_GRADES)
        rel = np.clip(rel, 0, RELEVANCE_GRADES)
        order = np.argsort(-u["prediction"].to_numpy(float))
        ideal = np.argsort(-rel)
        disc = 1.0 / np.log2(np.arange(2, len(u) + 2))
        dcg = float(np.sum(rel[order] * disc))
        idcg = float(np.sum(rel[ideal] * disc))
        if idcg > 0:
            scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else float("nan")


def _t_stat(s: pd.Series) -> float:
    c = pd.to_numeric(s, errors="coerce").dropna()
    if len(c) < 2:
        return float("nan")
    sd = float(c.std(ddof=1))
    return float(c.mean() / (sd / math.sqrt(len(c)))) if sd else float("nan")


def metrics_from_predictions(pred_frame: pd.DataFrame) -> dict[str, float]:
    daily = _daily_rank_ic(pred_frame)
    rank_ic = daily["rank_ic"] if not daily.empty else pd.Series(dtype=float)
    spread = daily["spread"] if not daily.empty else pd.Series(dtype=float)
    mean_ic = float(rank_ic.mean()) if len(rank_ic) else np.nan
    std_ic = float(rank_ic.std(ddof=1)) if len(rank_ic) > 1 else np.nan
    return {
        "mean_rank_ic": mean_ic,
        "icir": (mean_ic / std_ic) if std_ic else np.nan,
        "rank_ic_t_stat": _t_stat(rank_ic),
        "positive_ic_ratio": float((rank_ic > 0).mean()) if len(rank_ic) else np.nan,
        "mean_spread": float(spread.mean()) if len(spread) else np.nan,
        "spread_t_stat": _t_stat(spread),
        "mean_ndcg": _ndcg_by_date(pred_frame),
    }


def predict_frame(frame: pd.DataFrame, booster: xgb.Booster, feature_cols: list[str]) -> pd.DataFrame:
    ordered = frame.sort_values(["date", "ticker"]).reset_index(drop=True)
    dmat = build_dmatrix(ordered, feature_cols, with_label=False)
    out = ordered[["date", "ticker", config.TARGET_COLUMN]].rename(columns={config.TARGET_COLUMN: "target"})
    out["prediction"] = booster.predict(dmat)
    return out


# --------------------------------------------------------------------------- #
# Train one objective: grid search on val Rank IC
# --------------------------------------------------------------------------- #
def train_objective(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    objective: str,
    verbose: bool = True,
) -> tuple[xgb.Booster, dict[str, Any], list[dict[str, Any]]]:
    """Grid-search the covering grid; pick by validation mean daily Rank IC."""
    dtrain = build_dmatrix(train, feature_cols, with_label=True)
    dval = build_dmatrix(val, feature_cols, with_label=True)
    best_booster: xgb.Booster | None = None
    best_params: dict[str, Any] | None = None
    best_score = -np.inf
    trials: list[dict[str, Any]] = []
    for trial_id, (max_depth, eta, reg_lambda, reg_alpha, min_child_weight) in enumerate(COVERING_GRID, 1):
        params = {
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
        booster = xgb.train(
            params, dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dval, "validation")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        pred = predict_frame(val, booster, feature_cols)
        val_ic = metrics_from_predictions(pred)["mean_rank_ic"]
        best_iter = int(getattr(booster, "best_iteration", 0) or 0)
        trials.append({"trial_id": trial_id, "objective": objective,
                       "validation_rank_ic": val_ic, "best_iteration": best_iter, **params})
        if verbose:
            print(f"    [{objective}] trial {trial_id}/6 "
                  f"depth={max_depth} eta={eta} lambda={reg_lambda} alpha={reg_alpha} "
                  f"mcw={min_child_weight} -> val Rank IC={val_ic:.5f}")
        if np.isfinite(val_ic) and val_ic > best_score:
            best_score = val_ic
            best_booster = booster
            best_params = {**params, "best_iteration": best_iter, "trial_id": trial_id}
    if best_booster is None:
        raise RuntimeError(f"no valid ranker for objective={objective}")
    return best_booster, best_params, trials


# --------------------------------------------------------------------------- #
# Importance (aggregated by alpha)
# --------------------------------------------------------------------------- #
def importance_tables(
    booster: xgb.Booster,
    feature_pairs: list[tuple[int, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-column and per-alpha gain importance.

    Per-alpha sums the three transform columns of the same alpha, so a feature's
    total contribution is read at the alpha level.
    """
    gain = booster.get_score(importance_type="gain")
    rows = []
    for aid, col in feature_pairs:
        rows.append({
            "alpha_id": aid,
            "alpha_name": config.ALPHA_SPEC_BY_ID[aid].name if aid in config.ALPHA_SPEC_BY_ID else "",
            "feature_column": col,
            "gain_importance": float(gain.get(col, 0.0)),
        })
    per_col = pd.DataFrame(rows)
    if per_col.empty:
        return per_col, per_col
    total = per_col["gain_importance"].sum()
    per_col["normalized_gain"] = per_col["gain_importance"] / total if total > 0 else 0.0
    per_alpha = (
        per_col.groupby(["alpha_id", "alpha_name"], as_index=False)["gain_importance"].sum()
        .sort_values("gain_importance", ascending=False)
        .reset_index(drop=True)
    )
    atot = per_alpha["gain_importance"].sum()
    per_alpha["normalized_gain"] = per_alpha["gain_importance"] / atot if atot > 0 else 0.0
    per_col = per_col.sort_values("gain_importance", ascending=False).reset_index(drop=True)
    return per_col, per_alpha


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(
    panel: pd.DataFrame,
    alpha_ids: list[int] | None = None,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Train the ranker over the union alpha list and write artifacts.

    alpha_ids : the alpha ids to use. If None, the UNION of the two selection
        reports (Rank-IC selected_features.csv + multivariate_selected.csv) is
        read live from disk via ``load_union_alpha_ids`` — so it always reflects
        the latest selection outputs. Pass an explicit list to override.
    """
    out_dir = Path(output_dir) if output_dir is not None else config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if alpha_ids is None:
        alpha_ids = load_union_alpha_ids(verbose=verbose)
        if not alpha_ids:
            raise ValueError(
                "no alpha ids found in selection reports; run feature selection "
                "first or pass alpha_ids explicitly"
            )
    feature_pairs = feature_columns_for_alphas(panel.columns, alpha_ids)
    feature_cols = [col for _, col in feature_pairs]
    if not feature_cols:
        raise ValueError("no feature columns found for the requested alpha ids")

    if verbose:
        print(f"Stage 8: XGBoost ranker | {len(alpha_ids)} alphas x transforms "
              f"= {len(feature_cols)} feature columns | target={config.TARGET_COLUMN}")

    train, val, test = make_splits(panel, feature_cols)
    train, val, test = drop_small_groups(train), drop_small_groups(val), drop_small_groups(test)
    if verbose:
        print(f"  train: {len(train)} rows / {train['date'].nunique()} dates | "
              f"val: {len(val)} rows / {val['date'].nunique()} dates | "
              f"test: {len(test)} rows / {test['date'].nunique()} dates")

    perf_rows: list[dict[str, Any]] = []
    params_records: list[dict[str, Any]] = []
    per_col_frames: list[pd.DataFrame] = []
    per_alpha_frames: list[pd.DataFrame] = []
    boosters: dict[str, xgb.Booster] = {}

    for objective in OBJECTIVES:
        if verbose:
            print(f"  objective={objective}: grid search on validation Rank IC ...")
        booster, best_params, trials = train_objective(train, val, feature_cols, objective, verbose=verbose)
        boosters[objective] = booster

        val_pred = predict_frame(val, booster, feature_cols)
        test_pred = predict_frame(test, booster, feature_cols)
        val_m = metrics_from_predictions(val_pred)
        test_m = metrics_from_predictions(test_pred)
        perf_rows.append({
            "objective": objective,
            "n_features": len(feature_cols),
            "n_alphas": len(alpha_ids),
            "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
            "train_dates": train["date"].nunique(),
            "val_dates": val["date"].nunique(),
            "test_dates": test["date"].nunique(),
            "best_trial_id": best_params["trial_id"],
            "best_max_depth": best_params["max_depth"],
            "best_eta": best_params["eta"],
            "best_lambda": best_params["lambda"],
            "best_alpha": best_params["alpha"],
            "best_min_child_weight": best_params["min_child_weight"],
            "best_iteration": best_params["best_iteration"],
            "val_mean_rank_ic": val_m["mean_rank_ic"],
            "val_icir": val_m["icir"],
            "val_rank_ic_t_stat": val_m["rank_ic_t_stat"],
            "val_mean_spread": val_m["mean_spread"],
            "val_mean_ndcg": val_m["mean_ndcg"],
            "test_mean_rank_ic": test_m["mean_rank_ic"],
            "test_icir": test_m["icir"],
            "test_rank_ic_t_stat": test_m["rank_ic_t_stat"],
            "test_positive_ic_ratio": test_m["positive_ic_ratio"],
            "test_mean_spread": test_m["mean_spread"],
            "test_spread_t_stat": test_m["spread_t_stat"],
            "test_mean_ndcg": test_m["mean_ndcg"],
        })
        params_records.append({"objective": objective, "best_params": best_params, "all_trials": trials})

        per_col, per_alpha = importance_tables(booster, feature_pairs)
        per_col["objective"] = objective
        per_alpha["objective"] = objective
        per_col_frames.append(per_col)
        per_alpha_frames.append(per_alpha)

        if verbose:
            print(f"    val Rank IC={val_m['mean_rank_ic']:.5f} | "
                  f"test Rank IC={test_m['mean_rank_ic']:.5f} | "
                  f"test ICIR={test_m['icir']:.3f} | test NDCG={test_m['mean_ndcg']:.4f}")

    performance = pd.DataFrame(perf_rows)
    per_col_all = pd.concat(per_col_frames, ignore_index=True)
    per_alpha_all = pd.concat(per_alpha_frames, ignore_index=True)

    # Write artifacts.
    perf_path = out_dir / "ranker_performance.csv"
    col_imp_path = out_dir / "ranker_feature_importance.csv"
    alpha_imp_path = out_dir / "ranker_alpha_importance.csv"
    params_path = out_dir / "ranker_best_params.json"
    report_path = out_dir / "ranker_report.txt"

    performance.to_csv(perf_path, index=False)
    per_col_all.to_csv(col_imp_path, index=False)
    per_alpha_all.to_csv(alpha_imp_path, index=False)
    params_path.write_text(json.dumps(params_records, indent=2, default=str), encoding="utf-8")
    report_path.write_text(build_report(performance, per_alpha_all, alpha_ids, feature_cols), encoding="utf-8")

    if verbose:
        best = performance.sort_values("val_mean_rank_ic", ascending=False).iloc[0]
        print(f"  best objective by val Rank IC: {best['objective']} "
              f"(test Rank IC={best['test_mean_rank_ic']:.5f})")
        print(f"  artifacts -> {out_dir}")

    return {
        "performance": performance,
        "feature_importance": per_col_all,
        "alpha_importance": per_alpha_all,
        "boosters": boosters,
        "paths": {
            "performance": str(perf_path),
            "feature_importance": str(col_imp_path),
            "alpha_importance": str(alpha_imp_path),
            "best_params": str(params_path),
            "report": str(report_path),
        },
    }


def build_report(performance: pd.DataFrame, per_alpha_all: pd.DataFrame, alpha_ids: list[int], feature_cols: list[str]) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("XGBOOST RANKER (Stage 8)")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Target        : {config.TARGET_COLUMN}")
    lines.append(f"Alphas        : {len(alpha_ids)}  ({len(feature_cols)} feature columns = alphas x 3 transforms)")
    lines.append(f"Transforms    : {', '.join(TRANSFORM_SUFFIXES)} (fed jointly; GBDT picks)")
    lines.append(f"Objectives    : {', '.join(OBJECTIVES)}")
    lines.append(f"Relevance     : within-date return ranks, 0..{RELEVANCE_GRADES} grades")
    lines.append(f"Grid          : {len(COVERING_GRID)}-config covering grid, selected by VAL mean daily Rank IC")
    lines.append(f"Split         : train {config.TRAIN_START_DATE}..{config.TRAIN_END_DATE} | "
                 f"val {config.VAL_START_DATE}..{config.VAL_END_DATE} | "
                 f"test {config.TEST_START_DATE}..{config.TEST_END_DATE}")
    lines.append("")
    lines.append("-" * 72)
    lines.append("PERFORMANCE")
    lines.append("-" * 72)
    for _, r in performance.iterrows():
        lines.append(
            f"  {r['objective']:<14s} "
            f"val IC={r['val_mean_rank_ic']:+.5f}  "
            f"test IC={r['test_mean_rank_ic']:+.5f}  "
            f"test ICIR={r['test_icir']:+.3f}  "
            f"test spread={r['test_mean_spread']:+.5f}  "
            f"test NDCG={r['test_mean_ndcg']:.4f}"
        )
    lines.append("")
    lines.append("-" * 72)
    lines.append("TOP ALPHAS BY GAIN IMPORTANCE (aggregated over transforms, averaged over objectives)")
    lines.append("-" * 72)
    agg = (per_alpha_all.groupby(["alpha_id", "alpha_name"], as_index=False)["gain_importance"]
           .mean().sort_values("gain_importance", ascending=False))
    for _, r in agg.head(20).iterrows():
        lines.append(f"  alpha_{int(r['alpha_id']):<3d} {r['alpha_name']:<34s} gain={r['gain_importance']:.2f}")
    lines.append("")
    return "\n".join(lines) + "\n"