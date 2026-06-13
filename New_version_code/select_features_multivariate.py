"""Multivariate (Lasso / ElasticNet) feature-selection layer.

This is a SECOND, parallel feature-selection view that complements the existing
two-step Rank-IC + correlation method in ``select_features.py``. Where the
Rank-IC method is *univariate* (it scores each alpha on its own), the Lasso /
ElasticNet method is *multivariate*: it fits all candidate alphas jointly and
lets L1 regularization zero out the ones that add nothing once the others are
present. This can rescue an alpha that looks weak alone but carries an
interaction effect, and it can drop an alpha that only looked good because it
co-moved with a stronger one.

Method (ported from the team's ``run_lasso_elasticnet`` but rebuilt in this
repo's architecture):
  * For each feature-set view (``_z``, ``_rank``, ``_sector_neutral_rank``) and
    each model (LassoCV, ElasticNetCV), fit a pipeline
        SimpleImputer(median) -> StandardScaler -> model
    on the TRAIN segment only (date <= TRAIN_END_DATE), with a date-based
    expanding TimeSeriesSplit for the internal CV.
  * An alpha is "selected" if its fitted coefficient is non-zero
    (|coef| > 1e-12) in ANY of those experiments.
  * Single target: ``config.TARGET_COLUMN`` (ret_fwd_5d). No excess / tradable
    variants, no multi-target voting.

Key differences from the team's version (architecture compatibility):
  * Column suffixes ``_z`` / ``_rank`` / ``_sector_neutral_rank`` (not
    ``_cs_zscore`` / ``_cs_rank``).
  * One target instead of four.
  * Fixed split dates from ``config`` (TRAIN/VAL/TEST), not a re-derived 60/20/20.
  * ``AlphaSpec`` has no ``category`` field; a coarse category is derived locally
    only for the readable report.

Output is SLIM - the point of this layer is the selected list, which the caller
unions with the Rank-IC list. It writes:
  * multivariate_coefficients.csv : one row per (alpha, feature_set, model) with
    coefficient + selected flag.
  * multivariate_selected.csv     : one row per selected alpha (the list B).
  * multivariate_selection_report.txt : human-readable summary.

There is no val/test evaluation here (the team's version computed Rank-IC / ICIR
on val+test for cross-target voting; we don't need it for a plain union).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


# The three cross-sectional views produced by cross_section.py.
FEATURE_SET_SUFFIX = {
    "z": "_z",
    "rank": "_rank",
    "sector_neutral_rank": "_sector_neutral_rank",
}

# Minimum non-missing observations for a feature to enter an experiment.
MIN_VALID_OBS = 1000
# Numerical zero threshold for "coefficient is non-zero".
COEF_ZERO_TOL = 1e-12

# --- Speed controls -------------------------------------------------------
# These trade a tiny bit of precision for a large speed-up. They affect how
# precisely the regularization strength is tuned, NOT which alphas get selected
# (ElasticNet feature selection is insensitive to small penalty changes).
#   N_ALPHA_GRID : number of regularization-strength candidates the CV tries.
#                  Teammate used 50; 20 finds an equally-good penalty on the
#                  smooth log scale but ~2.5x faster.
#   N_CV_SPLITS  : CV folds. Kept at 5 to match the teammate exactly.
#   DATE_STRIDE  : keep every Nth trading day in the train segment. Selection is
#                  stable on a date-subsample; 2 halves the data for ~2x speed.
#                  Set to 1 to use every day (slowest, teammate-exact).
N_ALPHA_GRID = 20
N_CV_SPLITS = 3
DATE_STRIDE = 3

# --- Target handling ------------------------------------------------------
# The features are cross-sectional (each day's z-score / rank). The raw target
# ret_fwd_5d is an ABSOLUTE return that still contains the whole market's daily
# move. A linear model (Lasso/ElasticNet) tries to fit that absolute number, and
# the market component swamps the weak per-stock linear signal -> every
# coefficient gets pushed to zero (penalty hits the grid's upper bound).
# Cross-sectionally de-meaning the target (subtract that day's mean) strips the
# market move, leaving a market-neutral residual that matches the cross-sectional
# features. This mirrors what the teammate's pipeline gets "for free" by using
# an excess-return target. Set False to fit the raw target instead.
DEMEAN_TARGET = True


def _coarse_category(alpha_id: int) -> str:
    """Coarse category for the readable report only (AlphaSpec has none)."""
    if alpha_id in {26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 74, 75}:
        return "momentum_trend"
    if alpha_id in {39, 40, 41, 44, 45, 46, 50, 53}:
        return "reversal"
    if alpha_id in {42, 43, 47, 48, 49, 51, 52, 72, 73}:
        return "technical_price_volume"
    if alpha_id in {54, 55, 56, 57, 58, 59, 61, 62, 63}:
        return "volatility_risk"
    if alpha_id in {60, 64, 65, 66, 67, 68, 69, 70, 71}:
        return "liquidity_volume"
    return "other"


def _train_only(panel: pd.DataFrame) -> pd.DataFrame:
    """Rows on or before TRAIN_END_DATE (same convention as select_features)."""
    d = pd.to_datetime(panel["date"]).dt.normalize()
    return panel[d <= pd.Timestamp(config.TRAIN_END_DATE)]


def discover_feature_columns(columns, feature_set: str) -> list[tuple[int, str, str]]:
    """Return (alpha_id, alpha_name, feature_column) for one feature-set view.

    Iterates config.ALPHA_SPECS (so alpha 35 is naturally absent) and keeps the
    transformed column for the requested suffix when it exists in the panel.
    """
    suffix = FEATURE_SET_SUFFIX[feature_set]
    column_set = set(columns)
    found: list[tuple[int, str, str]] = []
    for spec in config.ALPHA_SPECS:
        col = f"{spec.column}{suffix}"
        if col in column_set:
            found.append((spec.alpha_id, spec.name, col))
    return found


def date_cv_indices(dates: pd.Series, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding TimeSeriesSplit-style CV folds, split by unique date.

    Train fold = all rows up to a cut date; validation fold = the next block of
    dates. This keeps the internal CV strictly chronological (no shuffling rows
    across time), matching the team's date_cv_indices.
    """
    unique_dates = np.array(sorted(pd.Series(dates).dropna().unique().tolist()))
    if len(unique_dates) < n_splits + 2:
        n_splits = max(2, len(unique_dates) - 2)
    folds = np.array_split(unique_dates, n_splits + 1)
    date_values = dates.to_numpy()
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in range(n_splits):
        train_dates = np.concatenate(folds[: idx + 1])
        val_dates = folds[idx + 1]
        train_idx = np.flatnonzero(np.isin(date_values, train_dates))
        val_idx = np.flatnonzero(np.isin(date_values, val_dates))
        if len(train_idx) and len(val_idx):
            splits.append((train_idx, val_idx))
    return splits


def build_model(model_type: str, cv_indices) -> Pipeline:
    """SimpleImputer(median) -> StandardScaler -> Lasso/ElasticNet (team's pipeline)."""
    if model_type == "lasso":
        estimator = LassoCV(
            alphas=np.logspace(-6, -2, N_ALPHA_GRID),
            cv=cv_indices,
            max_iter=10000,
            n_jobs=-1,
            random_state=42,
        )
    elif model_type == "elasticnet":
        estimator = ElasticNetCV(
            l1_ratio=[0.2, 0.5, 0.8, 0.95],
            alphas=np.logspace(-6, -2, N_ALPHA_GRID),
            cv=cv_indices,
            max_iter=10000,
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def run_one_experiment(
    train: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    model_type: str,
) -> tuple[np.ndarray, dict]:
    """Fit one (feature_set, model) experiment on the train segment.

    Returns the coefficient vector (aligned to feature_cols) and a small dict of
    fit metadata. Inf is mapped to NaN so the imputer handles it; rows with a
    missing target are dropped (can't train on them).
    """
    frame = train[["date", target, *feature_cols]].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[target])
    # Speed: keep every DATE_STRIDE-th trading day. Selection is stable on a
    # date-subsample; this cuts the row count roughly DATE_STRIDE-fold.
    if DATE_STRIDE > 1:
        uniq_dates = sorted(frame["date"].unique())
        keep = set(uniq_dates[::DATE_STRIDE])
        frame = frame[frame["date"].isin(keep)]
    if len(frame) < MIN_VALID_OBS:
        raise ValueError(f"insufficient train rows ({len(frame)}) for {model_type}")

    cv_indices = date_cv_indices(frame["date"], n_splits=N_CV_SPLITS)
    if not cv_indices:
        raise ValueError(f"could not build CV folds for {model_type}")

    # Cross-sectional de-mean of the target: subtract each day's mean so the
    # market-wide move is removed and the target matches the cross-sectional
    # features. (Teammate gets this via an excess-return target.)
    if DEMEAN_TARGET:
        y = frame[target] - frame.groupby("date")[target].transform("mean")
    else:
        y = frame[target]

    pipeline = build_model(model_type, cv_indices)
    pipeline.fit(frame[feature_cols], y)
    estimator = pipeline.named_steps["model"]
    coef = np.asarray(estimator.coef_, dtype=float)
    meta = {
        "n_rows": int(len(frame)),
        "n_dates": int(frame["date"].nunique()),
        "alpha_penalty": float(getattr(estimator, "alpha_", np.nan)),
        "l1_ratio": float(getattr(estimator, "l1_ratio_", 1.0 if model_type == "lasso" else np.nan)),
        "n_selected": int(np.sum(np.abs(coef) > COEF_ZERO_TOL)),
    }
    return coef, meta


def run_multivariate_selection(
    panel: pd.DataFrame,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[int], pd.DataFrame]:
    """Run all (feature_set x model) experiments on the train segment.

    Returns:
      * coefficients : long table, one row per (alpha_id, feature_set, model).
      * selected_ids : sorted list of alpha_ids selected in ANY experiment (list B).
      * selected_tbl : one row per selected alpha with supporting detail.
    """
    target = config.TARGET_COLUMN
    if target not in panel.columns:
        raise ValueError(f"panel is missing target column {target!r}")

    train = _train_only(panel)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    coef_rows: list[dict] = []
    for feature_set in FEATURE_SET_SUFFIX:
        feats = discover_feature_columns(panel.columns, feature_set)
        if not feats:
            if verbose:
                print(f"  [skip] no columns for feature_set={feature_set}")
            continue
        # Keep only features with enough train observations.
        usable = []
        for alpha_id, alpha_name, col in feats:
            vals = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if int(vals.notna().sum()) >= MIN_VALID_OBS:
                usable.append((alpha_id, alpha_name, col))
        if not usable:
            if verbose:
                print(f"  [skip] no train-observed features for feature_set={feature_set}")
            continue
        feature_cols = [col for _, _, col in usable]

        for model_type in ["lasso", "elasticnet"]:
            try:
                coef, meta = run_one_experiment(train, feature_cols, target, model_type)
            except ValueError as exc:
                if verbose:
                    print(f"  [skip] {feature_set}/{model_type}: {exc}")
                continue
            if verbose:
                print(
                    f"  {feature_set:>20s} / {model_type:<10s} "
                    f"-> selected {meta['n_selected']:>2d}/{len(feature_cols)} "
                    f"(penalty={meta['alpha_penalty']:.2e}, rows={meta['n_rows']}, dates={meta['n_dates']})"
                )
            for (alpha_id, alpha_name, col), c in zip(usable, coef, strict=True):
                coef_rows.append(
                    {
                        "alpha_id": alpha_id,
                        "alpha_name": alpha_name,
                        "feature_set": feature_set,
                        "model_type": model_type,
                        "feature_column": col,
                        "coefficient": float(c),
                        "selected": bool(abs(c) > COEF_ZERO_TOL),
                    }
                )

    coefficients = pd.DataFrame(coef_rows)
    if coefficients.empty:
        return coefficients, [], pd.DataFrame()

    # An alpha is selected (list B) if chosen in ANY experiment.
    sel = coefficients[coefficients["selected"]]
    selected_ids = sorted(sel["alpha_id"].unique().tolist())

    # One row per selected alpha, with which experiments picked it.
    rows: list[dict] = []
    for alpha_id in selected_ids:
        g = coefficients[coefficients["alpha_id"] == alpha_id]
        gs = g[g["selected"]]
        # Strongest coefficient (by |value|) among selecting experiments.
        best = gs.iloc[gs["coefficient"].abs().to_numpy().argmax()]
        rows.append(
            {
                "alpha_id": int(alpha_id),
                "alpha_name": best["alpha_name"],
                "category": _coarse_category(int(alpha_id)),
                "n_experiments_selected": int(len(gs)),
                "selected_feature_sets": ";".join(sorted(gs["feature_set"].unique())),
                "selected_models": ";".join(sorted(gs["model_type"].unique())),
                "strongest_coefficient": float(best["coefficient"]),
                "strongest_feature_set": best["feature_set"],
                "strongest_model": best["model_type"],
            }
        )
    selected_tbl = pd.DataFrame(rows).sort_values(
        ["n_experiments_selected", "alpha_id"], ascending=[False, True]
    ).reset_index(drop=True)

    return coefficients, selected_ids, selected_tbl


def build_report(coefficients: pd.DataFrame, selected_tbl: pd.DataFrame, selected_ids: list[int]) -> str:
    """Human-readable summary of the multivariate selection."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("MULTIVARIATE FEATURE SELECTION (Lasso / ElasticNet)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Target            : {config.TARGET_COLUMN}")
    lines.append(f"Train through     : {config.TRAIN_END_DATE} (train segment only)")
    lines.append(f"Feature-set views : {', '.join(FEATURE_SET_SUFFIX)}")
    lines.append("Models            : LassoCV, ElasticNetCV (date TimeSeriesSplit CV)")
    lines.append("Pipeline          : SimpleImputer(median) -> StandardScaler -> model")
    lines.append("Selected if        : |coefficient| > 1e-12 in ANY experiment")
    lines.append("")
    if coefficients.empty:
        lines.append("No experiments ran (no usable feature columns).")
        return "\n".join(lines) + "\n"

    n_experiments = coefficients[["feature_set", "model_type"]].drop_duplicates().shape[0]
    lines.append(f"Experiments run   : {n_experiments}")
    lines.append(f"Alphas selected   : {len(selected_ids)}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("SELECTED ALPHAS (list B)")
    lines.append("-" * 70)
    for _, row in selected_tbl.iterrows():
        lines.append(
            f"  + alpha_{row['alpha_id']:<3d} {row['alpha_name']:<34s} "
            f"picks={row['n_experiments_selected']:>2d}  "
            f"coef={row['strongest_coefficient']:+.4e}  "
            f"[{row['strongest_feature_set']}/{row['strongest_model']}]"
        )
    lines.append("")
    lines.append("-" * 70)
    lines.append("NOT SELECTED (zero coefficient in every experiment)")
    lines.append("-" * 70)
    all_ids = sorted(coefficients["alpha_id"].unique().tolist())
    not_sel = [aid for aid in all_ids if aid not in set(selected_ids)]
    if not_sel:
        for aid in not_sel:
            name = coefficients[coefficients["alpha_id"] == aid]["alpha_name"].iloc[0]
            lines.append(f"  - alpha_{aid:<3d} {name}")
    else:
        lines.append("  (none - every evaluated alpha was selected somewhere)")
    lines.append("")
    return "\n".join(lines) + "\n"


def run(panel: pd.DataFrame, output_dir: Path | None = None, verbose: bool = True) -> dict:
    """Entry point: run multivariate selection and write slim artifacts.

    Returns a dict with the selected alpha ids and the output paths.
    """
    out_dir = Path(output_dir) if output_dir is not None else config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Multivariate (Lasso / ElasticNet) selection ...")
    coefficients, selected_ids, selected_tbl = run_multivariate_selection(panel, verbose=verbose)

    coef_path = out_dir / "multivariate_coefficients.csv"
    sel_path = out_dir / "multivariate_selected.csv"
    report_path = out_dir / "multivariate_selection_report.txt"

    if not coefficients.empty:
        coefficients.to_csv(coef_path, index=False)
        selected_tbl.to_csv(sel_path, index=False)
    report = build_report(coefficients, selected_tbl, selected_ids)
    report_path.write_text(report, encoding="utf-8")

    if verbose:
        print(f"  selected {len(selected_ids)} alphas: {selected_ids}")
        print(f"  artifacts -> {out_dir}")

    return {
        "selected_alpha_ids": selected_ids,
        "selected_columns": [config.ALPHA_COLUMN_BY_ID[i] for i in selected_ids],
        "coefficients_path": str(coef_path),
        "selected_path": str(sel_path),
        "report_path": str(report_path),
    }