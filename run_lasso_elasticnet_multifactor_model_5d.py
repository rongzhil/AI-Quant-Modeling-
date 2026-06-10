from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.model_selection import TimeSeriesSplit


PROJECT_ROOT = Path(r"F:\icarus fund\Quantitative AI financial modeling\regression")
RESULT_PACKAGE = PROJECT_ROOT / "回归结果包"

CANDIDATE_SUMMARY_PATH = PROJECT_ROOT / "alpha_model_candidate_summary_5d.csv"
PANEL_PATH_PRIMARY = RESULT_PACKAGE / "sp500_alpha_target_panel.csv"
PANEL_PATH_FALLBACK = PROJECT_ROOT / "sp500_alpha_target_panel.csv"

COEFFICIENTS_OUT = PROJECT_ROOT / "lasso_elasticnet_model_coefficients_5d.csv"
PERFORMANCE_OUT = PROJECT_ROOT / "lasso_elasticnet_performance_summary_5d.csv"
DAILY_IC_OUT = PROJECT_ROOT / "lasso_elasticnet_daily_rank_ic_5d.csv"
PREDICTIONS_OUT = PROJECT_ROOT / "lasso_elasticnet_predictions_5d.csv"
REPORT_OUT = PROJECT_ROOT / "lasso_elasticnet_report_5d.txt"

TARGET_COL = "ret_fwd_5d"
DATE_COL = "date"
TICKER_COL = "ticker"
KEEP_RECOMMENDATIONS = {"Strong Keep", "Optional Keep"}
ALPHAS = np.logspace(-5, 0, 50)
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
RANDOM_STATE = 42
MAX_ITER = 10000
MIN_DAILY_OBS = 50
COEF_EPS = 1e-8


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def resolve_panel_path() -> Path:
    if PANEL_PATH_PRIMARY.exists():
        return PANEL_PATH_PRIMARY
    if PANEL_PATH_FALLBACK.exists():
        return PANEL_PATH_FALLBACK
    raise FileNotFoundError(
        "Could not find panel file. Tried:\n"
        f"  {PANEL_PATH_PRIMARY}\n"
        f"  {PANEL_PATH_FALLBACK}"
    )


def original_feature_from_aligned(aligned_name: str) -> str:
    if aligned_name.endswith("_aligned"):
        return aligned_name[: -len("_aligned")]
    return aligned_name


def load_candidate_features() -> pd.DataFrame:
    require_file(CANDIDATE_SUMMARY_PATH)
    candidates = pd.read_csv(CANDIDATE_SUMMARY_PATH)
    required = {
        "alpha",
        "aligned_action",
        "final_model_recommendation",
        "final_model_feature",
        "alpha_class",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate summary is missing required columns: {sorted(missing)}")

    selected = candidates.loc[
        candidates["final_model_recommendation"].isin(KEEP_RECOMMENDATIONS)
        & candidates["final_model_feature"].notna()
        & candidates["final_model_feature"].astype(str).str.strip().ne("")
    ].copy()
    selected["final_model_feature"] = selected["final_model_feature"].astype(str).str.strip()
    selected["panel_feature"] = selected["alpha"].astype(str).str.strip()
    selected.loc[selected["panel_feature"].eq(""), "panel_feature"] = selected["final_model_feature"].map(
        original_feature_from_aligned
    )
    selected = selected.drop_duplicates("final_model_feature").reset_index(drop=True)
    if selected.empty:
        raise ValueError("No Strong Keep or Optional Keep candidate features with non-empty final_model_feature were found.")
    return selected


def load_panel(panel_path: Path, candidate_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    header = pd.read_csv(panel_path, nrows=0)
    panel_columns = set(header.columns)
    required = {DATE_COL, TICKER_COL, TARGET_COL}
    missing_required = required - panel_columns
    if missing_required:
        raise ValueError(f"Panel is missing required columns: {sorted(missing_required)}")

    available = candidate_features.loc[candidate_features["panel_feature"].isin(panel_columns)].copy()
    missing_feature_count = int(len(candidate_features) - len(available))
    if available.empty:
        raise ValueError("None of the selected candidate features are present in the panel.")

    usecols = [DATE_COL, TICKER_COL, TARGET_COL] + available["panel_feature"].tolist()
    panel = pd.read_csv(panel_path, usecols=usecols)
    panel[DATE_COL] = pd.to_datetime(panel[DATE_COL], errors="coerce")
    panel[TICKER_COL] = panel[TICKER_COL].astype(str).str.strip().str.upper()
    panel[TARGET_COL] = pd.to_numeric(panel[TARGET_COL], errors="coerce")
    panel = panel.dropna(subset=[DATE_COL, TICKER_COL, TARGET_COL]).sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    return panel, available.reset_index(drop=True), missing_feature_count


def create_aligned_features(panel: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = panel[[DATE_COL, TICKER_COL, TARGET_COL]].copy()
    aligned_feature_names: list[str] = []
    for _, row in features.iterrows():
        panel_feature = row["panel_feature"]
        aligned_feature = row["final_model_feature"]
        values = pd.to_numeric(panel[panel_feature], errors="coerce").fillna(0.0)
        if row["aligned_action"] == "multiply_by_minus_one":
            values = -values
        elif row["aligned_action"] != "keep_as_is":
            raise ValueError(f"Unsupported aligned_action for {panel_feature}: {row['aligned_action']}")
        out[aligned_feature] = values.astype(float)
        aligned_feature_names.append(aligned_feature)
    return out, aligned_feature_names


def split_by_date(df: pd.DataFrame) -> pd.Series:
    unique_dates = pd.Series(df[DATE_COL].dropna().unique()).sort_values().reset_index(drop=True)
    if len(unique_dates) < 10:
        raise ValueError("Not enough unique dates for chronological train/validation/test split.")
    train_end = int(math.floor(0.60 * len(unique_dates)))
    val_end = int(math.floor(0.80 * len(unique_dates)))
    train_dates = set(unique_dates.iloc[:train_end])
    val_dates = set(unique_dates.iloc[train_end:val_end])
    test_dates = set(unique_dates.iloc[val_end:])

    split = pd.Series(index=df.index, dtype="object")
    split.loc[df[DATE_COL].isin(train_dates)] = "train"
    split.loc[df[DATE_COL].isin(val_dates)] = "validation"
    split.loc[df[DATE_COL].isin(test_dates)] = "test"
    if split.isna().any():
        raise ValueError("Some rows were not assigned to a chronological split.")
    return split


def date_based_cv_indices(train_df: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    train_dates = pd.Series(train_df[DATE_COL].dropna().unique()).sort_values().reset_index(drop=True)
    if len(train_dates) < 3:
        raise ValueError("Not enough training dates for time-series cross-validation.")
    splits = min(n_splits, len(train_dates) - 1)
    tss = TimeSeriesSplit(n_splits=splits)
    cv: list[tuple[np.ndarray, np.ndarray]] = []
    date_values = train_df[DATE_COL].reset_index(drop=True)
    for train_date_idx, test_date_idx in tss.split(train_dates):
        fold_train_dates = set(train_dates.iloc[train_date_idx])
        fold_test_dates = set(train_dates.iloc[test_date_idx])
        train_idx = np.flatnonzero(date_values.isin(fold_train_dates).to_numpy())
        test_idx = np.flatnonzero(date_values.isin(fold_test_dates).to_numpy())
        if len(train_idx) and len(test_idx):
            cv.append((train_idx, test_idx))
    if not cv:
        raise ValueError("Failed to create non-empty date-based CV folds.")
    return cv


def train_models(model_df: pd.DataFrame, feature_cols: list[str]) -> tuple[LassoCV, ElasticNetCV]:
    train_df = model_df.loc[model_df["split"].eq("train")].reset_index(drop=True)
    if train_df.empty:
        raise ValueError("Training split is empty.")
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    cv = date_based_cv_indices(train_df, n_splits=5)

    lasso = LassoCV(
        alphas=ALPHAS,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        cv=cv,
    )
    elasticnet = ElasticNetCV(
        l1_ratio=L1_RATIOS,
        alphas=ALPHAS,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        cv=cv,
    )

    print("Fitting LassoCV with date-based TimeSeriesSplit folds...")
    lasso.fit(X_train, y_train)
    print("Fitting ElasticNetCV with date-based TimeSeriesSplit folds...")
    elasticnet.fit(X_train, y_train)
    return lasso, elasticnet


def add_predictions(model_df: pd.DataFrame, feature_cols: list[str], lasso: LassoCV, elasticnet: ElasticNetCV) -> pd.DataFrame:
    X = model_df[feature_cols].to_numpy(dtype=float)
    pred = model_df[[DATE_COL, TICKER_COL, "split", TARGET_COL]].copy()
    pred["lasso_pred_score"] = lasso.predict(X)
    pred["elasticnet_pred_score"] = elasticnet.predict(X)
    return pred


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 2 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="pearson"))


def calculate_daily_rank_ic(split_df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_value, g in split_df.groupby(DATE_COL, sort=True):
        valid = g[[pred_col, TARGET_COL]].dropna()
        n_obs = len(valid)
        if n_obs < MIN_DAILY_OBS:
            continue
        if valid[pred_col].nunique() < 2 or valid[TARGET_COL].nunique() < 2:
            continue
        ic = spearmanr(valid[pred_col], valid[TARGET_COL]).correlation
        if not np.isnan(ic):
            rows.append({"date": date_value, "daily_rank_ic": float(ic), "n_obs": n_obs})
    return pd.DataFrame(rows)


def calculate_long_short_decile(split_df: pd.DataFrame, pred_col: str) -> pd.Series:
    daily_returns: list[float] = []
    for _, g in split_df.groupby(DATE_COL, sort=True):
        valid = g[[pred_col, TARGET_COL]].dropna().sort_values(pred_col)
        n_obs = len(valid)
        if n_obs < MIN_DAILY_OBS:
            continue
        decile_n = int(math.floor(n_obs * 0.10))
        if decile_n < 1:
            continue
        short_ret = valid.iloc[:decile_n][TARGET_COL].mean()
        long_ret = valid.iloc[-decile_n:][TARGET_COL].mean()
        daily_returns.append(float(long_ret - short_ret))
    return pd.Series(daily_returns, dtype=float)


def summarize_performance(predictions: pd.DataFrame, model_name: str, pred_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    perf_rows: list[dict[str, object]] = []
    daily_ic_rows: list[pd.DataFrame] = []
    for split_name in ["train", "validation", "test"]:
        split_df = predictions.loc[predictions["split"].eq(split_name)].copy()
        daily_ic = calculate_daily_rank_ic(split_df, pred_col)
        if not daily_ic.empty:
            daily_ic.insert(0, "split", split_name)
            daily_ic.insert(0, "model_name", model_name)
            daily_ic_rows.append(daily_ic)

        ic_values = daily_ic["daily_rank_ic"] if not daily_ic.empty else pd.Series(dtype=float)
        n_days = len(ic_values)
        mean_rank_ic = float(ic_values.mean()) if n_days else np.nan
        rank_ic_std = float(ic_values.std(ddof=1)) if n_days > 1 else np.nan
        rank_icir = mean_rank_ic / rank_ic_std if rank_ic_std and not np.isnan(rank_ic_std) else np.nan
        rank_ic_t = mean_rank_ic / (rank_ic_std / math.sqrt(n_days)) if rank_ic_std and n_days > 1 else np.nan
        positive_ratio = float((ic_values > 0).mean()) if n_days else np.nan

        ls = calculate_long_short_decile(split_df, pred_col)
        ls_mean = float(ls.mean()) if len(ls) else np.nan
        ls_vol = float(ls.std(ddof=1)) if len(ls) > 1 else np.nan
        ls_sharpe = ls_mean / ls_vol * math.sqrt(252) if ls_vol and not np.isnan(ls_vol) else np.nan

        perf_rows.append(
            {
                "model_name": model_name,
                "split": split_name,
                "n_rows": len(split_df),
                "n_dates": int(split_df[DATE_COL].nunique()),
                "pearson_corr": pearson_corr(split_df[pred_col], split_df[TARGET_COL]),
                "mean_rank_ic": mean_rank_ic,
                "rank_ic_std": rank_ic_std,
                "rank_icir": rank_icir,
                "rank_ic_t_stat": rank_ic_t,
                "positive_ic_ratio": positive_ratio,
                "long_short_mean_daily_return": ls_mean,
                "long_short_daily_vol": ls_vol,
                "long_short_annualized_sharpe": ls_sharpe,
            }
        )

    perf = pd.DataFrame(perf_rows)
    daily = pd.concat(daily_ic_rows, ignore_index=True) if daily_ic_rows else pd.DataFrame(
        columns=["model_name", "split", "date", "daily_rank_ic", "n_obs"]
    )
    return perf, daily


def coefficient_table(
    feature_cols: list[str],
    features: pd.DataFrame,
    lasso: LassoCV,
    elasticnet: ElasticNetCV,
) -> pd.DataFrame:
    meta = features.set_index("final_model_feature")
    rows: list[dict[str, object]] = []
    for model_name, coefs in [
        ("LassoCV", lasso.coef_),
        ("ElasticNetCV", elasticnet.coef_),
    ]:
        for feature, coef in zip(feature_cols, coefs):
            row = meta.loc[feature]
            abs_coef = abs(float(coef))
            rows.append(
                {
                    "model_name": model_name,
                    "feature": feature,
                    "coefficient": float(coef),
                    "abs_coefficient": abs_coef,
                    "selected_flag": bool(abs_coef > COEF_EPS),
                    "original_alpha": row["panel_feature"],
                    "aligned_action": row["aligned_action"],
                    "final_model_recommendation": row["final_model_recommendation"],
                    "alpha_class": row["alpha_class"],
                }
            )
    return pd.DataFrame(rows)


def model_usefulness(perf: pd.DataFrame, model_name: str) -> str:
    test = perf.loc[perf["model_name"].eq(model_name) & perf["split"].eq("test")]
    val = perf.loc[perf["model_name"].eq(model_name) & perf["split"].eq("validation")]
    if test.empty or val.empty:
        return f"{model_name}: not enough performance data to judge usefulness."
    test_row = test.iloc[0]
    val_row = val.iloc[0]
    test_icir = test_row["rank_icir"]
    test_sharpe = test_row["long_short_annualized_sharpe"]
    test_ic = test_row["mean_rank_ic"]
    val_ic = val_row["mean_rank_ic"]
    if pd.notna(test_icir) and pd.notna(test_sharpe) and test_ic > 0 and val_ic > 0 and test_icir > 0.2 and test_sharpe > 0.5:
        return f"{model_name}: looks useful on this first pass, with positive validation/test Rank IC and positive test long-short Sharpe."
    return (
        f"{model_name}: not yet proven effective. Test Rank IC, ICIR, or long-short Sharpe is weak or inconsistent; "
        "treat this as a baseline combination test, not a production signal."
    )


def save_report(
    panel_path: Path,
    features: pd.DataFrame,
    missing_feature_count: int,
    coefficients: pd.DataFrame,
    performance: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    strong_count = int(features["final_model_recommendation"].eq("Strong Keep").sum())
    optional_count = int(features["final_model_recommendation"].eq("Optional Keep").sum())

    lasso_selected = coefficients.loc[coefficients["model_name"].eq("LassoCV") & coefficients["selected_flag"]].sort_values(
        "abs_coefficient", ascending=False
    )
    elastic_selected = coefficients.loc[
        coefficients["model_name"].eq("ElasticNetCV") & coefficients["selected_flag"]
    ].sort_values("abs_coefficient", ascending=False)

    validation_test = performance.loc[performance["split"].isin(["validation", "test"])].copy()

    lasso_note = model_usefulness(performance, "LassoCV")
    elastic_note = model_usefulness(performance, "ElasticNetCV")
    improves_note = (
        "The linear multi-factor model improves the signal only if validation and test Rank IC, ICIR, and long-short Sharpe "
        "are consistently positive relative to the screened single-alpha baseline. If test metrics are weak, the model is not yet proven effective."
    )

    report = f"""Lasso / ElasticNet Multi-Factor Baseline Report: 5-Day Forward Return

Modeling Context
- This script is the next-stage multi-factor linear combination test after single-alpha screening.
- Panel loaded: {panel_path}
- Candidate features used: {len(feature_cols)}
- Strong Keep features: {strong_count}
- Optional Keep features: {optional_count}
- Features skipped because missing from panel: {missing_feature_count}
- Target: ret_fwd_5d
- Features: aligned alpha_z candidates only. Raw alpha columns and alpha_rank columns are excluded.
- Missing selected alpha_z values are filled with 0, the daily cross-sectional z-score mean.
- Chronological split: first 60% train, next 20% validation, final 20% test.
- CV: date-based TimeSeriesSplit within the training period only.

Lasso Selected Features and Coefficients
{lasso_selected[["feature", "coefficient", "original_alpha", "aligned_action", "final_model_recommendation", "alpha_class"]].to_string(index=False) if not lasso_selected.empty else "(none)"}

ElasticNet Selected Features and Coefficients
{elastic_selected[["feature", "coefficient", "original_alpha", "aligned_action", "final_model_recommendation", "alpha_class"]].to_string(index=False) if not elastic_selected.empty else "(none)"}

Train / Validation / Test Performance Summary
{performance.to_string(index=False)}

Validation and Test Focus
{validation_test[["model_name", "split", "mean_rank_ic", "rank_icir", "rank_ic_t_stat", "long_short_annualized_sharpe"]].to_string(index=False)}

Final Recommendation
- {lasso_note}
- {elastic_note}
- {improves_note}
- Use these outputs as a baseline diagnostic. This does not replace the single-alpha regression screen and does not include Review or Drop factors.
"""
    REPORT_OUT.write_text(report, encoding="utf-8")


def save_outputs(
    coefficients: pd.DataFrame,
    performance: pd.DataFrame,
    daily_ic: pd.DataFrame,
    predictions: pd.DataFrame,
    panel_path: Path,
    features: pd.DataFrame,
    missing_feature_count: int,
    feature_cols: list[str],
) -> None:
    coefficients.to_csv(COEFFICIENTS_OUT, index=False)
    performance.to_csv(PERFORMANCE_OUT, index=False)
    daily_ic.to_csv(DAILY_IC_OUT, index=False)
    predictions[[DATE_COL, TICKER_COL, "split", TARGET_COL, "lasso_pred_score", "elasticnet_pred_score"]].to_csv(
        PREDICTIONS_OUT, index=False
    )
    save_report(panel_path, features, missing_feature_count, coefficients, performance, feature_cols)


def main() -> None:
    panel_path = resolve_panel_path()
    candidates = load_candidate_features()
    panel, available_features, missing_feature_count = load_panel(panel_path, candidates)
    model_df, feature_cols = create_aligned_features(panel, available_features)
    model_df["split"] = split_by_date(model_df)

    lasso, elasticnet = train_models(model_df, feature_cols)
    predictions = add_predictions(model_df, feature_cols, lasso, elasticnet)

    lasso_perf, lasso_daily_ic = summarize_performance(predictions, "LassoCV", "lasso_pred_score")
    elastic_perf, elastic_daily_ic = summarize_performance(predictions, "ElasticNetCV", "elasticnet_pred_score")
    performance = pd.concat([lasso_perf, elastic_perf], ignore_index=True)
    daily_ic = pd.concat([lasso_daily_ic, elastic_daily_ic], ignore_index=True)
    coefficients = coefficient_table(feature_cols, available_features, lasso, elasticnet)

    save_outputs(coefficients, performance, daily_ic, predictions, panel_path, available_features, missing_feature_count, feature_cols)

    lasso_selected_count = int(
        coefficients.loc[coefficients["model_name"].eq("LassoCV"), "selected_flag"].sum()
    )
    elastic_selected_count = int(
        coefficients.loc[coefficients["model_name"].eq("ElasticNetCV"), "selected_flag"].sum()
    )

    focus = performance.loc[performance["split"].isin(["validation", "test"]), ["model_name", "split", "rank_icir"]]
    print("\nLasso / ElasticNet 5-day multi-factor baseline complete.")
    print(f"Number of features used: {len(feature_cols)}")
    print(f"Lasso selected feature count: {lasso_selected_count}")
    print(f"ElasticNet selected feature count: {elastic_selected_count}")
    print("Validation/test ICIR:")
    print(focus.to_string(index=False))
    print("Output file paths:")
    for path in [COEFFICIENTS_OUT, PERFORMANCE_OUT, DAILY_IC_OUT, PREDICTIONS_OUT, REPORT_OUT]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
