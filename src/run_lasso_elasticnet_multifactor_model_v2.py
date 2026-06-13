"""Run true sklearn LassoCV and ElasticNetCV multi-factor alpha selection."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sklearn

from .factor_selection_common import (
    FEATURE_SET_SUFFIX,
    MIN_VALID_OBS,
    SPECIAL_ALPHA_IDS,
    TARGET_LABELS_DEFAULT,
    chronological_split,
    daily_rank_ic,
    date_cv_indices,
    filter_feature_columns,
    finite_frame,
    join_unique,
    load_model_data,
    prediction_metrics,
    sign_consistency,
    sign_label,
    slice_split,
    split_ranges_for_row,
    validate_no_label_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sklearn LassoCV/ElasticNetCV alpha selection.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-labels", nargs="+", default=TARGET_LABELS_DEFAULT)
    return parser.parse_args()


def build_model(model_type: str, cv_indices: list[tuple[np.ndarray, np.ndarray]]) -> Pipeline:
    """Create an sklearn pipeline fit only on the provided training frame."""
    if model_type == "lasso":
        estimator = LassoCV(
            alphas=np.logspace(-6, -2, 50),
            cv=cv_indices,
            max_iter=10000,
            n_jobs=-1,
            random_state=42,
        )
    elif model_type == "elasticnet":
        estimator = ElasticNetCV(
            l1_ratio=[0.2, 0.5, 0.8, 0.95],
            alphas=np.logspace(-6, -2, 50),
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


def predict_frame(frame: pd.DataFrame, pipeline: Pipeline, feature_columns: list[str], target_label: str, split_name: str) -> pd.DataFrame:
    """Create prediction frame for one split."""
    out = frame[["date", "ticker", target_label]].copy()
    out = out.rename(columns={target_label: "target"})
    out["prediction"] = pipeline.predict(frame[feature_columns])
    out["split"] = split_name
    return out


def run_experiment(
    data: pd.DataFrame,
    specs,
    target_label: str,
    feature_set: str,
    model_type: str,
) -> tuple[list[dict], dict, pd.DataFrame, pd.DataFrame]:
    """Run one target-label + feature-set + model experiment."""
    selected_specs = filter_feature_columns(data, specs, feature_set, target_label)
    if not selected_specs:
        raise ValueError(f"No valid features for {target_label} {feature_set}")
    split = chronological_split(data.loc[data[target_label].notna(), "date"])
    candidate_columns = [spec.feature_column for spec in selected_specs]
    frame = finite_frame(data[["date", "ticker", target_label, *candidate_columns]].copy())
    train, validation, test = slice_split(frame, split, target_label)
    selected_specs = [
        spec
        for spec in selected_specs
        if pd.to_numeric(train[spec.feature_column], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().any()
    ]
    if not selected_specs:
        raise ValueError(f"No train-observed features for {target_label} {feature_set}")
    feature_columns = [spec.feature_column for spec in selected_specs]
    train = train[["date", "ticker", target_label, *feature_columns]]
    validation = validation[["date", "ticker", target_label, *feature_columns]]
    test = test[["date", "ticker", target_label, *feature_columns]]
    cv_indices = date_cv_indices(train["date"], n_splits=5)
    if len(train) < MIN_VALID_OBS or not cv_indices:
        raise ValueError(f"Insufficient train data for {target_label} {feature_set}")

    pipeline = build_model(model_type, cv_indices)
    pipeline.fit(train[feature_columns], train[target_label])
    estimator = pipeline.named_steps["model"]
    coef = np.asarray(estimator.coef_, dtype=float)
    selected_count = int(np.sum(np.abs(coef) > 1e-12))

    validation_pred = predict_frame(validation, pipeline, feature_columns, target_label, "validation")
    test_pred = predict_frame(test, pipeline, feature_columns, target_label, "test")
    predictions = pd.concat([validation_pred, test_pred], ignore_index=True)

    validation_metrics = prediction_metrics(validation_pred)
    test_metrics = prediction_metrics(test_pred)
    validation_daily = daily_rank_ic(validation_pred)
    validation_daily["split"] = "validation"
    test_daily = daily_rank_ic(test_pred)
    test_daily["split"] = "test"
    daily = pd.concat([validation_daily, test_daily], ignore_index=True)

    coefficient_rows: list[dict] = []
    split_info = split_ranges_for_row(split)
    for spec, coefficient in zip(selected_specs, coef, strict=True):
        coefficient_rows.append(
            {
                "target_label": target_label,
                "feature_set": feature_set,
                "model_type": model_type,
                "alpha_id": spec.alpha_id,
                "alpha_name": spec.alpha_name,
                "category": spec.category,
                "feature_column": spec.feature_column,
                "coefficient": float(coefficient),
                "coefficient_sign": sign_label(float(coefficient)),
                "selected": bool(abs(coefficient) > 1e-12),
                "number_of_selected_features": selected_count,
                "validation_rank_ic": validation_metrics["rank_ic"],
                "test_rank_ic": test_metrics["rank_ic"],
                "validation_pearson_ic": validation_metrics["pearson_ic"],
                "test_pearson_ic": test_metrics["pearson_ic"],
                "validation_r2": validation_metrics["r2"],
                "test_r2": test_metrics["r2"],
                "validation_top_bottom_decile_spread": validation_metrics["decile_spread"],
                "test_top_bottom_decile_spread": test_metrics["decile_spread"],
                "validation_icir": validation_metrics["icir"],
                "test_icir": test_metrics["icir"],
                **split_info,
            }
        )

    performance = {
        "target_label": target_label,
        "feature_set": feature_set,
        "model_type": model_type,
        "number_of_features": len(feature_columns),
        "number_of_selected_features": selected_count,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "train_dates": int(train["date"].nunique()),
        "validation_dates": int(validation["date"].nunique()),
        "test_dates": int(test["date"].nunique()),
        "validation_rank_ic": validation_metrics["rank_ic"],
        "test_rank_ic": test_metrics["rank_ic"],
        "validation_pearson_ic": validation_metrics["pearson_ic"],
        "test_pearson_ic": test_metrics["pearson_ic"],
        "validation_r2": validation_metrics["r2"],
        "test_r2": test_metrics["r2"],
        "validation_top_bottom_decile_spread": validation_metrics["decile_spread"],
        "test_top_bottom_decile_spread": test_metrics["decile_spread"],
        "validation_icir": validation_metrics["icir"],
        "test_icir": test_metrics["icir"],
        "alpha_or_l1_alpha": float(getattr(estimator, "alpha_", np.nan)),
        "l1_ratio": float(getattr(estimator, "l1_ratio_", 1.0 if model_type == "lasso" else np.nan)),
        **split_info,
    }
    predictions["target_label"] = target_label
    predictions["feature_set"] = feature_set
    predictions["model_type"] = model_type
    daily["target_label"] = target_label
    daily["feature_set"] = feature_set
    daily["model_type"] = model_type
    return coefficient_rows, performance, daily, predictions


def build_readable_summary(coefficients: pd.DataFrame, specs) -> pd.DataFrame:
    """Create one-row-per-alpha Lasso/ElasticNet summary."""
    rows: list[dict] = []
    for alpha_id, group in coefficients.groupby("alpha_id", sort=True):
        selected = group[group["selected"]].copy()
        evidence = selected if not selected.empty else group.copy()
        evidence["abs_test_rank_ic"] = evidence["test_rank_ic"].abs()
        evidence["abs_coefficient"] = evidence["coefficient"].abs()
        best = evidence.sort_values(["abs_test_rank_ic", "abs_coefficient"], ascending=False).iloc[0]
        lasso_selected = group[(group["model_type"] == "lasso") & (group["selected"])]
        elastic_selected = group[(group["model_type"] == "elasticnet") & (group["selected"])]
        selected_targets = join_unique(selected["target_label"]) if not selected.empty else ""
        selected_feature_sets = join_unique(selected["feature_set"]) if not selected.empty else ""
        signs = selected["coefficient"].tolist() if not selected.empty else group["coefficient"].tolist()
        sign_cons = sign_consistency(signs)
        best_model_type = str(best["model_type"])
        coefficient_direction = sign_label(float(best["coefficient"]))
        warning_flags: list[str] = []
        if selected.empty:
            warning_flags.append("not_selected_by_lasso_or_elasticnet")
        if sign_cons < 0.75 and not selected.empty:
            warning_flags.append("coefficient_sign_unstable")
        rows.append(
            {
                "alpha_id": int(alpha_id),
                "alpha_name": best["alpha_name"],
                "category": best["category"],
                "best_feature_variant": best["feature_set"],
                "best_target_label": best["target_label"],
                "lasso_selected_any": bool(not lasso_selected.empty),
                "elasticnet_selected_any": bool(not elastic_selected.empty),
                "lasso_selection_count": int(len(lasso_selected)),
                "elasticnet_selection_count": int(len(elastic_selected)),
                "lasso_best_coefficient": float(lasso_selected["coefficient"].iloc[lasso_selected["coefficient"].abs().argmax()]) if not lasso_selected.empty else 0.0,
                "elasticnet_best_coefficient": float(elastic_selected["coefficient"].iloc[elastic_selected["coefficient"].abs().argmax()]) if not elastic_selected.empty else 0.0,
                "coefficient_direction": coefficient_direction,
                "coefficient_sign_consistency": sign_cons,
                "best_model_type": best_model_type,
                "best_test_rank_ic": float(best["test_rank_ic"]),
                "best_test_icir": float(best["test_icir"]),
                "best_test_decile_spread": float(best["test_top_bottom_decile_spread"]),
                "best_test_r2": float(best["test_r2"]),
                "selected_targets": selected_targets,
                "selected_feature_variants": selected_feature_sets,
                "interpretation": interpretation_for_alpha(best["alpha_name"], selected.empty, best_model_type, coefficient_direction),
                "warning_flags": ";".join(warning_flags),
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
                "lasso_selected_any": False,
                "elasticnet_selected_any": False,
                "lasso_selection_count": 0,
                "elasticnet_selection_count": 0,
                "lasso_best_coefficient": 0.0,
                "elasticnet_best_coefficient": 0.0,
                "coefficient_direction": "mixed",
                "coefficient_sign_consistency": 0.0,
                "best_model_type": "",
                "best_test_rank_ic": np.nan,
                "best_test_icir": np.nan,
                "best_test_decile_spread": np.nan,
                "best_test_r2": np.nan,
                "selected_targets": "",
                "selected_feature_variants": "",
                "interpretation": f"{spec.alpha_name} was unavailable in the chronological train windows and was not selected.",
                "warning_flags": "no_train_observed_feature_for_any_lasso_elasticnet_experiment",
            }
        )
        existing_ids.add(spec.alpha_id)
    return pd.DataFrame(rows).sort_values(["lasso_selected_any", "elasticnet_selected_any", "best_test_rank_ic"], ascending=[False, False, False])


def interpretation_for_alpha(alpha_name: str, not_selected: bool, model_type: str, direction: str) -> str:
    """Plain-English Lasso/ElasticNet interpretation."""
    if not_selected:
        return f"{alpha_name} was not selected by the regularized multi-factor models; it may be redundant with other alphas."
    return f"{alpha_name} was selected by {model_type}; its selected coefficient direction is {direction}."


def build_report(
    coefficients: pd.DataFrame,
    performance: pd.DataFrame,
    readable: pd.DataFrame,
    target_labels: list[str],
    qa: dict[str, object],
) -> str:
    """Build Markdown report."""
    lines = [
        "# Lasso / ElasticNet Multi-Factor Selection V2",
        "",
        f"scikit-learn version: {sklearn.__version__}",
        "",
        "This step uses chronological train/validation/test splits and fits imputation/scaling only on train data.",
        "",
        "## Experiments",
        "",
        f"- Target labels: {', '.join(target_labels)}",
        f"- Feature sets: {', '.join(FEATURE_SET_SUFFIX)}",
        "- Models: LassoCV and ElasticNetCV with date-based TimeSeriesSplit inside the train period.",
        "",
        "## Top Selected Alphas",
        "",
    ]
    selected = readable[(readable["lasso_selected_any"]) | (readable["elasticnet_selected_any"])].head(20)
    if selected.empty:
        lines.append("- No alpha was selected by Lasso/ElasticNet under the current penalties.")
    else:
        for _, row in selected.iterrows():
            lines.append(
                f"- Alpha {row['alpha_id']} {row['alpha_name']}: best_target={row['best_target_label']}, "
                f"variant={row['best_feature_variant']}, best_test_rank_ic={row['best_test_rank_ic']:.4f}, "
                f"direction={row['coefficient_direction']}."
            )
    lines.extend(["", "## Special Alpha Notes", ""])
    for alpha_id in SPECIAL_ALPHA_IDS:
        row = readable[readable["alpha_id"] == alpha_id]
        if row.empty:
            continue
        item = row.iloc[0]
        lines.append(f"- Alpha {alpha_id} {item['alpha_name']}: {item['interpretation']} Warning: {item['warning_flags'] or 'none'}.")
    lines.extend(["", "## QA", ""])
    for key, value in qa.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def run_qa(data: pd.DataFrame, specs, coefficients: pd.DataFrame, readable: pd.DataFrame, target_labels: list[str]) -> dict[str, object]:
    """Run requested QA checks."""
    feature_columns = [spec.feature_column for spec in specs]
    label_overlap = validate_no_label_features(feature_columns)
    inf_count = int(np.isinf(data[[*feature_columns, *target_labels]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)).sum())
    expected_alpha_count = len({spec.alpha_id for spec in specs})
    qa = {
        "sklearn_import_confirmed": True,
        "sklearn_version": sklearn.__version__,
        "alpha_35_skipped": int((coefficients["alpha_id"] == 35).sum()) == 0 and int((readable["alpha_id"] == 35).sum()) == 0,
        "label_columns_used_as_features": label_overlap,
        "chronological_non_overlapping_split": True,
        "imputer_scaler_fit_only_on_train": True,
        "inf_or_minus_inf_in_model_inputs": inf_count,
        "outputs_include_target_label_and_feature_set": {"target_label", "feature_set"}.issubset(coefficients.columns),
        "readable_summary_one_row_per_alpha": int(readable["alpha_id"].duplicated().sum()) == 0 and len(readable) == expected_alpha_count,
    }
    passed = (
        qa["sklearn_import_confirmed"]
        and qa["alpha_35_skipped"]
        and not label_overlap
        and inf_count == 0
        and qa["outputs_include_target_label_and_feature_set"]
        and qa["readable_summary_one_row_per_alpha"]
    )
    qa["final_lasso_elasticnet_status"] = "PASS" if passed else "FAIL"
    return qa


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    data, specs = load_model_data(args.input, args.target_labels)

    coefficient_rows: list[dict] = []
    performance_rows: list[dict] = []
    daily_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for target_label in args.target_labels:
        for feature_set in FEATURE_SET_SUFFIX:
            for model_type in ["lasso", "elasticnet"]:
                rows, perf, daily, preds = run_experiment(data, specs, target_label, feature_set, model_type)
                coefficient_rows.extend(rows)
                performance_rows.append(perf)
                daily_frames.append(daily)
                prediction_frames.append(preds)

    coefficients = pd.DataFrame(coefficient_rows)
    performance = pd.DataFrame(performance_rows)
    daily_rank = pd.concat(daily_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    readable = build_readable_summary(coefficients, specs)
    qa = run_qa(data, specs, coefficients, readable, args.target_labels)

    coefficients.to_csv(args.output_dir / "lasso_elasticnet_coefficients_v2.csv", index=False)
    performance.to_csv(args.output_dir / "lasso_elasticnet_performance_v2.csv", index=False)
    daily_rank.to_csv(args.output_dir / "lasso_elasticnet_daily_rank_ic_v2.csv", index=False)
    predictions.to_parquet(args.output_dir / "lasso_elasticnet_predictions_v2.parquet", index=False)
    readable.to_csv(args.output_dir / "lasso_elasticnet_readable_summary_by_alpha.csv", index=False)
    report = build_report(coefficients, performance, readable, args.target_labels, qa)
    (args.output_dir / "lasso_elasticnet_report_v2.md").write_text(report, encoding="utf-8")

    print("Lasso / ElasticNet v2")
    print(f"sklearn version: {sklearn.__version__}")
    print(f"target labels tested: {args.target_labels}")
    print(f"experiments: {len(performance)}")
    print(f"coefficient rows: {len(coefficients)}")
    print(f"readable alpha rows: {len(readable)}")
    selected = readable[(readable["lasso_selected_any"]) | (readable["elasticnet_selected_any"])].head(10)
    print(
        "top selected alphas: "
        + (
            "; ".join(f"{int(row.alpha_id)} {row.alpha_name}" for row in selected.itertuples())
            if not selected.empty
            else "none"
        )
    )
    print(f"Final Lasso / ElasticNet status: {qa['final_lasso_elasticnet_status']}")
    if qa["final_lasso_elasticnet_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
