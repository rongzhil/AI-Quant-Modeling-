"""Build cross-sectional normalized and ranked S&P 500 alpha panels.

Step 5 consumes the enriched raw alpha panel and creates point-in-time
cross-sectional transforms. Every calculation is performed within a single
date, and sector-neutral calculations are performed within a single date and
sector. The module does not build labels, regressions, models, or backtests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .io_utils import write_csv, write_parquet


METADATA_COLUMNS = ["date", "ticker", *config.METADATA_COLUMNS]
MIN_CROSS_SECTIONAL_COUNT = 50
MIN_SECTOR_COUNT = 5
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99


@dataclass
class AlphaTransformStats:
    """Compact QA metrics for one alpha's transformed columns."""

    alpha_column: str
    missing_rates: dict[str, float]
    dates_below_min_count: int
    sector_date_groups_below_min_count: int


@dataclass
class Step5QA:
    """Terminal QA state for Step 5."""

    failures: list[str] = field(default_factory=list)
    alpha_stats: list[AlphaTransformStats] = field(default_factory=list)
    point_in_time_pass: bool = False
    point_in_time_details: dict[str, Any] = field(default_factory=dict)
    cs_rank_range_pass: bool = False
    sector_rank_range_pass: bool = False

    @property
    def passed(self) -> bool:
        return not self.failures


def parse_bool(value: str | bool) -> bool:
    """Parse flexible true/false command-line values."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def transformed_column_names(alpha_column: str) -> dict[str, str]:
    """Return output column names for one raw alpha column."""
    return {
        "raw": f"{alpha_column}_raw",
        "winsorized": f"{alpha_column}_cs_winsorized",
        "zscore": f"{alpha_column}_cs_zscore",
        "rank": f"{alpha_column}_cs_rank",
        "sector_value": f"{alpha_column}_sector_neutral_value",
        "sector_rank": f"{alpha_column}_sector_neutral_rank",
    }


def _inf_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return int(np.isinf(numeric.to_numpy(dtype=float)).sum())


def load_input_panel(path: Path) -> pd.DataFrame:
    """Load the enriched raw alpha panel."""
    if not path.exists():
        raise FileNotFoundError(f"Input panel not found: {path}")
    panel = pd.read_parquet(path)
    if "date" in panel.columns:
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def validate_input_panel(panel: pd.DataFrame) -> list[str]:
    """Validate the raw enriched panel before cross-sectional transforms."""
    failures: list[str] = []
    missing_metadata = [col for col in METADATA_COLUMNS if col not in panel.columns]
    if missing_metadata:
        failures.append(f"missing metadata columns: {missing_metadata}")
    missing_alpha_cols = [col for col in config.ALPHA_COLUMNS if col not in panel.columns]
    if missing_alpha_cols:
        failures.append(f"missing alpha columns 26-75: {missing_alpha_cols}")
    if {"date", "ticker"}.issubset(panel.columns):
        duplicate_count = int(panel.duplicated(["date", "ticker"]).sum())
        if duplicate_count:
            failures.append(f"input duplicate date-ticker rows: {duplicate_count}")
    available_alpha_cols = [col for col in config.ALPHA_COLUMNS if col in panel.columns]
    if available_alpha_cols:
        inf_count = _inf_count(panel[available_alpha_cols])
        if inf_count:
            failures.append(f"input alpha inf/-inf values: {inf_count}")
    if config.ALPHA_COLUMN_BY_ID[35] in panel.columns and panel[config.ALPHA_COLUMN_BY_ID[35]].notna().any():
        failures.append("alpha_35_industry_momentum is populated; expected dropped/NaN by design")
    return failures


def _transform_one_alpha(panel: pd.DataFrame, alpha_column: str) -> tuple[pd.DataFrame, AlphaTransformStats]:
    """Create raw, winsorized, z-score, rank, and sector-neutral columns for one alpha."""
    names = transformed_column_names(alpha_column)
    raw = pd.to_numeric(panel[alpha_column], errors="coerce")
    by_date = panel["date"]
    by_date_sector = [panel["date"], panel["sector"]]

    valid_count_by_date = raw.groupby(by_date).transform("count")
    lower = raw.groupby(by_date).transform(lambda values: values.quantile(WINSOR_LOWER))
    upper = raw.groupby(by_date).transform(lambda values: values.quantile(WINSOR_UPPER))
    winsorized = raw.clip(lower=lower, upper=upper)
    winsorized = winsorized.where(valid_count_by_date >= MIN_CROSS_SECTIONAL_COUNT)

    same_date_mean = winsorized.groupby(by_date).transform("mean")
    same_date_std = winsorized.groupby(by_date).transform("std")
    zscore = (winsorized - same_date_mean) / same_date_std.replace(0, np.nan)

    cs_rank = raw.groupby(by_date).rank(method="average", pct=True)
    cs_rank = cs_rank.where(valid_count_by_date >= MIN_CROSS_SECTIONAL_COUNT)

    sector_count = raw.groupby(by_date_sector).transform("count")
    sector_mean = raw.groupby(by_date_sector).transform("mean")
    sector_value = (raw - sector_mean).where(sector_count >= MIN_SECTOR_COUNT)
    sector_rank = raw.groupby(by_date_sector).rank(method="average", pct=True)
    sector_rank = sector_rank.where(sector_count >= MIN_SECTOR_COUNT)

    transformed = pd.DataFrame(
        {
            names["raw"]: raw,
            names["winsorized"]: winsorized,
            names["zscore"]: zscore,
            names["rank"]: cs_rank,
            names["sector_value"]: sector_value,
            names["sector_rank"]: sector_rank,
        },
        index=panel.index,
    ).replace([np.inf, -np.inf], np.nan)

    count_by_date = raw.groupby(by_date).count()
    sector_count_by_group = raw.groupby(by_date_sector).count()
    stats = AlphaTransformStats(
        alpha_column=alpha_column,
        missing_rates={col: float(transformed[col].isna().mean()) for col in transformed.columns},
        dates_below_min_count=int((count_by_date < MIN_CROSS_SECTIONAL_COUNT).sum()),
        sector_date_groups_below_min_count=int((sector_count_by_group < MIN_SECTOR_COUNT).sum()),
    )
    return transformed, stats


def build_ranked_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[AlphaTransformStats]]:
    """Build the full cross-sectional ranked panel."""
    output_parts = [panel[METADATA_COLUMNS].copy()]
    stats: list[AlphaTransformStats] = []
    for alpha_column in config.ALPHA_COLUMNS:
        transformed, alpha_stats = _transform_one_alpha(panel, alpha_column)
        output_parts.append(transformed)
        stats.append(alpha_stats)
    output = pd.concat(output_parts, axis=1)
    return output.sort_values(["date", "ticker"]).reset_index(drop=True), stats


def _compare_series(left: pd.Series, right: pd.Series, tolerance: float = 1e-10) -> tuple[float, int]:
    diff = (pd.to_numeric(left, errors="coerce") - pd.to_numeric(right, errors="coerce")).replace([np.inf, -np.inf], np.nan)
    max_abs = float(diff.abs().max(skipna=True)) if diff.notna().any() else 0.0
    mismatches = int((diff.abs() > tolerance).fillna(False).sum())
    return max_abs, mismatches


def point_in_time_alignment_check(input_panel: pd.DataFrame, ranked_panel: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
    """Recompute sampled dates from same-date rows only to verify point-in-time alignment."""
    unique_dates = sorted(input_panel["date"].dropna().unique().tolist())
    if not unique_dates:
        return False, {"reason": "input panel has no dates"}
    sample_positions = sorted(set([0, len(unique_dates) // 2, len(unique_dates) - 1]))
    sample_dates = [unique_dates[pos] for pos in sample_positions]
    sample_alpha_ids = [26, 33, 34, 35, 59, 60, 66, 75]
    sample_alpha_cols = [config.ALPHA_COLUMN_BY_ID[alpha_id] for alpha_id in sample_alpha_ids]

    failures: list[str] = []
    max_abs_diff = 0.0
    mismatched_rows = 0

    for date_value in sample_dates:
        input_subset = input_panel[input_panel["date"] == date_value].copy().sort_values(["date", "ticker"]).reset_index(drop=True)
        ranked_subset = ranked_panel[ranked_panel["date"] == date_value].copy().sort_values(["date", "ticker"]).reset_index(drop=True)
        local_parts = [input_subset[METADATA_COLUMNS].copy()]
        for alpha_col in sample_alpha_cols:
            local_parts.append(_transform_one_alpha(input_subset, alpha_col)[0])
        local_ranked = pd.concat(local_parts, axis=1)

        compare_cols: list[str] = []
        for alpha_col in sample_alpha_cols:
            compare_cols.extend(transformed_column_names(alpha_col).values())

        merged = ranked_subset[["date", "ticker", *compare_cols]].merge(
            local_ranked[["date", "ticker", *compare_cols]],
            on=["date", "ticker"],
            how="left",
            suffixes=("_panel", "_same_date"),
        )
        for col in compare_cols:
            col_max, col_mismatches = _compare_series(merged[f"{col}_panel"], merged[f"{col}_same_date"])
            max_abs_diff = max(max_abs_diff, col_max)
            mismatched_rows += col_mismatches
            if col_mismatches:
                failures.append(f"{date_value} {col}: {col_mismatches} mismatches")

    details = {
        "sample_dates": sample_dates,
        "sample_alpha_ids": sample_alpha_ids,
        "max_abs_diff": max_abs_diff,
        "mismatched_rows": mismatched_rows,
        "failures": failures[:20],
        "notes": [
            "All transforms are recomputed from rows with the same date only.",
            "Sector-neutral transforms are recomputed from rows with the same date and sector only.",
            "This step does not load or forward-fill market, factor, sector, or industry return data.",
        ],
    }
    return len(failures) == 0, details


def _rank_range_pass(panel: pd.DataFrame, suffix: str) -> bool:
    columns = [col for col in panel.columns if col.endswith(suffix)]
    if not columns:
        return False
    values = panel[columns].apply(pd.to_numeric, errors="coerce")
    mask = values.notna()
    if not bool(mask.to_numpy().any()):
        return True
    in_range_or_missing = values.between(0, 1) | values.isna()
    return bool(in_range_or_missing.to_numpy().all())


def run_step5_qa(input_panel: pd.DataFrame, ranked_panel: pd.DataFrame, alpha_stats: list[AlphaTransformStats]) -> Step5QA:
    """Run Step 5 output QA checks."""
    qa = Step5QA(alpha_stats=alpha_stats)
    if len(ranked_panel) != len(input_panel):
        qa.failures.append(f"output row count {len(ranked_panel)} does not equal input row count {len(input_panel)}")
    duplicate_count = int(ranked_panel.duplicated(["date", "ticker"]).sum())
    if duplicate_count:
        qa.failures.append(f"output duplicate date-ticker rows: {duplicate_count}")

    expected_cols = []
    for alpha_col in config.ALPHA_COLUMNS:
        expected_cols.extend(transformed_column_names(alpha_col).values())
    missing_cols = [col for col in expected_cols if col not in ranked_panel.columns]
    if missing_cols:
        qa.failures.append(f"missing transformed alpha columns: {missing_cols[:20]}")

    transform_cols = [col for col in ranked_panel.columns if col.startswith("alpha_")]
    inf_count = _inf_count(ranked_panel[transform_cols]) if transform_cols else 0
    if inf_count:
        qa.failures.append(f"output inf/-inf values: {inf_count}")

    qa.cs_rank_range_pass = _rank_range_pass(ranked_panel, "_cs_rank")
    qa.sector_rank_range_pass = _rank_range_pass(ranked_panel, "_sector_neutral_rank")
    if not qa.cs_rank_range_pass:
        qa.failures.append("one or more cs_rank values are outside [0, 1]")
    if not qa.sector_rank_range_pass:
        qa.failures.append("one or more sector_neutral_rank values are outside [0, 1]")

    alpha_35_names = transformed_column_names(config.ALPHA_COLUMN_BY_ID[35]).values()
    if any(ranked_panel[col].notna().any() for col in alpha_35_names):
        qa.failures.append("Alpha 35 transformed outputs are not fully NaN")

    qa.point_in_time_pass, qa.point_in_time_details = point_in_time_alignment_check(input_panel, ranked_panel)
    if not qa.point_in_time_pass:
        qa.failures.append("point-in-time alignment spot check failed")

    return qa


def print_missing_rates(alpha_stats: list[AlphaTransformStats]) -> None:
    print("missing rate by alpha:")
    for stats in alpha_stats:
        print(f"  {stats.alpha_column}:")
        for col, rate in stats.missing_rates.items():
            print(f"    {col}: {rate:.4f}")


def print_min_count_summaries(alpha_stats: list[AlphaTransformStats]) -> None:
    print("dates with fewer than 50 valid stocks by alpha:")
    for stats in alpha_stats:
        print(f"  {stats.alpha_column}: {stats.dates_below_min_count}")
    print("sector/date groups with fewer than 5 valid stocks by alpha:")
    for stats in alpha_stats:
        print(f"  {stats.alpha_column}: {stats.sector_date_groups_below_min_count}")


def print_step5_summary(input_panel: pd.DataFrame, ranked_panel: pd.DataFrame, qa: Step5QA, output: Path, csv_path: Path | None) -> None:
    """Print the final terminal summary requested for Step 5."""
    alpha_cols = [col for col in ranked_panel.columns if col.startswith("alpha_")]
    duplicate_count = int(ranked_panel.duplicated(["date", "ticker"]).sum()) if {"date", "ticker"}.issubset(ranked_panel.columns) else 0
    inf_count = _inf_count(ranked_panel[alpha_cols]) if alpha_cols else 0
    print("Step 5 Cross-Sectional Alpha Panel QA")
    print(f"input rows: {len(input_panel)}")
    print(f"output rows: {len(ranked_panel)}")
    print(f"columns: {ranked_panel.shape[1]}")
    print(f"unique tickers: {ranked_panel['ticker'].nunique() if 'ticker' in ranked_panel.columns else 0}")
    print(f"date range: {ranked_panel['date'].min() if 'date' in ranked_panel.columns else None} to {ranked_panel['date'].max() if 'date' in ranked_panel.columns else None}")
    print(f"duplicate date-ticker rows: {duplicate_count}")
    print(f"alpha count: {len(config.ALPHA_COLUMNS)}")
    print(f"inf / -inf count: {inf_count}")
    print(f"cs_rank range PASS/FAIL: {'PASS' if qa.cs_rank_range_pass else 'FAIL'}")
    print(f"sector_neutral_rank range PASS/FAIL: {'PASS' if qa.sector_rank_range_pass else 'FAIL'}")
    print(f"point-in-time alignment PASS/FAIL: {'PASS' if qa.point_in_time_pass else 'FAIL'}")
    print(f"point-in-time details: {qa.point_in_time_details}")
    print_missing_rates(qa.alpha_stats)
    print_min_count_summaries(qa.alpha_stats)
    print(f"output parquet: {output}")
    if csv_path is not None:
        print(f"output csv: {csv_path}")
    print(f"final Step 5 status PASS/FAIL: {'PASS' if qa.passed else 'FAIL'}")
    if qa.failures:
        print("failed checks:")
        for failure in qa.failures:
            print(f"  {failure}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build cross-sectional ranked S&P 500 alpha panel.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-csv", type=parse_bool, default=False)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_panel = load_input_panel(args.input)
    failures = validate_input_panel(input_panel)
    if failures:
        print("Input validation failed:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)

    ranked_panel, alpha_stats = build_ranked_panel(input_panel)
    qa = run_step5_qa(input_panel, ranked_panel, alpha_stats)
    csv_path = args.output.with_suffix(".csv") if args.export_csv else None
    if qa.passed:
        write_parquet(ranked_panel, args.output)
        if csv_path is not None:
            write_csv(ranked_panel, csv_path)
    print_step5_summary(input_panel, ranked_panel, qa, args.output, csv_path)
    if not qa.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
