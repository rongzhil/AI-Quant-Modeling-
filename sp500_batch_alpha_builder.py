"""Command-line entry point for S&P 500 batch alpha panel construction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config
from .alpha_features import compute_alpha_features
from .io_utils import (
    InputFileDescriptor,
    aggregate_minute_frame,
    find_input_files,
    inspect_company_file,
    load_auxiliary_data,
    load_raw_company_file,
    load_universe,
    read_table,
    standardize_daily_frame,
    validate_daily_ohlcv,
    write_csv,
    write_parquet,
)
from .qa_checks import BatchQAResult, MergeQAResult, print_batch_qa, print_merge_qa, run_batch_qa, run_merge_qa


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = value.strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def _metadata_for_ticker(universe: pd.DataFrame, ticker: str) -> dict[str, str]:
    row = universe[universe["ticker"] == ticker]
    if row.empty:
        return {col: "" for col in config.METADATA_COLUMNS}
    return {col: str(row.iloc[0].get(col, "") or "") for col in config.METADATA_COLUMNS}


def _empty_panel() -> pd.DataFrame:
    return pd.DataFrame(columns=config.OUTPUT_COLUMNS)


def _load_files_grouped_by_ticker(
    input_files: list[Path],
    universe_tickers: set[str],
) -> tuple[dict[str, list[InputFileDescriptor]], dict[str, str]]:
    grouped: dict[str, list[InputFileDescriptor]] = {}
    skipped: dict[str, str] = {}
    for file_path in input_files:
        try:
            descriptor = inspect_company_file(file_path, universe_tickers=universe_tickers)
            grouped.setdefault(descriptor.ticker, []).append(descriptor)
        except Exception as exc:
            skipped[file_path.stem] = str(exc)
    return grouped, skipped


def process_folder(
    universe_path: Path,
    input_dir: Path,
    batch_id: str,
    output_dir: Path,
    auxiliary_dir: Path,
    force: bool = False,
) -> tuple[pd.DataFrame, BatchQAResult]:
    """Process all valid ticker files in an input folder into one batch panel."""
    universe = load_universe(universe_path)
    universe_tickers = set(universe["ticker"])
    aux_data = load_auxiliary_data(auxiliary_dir)
    input_files = find_input_files(input_dir)

    grouped_files, skipped = _load_files_grouped_by_ticker(input_files, universe_tickers)
    panels: list[pd.DataFrame] = []
    daily_by_ticker: dict[str, pd.DataFrame] = {}
    ticker_input_stats: dict[str, dict[str, object]] = {}

    for ticker, loaded_files in sorted(grouped_files.items()):
        try:
            if ticker not in universe_tickers:
                skipped[ticker] = "ticker not found in universe file"
                continue

            input_types = sorted({item.input_type for item in loaded_files})
            if len(input_types) != 1:
                skipped[ticker] = f"mixed input types for same ticker: {input_types}"
                continue
            input_type = input_types[0]
            full_files = [load_raw_company_file(item.source_path, universe_tickers=universe_tickers) for item in loaded_files]
            full_tickers = sorted({item.ticker for item in full_files})
            full_input_types = sorted({item.input_type for item in full_files})
            if full_tickers != [ticker]:
                skipped[ticker] = f"ticker inference changed after full load: {full_tickers}"
                continue
            if full_input_types != [input_type]:
                skipped[ticker] = f"input type changed after full load: {full_input_types}"
                continue
            combined_raw = pd.concat([item.frame for item in full_files], ignore_index=True)

            if input_type == "minute":
                daily, warnings, duplicate_timestamp_count = aggregate_minute_frame(combined_raw, ticker)
            else:
                daily, warnings = standardize_daily_frame(combined_raw, ticker)
                duplicate_timestamp_count = None

            duplicate_daily_date_count = int(daily["date"].duplicated().sum()) if "date" in daily.columns else 0
            ticker_input_stats[ticker] = {
                "input_type": input_type,
                "file_count": len(loaded_files),
                "duplicate_timestamp_count": duplicate_timestamp_count,
                "duplicate_daily_date_count": duplicate_daily_date_count,
                "daily_row_count": len(daily),
            }

            for warning in warnings:
                print(f"{ticker}: {warning}")

            ok, reasons = validate_daily_ohlcv(daily)
            if not ok:
                skipped[ticker] = "; ".join(reasons)
                continue

            metadata = _metadata_for_ticker(universe, ticker)
            alphas = compute_alpha_features(daily, metadata=metadata, aux_data=aux_data)
            panel = daily[["date", "ticker"]].merge(alphas, on="date", how="left")
            for col in config.METADATA_COLUMNS:
                panel[col] = metadata.get(col, "")
            panel = panel[config.OUTPUT_COLUMNS]
            panels.append(panel)
            daily_by_ticker[ticker] = daily
        except Exception as exc:
            skipped[ticker] = str(exc)

    batch_panel = pd.concat(panels, ignore_index=True) if panels else _empty_panel()
    if not batch_panel.empty:
        batch_panel = batch_panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    qa = run_batch_qa(batch_panel, daily_by_ticker) if panels else BatchQAResult(
        passed=False,
        failures=["no valid tickers processed"],
        duplicate_date_ticker_count=0,
        alpha_column_count=len([col for col in batch_panel.columns if col.startswith("alpha_")]),
        inf_count=0,
        missing_rates={col: float("nan") for col in config.ALPHA_COLUMNS},
        formula_pass=False,
        leakage_pass=False,
    )

    output_path = output_dir / f"{batch_id}_alpha_panel.parquet"
    wrote_output = False
    if qa.passed or force:
        write_parquet(batch_panel, output_path)
        wrote_output = True

    print_batch_qa(
        batch_id=batch_id,
        input_dir=input_dir,
        files_found=len(input_files),
        tickers_processed=len(panels),
        skipped=skipped,
        ticker_input_stats=ticker_input_stats,
        output_path=output_path,
        panel=batch_panel,
        qa=qa,
        wrote_output=wrote_output,
    )
    if not qa.passed and not force:
        print("Output was not written because QA failed. Rerun with --force true only if you intentionally accept the failures.")
    return batch_panel, qa


def _read_batch_panels(batch_dir: Path) -> tuple[list[Path], list[pd.DataFrame]]:
    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch directory does not exist: {batch_dir}")
    batch_files = sorted(batch_dir.glob("*_alpha_panel.parquet"))
    panels = [read_table(path) for path in batch_files]
    return batch_files, panels


def _drop_exact_duplicate_rows_or_find_conflicts(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    if panel.empty or not panel.duplicated(["date", "ticker"]).any():
        return panel, []
    conflict_keys: list[tuple[str, str]] = []
    for (date, ticker), group in panel[panel.duplicated(["date", "ticker"], keep=False)].groupby(["date", "ticker"]):
        if len(group.drop_duplicates()) > 1:
            conflict_keys.append((str(date), str(ticker)))
    if conflict_keys:
        return panel, conflict_keys
    return panel.drop_duplicates().reset_index(drop=True), []


def merge_batches(batch_dir: Path, output: Path, export_csv: bool = False) -> tuple[pd.DataFrame, MergeQAResult]:
    """Merge all existing batch alpha panels into one final S&P 500 panel."""
    batch_files, panels = _read_batch_panels(batch_dir)
    if not panels:
        empty = _empty_panel()
        qa = MergeQAResult(passed=False, failures=["no *_alpha_panel.parquet files found"])
        print_merge_qa(0, output, empty, qa, wrote_output=False, csv_output_path=output.with_suffix(".csv") if export_csv else None)
        return empty, qa

    merged = pd.concat(panels, ignore_index=True)
    merged, conflicts = _drop_exact_duplicate_rows_or_find_conflicts(merged)
    if conflicts:
        preview = ", ".join(f"{ticker}@{date}" for date, ticker in conflicts[:20])
        qa = MergeQAResult(passed=False, failures=[f"conflicting duplicate date-ticker rows: {preview}"])
        print_merge_qa(
            len(batch_files),
            output,
            merged,
            qa,
            wrote_output=False,
            csv_output_path=output.with_suffix(".csv") if export_csv else None,
        )
        return merged, qa

    missing_output_columns = [col for col in config.OUTPUT_COLUMNS if col not in merged.columns]
    if missing_output_columns:
        qa = MergeQAResult(passed=False, failures=[f"missing expected columns before merge output: {missing_output_columns}"])
        print_merge_qa(
            len(batch_files),
            output,
            merged,
            qa,
            wrote_output=False,
            csv_output_path=output.with_suffix(".csv") if export_csv else None,
        )
        return merged, qa

    merged = merged[config.OUTPUT_COLUMNS].sort_values(["date", "ticker"]).reset_index(drop=True)
    qa = run_merge_qa(merged)
    wrote_output = False
    csv_output_path = output.with_suffix(".csv") if export_csv else None
    if qa.passed:
        write_parquet(merged, output)
        if export_csv and csv_output_path is not None:
            write_csv(merged, csv_output_path)
        wrote_output = True

    print_merge_qa(len(batch_files), output, merged, qa, wrote_output=wrote_output, csv_output_path=csv_output_path)
    if not qa.passed:
        print("Merged output was not written because QA failed.")
    return merged, qa


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build S&P 500 daily price/volume alpha panels from ticker batches.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process-folder", help="Process all valid ticker files in one input folder.")
    process.add_argument("--universe", type=Path, default=config.DEFAULT_UNIVERSE_PATH)
    process.add_argument("--input-dir", type=Path, required=True)
    process.add_argument("--batch-id", required=True)
    process.add_argument("--output-dir", type=Path, default=config.DEFAULT_BATCH_OUTPUT_DIR)
    process.add_argument("--auxiliary-dir", type=Path, default=config.DEFAULT_AUXILIARY_DIR)
    process.add_argument("--force", type=parse_bool, default=False)

    merge = subparsers.add_parser("merge-batches", help="Merge all *_alpha_panel.parquet files into one raw panel.")
    merge.add_argument("--batch-dir", type=Path, default=config.DEFAULT_BATCH_OUTPUT_DIR)
    merge.add_argument("--output", type=Path, default=config.DEFAULT_MERGED_PANEL_PATH)
    merge.add_argument("--export-csv", type=parse_bool, default=False)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "process-folder":
        process_folder(
            universe_path=args.universe,
            input_dir=args.input_dir,
            batch_id=args.batch_id,
            output_dir=args.output_dir,
            auxiliary_dir=args.auxiliary_dir,
            force=args.force,
        )
    elif args.command == "merge-batches":
        merge_batches(batch_dir=args.batch_dir, output=args.output, export_csv=args.export_csv)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
