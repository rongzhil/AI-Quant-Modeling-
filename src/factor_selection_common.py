"""Shared utilities for multi-method alpha factor selection v2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config


TARGET_LABELS_DEFAULT = [
    "forward_5d_excess_return",
    "tradable_forward_5d_return",
    "forward_21d_excess_return",
    "tradable_forward_21d_return",
]

LABEL_COLUMNS = [
    "close_to_close_forward_1d_return",
    "close_to_close_forward_5d_return",
    "close_to_close_forward_21d_return",
    "tradable_forward_1d_return",
    "tradable_forward_5d_return",
    "tradable_forward_21d_return",
    "vwap_forward_1d_return",
    "vwap_forward_5d_return",
    "vwap_forward_21d_return",
    "forward_1d_excess_return",
    "forward_5d_excess_return",
    "forward_21d_excess_return",
]

FEATURE_SET_SUFFIX = {
    "cs_rank": "_cs_rank",
    "cs_zscore": "_cs_zscore",
    "sector_neutral_rank": "_sector_neutral_rank",
}

SPECIAL_ALPHA_IDS = [66, 33, 34, 54, 55, 56, 57, 67, 73]
MIN_VALID_OBS = 1000
MIN_VALID_DATES = 50


@dataclass(frozen=True)
class FeatureSpec:
    """One transformed alpha feature."""

    alpha_id: int
    alpha_name: str
    category: str
    feature_set: str
    feature_column: str


@dataclass(frozen=True)
class DateSplit:
    """Chronological date split."""

    train_dates: list[str]
    validation_dates: list[str]
    test_dates: list[str]

    def summary(self) -> dict[str, str]:
        return {
            "train_start": self.train_dates[0] if self.train_dates else "",
            "train_end": self.train_dates[-1] if self.train_dates else "",
            "validation_start": self.validation_dates[0] if self.validation_dates else "",
            "validation_end": self.validation_dates[-1] if self.validation_dates else "",
            "test_start": self.test_dates[0] if self.test_dates else "",
            "test_end": self.test_dates[-1] if self.test_dates else "",
        }


def alpha_category(alpha_id: int) -> str:
    """Return a coarse category for an alpha id."""
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


def discover_feature_specs(columns: Iterable[str]) -> list[FeatureSpec]:
    """Find transformed alpha features, skipping Alpha 35 by design."""
    column_set = set(columns)
    label_set = set(LABEL_COLUMNS)
    specs: list[FeatureSpec] = []
    for alpha in config.ALPHA_SPECS:
        if alpha.alpha_id == 35:
            continue
        for feature_set, suffix in FEATURE_SET_SUFFIX.items():
            column = f"{alpha.column}{suffix}"
            if column in column_set and column not in label_set:
                specs.append(
                    FeatureSpec(
                        alpha_id=alpha.alpha_id,
                        alpha_name=alpha.name,
                        category=alpha_category(alpha.alpha_id),
                        feature_set=feature_set,
                        feature_column=column,
                    )
                )
    return specs


def load_model_data(path: Path, target_labels: list[str]) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Load model data with only target labels and transformed feature columns."""
    if not path.exists():
        raise FileNotFoundError(f"Input model dataset not found: {path}")
    schema = pd.read_parquet(path, columns=None)
    specs = discover_feature_specs(schema.columns)
    required = ["date", "ticker", *target_labels, *[spec.feature_column for spec in specs]]
    missing = [column for column in ["date", "ticker", *target_labels] if column not in schema.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = schema[required].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["ticker"] = data["ticker"].astype(str).str.strip().str.upper()
    data = data.sort_values(["date", "ticker"]).reset_index(drop=True)
    return data, specs


def chronological_split(dates: pd.Series) -> DateSplit:
    """Split unique dates into 60/20/20 chronological blocks."""
    unique_dates = sorted(pd.Series(dates).dropna().unique().tolist())
    if len(unique_dates) < 10:
        raise ValueError("Not enough unique dates for chronological split")
    train_end = int(len(unique_dates) * 0.60)
    validation_end = int(len(unique_dates) * 0.80)
    return DateSplit(
        train_dates=unique_dates[:train_end],
        validation_dates=unique_dates[train_end:validation_end],
        test_dates=unique_dates[validation_end:],
    )


def date_cv_indices(dates: pd.Series, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build expanding TimeSeriesSplit-style indices by date."""
    unique_dates = np.array(sorted(pd.Series(dates).dropna().unique().tolist()))
    if len(unique_dates) < n_splits + 2:
        n_splits = max(2, len(unique_dates) - 2)
    folds = np.array_split(unique_dates, n_splits + 1)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    date_values = dates.to_numpy()
    for idx in range(n_splits):
        train_dates = np.concatenate(folds[: idx + 1])
        validation_dates = folds[idx + 1]
        train_idx = np.flatnonzero(np.isin(date_values, train_dates))
        validation_idx = np.flatnonzero(np.isin(date_values, validation_dates))
        if len(train_idx) and len(validation_idx):
            splits.append((train_idx, validation_idx))
    return splits


def filter_feature_columns(
    data: pd.DataFrame,
    specs: list[FeatureSpec],
    feature_set: str,
    target_label: str,
) -> list[FeatureSpec]:
    """Keep features with enough target-specific non-missing observations."""
    selected = [spec for spec in specs if spec.feature_set == feature_set and spec.alpha_id != 35]
    usable = data.loc[data[target_label].notna(), ["date", target_label, *[spec.feature_column for spec in selected]]]
    kept: list[FeatureSpec] = []
    for spec in selected:
        values = pd.to_numeric(usable[spec.feature_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        mask = values.notna()
        if int(mask.sum()) < MIN_VALID_OBS:
            continue
        if int(usable.loc[mask, "date"].nunique()) < MIN_VALID_DATES:
            continue
        kept.append(spec)
    return kept


def slice_split(data: pd.DataFrame, split: DateSplit, target_label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Slice train/validation/test frames and drop only target NaNs."""
    train = data[data["date"].isin(split.train_dates)].dropna(subset=[target_label]).copy()
    validation = data[data["date"].isin(split.validation_dates)].dropna(subset=[target_label]).copy()
    test = data[data["date"].isin(split.test_dates)].dropna(subset=[target_label]).copy()
    return train, validation, test


def finite_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace inf values with NaN."""
    return frame.replace([np.inf, -np.inf], np.nan)


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation for finite pairs."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    left = y_true[mask]
    right = y_pred[mask]
    if np.std(left, ddof=1) == 0 or np.std(right, ddof=1) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R2 using numpy."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    y = y_true[mask]
    pred = y_pred[mask]
    denom = float(np.dot(y - y.mean(), y - y.mean()))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.dot(y - pred, y - pred) / denom)


def daily_rank_ic(frame: pd.DataFrame, pred_col: str = "prediction", target_col: str = "target") -> pd.DataFrame:
    """Daily Spearman and Pearson IC from a prediction frame."""
    rows: list[dict[str, float | str | int]] = []
    for date_value, group in frame.groupby("date", sort=True):
        usable = group[[target_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(usable) < 20:
            continue
        target_rank = usable[target_col].rank(method="average")
        pred_rank = usable[pred_col].rank(method="average")
        if target_rank.std(ddof=1) == 0 or pred_rank.std(ddof=1) == 0:
            spearman = float("nan")
        else:
            spearman = float(target_rank.corr(pred_rank))
        pearson = pearson_corr(usable[target_col].to_numpy(dtype=float), usable[pred_col].to_numpy(dtype=float))
        rows.append({"date": date_value, "rank_ic": spearman, "pearson_ic": pearson, "n_obs": int(len(usable))})
    return pd.DataFrame(rows)


def icir(values: pd.Series) -> float:
    """Information coefficient information ratio."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return float("nan")
    std = float(clean.std(ddof=1))
    if std == 0 or not np.isfinite(std):
        return float("nan")
    return float(clean.mean() / std)


def decile_spread(frame: pd.DataFrame, pred_col: str = "prediction", target_col: str = "target") -> float:
    """Mean daily top-bottom decile spread based on predictions."""
    spreads: list[float] = []
    for _, group in frame.groupby("date", sort=True):
        usable = group[[target_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(usable) < 50:
            continue
        ranks = usable[pred_col].rank(method="average", pct=True)
        top = usable.loc[ranks >= 0.9, target_col]
        bottom = usable.loc[ranks <= 0.1, target_col]
        if top.empty or bottom.empty:
            continue
        spreads.append(float(top.mean() - bottom.mean()))
    return float(np.mean(spreads)) if spreads else float("nan")


def prediction_metrics(frame: pd.DataFrame, pred_col: str = "prediction", target_col: str = "target") -> dict[str, float]:
    """Compute aggregate model diagnostics."""
    y_true = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(frame[pred_col], errors="coerce").to_numpy(dtype=float)
    daily_ic = daily_rank_ic(frame, pred_col=pred_col, target_col=target_col)
    return {
        "rank_ic": float(daily_ic["rank_ic"].mean()) if not daily_ic.empty else float("nan"),
        "pearson_ic": pearson_corr(y_true, y_pred),
        "r2": r2_score_np(y_true, y_pred),
        "decile_spread": decile_spread(frame, pred_col=pred_col, target_col=target_col),
        "icir": icir(daily_ic["rank_ic"]) if not daily_ic.empty else float("nan"),
    }


def target_to_rank_by_date(frame: pd.DataFrame, target_label: str) -> np.ndarray:
    """Convert continuous target to within-date percentile ranks for XGBoost ranking."""
    ranked = frame.groupby("date", sort=False)[target_label].rank(method="average", pct=True)
    return ranked.to_numpy(dtype=float)


def group_sizes_by_date(frame: pd.DataFrame) -> np.ndarray:
    """Return XGBoost ranking group sizes in row order sorted by date."""
    return frame.groupby("date", sort=False).size().to_numpy(dtype=np.uint32)


def sign_label(value: float) -> str:
    """Convert numeric sign to text."""
    if not np.isfinite(value) or abs(value) < 1e-15:
        return "mixed"
    return "positive" if value > 0 else "negative"


def sign_consistency(values: Iterable[float]) -> float:
    """Share of non-zero signs agreeing with the majority sign."""
    signs = [1 if value > 0 else -1 if value < 0 else 0 for value in values if np.isfinite(value)]
    signs = [sign for sign in signs if sign != 0]
    if not signs:
        return 0.0
    return float(max(signs.count(1), signs.count(-1)) / len(signs))


def join_unique(values: Iterable[object]) -> str:
    """Join sorted unique non-empty values."""
    unique = sorted({str(value) for value in values if pd.notna(value) and str(value) != ""})
    return ";".join(unique)


def validate_no_label_features(feature_columns: Iterable[str]) -> list[str]:
    """Return accidental label columns in feature list."""
    return sorted(set(feature_columns) & set(LABEL_COLUMNS))


def split_ranges_for_row(split: DateSplit) -> dict[str, str]:
    """Date split ranges as output row fields."""
    return split.summary()
