"""Cross-sectional transforms for the alpha panel.

For each trading date and each raw alpha column, this produces three
standardized views used downstream by the ranker and the Rank-IC evaluation:

  * ``<alpha>_z``                   : winsorized cross-sectional z-score
  * ``<alpha>_rank``                : cross-sectional rank in [0, 1]
  * ``<alpha>_sector_neutral_rank`` : rank in [0, 1] within the GICS sector

Pipeline per date, per alpha:
  1. winsorize the day's values to [WINSOR_LOWER, WINSOR_UPPER] quantiles;
  2. z-score the winsorized values (mean 0, std 1);
  3. rank the winsorized values to [0, 1] across the whole cross-section;
  4. rank within each sector to [0, 1].

Rules:
  * a date with fewer than MIN_CROSS_SECTIONAL_COUNT valid values for an alpha
    is left NaN for that alpha (cross-section too thin to standardize);
  * a (date, sector) group with fewer than MIN_SECTOR_COUNT valid values is
    left NaN for the sector-neutral rank;
  * NaN inputs are excluded from each day's statistics and stay NaN in output.

All operations are within a single date, so there is no look-ahead: today's
transform never uses any other day's data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


Z_SUFFIX = "_z"
RANK_SUFFIX = "_rank"
SECTOR_RANK_SUFFIX = "_sector_neutral_rank"


def _winsorize(s: pd.Series) -> pd.Series:
    """Clip to the configured lower/upper quantiles (NaNs ignored)."""
    valid = s.dropna()
    if valid.empty:
        return s
    lo = valid.quantile(config.WINSOR_LOWER)
    hi = valid.quantile(config.WINSOR_UPPER)
    return s.clip(lower=lo, upper=hi)


def _zscore(s: pd.Series) -> pd.Series:
    """Standardize to mean 0 / std 1 using the day's valid values."""
    mean = s.mean()
    std = s.std()
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mean) / std


def _rank01(s: pd.Series) -> pd.Series:
    """Rank to [0, 1]; NaNs stay NaN. Single valid value -> 0.5."""
    valid_count = s.notna().sum()
    if valid_count == 0:
        return pd.Series(np.nan, index=s.index)
    if valid_count == 1:
        return s.notna().astype(float).replace({0.0: np.nan, 1.0: 0.5})
    # average rank, normalized to [0, 1]
    return (s.rank(method="average") - 1.0) / (valid_count - 1.0)


def _transform_one_date(group: pd.DataFrame, alpha_cols: list[str]) -> pd.DataFrame:
    """Apply winsorize -> z / rank / sector-rank for all alphas on one date."""
    sectors = group["sector"] if "sector" in group.columns else pd.Series("", index=group.index)
    columns: dict[str, pd.Series] = {}

    for col in alpha_cols:
        raw = pd.to_numeric(group[col], errors="coerce")
        valid_count = int(raw.notna().sum())

        # Too few names in the whole cross-section: leave all three NaN.
        if valid_count < config.MIN_CROSS_SECTIONAL_COUNT:
            nan_series = pd.Series(np.nan, index=group.index)
            columns[col + Z_SUFFIX] = nan_series
            columns[col + RANK_SUFFIX] = nan_series
            columns[col + SECTOR_RANK_SUFFIX] = nan_series
            continue

        wins = _winsorize(raw)
        columns[col + Z_SUFFIX] = _zscore(wins)
        columns[col + RANK_SUFFIX] = _rank01(wins)

        # Sector-neutral rank: rank within each sector on the winsorized values.
        sector_rank = pd.Series(np.nan, index=group.index)
        for _, idx in group.groupby(sectors).groups.items():
            sub = wins.loc[idx]
            if int(sub.notna().sum()) >= config.MIN_SECTOR_COUNT:
                sector_rank.loc[idx] = _rank01(sub)
        columns[col + SECTOR_RANK_SUFFIX] = sector_rank

    return pd.DataFrame(columns, index=group.index)


def add_cross_sectional_transforms(panel: pd.DataFrame) -> pd.DataFrame:
    """Add z / rank / sector-neutral-rank columns for every alpha.

    Expects a long panel with columns date, ticker, sector, and the raw alpha
    columns in ``config.ALPHA_COLUMNS``. Returns the panel with the three
    transform columns appended per alpha.
    """
    alpha_cols = [c for c in config.ALPHA_COLUMNS if c in panel.columns]
    missing = [c for c in config.ALPHA_COLUMNS if c not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing alpha columns: {missing}")

    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    transformed_parts = []
    for _, group in panel.groupby("date", sort=True):
        transformed_parts.append(_transform_one_date(group, alpha_cols))
    transforms = pd.concat(transformed_parts).sort_index()

    result = pd.concat([panel, transforms], axis=1)
    return result


def transform_column_names() -> list[str]:
    """All transform column names produced, in stable order."""
    names: list[str] = []
    for col in config.ALPHA_COLUMNS:
        names.append(col + Z_SUFFIX)
        names.append(col + RANK_SUFFIX)
        names.append(col + SECTOR_RANK_SUFFIX)
    return names