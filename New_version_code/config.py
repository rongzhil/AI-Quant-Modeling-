"""Single source of truth for the S&P 500 price-volume alpha pipeline.

Everything that another module might need to agree on lives here: filesystem
paths, the trading universe, the backtest window, the train/val/test split,
and the definition of every alpha (id, human name, output column name, and
whether it depends on auxiliary data).

Edit paths and constants here only. No formula logic belongs in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
# Root of all input data. Note the space in "Quant Strat"; Path handles it.
DATA_ROOT = Path("/Users/qq/Desktop/Icarus/Quant Strat/data")

# Raw inputs.
INPUT_BATCH_DIRS = [
    DATA_ROOT / "batch_001",
    DATA_ROOT / "batch_002",
    DATA_ROOT / "batch_003",
]
GICS_MAPPING_PATH = DATA_ROOT / "sectors" / "SPX Sector_Industry Mapping.xlsx"
FLOAT_PATH = DATA_ROOT / "Equity Float.xlsx"
FACTOR_PATH = DATA_ROOT / "factors_carhart4_2023-05_2026-04.csv"
SPX_PATH = DATA_ROOT / "SPX_high_low_close.xlsx"

# All generated artifacts (final dataset + reports) go here.
OUTPUT_DIR = DATA_ROOT / "output"

# Cache of the aggregated all-market daily price panel (slow to rebuild from
# ~503 minute files, so it is written once and reused).
DAILY_PRICE_CACHE_PATH = OUTPUT_DIR / "daily_prices.parquet"

# Final dataset and the main reports.
ALPHA_PANEL_RAW_PATH = OUTPUT_DIR / "alpha_panel_raw.parquet"        # 49 raw alphas
ALPHA_PANEL_FINAL_PATH = OUTPUT_DIR / "alpha_panel_final.parquet"    # + cross-section + target
RANK_IC_REPORT_PATH = OUTPUT_DIR / "rank_ic_summary.csv"
SELECTED_FEATURES_PATH = OUTPUT_DIR / "selected_features.csv"
SELECTION_REPORT_PATH = OUTPUT_DIR / "feature_selection_report.txt"


# --------------------------------------------------------------------------- #
# Universe / data conventions
# --------------------------------------------------------------------------- #
# Minute bars are aggregated to daily using the regular US session only.
DEFAULT_TIMEZONE = "US/Eastern"
REGULAR_SESSION_OPEN = "09:30"
REGULAR_SESSION_CLOSE = "16:00"

# Float shares from Bloomberg are quoted in millions of shares.
FLOAT_UNIT_MULTIPLIER = 1_000_000.0

# Canonicalize share-class ticker variants (Bloomberg "/" vs panel "."), so
# BRK/B and BF/B match across files. Extend without touching formula code.
TICKER_ALIASES = {
    "BRK/B": "BRK.B",
    "BRK-B": "BRK.B",
    "BRK B": "BRK.B",
    "BF/B": "BF.B",
    "BF-B": "BF.B",
    "BF B": "BF.B",
}

# Metadata columns carried alongside every row.
METADATA_COLUMNS = ["company_name", "sector", "industry", "sub_industry"]


# ---------------------------------------------------------------------------
# Date boundaries
# ---------------------------------------------------------------------------
# Data start including warmup. Long-window alphas (e.g. 33, 60) need ~1-2 years
# of history before their first valid value. Once earlier price data is scraped,
# change ONLY this line (e.g. to "2021-05-01") and the pipeline will use the
# extra history for alpha computation without any other code change.
WARMUP_START_DATE = "2023-05-01"

# Evaluation / backtest window.
START_DATE = "2023-05-01"
END_DATE = "2026-04-30"

# Chronological split. Feature selection and (later) ranker training reference
# these exact dates so train/test stay consistent across the pipeline.
#   train : TRAIN_START_DATE .. TRAIN_END_DATE        (60%)
#   test  : TEST_START_DATE  .. TEST_END_DATE         (remaining 40% for now)
# Validation is not split out yet (data is currently too scarce); the params are
# kept as placeholders. When there is enough data, set VAL_START/END_DATE and
# move TEST_START_DATE to the day after VAL_END_DATE.
TRAIN_START_DATE = "2023-05-01"
TRAIN_END_DATE = "2025-10-09"
VAL_START_DATE = None            # placeholder; not used yet
VAL_END_DATE = None              # placeholder; not used yet
TEST_START_DATE = "2025-10-10"   # test directly follows train for now
TEST_END_DATE = "2026-04-30"

# Days to drop between train and the next segment so a 5-day forward-return label
# in train cannot peek into test. 0 for now; set to FORWARD_RETURN_HORIZON to be
# strict once val/test matter for final evaluation.
EMBARGO_DAYS = 0

FORWARD_RETURN_HORIZON = 5
TARGET_COLUMN = "ret_fwd_5d"


# --------------------------------------------------------------------------- #
# Cross-sectional and feature-selection thresholds
# --------------------------------------------------------------------------- #
MIN_CROSS_SECTIONAL_COUNT = 50   # min valid stocks per date for CS transforms
MIN_SECTOR_COUNT = 5             # min stocks per (date, sector) for neutralization
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

RANK_IC_T_STAT_THRESHOLD = 2.0   # drop alphas with |IC t-stat| below this
REDUNDANCY_CORR_THRESHOLD = 0.80 # |rank-corr| at/above this clusters as redundant


# --------------------------------------------------------------------------- #
# Alpha definitions (alpha 35 industry momentum removed by design)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AlphaSpec:
    """One raw price/volume alpha: id, human name, output column name."""

    alpha_id: int
    name: str
    column: str


ALPHA_SPECS = [
    AlphaSpec(26, "1-day return", "alpha_26_1_day_return"),
    AlphaSpec(27, "5-day return", "alpha_27_5_day_return"),
    AlphaSpec(28, "21-day return", "alpha_28_21_day_return"),
    AlphaSpec(29, "63-day return", "alpha_29_63_day_return"),
    AlphaSpec(30, "126-day return", "alpha_30_126_day_return"),
    AlphaSpec(31, "252-day return", "alpha_31_252_day_return"),
    AlphaSpec(32, "12-1 month momentum", "alpha_32_12_1_month_momentum"),
    AlphaSpec(33, "Residual momentum", "alpha_33_residual_momentum"),
    AlphaSpec(34, "Sector-neutral momentum", "alpha_34_sector_neutral_momentum"),
    # alpha 35 (industry momentum) intentionally removed: industry peer groups
    # are too small for a stable point-in-time median.
    AlphaSpec(36, "52-week high proximity", "alpha_36_52_week_high_proximity"),
    AlphaSpec(37, "20-day breakout", "alpha_37_20_day_breakout"),
    AlphaSpec(38, "20-day breakdown", "alpha_38_20_day_breakdown"),
    AlphaSpec(39, "1-day reversal", "alpha_39_1_day_reversal"),
    AlphaSpec(40, "5-day reversal", "alpha_40_5_day_reversal"),
    AlphaSpec(41, "21-day reversal", "alpha_41_21_day_reversal"),
    AlphaSpec(42, "Overnight return", "alpha_42_overnight_return"),
    AlphaSpec(43, "Intraday return", "alpha_43_intraday_return"),
    AlphaSpec(44, "Overnight-intraday reversal", "alpha_44_overnight_intraday_reversal"),
    AlphaSpec(45, "Gap-fill signal", "alpha_45_gap_fill_signal"),
    AlphaSpec(46, "VWAP deviation reversion", "alpha_46_vwap_deviation_reversion"),
    AlphaSpec(47, "20-day moving average distance", "alpha_47_20_day_moving_average_distance"),
    AlphaSpec(48, "MA20-MA60 crossover", "alpha_48_ma20_ma60_crossover"),
    AlphaSpec(49, "MA20 slope", "alpha_49_ma20_slope"),
    AlphaSpec(50, "RSI reversion score", "alpha_50_rsi_reversion_score"),
    AlphaSpec(51, "MACD normalized", "alpha_51_macd_normalized"),
    AlphaSpec(52, "Bollinger z-score", "alpha_52_bollinger_z_score"),
    AlphaSpec(53, "Bollinger reversion", "alpha_53_bollinger_reversion"),
    AlphaSpec(54, "Daily high-low range", "alpha_54_daily_high_low_range"),
    AlphaSpec(55, "Average true range", "alpha_55_average_true_range"),
    AlphaSpec(56, "21-day realized volatility", "alpha_56_21_day_realized_volatility"),
    AlphaSpec(57, "63-day realized volatility", "alpha_57_63_day_realized_volatility"),
    AlphaSpec(58, "Downside volatility", "alpha_58_downside_volatility"),
    AlphaSpec(59, "Idiosyncratic volatility", "alpha_59_idiosyncratic_volatility"),
    AlphaSpec(60, "Market beta", "alpha_60_market_beta"),
    AlphaSpec(61, "Return skewness", "alpha_61_return_skewness"),
    AlphaSpec(62, "Return kurtosis", "alpha_62_return_kurtosis"),
    AlphaSpec(63, "63-day max drawdown", "alpha_63_63_day_max_drawdown"),
    AlphaSpec(64, "Abnormal volume", "alpha_64_abnormal_volume"),
    AlphaSpec(65, "Dollar volume", "alpha_65_dollar_volume"),
    AlphaSpec(66, "Turnover", "alpha_66_turnover"),
    AlphaSpec(67, "Amihud illiquidity", "alpha_67_amihud_illiquidity"),
    AlphaSpec(68, "ADV trend", "alpha_68_adv_trend"),
    AlphaSpec(69, "Price-volume correlation", "alpha_69_price_volume_correlation"),
    AlphaSpec(70, "On-balance volume change", "alpha_70_on_balance_volume_change"),
    AlphaSpec(71, "Accumulation/distribution slope", "alpha_71_accumulation_distribution_slope"),
    AlphaSpec(72, "Money flow index", "alpha_72_money_flow_index"),
    AlphaSpec(73, "Chaikin money flow", "alpha_73_chaikin_money_flow"),
    AlphaSpec(74, "Volume-confirmed momentum", "alpha_74_volume_confirmed_momentum"),
    AlphaSpec(75, "Low-volatility momentum", "alpha_75_low_volatility_momentum"),
]

# Derived lookups used across the pipeline.
ALPHA_COLUMNS = [spec.column for spec in ALPHA_SPECS]
ALPHA_IDS = [spec.alpha_id for spec in ALPHA_SPECS]
ALPHA_COLUMN_BY_ID = {spec.alpha_id: spec.column for spec in ALPHA_SPECS}
ALPHA_SPEC_BY_ID = {spec.alpha_id: spec for spec in ALPHA_SPECS}

# Standard column order for the raw alpha panel.
OUTPUT_COLUMNS = ["date", "ticker", *METADATA_COLUMNS, *ALPHA_COLUMNS]


def column(alpha_id: int) -> str:
    """Return the output column name for an alpha id."""
    return ALPHA_COLUMN_BY_ID[alpha_id]