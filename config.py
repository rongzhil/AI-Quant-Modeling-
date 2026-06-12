"""Configuration constants for the portable S&P 500 alpha pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_UNIVERSE_PATH = Path("data/raw/auxiliary/sp500_tickers.csv")
DEFAULT_AUXILIARY_DIR = Path("data/raw/auxiliary")
DEFAULT_INPUT_BATCH_ROOT = Path("data/raw/input_batches")
DEFAULT_BATCH_OUTPUT_DIR = Path("data/processed/alpha_batches")
DEFAULT_MERGED_PANEL_PATH = Path("data/processed/sp500_alpha_panel_raw.parquet")

DEFAULT_TIMEZONE = "US/Eastern"
REGULAR_SESSION_OPEN = "09:30"
REGULAR_SESSION_CLOSE = "16:00"
MIN_DAILY_ROWS = 300

# Canonicalize common S&P 500 share-class file-name variants. Users may edit
# or extend this dictionary without touching the formula code.
DEFAULT_TICKER_ALIASES = {
    "BRK-B": "BRK.B",
    "BRK/B": "BRK.B",
    "BRK B": "BRK.B",
    "BF-B": "BF.B",
    "BF/B": "BF.B",
    "BF B": "BF.B",
}

METADATA_COLUMNS = ["company_name", "sector", "industry", "sub_industry"]


@dataclass(frozen=True)
class AlphaSpec:
    """Name and output column for one raw price/volume alpha."""

    alpha_id: int
    name: str
    column: str
    auxiliary_dependent: bool = False


ALPHA_SPECS = [
    AlphaSpec(26, "1-day return", "alpha_26_1_day_return"),
    AlphaSpec(27, "5-day return", "alpha_27_5_day_return"),
    AlphaSpec(28, "21-day return", "alpha_28_21_day_return"),
    AlphaSpec(29, "63-day return", "alpha_29_63_day_return"),
    AlphaSpec(30, "126-day return", "alpha_30_126_day_return"),
    AlphaSpec(31, "252-day return", "alpha_31_252_day_return"),
    AlphaSpec(32, "12-1 month momentum", "alpha_32_12_1_month_momentum"),
    AlphaSpec(33, "Residual momentum", "alpha_33_residual_momentum", True),
    AlphaSpec(34, "Sector-neutral momentum", "alpha_34_sector_neutral_momentum", True),
    AlphaSpec(35, "Industry momentum", "alpha_35_industry_momentum", True),
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
    AlphaSpec(59, "Idiosyncratic volatility", "alpha_59_idiosyncratic_volatility", True),
    AlphaSpec(60, "Market beta", "alpha_60_market_beta", True),
    AlphaSpec(61, "Return skewness", "alpha_61_return_skewness"),
    AlphaSpec(62, "Return kurtosis", "alpha_62_return_kurtosis"),
    AlphaSpec(63, "63-day max drawdown", "alpha_63_63_day_max_drawdown"),
    AlphaSpec(64, "Abnormal volume", "alpha_64_abnormal_volume"),
    AlphaSpec(65, "Dollar volume", "alpha_65_dollar_volume"),
    AlphaSpec(66, "Turnover", "alpha_66_turnover", True),
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

ALPHA_COLUMNS = [spec.column for spec in ALPHA_SPECS]
ALPHA_IDS = [spec.alpha_id for spec in ALPHA_SPECS]
ALPHA_COLUMN_BY_ID = {spec.alpha_id: spec.column for spec in ALPHA_SPECS}
OPTIONAL_ALPHA_IDS = [spec.alpha_id for spec in ALPHA_SPECS if spec.auxiliary_dependent]

OUTPUT_COLUMNS = ["date", "ticker", *METADATA_COLUMNS, *ALPHA_COLUMNS]

