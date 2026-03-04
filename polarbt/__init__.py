"""
PolarBT - A lightweight backtesting library for evolutionary strategy search.

Features:
- Polars-based for high performance
- Event-driven architecture with vectorized preprocessing
- Support for fractional and multi-asset positions
- ML model integration ready
- Parallel execution for evolutionary optimization
"""

from polarbt import analysis, data, indicators, metrics
from polarbt.analysis import (
    LookAheadResult,
    MonteCarloResult,
    PermutationTestResult,
    detect_look_ahead_bias,
    monte_carlo,
    permutation_test,
)
from polarbt.commissions import (
    CommissionModel,
    CustomCommission,
    FixedPlusPercentCommission,
    MakerTakerCommission,
    PercentCommission,
    TieredCommission,
)
from polarbt.core import (
    BacktestContext,
    Engine,
    Portfolio,
    Strategy,
    merge_asset_dataframes,
    standardize_dataframe,
)
from polarbt.metrics import format_results
from polarbt.orders import Order, OrderStatus, OrderType
from polarbt.results import BacktestMetrics, TradeStats
from polarbt.runner import backtest, backtest_batch, optimize, optimize_bayesian, optimize_multi, walk_forward_analysis
from polarbt.sizers import (
    FixedRiskSizer,
    FixedSizer,
    KellySizer,
    MaxPositionSizer,
    PercentSizer,
    Sizer,
    VolatilitySizer,
)
from polarbt.trades import Trade, TradeTracker

try:
    from polarbt import plotting
except ImportError:
    plotting = None  # type: ignore[assignment]

__version__ = "0.1.2"
__all__ = [
    "Strategy",
    "Portfolio",
    "Engine",
    "BacktestContext",
    "backtest",
    "backtest_batch",
    "optimize",
    "optimize_multi",
    "optimize_bayesian",
    "walk_forward_analysis",
    "standardize_dataframe",
    "merge_asset_dataframes",
    "indicators",
    "metrics",
    "Order",
    "OrderType",
    "OrderStatus",
    "Trade",
    "TradeTracker",
    "Sizer",
    "FixedSizer",
    "PercentSizer",
    "FixedRiskSizer",
    "KellySizer",
    "VolatilitySizer",
    "MaxPositionSizer",
    "CommissionModel",
    "PercentCommission",
    "FixedPlusPercentCommission",
    "MakerTakerCommission",
    "TieredCommission",
    "CustomCommission",
    "analysis",
    "data",
    "monte_carlo",
    "detect_look_ahead_bias",
    "permutation_test",
    "MonteCarloResult",
    "LookAheadResult",
    "PermutationTestResult",
    "BacktestMetrics",
    "TradeStats",
    "format_results",
    "plotting",
]
