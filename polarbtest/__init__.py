"""
PolarBtest - A lightweight backtesting library for evolutionary strategy search.

Features:
- Polars-based for high performance
- Event-driven architecture with vectorized preprocessing
- Support for fractional and multi-asset positions
- ML model integration ready
- Parallel execution for evolutionary optimization
"""

from polarbtest import indicators, metrics
from polarbtest.core import (
    BacktestContext,
    Engine,
    Portfolio,
    Strategy,
    merge_asset_dataframes,
    standardize_dataframe,
)
from polarbtest.orders import Order, OrderStatus, OrderType
from polarbtest.runner import backtest, backtest_batch
from polarbtest.trades import Trade, TradeTracker

try:
    from polarbtest import plotting
except ImportError:
    plotting = None  # type: ignore[assignment]

__version__ = "0.1.0"
__all__ = [
    "Strategy",
    "Portfolio",
    "Engine",
    "BacktestContext",
    "backtest",
    "backtest_batch",
    "standardize_dataframe",
    "merge_asset_dataframes",
    "indicators",
    "metrics",
    "Order",
    "OrderType",
    "OrderStatus",
    "Trade",
    "TradeTracker",
]
