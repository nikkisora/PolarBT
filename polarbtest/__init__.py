"""
PolarBtest - A lightweight backtesting library for evolutionary strategy search.

Features:
- Polars-based for high performance
- Event-driven architecture with vectorized preprocessing
- Support for fractional and multi-asset positions
- ML model integration ready
- Parallel execution for evolutionary optimization
"""

from polarbtest.core import Strategy, Portfolio, Engine, BacktestContext
from polarbtest.runner import backtest, backtest_batch
from polarbtest import indicators
from polarbtest import metrics

__version__ = "0.1.0"
__all__ = [
    "Strategy",
    "Portfolio",
    "Engine",
    "BacktestContext",
    "backtest",
    "backtest_batch",
    "indicators",
    "metrics",
]
