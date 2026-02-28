"""
Visualization module for PolarBtest backtesting results.

Requires plotly as an optional dependency: pip install plotly
"""

from polarbtest.plotting.charts import (
    plot_backtest,
    plot_returns_distribution,
)

__all__ = [
    "plot_backtest",
    "plot_returns_distribution",
]
