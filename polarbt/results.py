"""
Structured result types for backtesting.

Provides typed dataclasses for backtest results, replacing untyped dictionaries
with proper fields that enable IDE code completion and static type checking.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import polars as pl


@dataclass
class TradeStats:
    """Aggregate trade statistics.

    Attributes:
        total_trades: Total number of closed trades.
        winning_trades: Number of winning trades.
        losing_trades: Number of losing trades.
        win_rate: Win rate as percentage (e.g., 60.0 means 60%).
        avg_win: Average winning P&L in dollars.
        avg_loss: Average losing P&L in dollars (positive value).
        avg_pnl: Average P&L per trade.
        profit_factor: Total wins / total losses.
        total_pnl: Sum of all trade P&L.
    """

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_pnl: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0


@dataclass
class BacktestMetrics:
    """Complete backtest performance metrics.

    Returned by ``Engine.run()`` and ``backtest()``. All fields are typed
    for IDE autocompletion and static analysis.

    Attributes:
        total_return: Total return as fraction (e.g., 0.10 = 10%).
        cagr: Compound Annual Growth Rate.
        sharpe_ratio: Annualized Sharpe ratio.
        sortino_ratio: Annualized Sortino ratio.
        max_drawdown: Maximum drawdown as positive fraction.
        calmar_ratio: CAGR / Max Drawdown.
        volatility: Daily return standard deviation.
        volatility_annualized: Annualized volatility.
        ulcer_index: Ulcer Index (downside volatility).
        tail_ratio: Right tail / left tail ratio.
        max_drawdown_duration: Longest drawdown in bars.
        avg_drawdown_duration: Average drawdown duration in bars.
        drawdown_count: Number of drawdown periods.
        num_periods: Total number of bars.
        profit_factor: Total gains / total losses (bar-level).
        initial_equity: Starting capital.
        final_equity: Portfolio value at end.
        equity_peak: Maximum equity value reached.
        final_positions: Remaining positions mapping asset to size.
        final_cash: Remaining cash.
        trades: DataFrame of all completed trades.
        trade_stats: Aggregate trade statistics.
        win_rate: Win rate as percentage.
        return_annualized: Same as CAGR (explicit alias).
        buy_hold_return: Buy-and-hold return of first asset.
        best_trade_pct: Best trade return percentage.
        worst_trade_pct: Worst trade return percentage.
        avg_trade_pct: Average trade return percentage.
        max_trade_duration: Longest trade in bars.
        avg_trade_duration: Average trade duration in bars.
        expectancy: Average P&L per trade in dollars.
        sqn: System Quality Number.
        kelly_criterion: Kelly fraction.
        params: Strategy parameters (set by ``backtest()``).
        success: Whether backtest succeeded (set by ``backtest()``).
        error: Error message if backtest failed.
        traceback: Full traceback string if backtest failed.
    """

    # Core metrics (from calculate_metrics)
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0
    volatility_annualized: float = 0.0
    ulcer_index: float = 0.0
    tail_ratio: float = 0.0
    max_drawdown_duration: float = 0.0
    avg_drawdown_duration: float = 0.0
    drawdown_count: int = 0
    num_periods: int = 0
    profit_factor: float = 0.0
    initial_equity: float = 0.0
    final_equity: float = 0.0

    # Portfolio info
    equity_peak: float = 0.0
    final_positions: dict[str, float] = field(default_factory=dict)
    final_cash: float = 0.0

    # Trade info
    trades: pl.DataFrame = field(default_factory=pl.DataFrame)
    trade_stats: TradeStats = field(default_factory=TradeStats)
    win_rate: float = 0.0

    # Derived metrics
    return_annualized: float = 0.0
    buy_hold_return: float = 0.0

    # Trade-level detailed metrics
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    avg_trade_pct: float = 0.0
    max_trade_duration: float = 0.0
    avg_trade_duration: float = 0.0
    expectancy: float = 0.0
    sqn: float = 0.0
    kelly_criterion: float = 0.0

    # Optional fields populated by backtest() / optimize_bayesian()
    params: dict[str, Any] | None = None
    success: bool | None = None
    error: str | None = None
    traceback: str | None = None
    all_results_df: pl.DataFrame | None = None

    def to_scalar_dict(self) -> dict[str, Any]:
        """Convert to a flat dictionary of scalar values suitable for DataFrames.

        Excludes non-scalar fields (trades, trade_stats, final_positions, params)
        and flattens trade_stats fields with a ``ts_`` prefix.
        """
        skip = {"trades", "trade_stats", "final_positions", "params", "traceback"}
        result: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.name in skip:
                continue
            result[f.name] = getattr(self, f.name)
        # Flatten trade_stats
        for f in dataclasses.fields(self.trade_stats):
            result[f"ts_{f.name}"] = getattr(self.trade_stats, f.name)
        return result


def _backtest_metrics_from_dict(metrics_dict: dict[str, Any], trade_stats: TradeStats) -> BacktestMetrics:
    """Construct BacktestMetrics from a metrics dictionary and TradeStats.

    This is an internal helper used by ``Engine._calculate_results()`` to bridge
    the dict returned by ``calculate_metrics()`` into the typed dataclass.

    Args:
        metrics_dict: Raw metrics dictionary from ``calculate_metrics()`` and engine enrichments.
        trade_stats: Typed trade statistics.

    Returns:
        Populated BacktestMetrics instance.
    """
    return BacktestMetrics(
        total_return=metrics_dict.get("total_return", 0.0),
        cagr=metrics_dict.get("cagr", 0.0),
        sharpe_ratio=metrics_dict.get("sharpe_ratio", 0.0),
        sortino_ratio=metrics_dict.get("sortino_ratio", 0.0),
        max_drawdown=metrics_dict.get("max_drawdown", 0.0),
        calmar_ratio=metrics_dict.get("calmar_ratio", 0.0),
        volatility=metrics_dict.get("volatility", 0.0),
        volatility_annualized=metrics_dict.get("volatility_annualized", 0.0),
        ulcer_index=metrics_dict.get("ulcer_index", 0.0),
        tail_ratio=metrics_dict.get("tail_ratio", 0.0),
        max_drawdown_duration=metrics_dict.get("max_drawdown_duration", 0.0),
        avg_drawdown_duration=metrics_dict.get("avg_drawdown_duration", 0.0),
        drawdown_count=metrics_dict.get("drawdown_count", 0),
        num_periods=metrics_dict.get("num_periods", 0),
        profit_factor=metrics_dict.get("profit_factor", 0.0),
        initial_equity=metrics_dict.get("initial_equity", 0.0),
        final_equity=metrics_dict.get("final_equity", 0.0),
        equity_peak=metrics_dict.get("equity_peak", 0.0),
        final_positions=metrics_dict.get("final_positions", {}),
        final_cash=metrics_dict.get("final_cash", 0.0),
        trades=metrics_dict.get("trades", pl.DataFrame()),
        trade_stats=trade_stats,
        win_rate=metrics_dict.get("win_rate", 0.0),
        return_annualized=metrics_dict.get("return_annualized", 0.0),
        buy_hold_return=metrics_dict.get("buy_hold_return", 0.0),
        best_trade_pct=metrics_dict.get("best_trade_pct", 0.0),
        worst_trade_pct=metrics_dict.get("worst_trade_pct", 0.0),
        avg_trade_pct=metrics_dict.get("avg_trade_pct", 0.0),
        max_trade_duration=metrics_dict.get("max_trade_duration", 0.0),
        avg_trade_duration=metrics_dict.get("avg_trade_duration", 0.0),
        expectancy=metrics_dict.get("expectancy", 0.0),
        sqn=metrics_dict.get("sqn", 0.0),
        kelly_criterion=metrics_dict.get("kelly_criterion", 0.0),
    )
