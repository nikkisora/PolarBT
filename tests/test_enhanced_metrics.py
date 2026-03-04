"""Tests for enhanced metrics (Part 3 of the plan)."""

from datetime import date, timedelta

import polars as pl

from polarbt.metrics import (
    alpha_beta,
    calculate_metrics,
    drawdown_duration_stats,
    format_results,
    information_ratio,
    monthly_returns,
    tail_ratio,
    trade_level_metrics,
    ulcer_index,
)
from polarbt.results import BacktestMetrics, TradeStats
from polarbt.trades import Trade


def _make_equity_df(values: list[float], with_timestamps: bool = False) -> pl.DataFrame:
    """Helper to create equity DataFrames."""
    data: dict[str, list[float] | list[date]] = {"equity": values}
    if with_timestamps:
        start = date(2024, 1, 1)
        data["timestamp"] = [start + timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame(data)


class TestUlcerIndex:
    def test_flat_equity(self) -> None:
        df = _make_equity_df([100.0] * 20)
        assert ulcer_index(df) == 0.0

    def test_rising_equity(self) -> None:
        df = _make_equity_df([100.0 + i for i in range(20)])
        assert ulcer_index(df) == 0.0

    def test_with_drawdown(self) -> None:
        values = [100.0, 105.0, 110.0, 100.0, 95.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0]
        df = _make_equity_df(values)
        ui = ulcer_index(df, period=5)
        assert ui > 0.0

    def test_short_series(self) -> None:
        df = _make_equity_df([100.0, 101.0])
        assert ulcer_index(df, period=14) == 0.0

    def test_empty(self) -> None:
        df = pl.DataFrame({"equity": []}, schema={"equity": pl.Float64})
        assert ulcer_index(df) == 0.0


class TestTailRatio:
    def test_symmetric_returns(self) -> None:
        values = [100.0 + i * 0.1 * ((-1) ** i) for i in range(100)]
        df = _make_equity_df(values)
        tr = tail_ratio(df)
        assert tr > 0.0

    def test_positive_skew(self) -> None:
        # Mostly small losses, occasional large gains
        values = [100.0]
        for i in range(99):
            if i % 10 == 0:
                values.append(values[-1] * 1.05)
            else:
                values.append(values[-1] * 0.998)
        df = _make_equity_df(values)
        tr = tail_ratio(df)
        assert tr > 1.0

    def test_empty(self) -> None:
        df = pl.DataFrame({"equity": []}, schema={"equity": pl.Float64})
        assert tail_ratio(df) == 0.0


class TestInformationRatio:
    def test_outperforming(self) -> None:
        strategy = _make_equity_df([100.0 + i * 2 for i in range(50)])
        benchmark = _make_equity_df([100.0 + i for i in range(50)])
        ir = information_ratio(strategy, benchmark)
        assert ir > 0.0

    def test_underperforming(self) -> None:
        strategy = _make_equity_df([100.0 + i for i in range(50)])
        benchmark = _make_equity_df([100.0 + i * 2 for i in range(50)])
        ir = information_ratio(strategy, benchmark)
        assert ir < 0.0

    def test_identical(self) -> None:
        df = _make_equity_df([100.0 + i for i in range(50)])
        ir = information_ratio(df, df)
        assert ir == 0.0

    def test_empty(self) -> None:
        df = pl.DataFrame({"equity": []}, schema={"equity": pl.Float64})
        assert information_ratio(df, df) == 0.0


class TestAlphaBeta:
    def test_perfect_correlation(self) -> None:
        strategy = _make_equity_df([100.0 + i * 2 for i in range(50)])
        benchmark = _make_equity_df([100.0 + i for i in range(50)])
        result = alpha_beta(strategy, benchmark)
        assert "alpha" in result
        assert "beta" in result

    def test_beta_near_one_for_same(self) -> None:
        df = _make_equity_df([100.0 + i for i in range(50)])
        result = alpha_beta(df, df)
        assert abs(result["beta"] - 1.0) < 0.01
        assert abs(result["alpha"]) < 0.01

    def test_empty(self) -> None:
        df = pl.DataFrame({"equity": []}, schema={"equity": pl.Float64})
        result = alpha_beta(df, df)
        assert result["alpha"] == 0.0
        assert result["beta"] == 0.0


class TestDrawdownDurationStats:
    def test_no_drawdowns(self) -> None:
        df = _make_equity_df([100.0 + i for i in range(20)])
        stats = drawdown_duration_stats(df)
        assert stats["drawdown_count"] == 0
        assert stats["max_drawdown_duration"] == 0.0

    def test_single_drawdown(self) -> None:
        values = [100.0, 110.0, 105.0, 100.0, 95.0, 100.0, 105.0, 110.0, 115.0]
        df = _make_equity_df(values)
        stats = drawdown_duration_stats(df)
        assert stats["drawdown_count"] >= 1
        assert stats["max_drawdown_duration"] > 0.0

    def test_multiple_drawdowns(self) -> None:
        values = [100.0, 110.0, 105.0, 115.0, 110.0, 120.0]
        df = _make_equity_df(values)
        stats = drawdown_duration_stats(df)
        assert stats["drawdown_count"] == 2

    def test_empty(self) -> None:
        df = pl.DataFrame({"equity": []}, schema={"equity": pl.Float64})
        stats = drawdown_duration_stats(df)
        assert stats["drawdown_count"] == 0


class TestMonthlyReturns:
    def test_basic(self) -> None:
        start = date(2024, 1, 1)
        timestamps = [start + timedelta(days=i) for i in range(90)]
        equity = [100.0 + i * 0.5 for i in range(90)]
        df = pl.DataFrame({"timestamp": timestamps, "equity": equity})
        result = monthly_returns(df)
        assert "year" in result.columns
        assert "month" in result.columns
        assert "return" in result.columns
        assert len(result) >= 3

    def test_no_timestamp(self) -> None:
        df = pl.DataFrame({"equity": [100.0, 101.0]})
        result = monthly_returns(df)
        assert len(result) == 0

    def test_empty(self) -> None:
        df = pl.DataFrame({"timestamp": [], "equity": []}, schema={"timestamp": pl.Date, "equity": pl.Float64})
        result = monthly_returns(df)
        assert len(result) == 0


def _make_trade(pnl: float) -> Trade:
    """Helper to create a Trade with given P&L."""
    # Set entry/exit values so that the calculated pnl matches
    entry_value = 1000.0
    exit_value = entry_value + pnl
    return Trade(
        trade_id="test",
        asset="TEST",
        direction="long",
        entry_bar=0,
        entry_timestamp=None,
        entry_price=100.0,
        entry_size=10.0,
        entry_value=entry_value,
        exit_bar=1,
        exit_timestamp=None,
        exit_price=exit_value / 10.0,
        exit_size=10.0,
        exit_value=exit_value,
    )


class TestTradeLevelMetrics:
    def test_empty(self) -> None:
        result = trade_level_metrics([])
        assert result["expectancy"] == 0.0
        assert result["sqn"] == 0.0
        assert result["max_consecutive_wins"] == 0
        assert result["max_consecutive_losses"] == 0

    def test_all_winners(self) -> None:
        trades = [_make_trade(100.0), _make_trade(200.0), _make_trade(50.0)]
        result = trade_level_metrics(trades)
        assert result["expectancy"] > 0.0
        assert result["kelly_criterion"] == 1.0
        assert result["max_consecutive_wins"] == 3
        assert result["max_consecutive_losses"] == 0

    def test_all_losers(self) -> None:
        trades = [_make_trade(-100.0), _make_trade(-200.0)]
        result = trade_level_metrics(trades)
        assert result["expectancy"] < 0.0
        assert result["kelly_criterion"] == 0.0
        assert result["max_consecutive_losses"] == 2
        assert result["max_consecutive_wins"] == 0

    def test_mixed(self) -> None:
        trades = [
            _make_trade(100.0),
            _make_trade(200.0),
            _make_trade(-50.0),
            _make_trade(-30.0),
            _make_trade(-20.0),
            _make_trade(150.0),
        ]
        result = trade_level_metrics(trades)
        assert result["max_consecutive_wins"] == 2
        assert result["max_consecutive_losses"] == 3
        assert result["kelly_criterion"] > 0.0

    def test_sqn(self) -> None:
        trades = [_make_trade(100.0), _make_trade(200.0), _make_trade(-50.0), _make_trade(150.0)]
        result = trade_level_metrics(trades)
        assert result["sqn"] > 0.0


class TestCalculateMetricsEnhanced:
    def test_includes_new_fields(self) -> None:
        df = _make_equity_df([100.0 + i for i in range(30)])
        result = calculate_metrics(df, 100.0)
        assert "ulcer_index" in result
        assert "tail_ratio" in result
        assert "max_drawdown_duration" in result
        assert "avg_drawdown_duration" in result
        assert "drawdown_count" in result

    def test_with_drawdown(self) -> None:
        values = [
            100.0,
            110.0,
            105.0,
            100.0,
            95.0,
            100.0,
            105.0,
            110.0,
            115.0,
            120.0,
            125.0,
            130.0,
            135.0,
            140.0,
            145.0,
        ]
        df = _make_equity_df(values)
        result = calculate_metrics(df, 100.0)
        assert result["ulcer_index"] >= 0.0
        assert result["drawdown_count"] >= 1


class TestFormatResults:
    def _make_results(self) -> BacktestMetrics:
        """Create a minimal BacktestMetrics similar to backtest() output."""
        trades_df = pl.DataFrame(
            {
                "return_pct": [0.05, -0.02, 0.08, -0.01, 0.03],
                "pnl": [500.0, -200.0, 800.0, -100.0, 300.0],
                "bars_held": [10, 5, 20, 3, 15],
            }
        )
        return BacktestMetrics(
            final_equity=110_000.0,
            equity_peak=115_000.0,
            initial_equity=100_000.0,
            total_return=0.10,
            buy_hold_return=0.15,
            return_annualized=0.12,
            cagr=0.12,
            volatility_annualized=0.20,
            sharpe_ratio=1.5,
            sortino_ratio=2.1,
            calmar_ratio=0.8,
            max_drawdown=0.15,
            avg_drawdown_duration=5.0,
            max_drawdown_duration=12.0,
            best_trade_pct=8.0,
            worst_trade_pct=-2.0,
            avg_trade_pct=2.6,
            max_trade_duration=20.0,
            avg_trade_duration=10.6,
            expectancy=260.0,
            sqn=2.87,
            kelly_criterion=0.4333,
            trades=trades_df,
            trade_stats=TradeStats(
                total_trades=5,
                win_rate=60.0,
                profit_factor=2.67,
            ),
        )

    def test_contains_key_labels(self) -> None:
        output = format_results(self._make_results())
        assert "Equity Final [$]" in output
        assert "Equity Peak [$]" in output
        assert "Return [%]" in output
        assert "Buy & Hold Return [%]" in output
        assert "Return (Ann.) [%]" in output
        assert "Sharpe Ratio" in output
        assert "# Trades" in output
        assert "Win Rate [%]" in output
        assert "SQN" in output
        assert "Kelly Criterion" in output

    def test_values_formatted(self) -> None:
        output = format_results(self._make_results())
        assert "110,000.00" in output
        assert "10.00" in output  # Return [%]
        assert "1.50" in output  # Sharpe

    def test_empty_trades(self) -> None:
        results = self._make_results()
        results.trades = pl.DataFrame(schema={"return_pct": pl.Float64, "pnl": pl.Float64, "bars_held": pl.Int64})
        results.trade_stats = TradeStats(total_trades=0, win_rate=0.0, profit_factor=0.0)
        output = format_results(results)
        assert "# Trades" in output

    def test_no_trades_key(self) -> None:
        results = self._make_results()
        results.trades = pl.DataFrame()
        output = format_results(results)
        assert "# Trades" in output

    def test_returns_string(self) -> None:
        output = format_results(self._make_results())
        assert isinstance(output, str)
        assert len(output.splitlines()) > 10
