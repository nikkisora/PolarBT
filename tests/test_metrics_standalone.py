"""Tests for standalone metrics.py functions that had zero test coverage.

Covers: sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
omega_ratio, rolling_sharpe, underwater_plot_data, value_at_risk,
conditional_value_at_risk, liquidity_metrics.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from polarbt.metrics import (
    calmar_ratio,
    conditional_value_at_risk,
    liquidity_metrics,
    max_drawdown,
    omega_ratio,
    rolling_sharpe,
    sharpe_ratio,
    sortino_ratio,
    underwater_plot_data,
    value_at_risk,
)


def _make_equity(values: list[float], with_timestamps: bool = False) -> pl.DataFrame:
    data: dict[str, list[float] | list[date]] = {"equity": values}
    if with_timestamps:
        start = date(2024, 1, 1)
        data["timestamp"] = [start + timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------


class TestSharpeRatio:
    def test_flat_equity_zero(self):
        df = _make_equity([100.0] * 20)
        assert sharpe_ratio(df) == 0.0

    def test_positive_for_uptrend(self):
        df = _make_equity([100.0 + i for i in range(100)])
        assert sharpe_ratio(df) > 0.0

    def test_negative_for_downtrend(self):
        df = _make_equity([100.0 - i * 0.5 for i in range(100)])
        assert sharpe_ratio(df) < 0.0

    def test_risk_free_rate_reduces_ratio(self):
        df = _make_equity([100.0 + i * 0.1 for i in range(100)])
        sr_no_rf = sharpe_ratio(df, risk_free_rate=0.0)
        sr_with_rf = sharpe_ratio(df, risk_free_rate=0.05)
        assert sr_with_rf < sr_no_rf

    def test_empty_returns_zero(self):
        df = _make_equity([])
        assert sharpe_ratio(df) == 0.0


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------


class TestSortinoRatio:
    def test_zero_when_all_positive_returns(self):
        # Monotone uptrend -> no downside returns -> sortino returns 0.0
        df = _make_equity([100.0 + i for i in range(100)])
        assert sortino_ratio(df) == 0.0

    def test_zero_when_no_downside(self):
        # All returns are positive -> no downside -> returns 0.0 (our impl)
        df = _make_equity([100.0 + i for i in range(20)])
        assert sortino_ratio(df) == 0.0

    def test_empty_returns_zero(self):
        df = _make_equity([])
        assert sortino_ratio(df) == 0.0


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_no_drawdown(self):
        df = _make_equity([100.0 + i for i in range(20)])
        assert max_drawdown(df) == pytest.approx(0.0)

    def test_known_drawdown(self):
        # Peak at 200, drops to 150 -> 25% drawdown
        df = _make_equity([100.0, 150.0, 200.0, 150.0, 180.0])
        assert max_drawdown(df) == pytest.approx(0.25, abs=1e-10)

    def test_full_loss(self):
        df = _make_equity([100.0, 50.0, 0.01])
        assert max_drawdown(df) == pytest.approx(0.9999, abs=0.001)


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------


class TestCalmarRatio:
    def test_no_drawdown_returns_zero(self):
        df = _make_equity([100.0 + i for i in range(20)])
        assert calmar_ratio(df, 100.0) == 0.0

    def test_positive_for_uptrend_with_drawdown(self):
        values = [100.0, 110.0, 105.0, 115.0, 120.0, 115.0, 125.0, 130.0, 128.0, 135.0]
        values += [140.0 + i for i in range(40)]
        df = _make_equity(values)
        assert calmar_ratio(df, 100.0) > 0.0

    def test_single_period_returns_zero(self):
        df = _make_equity([100.0])
        assert calmar_ratio(df, 100.0) == 0.0


# ---------------------------------------------------------------------------
# omega_ratio
# ---------------------------------------------------------------------------


class TestOmegaRatio:
    def test_all_positive_returns(self):
        df = _make_equity([100.0 + i for i in range(50)])
        result = omega_ratio(df, threshold=0.0)
        assert result == float("inf") or result > 10.0

    def test_all_negative_returns(self):
        df = _make_equity([100.0 - i * 0.5 for i in range(50)])
        result = omega_ratio(df, threshold=0.0)
        assert result < 1.0

    def test_empty_returns_zero(self):
        df = _make_equity([])
        assert omega_ratio(df) == 0.0

    def test_higher_threshold_lowers_ratio(self):
        values = [100.0 + i * 0.5 + (i % 3 - 1) for i in range(50)]
        df = _make_equity(values)
        low = omega_ratio(df, threshold=-0.01)
        high = omega_ratio(df, threshold=0.01)
        assert low >= high


# ---------------------------------------------------------------------------
# rolling_sharpe
# ---------------------------------------------------------------------------


class TestRollingSharpe:
    def test_adds_column(self):
        df = _make_equity([100.0 + i for i in range(300)], with_timestamps=True)
        result = rolling_sharpe(df, window=50)
        assert "rolling_sharpe" in result.columns
        assert len(result) == len(df)

    def test_nulls_before_window(self):
        df = _make_equity([100.0 + i for i in range(100)])
        result = rolling_sharpe(df, window=50)
        # First window-1 values should be null
        assert result["rolling_sharpe"][0] is None


# ---------------------------------------------------------------------------
# underwater_plot_data
# ---------------------------------------------------------------------------


class TestUnderwaterPlotData:
    def test_columns(self):
        df = _make_equity([100.0, 110.0, 105.0, 115.0, 120.0], with_timestamps=True)
        result = underwater_plot_data(df)
        assert "timestamp" in result.columns
        assert "drawdown" in result.columns
        assert len(result) == len(df)

    def test_drawdown_values(self):
        df = _make_equity([100.0, 110.0, 100.0, 120.0], with_timestamps=True)
        result = underwater_plot_data(df)
        # At index 0: peak=100, dd=0
        assert result["drawdown"][0] == pytest.approx(0.0)
        # At index 2: peak=110, equity=100, dd = (100-110)/110
        assert result["drawdown"][2] == pytest.approx(-10.0 / 110.0, abs=1e-10)
        # At index 3: peak=120, dd=0
        assert result["drawdown"][3] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# value_at_risk
# ---------------------------------------------------------------------------


class TestValueAtRisk:
    def test_positive_result(self):
        np.random.seed(42)
        values = np.cumsum(np.random.randn(200) * 0.01) + 100.0
        df = _make_equity(values.tolist())
        var = value_at_risk(df, confidence=0.95)
        assert var > 0.0

    def test_empty_returns_zero(self):
        df = _make_equity([])
        assert value_at_risk(df) == 0.0

    def test_higher_confidence_higher_var(self):
        np.random.seed(42)
        values = np.cumsum(np.random.randn(500) * 0.01) + 100.0
        df = _make_equity(values.tolist())
        var_95 = value_at_risk(df, confidence=0.95)
        var_99 = value_at_risk(df, confidence=0.99)
        assert var_99 >= var_95


# ---------------------------------------------------------------------------
# conditional_value_at_risk
# ---------------------------------------------------------------------------


class TestConditionalVaR:
    def test_cvar_gte_var(self):
        """CVaR should be >= VaR (it's the expected loss beyond VaR)."""
        np.random.seed(42)
        values = np.cumsum(np.random.randn(500) * 0.01) + 100.0
        df = _make_equity(values.tolist())
        var = value_at_risk(df, confidence=0.95)
        cvar = conditional_value_at_risk(df, confidence=0.95)
        assert cvar >= var

    def test_empty_returns_zero(self):
        df = _make_equity([])
        assert conditional_value_at_risk(df) == 0.0


# ---------------------------------------------------------------------------
# liquidity_metrics
# ---------------------------------------------------------------------------


class TestLiquidityMetrics:
    def test_no_relevant_columns(self):
        trades_df = pl.DataFrame({"entry_price": [100.0], "exit_price": [110.0]})
        data = pl.DataFrame({"close": [100.0, 110.0]})
        result = liquidity_metrics(trades_df, data)
        assert result["buy_high_ratio"] is None
        assert result["sell_low_ratio"] is None
        assert result["capacity"] is None

    def test_empty_trades(self):
        trades_df = pl.DataFrame(schema={"entry_price": pl.Float64, "exit_price": pl.Float64, "entry_bar": pl.Int64})
        data = pl.DataFrame({"close": [100.0], "limit_up": [105.0]})
        result = liquidity_metrics(trades_df, data)
        assert result["buy_high_ratio"] is None

    def test_capacity_from_trading_value(self):
        data = pl.DataFrame({"close": [100.0] * 20, "trading_value": [float(i * 1000) for i in range(1, 21)]})
        trades_df = pl.DataFrame({"entry_price": [100.0], "exit_price": [110.0]})
        result = liquidity_metrics(trades_df, data)
        assert result["capacity"] is not None
        assert result["capacity"] > 0
