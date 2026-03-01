"""Tests for advanced analysis: Monte Carlo, look-ahead bias detection, permutation test."""

import numpy as np
import polars as pl
import pytest

from polarbtest import indicators as ind
from polarbtest.analysis import (
    LookAheadResult,
    MonteCarloResult,
    PermutationTestResult,
    detect_look_ahead_bias,
    monte_carlo,
    permutation_test,
)
from polarbtest.core import BacktestContext, Strategy
from polarbtest.trades import Trade

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trades(pnls: list[float]) -> list[Trade]:
    """Create Trade objects with given P&Ls."""
    trades = []
    for i, pnl in enumerate(pnls):
        entry_price = 100.0
        exit_price = entry_price + pnl
        t = Trade(
            trade_id=str(i),
            asset="TEST",
            direction="long",
            entry_bar=i * 10,
            entry_timestamp=i * 10,
            entry_price=entry_price,
            entry_size=1.0,
            entry_value=entry_price,
            exit_bar=i * 10 + 5,
            exit_timestamp=i * 10 + 5,
            exit_price=exit_price,
            exit_size=1.0,
            exit_value=exit_price,
        )
        trades.append(t)
    return trades


def _make_ohlcv_data(n: int = 200, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 10.0)  # keep positive
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    low = np.maximum(low, 1.0)
    opn = close + rng.normal(0, 0.5, n)
    opn = np.maximum(opn, 1.0)
    volume = rng.uniform(100, 1000, n)

    return pl.DataFrame(
        {
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class SimpleSMAStrategy(Strategy):
    """Simple SMA crossover for testing."""

    def __init__(self, fast: int = 10, slow: int = 30, **kwargs: object) -> None:
        super().__init__(fast=fast, slow=slow, **kwargs)
        self.fast = fast
        self.slow = slow

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            [
                ind.sma("close", self.fast).alias("sma_fast"),
                ind.sma("close", self.slow).alias("sma_slow"),
            ]
        )

    def next(self, ctx: BacktestContext) -> None:
        fast = ctx.row.get("sma_fast")
        slow = ctx.row.get("sma_slow")
        if fast is None or slow is None:
            return
        if fast > slow and ctx.portfolio.get_position("asset") == 0:
            ctx.portfolio.order("asset", 1.0)
        elif fast < slow and ctx.portfolio.get_position("asset") > 0:
            ctx.portfolio.close_position("asset")


class LookAheadStrategy(Strategy):
    """Strategy that intentionally leaks future data."""

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        # future_mean uses the mean of the entire column — clearly look-ahead
        return df.with_columns(
            [
                ind.sma("close", 10).alias("sma_10"),
                (pl.col("close") - pl.col("close").mean()).alias("future_demean"),
            ]
        )

    def next(self, ctx: BacktestContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Monte Carlo Tests
# ---------------------------------------------------------------------------


class TestMonteCarlo:
    def test_basic(self) -> None:
        trades = _make_trades([10, -5, 20, -3, 15, -8, 12, -2, 8, -4])
        result = monte_carlo(trades, initial_capital=10_000, n_simulations=500, seed=42)

        assert isinstance(result, MonteCarloResult)
        assert result.n_simulations == 500
        assert result.initial_capital == 10_000
        assert result.simulated_equities.shape == (500, 11)  # 10 trades + initial
        assert len(result.final_equities) == 500
        assert len(result.max_drawdowns) == 500

    def test_confidence_intervals(self) -> None:
        trades = _make_trades([10, -5, 20, -3, 15, -8, 12, -2, 8, -4])
        result = monte_carlo(trades, initial_capital=10_000, n_simulations=1000, seed=42)

        assert "final_equity" in result.confidence_intervals
        assert "max_drawdown" in result.confidence_intervals
        assert "total_return" in result.confidence_intervals

        lower, upper = result.confidence_intervals["final_equity"]
        assert lower < upper
        # All simulations should have the same total P&L on average (sum of resampled pnls)
        assert lower > 0  # shouldn't go to zero with these trades

    def test_percentiles(self) -> None:
        trades = _make_trades([10, -5, 20, -3, 15])
        result = monte_carlo(trades, n_simulations=200, seed=123)

        assert "final_equity" in result.percentiles
        assert len(result.percentiles["final_equity"]) == 9  # 1,5,10,25,50,75,90,95,99

    def test_empty_trades_raises(self) -> None:
        with pytest.raises(ValueError, match="trades list must not be empty"):
            monte_carlo([], initial_capital=10_000)

    def test_single_trade(self) -> None:
        trades = _make_trades([50.0])
        result = monte_carlo(trades, initial_capital=10_000, n_simulations=100, seed=1)

        # With only one trade, every simulation resamples the same trade
        assert np.allclose(result.final_equities, 10_050.0)
        assert np.allclose(result.max_drawdowns, 0.0)

    def test_all_losing_trades(self) -> None:
        trades = _make_trades([-10, -20, -5, -15])
        result = monte_carlo(trades, initial_capital=10_000, n_simulations=200, seed=42)

        # All simulations should end below initial capital
        assert np.all(result.final_equities < 10_000)
        assert np.all(result.max_drawdowns > 0)

    def test_reproducibility(self) -> None:
        trades = _make_trades([10, -5, 20, -3, 15])
        r1 = monte_carlo(trades, seed=99, n_simulations=100)
        r2 = monte_carlo(trades, seed=99, n_simulations=100)
        assert np.array_equal(r1.final_equities, r2.final_equities)


# ---------------------------------------------------------------------------
# Look-Ahead Bias Detection Tests
# ---------------------------------------------------------------------------


class TestLookAheadBias:
    def test_clean_strategy(self) -> None:
        data = _make_ohlcv_data(200)
        strategy = SimpleSMAStrategy(fast=10, slow=30)
        result = detect_look_ahead_bias(strategy, data, sample_bars=3)

        assert isinstance(result, LookAheadResult)
        assert "sma_fast" in result.clean_columns
        assert "sma_slow" in result.clean_columns
        assert len(result.biased_columns) == 0

    def test_biased_strategy(self) -> None:
        data = _make_ohlcv_data(200)
        strategy = LookAheadStrategy()
        result = detect_look_ahead_bias(strategy, data, sample_bars=3)

        assert "future_demean" in result.biased_columns
        assert "future_demean" in result.details
        # SMA should still be clean
        assert "sma_10" in result.clean_columns

    def test_no_new_columns(self) -> None:
        data = _make_ohlcv_data(50)

        class PassthroughStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def next(self, ctx: BacktestContext) -> None:
                pass

        result = detect_look_ahead_bias(PassthroughStrategy(), data)
        assert result.biased_columns == []
        assert result.clean_columns == []


# ---------------------------------------------------------------------------
# Permutation Test Tests
# ---------------------------------------------------------------------------


class TestPermutationTest:
    def test_basic(self) -> None:
        data = _make_ohlcv_data(200)
        result = permutation_test(
            SimpleSMAStrategy,
            data,
            metric="sharpe_ratio",
            n_permutations=10,
            seed=42,
            params={"fast": 10, "slow": 30},
        )

        assert isinstance(result, PermutationTestResult)
        assert result.n_permutations == 10
        assert len(result.null_distribution) == 10
        assert 0.0 <= result.p_value <= 1.0
        assert isinstance(result.mean_null, float)
        assert isinstance(result.std_null, float)

    def test_with_precomputed_metric(self) -> None:
        data = _make_ohlcv_data(200)
        result = permutation_test(
            SimpleSMAStrategy,
            data,
            original_metric=0.5,
            metric="sharpe_ratio",
            n_permutations=10,
            seed=42,
            params={"fast": 10, "slow": 30},
        )

        assert result.original_metric == 0.5

    def test_reproducibility(self) -> None:
        data = _make_ohlcv_data(100)
        r1 = permutation_test(
            SimpleSMAStrategy,
            data,
            n_permutations=5,
            seed=42,
            params={"fast": 10, "slow": 30},
        )
        r2 = permutation_test(
            SimpleSMAStrategy,
            data,
            n_permutations=5,
            seed=42,
            params={"fast": 10, "slow": 30},
        )
        assert np.allclose(r1.null_distribution, r2.null_distribution)

    def test_p_value_range(self) -> None:
        data = _make_ohlcv_data(150)
        result = permutation_test(
            SimpleSMAStrategy,
            data,
            n_permutations=20,
            seed=42,
            params={"fast": 10, "slow": 30},
        )
        assert 0.0 <= result.p_value <= 1.0

    def test_no_price_columns_raises(self) -> None:
        data = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        with pytest.raises(ValueError, match="at least one of"):
            permutation_test(SimpleSMAStrategy, data, n_permutations=1, params={"fast": 1, "slow": 2})
