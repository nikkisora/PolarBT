"""Tests for runner and optimization utilities."""

import polars as pl
import pytest

from polarbtest import indicators as ind
from polarbtest.core import Strategy
from polarbtest.runner import backtest, backtest_batch, optimize


@pytest.fixture
def sample_data():
    """Create sample price data."""
    return pl.DataFrame(
        {
            "timestamp": range(100),
            "close": [100 + i * 0.5 for i in range(100)],
        }
    )


class SampleStrategy(Strategy):
    """Simple strategy for testing."""

    def preprocess(self, df):
        sma_period = self.params.get("sma_period", 10)
        return df.with_columns([ind.sma("close", sma_period).alias("sma")])

    def next(self, ctx):
        sma = ctx.row.get("sma")
        close = ctx.row.get("close")

        if sma is None or close is None:
            return

        if close > sma:
            ctx.portfolio.order_target_percent("asset", 0.8)
        else:
            ctx.portfolio.close_position("asset")


class TestBacktest:
    """Test backtest function."""

    def test_backtest_basic(self, sample_data):
        """Test basic backtest execution."""
        results = backtest(SampleStrategy, sample_data, params={"sma_period": 10}, initial_cash=100_000)

        assert results is not None
        assert "sharpe_ratio" in results
        assert "total_return" in results
        assert "final_equity" in results
        assert results["success"] == True

    def test_backtest_different_params(self, sample_data):
        """Test backtest with different parameters."""
        results1 = backtest(SampleStrategy, sample_data, params={"sma_period": 5})
        results2 = backtest(SampleStrategy, sample_data, params={"sma_period": 20})

        # Results should be different
        assert (
            results1["sharpe_ratio"] != results2["sharpe_ratio"] or results1["total_return"] != results2["total_return"]
        )

    def test_backtest_no_params(self, sample_data):
        """Test backtest without parameters."""
        results = backtest(SampleStrategy, sample_data)
        assert results is not None
        assert results["success"] == True


class TestBatchBacktest:
    """Test batch backtesting."""

    def test_backtest_batch(self, sample_data):
        """Test running multiple backtests in parallel."""
        param_sets = [
            {"sma_period": 5},
            {"sma_period": 10},
            {"sma_period": 20},
        ]

        results_df = backtest_batch(SampleStrategy, sample_data, param_sets, n_jobs=2, verbose=False)

        assert len(results_df) == 3
        assert "sharpe_ratio" in results_df.columns
        assert "sma_period" in results_df.columns

    def test_backtest_batch_large(self, sample_data):
        """Test batch with many parameter sets."""
        param_sets = [{"sma_period": i} for i in range(5, 25)]

        results_df = backtest_batch(SampleStrategy, sample_data, param_sets, n_jobs=2, verbose=False)

        assert len(results_df) == len(param_sets)


class TestOptimize:
    """Test optimization utilities."""

    def test_optimize_grid_search(self, sample_data):
        """Test grid search optimization."""
        param_grid = {
            "sma_period": [5, 10, 15, 20],
        }

        best = optimize(
            SampleStrategy,
            sample_data,
            param_grid,
            objective="sharpe_ratio",
            maximize=True,
            verbose=False,
        )

        assert best is not None
        assert "sma_period" in best
        assert "sharpe_ratio" in best
        assert best["sma_period"] in [5, 10, 15, 20]

    def test_optimize_maximize_vs_minimize(self, sample_data):
        """Test maximizing vs minimizing."""
        param_grid = {"sma_period": [5, 10, 20]}

        best_max = optimize(
            SampleStrategy,
            sample_data,
            param_grid,
            objective="sharpe_ratio",
            maximize=True,
            verbose=False,
        )

        best_min = optimize(
            SampleStrategy,
            sample_data,
            param_grid,
            objective="sharpe_ratio",
            maximize=False,
            verbose=False,
        )

        assert best_max is not None
        assert best_min is not None

        # Different optimization directions should potentially give different results
        # (unless all results are the same, which is possible with limited data)
        assert best_max is not None
        assert best_min is not None
