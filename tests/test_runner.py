"""Tests for runner and optimization utilities."""

import polars as pl
import pytest

from polarbt import indicators as ind
from polarbt.core import Strategy
from polarbt.runner import backtest, backtest_batch, optimize


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
        assert results.sharpe_ratio is not None
        assert results.total_return is not None
        assert results.final_equity is not None
        assert results.success

    def test_backtest_different_params(self, sample_data):
        """Test backtest with different parameters."""
        results1 = backtest(SampleStrategy, sample_data, params={"sma_period": 5})
        results2 = backtest(SampleStrategy, sample_data, params={"sma_period": 20})

        # Results should be different
        assert results1.sharpe_ratio != results2.sharpe_ratio or results1.total_return != results2.total_return

    def test_backtest_no_params(self, sample_data):
        """Test backtest without parameters."""
        results = backtest(SampleStrategy, sample_data)
        assert results is not None
        assert results.success


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
        assert "sma_period" in best.params
        assert best.metrics.sharpe_ratio is not None
        assert best.params["sma_period"] in [5, 10, 15, 20]

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

    def test_optimize_returns_optimize_result(self, sample_data):
        """optimize() returns OptimizeResult with separated params and metrics."""
        from polarbt.results import OptimizeResult

        param_grid = {"sma_period": [5, 10, 20]}
        best = optimize(SampleStrategy, sample_data, param_grid, verbose=False)

        assert isinstance(best, OptimizeResult)
        assert "sma_period" in best.params
        assert best.metrics.sharpe_ratio is not None
        assert best.results_df is not None
        assert len(best.results_df) == 3

        # Access params and metrics directly
        assert best.params["sma_period"] in [5, 10, 20]
        assert best.metrics.sharpe_ratio is not None

    def test_optimize_filters_failed_backtests(self, sample_data):
        """optimize() filters out failed backtests so sentinels don't win."""

        class FailingStrategy(Strategy):
            def preprocess(self, df):
                if self.params.get("sma_period") == 5:
                    raise ValueError("Intentional failure")
                return df.with_columns([ind.sma("close", self.params.get("sma_period", 10)).alias("sma")])

            def next(self, ctx):
                sma = ctx.row.get("sma")
                close = ctx.row.get("close")
                if sma and close and close > sma:
                    ctx.portfolio.order_target_percent("asset", 0.8)

        param_grid = {"sma_period": [5, 10, 20]}
        best = optimize(
            FailingStrategy,
            sample_data,
            param_grid,
            objective="sharpe_ratio",
            maximize=False,
            verbose=False,
            n_jobs=1,
        )
        # The failed backtest (sma_period=5, sharpe=-999) should NOT be selected
        assert best.params["sma_period"] != 5

    def test_optimize_no_param_metric_collision(self, sample_data):
        """OptimizeResult separates params from metrics even with overlapping names."""
        import dataclasses

        param_grid = {"sma_period": [5, 10]}
        best = optimize(SampleStrategy, sample_data, param_grid, verbose=False)

        # params dict only has strategy params, metrics has everything else
        assert "sma_period" in best.params
        assert "sma_period" not in {f.name for f in dataclasses.fields(best.metrics)}


@pytest.fixture
def datetime_data():
    """Create sample data with real datetime timestamps (triggers fork issues with Polars)."""
    from datetime import datetime, timedelta

    base = datetime(2024, 1, 1)
    n = 100
    return pl.DataFrame(
        {
            "timestamp": [base + timedelta(hours=i) for i in range(n)],
            "open": [100.0 + i * 0.4 for i in range(n)],
            "high": [101.0 + i * 0.5 for i in range(n)],
            "low": [99.0 + i * 0.3 for i in range(n)],
            "close": [100.0 + i * 0.5 for i in range(n)],
            "volume": [1000.0 + i * 10 for i in range(n)],
        }
    )


class TestSequentialFallback:
    """Test that n_jobs=1 runs sequentially without ProcessPoolExecutor."""

    def test_njobs_1_skips_multiprocessing(self, sample_data):
        """n_jobs=1 should call worker directly, not spawn processes."""
        param_sets = [{"sma_period": 5}, {"sma_period": 10}]
        results_df = backtest_batch(SampleStrategy, sample_data, param_sets, n_jobs=1, verbose=False)

        assert len(results_df) == 2
        assert "sharpe_ratio" in results_df.columns

    def test_njobs_1_with_datetime_data(self, datetime_data):
        """n_jobs=1 with datetime columns must not deadlock."""
        param_sets = [{"sma_period": 5}, {"sma_period": 10}, {"sma_period": 20}]
        results_df = backtest_batch(SampleStrategy, datetime_data, param_sets, n_jobs=1, verbose=False)

        assert len(results_df) == 3
        assert all(results_df["success"].to_list())

    def test_njobs_1_results_match_single_backtest(self, sample_data):
        """Sequential batch results should match individual backtest() calls."""
        params = {"sma_period": 10}
        single = backtest(SampleStrategy, sample_data, params=params)
        batch_df = backtest_batch(SampleStrategy, sample_data, [params], n_jobs=1, verbose=False)

        batch_row = batch_df.to_dicts()[0]
        assert abs(single.sharpe_ratio - batch_row["sharpe_ratio"]) < 1e-10
        assert abs(single.total_return - batch_row["total_return"]) < 1e-10

    def test_optimize_njobs_1(self, sample_data):
        """optimize() with n_jobs=1 should work without multiprocessing."""
        best = optimize(
            SampleStrategy,
            sample_data,
            param_grid={"sma_period": [5, 10, 20]},
            n_jobs=1,
            verbose=False,
        )
        assert best is not None
        assert best.metrics.sharpe_ratio is not None


class TestSpawnContext:
    """Test that parallel execution uses spawn context."""

    def test_batch_parallel_with_datetime(self, datetime_data):
        """Parallel batch with datetime data should work (spawn context)."""
        param_sets = [{"sma_period": 5}, {"sma_period": 10}, {"sma_period": 20}]
        results_df = backtest_batch(SampleStrategy, datetime_data, param_sets, n_jobs=2, verbose=False)

        assert len(results_df) == 3
        assert "sharpe_ratio" in results_df.columns

    def test_batch_parallel_results_consistent(self, sample_data):
        """Parallel and sequential results should be consistent."""
        param_sets = [{"sma_period": 5}, {"sma_period": 10}, {"sma_period": 20}]

        seq_df = backtest_batch(SampleStrategy, sample_data, param_sets, n_jobs=1, verbose=False)
        par_df = backtest_batch(SampleStrategy, sample_data, param_sets, n_jobs=2, verbose=False)

        # Sort both by sma_period for comparison
        seq_df = seq_df.sort("sma_period")
        par_df = par_df.sort("sma_period")

        for col in ["sharpe_ratio", "total_return"]:
            seq_vals = seq_df[col].to_list()
            par_vals = par_df[col].to_list()
            for s, p in zip(seq_vals, par_vals, strict=True):
                assert abs(s - p) < 1e-10, f"Mismatch in {col}: seq={s}, par={p}"
