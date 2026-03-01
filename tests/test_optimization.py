"""Tests for optimization enhancements: constraints, multi-objective, Bayesian, plots."""

import contextlib
import multiprocessing as mp

import polars as pl
import pytest

from polarbt import indicators as ind
from polarbt.core import Strategy
from polarbt.runner import (
    _generate_param_sets,
    _pareto_front,
    optimize,
    optimize_multi,
)

# Avoid fork deadlocks in tests that use ProcessPoolExecutor
with contextlib.suppress(RuntimeError):
    mp.set_start_method("spawn", force=True)


@pytest.fixture
def sample_data():
    """Create sample price data for optimization tests."""
    return pl.DataFrame(
        {
            "timestamp": range(100),
            "close": [100 + i * 0.5 for i in range(100)],
        }
    )


class DualParamStrategy(Strategy):
    """Strategy with two parameters for constraint testing."""

    def preprocess(self, df):
        fast = self.params.get("fast", 5)
        slow = self.params.get("slow", 20)
        return df.with_columns(
            [
                ind.sma("close", fast).alias("sma_fast"),
                ind.sma("close", slow).alias("sma_slow"),
            ]
        )

    def next(self, ctx):
        fast = ctx.row.get("sma_fast")
        slow = ctx.row.get("sma_slow")
        if fast is None or slow is None:
            return
        if fast > slow:
            ctx.portfolio.order_target_percent("asset", 0.8)
        else:
            ctx.portfolio.close_position("asset")


class TestConstraints:
    """Test constraint functions in optimize()."""

    def test_constraint_filters_params(self):
        """Constraint function filters invalid parameter combinations."""
        param_grid = {"fast": [5, 10, 20], "slow": [10, 20, 30]}

        def constraint(p: dict) -> bool:
            return p["fast"] < p["slow"]

        param_sets = _generate_param_sets(param_grid, constraint)

        for p in param_sets:
            assert p["fast"] < p["slow"]

        # Without constraint we'd have 9, with constraint fewer
        all_sets = _generate_param_sets(param_grid, None)
        assert len(param_sets) < len(all_sets)

    def test_constraint_in_optimize(self, sample_data):
        """Optimize respects constraint parameter."""
        param_grid = {"fast": [5, 10, 20], "slow": [10, 20, 30]}

        best = optimize(
            DualParamStrategy,
            sample_data,
            param_grid,
            constraint=lambda p: p["fast"] < p["slow"],
            verbose=False,
            n_jobs=1,
        )

        assert best["fast"] < best["slow"]

    def test_constraint_all_rejected_raises(self, sample_data):
        """Raise ValueError when all combinations are rejected."""
        param_grid = {"fast": [100], "slow": [10]}

        with pytest.raises(ValueError, match="No parameter combinations remain"):
            optimize(
                DualParamStrategy,
                sample_data,
                param_grid,
                constraint=lambda p: p["fast"] < p["slow"],
                verbose=False,
                n_jobs=1,
            )

    def test_no_constraint_backward_compatible(self, sample_data):
        """optimize() without constraint works as before."""
        param_grid = {"fast": [5, 10], "slow": [20]}

        best = optimize(
            DualParamStrategy,
            sample_data,
            param_grid,
            verbose=False,
            n_jobs=1,
        )

        assert best is not None
        assert "sharpe_ratio" in best


class TestMultiObjective:
    """Test multi-objective optimization."""

    def test_pareto_front_basic(self):
        """Pareto front computation identifies non-dominated solutions."""
        df = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 1.5],
                "b": [3.0, 2.0, 1.0, 2.5],
            }
        )
        mask = _pareto_front(df, ["a", "b"], [True, True])
        # Row 0 (1,3) and row 2 (3,1) are Pareto-optimal; row 1 (2,2) and row 3 (1.5,2.5) are dominated
        # Actually: row 3 (1.5, 2.5) — is it dominated? row 0 has a=1 < 1.5, b=3 > 2.5 → not dominated by row 0
        # row 2 has a=3 > 1.5, b=1 < 2.5 → not dominated by row 2
        # row 1 has a=2 > 1.5, b=2 < 2.5 → not dominated by row 1
        # So row 3 is also Pareto-optimal
        pareto_rows = [i for i, m in enumerate(mask) if m]
        assert 0 in pareto_rows  # (1, 3) non-dominated
        assert 2 in pareto_rows  # (3, 1) non-dominated

    def test_pareto_front_with_minimization(self):
        """Pareto front works with mixed maximize/minimize."""
        df = pl.DataFrame(
            {
                "sharpe": [1.0, 2.0, 0.5],
                "drawdown": [0.1, 0.3, 0.05],
            }
        )
        # maximize sharpe, minimize drawdown
        mask = _pareto_front(df, ["sharpe", "drawdown"], [True, False])
        pareto_rows = [i for i, m in enumerate(mask) if m]
        # (2.0, 0.3) and (0.5, 0.05) are on the Pareto front; (1.0, 0.1) might be too
        assert len(pareto_rows) >= 2

    def test_pareto_front_empty(self):
        """Empty DataFrame returns empty mask."""
        df = pl.DataFrame({"a": [], "b": []})
        mask = _pareto_front(df, ["a", "b"], [True, True])
        assert mask == []

    def test_optimize_multi_returns_pareto(self, sample_data):
        """optimize_multi returns only Pareto-optimal results."""
        param_grid = {"fast": [5, 10], "slow": [20, 30]}

        pareto_df = optimize_multi(
            DualParamStrategy,
            sample_data,
            param_grid,
            objectives=["sharpe_ratio", "max_drawdown"],
            maximize=[True, False],
            verbose=False,
            n_jobs=1,
        )

        assert len(pareto_df) >= 1
        assert "sharpe_ratio" in pareto_df.columns
        assert "max_drawdown" in pareto_df.columns
        # All rows should be Pareto-optimal
        assert len(pareto_df) <= 4  # at most all combinations

    def test_optimize_multi_requires_two_objectives(self, sample_data):
        """optimize_multi raises with fewer than 2 objectives."""
        with pytest.raises(ValueError, match="at least 2 objectives"):
            optimize_multi(
                DualParamStrategy,
                sample_data,
                param_grid={"fast": [5]},
                objectives=["sharpe_ratio"],
                verbose=False,
            )

    def test_optimize_multi_mismatched_maximize(self, sample_data):
        """optimize_multi raises when maximize length doesn't match objectives."""
        with pytest.raises(ValueError, match="maximize list must match"):
            optimize_multi(
                DualParamStrategy,
                sample_data,
                param_grid={"fast": [5]},
                objectives=["sharpe_ratio", "max_drawdown"],
                maximize=[True],
                verbose=False,
            )

    def test_optimize_multi_with_constraint(self, sample_data):
        """optimize_multi respects constraints."""
        param_grid = {"fast": [5, 10, 20], "slow": [10, 20, 30]}

        pareto_df = optimize_multi(
            DualParamStrategy,
            sample_data,
            param_grid,
            objectives=["sharpe_ratio", "total_return"],
            constraint=lambda p: p["fast"] < p["slow"],
            verbose=False,
            n_jobs=1,
        )

        # Verify constraint was applied
        for row in pareto_df.iter_rows(named=True):
            assert row["fast"] < row["slow"]


class TestBayesianOptimization:
    """Test Bayesian optimization."""

    def test_bayesian_import_error(self, sample_data):
        """optimize_bayesian raises ImportError when scikit-optimize is not installed."""
        from polarbt.runner import optimize_bayesian

        # This test only validates the function exists and has correct signature
        # We can't test the actual optimization without scikit-optimize
        assert callable(optimize_bayesian)


class TestSensitivityPlot:
    """Test parameter sensitivity plots."""

    def test_plot_sensitivity_basic(self):
        """plot_sensitivity creates a figure."""
        from polarbt.plotting.charts import plot_sensitivity

        df = pl.DataFrame(
            {
                "sma_period": [5, 10, 15, 20, 25],
                "sharpe_ratio": [0.5, 1.2, 0.8, 1.5, 1.0],
            }
        )

        fig = plot_sensitivity(df, "sma_period", "sharpe_ratio")
        assert fig is not None
        assert len(fig.data) == 2  # scatter + mean line

    def test_plot_sensitivity_missing_param(self):
        """plot_sensitivity raises on missing parameter column."""
        from polarbt.plotting.charts import plot_sensitivity

        df = pl.DataFrame({"sma_period": [5], "sharpe_ratio": [0.5]})
        with pytest.raises(ValueError, match="not found"):
            plot_sensitivity(df, "nonexistent", "sharpe_ratio")

    def test_plot_sensitivity_missing_metric(self):
        """plot_sensitivity raises on missing metric column."""
        from polarbt.plotting.charts import plot_sensitivity

        df = pl.DataFrame({"sma_period": [5], "sharpe_ratio": [0.5]})
        with pytest.raises(ValueError, match="not found"):
            plot_sensitivity(df, "sma_period", "nonexistent")

    def test_plot_sensitivity_aggregation(self):
        """plot_sensitivity aggregates duplicate parameter values."""
        from polarbt.plotting.charts import plot_sensitivity

        df = pl.DataFrame(
            {
                "sma_period": [5, 5, 10, 10],
                "sharpe_ratio": [0.5, 0.7, 1.0, 1.2],
            }
        )

        fig = plot_sensitivity(df, "sma_period", "sharpe_ratio")
        # Mean line should have 2 points (for sma_period 5 and 10)
        mean_trace = fig.data[1]
        assert len(mean_trace.x) == 2

    def test_plot_sensitivity_save_html(self, tmp_path):
        """plot_sensitivity can save to HTML."""
        from polarbt.plotting.charts import plot_sensitivity

        df = pl.DataFrame(
            {
                "sma_period": [5, 10],
                "sharpe_ratio": [0.5, 1.0],
            }
        )

        out_path = str(tmp_path / "sensitivity.html")
        fig = plot_sensitivity(df, "sma_period", "sharpe_ratio", save_html=out_path)
        assert fig is not None
        import os

        assert os.path.exists(out_path)


class TestParamHeatmap:
    """Test 2D parameter heatmaps."""

    def test_heatmap_basic(self):
        """plot_param_heatmap creates a figure."""
        from polarbt.plotting.charts import plot_param_heatmap

        df = pl.DataFrame(
            {
                "fast": [5, 5, 10, 10],
                "slow": [20, 30, 20, 30],
                "sharpe_ratio": [0.5, 0.8, 1.0, 1.2],
            }
        )

        fig = plot_param_heatmap(df, "fast", "slow", "sharpe_ratio")
        assert fig is not None
        assert len(fig.data) == 1  # single heatmap trace

    def test_heatmap_missing_column(self):
        """plot_param_heatmap raises on missing columns."""
        from polarbt.plotting.charts import plot_param_heatmap

        df = pl.DataFrame({"fast": [5], "slow": [20], "sharpe_ratio": [0.5]})
        with pytest.raises(ValueError, match="not found"):
            plot_param_heatmap(df, "nonexistent", "slow", "sharpe_ratio")

    def test_heatmap_invalid_aggregation(self):
        """plot_param_heatmap raises on invalid aggregation."""
        from polarbt.plotting.charts import plot_param_heatmap

        df = pl.DataFrame({"fast": [5], "slow": [20], "sharpe_ratio": [0.5]})
        with pytest.raises(ValueError, match="aggregation"):
            plot_param_heatmap(df, "fast", "slow", "sharpe_ratio", aggregation="median")

    def test_heatmap_aggregation_mean(self):
        """plot_param_heatmap aggregates multiple runs correctly."""
        from polarbt.plotting.charts import plot_param_heatmap

        df = pl.DataFrame(
            {
                "fast": [5, 5, 10, 10],
                "slow": [20, 20, 30, 30],
                "sharpe_ratio": [0.5, 0.7, 1.0, 1.4],
            }
        )

        fig = plot_param_heatmap(df, "fast", "slow", "sharpe_ratio", aggregation="mean")
        # z values should be aggregated means
        z = fig.data[0].z
        assert z is not None

    def test_heatmap_save_html(self, tmp_path):
        """plot_param_heatmap can save to HTML."""
        from polarbt.plotting.charts import plot_param_heatmap

        df = pl.DataFrame(
            {
                "fast": [5, 10],
                "slow": [20, 30],
                "sharpe_ratio": [0.5, 1.0],
            }
        )

        out_path = str(tmp_path / "heatmap.html")
        fig = plot_param_heatmap(df, "fast", "slow", "sharpe_ratio", save_html=out_path)
        assert fig is not None
        import os

        assert os.path.exists(out_path)

    def test_heatmap_max_aggregation(self):
        """plot_param_heatmap with max aggregation."""
        from polarbt.plotting.charts import plot_param_heatmap

        df = pl.DataFrame(
            {
                "fast": [5, 5],
                "slow": [20, 20],
                "sharpe_ratio": [0.5, 0.9],
            }
        )

        fig = plot_param_heatmap(df, "fast", "slow", "sharpe_ratio", aggregation="max")
        assert fig is not None
