"""Tests for automatic warmup functionality."""

from typing import Any

import polars as pl
import pytest

from polarbt import backtest
from polarbt import indicators as ind
from polarbt.core import BacktestContext, Engine, Strategy


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """Create sample price data."""
    return pl.DataFrame(
        {
            "timestamp": range(100),
            "close": [100.0 + i * 0.5 for i in range(100)],
        }
    )


class TrackingStrategy(Strategy):
    """Strategy that tracks when it first executes."""

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.first_execution_bar: int | None = None
        self.execution_count = 0

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add indicators with configurable period."""
        period = self.params.get("period", 20)
        return df.with_columns(
            [
                ind.sma("close", period).alias("sma"),
                ind.rsi("close", 14).alias("rsi"),
            ]
        )

    def next(self, ctx: BacktestContext) -> None:
        """Track first execution."""
        if self.first_execution_bar is None:
            self.first_execution_bar = ctx.bar_index
        self.execution_count += 1


class NoIndicatorStrategy(Strategy):
    """Strategy without any indicators."""

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.first_execution_bar: int | None = None

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df  # No indicators

    def next(self, ctx: BacktestContext) -> None:
        if self.first_execution_bar is None:
            self.first_execution_bar = ctx.bar_index


class MultiIndicatorStrategy(Strategy):
    """Strategy with multiple indicators of different periods."""

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.first_execution_bar: int | None = None

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            [
                ind.sma("close", 10).alias("sma_10"),
                ind.sma("close", 30).alias("sma_30"),
                ind.rsi("close", 14).alias("rsi_14"),
            ]
        )

    def next(self, ctx: BacktestContext) -> None:
        if self.first_execution_bar is None:
            self.first_execution_bar = ctx.bar_index


class TestAutoWarmup:
    """Test automatic warmup calculation."""

    def test_auto_warmup_default(self, sample_data: pl.DataFrame) -> None:
        """Test that auto warmup is the default."""
        strategy = TrackingStrategy()
        engine = Engine(strategy, sample_data)

        # Verify auto is the default
        assert engine.warmup == "auto"

    def test_auto_warmup_calculation_sma_20(self, sample_data: pl.DataFrame) -> None:
        """Test auto warmup with 20-period SMA."""
        strategy = TrackingStrategy(period=20)
        engine = Engine(strategy, sample_data, warmup="auto")

        # Preprocess data
        processed = strategy.preprocess(sample_data)

        # Calculate warmup
        warmup_period = engine._calculate_auto_warmup(processed)

        # 20-period SMA needs 19 bars of history
        assert warmup_period == 19

    def test_auto_warmup_calculation_sma_50(self, sample_data: pl.DataFrame) -> None:
        """Test auto warmup with 50-period SMA."""
        strategy = TrackingStrategy(period=50)
        engine = Engine(strategy, sample_data, warmup="auto")

        processed = strategy.preprocess(sample_data)
        warmup_period = engine._calculate_auto_warmup(processed)

        # 50-period SMA needs 49 bars of history
        assert warmup_period == 49

    def test_auto_warmup_no_indicators(self, sample_data: pl.DataFrame) -> None:
        """Test auto warmup with no indicators."""
        strategy = NoIndicatorStrategy()
        engine = Engine(strategy, sample_data, warmup="auto")

        processed = strategy.preprocess(sample_data)
        warmup_period = engine._calculate_auto_warmup(processed)

        # No indicators, should be 0
        assert warmup_period == 0

    def test_auto_warmup_multiple_indicators(self, sample_data: pl.DataFrame) -> None:
        """Test auto warmup with multiple indicators."""
        strategy = MultiIndicatorStrategy()
        engine = Engine(strategy, sample_data, warmup="auto")

        processed = strategy.preprocess(sample_data)
        warmup_period = engine._calculate_auto_warmup(processed)

        # Should wait for slowest indicator (30-period SMA)
        assert warmup_period == 29

    def test_auto_warmup_excludes_timestamp(self, sample_data: pl.DataFrame) -> None:
        """Test that timestamp columns are excluded from warmup check."""

        class TimestampStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                # Add various timestamp column names
                return df.with_columns(
                    [
                        ind.sma("close", 20).alias("sma"),
                    ]
                )

            def next(self, ctx: BacktestContext) -> None:
                pass

        strategy = TimestampStrategy()
        engine = Engine(strategy, sample_data, warmup="auto")

        processed = strategy.preprocess(sample_data)
        warmup_period = engine._calculate_auto_warmup(processed)

        # Should ignore timestamp column and only check indicators
        assert warmup_period == 19


class TestWarmupExecution:
    """Test that strategies execute at the correct bar."""

    def test_auto_warmup_execution_start(self, sample_data: pl.DataFrame) -> None:
        """Test strategy starts at correct bar with auto warmup."""
        strategy = TrackingStrategy(period=20)
        backtest(TrackingStrategy, sample_data, params={"period": 20})

        # Should start at bar 19 (first bar where SMA is available)
        # Note: We can't easily access the strategy instance from backtest()
        # So we test via the engine directly
        engine = Engine(strategy, sample_data, warmup="auto")
        engine.run()

        assert strategy.first_execution_bar == 19

    def test_manual_warmup_execution(self, sample_data: pl.DataFrame) -> None:
        """Test strategy starts at correct bar with manual warmup."""
        strategy = TrackingStrategy()
        engine = Engine(strategy, sample_data, warmup=10)
        engine.run()

        assert strategy.first_execution_bar == 10

    def test_zero_warmup_execution(self, sample_data: pl.DataFrame) -> None:
        """Test strategy starts at bar 0 with no warmup."""
        strategy = TrackingStrategy()
        engine = Engine(strategy, sample_data, warmup=0)
        engine.run()

        assert strategy.first_execution_bar == 0

    def test_no_indicators_starts_immediately(self, sample_data: pl.DataFrame) -> None:
        """Test strategy with no indicators starts at bar 0."""
        strategy = NoIndicatorStrategy()
        engine = Engine(strategy, sample_data, warmup="auto")
        engine.run()

        assert strategy.first_execution_bar == 0

    def test_execution_count_with_warmup(self, sample_data: pl.DataFrame) -> None:
        """Test that warmup reduces execution count."""
        # With auto warmup (should skip ~19 bars)
        strategy_auto = TrackingStrategy(period=20)
        engine_auto = Engine(strategy_auto, sample_data, warmup="auto")
        engine_auto.run()

        # With no warmup
        strategy_no = TrackingStrategy(period=20)
        engine_no = Engine(strategy_no, sample_data, warmup=0)
        engine_no.run()

        # Auto warmup should execute fewer times
        assert strategy_auto.execution_count < strategy_no.execution_count
        assert strategy_auto.execution_count == len(sample_data) - 19
        assert strategy_no.execution_count == len(sample_data)


class TestWarmupValidation:
    """Test warmup parameter validation."""

    def test_valid_auto_string(self, sample_data: pl.DataFrame) -> None:
        """Test that 'auto' is accepted."""
        strategy = TrackingStrategy()
        engine = Engine(strategy, sample_data, warmup="auto")
        assert engine.warmup == "auto"

    def test_valid_integer(self, sample_data: pl.DataFrame) -> None:
        """Test that integer values are accepted."""
        strategy = TrackingStrategy()
        engine = Engine(strategy, sample_data, warmup=10)
        assert engine.warmup == 10

    def test_invalid_string_rejected(self, sample_data: pl.DataFrame) -> None:
        """Test that invalid strings are rejected."""
        strategy = TrackingStrategy()
        with pytest.raises(ValueError, match="warmup must be an integer or 'auto'"):
            Engine(strategy, sample_data, warmup="invalid")

    def test_float_rejected(self, sample_data: pl.DataFrame) -> None:
        """Test that float values are rejected."""
        strategy = TrackingStrategy()
        with pytest.raises(ValueError, match="warmup must be an integer or 'auto'"):
            Engine(strategy, sample_data, warmup=3.14)  # type: ignore[arg-type]

    def test_none_rejected(self, sample_data: pl.DataFrame) -> None:
        """Test that None is rejected."""
        strategy = TrackingStrategy()
        with pytest.raises(ValueError, match="warmup must be an integer or 'auto'"):
            Engine(strategy, sample_data, warmup=None)  # type: ignore[arg-type]


class TestWarmupWithBacktest:
    """Test warmup through the backtest() function."""

    def test_backtest_auto_warmup_default(self, sample_data: pl.DataFrame) -> None:
        """Test backtest uses auto warmup by default."""
        results = backtest(TrackingStrategy, sample_data, params={"period": 20})
        assert results.success

    def test_backtest_manual_warmup(self, sample_data: pl.DataFrame) -> None:
        """Test backtest with manual warmup."""
        results = backtest(TrackingStrategy, sample_data, params={"period": 20}, warmup=15)
        assert results.success

    def test_backtest_invalid_warmup(self, sample_data: pl.DataFrame) -> None:
        """Test backtest rejects invalid warmup."""
        results = backtest(TrackingStrategy, sample_data, warmup="invalid")
        assert not results.success
        assert results.error is not None
        assert "warmup must be an integer or 'auto'" in results.error


class TestWarmupEdgeCases:
    """Test edge cases for warmup."""

    def test_warmup_larger_than_data(self, sample_data: pl.DataFrame) -> None:
        """Test warmup period larger than dataset."""
        short_data = sample_data.head(10)
        strategy = TrackingStrategy()
        engine = Engine(strategy, short_data, warmup=20)
        engine.run()

        # Should handle gracefully - strategy never executes
        assert strategy.first_execution_bar is None

    def test_all_null_indicators(self) -> None:
        """Test data where indicators are always null."""
        data = pl.DataFrame(
            {
                "timestamp": range(10),
                "close": [None] * 10,
            }
        )

        # Use a strategy without RSI to avoid polars errors with null data
        class SimpleNullStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df.with_columns(
                    [
                        ind.sma("close", 5).alias("sma"),
                    ]
                )

            def next(self, ctx: BacktestContext) -> None:
                pass

        strategy = SimpleNullStrategy()
        engine = Engine(strategy, data, warmup="auto")

        processed = strategy.preprocess(data)
        warmup_period = engine._calculate_auto_warmup(processed)

        # Should return max warmup (len - 1) since all values are null
        assert warmup_period == 9

    def test_empty_dataframe(self) -> None:
        """Test with empty dataframe."""
        data = pl.DataFrame(
            {
                "timestamp": [],
                "close": [],
            }
        )

        # Use simple strategy without RSI
        class SimpleEmptyStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df.with_columns(
                    [
                        ind.sma("close", 5).alias("sma"),
                    ]
                )

            def next(self, ctx: BacktestContext) -> None:
                pass

        strategy = SimpleEmptyStrategy()
        engine = Engine(strategy, data, warmup="auto")

        processed = strategy.preprocess(data)
        warmup_period = engine._calculate_auto_warmup(processed)

        # Should return 0 for empty data
        assert warmup_period == 0

    def test_single_row_data(self) -> None:
        """Test with single row of data."""
        data = pl.DataFrame(
            {
                "timestamp": [0],
                "close": [100.0],
            }
        )

        strategy = NoIndicatorStrategy()
        engine = Engine(strategy, data, warmup="auto")
        engine.run()

        # Should execute once
        assert strategy.first_execution_bar == 0

    def test_partial_null_data(self) -> None:
        """Test data with some null values in the middle."""
        data = pl.DataFrame(
            {
                "timestamp": range(50),
                "close": [100.0 + i if i < 45 else None for i in range(50)],
            }
        )

        strategy = TrackingStrategy(period=10)
        engine = Engine(strategy, data, warmup="auto")

        processed = strategy.preprocess(data)
        warmup_period = engine._calculate_auto_warmup(processed)

        # Should find first non-null row after indicator warmup
        assert warmup_period >= 9  # At least the indicator period


class TestWarmupWithMultiAsset:
    """Test warmup with multiple assets."""

    def test_multi_asset_auto_warmup(self) -> None:
        """Test auto warmup with multiple assets."""
        data = {
            "BTC": pl.DataFrame(
                {
                    "timestamp": range(50),
                    "close": [50000.0 + i * 100 for i in range(50)],
                }
            ),
            "ETH": pl.DataFrame(
                {
                    "timestamp": range(50),
                    "close": [3000.0 + i * 10 for i in range(50)],
                }
            ),
        }

        class MultiAssetTracker(Strategy):
            def __init__(self, **params: Any) -> None:
                super().__init__(**params)
                self.first_bar: int | None = None

            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df.with_columns(
                    ind.sma("close", 20).over("symbol").alias("sma"),
                )

            def next(self, ctx: BacktestContext) -> None:
                if self.first_bar is None:
                    self.first_bar = ctx.bar_index

        results = backtest(MultiAssetTracker, data, params={})

        assert results.success


class TestWarmupIntegration:
    """Integration tests for warmup feature."""

    def test_strategy_without_none_checks(self, sample_data: pl.DataFrame) -> None:
        """Test that strategies don't need None checks with auto warmup."""

        class NoNoneCheckStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df.with_columns(
                    [
                        ind.sma("close", 20).alias("sma"),
                    ]
                )

            def next(self, ctx: BacktestContext) -> None:
                # No None checks - should work with auto warmup
                if ctx.row["close"] > ctx.row["sma"]:
                    ctx.portfolio.order_target_percent("asset", 1.0)
                else:
                    ctx.portfolio.close_position("asset")

        results = backtest(NoNoneCheckStrategy, sample_data)
        assert results.success

    def test_strategy_with_manual_warmup_needs_checks(self, sample_data: pl.DataFrame) -> None:
        """Test that manual warmup=0 requires None checks."""

        class NoNoneCheckStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df.with_columns(
                    [
                        ind.sma("close", 20).alias("sma"),
                    ]
                )

            def next(self, ctx: BacktestContext) -> None:
                # No None checks - will fail with warmup=0
                if ctx.row["close"] > ctx.row["sma"]:
                    ctx.portfolio.order_target_percent("asset", 1.0)

        # This should fail because SMA is None at early bars
        results = backtest(NoNoneCheckStrategy, sample_data, warmup=0)
        assert not results.success

    def test_warmup_consistency_across_runs(self, sample_data: pl.DataFrame) -> None:
        """Test that warmup is consistent across multiple runs."""
        results1 = backtest(TrackingStrategy, sample_data, params={"period": 20})
        results2 = backtest(TrackingStrategy, sample_data, params={"period": 20})

        # Results should be identical
        assert results1.total_return == results2.total_return
        assert results1.final_equity == results2.final_equity
