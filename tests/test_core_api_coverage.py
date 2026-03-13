"""Tests for core.py methods that had no direct test coverage.

Covers: order_target_value, close_all_positions, cancel_order,
Strategy.on_start, Strategy.on_finish.
"""

import polars as pl
import pytest

from polarbt.core import BacktestContext, Engine, Portfolio, Strategy


def _make_data(n: int = 50) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": list(range(n)),
            "open": [100.0 + i * 0.5 for i in range(n)],
            "high": [101.0 + i * 0.5 for i in range(n)],
            "low": [99.0 + i * 0.5 for i in range(n)],
            "close": [100.0 + i * 0.5 for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# order_target_value
# ---------------------------------------------------------------------------


class TestOrderTargetValue:
    def test_buy_to_target_value(self):
        """Ordering target_value=10000 at price~102.5 should buy ~97 shares."""

        class ValueStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def next(self, ctx: BacktestContext) -> None:
                if ctx.bar_index == 5:
                    ctx.portfolio.order_target_value("asset", 10_000.0)

        engine = Engine(ValueStrategy(), _make_data(), initial_cash=100_000, commission=0.0, slippage=0.0)
        results = engine.run()

        pos = results.final_positions.get("asset", 0.0)
        price_at_5 = 100.0 + 5 * 0.5  # 102.5
        expected_qty = 10_000.0 / price_at_5
        # Position should be close to expected
        assert abs(pos - expected_qty) < expected_qty * 0.05

    def test_sell_down_to_target(self):
        """If holding more than target value, should sell to reach target."""

        class SellDownStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def next(self, ctx: BacktestContext) -> None:
                if ctx.bar_index == 2:
                    ctx.portfolio.order("asset", 200.0)
                elif ctx.bar_index == 10:
                    ctx.portfolio.order_target_value("asset", 5_000.0)

        engine = Engine(SellDownStrategy(), _make_data(), initial_cash=100_000, commission=0.0, slippage=0.0)
        results = engine.run()
        pos = results.final_positions.get("asset", 0.0)
        price_at_10 = 100.0 + 10 * 0.5  # 105.0
        expected_qty = 5_000.0 / price_at_10
        assert pos == pytest.approx(expected_qty, abs=1.0)

    def test_returns_none_for_zero_price(self):
        portfolio = Portfolio(initial_cash=100_000)
        # No current price set -> returns None
        result = portfolio.order_target_value("unknown_asset", 10_000.0)
        assert result is None


# ---------------------------------------------------------------------------
# close_all_positions
# ---------------------------------------------------------------------------


class TestCloseAllPositions:
    def test_closes_multiple_assets(self):
        """close_all_positions should close all open positions."""

        class MultiOpenStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def next(self, ctx: BacktestContext) -> None:
                if ctx.bar_index == 2:
                    ctx.portfolio.order("BTC", 1.0)
                    ctx.portfolio.order("ETH", 10.0)
                elif ctx.bar_index == 10:
                    ctx.portfolio.close_all_positions()

        btc_df = pl.DataFrame({"timestamp": list(range(30)), "close": [50_000.0 + i * 100 for i in range(30)]})
        eth_df = pl.DataFrame({"timestamp": list(range(30)), "close": [3_000.0 + i * 10 for i in range(30)]})
        engine = Engine(MultiOpenStrategy(), {"BTC": btc_df, "ETH": eth_df}, initial_cash=200_000)
        results = engine.run()

        # All positions should be 0 at the end
        for asset, qty in results.final_positions.items():
            assert qty == 0.0, f"{asset} position should be 0 after close_all_positions"


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------


class TestCancelOrder:
    def test_cancel_pending_limit_order(self):
        """Cancelling a pending limit order should return True."""

        class CancelStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def next(self, ctx: BacktestContext) -> None:
                if ctx.bar_index == 2:
                    # Place a limit order far from market price (won't fill)
                    oid = ctx.portfolio.order("asset", 10.0, limit_price=50.0)
                    if oid:
                        self._limit_order_id = oid  # type: ignore[attr-defined]
                elif ctx.bar_index == 5:
                    oid = getattr(self, "_limit_order_id", None)
                    if oid:
                        cancelled = ctx.portfolio.cancel_order(oid)
                        self._cancel_result = cancelled  # type: ignore[attr-defined]

        strategy = CancelStrategy()
        engine = Engine(strategy, _make_data(), initial_cash=100_000)
        engine.run()
        assert getattr(strategy, "_cancel_result", None) is True

    def test_cancel_nonexistent_order(self):
        portfolio = Portfolio(initial_cash=100_000)
        assert portfolio.cancel_order("nonexistent_id") is False

    def test_cancel_already_filled_order(self):
        """Cannot cancel a filled order."""

        class FilledCancelStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def next(self, ctx: BacktestContext) -> None:
                if ctx.bar_index == 2:
                    oid = ctx.portfolio.order("asset", 10.0)
                    if oid:
                        self._market_order_id = oid  # type: ignore[attr-defined]
                elif ctx.bar_index == 5:
                    oid = getattr(self, "_market_order_id", None)
                    if oid:
                        cancelled = ctx.portfolio.cancel_order(oid)
                        self._cancel_result = cancelled  # type: ignore[attr-defined]

        strategy = FilledCancelStrategy()
        engine = Engine(strategy, _make_data(), initial_cash=100_000)
        engine.run()
        # Market order fills immediately, cancel should fail
        assert getattr(strategy, "_cancel_result", None) is False


# ---------------------------------------------------------------------------
# Strategy.on_start / Strategy.on_finish
# ---------------------------------------------------------------------------


class TestOnStartOnFinish:
    def test_on_start_called_before_first_bar(self):
        """on_start should be called once before the first next() call."""

        class StartTracker(Strategy):
            def __init__(self) -> None:
                super().__init__()
                self.start_called = False
                self.start_cash: float = 0.0
                self.first_bar_start_was_called = False

            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def on_start(self, portfolio: Portfolio) -> None:
                self.start_called = True
                self.start_cash = portfolio.cash

            def next(self, ctx: BacktestContext) -> None:
                if ctx.bar_index == 0:
                    self.first_bar_start_was_called = self.start_called

        strategy = StartTracker()
        Engine(strategy, _make_data(), initial_cash=50_000).run()

        assert strategy.start_called is True
        assert strategy.start_cash == 50_000.0
        assert strategy.first_bar_start_was_called is True

    def test_on_finish_called_after_last_bar(self):
        """on_finish should be called once after all bars processed."""

        class FinishTracker(Strategy):
            def __init__(self) -> None:
                super().__init__()
                self.finish_called = False
                self.bar_count = 0

            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def on_finish(self, portfolio: Portfolio) -> None:
                self.finish_called = True

            def next(self, ctx: BacktestContext) -> None:
                self.bar_count += 1

        strategy = FinishTracker()
        Engine(strategy, _make_data(n=20), initial_cash=50_000).run()

        assert strategy.finish_called is True
        assert strategy.bar_count > 0

    def test_on_start_and_finish_default_noop(self):
        """Default on_start/on_finish should not raise."""

        class NoopStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def next(self, ctx: BacktestContext) -> None:
                pass

        engine = Engine(NoopStrategy(), _make_data(n=10), initial_cash=50_000)
        results = engine.run()
        assert results is not None


# ---------------------------------------------------------------------------
# _RowAccessor.get() fix
# ---------------------------------------------------------------------------


class TestRowAccessorGet:
    def test_get_returns_default_for_multi_symbol(self):
        """_RowAccessor.get() should return default for multi-symbol mode."""

        class MultiAccessStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def next(self, ctx: BacktestContext) -> None:
                result = ctx.row.get("nonexistent", "default_val")
                self._get_result = result  # type: ignore[attr-defined]

        btc_df = pl.DataFrame({"timestamp": list(range(10)), "close": [50_000.0 + i * 100 for i in range(10)]})
        eth_df = pl.DataFrame({"timestamp": list(range(10)), "close": [3_000.0 + i * 10 for i in range(10)]})

        strategy = MultiAccessStrategy()
        Engine(strategy, {"BTC": btc_df, "ETH": eth_df}, initial_cash=200_000).run()
        assert getattr(strategy, "_get_result", None) == "default_val"
