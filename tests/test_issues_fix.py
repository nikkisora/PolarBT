"""Tests for fixes to issues discovered during MOEX M5 integration testing.

Covers:
- Issue 1: Order dict corruption on micro-price stocks (purging, snapshot iteration)
- Issue 2: Segfault on sequential backtests (Engine.cleanup)
- Issue 3: Concurrent dict mutation in stop-loss fill path (two-phase stops)
"""

import gc

import numpy as np
import polars as pl

from polarbt import indicators as ind
from polarbt.core import BacktestContext, Engine, Portfolio, Strategy
from polarbt.orders import OrderStatus

# ---------------------------------------------------------------------------
# Helper strategy used across tests
# ---------------------------------------------------------------------------


class SimpleEMA(Strategy):
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            ind.ema("close", 12).alias("ema_fast"),
            ind.ema("close", 36).alias("ema_slow"),
        )
        df = df.with_columns(
            ind.crossover("ema_fast", "ema_slow").alias("buy"),
            ind.crossunder("ema_fast", "ema_slow").alias("sell"),
        )
        return df

    def next(self, ctx: BacktestContext) -> None:
        pos = ctx.portfolio.get_position("asset")
        if ctx.row.get("buy") and pos <= 0:
            ctx.portfolio.order_target_percent("asset", 0.95)
        elif ctx.row.get("sell") and pos > 0:
            ctx.portfolio.close_position("asset")


class WithStops(Strategy):
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            ind.ema("close", 12).alias("ema_fast"),
            ind.ema("close", 36).alias("ema_slow"),
            ind.atr("high", "low", "close", 14).alias("atr"),
        )
        df = df.with_columns(
            ind.crossover("ema_fast", "ema_slow").alias("buy"),
            ind.crossunder("ema_fast", "ema_slow").alias("sell"),
        )
        return df

    def next(self, ctx: BacktestContext) -> None:
        pos = ctx.portfolio.get_position("asset")
        price = ctx.row["close"]
        atr = ctx.row.get("atr")
        if ctx.row.get("buy") and pos <= 0:
            ctx.portfolio.order_target_percent("asset", 0.95)
            if atr and atr > 0:
                ctx.portfolio.set_stop_loss("asset", stop_price=price - 2.0 * atr)
        elif ctx.row.get("sell") and pos > 0:
            ctx.portfolio.close_position("asset")


def _make_micro_price_data(n: int = 5000, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic micro-price data similar to VTBR."""
    rng = np.random.default_rng(seed)
    close = np.cumsum(rng.standard_normal(n) * 0.0001) + 0.03
    close = np.clip(close, 0.005, 0.1)
    return pl.DataFrame(
        {
            "timestamp": pl.Series(range(n)),
            "open": pl.Series(close + rng.standard_normal(n) * 0.0001),
            "high": pl.Series(close + np.abs(rng.standard_normal(n) * 0.0005)),
            "low": pl.Series(close - np.abs(rng.standard_normal(n) * 0.0005)),
            "close": pl.Series(close),
            "volume": pl.Series(rng.integers(100_000, 10_000_000, n).astype(float)),
        }
    )


def _make_normal_data(n: int = 500, base_price: float = 100.0, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic normal-price data for sequential backtest tests."""
    rng = np.random.default_rng(seed)
    close = np.cumsum(rng.standard_normal(n) * 0.5) + base_price
    close = np.clip(close, 10.0, 500.0)
    return pl.DataFrame(
        {
            "timestamp": pl.Series(range(n)),
            "open": pl.Series(close + rng.standard_normal(n) * 0.1),
            "high": pl.Series(close + np.abs(rng.standard_normal(n) * 0.5)),
            "low": pl.Series(close - np.abs(rng.standard_normal(n) * 0.5)),
            "close": pl.Series(close),
            "volume": pl.Series(rng.integers(1_000, 100_000, n).astype(float)),
        }
    )


# ---------------------------------------------------------------------------
# Issue 1: Order dict corruption / purging
# ---------------------------------------------------------------------------


class TestOrderPurging:
    """Verify that inactive orders are purged to prevent unbounded dict growth."""

    def test_purge_removes_filled_orders(self):
        """Portfolio._purge_inactive_orders removes non-active orders."""
        portfolio = Portfolio(initial_cash=100_000)
        portfolio.update_prices(
            {"A": 100.0},
            bar_index=0,
            ohlc_data={"A": {"open": 100, "high": 100, "low": 100, "close": 100}},
        )

        # Create and fill several orders
        for _ in range(10):
            portfolio.order("A", 1.0)
            portfolio.order("A", -1.0)

        filled_count = sum(1 for o in portfolio.orders.values() if o.is_filled())
        assert filled_count > 0

        portfolio._purge_inactive_orders()

        # Only active orders remain (should be 0 since all were filled)
        assert all(o.is_active() for o in portfolio.orders.values())

    def test_purge_keeps_active_orders(self):
        """Purging should not remove pending/active orders."""
        portfolio = Portfolio(initial_cash=100_000, order_delay=1)
        portfolio.update_prices(
            {"A": 100.0},
            bar_index=0,
            ohlc_data={"A": {"open": 100, "high": 100, "low": 100, "close": 100}},
        )

        # Place an order with delay — stays PENDING
        oid = portfolio.order("A", 1.0)
        assert oid is not None
        assert portfolio.orders[oid].status == OrderStatus.PENDING

        portfolio._purge_inactive_orders()

        # Pending order should survive purge
        assert oid in portfolio.orders

    def test_auto_purge_triggers_on_threshold(self):
        """Orders dict is automatically purged when exceeding threshold."""
        portfolio = Portfolio(initial_cash=1_000_000)
        portfolio.update_prices(
            {"A": 100.0},
            bar_index=0,
            ohlc_data={"A": {"open": 100, "high": 100, "low": 100, "close": 100}},
        )

        # Create many orders to exceed the purge threshold
        for _ in range(600):
            portfolio.order("A", 1.0)
            portfolio.order("A", -1.0)

        # At this point orders should have been auto-purged during update_prices
        # but we created them directly via order(), not through update_prices
        # Let's trigger update_prices to run the purge
        portfolio.update_prices(
            {"A": 100.0},
            bar_index=1,
            ohlc_data={"A": {"open": 100, "high": 100, "low": 100, "close": 100}},
        )

        # After purge, only active orders should remain
        assert len(portfolio.orders) <= portfolio._MAX_ORDERS_BEFORE_PURGE

    def test_micro_price_backtest_completes(self):
        """Backtest on micro-price data completes without crashing (Issue 1 reproducer)."""
        df = _make_micro_price_data(n=5000)

        engine = Engine(
            strategy=SimpleEMA(),
            data=df,
            initial_cash=100_000,
            commission=0.001,
            slippage=0.0005,
            warmup="auto",
        )
        results = engine.run()

        assert results is not None
        assert results.total_return is not None

    def test_micro_price_orders_dict_bounded(self):
        """Orders dict stays bounded during micro-price backtest."""
        df = _make_micro_price_data(n=5000)

        engine = Engine(
            strategy=SimpleEMA(),
            data=df,
            initial_cash=100_000,
            commission=0.001,
            slippage=0.0005,
            warmup="auto",
        )
        engine.run()

        assert engine.portfolio is not None
        # Orders dict should be bounded, not grow to thousands
        assert len(engine.portfolio.orders) <= Portfolio._MAX_ORDERS_BEFORE_PURGE


# ---------------------------------------------------------------------------
# Issue 2: Segfault on sequential backtests (Engine.cleanup)
# ---------------------------------------------------------------------------


class TestEngineCleanup:
    """Verify Engine.cleanup releases memory between sequential runs."""

    def test_cleanup_releases_portfolio(self):
        """Engine.cleanup sets portfolio to None."""
        df = _make_normal_data(n=200)
        engine = Engine(strategy=SimpleEMA(), data=df, warmup="auto")
        engine.run()

        assert engine.portfolio is not None
        assert engine.processed_data is not None
        assert engine.results is not None

        engine.cleanup()

        assert engine.portfolio is None
        assert engine.processed_data is None
        assert engine.results is None

    def test_cleanup_allows_rerun(self):
        """Engine can be run again after cleanup."""
        df = _make_normal_data(n=200)
        engine = Engine(strategy=SimpleEMA(), data=df, warmup="auto")

        r1 = engine.run()
        engine.cleanup()

        r2 = engine.run()
        assert r2 is not None
        assert r2.total_return is not None

        # Results should be identical since same data and strategy
        assert abs(r1.total_return - r2.total_return) < 1e-10

    def test_sequential_backtests_with_cleanup(self):
        """Multiple sequential backtests with cleanup don't crash."""
        seeds = [42, 123, 456, 789, 1011]

        for seed in seeds:
            df = _make_normal_data(n=300, seed=seed)
            engine = Engine(
                strategy=SimpleEMA(),
                data=df,
                initial_cash=100_000,
                commission=0.001,
                slippage=0.0005,
                warmup="auto",
            )
            results = engine.run()
            assert results is not None
            engine.cleanup()

        # Force GC to verify no dangling references cause issues
        gc.collect()

    def test_run_clears_previous_state(self):
        """Engine.run() clears processed_data from previous runs."""
        df = _make_normal_data(n=200)
        engine = Engine(strategy=SimpleEMA(), data=df, warmup="auto")

        engine.run()
        first_processed_id = id(engine.processed_data)

        engine.run()
        # processed_data should be freshly created (previous one cleared)
        assert engine.processed_data is not None
        assert id(engine.processed_data) != first_processed_id


# ---------------------------------------------------------------------------
# Issue 3: Two-phase stop checking
# ---------------------------------------------------------------------------


class TestTwoPhaseStops:
    """Verify stop-loss/TP/trailing stops work correctly with two-phase approach."""

    def test_stop_loss_still_triggers(self):
        """Basic SL trigger should work with the two-phase approach."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.001)
        portfolio.update_prices(
            {"A": 100.0},
            bar_index=0,
            ohlc_data={"A": {"open": 100, "high": 100, "low": 100, "close": 100}},
        )
        portfolio.order("A", 10.0)
        portfolio.set_stop_loss("A", stop_price=95.0)

        # Bar where low hits stop
        portfolio.update_prices(
            {"A": 93.0},
            bar_index=1,
            ohlc_data={"A": {"open": 98, "high": 98, "low": 92, "close": 93}},
        )

        assert portfolio.get_position("A") == 0

    def test_take_profit_still_triggers(self):
        """Basic TP trigger should work with the two-phase approach."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.001)
        portfolio.update_prices(
            {"A": 100.0},
            bar_index=0,
            ohlc_data={"A": {"open": 100, "high": 100, "low": 100, "close": 100}},
        )
        portfolio.order("A", 10.0)
        portfolio.set_take_profit("A", target_price=110.0)

        # Bar where high hits TP
        portfolio.update_prices(
            {"A": 108.0},
            bar_index=1,
            ohlc_data={"A": {"open": 105, "high": 112, "low": 105, "close": 108}},
        )

        assert portfolio.get_position("A") == 0

    def test_gap_open_triggers_at_open_price(self):
        """Gap open past stop should fill at open price, not stop price."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.001)
        portfolio.update_prices(
            {"A": 100.0},
            bar_index=0,
            ohlc_data={"A": {"open": 100, "high": 100, "low": 100, "close": 100}},
        )
        portfolio.order("A", 10.0)
        portfolio.set_stop_loss("A", stop_price=95.0)

        # Gap down open below stop
        portfolio.update_prices(
            {"A": 88.0},
            bar_index=1,
            ohlc_data={"A": {"open": 90, "high": 91, "low": 87, "close": 88}},
        )

        assert portfolio.get_position("A") == 0
        # Should fill at open price (90), not stop price (95)
        filled_sells = [o for o in portfolio.orders.values() if o.is_filled() and o.size < 0]
        assert len(filled_sells) == 1
        fill_price = filled_sells[0].filled_price
        assert fill_price is not None
        expected = 90.0 * (1 - 0.001)  # open with sell slippage
        assert abs(fill_price - expected) < 0.01

    def test_trailing_stop_updates_when_no_trigger(self):
        """Trailing stop high-water mark updates when no exit is triggered."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.001)
        portfolio.update_prices(
            {"A": 100.0},
            bar_index=0,
            ohlc_data={"A": {"open": 100, "high": 100, "low": 100, "close": 100}},
        )
        portfolio.order("A", 10.0)
        portfolio.set_trailing_stop("A", trail_pct=0.05)

        initial_stop = portfolio.get_trailing_stop("A")
        assert initial_stop is not None
        assert abs(initial_stop - 95.0) < 0.01

        # Price rises — trailing stop should ratchet up
        portfolio.update_prices(
            {"A": 110.0},
            bar_index=1,
            ohlc_data={"A": {"open": 105, "high": 112, "low": 104, "close": 110}},
        )

        updated_stop = portfolio.get_trailing_stop("A")
        assert updated_stop is not None
        expected_stop = 112.0 * (1 - 0.05)  # 106.4
        assert abs(updated_stop - expected_stop) < 0.01

    def test_multiple_assets_stops_independent(self):
        """Stops on different assets fire independently in the same bar."""
        portfolio = Portfolio(initial_cash=200_000, slippage=0.001)

        # Open positions in two assets
        portfolio.update_prices(
            {"A": 100.0, "B": 200.0},
            bar_index=0,
            ohlc_data={
                "A": {"open": 100, "high": 100, "low": 100, "close": 100},
                "B": {"open": 200, "high": 200, "low": 200, "close": 200},
            },
        )
        portfolio.order("A", 10.0)
        portfolio.order("B", 5.0)
        portfolio.set_stop_loss("A", stop_price=95.0)
        portfolio.set_stop_loss("B", stop_price=190.0)

        # Both stops trigger on the same bar
        portfolio.update_prices(
            {"A": 93.0, "B": 185.0},
            bar_index=1,
            ohlc_data={
                "A": {"open": 97, "high": 97, "low": 92, "close": 93},
                "B": {"open": 195, "high": 195, "low": 183, "close": 185},
            },
        )

        assert portfolio.get_position("A") == 0
        assert portfolio.get_position("B") == 0

    def test_stop_loss_with_many_orders_micro_price(self):
        """Strategy with stops on micro-price data completes without issues."""
        df = _make_micro_price_data(n=3000)

        engine = Engine(
            strategy=WithStops(),
            data=df,
            initial_cash=100_000,
            commission=0.001,
            slippage=0.0005,
            warmup="auto",
        )
        results = engine.run()

        assert results is not None
        assert results.total_return is not None


# ---------------------------------------------------------------------------
# Integration: combined scenario
# ---------------------------------------------------------------------------


class TestCombinedScenario:
    """Integration tests combining all three fixes."""

    def test_sequential_micro_price_backtests(self):
        """Run multiple micro-price backtests sequentially without crashing."""
        for seed in [42, 123, 456]:
            df = _make_micro_price_data(n=2000, seed=seed)
            engine = Engine(
                strategy=SimpleEMA(),
                data=df,
                initial_cash=100_000,
                commission=0.001,
                slippage=0.0005,
                warmup="auto",
            )
            results = engine.run()
            assert results is not None
            engine.cleanup()

        gc.collect()

    def test_sequential_stop_strategy_backtests(self):
        """Run multiple stop-loss strategy backtests sequentially."""
        for seed in [42, 123, 456]:
            df = _make_micro_price_data(n=2000, seed=seed)
            engine = Engine(
                strategy=WithStops(),
                data=df,
                initial_cash=100_000,
                commission=0.001,
                slippage=0.0005,
                warmup="auto",
            )
            results = engine.run()
            assert results is not None
            engine.cleanup()

        gc.collect()


# ---------------------------------------------------------------------------
# Issue 4: Sequential Engine.run() without explicit cleanup
# ---------------------------------------------------------------------------


class TestNoCleanupSequentialBacktests:
    """Verify that sequential backtests work without explicit cleanup().

    Reproduces the pattern used by external evaluation scripts that create a
    new Engine per stock and call engine.run() directly, relying on Python GC
    rather than explicit engine.cleanup().
    """

    def test_sequential_backtests_no_cleanup(self):
        """Multiple backtests with fresh Engine instances complete without cleanup."""
        all_results = []
        for seed in [42, 123, 456, 789, 1011]:
            df = _make_normal_data(n=500, seed=seed)
            strategy = SimpleEMA()
            engine = Engine(
                strategy=strategy,
                data=df,
                initial_cash=100_000,
                commission=0.001,
                slippage=0.0005,
                warmup="auto",
            )
            results = engine.run()
            assert results is not None
            assert results.total_return is not None
            all_results.append(
                {
                    "sharpe": results.sharpe_ratio,
                    "trades": results.trade_stats.total_trades,
                    "_results": results,
                }
            )
            # Deliberately NO cleanup — mirrors evaluate.py pattern

        assert len(all_results) == 5
        # All results should be accessible after the loop
        for r in all_results:
            assert isinstance(r["sharpe"], float)

    def test_sequential_micro_price_no_cleanup(self):
        """Micro-price backtests without cleanup do not crash."""
        all_results = []
        for seed in [42, 123, 456, 789, 1011, 1213]:
            df = _make_micro_price_data(n=3000, seed=seed)
            engine = Engine(
                strategy=SimpleEMA(),
                data=df,
                initial_cash=100_000,
                commission=0.001,
                slippage=0.0005,
                warmup="auto",
                bars_per_day=163,
            )
            results = engine.run()
            assert results is not None
            all_results.append({"_results": results})

        assert len(all_results) == 6

    def test_del_cleans_up_portfolio(self):
        """Engine.__del__ releases portfolio resources."""
        df = _make_normal_data(n=200)
        engine = Engine(strategy=SimpleEMA(), data=df, warmup="auto")
        engine.run()

        assert engine.portfolio is not None
        assert len(engine.portfolio.equity_curve) > 0

        # Trigger __del__ via explicit deletion
        del engine
        gc.collect()  # Ensure finalizer runs

    def test_rerun_clears_previous_portfolio(self):
        """Engine.run() clears the previous portfolio before allocating a new one."""
        df = _make_normal_data(n=200)
        engine = Engine(strategy=SimpleEMA(), data=df, warmup="auto")

        r1 = engine.run()
        assert engine.portfolio is not None
        first_equity_len = len(engine.portfolio.equity_curve)
        assert first_equity_len > 0

        r2 = engine.run()
        assert engine.portfolio is not None
        # The equity curve should be the same length (same data, same warmup)
        assert len(engine.portfolio.equity_curve) == first_equity_len
        assert abs(r1.total_return - r2.total_return) < 1e-10
