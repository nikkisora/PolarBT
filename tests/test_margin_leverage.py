"""Tests for margin and leverage functionality."""

import polars as pl

from polarbtest.core import Engine, Portfolio, Strategy

# --- Helper Strategy ---


class BuyAndHoldStrategy(Strategy):
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def next(self, ctx):
        if ctx.bar_index == 0 and ctx.portfolio.get_position("asset") == 0:
            pct = self.params.get("pct", 1.0)
            ctx.portfolio.order_target_percent("asset", pct)


class BuyOnBarStrategy(Strategy):
    """Buy a specific quantity on a specific bar."""

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def next(self, ctx):
        buy_bar = self.params.get("buy_bar", 0)
        qty = self.params.get("qty", 1.0)
        if ctx.bar_index == buy_bar and ctx.portfolio.get_position("asset") == 0:
            ctx.portfolio.order("asset", qty)


# --- Portfolio unit tests ---


class TestGetBuyingPower:
    def test_no_leverage_buying_power_equals_cash(self):
        p = Portfolio(initial_cash=100_000, leverage=1.0)
        assert p.get_buying_power() == 100_000

    def test_leverage_2x_doubles_buying_power(self):
        p = Portfolio(initial_cash=100_000, leverage=2.0)
        assert p.get_buying_power() == 200_000

    def test_buying_power_decreases_with_positions(self):
        p = Portfolio(initial_cash=100_000, leverage=2.0)
        p._current_prices = {"BTC": 50_000}
        p.positions["BTC"] = 1.0  # Position worth 50k
        p.cash = 50_000  # Spent 50k buying
        # equity = 50k cash + 50k position = 100k, buying_power = 100k*2 - 50k = 150k
        assert p.get_buying_power() == 150_000

    def test_buying_power_never_negative(self):
        p = Portfolio(initial_cash=10_000, leverage=1.0)
        p._current_prices = {"BTC": 50_000}
        p.positions["BTC"] = 1.0
        # equity = 10k + 50k = 60k, buying_power = 60k - 50k = 10k
        assert p.get_buying_power() == 10_000


class TestMarginMethods:
    def test_margin_used(self):
        p = Portfolio(initial_cash=100_000, leverage=2.0)
        p._current_prices = {"BTC": 50_000}
        p.positions["BTC"] = 1.0
        # margin_used = 50k / 2 = 25k
        assert p.get_margin_used() == 25_000

    def test_margin_available(self):
        p = Portfolio(initial_cash=100_000, leverage=2.0)
        p._current_prices = {"BTC": 50_000}
        p.positions["BTC"] = 1.0
        p.cash = 50_000  # Spent 50k
        # equity = 100k, margin_used = 50k/2 = 25k, margin_available = 75k
        assert p.get_margin_available() == 75_000

    def test_margin_ratio_no_positions(self):
        p = Portfolio(initial_cash=100_000, leverage=2.0)
        assert p.get_margin_ratio() is None

    def test_margin_ratio_with_positions(self):
        p = Portfolio(initial_cash=100_000, leverage=2.0)
        p._current_prices = {"BTC": 50_000}
        p.positions["BTC"] = 1.0
        p.cash = 50_000  # Spent 50k
        # equity = 100k, position_value = 50k, ratio = 2.0
        assert p.get_margin_ratio() == 2.0

    def test_margin_ratio_leveraged_position(self):
        p = Portfolio(initial_cash=50_000, leverage=2.0)
        p._current_prices = {"BTC": 50_000}
        p.positions["BTC"] = 2.0  # 100k position on 50k equity
        p.cash = -50_000  # Borrowed 50k
        # equity = -50k + 100k = 50k, position_value = 100k, ratio = 0.5
        assert p.get_margin_ratio() == 0.5


class TestLeveragedOrders:
    def test_leverage_allows_larger_position(self):
        """With 2x leverage, can buy 2x the position."""
        p = Portfolio(initial_cash=100_000, leverage=2.0)
        p._current_prices = {"BTC": 50_000}
        p.update_prices({"BTC": 50_000}, bar_index=0)

        # Without leverage, can only buy 2 BTC (100k / 50k)
        # With 2x leverage, should be able to buy up to ~4 BTC
        order_id = p.order("BTC", 3.0)
        assert order_id is not None
        order = p.orders[order_id]
        assert order.is_filled()
        assert p.get_position("BTC") == 3.0
        # Cash goes negative (borrowed funds)
        assert p.cash < 0

    def test_no_leverage_rejects_insufficient_cash(self):
        """Without leverage, can't buy more than cash allows."""
        p = Portfolio(initial_cash=100_000, leverage=1.0)
        p._current_prices = {"BTC": 50_000}
        p.update_prices({"BTC": 50_000}, bar_index=0)

        order_id = p.order("BTC", 3.0)  # Would cost 150k
        assert order_id is not None
        order = p.orders[order_id]
        assert order.status.value == "rejected"

    def test_leverage_respects_margin_limit(self):
        """Can't exceed max leverage."""
        p = Portfolio(initial_cash=100_000, leverage=2.0)
        p._current_prices = {"BTC": 50_000}
        p.update_prices({"BTC": 50_000}, bar_index=0)

        # 5 BTC = 250k, needs 125k margin but equity is 100k
        order_id = p.order("BTC", 5.0)
        assert order_id is not None
        order = p.orders[order_id]
        assert order.status.value == "rejected"

    def test_default_leverage_backward_compatible(self):
        """Default leverage=1.0 should not change existing behavior."""
        p = Portfolio(initial_cash=100_000)
        assert p.leverage == 1.0
        p._current_prices = {"BTC": 1000}
        p.update_prices({"BTC": 1000}, bar_index=0)

        order_id = p.order("BTC", 100.0)  # Exactly 100k
        assert order_id is not None
        assert p.orders[order_id].is_filled()


class TestMarginCall:
    def test_margin_call_closes_positions(self):
        """When margin ratio drops below maintenance, positions are closed."""
        p = Portfolio(initial_cash=50_000, leverage=2.0, maintenance_margin=0.3)
        p._current_prices = {"BTC": 50_000}
        p.update_prices({"BTC": 50_000}, bar_index=0)

        # Buy 2 BTC at 50k each = 100k position, using 50k cash + 50k borrowed
        p.order("BTC", 2.0)
        assert p.get_position("BTC") == 2.0
        assert p.cash == -50_000  # Borrowed 50k

        # Price drops: equity = -50k + 2*30k = 10k, ratio = 10k/60k = 0.167 < 0.3
        p.update_prices({"BTC": 30_000}, bar_index=1)

        # Margin call should have closed the position
        assert p.get_position("BTC") == 0.0
        assert p._margin_called is True

    def test_no_margin_call_when_ratio_ok(self):
        """No margin call when ratio is above maintenance."""
        p = Portfolio(initial_cash=50_000, leverage=2.0, maintenance_margin=0.25)
        p._current_prices = {"BTC": 50_000}
        p.update_prices({"BTC": 50_000}, bar_index=0)

        p.order("BTC", 1.5)
        assert p.get_position("BTC") == 1.5

        # Price drops slightly: equity = (50k - 75k) + 1.5*45k = -25k + 67.5k = 42.5k
        # ratio = 42.5k / 67.5k = 0.63 > 0.25
        p.update_prices({"BTC": 45_000}, bar_index=1)
        assert p.get_position("BTC") == 1.5  # Position kept
        assert p._margin_called is False

    def test_no_margin_call_without_maintenance_margin(self):
        """No margin call when maintenance_margin is None."""
        p = Portfolio(initial_cash=50_000, leverage=2.0)
        p._current_prices = {"BTC": 50_000}
        p.update_prices({"BTC": 50_000}, bar_index=0)

        p.order("BTC", 2.0)
        # Huge price drop
        p.update_prices({"BTC": 10_000}, bar_index=1)
        # Position should still be open
        assert p.get_position("BTC") == 2.0


class TestEngineIntegration:
    def _make_df(self, prices: list[float]) -> pl.DataFrame:
        return pl.DataFrame({"close": prices})

    def test_engine_with_leverage(self):
        """Engine passes leverage to Portfolio."""
        df = self._make_df([100.0, 101.0, 102.0, 103.0, 104.0])
        engine = Engine(
            strategy=BuyAndHoldStrategy(),
            data=df,
            initial_cash=10_000,
            leverage=2.0,
            commission=0.0,
            slippage=0.0,
        )
        engine.run()
        assert engine.portfolio is not None
        assert engine.portfolio.leverage == 2.0

    def test_engine_with_margin_call(self):
        """Engine with maintenance margin triggers margin call on price drop."""
        # Start at 100, drop to 40 — with 2x leverage on large position, margin call triggers
        prices = [100.0, 100.0, 40.0, 41.0, 42.0]
        df = self._make_df(prices)

        engine = Engine(
            strategy=BuyOnBarStrategy(buy_bar=1, qty=150.0),
            data=df,
            initial_cash=10_000,
            leverage=2.0,
            maintenance_margin=0.3,
            commission=0.0,
            slippage=0.0,
        )
        engine.run()
        assert engine.portfolio is not None
        # After price drop, margin call should have closed positions
        assert engine.portfolio.get_position("asset") == 0.0
        assert engine.portfolio._margin_called is True

    def test_engine_default_no_leverage(self):
        """Default engine has leverage=1.0."""
        df = self._make_df([100.0, 101.0])
        engine = Engine(
            strategy=BuyAndHoldStrategy(),
            data=df,
        )
        engine.run()
        assert engine.portfolio is not None
        assert engine.portfolio.leverage == 1.0

    def test_leverage_with_commission(self):
        """Leverage works correctly with commissions."""
        df = self._make_df([100.0, 100.0, 100.0])
        engine = Engine(
            strategy=BuyOnBarStrategy(buy_bar=0, qty=150.0),
            data=df,
            initial_cash=10_000,
            leverage=2.0,
            commission=0.001,
            slippage=0.0,
        )
        engine.run()
        assert engine.portfolio is not None
        assert engine.portfolio.get_position("asset") == 150.0


class TestRunnerIntegration:
    def test_backtest_with_leverage(self):
        """backtest() passes leverage parameter."""
        from polarbtest.runner import backtest

        df = pl.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})
        results = backtest(
            BuyAndHoldStrategy,
            df,
            initial_cash=10_000,
            leverage=2.0,
            commission=0.0,
            slippage=0.0,
        )
        assert results["success"] is True
