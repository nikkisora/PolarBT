"""Tests for percentage-based orders, fractional shares, and fee-adjusted sizing."""

import polars as pl

from polarbt.core import Engine, Portfolio, Strategy


def make_portfolio(**kwargs) -> Portfolio:
    """Create a portfolio with prices set."""
    defaults = {"initial_cash": 100_000.0}
    defaults.update(kwargs)
    p = Portfolio(**defaults)
    p._current_prices = {"BTC": 50_000.0, "ETH": 2_000.0}
    return p


# --- Fractional shares ---


class TestFractionalShares:
    def test_default_allows_fractional(self):
        p = make_portfolio()
        p.order("BTC", 0.123)
        assert p.get_position("BTC") == 0.123

    def test_whole_shares_truncates_buy(self):
        p = make_portfolio(fractional_shares=False)
        p.order("ETH", 3.7)
        assert p.get_position("ETH") == 3.0

    def test_whole_shares_truncates_sell(self):
        p = make_portfolio(fractional_shares=False)
        p.order("ETH", 10.0)
        p.order("ETH", -3.7)
        assert p.get_position("ETH") == 7.0  # Sold 3, not 3.7

    def test_whole_shares_rejects_sub_one(self):
        p = make_portfolio(fractional_shares=False)
        result = p.order("BTC", 0.5)
        assert result is None  # Truncated to 0, rejected

    def test_whole_shares_order_target_percent(self):
        p = make_portfolio(fractional_shares=False)
        # 50% of 100k / 50k = 1 BTC exactly
        p.order_target_percent("BTC", 0.5)
        assert p.get_position("BTC") == 1.0

    def test_whole_shares_order_target_percent_rounds_down(self):
        p = make_portfolio(fractional_shares=False)
        # 60% of 100k = 60k / 50k = 1.2 -> truncated to 1
        p.order_target_percent("BTC", 0.6)
        assert p.get_position("BTC") == 1.0

    def test_whole_shares_order_target(self):
        p = make_portfolio(fractional_shares=False)
        p.order_target("ETH", 7.9)
        assert p.get_position("ETH") == 7.0

    def test_whole_shares_order_target_value(self):
        p = make_portfolio(fractional_shares=False)
        # 5000 / 2000 = 2.5 -> truncated to 2
        p.order_target_value("ETH", 5_000.0)
        assert p.get_position("ETH") == 2.0

    def test_fractional_shares_with_engine(self):
        """Test fractional_shares=False flows through Engine."""

        class BuyStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def next(self, ctx):
                if ctx.bar_index == 0:
                    ctx.portfolio.order("asset", 2.7)

        df = pl.DataFrame({"close": [100.0, 101.0, 102.0]})
        engine = Engine(
            strategy=BuyStrategy(),
            data=df,
            initial_cash=10_000.0,
            warmup=0,
            fractional_shares=False,
        )
        engine.run()
        assert engine.portfolio is not None
        assert engine.portfolio.get_position("asset") == 2.0


# --- Percentage orders with fees ---


class TestPercentOrdersWithFees:
    def test_100_percent_order_succeeds_with_commission(self):
        """Ordering 100% of portfolio should not be rejected due to fees."""
        p = make_portfolio(commission=0.001)  # 0.1% commission
        p._current_prices = {"ETH": 100.0}
        order_id = p.order_target_percent("ETH", 1.0)
        assert order_id is not None
        # Position should be filled (slightly less than 1000 shares due to fee adjustment)
        position = p.get_position("ETH")
        assert position > 0
        # Portfolio value should be close to initial
        assert abs(p.get_value() - 100_000) < 200  # within $200 of initial (commission)

    def test_100_percent_order_succeeds_with_slippage(self):
        """Ordering 100% should work with slippage."""
        p = make_portfolio(slippage=0.001)
        p._current_prices = {"ETH": 100.0}
        order_id = p.order_target_percent("ETH", 1.0)
        assert order_id is not None
        assert p.get_position("ETH") > 0

    def test_100_percent_order_succeeds_with_both(self):
        """Ordering 100% should work with both commission and slippage."""
        p = make_portfolio(commission=0.001, slippage=0.001)
        p._current_prices = {"ETH": 100.0}
        order_id = p.order_target_percent("ETH", 1.0)
        assert order_id is not None
        assert p.get_position("ETH") > 0
        # Cash should be near zero (small remainder from fee adjustment)
        assert p.cash >= -0.01  # Should not go negative without leverage

    def test_50_percent_still_accurate(self):
        """50% order should still allocate roughly 50%."""
        p = make_portfolio(commission=0.001)
        p._current_prices = {"ETH": 100.0}
        p.order_target_percent("ETH", 0.5)
        position_value = p.get_position("ETH") * 100.0
        ratio = position_value / p.get_value()
        assert 0.48 < ratio < 0.52

    def test_sell_orders_not_over_adjusted(self):
        """Selling should not apply buy-side fee adjustment."""
        p = make_portfolio(commission=0.001)
        p._current_prices = {"ETH": 100.0}
        p.order("ETH", 500.0)  # Buy 500 shares
        # Now reduce to 25% — this is a sell, fees shouldn't reduce the sell
        p.order_target_percent("ETH", 0.25)
        position = p.get_position("ETH")
        # Should be close to 25% of portfolio
        ratio = (position * 100.0) / p.get_value()
        assert 0.23 < ratio < 0.27

    def test_zero_fees_no_adjustment(self):
        """Without fees, ordering 100% should use all cash."""
        p = make_portfolio(commission=0.0, slippage=0.0)
        p._current_prices = {"ETH": 100.0}
        p.order_target_percent("ETH", 1.0)
        assert p.get_position("ETH") == 1000.0  # Exactly 100k / 100
        assert abs(p.cash) < 0.01

    def test_fixed_plus_percent_commission_100_percent(self):
        """100% order with fixed + percent commission model."""
        p = make_portfolio(commission=(5.0, 0.001))
        p._current_prices = {"ETH": 100.0}
        order_id = p.order_target_percent("ETH", 1.0)
        assert order_id is not None
        assert p.get_position("ETH") > 0
        assert p.cash >= -0.01


# --- Percentage orders with leverage ---


class TestPercentOrdersWithLeverage:
    def test_100_percent_with_leverage(self):
        """100% means 100% of equity, not leveraged buying power."""
        p = make_portfolio(leverage=2.0)
        p._current_prices = {"ETH": 100.0}
        p.order_target_percent("ETH", 1.0)
        position_value = p.get_position("ETH") * 100.0
        # Should be close to 100k (equity), not 200k
        assert 99_000 < position_value < 101_000

    def test_200_percent_with_2x_leverage(self):
        """With 2x leverage, user can order 200% of equity."""
        p = make_portfolio(leverage=2.0)
        p._current_prices = {"ETH": 100.0}
        p.order_target_percent("ETH", 2.0)
        position = p.get_position("ETH")
        assert position > 0
        # Should use most of buying power (2x equity = 200k)
        position_value = position * 100.0
        assert position_value > 150_000  # At least 150k allocated

    def test_percent_order_with_leverage_and_fees(self):
        """100% + leverage + fees should all work together."""
        p = make_portfolio(leverage=2.0, commission=0.001, slippage=0.001)
        p._current_prices = {"ETH": 100.0}
        order_id = p.order_target_percent("ETH", 1.0)
        assert order_id is not None
        assert p.get_position("ETH") > 0


# --- Engine integration ---


class TestPercentOrdersEngine:
    def test_100_percent_in_backtest(self):
        """Full backtest with 100% allocation should not reject the order."""

        class AllInStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def next(self, ctx):
                if ctx.bar_index == 0 and ctx.portfolio.get_position("asset") == 0:
                    ctx.portfolio.order_target_percent("asset", 1.0)

        df = pl.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})
        engine = Engine(
            strategy=AllInStrategy(),
            data=df,
            initial_cash=10_000.0,
            commission=0.001,
            slippage=0.0005,
            warmup=0,
        )
        results = engine.run()
        assert engine.portfolio is not None
        assert engine.portfolio.get_position("asset") > 0
        assert results.final_equity > 0

    def test_whole_shares_100_percent_in_backtest(self):
        """Full backtest with 100% and whole shares."""

        class AllInStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def next(self, ctx):
                if ctx.bar_index == 0 and ctx.portfolio.get_position("asset") == 0:
                    ctx.portfolio.order_target_percent("asset", 1.0)

        df = pl.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})
        engine = Engine(
            strategy=AllInStrategy(),
            data=df,
            initial_cash=10_000.0,
            commission=0.001,
            warmup=0,
            fractional_shares=False,
        )
        results = engine.run()
        assert engine.portfolio is not None
        position = engine.portfolio.get_position("asset")
        assert position == int(position)  # Whole number
        assert position > 0
        assert results.final_equity > 0
