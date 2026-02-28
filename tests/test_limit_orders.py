"""Tests for limit order execution."""

from polarbtest.core import Portfolio
from polarbtest.orders import OrderStatus, OrderType


class TestLimitOrders:
    """Test limit order functionality."""

    def test_buy_limit_fills_when_low_touches_limit(self):
        """Test buy limit order fills when low reaches limit price."""
        portfolio = Portfolio(initial_cash=100_000)

        # Place buy limit order at 49500
        order_id = portfolio.order(asset="BTC", quantity=1.0, limit_price=49500.0, order_type=OrderType.LIMIT)

        assert order_id is not None
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.PENDING

        # Update with OHLC where low touches limit
        portfolio.update_prices(
            prices={"BTC": 50000.0},
            bar_index=1,
            ohlc_data={"BTC": {"open": 50100.0, "high": 50200.0, "low": 49400.0, "close": 50000.0}},
            timestamp=1000,
        )

        # Order should be filled
        assert order.status == OrderStatus.FILLED  # type: ignore[comparison-overlap]
        assert portfolio.get_position("BTC") == 1.0

    def test_buy_limit_does_not_fill_when_low_above_limit(self):
        """Test buy limit order doesn't fill when low is above limit."""
        portfolio = Portfolio(initial_cash=100_000)

        # Place buy limit order at 49500
        order_id = portfolio.order(asset="BTC", quantity=1.0, limit_price=49500.0, order_type=OrderType.LIMIT)
        assert order_id is not None

        # Update with OHLC where low is above limit
        portfolio.update_prices(
            prices={"BTC": 50000.0},
            bar_index=1,
            ohlc_data={"BTC": {"open": 50100.0, "high": 50200.0, "low": 49600.0, "close": 50000.0}},
            timestamp=1000,
        )

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.PENDING
        assert portfolio.get_position("BTC") == 0.0

    def test_sell_limit_fills_when_high_touches_limit(self):
        """Test sell limit order fills when high reaches limit price."""
        portfolio = Portfolio(initial_cash=100_000)

        # First buy at market
        portfolio.update_prices(
            prices={"BTC": 50000.0},
            bar_index=0,
            ohlc_data={"BTC": {"open": 50000.0, "high": 50000.0, "low": 50000.0, "close": 50000.0}},
            timestamp=900,
        )
        portfolio.order(asset="BTC", quantity=1.0, order_type=OrderType.MARKET)

        # Place sell limit order at 51000
        order_id = portfolio.order(asset="BTC", quantity=-1.0, limit_price=51000.0, order_type=OrderType.LIMIT)

        assert order_id is not None

        # Update with OHLC where high touches limit
        portfolio.update_prices(
            prices={"BTC": 50500.0},
            bar_index=1,
            ohlc_data={"BTC": {"open": 50400.0, "high": 51100.0, "low": 50300.0, "close": 50500.0}},
            timestamp=1000,
        )

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert portfolio.get_position("BTC") == 0.0

    def test_sell_limit_does_not_fill_when_high_below_limit(self):
        """Test sell limit order doesn't fill when high is below limit."""
        portfolio = Portfolio(initial_cash=100_000)

        # First buy at market
        portfolio.update_prices(
            prices={"BTC": 50000.0},
            bar_index=0,
            ohlc_data={"BTC": {"open": 50000.0, "high": 50000.0, "low": 50000.0, "close": 50000.0}},
            timestamp=900,
        )
        portfolio.order(asset="BTC", quantity=1.0, order_type=OrderType.MARKET)

        # Place sell limit order at 51000
        order_id = portfolio.order(asset="BTC", quantity=-1.0, limit_price=51000.0, order_type=OrderType.LIMIT)
        assert order_id is not None

        # Update with OHLC where high is below limit
        portfolio.update_prices(
            prices={"BTC": 50500.0},
            bar_index=1,
            ohlc_data={"BTC": {"open": 50400.0, "high": 50900.0, "low": 50300.0, "close": 50500.0}},
            timestamp=1000,
        )

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.PENDING
        assert portfolio.get_position("BTC") == 1.0

    def test_limit_order_without_ohlc_rejected(self):
        """Test limit order is rejected if no OHLC data provided."""
        portfolio = Portfolio(initial_cash=100_000)

        # Place buy limit order
        order_id = portfolio.order(asset="BTC", quantity=1.0, limit_price=49500.0, order_type=OrderType.LIMIT)
        assert order_id is not None

        # Update without OHLC data
        portfolio.update_prices(prices={"BTC": 50000.0}, bar_index=1, timestamp=1000)

        order = portfolio.get_order(order_id)
        assert order is not None
        # Should remain pending or be rejected (depending on implementation)
        assert order.status in (OrderStatus.PENDING, OrderStatus.REJECTED)

    def test_market_order_works_without_ohlc(self):
        """Test market orders still work without OHLC data."""
        portfolio = Portfolio(initial_cash=100_000)

        # Update prices without OHLC
        portfolio.update_prices(prices={"BTC": 50000.0}, bar_index=0, timestamp=1000)

        # Place market order
        order_id = portfolio.order(asset="BTC", quantity=1.0, order_type=OrderType.MARKET)

        assert order_id is not None
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert portfolio.get_position("BTC") == 1.0

    def test_limit_order_execution_price(self):
        """Test limit order executes at limit price."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0)

        # Place buy limit order at 49500
        order_id = portfolio.order(asset="BTC", quantity=1.0, limit_price=49500.0, order_type=OrderType.LIMIT)
        assert order_id is not None

        # Update with OHLC where low touches limit
        portfolio.update_prices(
            prices={"BTC": 50000.0},
            bar_index=1,
            ohlc_data={"BTC": {"open": 50100.0, "high": 50200.0, "low": 49400.0, "close": 50000.0}},
            timestamp=1000,
        )

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.FILLED
        # Should execute at limit price (with slippage if configured)
        assert order.filled_price == 49500.0

    def test_multiple_limit_orders(self):
        """Test multiple limit orders at different prices."""
        portfolio = Portfolio(initial_cash=100_000)

        # Place multiple buy limit orders
        order1_id = portfolio.order("BTC", 0.5, limit_price=49000.0, order_type=OrderType.LIMIT)
        order2_id = portfolio.order("BTC", 0.5, limit_price=48000.0, order_type=OrderType.LIMIT)

        # Update with OHLC where low touches first order
        portfolio.update_prices(
            prices={"BTC": 50000.0},
            bar_index=1,
            ohlc_data={"BTC": {"open": 50100.0, "high": 50200.0, "low": 48900.0, "close": 50000.0}},
            timestamp=1000,
        )

        # First order should fill, second should not
        assert order1_id is not None
        assert order2_id is not None
        order1 = portfolio.get_order(order1_id)
        order2 = portfolio.get_order(order2_id)
        assert order1 is not None
        assert order2 is not None
        assert order1.status == OrderStatus.FILLED
        assert order2.status == OrderStatus.PENDING
        assert portfolio.get_position("BTC") == 0.5

    def test_limit_order_with_delay(self):
        """Test limit order with order delay."""
        portfolio = Portfolio(initial_cash=100_000, order_delay=1)

        # Place limit order
        order_id = portfolio.order("BTC", 1.0, limit_price=49500.0, order_type=OrderType.LIMIT)
        assert order_id is not None

        # First bar - order should not execute (delay=1)
        portfolio.update_prices(
            prices={"BTC": 50000.0},
            bar_index=0,
            ohlc_data={"BTC": {"open": 50100.0, "high": 50200.0, "low": 49400.0, "close": 50000.0}},
            timestamp=1000,
        )

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.PENDING

        # Second bar - order should execute
        portfolio.update_prices(
            prices={"BTC": 50000.0},
            bar_index=1,
            ohlc_data={"BTC": {"open": 50100.0, "high": 50200.0, "low": 49400.0, "close": 50000.0}},
            timestamp=2000,
        )

        assert order.status == OrderStatus.FILLED  # type: ignore[comparison-overlap]
