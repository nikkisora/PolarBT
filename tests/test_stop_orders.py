"""Tests for STOP and STOP_LIMIT order execution."""

import pytest

from polarbtest.core import Portfolio
from polarbtest.orders import OrderStatus, OrderType


class TestStopOrders:
    """Test STOP order execution."""

    def test_buy_stop_triggers_on_high(self):
        """Buy stop triggers when high >= stop_price (breakout entry)."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0)

        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )

        # Place buy stop at 52000 (breakout entry)
        order_id = portfolio.order("BTC", 0.1, stop_price=52000)
        assert order_id is not None

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.order_type == OrderType.STOP
        assert order.status == OrderStatus.PENDING  # Not triggered yet

        # Bar where high doesn't reach stop price
        portfolio.update_prices(
            {"BTC": 51000}, bar_index=1, ohlc_data={"BTC": {"open": 50500, "high": 51500, "low": 50000, "close": 51000}}
        )
        assert order.status == OrderStatus.PENDING  # Still not triggered

        # Bar where high crosses stop price
        portfolio.update_prices(
            {"BTC": 53000}, bar_index=2, ohlc_data={"BTC": {"open": 51500, "high": 53000, "low": 51000, "close": 52500}}
        )
        assert order.status == OrderStatus.FILLED
        assert order.filled_price == 52000  # Filled at stop price
        assert portfolio.get_position("BTC") == pytest.approx(0.1)

    def test_sell_stop_triggers_on_low(self):
        """Sell stop triggers when low <= stop_price (breakdown entry)."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0)

        # First buy some BTC
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )
        portfolio.order("BTC", 1.0)
        assert portfolio.get_position("BTC") == 1.0

        # Place sell stop at 48000 (breakdown exit)
        order_id = portfolio.order("BTC", -0.5, stop_price=48000)
        assert order_id is not None

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.order_type == OrderType.STOP
        assert order.status == OrderStatus.PENDING

        # Bar where low doesn't reach stop price
        portfolio.update_prices(
            {"BTC": 49000}, bar_index=1, ohlc_data={"BTC": {"open": 50000, "high": 50500, "low": 48500, "close": 49000}}
        )
        assert order.status == OrderStatus.PENDING

        # Bar where low crosses stop price
        portfolio.update_prices(
            {"BTC": 47000}, bar_index=2, ohlc_data={"BTC": {"open": 49000, "high": 49500, "low": 47000, "close": 47500}}
        )
        assert order.status == OrderStatus.FILLED
        assert order.filled_price == 48000
        assert portfolio.get_position("BTC") == pytest.approx(0.5)

    def test_stop_order_with_no_ohlc_uses_close(self):
        """Stop order uses close price when no OHLC data available."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0)

        portfolio.update_prices({"BTC": 50000}, bar_index=0)

        # Place buy stop at 52000
        order_id = portfolio.order("BTC", 0.1, stop_price=52000)
        assert order_id is not None

        order = portfolio.get_order(order_id)
        assert order is not None

        # Update with close price above stop (no OHLC)
        portfolio.update_prices({"BTC": 53000}, bar_index=1)
        assert order.status == OrderStatus.FILLED

    def test_stop_order_rejected_without_stop_price(self):
        """STOP order without stop_price is rejected at creation time."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0)

        portfolio.update_prices({"BTC": 50000}, bar_index=0)

        # Place stop order without stop_price — should be rejected
        order_id = portfolio.order("BTC", 0.1, order_type=OrderType.STOP)
        assert order_id is None

    def test_stop_limit_rejected_without_stop_price(self):
        """STOP_LIMIT order without stop_price is rejected at creation time."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0)

        portfolio.update_prices({"BTC": 50000}, bar_index=0)

        order_id = portfolio.order("BTC", 0.1, order_type=OrderType.STOP_LIMIT, limit_price=52000)
        assert order_id is None

    def test_buy_stop_with_slippage(self):
        """Buy stop applies slippage on top of stop price."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.01)  # 1% slippage

        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )

        order_id = portfolio.order("BTC", 0.1, stop_price=52000)
        assert order_id is not None

        # Trigger the stop
        portfolio.update_prices(
            {"BTC": 53000}, bar_index=1, ohlc_data={"BTC": {"open": 51000, "high": 53000, "low": 51000, "close": 52500}}
        )

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.FILLED
        # Execution price = stop_price * (1 + slippage) = 52000 * 1.01 = 52520
        assert order.filled_price == pytest.approx(52520, abs=1)


class TestStopLimitOrders:
    """Test STOP_LIMIT order execution (two-phase)."""

    def test_buy_stop_limit_two_phase(self):
        """Buy stop-limit: stop triggers, then limit fills."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0)

        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )

        # Buy stop at 52000, limit at 52500
        order_id = portfolio.order("BTC", 0.1, stop_price=52000, limit_price=52500)
        assert order_id is not None

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.order_type == OrderType.STOP_LIMIT
        assert order.status == OrderStatus.PENDING
        assert not order.triggered

        # Bar 1: price doesn't reach stop
        portfolio.update_prices(
            {"BTC": 51000}, bar_index=1, ohlc_data={"BTC": {"open": 50500, "high": 51500, "low": 50000, "close": 51000}}
        )
        assert order.status == OrderStatus.PENDING
        assert not order.triggered

        # Bar 2: stop triggers (high >= 52000) AND limit can fill (low <= 52500)
        portfolio.update_prices(
            {"BTC": 52200}, bar_index=2, ohlc_data={"BTC": {"open": 51500, "high": 53000, "low": 51000, "close": 52200}}
        )
        assert order.triggered
        assert order.status == OrderStatus.FILLED
        assert order.filled_price == 52500  # Filled at limit price

    def test_stop_limit_stop_triggers_but_limit_not_filled(self):
        """Stop triggers but limit price not reached — order stays pending."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0)

        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )

        # Buy stop at 52000, limit at 51500 (limit below stop — gap scenario)
        order_id = portfolio.order("BTC", 0.1, stop_price=52000, limit_price=51500)
        assert order_id is not None

        order = portfolio.get_order(order_id)
        assert order is not None

        # Bar where stop triggers but price gaps above limit
        portfolio.update_prices(
            {"BTC": 53000}, bar_index=1, ohlc_data={"BTC": {"open": 52500, "high": 53500, "low": 52200, "close": 53000}}
        )
        assert order.triggered  # Stop was triggered
        assert order.status == OrderStatus.PENDING  # But limit not filled (low=52200 > 51500)

        # Next bar: price comes back down to fill limit
        portfolio.update_prices(
            {"BTC": 51000}, bar_index=2, ohlc_data={"BTC": {"open": 52000, "high": 52500, "low": 51000, "close": 51200}}
        )
        assert order.status == OrderStatus.FILLED
        assert order.filled_price == 51500

    def test_sell_stop_limit(self):
        """Sell stop-limit order for exiting a position."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0)

        # Buy first
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )
        portfolio.order("BTC", 1.0)

        # Sell stop at 48000, limit at 47500
        order_id = portfolio.order("BTC", -0.5, stop_price=48000, limit_price=47500)
        assert order_id is not None

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.order_type == OrderType.STOP_LIMIT

        # Stop triggers and limit fills on same bar
        portfolio.update_prices(
            {"BTC": 47000}, bar_index=1, ohlc_data={"BTC": {"open": 49000, "high": 49500, "low": 47000, "close": 47200}}
        )
        assert order.status == OrderStatus.FILLED
        assert order.filled_price == 47500
        assert portfolio.get_position("BTC") == pytest.approx(0.5)

    def test_stop_limit_same_bar_trigger_and_fill(self):
        """Stop and limit can both trigger on the same bar."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0)

        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )

        # Buy stop at 52000, limit at 53000 (limit above stop)
        order_id = portfolio.order("BTC", 0.1, stop_price=52000, limit_price=53000)
        assert order_id is not None

        # Wide bar that crosses both stop and limit
        portfolio.update_prices(
            {"BTC": 52500}, bar_index=1, ohlc_data={"BTC": {"open": 51000, "high": 54000, "low": 50500, "close": 52500}}
        )

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.filled_price == 53000  # Filled at limit price
