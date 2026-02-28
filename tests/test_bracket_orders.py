"""Tests for bracket order functionality (OCO orders)."""

from polarbtest.core import Portfolio


class TestBracketOrders:
    """Test bracket order functionality."""

    def test_order_bracket_with_absolute_prices(self):
        """Test bracket order with absolute stop-loss and take-profit prices."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter position
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )

        # Place bracket order
        result = portfolio.order_bracket("BTC", 0.1, stop_loss=48000, take_profit=55000)

        assert result["entry"] is not None
        assert result["stop_loss"] is not None
        assert result["take_profit"] is not None
        assert portfolio.get_position("BTC") == 0.1
        assert portfolio.get_stop_loss("BTC") == 48000
        assert portfolio.get_take_profit("BTC") == 55000

    def test_order_bracket_with_percentages(self):
        """Test bracket order with percentage-based stops."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter position
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )

        # Place bracket order with 5% SL and 10% TP
        result = portfolio.order_bracket("BTC", 0.1, stop_loss_pct=0.05, take_profit_pct=0.10)

        assert result["entry"] is not None
        assert result["stop_loss"] is not None
        assert result["take_profit"] is not None
        assert portfolio.get_position("BTC") == 0.1

        # Check calculated prices
        sl_price = portfolio.get_stop_loss("BTC")
        tp_price = portfolio.get_take_profit("BTC")
        assert sl_price is not None
        assert tp_price is not None
        assert abs(sl_price - 47500) < 1  # 50000 * (1 - 0.05)
        assert abs(tp_price - 55000) < 1  # 50000 * (1 + 0.10)

    def test_bracket_oco_take_profit_cancels_stop_loss(self):
        """Test that when take-profit hits, stop-loss is cancelled (OCO)."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter position at 50000
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )
        portfolio.order_bracket("BTC", 0.1, stop_loss=48000, take_profit=55000)

        assert portfolio.get_stop_loss("BTC") == 48000
        assert portfolio.get_take_profit("BTC") == 55000

        # Price rises and hits take-profit
        portfolio.update_prices(
            {"BTC": 56000}, bar_index=1, ohlc_data={"BTC": {"open": 52000, "high": 56000, "low": 51000, "close": 55500}}
        )

        # Position should be closed
        assert portfolio.get_position("BTC") == 0.0

        # Both SL and TP should be removed (OCO behavior)
        assert portfolio.get_stop_loss("BTC") is None
        assert portfolio.get_take_profit("BTC") is None

    def test_bracket_oco_stop_loss_cancels_take_profit(self):
        """Test that when stop-loss hits, take-profit is cancelled (OCO)."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter position at 50000
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )
        portfolio.order_bracket("BTC", 0.1, stop_loss=48000, take_profit=55000)

        # Price falls and hits stop-loss
        portfolio.update_prices(
            {"BTC": 47000}, bar_index=1, ohlc_data={"BTC": {"open": 49000, "high": 49500, "low": 47000, "close": 47500}}
        )

        # Position should be closed
        assert portfolio.get_position("BTC") == 0.0

        # Both SL and TP should be removed (OCO behavior)
        assert portfolio.get_stop_loss("BTC") is None
        assert portfolio.get_take_profit("BTC") is None

    def test_bracket_short_position(self):
        """Test bracket order for short positions."""
        portfolio = Portfolio(initial_cash=100000)

        # Enter short position at 50000
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )

        # Short 0.1 BTC with 5% SL and 10% TP
        # For shorts: SL is above entry, TP is below entry
        portfolio.order_bracket("BTC", -0.1, stop_loss_pct=0.05, take_profit_pct=0.10)

        assert portfolio.get_position("BTC") == -0.1

        # Check prices are inverted for short
        sl_price = portfolio.get_stop_loss("BTC")
        tp_price = portfolio.get_take_profit("BTC")
        assert sl_price is not None
        assert tp_price is not None
        assert sl_price > 50000  # Stop-loss above for short (52500)
        assert tp_price < 50000  # Take-profit below for short (45000)
        assert abs(sl_price - 52500) < 1  # 50000 * (1 + 0.05)
        assert abs(tp_price - 45000) < 1  # 50000 * (1 - 0.10)

    def test_bracket_with_limit_order(self):
        """Test bracket order with limit entry."""
        portfolio = Portfolio(initial_cash=10000)

        # Set price at 50000
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )

        # Place limit buy at 49000 (below current price)
        result = portfolio.order_bracket("BTC", 0.1, stop_loss=47000, take_profit=52000, limit_price=49000)

        assert result["entry"] is not None

        # Position not filled yet (price hasn't reached limit)
        assert portfolio.get_position("BTC") == 0.0

        # Stops not set yet (entry not filled)
        assert portfolio.get_stop_loss("BTC") is None
        assert portfolio.get_take_profit("BTC") is None

    def test_bracket_no_position(self):
        """Test that bracket order fails if position can't be opened."""
        portfolio = Portfolio(initial_cash=100)  # Not enough cash

        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )

        # Try to buy 0.1 BTC (costs 5000) with only 100 cash
        portfolio.order_bracket("BTC", 0.1, stop_loss=48000, take_profit=55000)

        # Entry order should be rejected
        assert portfolio.get_position("BTC") == 0.0
