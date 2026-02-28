"""Tests for take-profit order functionality."""

from polarbtest.core import Portfolio


class TestTakeProfit:
    """Test take-profit order functionality."""

    def test_set_take_profit_with_price(self):
        """Test setting take-profit with absolute price."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter position
        portfolio.update_prices({"BTC": 50000}, bar_index=0)
        portfolio.order("BTC", 0.1)

        # Set take-profit
        order_id = portfolio.set_take_profit("BTC", target_price=55000)

        assert order_id is not None
        assert portfolio.get_take_profit("BTC") == 55000

    def test_set_take_profit_with_percentage(self):
        """Test setting take-profit with percentage."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter position
        portfolio.update_prices({"BTC": 50000}, bar_index=0)
        portfolio.order("BTC", 0.1)

        # Set take-profit at 10% above current price
        order_id = portfolio.set_take_profit("BTC", target_pct=0.10)

        assert order_id is not None
        tp_price = portfolio.get_take_profit("BTC")
        assert tp_price is not None
        assert abs(tp_price - 55000) < 0.01  # 50000 * 1.10 (with float tolerance)

    def test_take_profit_triggers_long_position(self):
        """Test that take-profit triggers when price rises (long position)."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter long position at 50000
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )
        portfolio.order("BTC", 0.1)

        assert portfolio.get_position("BTC") == 0.1

        # Set take-profit at 55000
        portfolio.set_take_profit("BTC", target_price=55000)

        # Price rises to 56000 (high crosses take-profit)
        portfolio.update_prices(
            {"BTC": 56000}, bar_index=1, ohlc_data={"BTC": {"open": 52000, "high": 56000, "low": 51000, "close": 55500}}
        )

        # Position should be closed
        assert portfolio.get_position("BTC") == 0.0
        assert portfolio.get_take_profit("BTC") is None  # Take-profit removed

    def test_take_profit_not_triggered_when_not_hit(self):
        """Test that take-profit doesn't trigger when price doesn't reach target."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter long position
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )
        portfolio.order("BTC", 0.1)

        # Set take-profit at 55000
        portfolio.set_take_profit("BTC", target_price=55000)

        # Price rises but doesn't hit take-profit
        portfolio.update_prices(
            {"BTC": 54000}, bar_index=1, ohlc_data={"BTC": {"open": 52000, "high": 54500, "low": 51000, "close": 54000}}
        )

        # Position should still be open
        assert portfolio.get_position("BTC") == 0.1
        assert portfolio.get_take_profit("BTC") == 55000

    def test_take_profit_triggers_short_position(self):
        """Test that take-profit triggers when price falls (short position)."""
        portfolio = Portfolio(initial_cash=100000)

        # Enter short position at 50000
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )
        portfolio.order("BTC", -0.1)
        assert portfolio.get_position("BTC") == -0.1

        # Set take-profit at 45000 (below entry for short)
        portfolio.set_take_profit("BTC", target_price=45000)
        assert portfolio.get_take_profit("BTC") == 45000

        # Price falls to 44000 (low crosses take-profit)
        portfolio.update_prices(
            {"BTC": 44000}, bar_index=1, ohlc_data={"BTC": {"open": 48000, "high": 49000, "low": 44000, "close": 44500}}
        )

        # Position should be closed
        assert portfolio.get_position("BTC") == 0.0
        assert portfolio.get_take_profit("BTC") is None

    def test_remove_take_profit(self):
        """Test removing take-profit."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter position and set take-profit
        portfolio.update_prices({"BTC": 50000}, bar_index=0)
        portfolio.order("BTC", 0.1)
        portfolio.set_take_profit("BTC", target_price=55000)

        assert portfolio.get_take_profit("BTC") == 55000

        # Remove take-profit
        result = portfolio.remove_take_profit("BTC")

        assert result is True
        assert portfolio.get_take_profit("BTC") is None

    def test_remove_nonexistent_take_profit(self):
        """Test removing take-profit that doesn't exist."""
        portfolio = Portfolio(initial_cash=10000)

        result = portfolio.remove_take_profit("BTC")

        assert result is False

    def test_take_profit_no_position(self):
        """Test that take-profit can't be set without a position."""
        portfolio = Portfolio(initial_cash=10000)
        portfolio.update_prices({"BTC": 50000}, bar_index=0)

        # Try to set take-profit without position
        order_id = portfolio.set_take_profit("BTC", target_price=55000)

        assert order_id is None

    def test_take_profit_and_stop_loss_together(self):
        """Test using both take-profit and stop-loss on same position."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter position
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )
        portfolio.order("BTC", 0.1)

        # Set both stop-loss and take-profit
        sl_id = portfolio.set_stop_loss("BTC", stop_price=48000)
        tp_id = portfolio.set_take_profit("BTC", target_price=55000)

        assert sl_id is not None
        assert tp_id is not None
        assert portfolio.get_stop_loss("BTC") == 48000
        assert portfolio.get_take_profit("BTC") == 55000

        # Price rises to hit take-profit
        portfolio.update_prices(
            {"BTC": 56000}, bar_index=1, ohlc_data={"BTC": {"open": 52000, "high": 56000, "low": 51000, "close": 55500}}
        )

        # Position closed, both should be removed
        assert portfolio.get_position("BTC") == 0.0
        assert portfolio.get_take_profit("BTC") is None
        # Note: stop-loss will still exist unless we add cleanup logic
