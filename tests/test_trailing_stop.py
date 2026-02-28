"""Tests for trailing stop-loss functionality."""

from polarbtest.core import Portfolio


class TestTrailingStop:
    """Test trailing stop-loss functionality."""

    def test_set_trailing_stop_with_percentage(self):
        """Test setting trailing stop with percentage."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter position at 50000
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )
        portfolio.order("BTC", 0.1)

        # Set 5% trailing stop
        order_id = portfolio.set_trailing_stop("BTC", trail_pct=0.05)

        assert order_id is not None
        # Initial stop should be 5% below entry
        stop_price = portfolio.get_trailing_stop("BTC")
        assert stop_price is not None
        assert abs(stop_price - 47500) < 1  # 50000 * 0.95

    def test_set_trailing_stop_with_amount(self):
        """Test setting trailing stop with absolute amount."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter position
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )
        portfolio.order("BTC", 0.1)

        # Set $2000 trailing stop
        order_id = portfolio.set_trailing_stop("BTC", trail_amount=2000)

        assert order_id is not None
        assert portfolio.get_trailing_stop("BTC") == 48000  # 50000 - 2000

    def test_trailing_stop_moves_up_with_price(self):
        """Test that trailing stop moves up when price rises."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter position at 50000
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )
        portfolio.order("BTC", 0.1)

        # Set 5% trailing stop
        portfolio.set_trailing_stop("BTC", trail_pct=0.05)
        initial_stop = portfolio.get_trailing_stop("BTC")
        assert initial_stop is not None
        assert abs(initial_stop - 47500) < 1

        # Price rises to 55000 (low stays above stop)
        portfolio.update_prices(
            {"BTC": 55000}, bar_index=1, ohlc_data={"BTC": {"open": 52000, "high": 55000, "low": 51000, "close": 55000}}
        )
        new_stop = portfolio.get_trailing_stop("BTC")

        # Stop should have moved up to 55000 * 0.95 = 52250
        assert new_stop is not None
        assert abs(new_stop - 52250) < 1
        assert new_stop > initial_stop

    def test_trailing_stop_doesnt_move_down(self):
        """Test that trailing stop doesn't move down when price falls."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter position at 50000
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )
        portfolio.order("BTC", 0.1)

        # Set 5% trailing stop
        portfolio.set_trailing_stop("BTC", trail_pct=0.05)

        # Price rises to 55000 (low stays above previous stop at 47500)
        portfolio.update_prices(
            {"BTC": 55000}, bar_index=1, ohlc_data={"BTC": {"open": 52000, "high": 55000, "low": 51000, "close": 55000}}
        )
        stop_at_high = portfolio.get_trailing_stop("BTC")
        assert stop_at_high is not None
        assert abs(stop_at_high - 52250) < 1

        # Price falls back to 53000 (low at 52500, which is above stop at 52250)
        portfolio.update_prices(
            {"BTC": 53000}, bar_index=2, ohlc_data={"BTC": {"open": 54000, "high": 54500, "low": 52500, "close": 53000}}
        )
        stop_after_fall = portfolio.get_trailing_stop("BTC")

        # Stop should NOT have moved down
        assert stop_after_fall == stop_at_high

    def test_trailing_stop_triggers(self):
        """Test that trailing stop triggers when price retraces."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter position at 50000
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )
        portfolio.order("BTC", 0.1)

        assert portfolio.get_position("BTC") == 0.1

        # Set 5% trailing stop
        portfolio.set_trailing_stop("BTC", trail_pct=0.05)

        # Price rises to 60000
        portfolio.update_prices(
            {"BTC": 60000}, bar_index=1, ohlc_data={"BTC": {"open": 55000, "high": 60000, "low": 54000, "close": 60000}}
        )

        # Trailing stop should be at 60000 * 0.95 = 57000
        stop_price_after_rise = portfolio.get_trailing_stop("BTC")
        assert stop_price_after_rise is not None
        assert abs(stop_price_after_rise - 57000) < 1

        # Position should still be open
        assert portfolio.get_position("BTC") == 0.1

        # Price falls to 56000 (crosses trailing stop at 57000)
        portfolio.update_prices(
            {"BTC": 56000}, bar_index=2, ohlc_data={"BTC": {"open": 59000, "high": 59000, "low": 56000, "close": 56500}}
        )

        # Position should be closed
        assert portfolio.get_position("BTC") == 0.0
        assert portfolio.get_trailing_stop("BTC") is None

    def test_trailing_stop_not_triggered_when_not_hit(self):
        """Test that trailing stop doesn't trigger prematurely."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter position
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )
        portfolio.order("BTC", 0.1)

        # Set 5% trailing stop
        portfolio.set_trailing_stop("BTC", trail_pct=0.05)

        # Price rises to 55000
        portfolio.update_prices(
            {"BTC": 55000}, bar_index=1, ohlc_data={"BTC": {"open": 52000, "high": 55000, "low": 51000, "close": 55000}}
        )

        # Price falls but doesn't hit trailing stop (stop at 52250, low at 51000)
        portfolio.update_prices(
            {"BTC": 53000}, bar_index=2, ohlc_data={"BTC": {"open": 54000, "high": 54500, "low": 52500, "close": 53000}}
        )

        # Position should still be open
        assert portfolio.get_position("BTC") == 0.1

    def test_remove_trailing_stop(self):
        """Test removing trailing stop."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter position and set trailing stop
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )
        portfolio.order("BTC", 0.1)
        portfolio.set_trailing_stop("BTC", trail_pct=0.05)

        assert portfolio.get_trailing_stop("BTC") is not None

        # Remove trailing stop
        result = portfolio.remove_trailing_stop("BTC")

        assert result is True
        assert portfolio.get_trailing_stop("BTC") is None

    def test_trailing_stop_no_position(self):
        """Test that trailing stop can't be set without a position."""
        portfolio = Portfolio(initial_cash=10000)
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )

        # Try to set trailing stop without position
        order_id = portfolio.set_trailing_stop("BTC", trail_pct=0.05)

        assert order_id is None

    def test_trailing_stop_with_absolute_amount(self):
        """Test trailing stop with absolute dollar amount."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter position at 50000
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )
        portfolio.order("BTC", 0.1)

        # Set $3000 trailing stop
        portfolio.set_trailing_stop("BTC", trail_amount=3000)
        assert portfolio.get_trailing_stop("BTC") == 47000

        # Price rises to 55000 (low stays above old stop)
        portfolio.update_prices(
            {"BTC": 55000}, bar_index=1, ohlc_data={"BTC": {"open": 52000, "high": 55000, "low": 51000, "close": 55000}}
        )

        # Stop should move to 52000 (55000 - 3000)
        assert portfolio.get_trailing_stop("BTC") == 52000

    def test_trailing_stop_cleanup_on_position_close(self):
        """Test that trailing stop is removed when position is manually closed."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter position
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )
        portfolio.order("BTC", 0.1)

        # Set trailing stop
        portfolio.set_trailing_stop("BTC", trail_pct=0.05)
        assert portfolio.get_trailing_stop("BTC") is not None

        # Manually close position
        portfolio.close_position("BTC")

        # Trailing stop should be removed (cleanup happens in _update_trade_tracker)
        # Actually, cleanup happens on next update_prices call when position is 0
        portfolio.update_prices(
            {"BTC": 51000}, bar_index=1, ohlc_data={"BTC": {"open": 51000, "high": 51000, "low": 51000, "close": 51000}}
        )
        assert portfolio.get_trailing_stop("BTC") is None
