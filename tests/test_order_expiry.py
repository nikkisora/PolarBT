"""Tests for order expiry functionality (GTC vs Day orders)."""

from polarbt.core import Portfolio
from polarbt.orders import OrderStatus


class TestOrderExpiry:
    """Test order expiry functionality."""

    def test_gtc_order_does_not_expire(self):
        """Test that GTC orders remain active indefinitely."""
        portfolio = Portfolio(initial_cash=10000)

        # Set price at 50000
        portfolio.update_prices({"BTC": 50000}, bar_index=0)

        # Place GTC limit buy below market (won't fill immediately)
        order_id = portfolio.order_gtc("BTC", 0.1, limit_price=49000)
        assert order_id is not None

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.valid_until is None  # GTC has no expiry
        assert order.status == OrderStatus.PENDING

        # Advance many bars without filling - order should remain active
        for bar in range(1, 100):
            portfolio.update_prices({"BTC": 50000}, bar_index=bar)

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.PENDING  # Still active

    def test_day_order_expires_after_bars(self):
        """Test that Day orders expire after specified bars."""
        portfolio = Portfolio(initial_cash=10000)

        # Set price at 50000
        portfolio.update_prices({"BTC": 50000}, bar_index=0)

        # Place Day order that expires after 2 bars
        order_id = portfolio.order_day("BTC", 0.1, limit_price=49000, bars_valid=2)
        assert order_id is not None

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.valid_until == 2  # Expires at bar 2 (0 + 2)
        assert order.status == OrderStatus.PENDING

        # Bar 1 - order should still be active
        portfolio.update_prices({"BTC": 50000}, bar_index=1)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.PENDING

        # Bar 2 - order should still be active (expires AFTER bar 2)
        portfolio.update_prices({"BTC": 50000}, bar_index=2)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.PENDING

        # Bar 3 - order should be expired
        portfolio.update_prices({"BTC": 50000}, bar_index=3)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.EXPIRED

    def test_day_order_default_expiry(self):
        """Test Day order with default 1-bar expiry."""
        portfolio = Portfolio(initial_cash=10000)

        # Set price at 50000
        portfolio.update_prices({"BTC": 50000}, bar_index=0)

        # Place Day order with default expiry (1 bar)
        order_id = portfolio.order_day("BTC", 0.1, limit_price=49000)
        assert order_id is not None

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.valid_until == 1  # Expires after bar 1

        # Bar 1 - order should still be pending
        portfolio.update_prices({"BTC": 50000}, bar_index=1)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.PENDING

        # Bar 2 - order should be expired
        portfolio.update_prices({"BTC": 50000}, bar_index=2)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.EXPIRED

    def test_expired_order_not_filled(self):
        """Test that expired orders don't execute even if price reaches limit."""
        portfolio = Portfolio(initial_cash=10000)

        # Set price at 50000
        portfolio.update_prices({"BTC": 50000}, bar_index=0)

        # Place Day order that expires after 1 bar
        order_id = portfolio.order_day("BTC", 0.1, limit_price=49000, bars_valid=1)
        assert order_id is not None

        # Advance to expiry without hitting limit
        portfolio.update_prices({"BTC": 50000}, bar_index=1)

        # Bar 2 - order expires
        portfolio.update_prices({"BTC": 50000}, bar_index=2)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.EXPIRED

        # Bar 3 - price hits limit, but order already expired
        portfolio.update_prices(
            {"BTC": 49000}, bar_index=3, ohlc_data={"BTC": {"open": 49500, "high": 49500, "low": 49000, "close": 49000}}
        )

        # Order should remain expired, not filled
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.EXPIRED
        assert portfolio.get_position("BTC") == 0.0

    def test_day_order_fills_before_expiry(self):
        """Test that Day order fills if limit hit before expiry."""
        portfolio = Portfolio(initial_cash=10000)

        # Set price at 50000
        portfolio.update_prices(
            {"BTC": 50000}, bar_index=0, ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}}
        )

        # Place Day order that expires after 2 bars
        order_id = portfolio.order_day("BTC", 0.1, limit_price=49000, bars_valid=2)
        assert order_id is not None

        # Bar 1 - price hits limit
        portfolio.update_prices(
            {"BTC": 49000}, bar_index=1, ohlc_data={"BTC": {"open": 49500, "high": 49500, "low": 49000, "close": 49000}}
        )

        # Order should be filled
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert portfolio.get_position("BTC") == 0.1

        # Bar 2 - order already filled, so expiry doesn't matter
        portfolio.update_prices({"BTC": 49000}, bar_index=2)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.FILLED

    def test_multiple_day_orders_expiry(self):
        """Test multiple Day orders with different expiry times."""
        portfolio = Portfolio(initial_cash=10000)

        # Set price at 50000
        portfolio.update_prices({"BTC": 50000}, bar_index=0)

        # Place three Day orders with different expiries
        order1_id = portfolio.order_day("BTC", 0.01, limit_price=49000, bars_valid=1)
        order2_id = portfolio.order_day("BTC", 0.01, limit_price=48000, bars_valid=2)
        order3_id = portfolio.order_day("BTC", 0.01, limit_price=47000, bars_valid=3)
        assert order1_id is not None
        assert order2_id is not None
        assert order3_id is not None

        # Bar 1
        portfolio.update_prices({"BTC": 50000}, bar_index=1)
        order1 = portfolio.get_order(order1_id)
        order2 = portfolio.get_order(order2_id)
        order3 = portfolio.get_order(order3_id)
        assert order1 is not None
        assert order2 is not None
        assert order3 is not None
        assert order1.status == OrderStatus.PENDING
        assert order2.status == OrderStatus.PENDING
        assert order3.status == OrderStatus.PENDING

        # Bar 2 - order1 expires
        portfolio.update_prices({"BTC": 50000}, bar_index=2)
        order1 = portfolio.get_order(order1_id)
        order2 = portfolio.get_order(order2_id)
        order3 = portfolio.get_order(order3_id)
        assert order1 is not None
        assert order2 is not None
        assert order3 is not None
        assert order1.status == OrderStatus.EXPIRED
        assert order2.status == OrderStatus.PENDING
        assert order3.status == OrderStatus.PENDING

        # Bar 3 - order2 expires
        portfolio.update_prices({"BTC": 50000}, bar_index=3)
        order1 = portfolio.get_order(order1_id)
        order2 = portfolio.get_order(order2_id)
        order3 = portfolio.get_order(order3_id)
        assert order1 is not None
        assert order2 is not None
        assert order3 is not None
        assert order1.status == OrderStatus.EXPIRED
        assert order2.status == OrderStatus.EXPIRED
        assert order3.status == OrderStatus.PENDING

        # Bar 4 - order3 expires
        portfolio.update_prices({"BTC": 50000}, bar_index=4)
        order1 = portfolio.get_order(order1_id)
        order2 = portfolio.get_order(order2_id)
        order3 = portfolio.get_order(order3_id)
        assert order1 is not None
        assert order2 is not None
        assert order3 is not None
        assert order1.status == OrderStatus.EXPIRED
        assert order2.status == OrderStatus.EXPIRED
        assert order3.status == OrderStatus.EXPIRED

    def test_regular_order_is_gtc(self):
        """Test that regular order() behaves as GTC."""
        portfolio = Portfolio(initial_cash=10000)

        # Set price at 50000
        portfolio.update_prices({"BTC": 50000}, bar_index=0)

        # Regular order without expiry
        order_id = portfolio.order("BTC", 0.1, limit_price=49000)
        assert order_id is not None

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.valid_until is None  # GTC behavior

        # Advance many bars
        for bar in range(1, 50):
            portfolio.update_prices({"BTC": 50000}, bar_index=bar)

        # Order should still be pending
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.PENDING


class TestDayOrderAutoExpiry:
    """Test automatic day order expiry using timestamps."""

    def test_day_order_with_datetime_expires_on_date_change(self):
        """Test day order expires when date changes (using datetime timestamps)."""
        from datetime import datetime

        portfolio = Portfolio(initial_cash=10000)

        # Day 1: 2024-01-15 09:30
        ts1 = datetime(2024, 1, 15, 9, 30)
        portfolio.update_prices({"BTC": 50000}, bar_index=0, timestamp=ts1)

        # Place day order - should expire at end of day
        order_id = portfolio.order_day("BTC", 0.1, limit_price=49000)
        assert order_id is not None
        order = portfolio.get_order(order_id)
        assert order is not None

        # Order should have expiry_date set to 2024-01-15
        assert order.expiry_date is not None
        assert order.expiry_date.year == 2024
        assert order.expiry_date.month == 1
        assert order.expiry_date.day == 15

        # Still day 1: 2024-01-15 15:30 - order should remain active
        ts2 = datetime(2024, 1, 15, 15, 30)
        portfolio.update_prices({"BTC": 50000}, bar_index=1, timestamp=ts2)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.PENDING

        # Day 2: 2024-01-16 09:30 - order should expire
        ts3 = datetime(2024, 1, 16, 9, 30)
        portfolio.update_prices({"BTC": 50000}, bar_index=2, timestamp=ts3)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.EXPIRED

    def test_day_order_with_unix_timestamp_expires_on_date_change(self):
        """Test day order expires when date changes (using unix timestamps)."""
        portfolio = Portfolio(initial_cash=10000)

        # Day 1: 2024-01-15 (unix timestamp)
        ts1 = 1705305000  # 2024-01-15 09:30:00 UTC
        portfolio.update_prices({"BTC": 50000}, bar_index=0, timestamp=ts1)

        # Place day order
        order_id = portfolio.order_day("BTC", 0.1, limit_price=49000)
        assert order_id is not None
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.expiry_date is not None

        # Same day - order active
        ts2 = 1705326600  # 2024-01-15 15:30:00 UTC
        portfolio.update_prices({"BTC": 50000}, bar_index=1, timestamp=ts2)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.PENDING

        # Next day - order expired
        ts3 = 1705391400  # 2024-01-16 09:30:00 UTC
        portfolio.update_prices({"BTC": 50000}, bar_index=2, timestamp=ts3)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.EXPIRED

    def test_day_order_fills_before_date_expiry(self):
        """Test day order fills before date changes."""
        from datetime import datetime

        portfolio = Portfolio(initial_cash=10000)

        # Day 1: Place order
        ts1 = datetime(2024, 1, 15, 9, 30)
        portfolio.update_prices(
            {"BTC": 50000},
            bar_index=0,
            timestamp=ts1,
            ohlc_data={"BTC": {"open": 50000, "high": 50000, "low": 50000, "close": 50000}},
        )

        order_id = portfolio.order_day("BTC", 0.1, limit_price=49000)
        assert order_id is not None

        # Day 1: Price hits limit - order fills
        ts2 = datetime(2024, 1, 15, 15, 30)
        portfolio.update_prices(
            {"BTC": 49000},
            bar_index=1,
            timestamp=ts2,
            ohlc_data={"BTC": {"open": 49500, "high": 49500, "low": 49000, "close": 49000}},
        )

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert portfolio.get_position("BTC") == 0.1

        # Day 2: Order already filled, expiry doesn't matter
        ts3 = datetime(2024, 1, 16, 9, 30)
        portfolio.update_prices({"BTC": 49000}, bar_index=2, timestamp=ts3)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.FILLED

    def test_day_order_with_bars_per_day_config(self):
        """Test day order using bars_per_day configuration (no timestamps)."""
        # Portfolio configured for 1-min bars (390 bars per 6.5 hour trading day)
        portfolio = Portfolio(initial_cash=10000, bars_per_day=390)

        # Start of day: bar 0
        portfolio.update_prices({"BTC": 50000}, bar_index=0)

        # Place day order at start of day
        order_id = portfolio.order_day("BTC", 0.1, limit_price=49000)
        assert order_id is not None
        order = portfolio.get_order(order_id)
        assert order is not None

        # Should expire at end of day 0 (bar 389)
        # current_bar_in_day = 0 % 390 = 0
        # bars_until_eod = 390 - 0 - 1 = 389
        # valid_until = 0 + 389 = 389
        assert order.valid_until == 389

        # Advance to bar 389 - still active
        portfolio.update_prices({"BTC": 50000}, bar_index=389)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.PENDING

        # Bar 390 (start of next day) - expired
        portfolio.update_prices({"BTC": 50000}, bar_index=390)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.EXPIRED

    def test_day_order_with_bars_per_day_mid_day(self):
        """Test day order placed mid-day expires at end of same day."""
        # Portfolio configured for 1-min bars (390 bars per day)
        portfolio = Portfolio(initial_cash=10000, bars_per_day=390)

        # Mid-day: bar 100 (within day 0)
        portfolio.update_prices({"BTC": 50000}, bar_index=100)

        # Place day order mid-day
        order_id = portfolio.order_day("BTC", 0.1, limit_price=49000)
        assert order_id is not None
        order = portfolio.get_order(order_id)
        assert order is not None

        # Should expire at end of day 0 (bar 389), not bar 490
        # current_bar_in_day = 100 % 390 = 100
        # bars_until_eod = 390 - 100 - 1 = 289
        # valid_until = 100 + 289 = 389
        assert order.valid_until == 389

        # Advance to bar 389 - still active
        portfolio.update_prices({"BTC": 50000}, bar_index=389)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.PENDING

        # Bar 390 - expired
        portfolio.update_prices({"BTC": 50000}, bar_index=390)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.EXPIRED

    def test_day_order_with_bars_per_day_second_day(self):
        """Test day order placed on second day expires at end of that day."""
        # Portfolio configured for 390 bars per day
        portfolio = Portfolio(initial_cash=10000, bars_per_day=390)

        # Day 1, bar 450 (bar 60 of day 1)
        portfolio.update_prices({"BTC": 50000}, bar_index=450)

        # Place day order on day 1
        order_id = portfolio.order_day("BTC", 0.1, limit_price=49000)
        assert order_id is not None
        order = portfolio.get_order(order_id)
        assert order is not None

        # Should expire at end of day 1 (bar 779)
        # current_bar_in_day = 450 % 390 = 60
        # bars_until_eod = 390 - 60 - 1 = 329
        # valid_until = 450 + 329 = 779
        assert order.valid_until == 779

        # Advance to bar 779 - still active
        portfolio.update_prices({"BTC": 50000}, bar_index=779)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.PENDING

        # Bar 780 (start of day 2) - expired
        portfolio.update_prices({"BTC": 50000}, bar_index=780)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.EXPIRED

    def test_day_order_explicit_bars_valid_overrides_auto(self):
        """Test explicit bars_valid parameter overrides automatic detection."""
        from datetime import datetime

        portfolio = Portfolio(initial_cash=10000, bars_per_day=10)

        # Day 1 with timestamp
        ts1 = datetime(2024, 1, 15, 9, 30)
        portfolio.update_prices({"BTC": 50000}, bar_index=0, timestamp=ts1)

        # Explicitly set bars_valid=2 (overrides both timestamp and bars_per_day)
        order_id = portfolio.order_day("BTC", 0.1, limit_price=49000, bars_valid=2)
        assert order_id is not None
        order = portfolio.get_order(order_id)
        assert order is not None

        # Should use explicit bars_valid
        assert order.valid_until == 2  # 0 + 2
        assert order.expiry_date is None  # Not using date-based expiry

        # Bar 2 - still active
        ts2 = datetime(2024, 1, 15, 10, 30)
        portfolio.update_prices({"BTC": 50000}, bar_index=2, timestamp=ts2)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.PENDING

        # Bar 3 - expired
        ts3 = datetime(2024, 1, 15, 11, 30)
        portfolio.update_prices({"BTC": 50000}, bar_index=3, timestamp=ts3)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.EXPIRED

    def test_day_order_multiple_orders_different_days(self):
        """Test multiple day orders placed on different days."""
        from datetime import datetime

        portfolio = Portfolio(initial_cash=10000)

        # Day 1: Place first order
        ts1 = datetime(2024, 1, 15, 9, 30)
        portfolio.update_prices({"BTC": 50000}, bar_index=0, timestamp=ts1)
        order1_id = portfolio.order_day("BTC", 0.1, limit_price=49000)
        assert order1_id is not None

        # Day 2: Place second order
        ts2 = datetime(2024, 1, 16, 9, 30)
        portfolio.update_prices({"BTC": 50000}, bar_index=1, timestamp=ts2)
        order2_id = portfolio.order_day("BTC", 0.1, limit_price=48000)
        assert order2_id is not None

        # Check orders
        order1 = portfolio.get_order(order1_id)
        order2 = portfolio.get_order(order2_id)
        assert order1 is not None
        assert order2 is not None

        # Order1 should be expired (placed on day 1, now day 2)
        assert order1.status == OrderStatus.EXPIRED

        # Order2 should be active (placed on day 2, still day 2)
        assert order2.status == OrderStatus.PENDING

        # Day 3: Order2 should expire
        ts3 = datetime(2024, 1, 17, 9, 30)
        portfolio.update_prices({"BTC": 50000}, bar_index=2, timestamp=ts3)
        order2 = portfolio.get_order(order2_id)
        assert order2 is not None
        assert order2.status == OrderStatus.EXPIRED

    def test_day_order_fallback_to_default(self):
        """Test day order falls back to 1 bar when no config provided."""
        portfolio = Portfolio(initial_cash=10000)  # No bars_per_day, no timestamp

        # No timestamp
        portfolio.update_prices({"BTC": 50000}, bar_index=0)

        # Place day order - should default to 1 bar
        order_id = portfolio.order_day("BTC", 0.1, limit_price=49000)
        assert order_id is not None
        order = portfolio.get_order(order_id)
        assert order is not None

        # Should expire after 1 bar
        assert order.valid_until == 1  # 0 + 1

        # Bar 1 - still active
        portfolio.update_prices({"BTC": 50000}, bar_index=1)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.PENDING

        # Bar 2 - expired
        portfolio.update_prices({"BTC": 50000}, bar_index=2)
        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.EXPIRED
