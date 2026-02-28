"""Tests for bugfixes identified during code review.

Covers:
- SL/TP/trailing stop fill at trigger price (not market close)
- Fixed commission charged twice on position reversals
- Position increase updates average entry price in TradeTracker
- order_target_percent no longer double-applies slippage
- monthly_returns uses previous month's end equity
- Equity curve excludes warmup bars
- daily_win_rate rename in calculate_metrics
"""

from datetime import datetime

import polars as pl

from polarbtest.core import Engine, Portfolio, Strategy
from polarbtest.metrics import calculate_metrics, monthly_returns
from polarbtest.trades import TradeTracker

# ---------------------------------------------------------------------------
# Fix 2: SL/TP/trailing stop fill at trigger price + slippage
# ---------------------------------------------------------------------------


class TestSLTPFillPrice:
    """Verify SL/TP/trailing stop orders fill at the stop/target price, not at close."""

    def test_stop_loss_fills_at_stop_price(self):
        """Long SL at 95 should fill at 95 (+ slippage), not at the bar close."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.001)
        portfolio.update_prices(
            {"BTC": 100.0}, bar_index=0, ohlc_data={"BTC": {"open": 100, "high": 100, "low": 100, "close": 100}}
        )
        portfolio.order("BTC", 10.0)  # buy 10 at 100

        portfolio.set_stop_loss("BTC", stop_price=95.0)

        # Bar where low touches stop but close is 98
        portfolio.update_prices(
            {"BTC": 98.0},
            bar_index=1,
            ohlc_data={"BTC": {"open": 99, "high": 99, "low": 94, "close": 98}},
        )

        # Position should be closed
        assert portfolio.get_position("BTC") == 0

        # Find the closing order
        filled_orders = [o for o in portfolio.orders.values() if o.is_filled() and o.size < 0]
        assert len(filled_orders) == 1
        fill_price = filled_orders[0].filled_price
        assert fill_price is not None
        # Should fill at 95 * (1 - slippage) = 94.905, NOT at 98
        expected = 95.0 * (1 - 0.001)
        assert abs(fill_price - expected) < 0.01

    def test_take_profit_fills_at_target_price(self):
        """Long TP at 110 should fill at 110 (+ slippage), not at close."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.001)
        portfolio.update_prices(
            {"BTC": 100.0}, bar_index=0, ohlc_data={"BTC": {"open": 100, "high": 100, "low": 100, "close": 100}}
        )
        portfolio.order("BTC", 10.0)

        portfolio.set_take_profit("BTC", target_price=110.0)

        # Bar where high touches target but close is 108
        portfolio.update_prices(
            {"BTC": 108.0},
            bar_index=1,
            ohlc_data={"BTC": {"open": 105, "high": 112, "low": 105, "close": 108}},
        )

        assert portfolio.get_position("BTC") == 0
        filled_orders = [o for o in portfolio.orders.values() if o.is_filled() and o.size < 0]
        assert len(filled_orders) == 1
        fill_price = filled_orders[0].filled_price
        assert fill_price is not None
        expected = 110.0 * (1 - 0.001)  # sell with slippage
        assert abs(fill_price - expected) < 0.01

    def test_trailing_stop_fills_at_stop_price(self):
        """Trailing stop should fill at the trailing stop price, not close."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.001)
        portfolio.update_prices(
            {"BTC": 100.0}, bar_index=0, ohlc_data={"BTC": {"open": 100, "high": 100, "low": 100, "close": 100}}
        )
        portfolio.order("BTC", 10.0)

        portfolio.set_trailing_stop("BTC", trail_pct=0.05)
        trailing_stop_price = portfolio.get_trailing_stop("BTC")
        assert trailing_stop_price is not None
        assert abs(trailing_stop_price - 95.0) < 0.01

        # Bar where low triggers the trailing stop but close is higher
        portfolio.update_prices(
            {"BTC": 97.0},
            bar_index=1,
            ohlc_data={"BTC": {"open": 99, "high": 99, "low": 94, "close": 97}},
        )

        assert portfolio.get_position("BTC") == 0
        filled_orders = [o for o in portfolio.orders.values() if o.is_filled() and o.size < 0]
        assert len(filled_orders) == 1
        fill_price = filled_orders[0].filled_price
        assert fill_price is not None
        expected = 95.0 * (1 - 0.001)
        assert abs(fill_price - expected) < 0.01

    def test_short_stop_loss_fills_at_stop_price(self):
        """Short SL at 105 should fill at 105 (+ slippage for buy)."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.001)
        portfolio.update_prices(
            {"BTC": 100.0}, bar_index=0, ohlc_data={"BTC": {"open": 100, "high": 100, "low": 100, "close": 100}}
        )
        portfolio.order("BTC", -10.0)  # short 10 at 100

        portfolio.set_stop_loss("BTC", stop_price=105.0)

        # Bar where high triggers stop but close is 103
        portfolio.update_prices(
            {"BTC": 103.0},
            bar_index=1,
            ohlc_data={"BTC": {"open": 102, "high": 106, "low": 102, "close": 103}},
        )

        assert portfolio.get_position("BTC") == 0
        filled_orders = [o for o in portfolio.orders.values() if o.is_filled() and o.size > 0]
        assert len(filled_orders) == 1
        fill_price = filled_orders[0].filled_price
        assert fill_price is not None
        expected = 105.0 * (1 + 0.001)  # buy with slippage
        assert abs(fill_price - expected) < 0.01


# ---------------------------------------------------------------------------
# Fix 3+6: Fixed commission on reversals
# ---------------------------------------------------------------------------


class TestReversalCommission:
    """Verify fixed commission is charged twice when an order crosses zero."""

    def test_reversal_charges_double_fixed_commission(self):
        """Long→short reversal should charge 2x fixed commission."""
        fixed = 10.0
        portfolio = Portfolio(initial_cash=100_000, commission=(fixed, 0.0))
        portfolio.update_prices(
            {"A": 100.0}, bar_index=0, ohlc_data={"A": {"open": 100, "high": 100, "low": 100, "close": 100}}
        )

        # Open long position of 10 at 100 → cost 1000 + 10 commission
        portfolio.order("A", 10.0)
        cash_after_buy = portfolio.cash
        assert abs(cash_after_buy - (100_000 - 1000 - 10)) < 0.01

        # Reverse to short 10 → sell 20 units. Should charge 2x fixed = 20
        portfolio.update_prices(
            {"A": 100.0}, bar_index=1, ohlc_data={"A": {"open": 100, "high": 100, "low": 100, "close": 100}}
        )
        portfolio.order("A", -20.0)

        # Proceeds: sell 20 * 100 = 2000, minus 2 * 10 fixed = 20
        # Cash = cash_after_buy + 2000 - 20
        expected_cash = cash_after_buy + 2000 - 20
        assert abs(portfolio.cash - expected_cash) < 0.01
        assert portfolio.get_position("A") == -10.0

    def test_cover_short_to_long_charges_double_fixed(self):
        """Short→long reversal via buy should charge 2x fixed commission."""
        fixed = 5.0
        portfolio = Portfolio(initial_cash=100_000, commission=(fixed, 0.0))
        portfolio.update_prices(
            {"A": 50.0}, bar_index=0, ohlc_data={"A": {"open": 50, "high": 50, "low": 50, "close": 50}}
        )

        # Open short: sell 10 at 50 → receive 500 - 5 = 495
        portfolio.order("A", -10.0)
        cash_after_short = portfolio.cash

        # Buy 20 to reverse to long 10
        portfolio.update_prices(
            {"A": 50.0}, bar_index=1, ohlc_data={"A": {"open": 50, "high": 50, "low": 50, "close": 50}}
        )
        portfolio.order("A", 20.0)

        # Cost: 20 * 50 = 1000 + 2 * 5 = 10 fixed commission
        expected_cash = cash_after_short - 1000 - 10
        assert abs(portfolio.cash - expected_cash) < 0.01
        assert portfolio.get_position("A") == 10.0

    def test_non_reversal_single_fixed_commission(self):
        """Simple close (no reversal) should charge 1x fixed commission."""
        fixed = 10.0
        portfolio = Portfolio(initial_cash=100_000, commission=(fixed, 0.0))
        portfolio.update_prices(
            {"A": 100.0}, bar_index=0, ohlc_data={"A": {"open": 100, "high": 100, "low": 100, "close": 100}}
        )

        portfolio.order("A", 10.0)
        cash_after_buy = portfolio.cash

        # Close position (sell 10)
        portfolio.update_prices(
            {"A": 100.0}, bar_index=1, ohlc_data={"A": {"open": 100, "high": 100, "low": 100, "close": 100}}
        )
        portfolio.order("A", -10.0)

        # Proceeds: 10 * 100 - 10 fixed = 990
        expected_cash = cash_after_buy + 990
        assert abs(portfolio.cash - expected_cash) < 0.01


# ---------------------------------------------------------------------------
# Fix 4: Position increase updates average entry price
# ---------------------------------------------------------------------------


class TestPositionIncrease:
    """Verify that adding to an existing position updates the average entry price."""

    def test_increase_long_updates_avg_price(self):
        tracker = TradeTracker()
        tracker.on_position_opened("BTC", 5.0, 100.0, bar=0, timestamp=0, commission=1.0)

        # Add 5 more at 120
        tracker.on_position_increased("BTC", added_size=5.0, price=120.0, commission=1.0)

        pos = tracker.open_positions["BTC"]
        assert pos["entry_size"] == 10.0
        assert abs(pos["entry_price"] - 110.0) < 0.01  # (5*100 + 5*120) / 10 = 110
        assert abs(pos["entry_value"] - 1100.0) < 0.01
        assert abs(pos["entry_commission"] - 2.0) < 0.01

    def test_increase_then_close_pnl(self):
        """Verify P&L is correct when position was increased then closed."""
        tracker = TradeTracker()
        tracker.on_position_opened("BTC", 5.0, 100.0, bar=0, timestamp=0, commission=0.0)
        tracker.on_position_increased("BTC", added_size=5.0, price=120.0, commission=0.0)

        # Close all at 130
        trade = tracker.on_position_closed("BTC", 10.0, 130.0, bar=5, timestamp=5, commission=0.0)
        assert trade is not None
        # P&L: exit_value - entry_value = 10*130 - (5*100+5*120) = 1300 - 1100 = 200
        assert abs(trade.pnl - 200.0) < 0.01

    def test_engine_tracks_position_increase(self):
        """Verify the Engine properly routes position increases to TradeTracker."""
        portfolio = Portfolio(initial_cash=100_000)
        portfolio.update_prices(
            {"A": 100.0}, bar_index=0, ohlc_data={"A": {"open": 100, "high": 100, "low": 100, "close": 100}}
        )
        portfolio.order("A", 5.0)

        portfolio.update_prices(
            {"A": 120.0}, bar_index=1, ohlc_data={"A": {"open": 120, "high": 120, "low": 120, "close": 120}}
        )
        portfolio.order("A", 5.0)

        # Check the open position was updated
        pos = portfolio.trade_tracker.open_positions.get("A")
        assert pos is not None
        assert pos["entry_size"] == 10.0
        assert abs(pos["entry_price"] - 110.0) < 0.01


# ---------------------------------------------------------------------------
# Fix 5: order_target_percent no double slippage
# ---------------------------------------------------------------------------


class TestOrderTargetPercentSlippage:
    """Verify order_target_percent doesn't double-apply slippage."""

    def test_target_percent_reasonable_position(self):
        """With 50% target, the resulting position should be close to 50% of portfolio."""
        slippage = 0.01  # 1% slippage — large enough to notice doubling
        portfolio = Portfolio(initial_cash=100_000, slippage=slippage)
        portfolio.update_prices(
            {"A": 100.0}, bar_index=0, ohlc_data={"A": {"open": 100, "high": 100, "low": 100, "close": 100}}
        )

        portfolio.order_target_percent("A", 0.5)

        position = portfolio.get_position("A")
        position_value = position * 100.0  # at current price
        portfolio_value = portfolio.get_value()

        # Position should be approximately 50% of portfolio value
        # Allow tolerance for slippage + commission but it should not be wildly off
        actual_pct = position_value / portfolio_value
        assert 0.45 < actual_pct < 0.55, f"Position is {actual_pct * 100:.1f}% of portfolio, expected ~50%"


# ---------------------------------------------------------------------------
# Fix 7: monthly_returns uses previous month's end equity
# ---------------------------------------------------------------------------


class TestMonthlyReturns:
    """Verify monthly returns calculation uses previous month-end equity."""

    def test_monthly_returns_cross_month_boundary(self):
        """Monthly return should be calculated from previous month's last equity."""
        # Month 1: equity goes 100 -> 110
        # Month 2: equity goes 115 -> 120  (note: gap from 110 to 115 within month 2)
        timestamps = [
            datetime(2024, 1, 15),
            datetime(2024, 1, 31),
            datetime(2024, 2, 1),
            datetime(2024, 2, 28),
        ]
        equities = [100.0, 110.0, 115.0, 120.0]

        df = pl.DataFrame({"timestamp": timestamps, "equity": equities})
        result = monthly_returns(df)

        assert len(result) == 2
        rows = result.sort(["year", "month"]).to_dicts()

        # January: (110 - 100) / 100 = 0.10
        assert abs(rows[0]["return"] - 0.10) < 0.001

        # February: (120 - 110) / 110 ≈ 0.0909 (uses Jan end=110 as start)
        assert abs(rows[1]["return"] - (120 - 110) / 110) < 0.001


# ---------------------------------------------------------------------------
# Fix 9: Equity curve excludes warmup
# ---------------------------------------------------------------------------


class TestEquityCurveWarmup:
    """Verify equity is only recorded after warmup period."""

    def test_warmup_bars_excluded_from_equity(self):
        class DoNothing(Strategy):
            def preprocess(self, df):
                return df

            def next(self, ctx):
                pass

        df = pl.DataFrame({"close": [100.0] * 50})
        engine = Engine(strategy=DoNothing(), data=df, warmup=10)
        engine.run()

        # With 50 bars and 10 warmup, equity should have 40 entries
        assert engine.portfolio is not None
        assert len(engine.portfolio.equity_curve) == 40


# ---------------------------------------------------------------------------
# Fix 10: daily_win_rate rename
# ---------------------------------------------------------------------------


class TestDailyWinRateRename:
    """Verify calculate_metrics uses daily_win_rate key, not win_rate."""

    def test_metrics_has_daily_win_rate(self):
        df = pl.DataFrame({"timestamp": list(range(10)), "equity": [100 + i for i in range(10)]})
        metrics = calculate_metrics(df, 100.0)

        assert "daily_win_rate" in metrics
        assert "win_rate" not in metrics
        assert "daily_avg_win" in metrics
        assert "daily_avg_loss" in metrics
