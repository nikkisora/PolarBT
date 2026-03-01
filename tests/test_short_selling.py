"""Tests for short selling and borrow costs."""

import polars as pl
import pytest

from polarbt.core import Engine, Portfolio, Strategy
from polarbt.orders import OrderStatus


class TestShortSelling:
    """Test short selling functionality."""

    def test_open_short_position(self):
        """Test opening a short position from zero."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0)
        portfolio.update_prices({"BTC": 50_000}, bar_index=0)

        order_id = portfolio.order("BTC", -1.0)
        assert order_id is not None

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert portfolio.get_position("BTC") == -1.0
        # Cash increases by sale proceeds
        assert portfolio.cash == pytest.approx(150_000, abs=1)

    def test_close_short_position(self):
        """Test closing a short position (buying to cover)."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0)
        portfolio.update_prices({"BTC": 50_000}, bar_index=0)

        # Open short
        portfolio.order("BTC", -1.0)
        assert portfolio.get_position("BTC") == -1.0
        assert portfolio.cash == pytest.approx(150_000, abs=1)

        # Close short (buy to cover) at same price
        portfolio.order("BTC", 1.0)
        assert portfolio.get_position("BTC") == 0.0
        # Cash: 150_000 - 50_000 (buy back) = 100_000 (break even)
        assert portfolio.cash == pytest.approx(100_000, abs=1)

    def test_short_position_profit(self):
        """Test profit from short selling when price drops."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0)

        # Short at 50000
        portfolio.update_prices({"BTC": 50_000}, bar_index=0)
        portfolio.order("BTC", -1.0)
        assert portfolio.cash == pytest.approx(150_000, abs=1)

        # Cover at 45000 (price dropped)
        portfolio.update_prices({"BTC": 45_000}, bar_index=1)
        portfolio.order("BTC", 1.0)
        assert portfolio.get_position("BTC") == 0.0
        # Cash: 150_000 - 45_000 = 105_000 (profit of 5000)
        assert portfolio.cash == pytest.approx(105_000, abs=1)

    def test_short_position_loss(self):
        """Test loss from short selling when price rises."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0)

        # Short at 50000
        portfolio.update_prices({"BTC": 50_000}, bar_index=0)
        portfolio.order("BTC", -1.0)

        # Cover at 55000 (price rose)
        portfolio.update_prices({"BTC": 55_000}, bar_index=1)
        portfolio.order("BTC", 1.0)
        assert portfolio.get_position("BTC") == 0.0
        # Cash: 150_000 - 55_000 = 95_000 (loss of 5000)
        assert portfolio.cash == pytest.approx(95_000, abs=1)

    def test_short_portfolio_value(self):
        """Test portfolio value calculation with short positions."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0)

        portfolio.update_prices({"BTC": 50_000}, bar_index=0)
        portfolio.order("BTC", -1.0)

        # Value = cash + positions = 150_000 + (-1 * 50_000) = 100_000
        assert portfolio.get_value() == pytest.approx(100_000, abs=1)

        # Price drops — short is profitable
        portfolio.update_prices({"BTC": 45_000}, bar_index=1)
        # Value = 150_000 + (-1 * 45_000) = 105_000
        assert portfolio.get_value() == pytest.approx(105_000, abs=1)

        # Price rises — short is losing
        portfolio.update_prices({"BTC": 55_000}, bar_index=2)
        # Value = 150_000 + (-1 * 55_000) = 95_000
        assert portfolio.get_value() == pytest.approx(95_000, abs=1)

    def test_short_with_commission(self):
        """Test short selling with commission."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.001)

        portfolio.update_prices({"BTC": 50_000}, bar_index=0)
        portfolio.order("BTC", -1.0)

        # Proceeds = 50_000 - commission (50_000 * 0.001 = 50) = 49_950
        assert portfolio.cash == pytest.approx(149_950, abs=1)

    def test_short_with_slippage(self):
        """Test short selling with slippage."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.01, commission=0.0)

        portfolio.update_prices({"BTC": 50_000}, bar_index=0)
        portfolio.order("BTC", -1.0)

        # Execution price = 50_000 * (1 - 0.01) = 49_500 (slippage hurts seller)
        # Proceeds = 49_500
        assert portfolio.cash == pytest.approx(149_500, abs=1)

    def test_short_close_position(self):
        """Test close_position works for short positions."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0)

        portfolio.update_prices({"BTC": 50_000}, bar_index=0)
        portfolio.order("BTC", -1.0)
        assert portfolio.get_position("BTC") == -1.0

        portfolio.close_position("BTC")
        assert portfolio.get_position("BTC") == 0.0

    def test_short_order_target(self):
        """Test order_target with short positions."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0)

        portfolio.update_prices({"BTC": 50_000}, bar_index=0)

        # Target -1 BTC (short)
        portfolio.order_target("BTC", -1.0)
        assert portfolio.get_position("BTC") == -1.0

        # Target -0.5 BTC (reduce short)
        portfolio.order_target("BTC", -0.5)
        assert portfolio.get_position("BTC") == pytest.approx(-0.5)

    def test_position_reversal_long_to_short(self):
        """Test reversing from long to short in a single order."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0)

        portfolio.update_prices({"BTC": 50_000}, bar_index=0)

        # Go long 1 BTC
        portfolio.order("BTC", 1.0)
        assert portfolio.get_position("BTC") == 1.0
        assert portfolio.cash == pytest.approx(50_000, abs=1)

        # Sell 2 BTC (close long + open short)
        portfolio.order("BTC", -2.0)
        assert portfolio.get_position("BTC") == -1.0
        # Cash: 50_000 + (1 * 50_000 for closing long) + (1 * 50_000 for opening short) = 150_000
        assert portfolio.cash == pytest.approx(150_000, abs=1)

    def test_position_reversal_short_to_long(self):
        """Test reversing from short to long in a single order."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0)

        portfolio.update_prices({"BTC": 50_000}, bar_index=0)

        # Go short 1 BTC
        portfolio.order("BTC", -1.0)
        assert portfolio.get_position("BTC") == -1.0
        assert portfolio.cash == pytest.approx(150_000, abs=1)

        # Buy 2 BTC (cover short + open long)
        portfolio.order("BTC", 2.0)
        assert portfolio.get_position("BTC") == 1.0
        # Cash: 150_000 - (1 * 50_000 for covering) - (1 * 50_000 for new long) = 50_000
        assert portfolio.cash == pytest.approx(50_000, abs=1)

    def test_short_trade_tracking(self):
        """Test that short trades are tracked correctly."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0)

        # Open short at 50000
        portfolio.update_prices({"BTC": 50_000}, bar_index=0)
        portfolio.order("BTC", -1.0)

        # Close short at 45000
        portfolio.update_prices({"BTC": 45_000}, bar_index=5)
        portfolio.order("BTC", 1.0)

        trades = portfolio.trade_tracker.get_trades()
        assert len(trades) == 1

        trade = trades[0]
        assert trade.direction == "short"
        assert trade.entry_price == 50_000
        assert trade.exit_price == 45_000
        assert trade.pnl == pytest.approx(5_000, abs=1)  # Profit from short
        assert trade.bars_held == 5

    def test_short_stop_loss(self):
        """Test stop-loss on short position (triggers when price rises)."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0)

        # Open short
        portfolio.update_prices(
            {"BTC": 50_000},
            bar_index=0,
            ohlc_data={"BTC": {"open": 50_000, "high": 50_000, "low": 50_000, "close": 50_000}},
        )
        portfolio.order("BTC", -1.0)

        # Set stop-loss at 52000 (above entry for short)
        portfolio.set_stop_loss("BTC", stop_price=52_000)
        assert portfolio.get_stop_loss("BTC") == 52_000

        # Price rises and hits stop
        portfolio.update_prices(
            {"BTC": 53_000},
            bar_index=1,
            ohlc_data={"BTC": {"open": 51_000, "high": 53_000, "low": 50_500, "close": 52_500}},
        )

        assert portfolio.get_position("BTC") == 0.0
        assert portfolio.get_stop_loss("BTC") is None

    def test_short_trailing_stop(self):
        """Test trailing stop on short position."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0)

        # Open short
        portfolio.update_prices(
            {"BTC": 50_000},
            bar_index=0,
            ohlc_data={"BTC": {"open": 50_000, "high": 50_000, "low": 50_000, "close": 50_000}},
        )
        portfolio.order("BTC", -1.0)

        # Set trailing stop at 5%
        portfolio.set_trailing_stop("BTC", trail_pct=0.05)
        # Initial stop = 50000 * 1.05 = 52500
        assert portfolio.get_trailing_stop("BTC") == pytest.approx(52_500, abs=1)

        # Price drops (favorable for short) — stop should move down
        portfolio.update_prices(
            {"BTC": 45_000},
            bar_index=1,
            ohlc_data={"BTC": {"open": 49_000, "high": 49_500, "low": 45_000, "close": 45_500}},
        )
        # New stop = 45000 * 1.05 = 47250
        assert portfolio.get_trailing_stop("BTC") == pytest.approx(47_250, abs=1)
        assert portfolio.get_position("BTC") == -1.0  # Still open

    def test_short_insufficient_cash_to_cover(self):
        """Test that covering a short fails if not enough cash."""
        portfolio = Portfolio(initial_cash=10_000, slippage=0.0, commission=0.0)

        # Short 1 BTC at 50000 — receive 50000, cash = 60000
        portfolio.update_prices({"BTC": 50_000}, bar_index=0)
        portfolio.order("BTC", -1.0)
        assert portfolio.cash == pytest.approx(60_000, abs=1)

        # Price rises to 100000 — covering costs 100000, but we only have 60000
        portfolio.update_prices({"BTC": 100_000}, bar_index=1)
        order_id = portfolio.order("BTC", 1.0)
        assert order_id is not None

        order = portfolio.get_order(order_id)
        assert order is not None
        assert order.status == OrderStatus.REJECTED


class TestShortSellingWithEngine:
    """Test short selling through the Engine."""

    def test_short_strategy(self):
        """Test a strategy that shorts."""

        class ShortStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def next(self, ctx):
                if ctx.bar_index == 1:
                    ctx.portfolio.order("asset", -1.0)
                elif ctx.bar_index == 5:
                    ctx.portfolio.close_position("asset")

        data = pl.DataFrame(
            {
                "timestamp": range(10),
                "close": [100.0, 100.0, 95.0, 90.0, 85.0, 80.0, 85.0, 90.0, 95.0, 100.0],
            }
        )

        engine = Engine(ShortStrategy(), data, initial_cash=100_000, warmup=0)
        results = engine.run()

        assert results is not None
        # Short at 100, cover at 80 — profit of 20
        assert results["total_return"] > 0

        trades_df = results["trades"]
        assert len(trades_df) == 1
        assert trades_df["direction"][0] == "short"


class TestBorrowCosts:
    """Test borrow cost deduction for short positions."""

    def test_borrow_cost_deducted_daily(self):
        """Test that borrow costs are deducted each bar for short positions."""
        # 10% annual borrow rate, daily bars (default bars_per_day=None means 1 bar = 1 day)
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0, borrow_rate=0.10)

        portfolio.update_prices({"BTC": 50_000}, bar_index=0)
        portfolio.order("BTC", -1.0)
        cash_after_short = portfolio.cash  # 150_000

        # Next bar: borrow cost deducted
        portfolio.update_prices({"BTC": 50_000}, bar_index=1)

        # Daily rate = 0.10 / 252 ≈ 0.000397
        # Borrow cost = 50_000 * 0.10 / 252 ≈ 19.84
        daily_rate = 0.10 / 252
        expected_cost = 50_000 * daily_rate
        assert portfolio.cash == pytest.approx(cash_after_short - expected_cost, abs=0.01)

    def test_borrow_cost_with_bars_per_day(self):
        """Test borrow cost with intraday bars."""
        # 10% annual, 390 bars per day (1-min bars)
        portfolio = Portfolio(
            initial_cash=100_000,
            slippage=0.0,
            commission=0.0,
            borrow_rate=0.10,
            bars_per_day=390,
        )

        portfolio.update_prices({"BTC": 50_000}, bar_index=0)
        portfolio.order("BTC", -1.0)
        cash_after_short = portfolio.cash

        # Next bar: borrow cost for 1 intraday bar
        portfolio.update_prices({"BTC": 50_000}, bar_index=1)

        bar_rate = 0.10 / 252 / 390
        expected_cost = 50_000 * bar_rate
        assert portfolio.cash == pytest.approx(cash_after_short - expected_cost, abs=0.001)

    def test_no_borrow_cost_for_long_positions(self):
        """Test that borrow costs are NOT deducted for long positions."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0, borrow_rate=0.10)

        portfolio.update_prices({"BTC": 50_000}, bar_index=0)
        portfolio.order("BTC", 1.0)
        cash_after_buy = portfolio.cash  # 50_000

        portfolio.update_prices({"BTC": 50_000}, bar_index=1)
        assert portfolio.cash == pytest.approx(cash_after_buy, abs=0.01)

    def test_no_borrow_cost_when_rate_is_zero(self):
        """Test that no borrow cost is deducted when rate is 0."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0, borrow_rate=0.0)

        portfolio.update_prices({"BTC": 50_000}, bar_index=0)
        portfolio.order("BTC", -1.0)
        cash_after_short = portfolio.cash

        portfolio.update_prices({"BTC": 50_000}, bar_index=1)
        assert portfolio.cash == pytest.approx(cash_after_short, abs=0.01)

    def test_borrow_cost_scales_with_position_size(self):
        """Test that borrow cost scales with the size of the short position."""
        portfolio = Portfolio(initial_cash=200_000, slippage=0.0, commission=0.0, borrow_rate=0.10)

        portfolio.update_prices({"BTC": 50_000}, bar_index=0)
        portfolio.order("BTC", -2.0)  # Short 2 BTC
        cash_after_short = portfolio.cash

        portfolio.update_prices({"BTC": 50_000}, bar_index=1)

        daily_rate = 0.10 / 252
        expected_cost = 2.0 * 50_000 * daily_rate  # 2x the single position cost
        assert portfolio.cash == pytest.approx(cash_after_short - expected_cost, abs=0.01)

    def test_borrow_cost_uses_current_price(self):
        """Test that borrow cost is based on current market price, not entry price."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0, borrow_rate=0.10)

        portfolio.update_prices({"BTC": 50_000}, bar_index=0)
        portfolio.order("BTC", -1.0)
        cash_after_short = portfolio.cash

        # Price rises — borrow cost should be higher
        portfolio.update_prices({"BTC": 60_000}, bar_index=1)

        daily_rate = 0.10 / 252
        expected_cost = 60_000 * daily_rate  # Based on current price, not entry
        assert portfolio.cash == pytest.approx(cash_after_short - expected_cost, abs=0.01)


class TestBracketPendingFill:
    """Test automatic SL/TP placement when pending bracket entry fills."""

    def test_bracket_limit_entry_sets_stops_on_fill(self):
        """Test that SL/TP are set when a pending bracket limit entry fills."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0)

        # Set price at 50000
        portfolio.update_prices(
            {"BTC": 50_000},
            bar_index=0,
            ohlc_data={"BTC": {"open": 50_000, "high": 50_000, "low": 50_000, "close": 50_000}},
        )

        # Place bracket with limit entry at 49000 (below current)
        result = portfolio.order_bracket("BTC", 0.1, stop_loss=47_000, take_profit=55_000, limit_price=49_000)
        assert result["entry"] is not None
        assert result["stop_loss"] is None  # Not set yet (entry pending)
        assert result["take_profit"] is None

        assert portfolio.get_position("BTC") == 0.0
        assert portfolio.get_stop_loss("BTC") is None
        assert portfolio.get_take_profit("BTC") is None

        # Price drops to fill the limit entry
        portfolio.update_prices(
            {"BTC": 48_500},
            bar_index=1,
            ohlc_data={"BTC": {"open": 49_500, "high": 50_000, "low": 48_000, "close": 48_500}},
        )

        # Entry should have filled and SL/TP should be set
        assert portfolio.get_position("BTC") == pytest.approx(0.1)
        assert portfolio.get_stop_loss("BTC") == 47_000
        assert portfolio.get_take_profit("BTC") == 55_000

    def test_bracket_delayed_entry_sets_stops_on_fill(self):
        """Test bracket with order_delay sets stops when entry fills."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0, order_delay=1)

        portfolio.update_prices(
            {"BTC": 50_000},
            bar_index=0,
            ohlc_data={"BTC": {"open": 50_000, "high": 50_000, "low": 50_000, "close": 50_000}},
        )

        # Place bracket (market entry, but delayed)
        result = portfolio.order_bracket("BTC", 0.1, stop_loss=48_000, take_profit=55_000)
        assert result["entry"] is not None
        assert result["stop_loss"] is None  # Not set yet (entry pending due to delay)

        assert portfolio.get_position("BTC") == 0.0

        # Next bar: entry fills
        portfolio.update_prices(
            {"BTC": 50_500},
            bar_index=1,
            ohlc_data={"BTC": {"open": 50_200, "high": 50_800, "low": 50_000, "close": 50_500}},
        )

        assert portfolio.get_position("BTC") == pytest.approx(0.1)
        assert portfolio.get_stop_loss("BTC") == 48_000
        assert portfolio.get_take_profit("BTC") == 55_000

    def test_bracket_pending_with_percentage_stops(self):
        """Test bracket with percentage stops stores calculated prices."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0)

        portfolio.update_prices(
            {"BTC": 50_000},
            bar_index=0,
            ohlc_data={"BTC": {"open": 50_000, "high": 50_000, "low": 50_000, "close": 50_000}},
        )

        # Place bracket with limit entry and percentage stops
        result = portfolio.order_bracket("BTC", 0.1, stop_loss_pct=0.05, take_profit_pct=0.10, limit_price=49_000)
        assert result["entry"] is not None

        # Check that the entry order has bracket metadata with calculated prices
        entry_order = portfolio.get_order(result["entry"])
        assert entry_order is not None
        # SL = 49000 * (1 - 0.05) = 46550 (based on limit_price estimate)
        assert entry_order.bracket_stop_loss is not None
        assert entry_order.bracket_take_profit is not None
