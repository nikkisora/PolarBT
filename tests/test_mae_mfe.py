"""Tests for MAE/MFE and enhanced trade metrics (BMFE, trade_mdd, pdays)."""

import polars as pl

from polarbt.core import Portfolio


class TestMAEMFE:
    """Test MAE/MFE tracking functionality."""

    def test_mae_tracks_worst_drawdown(self):
        """Test that MAE tracks the worst unrealized loss during a trade."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter long position at 50000
        portfolio.update_prices({"BTC": 50000}, bar_index=0)
        portfolio.order("BTC", 0.1)

        # Price moves favorably, then adversely, then closes in profit
        portfolio.update_prices({"BTC": 52000}, bar_index=1)  # Up 2000 (MFE)
        portfolio.update_prices({"BTC": 48000}, bar_index=2)  # Down 2000 (MAE)
        portfolio.update_prices({"BTC": 49000}, bar_index=3)  # Slight recovery

        # Close position
        portfolio.close_position("BTC")

        trades = portfolio.get_trades()
        assert len(trades) == 1

        trade = trades.row(0, named=True)
        assert trade["entry_price"] == 50000
        assert trade["exit_price"] == 49000

        # MAE should be -2000/50000 = -0.04 (worst drawdown as pct)
        assert trade["mae"] is not None
        assert abs(trade["mae"] - (-0.04)) < 0.001

        # MFE should be 2000/50000 = 0.04 (best profit as pct)
        assert trade["mfe"] is not None
        assert abs(trade["mfe"] - 0.04) < 0.001

    def test_mfe_tracks_best_profit(self):
        """Test that MFE tracks the best unrealized profit during a trade."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter long position at 50000
        portfolio.update_prices({"BTC": 50000}, bar_index=0)
        portfolio.order("BTC", 0.1)

        # Price moves up to peak, then back down
        portfolio.update_prices({"BTC": 55000}, bar_index=1)  # Peak: +5000
        portfolio.update_prices({"BTC": 54000}, bar_index=2)  # Slight drop
        portfolio.update_prices({"BTC": 53000}, bar_index=3)  # More drop

        # Close at 53000
        portfolio.close_position("BTC")

        trades = portfolio.get_trades()
        trade = trades.row(0, named=True)

        # MFE should be 5000/50000 = 0.10
        assert trade["mfe"] is not None
        assert abs(trade["mfe"] - 0.10) < 0.001

    def test_mae_mfe_both_zero_on_immediate_exit(self):
        """Test MAE/MFE when position is closed immediately."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter and exit at same price
        portfolio.update_prices({"BTC": 50000}, bar_index=0)
        portfolio.order("BTC", 0.1)
        portfolio.close_position("BTC")

        trades = portfolio.get_trades()
        trade = trades.row(0, named=True)

        # Both should be 0 or very close (no price movement)
        assert trade["mae"] == 0.0
        assert trade["mfe"] == 0.0

    def test_mae_mfe_with_winning_trade(self):
        """Test MAE/MFE on a winning trade that never went negative."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter at 50000
        portfolio.update_prices({"BTC": 50000}, bar_index=0)
        portfolio.order("BTC", 0.1)

        # Price only goes up
        portfolio.update_prices({"BTC": 51000}, bar_index=1)
        portfolio.update_prices({"BTC": 52000}, bar_index=2)
        portfolio.update_prices({"BTC": 53000}, bar_index=3)

        portfolio.close_position("BTC")

        trades = portfolio.get_trades()
        trade = trades.row(0, named=True)

        # MAE should be 0 or positive (never went negative)
        assert trade["mae"] >= 0

        # MFE should be 3000/50000 = 0.06
        assert trade["mfe"] is not None
        assert abs(trade["mfe"] - 0.06) < 0.001

    def test_mae_mfe_with_losing_trade(self):
        """Test MAE/MFE on a losing trade that never went positive."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter at 50000
        portfolio.update_prices({"BTC": 50000}, bar_index=0)
        portfolio.order("BTC", 0.1)

        # Price only goes down
        portfolio.update_prices({"BTC": 49000}, bar_index=1)
        portfolio.update_prices({"BTC": 48000}, bar_index=2)
        portfolio.update_prices({"BTC": 47000}, bar_index=3)

        portfolio.close_position("BTC")

        trades = portfolio.get_trades()
        trade = trades.row(0, named=True)

        # MAE should be -3000/50000 = -0.06
        assert trade["mae"] is not None
        assert abs(trade["mae"] - (-0.06)) < 0.001

        # MFE should be 0 or negative (never went positive)
        assert trade["mfe"] <= 0

    def test_mae_mfe_multi_bar_tracking(self):
        """Test that MAE/MFE track correctly over many bars."""
        portfolio = Portfolio(initial_cash=10000)

        # Enter at 50000
        portfolio.update_prices({"BTC": 50000}, bar_index=0)
        portfolio.order("BTC", 0.1)

        # Simulate volatile price action
        prices = [51000, 49000, 52000, 48000, 53000, 47000, 54000, 50000]
        for i, price in enumerate(prices, start=1):
            portfolio.update_prices({"BTC": price}, bar_index=i)

        portfolio.close_position("BTC")

        trades = portfolio.get_trades()
        trade = trades.row(0, named=True)

        # MAE should capture lowest point: (47000-50000)/50000 = -0.06
        assert trade["mae"] is not None
        assert abs(trade["mae"] - (-0.06)) < 0.001

        # MFE should capture highest point: (54000-50000)/50000 = 0.08
        assert trade["mfe"] is not None
        assert abs(trade["mfe"] - 0.08) < 0.001

    def test_mae_mfe_exported_in_dataframe(self):
        """Test that MAE/MFE are included in trades DataFrame."""
        portfolio = Portfolio(initial_cash=10000)

        # Create a simple trade
        portfolio.update_prices({"BTC": 50000}, bar_index=0)
        portfolio.order("BTC", 0.1)
        portfolio.update_prices({"BTC": 55000}, bar_index=1)  # MFE
        portfolio.update_prices({"BTC": 48000}, bar_index=2)  # MAE
        portfolio.close_position("BTC")

        trades_df = portfolio.get_trades()

        # Check that MAE and MFE columns exist
        assert "mae" in trades_df.columns
        assert "mfe" in trades_df.columns

        # Check values
        assert trades_df["mae"][0] is not None
        assert trades_df["mfe"][0] is not None

    def test_mae_mfe_multiple_trades(self):
        """Test MAE/MFE tracking across multiple independent trades."""
        portfolio = Portfolio(initial_cash=20000)

        # Trade 1: BTC
        portfolio.update_prices({"BTC": 50000, "ETH": 3000}, bar_index=0)
        portfolio.order("BTC", 0.1)
        portfolio.update_prices({"BTC": 52000, "ETH": 3000}, bar_index=1)  # MFE: +2000
        portfolio.update_prices({"BTC": 49000, "ETH": 3000}, bar_index=2)  # MAE: -1000
        portfolio.close_position("BTC")

        # Trade 2: ETH
        portfolio.order("ETH", 1.0)
        portfolio.update_prices({"BTC": 49000, "ETH": 3500}, bar_index=3)  # MFE: +500
        portfolio.update_prices({"BTC": 49000, "ETH": 2800}, bar_index=4)  # MAE: -200
        portfolio.close_position("ETH")

        trades_df = portfolio.get_trades()
        assert len(trades_df) == 2

        # Trade 1 (BTC): MAE = -1000/50000 = -0.02, MFE = 2000/50000 = 0.04
        btc_trade = trades_df.filter(pl.col("asset") == "BTC").row(0, named=True)
        assert abs(btc_trade["mae"] - (-0.02)) < 0.001
        assert abs(btc_trade["mfe"] - 0.04) < 0.001

        # Trade 2 (ETH): MAE = -200/3000 = -0.0667, MFE = 500/3000 = 0.1667
        eth_trade = trades_df.filter(pl.col("asset") == "ETH").row(0, named=True)
        assert abs(eth_trade["mae"] - (-200 / 3000)) < 0.001
        assert abs(eth_trade["mfe"] - (500 / 3000)) < 0.001


class TestBMFE:
    """Test Before-MAE MFE tracking."""

    def test_bmfe_captures_mfe_before_crash(self):
        """Long trade going up then down: BMFE = MFE at the bar before the crash."""
        portfolio = Portfolio(initial_cash=100000)

        portfolio.update_prices({"BTC": 100}, bar_index=0)
        portfolio.order("BTC", 10)

        # Price goes up
        portfolio.update_prices({"BTC": 110}, bar_index=1)  # MFE = 0.10
        portfolio.update_prices({"BTC": 115}, bar_index=2)  # MFE = 0.15

        # Price crashes below entry — MAE worsens, BMFE snapshots MFE at that moment
        portfolio.update_prices({"BTC": 85}, bar_index=3)  # MAE = -0.15, BMFE = 0.15

        portfolio.close_position("BTC")

        trades = portfolio.get_trades()
        trade = trades.row(0, named=True)

        assert abs(trade["bmfe"] - 0.15) < 0.001
        assert abs(trade["mae"] - (-0.15)) < 0.001
        assert abs(trade["mfe"] - 0.15) < 0.001

    def test_bmfe_zero_when_never_negative(self):
        """Trade that never goes negative: BMFE = 0.0."""
        portfolio = Portfolio(initial_cash=100000)

        portfolio.update_prices({"BTC": 100}, bar_index=0)
        portfolio.order("BTC", 10)

        portfolio.update_prices({"BTC": 105}, bar_index=1)
        portfolio.update_prices({"BTC": 110}, bar_index=2)
        portfolio.update_prices({"BTC": 108}, bar_index=3)

        portfolio.close_position("BTC")

        trades = portfolio.get_trades()
        trade = trades.row(0, named=True)

        # MAE never went below 0, so BMFE stays at initial 0.0
        assert trade["bmfe"] == 0.0

    def test_bmfe_short_trade(self):
        """Short trade: verify direction-aware calculation."""
        portfolio = Portfolio(initial_cash=100000)

        portfolio.update_prices({"BTC": 100}, bar_index=0)
        portfolio.order("BTC", -10)  # Short

        # Price goes down (favorable for short)
        portfolio.update_prices({"BTC": 90}, bar_index=1)  # MFE = 0.10
        # Price goes up (adverse for short)
        portfolio.update_prices({"BTC": 115}, bar_index=2)  # MAE = -0.15, BMFE = 0.10

        portfolio.close_position("BTC")

        trades = portfolio.get_trades()
        trade = trades.row(0, named=True)

        assert abs(trade["bmfe"] - 0.10) < 0.001
        assert abs(trade["mae"] - (-0.15)) < 0.001


class TestTradeMDD:
    """Test per-trade max drawdown."""

    def test_trade_mdd_with_drawdown(self):
        """Trade with internal drawdown."""
        portfolio = Portfolio(initial_cash=100000)

        portfolio.update_prices({"BTC": 100}, bar_index=0)
        portfolio.order("BTC", 10)

        # Price goes up, then drops, then recovers
        portfolio.update_prices({"BTC": 120}, bar_index=1)  # peak_pnl = 0.20
        portfolio.update_prices({"BTC": 105}, bar_index=2)  # dd = 0.05 - 0.20 = -0.15
        portfolio.update_prices({"BTC": 130}, bar_index=3)  # new peak 0.30

        portfolio.close_position("BTC")

        trades = portfolio.get_trades()
        trade = trades.row(0, named=True)

        # trade_mdd = worst drawdown from peak = 0.05 - 0.20 = -0.15
        assert abs(trade["trade_mdd"] - (-0.15)) < 0.001

    def test_trade_mdd_always_rising(self):
        """Trade that never drops: trade_mdd = 0."""
        portfolio = Portfolio(initial_cash=100000)

        portfolio.update_prices({"BTC": 100}, bar_index=0)
        portfolio.order("BTC", 10)

        portfolio.update_prices({"BTC": 105}, bar_index=1)
        portfolio.update_prices({"BTC": 110}, bar_index=2)
        portfolio.update_prices({"BTC": 115}, bar_index=3)

        portfolio.close_position("BTC")

        trades = portfolio.get_trades()
        trade = trades.row(0, named=True)

        assert trade["trade_mdd"] == 0.0

    def test_trade_mdd_equals_mae_when_never_positive(self):
        """Trade that never goes positive: pdays = 0, trade_mdd = MAE."""
        portfolio = Portfolio(initial_cash=100000)

        portfolio.update_prices({"BTC": 100}, bar_index=0)
        portfolio.order("BTC", 10)

        portfolio.update_prices({"BTC": 95}, bar_index=1)
        portfolio.update_prices({"BTC": 90}, bar_index=2)
        portfolio.update_prices({"BTC": 85}, bar_index=3)

        portfolio.close_position("BTC")

        trades = portfolio.get_trades()
        trade = trades.row(0, named=True)

        # Peak PnL stays at 0.0 (initial), drawdown = current - 0 = current
        # trade_mdd should equal mae since peak never moved above 0
        assert abs(trade["trade_mdd"] - trade["mae"]) < 0.001
        assert trade["pdays"] == 0


class TestPdays:
    """Test profitable days count."""

    def test_pdays_all_profitable(self):
        """Trade where all bars are profitable: pdays = bars_held."""
        portfolio = Portfolio(initial_cash=100000)

        portfolio.update_prices({"BTC": 100}, bar_index=0)
        portfolio.order("BTC", 10)

        portfolio.update_prices({"BTC": 105}, bar_index=1)  # +5%
        portfolio.update_prices({"BTC": 110}, bar_index=2)  # +10%
        portfolio.update_prices({"BTC": 108}, bar_index=3)  # +8%

        portfolio.close_position("BTC")

        trades = portfolio.get_trades()
        trade = trades.row(0, named=True)

        # All 3 update bars had positive unrealized P&L
        assert trade["pdays"] == 3

    def test_pdays_none_profitable(self):
        """Trade where no bars are profitable: pdays = 0."""
        portfolio = Portfolio(initial_cash=100000)

        portfolio.update_prices({"BTC": 100}, bar_index=0)
        portfolio.order("BTC", 10)

        portfolio.update_prices({"BTC": 95}, bar_index=1)
        portfolio.update_prices({"BTC": 90}, bar_index=2)

        portfolio.close_position("BTC")

        trades = portfolio.get_trades()
        trade = trades.row(0, named=True)

        assert trade["pdays"] == 0

    def test_pdays_mixed(self):
        """Trade with mixed profitable/unprofitable bars."""
        portfolio = Portfolio(initial_cash=100000)

        portfolio.update_prices({"BTC": 100}, bar_index=0)
        portfolio.order("BTC", 10)

        portfolio.update_prices({"BTC": 105}, bar_index=1)  # profitable
        portfolio.update_prices({"BTC": 95}, bar_index=2)  # not profitable
        portfolio.update_prices({"BTC": 110}, bar_index=3)  # profitable
        portfolio.update_prices({"BTC": 99}, bar_index=4)  # not profitable

        portfolio.close_position("BTC")

        trades = portfolio.get_trades()
        trade = trades.row(0, named=True)

        assert trade["pdays"] == 2


class TestEnhancedMetricsInDataFrame:
    """Test that new columns appear in the trades DataFrame."""

    def test_new_columns_present(self):
        """Verify bmfe, trade_mdd, pdays columns exist."""
        portfolio = Portfolio(initial_cash=100000)

        portfolio.update_prices({"BTC": 100}, bar_index=0)
        portfolio.order("BTC", 10)
        portfolio.update_prices({"BTC": 110}, bar_index=1)
        portfolio.update_prices({"BTC": 90}, bar_index=2)
        portfolio.close_position("BTC")

        trades_df = portfolio.get_trades()
        assert "bmfe" in trades_df.columns
        assert "trade_mdd" in trades_df.columns
        assert "pdays" in trades_df.columns

    def test_empty_df_has_new_columns(self):
        """Verify empty trades DataFrame has new columns."""
        from polarbt.trades import TradeTracker

        tracker = TradeTracker()
        df = tracker.get_trades_df()
        assert "bmfe" in df.columns
        assert "trade_mdd" in df.columns
        assert "pdays" in df.columns

    def test_trade_stats_aggregates(self):
        """Verify TradeStats includes avg_bmfe, avg_trade_mdd, avg_pdays."""
        portfolio = Portfolio(initial_cash=100000)

        # Trade 1
        portfolio.update_prices({"BTC": 100}, bar_index=0)
        portfolio.order("BTC", 10)
        portfolio.update_prices({"BTC": 110}, bar_index=1)
        portfolio.update_prices({"BTC": 90}, bar_index=2)
        portfolio.close_position("BTC")

        # Trade 2
        portfolio.update_prices({"BTC": 100}, bar_index=3)
        portfolio.order("BTC", 10)
        portfolio.update_prices({"BTC": 105}, bar_index=4)
        portfolio.close_position("BTC")

        stats = portfolio.get_trade_stats()
        assert hasattr(stats, "avg_bmfe")
        assert hasattr(stats, "avg_trade_mdd")
        assert hasattr(stats, "avg_pdays")
