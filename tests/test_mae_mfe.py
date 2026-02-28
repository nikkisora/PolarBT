"""Tests for MAE (Maximum Adverse Excursion) and MFE (Maximum Favorable Excursion) tracking."""

import polars as pl

from polarbtest.core import Portfolio


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

        # MAE should be -2000 (worst drawdown was at 48000)
        assert trade["mae"] is not None
        assert abs(trade["mae"] - (-2000)) < 1

        # MFE should be 2000 (best profit was at 52000)
        assert trade["mfe"] is not None
        assert abs(trade["mfe"] - 2000) < 1

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

        # MFE should be 5000 (peak profit at 55000)
        assert trade["mfe"] is not None
        assert abs(trade["mfe"] - 5000) < 1

        # Final profit is 3000, but MFE captured the 5000 peak
        assert abs(trade["pnl"] - 300) < 50  # ~300 after commission

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

        # MFE should be 3000 (final price)
        assert trade["mfe"] is not None
        assert abs(trade["mfe"] - 3000) < 1

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

        # MAE should be -3000 (final price)
        assert trade["mae"] is not None
        assert abs(trade["mae"] - (-3000)) < 1

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

        # MAE should capture lowest point: 47000 - 50000 = -3000
        assert trade["mae"] is not None
        assert abs(trade["mae"] - (-3000)) < 1

        # MFE should capture highest point: 54000 - 50000 = 4000
        assert trade["mfe"] is not None
        assert abs(trade["mfe"] - 4000) < 1

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

        # Trade 1 (BTC)
        btc_trade = trades_df.filter(pl.col("asset") == "BTC").row(0, named=True)
        assert abs(btc_trade["mae"] - (-1000)) < 1
        assert abs(btc_trade["mfe"] - 2000) < 1

        # Trade 2 (ETH)
        eth_trade = trades_df.filter(pl.col("asset") == "ETH").row(0, named=True)
        assert abs(eth_trade["mae"] - (-200)) < 1
        assert abs(eth_trade["mfe"] - 500) < 1
