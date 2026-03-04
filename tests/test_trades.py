"""Tests for trade tracking system."""

import polars as pl

from polarbt.trades import Trade, TradeTracker


class TestTrade:
    """Test Trade dataclass."""

    def test_long_trade_creation(self):
        """Test creating a long trade."""
        trade = Trade(
            trade_id="1",
            asset="BTC",
            direction="long",
            entry_bar=10,
            entry_timestamp=1000,
            entry_price=50000.0,
            entry_size=1.0,
            entry_value=50000.0,
            entry_commission=50.0,
            exit_bar=20,
            exit_timestamp=2000,
            exit_price=52000.0,
            exit_size=1.0,
            exit_value=52000.0,
            exit_commission=52.0,
        )

        assert trade.asset == "BTC"
        assert trade.direction == "long"
        assert trade.pnl == 52000.0 - 50000.0 - 50.0 - 52.0
        assert trade.pnl == 1898.0
        assert trade.bars_held == 10

    def test_short_trade_creation(self):
        """Test creating a short trade."""
        trade = Trade(
            trade_id="2",
            asset="ETH",
            direction="short",
            entry_bar=10,
            entry_timestamp=1000,
            entry_price=2000.0,
            entry_size=10.0,
            entry_value=20000.0,
            entry_commission=20.0,
            exit_bar=15,
            exit_timestamp=1500,
            exit_price=1900.0,
            exit_size=10.0,
            exit_value=19000.0,
            exit_commission=19.0,
        )

        assert trade.direction == "short"
        # For short: pnl = entry_value - exit_value - commissions
        assert trade.pnl == 20000.0 - 19000.0 - 20.0 - 19.0
        assert trade.pnl == 961.0

    def test_pnl_percentage_calculation(self):
        """Test P&L percentage calculation."""
        trade = Trade(
            trade_id="1",
            asset="BTC",
            direction="long",
            entry_bar=10,
            entry_timestamp=1000,
            entry_price=50000.0,
            entry_size=1.0,
            entry_value=50000.0,
            entry_commission=50.0,
            exit_bar=20,
            exit_timestamp=2000,
            exit_price=55000.0,
            exit_size=1.0,
            exit_value=55000.0,
            exit_commission=55.0,
        )

        # pnl = 55000 - 50000 - 50 - 55 = 4895
        # pnl_pct = 4895 / 50000 = 0.0979
        assert abs(trade.pnl_pct - 0.0979) < 0.0001

    def test_is_winner(self):
        """Test is_winner method."""
        winning_trade = Trade(
            trade_id="1",
            asset="BTC",
            direction="long",
            entry_bar=10,
            entry_timestamp=1000,
            entry_price=50000.0,
            entry_size=1.0,
            entry_value=50000.0,
            exit_bar=20,
            exit_timestamp=2000,
            exit_price=52000.0,
            exit_size=1.0,
            exit_value=52000.0,
        )

        losing_trade = Trade(
            trade_id="2",
            asset="BTC",
            direction="long",
            entry_bar=10,
            entry_timestamp=1000,
            entry_price=50000.0,
            entry_size=1.0,
            entry_value=50000.0,
            exit_bar=20,
            exit_timestamp=2000,
            exit_price=48000.0,
            exit_size=1.0,
            exit_value=48000.0,
        )

        assert winning_trade.is_winner()
        assert not losing_trade.is_winner()

    def test_is_loser(self):
        """Test is_loser method."""
        winning_trade = Trade(
            trade_id="1",
            asset="BTC",
            direction="long",
            entry_bar=10,
            entry_timestamp=1000,
            entry_price=50000.0,
            entry_size=1.0,
            entry_value=50000.0,
            exit_bar=20,
            exit_timestamp=2000,
            exit_price=52000.0,
            exit_size=1.0,
            exit_value=52000.0,
        )

        losing_trade = Trade(
            trade_id="2",
            asset="BTC",
            direction="long",
            entry_bar=10,
            entry_timestamp=1000,
            entry_price=50000.0,
            entry_size=1.0,
            entry_value=50000.0,
            exit_bar=20,
            exit_timestamp=2000,
            exit_price=48000.0,
            exit_size=1.0,
            exit_value=48000.0,
        )

        assert not winning_trade.is_loser()
        assert losing_trade.is_loser()

    def test_is_long(self):
        """Test is_long method."""
        long_trade = Trade(
            trade_id="1",
            asset="BTC",
            direction="long",
            entry_bar=10,
            entry_timestamp=1000,
            entry_price=50000.0,
            entry_size=1.0,
            entry_value=50000.0,
            exit_bar=20,
            exit_timestamp=2000,
            exit_price=52000.0,
            exit_size=1.0,
            exit_value=52000.0,
        )

        short_trade = Trade(
            trade_id="2",
            asset="BTC",
            direction="short",
            entry_bar=10,
            entry_timestamp=1000,
            entry_price=50000.0,
            entry_size=1.0,
            entry_value=50000.0,
            exit_bar=20,
            exit_timestamp=2000,
            exit_price=48000.0,
            exit_size=1.0,
            exit_value=48000.0,
        )

        assert long_trade.is_long()
        assert not short_trade.is_long()

    def test_is_short(self):
        """Test is_short method."""
        long_trade = Trade(
            trade_id="1",
            asset="BTC",
            direction="long",
            entry_bar=10,
            entry_timestamp=1000,
            entry_price=50000.0,
            entry_size=1.0,
            entry_value=50000.0,
            exit_bar=20,
            exit_timestamp=2000,
            exit_price=52000.0,
            exit_size=1.0,
            exit_value=52000.0,
        )

        short_trade = Trade(
            trade_id="2",
            asset="BTC",
            direction="short",
            entry_bar=10,
            entry_timestamp=1000,
            entry_price=50000.0,
            entry_size=1.0,
            entry_value=50000.0,
            exit_bar=20,
            exit_timestamp=2000,
            exit_price=48000.0,
            exit_size=1.0,
            exit_value=48000.0,
        )

        assert not long_trade.is_short()
        assert short_trade.is_short()


class TestTradeTracker:
    """Test TradeTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = TradeTracker()
        assert len(tracker.trades) == 0
        assert len(tracker.open_positions) == 0

    def test_open_and_close_position(self):
        """Test opening and closing a position."""
        tracker = TradeTracker()

        # Open position
        tracker.on_position_opened(asset="BTC", size=1.0, price=50000.0, bar=10, timestamp=1000, commission=50.0)

        assert "BTC" in tracker.open_positions
        assert len(tracker.trades) == 0

        # Close position
        trade = tracker.on_position_closed(
            asset="BTC", size_closed=1.0, price=52000.0, bar=20, timestamp=2000, commission=52.0
        )

        assert trade is not None
        assert len(tracker.trades) == 1
        assert "BTC" not in tracker.open_positions
        assert trade.pnl == 52000.0 - 50000.0 - 50.0 - 52.0

    def test_partial_close(self):
        """Test partial position close."""
        tracker = TradeTracker()

        # Open position
        tracker.on_position_opened(asset="BTC", size=2.0, price=50000.0, bar=10, timestamp=1000, commission=100.0)

        # Partial close
        trade = tracker.on_position_closed(
            asset="BTC", size_closed=1.0, price=52000.0, bar=15, timestamp=1500, commission=52.0
        )

        assert trade is not None
        assert len(tracker.trades) == 1
        assert "BTC" in tracker.open_positions  # Still have 1.0 BTC
        assert tracker.open_positions["BTC"]["entry_size"] == 1.0

    def test_multiple_trades(self):
        """Test multiple sequential trades."""
        tracker = TradeTracker()

        # First trade
        tracker.on_position_opened("BTC", 1.0, 50000.0, 10, 1000, 50.0)
        tracker.on_position_closed("BTC", 1.0, 52000.0, 20, 2000, 52.0)

        # Second trade
        tracker.on_position_opened("BTC", 1.0, 51000.0, 30, 3000, 51.0)
        tracker.on_position_closed("BTC", 1.0, 53000.0, 40, 4000, 53.0)

        assert len(tracker.trades) == 2
        assert tracker.trades[0].entry_price == 50000.0
        assert tracker.trades[1].entry_price == 51000.0

    def test_get_trades_df_empty(self):
        """Test getting trades DataFrame when no trades."""
        tracker = TradeTracker()
        df = tracker.get_trades_df()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        assert "trade_id" in df.columns
        assert "pnl" in df.columns

    def test_get_trades_df_with_trades(self):
        """Test getting trades DataFrame with trades."""
        tracker = TradeTracker()

        tracker.on_position_opened("BTC", 1.0, 50000.0, 10, 1000, 50.0)
        tracker.on_position_closed("BTC", 1.0, 52000.0, 20, 2000, 52.0)

        df = tracker.get_trades_df()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 1
        assert df["asset"][0] == "BTC"
        assert df["entry_price"][0] == 50000.0
        assert df["exit_price"][0] == 52000.0

    def test_get_trade_stats_empty(self):
        """Test trade stats with no trades."""
        tracker = TradeTracker()
        stats = tracker.get_trade_stats()

        assert stats.total_trades == 0
        assert stats.win_rate == 0.0

    def test_get_trade_stats(self):
        """Test trade stats calculation."""
        tracker = TradeTracker()

        # Winning trade
        tracker.on_position_opened("BTC", 1.0, 50000.0, 10, 1000, 50.0)
        tracker.on_position_closed("BTC", 1.0, 52000.0, 20, 2000, 52.0)

        # Losing trade
        tracker.on_position_opened("ETH", 10.0, 2000.0, 30, 3000, 20.0)
        tracker.on_position_closed("ETH", 10.0, 1900.0, 40, 4000, 19.0)

        stats = tracker.get_trade_stats()

        assert stats.total_trades == 2
        assert stats.winning_trades == 1
        assert stats.losing_trades == 1
        assert stats.win_rate == 0.5
        assert stats.avg_win > 0
        assert stats.avg_loss > 0
        assert stats.profit_factor > 0

    def test_position_reversal(self):
        """Test position reversal (long to short)."""
        tracker = TradeTracker()

        # Open long position
        tracker.on_position_opened("BTC", 1.0, 50000.0, 10, 1000, 50.0)

        # Reverse to short
        trade = tracker.on_position_reversed(
            asset="BTC", old_size=1.0, new_size=-1.0, price=52000.0, bar=20, timestamp=2000, commission=52.0
        )

        assert trade is not None
        assert len(tracker.trades) == 1
        assert "BTC" in tracker.open_positions
        assert tracker.open_positions["BTC"]["direction"] == "short"
