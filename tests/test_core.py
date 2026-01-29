"""Tests for core backtesting components."""

import polars as pl
import pytest
from polarbtest.core import Portfolio, Strategy, Engine, BacktestContext
from polarbtest import indicators as ind


@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    return pl.DataFrame(
        {
            "timestamp": range(100),
            "close": [100 + i * 0.5 for i in range(100)],
            "high": [101 + i * 0.5 for i in range(100)],
            "low": [99 + i * 0.5 for i in range(100)],
            "volume": [1000] * 100,
        }
    )


class TestPortfolio:
    """Test Portfolio class."""

    def test_initialization(self):
        """Test portfolio initialization."""
        portfolio = Portfolio(initial_cash=100_000)
        assert portfolio.cash == 100_000
        assert portfolio.get_value() == 100_000
        assert len(portfolio.positions) == 0

    def test_order_buy(self):
        """Test buying assets."""
        portfolio = Portfolio(initial_cash=100_000, commission=0.001)
        portfolio.update_prices({"BTC": 50_000})

        # Buy 0.5 BTC
        success = portfolio.order("BTC", 0.5)
        assert success
        assert portfolio.get_position("BTC") == 0.5

        # Check cash decreased (price + commission + slippage)
        assert portfolio.cash < 100_000
        assert (
            portfolio.cash > 100_000 - 50_000
        )  # Less than full amount due to fractional buy

    def test_order_sell(self):
        """Test selling assets."""
        portfolio = Portfolio(initial_cash=100_000)
        portfolio.update_prices({"BTC": 50_000})

        # Buy then sell
        portfolio.order("BTC", 1.0)
        initial_cash = portfolio.cash
        portfolio.order("BTC", -0.5)  # Sell half

        assert portfolio.get_position("BTC") == 0.5
        assert portfolio.cash > initial_cash  # Cash increased from sale

    def test_order_target(self):
        """Test order_target."""
        portfolio = Portfolio(initial_cash=100_000)
        portfolio.update_prices({"BTC": 50_000})

        # Target 1 BTC
        portfolio.order_target("BTC", 1.0)
        assert portfolio.get_position("BTC") == 1.0

        # Target 0.5 BTC (should sell 0.5)
        portfolio.order_target("BTC", 0.5)
        assert portfolio.get_position("BTC") == 0.5

    def test_order_target_percent(self):
        """Test order_target_percent."""
        portfolio = Portfolio(initial_cash=100_000)
        portfolio.update_prices({"BTC": 50_000})

        # Allocate 50% to BTC
        portfolio.order_target_percent("BTC", 0.5)

        # Position value should be approximately 50% of portfolio
        position_value = portfolio.get_position("BTC") * 50_000
        total_value = portfolio.get_value()

        # Allow some tolerance for commission/slippage
        assert 0.45 < position_value / total_value < 0.55

    def test_close_position(self):
        """Test closing a position."""
        portfolio = Portfolio(initial_cash=100_000)
        portfolio.update_prices({"BTC": 50_000})

        portfolio.order("BTC", 1.0)
        portfolio.close_position("BTC")

        assert portfolio.get_position("BTC") == 0.0
        assert "BTC" not in portfolio.positions

    def test_insufficient_cash(self):
        """Test that orders fail with insufficient cash."""
        portfolio = Portfolio(initial_cash=1_000)
        portfolio.update_prices({"BTC": 50_000})

        # Try to buy 1 BTC (costs 50k but we only have 1k)
        success = portfolio.order("BTC", 1.0)
        assert not success
        assert portfolio.get_position("BTC") == 0.0

    def test_insufficient_shares(self):
        """Test that sells fail with insufficient shares."""
        portfolio = Portfolio(initial_cash=100_000)
        portfolio.update_prices({"BTC": 50_000})

        # Try to sell BTC we don't have
        success = portfolio.order("BTC", -1.0)
        assert not success


class SimpleStrategy(Strategy):
    """Simple test strategy."""

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            [ind.sma("close", self.params.get("sma_period", 10)).alias("sma")]
        )

    def next(self, ctx: BacktestContext) -> None:
        if ctx.row.get("sma") is None:
            return

        # Simple strategy: buy when above SMA, sell when below
        if ctx.row["close"] > ctx.row["sma"]:
            ctx.portfolio.order_target_percent("asset", 1.0)
        else:
            ctx.portfolio.close_position("asset")


class TestEngine:
    """Test Engine class."""

    def test_engine_initialization(self, sample_data):
        """Test engine initialization."""
        strategy = SimpleStrategy(sma_period=10)
        engine = Engine(strategy, sample_data)

        assert engine.strategy is not None
        assert engine.portfolio is None  # Not initialized until run()

    def test_engine_run(self, sample_data):
        """Test running a backtest."""
        strategy = SimpleStrategy(sma_period=10)
        engine = Engine(strategy, sample_data, initial_cash=100_000)

        results = engine.run()

        assert results is not None
        assert "final_equity" in results
        assert "sharpe_ratio" in results
        assert "total_return" in results
        assert engine.portfolio is not None

    def test_engine_with_uptrend(self):
        """Test engine with upward trending data."""
        # Create strongly upward trending data
        data = pl.DataFrame(
            {
                "timestamp": range(50),
                "close": [100 + i * 2 for i in range(50)],
            }
        )

        strategy = SimpleStrategy(sma_period=5)
        engine = Engine(strategy, data, initial_cash=100_000)
        results = engine.run()

        # Should make profit in uptrend
        assert results["total_return"] > 0

    def test_multiple_assets(self):
        """Test engine with multiple assets."""
        data = pl.DataFrame(
            {
                "timestamp": range(50),
                "btc_close": [50000 + i * 100 for i in range(50)],
                "eth_close": [3000 + i * 10 for i in range(50)],
            }
        )

        class MultiAssetStrategy(Strategy):
            def preprocess(self, df):
                return df

            def next(self, ctx):
                # Simple allocation
                ctx.portfolio.order_target_percent("BTC", 0.5)
                ctx.portfolio.order_target_percent("ETH", 0.5)

        strategy = MultiAssetStrategy()
        engine = Engine(
            strategy, data, price_columns={"BTC": "btc_close", "ETH": "eth_close"}
        )
        results = engine.run()

        assert results is not None
        assert (
            "BTC" in results["final_positions"] or "ETH" in results["final_positions"]
        )


class TestBacktestContext:
    """Test BacktestContext."""

    def test_context_creation(self):
        """Test creating a context object."""
        portfolio = Portfolio()
        portfolio.update_prices({"BTC": 50000})

        ctx = BacktestContext(
            timestamp=0,
            row={"close": 100, "sma": 95},
            portfolio=portfolio,
            bar_index=10,
        )

        assert ctx.timestamp == 0
        assert ctx.row["close"] == 100
        assert ctx.bar_index == 10
        assert ctx.portfolio is portfolio
