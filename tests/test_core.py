"""Tests for core backtesting components."""

import polars as pl
import pytest

from polarbt import indicators as ind
from polarbt.core import BacktestContext, Engine, Portfolio, Strategy, standardize_dataframe


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
        assert portfolio.cash > 100_000 - 50_000  # Less than full amount due to fractional buy

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
        order_id = portfolio.order("BTC", 1.0)
        # Order is created but should be rejected on execution
        assert order_id is not None
        order = portfolio.get_order(order_id)
        assert order is not None
        # With order_delay=0, order is executed immediately and rejected
        from polarbt.orders import OrderStatus

        assert order.status == OrderStatus.REJECTED
        assert portfolio.get_position("BTC") == 0.0

    def test_short_selling(self):
        """Test that selling without holding opens a short position."""
        portfolio = Portfolio(initial_cash=100_000)
        portfolio.update_prices({"BTC": 50_000})

        # Sell BTC we don't have — opens a short position
        order_id = portfolio.order("BTC", -1.0)
        assert order_id is not None
        order = portfolio.get_order(order_id)
        assert order is not None
        from polarbt.orders import OrderStatus

        assert order.status == OrderStatus.FILLED
        assert portfolio.get_position("BTC") == -1.0
        # Cash increases by sale proceeds minus commission
        assert portfolio.cash > 100_000

    def test_fixed_commission(self):
        """Test fixed commission only."""
        # $5 fixed commission per trade, no percentage
        portfolio = Portfolio(initial_cash=100_000, commission=(5.0, 0.0), slippage=0.0)
        portfolio.update_prices({"BTC": 50_000})

        # Buy 1 BTC
        success = portfolio.order("BTC", 1.0)
        assert success
        assert portfolio.get_position("BTC") == 1.0

        # Cash should be: 100_000 - 50_000 (price) - 5 (fixed commission) = 49_995
        assert portfolio.cash == pytest.approx(49_995, abs=0.01)

    def test_mixed_commission(self):
        """Test mixed fixed + percentage commission."""
        # $5 fixed + 0.1% per trade
        portfolio = Portfolio(initial_cash=100_000, commission=(5.0, 0.001), slippage=0.0)
        portfolio.update_prices({"BTC": 50_000})

        # Buy 1 BTC
        success = portfolio.order("BTC", 1.0)
        assert success
        assert portfolio.get_position("BTC") == 1.0

        # Total cost = 50_000 (price) + 5 (fixed) + 50 (0.1% of 50_000) = 50_055
        # Cash should be: 100_000 - 50_055 = 49_945
        assert portfolio.cash == pytest.approx(49_945, abs=0.01)

    def test_percentage_commission_backward_compatible(self):
        """Test that percentage-only commission still works (backward compatibility)."""
        # 0.1% commission (old style, should work the same)
        portfolio = Portfolio(initial_cash=100_000, commission=0.001, slippage=0.0)
        portfolio.update_prices({"BTC": 50_000})

        # Buy 1 BTC
        success = portfolio.order("BTC", 1.0)
        assert success
        assert portfolio.get_position("BTC") == 1.0

        # Total cost = 50_000 + 50 (0.1% commission) = 50_050
        # Cash should be: 100_000 - 50_050 = 49_950
        assert portfolio.cash == pytest.approx(49_950, abs=0.01)

    def test_fixed_commission_sell(self):
        """Test fixed commission on sell orders."""
        portfolio = Portfolio(initial_cash=100_000, commission=(5.0, 0.0), slippage=0.0)
        portfolio.update_prices({"BTC": 50_000})

        # Buy 1 BTC (costs 50_000 + 5 = 50_005)
        portfolio.order("BTC", 1.0)
        assert portfolio.cash == pytest.approx(49_995, abs=0.01)

        # Sell 1 BTC (receives 50_000 - 5 = 49_995)
        portfolio.order("BTC", -1.0)
        assert portfolio.cash == pytest.approx(99_990, abs=0.01)
        assert portfolio.get_position("BTC") == 0.0

    def test_order_target_percent_with_fixed_commission(self):
        """Test order_target_percent with fixed commission."""
        portfolio = Portfolio(initial_cash=100_000, commission=(5.0, 0.0), slippage=0.0)
        portfolio.update_prices({"BTC": 50_000})

        # Allocate 50% to BTC
        portfolio.order_target_percent("BTC", 0.5)

        # Position value should be approximately 50% of portfolio (accounting for fixed fee)
        position_value = portfolio.get_position("BTC") * 50_000
        total_value = portfolio.get_value()

        # With fixed commission, we should be close to 50%
        # Target is ~50k, but we pay $5 commission, so actual allocation is slightly less
        assert 0.48 < position_value / total_value < 0.52

    def test_order_target_percent_with_mixed_commission(self):
        """Test order_target_percent with mixed commission."""
        portfolio = Portfolio(initial_cash=100_000, commission=(5.0, 0.001), slippage=0.0)
        portfolio.update_prices({"BTC": 50_000})

        # Allocate 50% to BTC
        portfolio.order_target_percent("BTC", 0.5)

        # Position value should be approximately 50% of portfolio
        position_value = portfolio.get_position("BTC") * 50_000
        total_value = portfolio.get_value()

        # Allow tolerance for commission (both fixed and percentage)
        assert 0.47 < position_value / total_value < 0.52


class SimpleStrategy(Strategy):
    """Simple test strategy."""

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([ind.sma("close", self.params.get("sma_period", 10)).alias("sma")])

    def next(self, ctx: BacktestContext) -> None:
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
                "close": [100.0 + i * 2.0 for i in range(50)],  # Use floats
            }
        )

        strategy = SimpleStrategy(sma_period=5)
        # Auto warmup is default, which will skip first 4 bars (5-period SMA)
        engine = Engine(strategy, data, initial_cash=100_000)
        results = engine.run()

        # Should make profit in uptrend (enough bars left to trade)
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
        engine = Engine(strategy, data, price_columns={"BTC": "btc_close", "ETH": "eth_close"})
        results = engine.run()

        assert results is not None
        assert "BTC" in results["final_positions"] or "ETH" in results["final_positions"]


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


class TestStandardizeDataframe:
    """Test standardize_dataframe OHLCV column name normalization."""

    def test_capitalized_ohlcv_renamed(self):
        df = pl.DataFrame({"Date": [1], "Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [100]})
        result = standardize_dataframe(df)
        assert result.columns == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_already_lowercase_unchanged(self):
        df = pl.DataFrame(
            {"timestamp": [1], "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100]}
        )
        result = standardize_dataframe(df)
        assert result.columns == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_mixed_case_columns(self):
        df = pl.DataFrame({"Date": [1], "open": [1.0], "High": [2.0], "low": [0.5], "Close": [1.5], "volume": [100]})
        result = standardize_dataframe(df)
        assert result.columns == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_adj_close_alias(self):
        df = pl.DataFrame({"timestamp": [1], "Adj_Close": [1.5]})
        result = standardize_dataframe(df)
        assert "close" in result.columns

    def test_auto_detect_disabled(self):
        df = pl.DataFrame({"Open": [1.0], "High": [2.0]})
        result = standardize_dataframe(df, auto_detect=False)
        assert result.columns == ["Open", "High"]

    def test_engine_standardizes_ohlcv(self):
        """Engine should auto-standardize capitalized OHLCV columns."""

        class Noop(Strategy):
            def preprocess(self, df):
                return df

            def next(self, ctx):
                pass

        df = pl.DataFrame(
            {
                "Date": range(10),
                "Open": [1.0] * 10,
                "High": [2.0] * 10,
                "Low": [0.5] * 10,
                "Close": [1.5] * 10,
                "Volume": [100] * 10,
            }
        )
        engine = Engine(Noop(), df)
        assert "open" in engine.data.columns
        assert "close" in engine.data.columns
        assert "timestamp" in engine.data.columns
