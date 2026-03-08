"""Tests for core backtesting components."""

import polars as pl
import pytest

from polarbt import indicators as ind
from polarbt.core import BacktestContext, Engine, Portfolio, Strategy, param, standardize_dataframe


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
        assert results.final_equity is not None
        assert results.sharpe_ratio is not None
        assert results.total_return is not None
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
        assert results.total_return > 0

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
        assert "BTC" in results.final_positions or "ETH" in results.final_positions


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


class TestParam:
    """Test param() descriptor for strategy parameters."""

    def test_default_value(self):
        """param() returns default when no kwarg is provided."""

        class S(Strategy):
            period = param(10)

            def preprocess(self, df):
                return df

            def next(self, ctx):
                pass

        s = S()
        assert s.period == 10

    def test_kwarg_overrides_default(self):
        """Keyword argument overrides the declared default."""

        class S(Strategy):
            period = param(10)

            def preprocess(self, df):
                return df

            def next(self, ctx):
                pass

        s = S(period=42)
        assert s.period == 42

    def test_set_updates_params(self):
        """Setting a param attribute writes to self.params."""

        class S(Strategy):
            period = param(10)

            def preprocess(self, df):
                return df

            def next(self, ctx):
                pass

        s = S()
        s.period = 99
        assert s.params["period"] == 99
        assert s.period == 99

    def test_params_dict_still_works(self):
        """self.params.get() pattern still works alongside param()."""

        class S(Strategy):
            fast = param(5)

            def preprocess(self, df):
                return df

            def next(self, ctx):
                pass

        s = S(fast=15)
        assert s.params.get("fast", 5) == 15
        assert s.fast == 15

    def test_class_access_returns_descriptor(self):
        """Accessing param on the class returns the Param descriptor."""

        class S(Strategy):
            period = param(10)

            def preprocess(self, df):
                return df

            def next(self, ctx):
                pass

        from polarbt.core import Param

        assert isinstance(S.period, Param)

    def test_multiple_params(self):
        """Multiple param() declarations work independently."""

        class S(Strategy):
            fast = param(5)
            slow = param(20)

            def preprocess(self, df):
                return df

            def next(self, ctx):
                pass

        s = S(fast=10)
        assert s.fast == 10
        assert s.slow == 20

    def test_param_in_backtest(self, sample_data):
        """Strategy using param() works end-to-end with backtest."""

        class ParamStrategy(Strategy):
            sma_period = param(10)

            def preprocess(self, df):
                return df.with_columns([ind.sma("close", self.sma_period).alias("sma")])

            def next(self, ctx):
                if ctx.row["close"] > ctx.row["sma"]:
                    ctx.portfolio.order_target_percent("asset", 1.0)
                else:
                    ctx.portfolio.close_position("asset")

        engine = Engine(ParamStrategy(sma_period=10), sample_data, initial_cash=100_000)
        results = engine.run()
        assert results is not None
        assert results.total_return is not None


class TestTouchedExitPriority:
    """Test OHLC priority-based stop checking (Feature 5)."""

    def test_gap_down_through_stop_loss_exits_at_open(self):
        """Gap down through stop-loss: open < stop -> exit at open, not stop price."""
        portfolio = Portfolio(initial_cash=100_000)

        # Enter long at 100
        portfolio.update_prices(
            {"BTC": 100},
            bar_index=0,
            ohlc_data={"BTC": {"open": 100, "high": 100, "low": 100, "close": 100}},
        )
        portfolio.order("BTC", 10)
        portfolio.set_stop_loss("BTC", stop_price=90)

        # Bar 1: gap down to 80, below stop of 90
        portfolio.update_prices(
            {"BTC": 82},
            bar_index=1,
            ohlc_data={"BTC": {"open": 80, "high": 85, "low": 78, "close": 82}},
        )

        # Position should be closed
        assert portfolio.get_position("BTC") == 0
        # Should have exited at open price (80), not stop price (90)
        trades = portfolio.get_trades()
        assert len(trades) == 1
        assert trades["exit_price"][0] == 80

    def test_gap_up_through_take_profit_exits_at_open(self):
        """Gap up through take-profit: open > TP -> exit at open, not TP price."""
        portfolio = Portfolio(initial_cash=100_000)

        # Enter long at 100
        portfolio.update_prices(
            {"BTC": 100},
            bar_index=0,
            ohlc_data={"BTC": {"open": 100, "high": 100, "low": 100, "close": 100}},
        )
        portfolio.order("BTC", 10)
        portfolio.set_take_profit("BTC", target_price=110)

        # Bar 1: gap up to 120, above TP of 110
        portfolio.update_prices(
            {"BTC": 118},
            bar_index=1,
            ohlc_data={"BTC": {"open": 120, "high": 125, "low": 115, "close": 118}},
        )

        # Position should be closed at open price (120)
        assert portfolio.get_position("BTC") == 0
        trades = portfolio.get_trades()
        assert len(trades) == 1
        assert trades["exit_price"][0] == 120

    def test_normal_stop_loss_trigger_exits_at_stop_price(self):
        """Normal SL trigger: low touches stop but open is above -> exit at stop price."""
        portfolio = Portfolio(initial_cash=100_000)

        # Enter long at 100
        portfolio.update_prices(
            {"BTC": 100},
            bar_index=0,
            ohlc_data={"BTC": {"open": 100, "high": 100, "low": 100, "close": 100}},
        )
        portfolio.order("BTC", 10)
        portfolio.set_stop_loss("BTC", stop_price=90)

        # Bar 1: open at 95 (above stop), low touches 88 (below stop)
        portfolio.update_prices(
            {"BTC": 92},
            bar_index=1,
            ohlc_data={"BTC": {"open": 95, "high": 96, "low": 88, "close": 92}},
        )

        # Position should be closed at stop price (90)
        assert portfolio.get_position("BTC") == 0
        trades = portfolio.get_trades()
        assert len(trades) == 1
        assert trades["exit_price"][0] == 90

    def test_normal_take_profit_trigger_exits_at_tp_price(self):
        """Normal TP trigger: high touches TP but open is below -> exit at TP price."""
        portfolio = Portfolio(initial_cash=100_000)

        # Enter long at 100
        portfolio.update_prices(
            {"BTC": 100},
            bar_index=0,
            ohlc_data={"BTC": {"open": 100, "high": 100, "low": 100, "close": 100}},
        )
        portfolio.order("BTC", 10)
        portfolio.set_take_profit("BTC", target_price=110)

        # Bar 1: open at 105 (below TP), high touches 112 (above TP)
        portfolio.update_prices(
            {"BTC": 108},
            bar_index=1,
            ohlc_data={"BTC": {"open": 105, "high": 112, "low": 104, "close": 108}},
        )

        # Position should be closed at TP price (110)
        assert portfolio.get_position("BTC") == 0
        trades = portfolio.get_trades()
        assert len(trades) == 1
        assert trades["exit_price"][0] == 110

    def test_both_sl_and_tp_touched_priority_determines_which(self):
        """Both SL and TP touched in same bar: TP fires first via high priority."""
        portfolio = Portfolio(initial_cash=100_000)

        # Enter long at 100
        portfolio.update_prices(
            {"BTC": 100},
            bar_index=0,
            ohlc_data={"BTC": {"open": 100, "high": 100, "low": 100, "close": 100}},
        )
        portfolio.order("BTC", 10)
        portfolio.set_stop_loss("BTC", stop_price=90)
        portfolio.set_take_profit("BTC", target_price=110)

        # Bar 1: both high > TP and low < SL, open between them
        # Priority: high (TP for long) checked before low (SL for long)
        portfolio.update_prices(
            {"BTC": 100},
            bar_index=1,
            ohlc_data={"BTC": {"open": 100, "high": 115, "low": 85, "close": 100}},
        )

        assert portfolio.get_position("BTC") == 0
        trades = portfolio.get_trades()
        assert len(trades) == 1
        # TP should fire since high is checked before low for long positions
        assert trades["exit_price"][0] == 110

    def test_no_ohlc_data_uses_existing_behavior(self):
        """No OHLC data: existing behavior unchanged (close-price checks)."""
        portfolio = Portfolio(initial_cash=100_000)

        # Enter long at 100
        portfolio.update_prices({"BTC": 100}, bar_index=0)
        portfolio.order("BTC", 10)
        portfolio.set_stop_loss("BTC", stop_price=90)

        # Update with just close price (no OHLC) — falls to 85
        portfolio.update_prices({"BTC": 85}, bar_index=1)

        # Should still trigger stop
        assert portfolio.get_position("BTC") == 0

    def test_trailing_stop_with_gap(self):
        """Trailing stop with gap: verify open-price exit."""
        portfolio = Portfolio(initial_cash=100_000)

        # Enter long at 100
        portfolio.update_prices(
            {"BTC": 100},
            bar_index=0,
            ohlc_data={"BTC": {"open": 100, "high": 100, "low": 100, "close": 100}},
        )
        portfolio.order("BTC", 10)
        portfolio.set_trailing_stop("BTC", trail_pct=0.05)

        # Price goes to 120 — trailing stop moves up to 120 * 0.95 = 114
        portfolio.update_prices(
            {"BTC": 120},
            bar_index=1,
            ohlc_data={"BTC": {"open": 115, "high": 120, "low": 114, "close": 120}},
        )

        # Verify trailing stop has moved up
        assert portfolio.get_position("BTC") != 0

        # Gap down below trailing stop of 114
        portfolio.update_prices(
            {"BTC": 105},
            bar_index=2,
            ohlc_data={"BTC": {"open": 110, "high": 112, "low": 105, "close": 105}},
        )

        # Should have exited at open price 110 (gap through stop at 114)
        assert portfolio.get_position("BTC") == 0
        trades = portfolio.get_trades()
        assert len(trades) == 1
        assert trades["exit_price"][0] == 110

    def test_short_position_gap_up_through_stop_loss(self):
        """Short position: gap up through stop -> exit at open."""
        portfolio = Portfolio(initial_cash=100_000)

        # Enter short at 100
        portfolio.update_prices(
            {"BTC": 100},
            bar_index=0,
            ohlc_data={"BTC": {"open": 100, "high": 100, "low": 100, "close": 100}},
        )
        portfolio.order("BTC", -10)
        portfolio.set_stop_loss("BTC", stop_price=110)

        # Gap up to 115 (above stop of 110)
        portfolio.update_prices(
            {"BTC": 118},
            bar_index=1,
            ohlc_data={"BTC": {"open": 115, "high": 120, "low": 114, "close": 118}},
        )

        assert portfolio.get_position("BTC") == 0
        trades = portfolio.get_trades()
        assert len(trades) == 1
        # Should exit at open (115), not stop (110)
        assert trades["exit_price"][0] == 115


class TestFactorBasedPriceAdjustment:
    """Test Feature 7: Factor-based price adjustment for commissions."""

    def test_no_factor_column_unchanged(self):
        """Without factor_column, behavior is identical to before."""
        portfolio = Portfolio(initial_cash=100_000, commission=0.01)
        portfolio.update_prices({"AAPL": 200.0})
        portfolio.order("AAPL", 100)

        # Commission should be on the adjusted price (200)
        filled_orders = portfolio.get_orders()
        order = filled_orders[0]
        assert order.is_filled()
        # 100 shares * 200 * 0.01 = 200
        assert abs(order.commission_paid - 200.0) < 0.01

    def test_factor_column_halves_commission(self):
        """With factor=2.0, raw_price = adjusted/2, so commission is halved."""
        portfolio = Portfolio(initial_cash=100_000, commission=0.01, factor_column="factor")
        portfolio._factors["AAPL"] = 2.0
        portfolio.update_prices({"AAPL": 200.0})
        portfolio.order("AAPL", 100)

        filled_orders = portfolio.get_orders()
        order = filled_orders[0]
        assert order.is_filled()
        # raw_price = 200 / 2 = 100, commission = 100 * 100 * 0.01 = 100
        assert abs(order.commission_paid - 100.0) < 0.01

    def test_factor_of_one_identical(self):
        """Factor of 1.0 should produce identical results to no factor."""
        portfolio_no_factor = Portfolio(initial_cash=100_000, commission=0.01)
        portfolio_no_factor.update_prices({"AAPL": 150.0})
        portfolio_no_factor.order("AAPL", 50)

        portfolio_with_factor = Portfolio(initial_cash=100_000, commission=0.01, factor_column="factor")
        portfolio_with_factor._factors["AAPL"] = 1.0
        portfolio_with_factor.update_prices({"AAPL": 150.0})
        portfolio_with_factor.order("AAPL", 50)

        orders_no = portfolio_no_factor.get_orders()
        orders_with = portfolio_with_factor.get_orders()
        assert abs(orders_no[0].commission_paid - orders_with[0].commission_paid) < 1e-10

    def test_factor_changes_mid_backtest(self):
        """Factor changing mid-backtest (simulating a 2:1 split)."""
        portfolio = Portfolio(initial_cash=100_000, commission=0.01, factor_column="factor")

        # Before split: factor=1.0, price=200
        portfolio._factors["AAPL"] = 1.0
        portfolio.update_prices({"AAPL": 200.0})
        portfolio.order("AAPL", 10)

        order1 = portfolio.get_orders()[0]
        # Commission on raw price 200: 10 * 200 * 0.01 = 20
        assert abs(order1.commission_paid - 20.0) < 0.01

        # After 2:1 split: factor=2.0, adjusted price=200 (raw=100)
        portfolio._factors["AAPL"] = 2.0
        portfolio.update_prices({"AAPL": 200.0}, bar_index=1)
        portfolio.order("AAPL", 10)

        order2 = portfolio.get_orders()[1]
        # Commission on raw price 100: 10 * 100 * 0.01 = 10
        assert abs(order2.commission_paid - 10.0) < 0.01

    def test_engine_with_factor_column(self):
        """Engine passes factor_column to Portfolio and extracts factor data."""

        class BuyAndHold(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def next(self, ctx: BacktestContext) -> None:
                if ctx.bar_index == 0:
                    ctx.portfolio.order("asset", 10)

        # Create data with a factor column
        df = pl.DataFrame(
            {
                "timestamp": list(range(5)),
                "close": [100.0, 102.0, 104.0, 106.0, 108.0],
                "factor": [2.0, 2.0, 2.0, 2.0, 2.0],
            }
        )

        # With factor_column: commission on raw price (close/factor)
        engine = Engine(
            strategy=BuyAndHold(),
            data=df,
            initial_cash=100_000,
            commission=0.01,
            factor_column="factor",
        )
        result = engine.run()
        assert result.success is not False

        # Verify factor was used: raw_price = 100/2 = 50
        # Commission = 10 * 50 * 0.01 = 5
        order = engine.portfolio.get_orders()[0]
        assert abs(order.commission_paid - 5.0) < 0.01

    def test_engine_without_factor_column(self):
        """Engine without factor_column uses adjusted prices for commission."""

        class BuyAndHold(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def next(self, ctx: BacktestContext) -> None:
                if ctx.bar_index == 0:
                    ctx.portfolio.order("asset", 10)

        df = pl.DataFrame(
            {
                "timestamp": list(range(5)),
                "close": [100.0, 102.0, 104.0, 106.0, 108.0],
                "factor": [2.0, 2.0, 2.0, 2.0, 2.0],
            }
        )

        # Without factor_column: commission on adjusted price
        engine = Engine(
            strategy=BuyAndHold(),
            data=df,
            initial_cash=100_000,
            commission=0.01,
        )
        result = engine.run()
        assert result.success is not False

        # Commission = 10 * 100 * 0.01 = 10
        order = engine.portfolio.get_orders()[0]
        assert abs(order.commission_paid - 10.0) < 0.01

    def test_standardize_dataframe_recognizes_factor(self):
        """standardize_dataframe should rename Factor -> factor."""
        df = pl.DataFrame(
            {
                "Date": ["2024-01-01"],
                "Close": [100.0],
                "Factor": [1.5],
            }
        )
        result = standardize_dataframe(df)
        assert "factor" in result.columns
        assert "close" in result.columns
        assert "timestamp" in result.columns

    def test_runner_backtest_with_factor_column(self):
        """runner.backtest() passes factor_column through to Engine."""
        from polarbt.runner import backtest

        class SimpleStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def next(self, ctx: BacktestContext) -> None:
                if ctx.bar_index == 0:
                    ctx.portfolio.order("asset", 10)

        df = pl.DataFrame(
            {
                "timestamp": list(range(10)),
                "close": [100.0 + i for i in range(10)],
                "factor": [2.0] * 10,
            }
        )

        result = backtest(
            SimpleStrategy,
            df,
            initial_cash=100_000,
            commission=0.01,
            slippage=0.0,
            factor_column="factor",
        )
        assert result.success is not False
