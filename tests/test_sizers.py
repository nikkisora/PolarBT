"""Tests for position sizing module."""

import polars as pl
import pytest

from polarbt.core import Engine, Portfolio, Strategy
from polarbt.sizers import (
    FixedRiskSizer,
    FixedSizer,
    KellySizer,
    MaxPositionSizer,
    PercentSizer,
    VolatilitySizer,
)


def make_portfolio(cash: float = 100_000.0) -> Portfolio:
    """Create a minimal Portfolio for testing sizers."""
    portfolio = Portfolio(initial_cash=cash)
    portfolio._current_prices["BTC"] = 50_000.0
    portfolio._current_prices["ETH"] = 2_000.0
    return portfolio


class TestFixedSizer:
    def test_returns_fixed_quantity(self):
        sizer = FixedSizer(2.5)
        qty = sizer.size(make_portfolio(), "BTC", 50_000.0)
        assert qty == 2.5

    def test_ignores_portfolio_and_price(self):
        sizer = FixedSizer(10.0)
        qty1 = sizer.size(make_portfolio(1_000), "BTC", 100.0)
        qty2 = sizer.size(make_portfolio(1_000_000), "ETH", 99_999.0)
        assert qty1 == qty2 == 10.0

    def test_rejects_zero_quantity(self):
        with pytest.raises(ValueError, match="positive"):
            FixedSizer(0)

    def test_rejects_negative_quantity(self):
        with pytest.raises(ValueError, match="positive"):
            FixedSizer(-1.0)


class TestPercentSizer:
    def test_basic_calculation(self):
        sizer = PercentSizer(0.5)
        portfolio = make_portfolio(100_000.0)
        qty = sizer.size(portfolio, "ETH", 2_000.0)
        assert qty == 25.0  # 50% of 100k / 2000

    def test_full_portfolio(self):
        sizer = PercentSizer(1.0)
        portfolio = make_portfolio(100_000.0)
        qty = sizer.size(portfolio, "BTC", 50_000.0)
        assert qty == 2.0

    def test_zero_price_returns_zero(self):
        sizer = PercentSizer(0.5)
        assert sizer.size(make_portfolio(), "BTC", 0.0) == 0.0

    def test_rejects_invalid_percent(self):
        with pytest.raises(ValueError):
            PercentSizer(0.0)
        with pytest.raises(ValueError):
            PercentSizer(1.5)
        with pytest.raises(ValueError):
            PercentSizer(-0.1)


class TestFixedRiskSizer:
    def test_basic_calculation(self):
        sizer = FixedRiskSizer(0.02)
        portfolio = make_portfolio(100_000.0)
        # Risk 2% = $2000, stop distance = $100 -> 20 units
        qty = sizer.size(portfolio, "ETH", 2_000.0, stop_distance=100.0)
        assert qty == 20.0

    def test_small_stop_gives_large_position(self):
        sizer = FixedRiskSizer(0.01)
        portfolio = make_portfolio(100_000.0)
        qty = sizer.size(portfolio, "ETH", 2_000.0, stop_distance=10.0)
        assert qty == 100.0  # $1000 risk / $10 = 100

    def test_zero_stop_returns_zero(self):
        sizer = FixedRiskSizer(0.02)
        assert sizer.size(make_portfolio(), "BTC", 50_000.0, stop_distance=0.0) == 0.0

    def test_missing_stop_distance_raises(self):
        sizer = FixedRiskSizer(0.02)
        with pytest.raises(ValueError, match="stop_distance"):
            sizer.size(make_portfolio(), "BTC", 50_000.0)

    def test_rejects_invalid_risk(self):
        with pytest.raises(ValueError):
            FixedRiskSizer(0.0)
        with pytest.raises(ValueError):
            FixedRiskSizer(1.5)


class TestKellySizer:
    def test_positive_edge(self):
        # win_rate=0.6, avg_win=2.0, avg_loss=1.0 -> kelly = 0.6 - 0.4/2 = 0.4
        sizer = KellySizer(win_rate=0.6, avg_win=2.0, avg_loss=1.0, max_fraction=1.0)
        assert pytest.approx(sizer.kelly_fraction, abs=1e-10) == 0.4
        portfolio = make_portfolio(100_000.0)
        qty = sizer.size(portfolio, "ETH", 2_000.0)
        assert qty == 20.0  # 40% of 100k / 2000

    def test_negative_edge_returns_zero(self):
        # win_rate=0.3, avg_win=1.0, avg_loss=1.0 -> kelly = 0.3 - 0.7 = -0.4
        sizer = KellySizer(win_rate=0.3, avg_win=1.0, avg_loss=1.0)
        assert sizer.kelly_fraction < 0
        assert sizer.size(make_portfolio(), "ETH", 2_000.0) == 0.0

    def test_max_fraction_caps(self):
        sizer = KellySizer(win_rate=0.6, avg_win=2.0, avg_loss=1.0, max_fraction=0.1)
        portfolio = make_portfolio(100_000.0)
        qty = sizer.size(portfolio, "ETH", 2_000.0)
        assert qty == 5.0  # 10% of 100k / 2000 (capped from 40%)

    def test_rejects_invalid_params(self):
        with pytest.raises(ValueError):
            KellySizer(win_rate=0.0, avg_win=1.0, avg_loss=1.0)
        with pytest.raises(ValueError):
            KellySizer(win_rate=0.5, avg_win=0, avg_loss=1.0)
        with pytest.raises(ValueError):
            KellySizer(win_rate=0.5, avg_win=1.0, avg_loss=-1.0)


class TestVolatilitySizer:
    def test_basic_calculation(self):
        sizer = VolatilitySizer(0.02)
        portfolio = make_portfolio(100_000.0)
        # Risk $2000 / ATR $500 = 4 units
        qty = sizer.size(portfolio, "BTC", 50_000.0, atr=500.0)
        assert qty == 4.0

    def test_high_volatility_reduces_size(self):
        sizer = VolatilitySizer(0.02)
        portfolio = make_portfolio(100_000.0)
        qty_low = sizer.size(portfolio, "BTC", 50_000.0, atr=100.0)
        qty_high = sizer.size(portfolio, "BTC", 50_000.0, atr=1_000.0)
        assert qty_low > qty_high

    def test_zero_atr_returns_zero(self):
        sizer = VolatilitySizer(0.02)
        assert sizer.size(make_portfolio(), "BTC", 50_000.0, atr=0.0) == 0.0

    def test_missing_atr_raises(self):
        sizer = VolatilitySizer(0.02)
        with pytest.raises(ValueError, match="atr"):
            sizer.size(make_portfolio(), "BTC", 50_000.0)


class TestMaxPositionSizer:
    def test_caps_by_quantity(self):
        inner = FixedSizer(100.0)
        sizer = MaxPositionSizer(inner, max_quantity=10.0)
        qty = sizer.size(make_portfolio(), "ETH", 2_000.0)
        assert qty == 10.0

    def test_caps_by_percent(self):
        inner = FixedSizer(100.0)
        sizer = MaxPositionSizer(inner, max_percent=0.1)
        portfolio = make_portfolio(100_000.0)
        # 10% of 100k / 2000 = 5
        qty = sizer.size(portfolio, "ETH", 2_000.0)
        assert qty == 5.0

    def test_both_caps_most_restrictive_wins(self):
        inner = FixedSizer(100.0)
        sizer = MaxPositionSizer(inner, max_quantity=3.0, max_percent=0.1)
        portfolio = make_portfolio(100_000.0)
        # max_quantity=3 is more restrictive than max_percent (5 units)
        qty = sizer.size(portfolio, "ETH", 2_000.0)
        assert qty == 3.0

    def test_passthrough_when_under_limits(self):
        inner = FixedSizer(2.0)
        sizer = MaxPositionSizer(inner, max_quantity=10.0)
        qty = sizer.size(make_portfolio(), "ETH", 2_000.0)
        assert qty == 2.0

    def test_requires_at_least_one_limit(self):
        with pytest.raises(ValueError, match="At least one"):
            MaxPositionSizer(FixedSizer(1.0))


class TestOrderWithSizer:
    """Test Portfolio.order_with_sizer() integration."""

    def test_buy_with_fixed_sizer(self):
        portfolio = make_portfolio(100_000.0)
        sizer = FixedSizer(1.0)
        order_id = portfolio.order_with_sizer("BTC", sizer, direction=1.0, price=50_000.0)
        assert order_id is not None
        assert portfolio.get_position("BTC") == 1.0

    def test_sell_with_fixed_sizer(self):
        portfolio = make_portfolio(100_000.0)
        # First buy
        portfolio.order("BTC", 2.0)
        # Then sell 1 via sizer
        sizer = FixedSizer(1.0)
        order_id = portfolio.order_with_sizer("BTC", sizer, direction=-1.0, price=50_000.0)
        assert order_id is not None
        assert portfolio.get_position("BTC") == 1.0

    def test_zero_direction_returns_none(self):
        portfolio = make_portfolio(100_000.0)
        sizer = FixedSizer(1.0)
        assert portfolio.order_with_sizer("BTC", sizer, direction=0.0, price=50_000.0) is None

    def test_uses_current_price_when_none(self):
        portfolio = make_portfolio(100_000.0)
        sizer = PercentSizer(0.5)
        order_id = portfolio.order_with_sizer("ETH", sizer, direction=1.0)
        assert order_id is not None
        # 50% of 100k / 2000 = 25
        assert portfolio.get_position("ETH") == 25.0

    def test_passes_kwargs_to_sizer(self):
        portfolio = make_portfolio(100_000.0)
        sizer = FixedRiskSizer(0.01)
        order_id = portfolio.order_with_sizer("ETH", sizer, direction=1.0, price=2_000.0, stop_distance=50.0)
        assert order_id is not None
        # Risk $1000 / $50 = 20 units
        assert portfolio.get_position("ETH") == 20.0

    def test_rejects_non_sizer(self):
        portfolio = make_portfolio(100_000.0)
        with pytest.raises(TypeError, match="Sizer"):
            portfolio.order_with_sizer("BTC", "not_a_sizer", direction=1.0, price=50_000.0)

    def test_with_engine_integration(self):
        """Test sizer works within a full backtest."""

        class SizerStrategy(Strategy):
            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def next(self, ctx):
                if ctx.bar_index == 0 and ctx.portfolio.get_position("asset") == 0:
                    sizer = PercentSizer(0.5)
                    ctx.portfolio.order_with_sizer("asset", sizer, direction=1.0)

        df = pl.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})
        engine = Engine(strategy=SizerStrategy(), data=df, initial_cash=10_000.0, warmup=0)
        results = engine.run()
        assert results["final_equity"] > 0
