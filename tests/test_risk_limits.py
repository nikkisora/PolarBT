"""Tests for risk limit enforcement in Portfolio."""

from datetime import datetime

import polars as pl
import pytest

from polarbt.core import Engine, Portfolio, Strategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_portfolio(**kwargs: object) -> Portfolio:
    """Create a Portfolio with sensible defaults and apply kwargs."""
    defaults: dict[str, object] = {"initial_cash": 100_000.0, "commission": 0.0, "slippage": 0.0}
    defaults.update(kwargs)
    return Portfolio(**defaults)  # type: ignore[arg-type]


def _update(
    portfolio: Portfolio,
    prices: dict[str, float],
    bar: int = 0,
    ohlc: dict[str, dict[str, float]] | None = None,
    ts: object = None,
) -> None:
    """Shortcut to update portfolio prices."""
    if ohlc is None:
        ohlc = {asset: {"open": p, "high": p, "low": p, "close": p} for asset, p in prices.items()}
    portfolio.update_prices(prices, bar_index=bar, ohlc_data=ohlc, timestamp=ts)


# ===========================================================================
# max_position_size
# ===========================================================================


class TestMaxPositionSize:
    def test_clamps_order_to_limit(self):
        """Order that would exceed max_position_size is clamped."""
        p = _make_portfolio(max_position_size=0.5)
        _update(p, {"BTC": 100.0}, bar=0)

        # Portfolio value = 100k. Max position = 50k. 50k / 100 = 500 units.
        # Requesting 600 units should be clamped to 500.
        oid = p.order("BTC", 600.0)
        assert oid is not None
        order = p.orders[oid]
        assert order.is_filled()
        assert p.get_position("BTC") == pytest.approx(500.0, abs=1.0)

    def test_allows_within_limit(self):
        """Order within limit executes fully."""
        p = _make_portfolio(max_position_size=0.5)
        _update(p, {"BTC": 100.0}, bar=0)

        oid = p.order("BTC", 400.0)
        assert oid is not None
        assert p.get_position("BTC") == pytest.approx(400.0)

    def test_no_limit_allows_full_order(self):
        """Without max_position_size, no clamping occurs."""
        p = _make_portfolio()
        _update(p, {"BTC": 100.0}, bar=0)

        oid = p.order("BTC", 900.0)
        assert oid is not None
        assert p.get_position("BTC") == pytest.approx(900.0)

    def test_risk_reducing_order_bypasses_limit(self):
        """Closing a position should bypass max_position_size."""
        p = _make_portfolio(max_position_size=0.5)
        _update(p, {"BTC": 100.0}, bar=0)

        # Open a position at limit
        p.order("BTC", 500.0)
        assert p.get_position("BTC") == pytest.approx(500.0, abs=1.0)

        # Close it — should always work
        p.close_position("BTC")
        assert p.get_position("BTC") == 0.0

    def test_short_position_clamped(self):
        """Short positions are also clamped by max_position_size."""
        p = _make_portfolio(max_position_size=0.5)
        _update(p, {"BTC": 100.0}, bar=0)

        # Max abs position = 500 units. Requesting -600 should clamp to -500.
        oid = p.order("BTC", -600.0)
        assert oid is not None
        assert p.get_position("BTC") == pytest.approx(-500.0, abs=1.0)


# ===========================================================================
# max_total_exposure
# ===========================================================================


class TestMaxTotalExposure:
    def test_clamps_when_exposure_exceeded(self):
        """Second position is clamped when total exposure would exceed limit."""
        p = _make_portfolio(max_total_exposure=1.0)
        _update(p, {"BTC": 100.0, "ETH": 50.0}, bar=0)

        # Buy 500 BTC = 50k exposure (50% of 100k)
        p.order("BTC", 500.0)

        # Try to buy 1200 ETH = 60k. Total would be 110k > 100k.
        # Should clamp ETH to ~50k / 50 = 1000 units
        oid = p.order("ETH", 1200.0)
        assert oid is not None
        # After BTC purchase: cash ~50k, BTC worth 50k, portfolio ~100k
        # Max total exposure = 100k. Already 50k in BTC. Room = 50k for ETH.
        assert abs(p.get_position("ETH")) <= 1001.0

    def test_allows_within_limit(self):
        """Orders within exposure limit execute fully."""
        p = _make_portfolio(max_total_exposure=1.0)
        _update(p, {"BTC": 100.0}, bar=0)

        # 400 * 100 = 40k. 40% exposure < 100%.
        p.order("BTC", 400.0)
        assert p.get_position("BTC") == pytest.approx(400.0)

    def test_risk_reducing_bypasses_exposure_limit(self):
        """Closing positions bypasses exposure limit."""
        p = _make_portfolio(max_total_exposure=0.5)
        _update(p, {"BTC": 100.0}, bar=0)

        p.order("BTC", 500.0)
        # Now exposure is at limit. Closing should work.
        p.close_position("BTC")
        assert p.get_position("BTC") == 0.0


# ===========================================================================
# max_drawdown_stop
# ===========================================================================


class TestMaxDrawdownStop:
    def test_halts_trading_on_drawdown(self):
        """Trading halts when drawdown exceeds max_drawdown_stop."""
        p = _make_portfolio(max_drawdown_stop=0.1)
        _update(p, {"BTC": 100.0}, bar=0)

        # Buy some BTC
        p.order("BTC", 500.0)
        assert p.get_position("BTC") == pytest.approx(500.0)

        # Price drops 15% — drawdown > 10%
        _update(p, {"BTC": 85.0}, bar=1)

        # Peak was ~100k, now ~(100k - 500*15) = ~92.5k. Drawdown = 7.5% (may vary)
        # Let's drop more to ensure halt
        _update(p, {"BTC": 70.0}, bar=2)

        # Now: cash ~50k, position 500*70=35k, total=85k. Peak=100k. DD=15% > 10%.
        assert p.trading_halted

        # New orders should be rejected
        oid = p.order("BTC", 100.0)
        assert oid is None

    def test_allows_close_when_halted(self):
        """Risk-reducing orders still execute when halted."""
        p = _make_portfolio(max_drawdown_stop=0.1)
        _update(p, {"BTC": 100.0}, bar=0)
        p.order("BTC", 500.0)
        _update(p, {"BTC": 70.0}, bar=1)

        assert p.trading_halted

        # Close position should still work
        oid = p.close_position("BTC")
        assert oid is not None
        assert p.get_position("BTC") == 0.0

    def test_no_halt_within_limit(self):
        """No halt if drawdown stays within limit."""
        p = _make_portfolio(max_drawdown_stop=0.2)
        _update(p, {"BTC": 100.0}, bar=0)
        p.order("BTC", 500.0)
        _update(p, {"BTC": 95.0}, bar=1)  # Small drop

        assert not p.trading_halted
        oid = p.order("BTC", 50.0)
        assert oid is not None

    def test_stop_loss_executes_when_halted(self):
        """SL/TP triggered by update_prices should work even when halted."""
        p = _make_portfolio(max_drawdown_stop=0.1)
        _update(p, {"BTC": 100.0}, bar=0)
        p.order("BTC", 500.0)
        p.set_stop_loss("BTC", stop_price=75.0)

        # Big drop triggers both halt and SL
        _update(p, {"BTC": 70.0}, bar=1, ohlc={"BTC": {"open": 100.0, "high": 100.0, "low": 70.0, "close": 70.0}})

        # SL should have closed the position
        assert p.get_position("BTC") == 0.0


# ===========================================================================
# daily_loss_limit
# ===========================================================================


class TestDailyLossLimit:
    def test_halts_on_daily_loss(self):
        """Trading halts when daily loss exceeds limit."""
        p = _make_portfolio(daily_loss_limit=0.05)
        day1 = datetime(2024, 1, 1, 9, 30)
        _update(p, {"BTC": 100.0}, bar=0, ts=day1)

        p.order("BTC", 500.0)
        # Equity ~100k. If price drops 10%, equity ~95k. Daily loss = 5%.
        day1_bar2 = datetime(2024, 1, 1, 10, 30)
        _update(p, {"BTC": 90.0}, bar=1, ts=day1_bar2)

        assert p._daily_halted

        # New orders rejected
        oid = p.order("BTC", 100.0)
        assert oid is None

    def test_resets_next_day(self):
        """Daily halt resets on next calendar day."""
        p = _make_portfolio(daily_loss_limit=0.05)
        day1 = datetime(2024, 1, 1, 9, 30)
        _update(p, {"BTC": 100.0}, bar=0, ts=day1)
        p.order("BTC", 500.0)

        day1_bar2 = datetime(2024, 1, 1, 10, 30)
        _update(p, {"BTC": 90.0}, bar=1, ts=day1_bar2)
        assert p._daily_halted

        # Next day
        day2 = datetime(2024, 1, 2, 9, 30)
        _update(p, {"BTC": 90.0}, bar=2, ts=day2)

        assert not p._daily_halted
        oid = p.order("BTC", 10.0)
        assert oid is not None

    def test_allows_close_when_daily_halted(self):
        """Close positions even when daily halted."""
        p = _make_portfolio(daily_loss_limit=0.05)
        day1 = datetime(2024, 1, 1, 9, 30)
        _update(p, {"BTC": 100.0}, bar=0, ts=day1)
        p.order("BTC", 500.0)

        day1_bar2 = datetime(2024, 1, 1, 10, 30)
        _update(p, {"BTC": 90.0}, bar=1, ts=day1_bar2)
        assert p._daily_halted

        oid = p.close_position("BTC")
        assert oid is not None
        assert p.get_position("BTC") == 0.0


# ===========================================================================
# Engine integration
# ===========================================================================


class TestRiskLimitsEngine:
    def test_max_position_size_via_engine(self):
        """Risk limits are passed through Engine to Portfolio."""

        class BuyAllStrategy(Strategy):
            def preprocess(self, df):
                return df

            def next(self, ctx):
                if ctx.bar_index == 0:
                    ctx.portfolio.order("asset", 2000.0)

        df = pl.DataFrame({"close": [100.0, 101.0, 102.0]})
        engine = Engine(
            strategy=BuyAllStrategy(),
            data=df,
            initial_cash=100_000.0,
            max_position_size=0.5,
            warmup=0,
        )
        engine.run()
        assert engine.portfolio is not None
        pos = engine.portfolio.get_position("asset")
        # Max 50% of 100k = 50k / 100 = 500 units
        assert pos <= 501.0

    def test_max_drawdown_stop_via_engine(self):
        """Drawdown halt works through Engine."""

        class AlwaysBuyStrategy(Strategy):
            def preprocess(self, df):
                return df

            def next(self, ctx):
                if ctx.portfolio.get_position("asset") == 0.0:
                    ctx.portfolio.order("asset", 500.0)

        # Price drops significantly mid-way
        prices = [100.0] * 3 + [70.0] * 5 + [100.0] * 3
        df = pl.DataFrame({"close": prices})
        engine = Engine(
            strategy=AlwaysBuyStrategy(),
            data=df,
            initial_cash=100_000.0,
            max_drawdown_stop=0.1,
            warmup=0,
        )
        engine.run()
        assert engine.portfolio is not None
        # After the drop, trading should be halted. Position opened on bar 0 stays.
        assert engine.portfolio._trading_halted


# ===========================================================================
# trading_halted property
# ===========================================================================


class TestTradingHaltedProperty:
    def test_property_combines_halts(self):
        p = _make_portfolio(max_drawdown_stop=0.1, daily_loss_limit=0.05)
        assert not p.trading_halted

        p._trading_halted = True
        assert p.trading_halted

        p._trading_halted = False
        p._daily_halted = True
        assert p.trading_halted

    def test_is_order_risk_reducing(self):
        p = _make_portfolio()
        _update(p, {"BTC": 100.0}, bar=0)
        p.order("BTC", 100.0)

        # Selling 50 from a long of 100 is risk-reducing
        assert p._is_order_risk_reducing("BTC", -50.0)
        # Selling 100 (closing) is risk-reducing
        assert p._is_order_risk_reducing("BTC", -100.0)
        # Selling 150 (reversal) is NOT purely risk-reducing
        assert not p._is_order_risk_reducing("BTC", -150.0)
        # Buying more is NOT risk-reducing
        assert not p._is_order_risk_reducing("BTC", 50.0)

    def test_get_total_exposure(self):
        p = _make_portfolio()
        _update(p, {"BTC": 100.0, "ETH": 50.0}, bar=0)
        p.order("BTC", 300.0)  # 30k
        p.order("ETH", -200.0)  # 10k abs

        # Portfolio value ~= 100k - 30k + 30k + 10k - 10k = ~100k (ignoring commission)
        # Total abs exposure = 30k + 10k = 40k
        exposure = p._get_total_exposure()
        assert exposure == pytest.approx(0.4, abs=0.05)
