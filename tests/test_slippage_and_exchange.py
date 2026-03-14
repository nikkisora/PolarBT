"""Tests for AMM-aware slippage models (Phase 4) and exchange rate support (Phase 5)."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from polarbt.core import BacktestContext, Engine, Strategy
from polarbt.slippage import AMMSlippage, FlatSlippage, make_slippage_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_single_asset_df(n: int = 20) -> pl.DataFrame:
    base = datetime(2025, 1, 1)
    return pl.DataFrame(
        {
            "timestamp": [base + timedelta(hours=i) for i in range(n)],
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.0 + i for i in range(n)],
            "volume": [1000.0] * n,
        },
        schema={
            "timestamp": pl.Datetime("us"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )


def _make_amm_df(n: int = 10) -> pl.DataFrame:
    """DataFrame with pool_reserve_last column for AMM slippage tests."""
    base = datetime(2025, 1, 1)
    return pl.DataFrame(
        {
            "timestamp": [base + timedelta(hours=i) for i in range(n)],
            "symbol": ["TOKEN"] * n,
            "open": [1.0] * n,
            "high": [1.1] * n,
            "low": [0.9] * n,
            "close": [1.0] * n,
            "volume": [100.0] * n,
            "pool_reserve_last": [10000.0 - i * 500 for i in range(n)],
        },
        schema={
            "timestamp": pl.Datetime("us"),
            "symbol": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "pool_reserve_last": pl.Float64,
        },
    )


class BuyOnceStrategy(Strategy):
    """Buy 10 units of asset on bar 1."""

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def next(self, ctx: BacktestContext) -> None:
        if ctx.bar_index == 1:
            sym = ctx.symbols[0] if ctx.symbols else "asset"
            ctx.portfolio.order(sym, 10.0)


class RecordPriceStrategy(Strategy):
    """Record execution prices for verification."""

    def __init__(self) -> None:
        self.filled_prices: list[float | None] = []
        self.ordered_on: int | None = None

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def next(self, ctx: BacktestContext) -> None:
        if ctx.bar_index == 1:
            sym = ctx.symbols[0] if ctx.symbols else "asset"
            order_id = ctx.portfolio.order(sym, 10.0)
            self.ordered_on = ctx.bar_index
            if order_id:
                order = ctx.portfolio.orders.get(order_id)
                if order and order.filled_price is not None:
                    self.filled_prices.append(order.filled_price)


# ===========================================================================
# Phase 4.1: SlippageModel unit tests
# ===========================================================================


class TestFlatSlippage:
    def test_buy_increases_price(self) -> None:
        model = FlatSlippage(pct=0.01)
        result = model.calculate(100.0, 10.0, is_buy=True)
        assert result == pytest.approx(101.0)

    def test_sell_decreases_price(self) -> None:
        model = FlatSlippage(pct=0.01)
        result = model.calculate(100.0, 10.0, is_buy=False)
        assert result == pytest.approx(99.0)

    def test_zero_slippage(self) -> None:
        model = FlatSlippage(pct=0.0)
        assert model.calculate(100.0, 10.0, is_buy=True) == 100.0
        assert model.calculate(100.0, 10.0, is_buy=False) == 100.0

    def test_estimate_rate(self) -> None:
        model = FlatSlippage(pct=0.005)
        assert model.estimate_rate(100.0, 10.0, is_buy=True) == 0.005

    def test_negative_pct_raises(self) -> None:
        with pytest.raises(ValueError, match="pct must be non-negative"):
            FlatSlippage(pct=-0.01)


class TestAMMSlippage:
    def test_small_trade_low_impact(self) -> None:
        model = AMMSlippage()
        ctx = {"pool_reserve_last": 10000.0}
        result = model.calculate(1.0, 10.0, is_buy=True, context=ctx)
        # trade_value = 10 * 1.0 = 10, rate = 10 / (10000 + 10) ~ 0.001
        expected_rate = 10.0 / 10010.0
        assert result == pytest.approx(1.0 * (1 + expected_rate))

    def test_large_trade_high_impact(self) -> None:
        model = AMMSlippage()
        ctx = {"pool_reserve_last": 1000.0}
        result = model.calculate(1.0, 1000.0, is_buy=True, context=ctx)
        # trade_value = 1000, rate = 1000 / (1000 + 1000) = 0.5
        assert result == pytest.approx(1.5)

    def test_sell_decreases_price(self) -> None:
        model = AMMSlippage()
        ctx = {"pool_reserve_last": 10000.0}
        buy_price = model.calculate(1.0, 100.0, is_buy=True, context=ctx)
        sell_price = model.calculate(1.0, 100.0, is_buy=False, context=ctx)
        assert buy_price > 1.0
        assert sell_price < 1.0

    def test_no_context_returns_min_slippage(self) -> None:
        model = AMMSlippage(min_slippage=0.001)
        result = model.calculate(100.0, 10.0, is_buy=True)
        assert result == pytest.approx(100.0 * 1.001)

    def test_zero_reserve_returns_min_slippage(self) -> None:
        model = AMMSlippage(min_slippage=0.002)
        ctx = {"pool_reserve_last": 0.0}
        result = model.calculate(100.0, 10.0, is_buy=True, context=ctx)
        assert result == pytest.approx(100.0 * 1.002)

    def test_custom_reserve_key(self) -> None:
        model = AMMSlippage(reserve_key="liquidity")
        ctx = {"liquidity": 5000.0}
        rate = model.estimate_rate(1.0, 100.0, is_buy=True, context=ctx)
        assert rate == pytest.approx(100.0 / 5100.0)

    def test_min_slippage_floor(self) -> None:
        model = AMMSlippage(min_slippage=0.05)
        ctx = {"pool_reserve_last": 1_000_000.0}  # very deep pool
        rate = model.estimate_rate(1.0, 1.0, is_buy=True, context=ctx)
        assert rate == pytest.approx(0.05)  # min_slippage dominates


class TestMakeSlippageModel:
    def test_float_creates_flat(self) -> None:
        model = make_slippage_model(0.01)
        assert isinstance(model, FlatSlippage)
        assert model.pct == 0.01

    def test_model_passthrough(self) -> None:
        amm = AMMSlippage()
        assert make_slippage_model(amm) is amm


# ===========================================================================
# Phase 4.1: Engine integration with SlippageModel
# ===========================================================================


class TestSlippageEngineIntegration:
    def test_flat_slippage_float_backward_compat(self) -> None:
        """Float slippage should work exactly as before."""
        df = _make_single_asset_df()
        strategy = BuyOnceStrategy()
        engine = Engine(strategy, df, slippage=0.01, warmup=0)
        result = engine.run()
        # Should complete without errors and apply slippage
        assert result.final_equity > 0

    def test_flat_slippage_model(self) -> None:
        """FlatSlippage model should produce same results as float."""
        df = _make_single_asset_df()
        strategy1 = BuyOnceStrategy()
        engine1 = Engine(strategy1, df, slippage=0.01, warmup=0)
        result1 = engine1.run()

        strategy2 = BuyOnceStrategy()
        engine2 = Engine(strategy2, df, slippage=FlatSlippage(0.01), warmup=0)
        result2 = engine2.run()

        assert result1.final_equity == pytest.approx(result2.final_equity, rel=1e-10)

    def test_amm_slippage_in_engine(self) -> None:
        """AMMSlippage should use pool_reserve_last from bar data."""
        df = _make_amm_df()
        strategy = BuyOnceStrategy()
        engine = Engine(strategy, df, slippage=AMMSlippage(), warmup=0)
        result = engine.run()
        assert result.final_equity > 0

    def test_amm_slippage_higher_than_zero(self) -> None:
        """AMM slippage should result in worse price than zero slippage."""
        df = _make_amm_df()

        strategy_no_slip = BuyOnceStrategy()
        engine_no_slip = Engine(strategy_no_slip, df, slippage=0.0, warmup=0)
        result_no_slip = engine_no_slip.run()

        strategy_amm = BuyOnceStrategy()
        engine_amm = Engine(strategy_amm, df, slippage=AMMSlippage(), warmup=0)
        result_amm = engine_amm.run()

        # With slippage, we pay more for buys -> less final equity
        assert result_amm.final_equity < result_no_slip.final_equity


# ===========================================================================
# Phase 5: Exchange rate support
# ===========================================================================


class TestExchangeRate:
    def _make_rate_df(self, n: int = 20, rate: float = 150.0) -> pl.DataFrame:
        """Create a constant-rate exchange rate DataFrame."""
        base = datetime(2025, 1, 1)
        return pl.DataFrame(
            {
                "timestamp": [base + timedelta(hours=i) for i in range(n)],
                "rate": [rate] * n,
            },
            schema={"timestamp": pl.Datetime("us"), "rate": pl.Float64},
        )

    def test_usd_metrics_populated(self) -> None:
        df = _make_single_asset_df()
        rate_df = self._make_rate_df()
        strategy = BuyOnceStrategy()
        engine = Engine(strategy, df, warmup=0, exchange_rate=rate_df)
        result = engine.run()

        assert result.final_equity_usd is not None
        assert result.total_return_usd is not None
        assert result.sharpe_ratio_usd is not None
        assert result.max_drawdown_usd is not None

    def test_usd_equity_scales_with_rate(self) -> None:
        df = _make_single_asset_df()
        rate = 150.0
        rate_df = self._make_rate_df(rate=rate)
        strategy = BuyOnceStrategy()
        engine = Engine(strategy, df, warmup=0, exchange_rate=rate_df)
        result = engine.run()

        # USD equity should be approximately quote equity * rate
        assert result.final_equity_usd is not None
        assert result.final_equity_usd == pytest.approx(result.final_equity * rate, rel=0.01)

    def test_no_exchange_rate_leaves_usd_none(self) -> None:
        df = _make_single_asset_df()
        strategy = BuyOnceStrategy()
        engine = Engine(strategy, df, warmup=0)
        result = engine.run()

        assert result.final_equity_usd is None
        assert result.total_return_usd is None
        assert result.sharpe_ratio_usd is None
        assert result.max_drawdown_usd is None

    def test_varying_exchange_rate(self) -> None:
        """USD return should differ from quote return when rate changes."""
        df = _make_single_asset_df(n=10)
        base = datetime(2025, 1, 1)
        # Rate doubles over the backtest period
        rate_df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(hours=i) for i in range(10)],
                "rate": [100.0 + i * 10 for i in range(10)],
            },
            schema={"timestamp": pl.Datetime("us"), "rate": pl.Float64},
        )
        strategy = BuyOnceStrategy()
        engine = Engine(strategy, df, warmup=0, exchange_rate=rate_df)
        result = engine.run()

        assert result.total_return_usd is not None
        # With rising exchange rate, USD return should be higher than quote return
        # (the same quote equity is worth more USD over time)
        assert result.total_return_usd != result.total_return
