"""Tests for commission models."""

import polars as pl
import pytest

from polarbtest import (
    BacktestContext,
    CustomCommission,
    Engine,
    FixedPlusPercentCommission,
    MakerTakerCommission,
    PercentCommission,
    Strategy,
    TieredCommission,
)
from polarbtest.commissions import make_commission_model
from polarbtest.core import Portfolio

# --- Unit tests for CommissionModel implementations ---


class TestPercentCommission:
    def test_basic_calculation(self) -> None:
        model = PercentCommission(rate=0.001)
        assert model.calculate(10.0, 100.0) == pytest.approx(1.0)

    def test_zero_rate(self) -> None:
        model = PercentCommission(rate=0.0)
        assert model.calculate(10.0, 100.0) == 0.0

    def test_reversal_has_no_effect(self) -> None:
        model = PercentCommission(rate=0.001)
        assert model.calculate(10.0, 100.0, is_reversal=True) == model.calculate(10.0, 100.0, is_reversal=False)

    def test_negative_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            PercentCommission(rate=-0.001)


class TestFixedPlusPercentCommission:
    def test_basic_calculation(self) -> None:
        model = FixedPlusPercentCommission(fixed=5.0, percent=0.001)
        # 5 + 10*100*0.001 = 5 + 1 = 6
        assert model.calculate(10.0, 100.0) == pytest.approx(6.0)

    def test_reversal_doubles_fixed(self) -> None:
        model = FixedPlusPercentCommission(fixed=5.0, percent=0.001)
        # 2*5 + 10*100*0.001 = 10 + 1 = 11
        assert model.calculate(10.0, 100.0, is_reversal=True) == pytest.approx(11.0)

    def test_zero_fixed(self) -> None:
        model = FixedPlusPercentCommission(fixed=0.0, percent=0.001)
        assert model.calculate(10.0, 100.0) == pytest.approx(1.0)

    def test_negative_values_raise(self) -> None:
        with pytest.raises(ValueError):
            FixedPlusPercentCommission(fixed=-1.0, percent=0.001)
        with pytest.raises(ValueError):
            FixedPlusPercentCommission(fixed=5.0, percent=-0.001)


class TestMakerTakerCommission:
    def test_taker_default(self) -> None:
        model = MakerTakerCommission(maker_rate=0.0002, taker_rate=0.001)
        # taker by default: 10*100*0.001 = 1.0
        assert model.calculate(10.0, 100.0) == pytest.approx(1.0)

    def test_maker_mode(self) -> None:
        model = MakerTakerCommission(maker_rate=0.0002, taker_rate=0.001, is_maker=True)
        # maker: 10*100*0.0002 = 0.2
        assert model.calculate(10.0, 100.0) == pytest.approx(0.2)

    def test_with_fixed_fee(self) -> None:
        model = MakerTakerCommission(maker_rate=0.0002, taker_rate=0.001, fixed=3.0)
        assert model.calculate(10.0, 100.0) == pytest.approx(3.0 + 1.0)

    def test_reversal_doubles_fixed(self) -> None:
        model = MakerTakerCommission(maker_rate=0.0002, taker_rate=0.001, fixed=3.0)
        assert model.calculate(10.0, 100.0, is_reversal=True) == pytest.approx(6.0 + 1.0)

    def test_negative_rates_raise(self) -> None:
        with pytest.raises(ValueError):
            MakerTakerCommission(maker_rate=-0.001, taker_rate=0.001)
        with pytest.raises(ValueError):
            MakerTakerCommission(maker_rate=0.001, taker_rate=-0.001)


class TestTieredCommission:
    def test_single_tier(self) -> None:
        model = TieredCommission(tiers=[(0, 0.001)])
        assert model.calculate(10.0, 100.0) == pytest.approx(1.0)

    def test_volume_based_tier_progression(self) -> None:
        model = TieredCommission(tiers=[(0, 0.001), (100_000, 0.0005)])
        # First trade: volume=0 -> rate=0.001
        assert model.calculate(100.0, 500.0) == pytest.approx(50.0)
        # cumulative_volume is now 50_000
        assert model.cumulative_volume == pytest.approx(50_000.0)
        # Second trade: volume=50_000 -> still rate=0.001
        assert model.calculate(100.0, 500.0) == pytest.approx(50.0)
        # cumulative_volume is now 100_000
        # Third trade: volume=100_000 -> rate=0.0005
        assert model.calculate(100.0, 500.0) == pytest.approx(25.0)

    def test_reset_volume(self) -> None:
        model = TieredCommission(tiers=[(0, 0.001), (100_000, 0.0005)])
        model.cumulative_volume = 200_000.0
        assert model._get_rate() == 0.0005
        model.reset_volume()
        assert model.cumulative_volume == 0.0
        assert model._get_rate() == 0.001

    def test_with_fixed_fee(self) -> None:
        model = TieredCommission(tiers=[(0, 0.001)], fixed=2.0)
        assert model.calculate(10.0, 100.0) == pytest.approx(3.0)

    def test_reversal_doubles_fixed(self) -> None:
        model = TieredCommission(tiers=[(0, 0.001)], fixed=2.0)
        assert model.calculate(10.0, 100.0, is_reversal=True) == pytest.approx(5.0)

    def test_empty_tiers_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            TieredCommission(tiers=[])

    def test_unsorted_tiers_are_sorted(self) -> None:
        model = TieredCommission(tiers=[(100_000, 0.0005), (0, 0.001)])
        assert model.tiers[0] == (0, 0.001)
        assert model.tiers[1] == (100_000, 0.0005)


class TestCustomCommission:
    def test_basic_callable(self) -> None:
        model = CustomCommission(lambda size, price, is_reversal: max(1.0, size * price * 0.001))
        assert model.calculate(10.0, 100.0) == pytest.approx(1.0)
        assert model.calculate(100.0, 100.0) == pytest.approx(10.0)

    def test_reversal_passed_through(self) -> None:
        def my_func(size: float, price: float, is_reversal: bool) -> float:
            base = size * price * 0.001
            return base * 2 if is_reversal else base

        model = CustomCommission(my_func)
        assert model.calculate(10.0, 100.0) == pytest.approx(1.0)
        assert model.calculate(10.0, 100.0, is_reversal=True) == pytest.approx(2.0)


# --- make_commission_model tests ---


class TestMakeCommissionModel:
    def test_float_creates_percent(self) -> None:
        model = make_commission_model(0.001)
        assert isinstance(model, PercentCommission)

    def test_tuple_creates_fixed_plus_percent(self) -> None:
        model = make_commission_model((5.0, 0.001))
        assert isinstance(model, FixedPlusPercentCommission)

    def test_model_returned_as_is(self) -> None:
        original = MakerTakerCommission(0.0002, 0.001)
        assert make_commission_model(original) is original


# --- Integration tests with Portfolio ---


class TestPortfolioCommissionModel:
    def test_portfolio_accepts_commission_model(self) -> None:
        model = PercentCommission(rate=0.001)
        portfolio = Portfolio(commission=model)
        assert portfolio.commission_model is model

    def test_portfolio_backward_compat_float(self) -> None:
        portfolio = Portfolio(commission=0.001)
        assert isinstance(portfolio.commission_model, PercentCommission)

    def test_portfolio_backward_compat_tuple(self) -> None:
        portfolio = Portfolio(commission=(5.0, 0.001))
        assert isinstance(portfolio.commission_model, FixedPlusPercentCommission)


# --- Integration test with Engine ---


class BuyAndHoldStrategy(Strategy):
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def next(self, ctx: "BacktestContext") -> None:
        if ctx.bar_index == 0:
            ctx.portfolio.order("asset", 10.0)


class TestEngineCommissionModel:
    def _make_data(self) -> pl.DataFrame:
        return pl.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})

    def test_engine_with_maker_taker(self) -> None:
        data = self._make_data()
        engine = Engine(
            strategy=BuyAndHoldStrategy(),
            data=data,
            commission=MakerTakerCommission(maker_rate=0.0001, taker_rate=0.001),
            warmup=0,
        )
        results = engine.run()
        assert results["final_equity"] > 0

        # Commission should be taker rate: 10 * 100 * 0.001 = 1.0
        assert engine.portfolio is not None
        orders = engine.portfolio.get_orders()
        filled = [o for o in orders if o.status.name == "FILLED"]
        assert len(filled) == 1
        assert filled[0].commission_paid == pytest.approx(1.0)

    def test_engine_with_tiered(self) -> None:
        data = self._make_data()
        engine = Engine(
            strategy=BuyAndHoldStrategy(),
            data=data,
            commission=TieredCommission(tiers=[(0, 0.002), (50_000, 0.001)]),
            warmup=0,
        )
        engine.run()
        # Commission: 10 * 100 * 0.002 = 2.0
        assert engine.portfolio is not None
        orders = engine.portfolio.get_orders()
        filled = [o for o in orders if o.status.name == "FILLED"]
        assert filled[0].commission_paid == pytest.approx(2.0)

    def test_engine_with_custom(self) -> None:
        data = self._make_data()
        engine = Engine(
            strategy=BuyAndHoldStrategy(),
            data=data,
            commission=CustomCommission(lambda s, p, r: max(5.0, s * p * 0.001)),
            warmup=0,
        )
        engine.run()
        # min commission = 5.0 (since 10*100*0.001=1.0 < 5.0)
        assert engine.portfolio is not None
        orders = engine.portfolio.get_orders()
        filled = [o for o in orders if o.status.name == "FILLED"]
        assert filled[0].commission_paid == pytest.approx(5.0)

    def test_legacy_float_still_works(self) -> None:
        data = self._make_data()
        engine = Engine(strategy=BuyAndHoldStrategy(), data=data, commission=0.001, warmup=0)
        engine.run()
        assert engine.portfolio is not None
        orders = engine.portfolio.get_orders()
        filled = [o for o in orders if o.status.name == "FILLED"]
        assert filled[0].commission_paid == pytest.approx(1.0)

    def test_legacy_tuple_still_works(self) -> None:
        data = self._make_data()
        engine = Engine(strategy=BuyAndHoldStrategy(), data=data, commission=(5.0, 0.001), warmup=0)
        engine.run()
        assert engine.portfolio is not None
        orders = engine.portfolio.get_orders()
        filled = [o for o in orders if o.status.name == "FILLED"]
        # 5.0 + 10*100*0.001 = 6.0
        assert filled[0].commission_paid == pytest.approx(6.0)
