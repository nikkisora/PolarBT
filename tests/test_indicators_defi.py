"""Tests for DeFi/memecoin indicators (Phase 3)."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from polarbt import indicators_defi as defi
from polarbt.data.trades import aggregate_trades

REAL_DATA_PATH = Path("/mnt/d/downloads/02_datasets/pumpfun_standard.parquet")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bar_df() -> pl.DataFrame:
    """Multi-symbol aggregated bar DataFrame with all relevant columns."""
    base = datetime(2025, 1, 1)
    n = 20
    rows: list[dict] = []
    for sym, base_price, base_vol in [("AAA", 10.0, 100.0), ("BBB", 50.0, 500.0)]:
        for i in range(n):
            price = base_price * (1 + 0.05 * i)
            vol = base_vol + 10 * i
            buy_pct = 0.6 if i < 15 else 0.2
            rows.append(
                {
                    "timestamp": base + timedelta(hours=i),
                    "symbol": sym,
                    "open": price * 0.99,
                    "high": price * 1.02,
                    "low": price * 0.97,
                    "close": price,
                    "volume": vol,
                    "trades": 20 + i,
                    "buy_volume": vol * buy_pct,
                    "sell_volume": vol * (1 - buy_pct),
                    "unique_traders": 5 + i,
                    "pool_reserve_last": 10000.0 - i * 100,
                    "vwap": price * 1.001,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "timestamp": pl.Datetime("us"),
            "symbol": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "trades": pl.Int64,
            "buy_volume": pl.Float64,
            "sell_volume": pl.Float64,
            "unique_traders": pl.Int64,
            "pool_reserve_last": pl.Float64,
            "vwap": pl.Float64,
        },
    )


@pytest.fixture
def real_bars() -> pl.DataFrame | None:
    """Aggregate a slice of real pumpfun data into 1m bars."""
    if not REAL_DATA_PATH.exists():
        pytest.skip("Real parquet data not available")
    df = pl.read_parquet(REAL_DATA_PATH)
    top_symbols = df.group_by("symbol").len().sort("len", descending=True).head(3)["symbol"].to_list()
    subset = df.filter(pl.col("symbol").is_in(top_symbols)).sort("symbol", "timestamp")
    return aggregate_trades(subset, "5m")


# ===========================================================================
# Token activity indicators
# ===========================================================================


class TestTokenAge:
    def test_increments_per_symbol(self, bar_df: pl.DataFrame) -> None:
        result = bar_df.with_columns(defi.token_age().alias("age"))
        for sym in ["AAA", "BBB"]:
            ages = result.filter(pl.col("symbol") == sym)["age"].to_list()
            assert ages == list(range(1, 21))

    def test_custom_symbol_col(self) -> None:
        df = pl.DataFrame(
            {
                "ts": [1, 2, 3],
                "tkn": ["X", "X", "X"],
                "close": [1.0, 2.0, 3.0],
            }
        )
        result = df.with_columns(defi.token_age(symbol_col="tkn").alias("age"))
        assert result["age"].to_list() == [1, 2, 3]


class TestBuySellRatio:
    def test_basic_ratio(self) -> None:
        df = pl.DataFrame({"buy_volume": [60.0, 40.0], "sell_volume": [40.0, 60.0]})
        result = df.with_columns(defi.buy_sell_ratio().alias("ratio"))
        assert result["ratio"][0] == pytest.approx(0.6)
        assert result["ratio"][1] == pytest.approx(0.4)

    def test_zero_volume_returns_null(self) -> None:
        df = pl.DataFrame({"buy_volume": [0.0], "sell_volume": [0.0]})
        result = df.with_columns(defi.buy_sell_ratio().alias("ratio"))
        assert result["ratio"][0] is None

    def test_all_buy(self) -> None:
        df = pl.DataFrame({"buy_volume": [100.0], "sell_volume": [0.0]})
        result = df.with_columns(defi.buy_sell_ratio().alias("ratio"))
        assert result["ratio"][0] == pytest.approx(1.0)

    def test_custom_columns(self) -> None:
        df = pl.DataFrame({"bv": [30.0], "sv": [70.0]})
        result = df.with_columns(defi.buy_sell_ratio("bv", "sv").alias("ratio"))
        assert result["ratio"][0] == pytest.approx(0.3)


class TestNetFlow:
    def test_positive_flow(self) -> None:
        df = pl.DataFrame({"buy_volume": [100.0], "sell_volume": [40.0]})
        result = df.with_columns(defi.net_flow().alias("flow"))
        assert result["flow"][0] == pytest.approx(60.0)

    def test_negative_flow(self) -> None:
        df = pl.DataFrame({"buy_volume": [20.0], "sell_volume": [80.0]})
        result = df.with_columns(defi.net_flow().alias("flow"))
        assert result["flow"][0] == pytest.approx(-60.0)


class TestTradeIntensity:
    def test_spike_detection(self) -> None:
        # Flat trade count with a spike
        trades = [10] * 15 + [100] + [10] * 4
        df = pl.DataFrame({"trades": trades})
        result = df.with_columns(defi.trade_intensity("trades", window=10).alias("intensity"))
        intensities = result["intensity"].to_list()
        # The spike bar should have the highest intensity
        spike_idx = 15
        assert intensities[spike_idx] == max(i for i in intensities if i is not None and i == i)  # skip NaN

    def test_constant_returns_zero(self) -> None:
        df = pl.DataFrame({"trades": [10] * 20})
        result = df.with_columns(defi.trade_intensity("trades", window=5).alias("intensity"))
        # After warmup, constant series should have intensity ~0
        vals = result["intensity"].to_list()[5:]
        assert all(v is not None and abs(v) < 1e-10 for v in vals)


class TestUniqueTraderGrowth:
    def test_growing_traders(self) -> None:
        df = pl.DataFrame({"unique_traders": list(range(1, 21))})
        result = df.with_columns(defi.unique_trader_growth("unique_traders", window=3).alias("growth"))
        # Growth should be positive for linearly increasing traders
        vals = [v for v in result["growth"].to_list() if v is not None and v == v]
        assert all(v > 0 for v in vals)


# ===========================================================================
# Liquidity indicators
# ===========================================================================


class TestPoolDepth:
    def test_passthrough(self) -> None:
        df = pl.DataFrame({"pool_reserve_last": [1000.0, 2000.0]})
        result = df.with_columns(defi.pool_depth().alias("depth"))
        assert result["depth"].to_list() == [1000.0, 2000.0]


class TestPriceImpactEstimate:
    def test_small_trade(self) -> None:
        df = pl.DataFrame({"pool_reserve_last": [10000.0]})
        result = df.with_columns(defi.price_impact_estimate(100.0).alias("impact"))
        # 100 / (10000 + 100) = 0.0099...
        assert result["impact"][0] == pytest.approx(100.0 / 10100.0)

    def test_large_trade_high_impact(self) -> None:
        df = pl.DataFrame({"pool_reserve_last": [1000.0]})
        result = df.with_columns(defi.price_impact_estimate(1000.0).alias("impact"))
        # 1000 / (1000 + 1000) = 0.5
        assert result["impact"][0] == pytest.approx(0.5)

    def test_impact_increases_with_size(self) -> None:
        df = pl.DataFrame({"pool_reserve_last": [10000.0, 10000.0, 10000.0]})
        small = df.with_columns(defi.price_impact_estimate(10.0).alias("impact"))["impact"][0]
        medium = df.with_columns(defi.price_impact_estimate(1000.0).alias("impact"))["impact"][0]
        large = df.with_columns(defi.price_impact_estimate(5000.0).alias("impact"))["impact"][0]
        assert small < medium < large


class TestLiquidityRatio:
    def test_basic(self) -> None:
        df = pl.DataFrame({"volume": [500.0], "pool_reserve_last": [10000.0]})
        result = df.with_columns(defi.liquidity_ratio().alias("lr"))
        assert result["lr"][0] == pytest.approx(0.05)

    def test_zero_reserve_returns_null(self) -> None:
        df = pl.DataFrame({"volume": [500.0], "pool_reserve_last": [0.0]})
        result = df.with_columns(defi.liquidity_ratio().alias("lr"))
        assert result["lr"][0] is None

    def test_high_ratio_means_thin_pool(self) -> None:
        df = pl.DataFrame({"volume": [5000.0], "pool_reserve_last": [1000.0]})
        result = df.with_columns(defi.liquidity_ratio().alias("lr"))
        assert result["lr"][0] == pytest.approx(5.0)


# ===========================================================================
# Momentum indicators
# ===========================================================================


class TestLaunchVelocity:
    def test_positive_launch(self) -> None:
        # Price doubles over 5 bars
        prices = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        df = pl.DataFrame({"close": prices})
        result = df.with_columns(defi.launch_velocity("close", window=5).alias("velocity"))
        # Last bar: (2.0 - 1.0) / 1.0 = 1.0
        assert result["velocity"][5] == pytest.approx(1.0)

    def test_negative_velocity(self) -> None:
        prices = [10.0, 8.0, 6.0, 4.0]
        df = pl.DataFrame({"close": prices})
        result = df.with_columns(defi.launch_velocity("close", window=1).alias("velocity"))
        assert all(v < 0 for v in result["velocity"].to_list() if v is not None and v == v)


class TestPumpDetector:
    def test_detects_pump(self) -> None:
        # Normal bars then a pump bar
        prices = [10.0] * 20 + [20.0]
        volumes = [100.0] * 20 + [1000.0]
        df = pl.DataFrame({"close": prices, "volume": volumes})
        result = df.with_columns(defi.pump_detector(price_std=2.0, volume_std=2.0, window=15).alias("is_pump"))
        # The last bar should be flagged
        assert result["is_pump"][-1] is True

    def test_no_false_positive_on_flat(self) -> None:
        df = pl.DataFrame({"close": [10.0] * 30, "volume": [100.0] * 30})
        result = df.with_columns(defi.pump_detector(price_std=2.0, volume_std=2.0, window=10).alias("is_pump"))
        # No bars should be flagged (all constant)
        flagged = result.filter(pl.col("is_pump") == True).height  # noqa: E712
        assert flagged == 0

    def test_price_spike_only_not_flagged(self) -> None:
        prices = [10.0] * 20 + [20.0]
        volumes = [100.0] * 21  # volume stays flat
        df = pl.DataFrame({"close": prices, "volume": volumes})
        result = df.with_columns(defi.pump_detector(price_std=2.0, volume_std=2.0, window=15).alias("is_pump"))
        assert result["is_pump"][-1] is not True


class TestRugPullDetector:
    def test_detects_rug(self) -> None:
        # Price drops 50%, sell volume is 90%
        prices = [10.0, 5.0]
        volumes = [100.0, 100.0]
        sell_volumes = [50.0, 90.0]
        df = pl.DataFrame(
            {
                "close": prices,
                "volume": volumes,
                "sell_volume": sell_volumes,
            }
        )
        result = df.with_columns(defi.rug_pull_detector(price_drop=-0.3, sell_ratio=0.8, window=1).alias("is_rug"))
        assert result["is_rug"][-1] is True

    def test_no_rug_on_mild_drop(self) -> None:
        prices = [10.0, 9.0]  # -10%
        volumes = [100.0, 100.0]
        sell_volumes = [50.0, 90.0]
        df = pl.DataFrame(
            {
                "close": prices,
                "volume": volumes,
                "sell_volume": sell_volumes,
            }
        )
        result = df.with_columns(defi.rug_pull_detector(price_drop=-0.3, sell_ratio=0.8, window=1).alias("is_rug"))
        assert result["is_rug"][-1] is not True

    def test_no_rug_on_low_sell_ratio(self) -> None:
        prices = [10.0, 5.0]  # -50%
        volumes = [100.0, 100.0]
        sell_volumes = [50.0, 50.0]  # only 50%
        df = pl.DataFrame(
            {
                "close": prices,
                "volume": volumes,
                "sell_volume": sell_volumes,
            }
        )
        result = df.with_columns(defi.rug_pull_detector(price_drop=-0.3, sell_ratio=0.8, window=1).alias("is_rug"))
        assert result["is_rug"][-1] is not True


# ===========================================================================
# Integration with aggregated data
# ===========================================================================


class TestWithAggregatedData:
    def test_all_indicators_on_bar_df(self, bar_df: pl.DataFrame) -> None:
        """All indicators should compute without error on properly shaped bar data."""
        result = bar_df.with_columns(
            defi.token_age().alias("age"),
            defi.buy_sell_ratio().over("symbol").alias("bs_ratio"),
            defi.net_flow().alias("flow"),
            defi.trade_intensity(window=5).over("symbol").alias("intensity"),
            defi.unique_trader_growth(window=3).over("symbol").alias("trader_growth"),
            defi.pool_depth().alias("depth"),
            defi.price_impact_estimate(100.0).alias("impact"),
            defi.liquidity_ratio().alias("liq_ratio"),
            defi.launch_velocity(window=3).over("symbol").alias("velocity"),
            defi.pump_detector(window=10).over("symbol").alias("is_pump"),
            defi.rug_pull_detector(window=1).over("symbol").alias("is_rug"),
        )
        assert result.height == bar_df.height
        expected_cols = {
            "age",
            "bs_ratio",
            "flow",
            "intensity",
            "trader_growth",
            "depth",
            "impact",
            "liq_ratio",
            "velocity",
            "is_pump",
            "is_rug",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_real_data_indicators(self, real_bars: pl.DataFrame) -> None:
        """Indicators should compute on real aggregated data."""
        result = real_bars.with_columns(
            defi.token_age().alias("age"),
            defi.buy_sell_ratio().over("symbol").alias("bs_ratio"),
            defi.net_flow().alias("flow"),
            defi.trade_intensity(window=10).over("symbol").alias("intensity"),
            defi.launch_velocity(window=5).over("symbol").alias("velocity"),
            defi.pump_detector(window=20).over("symbol").alias("is_pump"),
            defi.rug_pull_detector(window=1).over("symbol").alias("is_rug"),
        )
        assert result.height == real_bars.height
        # buy_sell_ratio should be in [0, 1] where not null
        ratios = result.filter(pl.col("bs_ratio").is_not_null())["bs_ratio"]
        assert ratios.min() >= 0.0  # type: ignore[operator]
        assert ratios.max() <= 1.0  # type: ignore[operator]
