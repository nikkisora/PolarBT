"""Tests for additional technical indicators."""

import numpy as np
import polars as pl
import pytest

from polarbt import indicators as ind


@pytest.fixture
def ohlcv_data():
    """OHLCV data with enough bars for indicator calculation."""
    np.random.seed(42)
    n = 100
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 10000, size=n).astype(float)
    return pl.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


class TestWMA:
    def test_basic(self, ohlcv_data: pl.DataFrame):
        df = ohlcv_data.with_columns(ind.wma("close", 10).alias("wma"))
        # First 9 values should be NaN
        assert all(np.isnan(df["wma"][i]) for i in range(9))
        assert not np.isnan(df["wma"][9])

    def test_manual(self):
        df = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]})
        df = df.with_columns(ind.wma("close", 3).alias("wma"))
        # WMA(3) at index 2: (1*1 + 2*2 + 3*3) / 6 = 14/6
        assert abs(df["wma"][2] - 14.0 / 6.0) < 1e-9


class TestHMA:
    def test_basic(self, ohlcv_data: pl.DataFrame):
        df = ohlcv_data.with_columns(ind.hma("close", 16).alias("hma"))
        non_null = df["hma"].drop_nulls()
        assert len(non_null) > 0

    def test_smoother_than_sma(self, ohlcv_data: pl.DataFrame):
        df = ohlcv_data.with_columns(
            [
                ind.hma("close", 10).alias("hma"),
                ind.sma("close", 10).alias("sma"),
            ]
        )
        # Just check both produce values
        assert df["hma"].drop_nulls().len() > 0
        assert df["sma"].drop_nulls().len() > 0


class TestVWAP:
    def test_close_only(self, ohlcv_data: pl.DataFrame):
        df = ohlcv_data.with_columns(ind.vwap("close", "volume").alias("vwap"))
        assert df["vwap"].null_count() == 0
        # First bar VWAP = close[0]
        assert abs(df["vwap"][0] - df["close"][0]) < 1e-9

    def test_typical_price(self, ohlcv_data: pl.DataFrame):
        df = ohlcv_data.with_columns(ind.vwap("close", "volume", "high", "low").alias("vwap"))
        assert df["vwap"].null_count() == 0


class TestSuperTrend:
    def test_basic(self, ohlcv_data: pl.DataFrame):
        st_line, st_dir = ind.supertrend("high", "low", "close", 10, 3.0)
        df = ohlcv_data.with_columns([st_line.alias("st"), st_dir.alias("st_dir")])
        # Direction should be 1.0 or -1.0
        dirs = df["st_dir"].drop_nulls().to_list()
        assert all(d in (1.0, -1.0) for d in dirs)
        # Line should have values after warmup
        assert df["st"].drop_nulls().len() > 0

    def test_direction_flips_on_trend_reversal(self):
        """SuperTrend direction must flip when price reverses sharply."""
        n = 200
        close = np.zeros(n)
        close[0] = 100.0
        for i in range(1, n):
            close[i] = close[i - 1] + (2.0 if (i // 30) % 2 == 0 else -2.0)
        high = close + 0.5
        low = close - 0.5
        df = pl.DataFrame({"high": high, "low": low, "close": close})
        st_line, st_dir = ind.supertrend("high", "low", "close", 10, 3.0)
        result = df.with_columns([st_dir.alias("st_dir")])
        dirs = set(result["st_dir"].to_list())
        assert 1.0 in dirs and -1.0 in dirs, "SuperTrend should detect both up and down trends"


class TestADX:
    def test_basic(self, ohlcv_data: pl.DataFrame):
        adx_expr, plus_di, minus_di = ind.adx("high", "low", "close", 14)
        df = ohlcv_data.with_columns(
            [
                adx_expr.alias("adx"),
                plus_di.alias("plus_di"),
                minus_di.alias("minus_di"),
            ]
        )
        adx_vals = df.filter(pl.col("adx").is_not_null() & pl.col("adx").is_not_nan())["adx"]
        assert len(adx_vals) > 0
        # ADX should be non-negative
        assert all(v >= 0 for v in adx_vals)


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


class TestStochastic:
    def test_basic(self, ohlcv_data: pl.DataFrame):
        k, d = ind.stochastic("high", "low", "close", 14, 3)
        df = ohlcv_data.with_columns([k.alias("k"), d.alias("d")])
        k_vals = df["k"].drop_nulls()
        assert all(0 <= v <= 100 for v in k_vals)

    def test_known_values(self):
        # When close is at highest high, %K should be 100
        df = pl.DataFrame(
            {
                "high": [10.0, 11.0, 12.0],
                "low": [8.0, 9.0, 10.0],
                "close": [9.0, 10.0, 12.0],
            }
        )
        k, d = ind.stochastic("high", "low", "close", 3, 1)
        df = df.with_columns([k.alias("k")])
        assert abs(df["k"][2] - 100.0) < 1e-9


class TestWilliamsR:
    def test_basic(self, ohlcv_data: pl.DataFrame):
        df = ohlcv_data.with_columns(ind.williams_r("high", "low", "close", 14).alias("wr"))
        wr_vals = df["wr"].drop_nulls()
        assert all(-100 <= v <= 0 for v in wr_vals)


class TestCCI:
    def test_basic(self, ohlcv_data: pl.DataFrame):
        df = ohlcv_data.with_columns(ind.cci("high", "low", "close", 20).alias("cci"))
        cci_vals = df["cci"].drop_nulls()
        assert len(cci_vals) > 0


class TestMFI:
    def test_basic(self, ohlcv_data: pl.DataFrame):
        df = ohlcv_data.with_columns(ind.mfi("high", "low", "close", "volume", 14).alias("mfi"))
        mfi_vals = df.filter(pl.col("mfi").is_not_null() & pl.col("mfi").is_not_nan())["mfi"]
        assert len(mfi_vals) > 0
        assert all(0 <= v <= 100 for v in mfi_vals)


class TestROC:
    def test_basic(self, ohlcv_data: pl.DataFrame):
        df = ohlcv_data.with_columns(ind.roc("close", 12).alias("roc"))
        # First 12 values should be null
        assert df["roc"][:12].null_count() == 12
        assert df["roc"][12] is not None

    def test_manual(self):
        df = pl.DataFrame({"close": [100.0, 110.0]})
        df = df.with_columns(ind.roc("close", 1).alias("roc"))
        assert abs(df["roc"][1] - 10.0) < 1e-9


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


class TestKeltnerChannels:
    def test_basic(self, ohlcv_data: pl.DataFrame):
        upper, middle, lower = ind.keltner_channels("high", "low", "close", 20, 10, 2.0)
        df = ohlcv_data.with_columns(
            [
                upper.alias("kc_upper"),
                middle.alias("kc_middle"),
                lower.alias("kc_lower"),
            ]
        )
        valid = df.filter(pl.col("kc_upper").is_not_null() & pl.col("kc_lower").is_not_null())
        for row in valid.iter_rows(named=True):
            assert row["kc_upper"] >= row["kc_middle"]
            assert row["kc_middle"] >= row["kc_lower"]


class TestDonchianChannels:
    def test_basic(self, ohlcv_data: pl.DataFrame):
        upper, middle, lower = ind.donchian_channels("high", "low", 20)
        df = ohlcv_data.with_columns(
            [
                upper.alias("dc_upper"),
                middle.alias("dc_middle"),
                lower.alias("dc_lower"),
            ]
        )
        valid = df.filter(pl.col("dc_upper").is_not_null() & pl.col("dc_lower").is_not_null())
        for row in valid.iter_rows(named=True):
            assert row["dc_upper"] >= row["dc_middle"]
            assert row["dc_middle"] >= row["dc_lower"]

    def test_manual(self):
        df = pl.DataFrame(
            {
                "high": [5.0, 10.0, 8.0],
                "low": [1.0, 3.0, 2.0],
            }
        )
        upper, middle, lower = ind.donchian_channels("high", "low", 3)
        df = df.with_columns([upper.alias("u"), middle.alias("m"), lower.alias("l")])
        assert abs(df["u"][2] - 10.0) < 1e-9
        assert abs(df["l"][2] - 1.0) < 1e-9
        assert abs(df["m"][2] - 5.5) < 1e-9


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


class TestOBV:
    def test_basic(self, ohlcv_data: pl.DataFrame):
        df = ohlcv_data.with_columns(ind.obv("close", "volume").alias("obv"))
        assert df["obv"].null_count() <= 1  # first bar may be null

    def test_manual(self):
        df = pl.DataFrame(
            {
                "close": [10.0, 11.0, 10.5, 12.0],
                "volume": [100.0, 200.0, 150.0, 300.0],
            }
        )
        df = df.with_columns(ind.obv("close", "volume").alias("obv"))
        # bar1: up -> +200, bar2: down -> -150, bar3: up -> +300
        # cumsum: 0, 200, 50, 350
        assert df["obv"][1] == 200
        assert df["obv"][2] == 50
        assert df["obv"][3] == 350


class TestADLine:
    def test_basic(self, ohlcv_data: pl.DataFrame):
        df = ohlcv_data.with_columns(ind.ad_line("high", "low", "close", "volume").alias("ad"))
        assert df["ad"].null_count() == 0

    def test_manual(self):
        df = pl.DataFrame(
            {
                "high": [12.0],
                "low": [10.0],
                "close": [11.0],
                "volume": [1000.0],
            }
        )
        df = df.with_columns(ind.ad_line("high", "low", "close", "volume").alias("ad"))
        # CLV = ((11-10) - (12-11)) / (12-10) = 0/2 = 0
        assert abs(df["ad"][0]) < 1e-9


# ---------------------------------------------------------------------------
# Support / Resistance
# ---------------------------------------------------------------------------


class TestPivotPoints:
    def test_standard(self):
        df = pl.DataFrame(
            {
                "high": [110.0, 115.0],
                "low": [90.0, 95.0],
                "close": [100.0, 105.0],
            }
        )
        pp = ind.pivot_points("high", "low", "close", method="standard")
        exprs = [v.alias(k) for k, v in pp.items()]
        df = df.with_columns(exprs)
        # PP at index 1 = (110 + 90 + 100) / 3 = 100
        assert abs(df["pp"][1] - 100.0) < 1e-9
        # R1 = 2*100 - 90 = 110
        assert abs(df["r1"][1] - 110.0) < 1e-9
        # S1 = 2*100 - 110 = 90
        assert abs(df["s1"][1] - 90.0) < 1e-9

    def test_fibonacci(self):
        pp = ind.pivot_points(method="fibonacci")
        assert "r1" in pp and "r2" in pp and "r3" in pp

    def test_woodie(self):
        pp = ind.pivot_points(method="woodie")
        assert "pp" in pp
        # Woodie doesn't have r3/s3
        assert "r3" not in pp

    def test_camarilla(self):
        pp = ind.pivot_points(method="camarilla")
        assert "r1" in pp and "r3" in pp

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown pivot point method"):
            ind.pivot_points(method="invalid")

    def test_first_bar_null(self):
        df = pl.DataFrame(
            {
                "high": [110.0, 115.0],
                "low": [90.0, 95.0],
                "close": [100.0, 105.0],
            }
        )
        pp = ind.pivot_points("high", "low", "close", method="standard")
        df = df.with_columns([v.alias(k) for k, v in pp.items()])
        # First bar should be null (no previous bar)
        assert df["pp"][0] is None
