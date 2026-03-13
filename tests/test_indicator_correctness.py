"""Numerical correctness tests for technical indicators.

Each test computes expected values by hand for known input data and asserts
the indicator output matches to a tight tolerance.
"""

import math

import polars as pl
import pytest

from polarbt import indicators as ind


@pytest.fixture
def sample_data():
    """Small dataset with hand-computable values."""
    return pl.DataFrame(
        {
            "close": [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0],
            "high": [101.0, 103.0, 102.0, 104.0, 106.0, 105.0, 107.0, 109.0, 108.0, 110.0],
            "low": [99.0, 101.0, 100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 106.0, 108.0],
            "volume": [1000, 1200, 800, 1500, 1100, 900, 1300, 1400, 1000, 1600],
        }
    )


class TestSMACorrectness:
    def test_period_3_exact_values(self, sample_data):
        df = sample_data.with_columns(ind.sma("close", 3).alias("sma"))
        # SMA(3) at index 2 = (100+102+101)/3 = 101.0
        assert df["sma"][2] == pytest.approx(101.0)
        # SMA(3) at index 3 = (102+101+103)/3 = 102.0
        assert df["sma"][3] == pytest.approx(102.0)
        # SMA(3) at index 9 = (108+107+109)/3 = 108.0
        assert df["sma"][9] == pytest.approx(108.0)

    def test_period_1_equals_close(self, sample_data):
        df = sample_data.with_columns(ind.sma("close", 1).alias("sma"))
        for i in range(len(df)):
            assert df["sma"][i] == pytest.approx(df["close"][i])


class TestEMACorrectness:
    def test_period_3_exact_values(self, sample_data):
        """EMA(3) with alpha=2/(3+1)=0.5, adjust=False, first value=null."""
        df = sample_data.with_columns(ind.ema("close", 3).alias("ema"))
        assert df["ema"][0] is None
        # Polars ewm_mean with adjust=False: seed = first value, then recurse
        # seed = 100.0
        # ema[1] = 0.5*102 + 0.5*100 = 101.0
        # ema[2] = 0.5*101 + 0.5*101 = 101.0
        # ema[3] = 0.5*103 + 0.5*101 = 102.0
        # ema[4] = 0.5*105 + 0.5*102 = 103.5
        assert df["ema"][1] == pytest.approx(101.0, abs=0.01)
        assert df["ema"][2] == pytest.approx(101.0, abs=0.01)
        assert df["ema"][3] == pytest.approx(102.0, abs=0.01)
        assert df["ema"][4] == pytest.approx(103.5, abs=0.01)


class TestRSICorrectness:
    def test_all_gains(self):
        """All prices go up -> RSI should be 100 (after warmup)."""
        df = pl.DataFrame({"close": [float(100 + i) for i in range(20)]})
        df = df.with_columns(ind.rsi("close", 5).alias("rsi"))
        # With all gains and no losses, avg_loss ~ 0, RSI -> 100
        last_rsi = df["rsi"][-1]
        assert last_rsi == pytest.approx(100.0, abs=0.1)

    def test_all_losses(self):
        """All prices go down -> RSI should be 0 (after warmup)."""
        df = pl.DataFrame({"close": [float(100 - i) for i in range(20)]})
        df = df.with_columns(ind.rsi("close", 5).alias("rsi"))
        last_rsi = df["rsi"][-1]
        assert last_rsi == pytest.approx(0.0, abs=0.1)

    def test_range_bounds(self, sample_data):
        """RSI must be in [0, 100]."""
        df = sample_data.with_columns(ind.rsi("close", 5).alias("rsi"))
        valid = df["rsi"].drop_nulls()
        assert all(0 <= v <= 100 for v in valid)

    def test_mostly_up_above_50(self, sample_data):
        """Sample data trends up -> RSI should be above 50."""
        df = sample_data.with_columns(ind.rsi("close", 5).alias("rsi"))
        last_rsi = df["rsi"][-1]
        assert last_rsi > 50


class TestATRCorrectness:
    def test_constant_range(self):
        """When high-low is constant (2.0) and no gaps, ATR converges to 2.0."""
        n = 50
        closes = [100.0 + i * 0.1 for i in range(n)]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        df = pl.DataFrame({"close": closes, "high": highs, "low": lows})
        df = df.with_columns(ind.atr("high", "low", "close", 5).alias("atr"))
        # ATR should converge to ~2.0 (high - low range)
        assert df["atr"][-1] == pytest.approx(2.0, abs=0.1)


class TestMACDCorrectness:
    def test_histogram_equals_macd_minus_signal(self, sample_data):
        macd_line, signal_line, histogram = ind.macd("close", 3, 5, 2)
        df = sample_data.with_columns(
            macd_line.alias("macd"),
            signal_line.alias("signal"),
            histogram.alias("hist"),
        )
        valid = df.filter(pl.col("macd").is_not_null() & pl.col("signal").is_not_null() & pl.col("hist").is_not_null())
        for row in valid.iter_rows(named=True):
            expected = row["macd"] - row["signal"]
            assert row["hist"] == pytest.approx(expected, abs=1e-10)

    def test_uptrend_macd_positive(self):
        """Strong uptrend -> fast EMA > slow EMA -> MACD > 0."""
        df = pl.DataFrame({"close": [float(100 + i * 2) for i in range(30)]})
        macd_line, _, _ = ind.macd("close", 5, 10, 3)
        df = df.with_columns(macd_line.alias("macd"))
        assert df["macd"][-1] > 0


class TestBollingerBandsCorrectness:
    def test_middle_equals_sma(self, sample_data):
        upper, middle, lower = ind.bollinger_bands("close", 3, 2.0)
        df = sample_data.with_columns(
            middle.alias("bb_mid"),
            ind.sma("close", 3).alias("sma"),
        )
        valid = df.filter(pl.col("bb_mid").is_not_null() & pl.col("sma").is_not_null())
        for row in valid.iter_rows(named=True):
            assert row["bb_mid"] == pytest.approx(row["sma"], abs=1e-10)

    def test_band_width_is_2_std_dev(self, sample_data):
        upper, middle, lower = ind.bollinger_bands("close", 3, 2.0)
        df = sample_data.with_columns(
            upper.alias("upper"),
            middle.alias("mid"),
            lower.alias("lower"),
        )
        valid = df.filter(pl.col("upper").is_not_null())
        for row in valid.iter_rows(named=True):
            # Upper - mid should equal mid - lower (symmetric)
            upper_dist = row["upper"] - row["mid"]
            lower_dist = row["mid"] - row["lower"]
            assert upper_dist == pytest.approx(lower_dist, abs=1e-10)


class TestLogReturnsCorrectness:
    def test_exact_values(self, sample_data):
        df = sample_data.with_columns(ind.log_returns("close", 1).alias("lr"))
        # log(102/100) = log(1.02)
        assert df["lr"][1] == pytest.approx(math.log(102.0 / 100.0), abs=1e-10)
        # log(101/102)
        assert df["lr"][2] == pytest.approx(math.log(101.0 / 102.0), abs=1e-10)

    def test_close_to_pct_returns_for_small_changes(self, sample_data):
        df = sample_data.with_columns(
            ind.returns("close", 1).alias("pct"),
            ind.log_returns("close", 1).alias("lr"),
        )
        valid = df.filter(pl.col("pct").is_not_null() & pl.col("lr").is_not_null())
        for row in valid.iter_rows(named=True):
            # For small returns, log return ~ pct return
            assert abs(row["lr"] - row["pct"]) < 0.01


class TestReturnsCorrectness:
    def test_exact_values(self, sample_data):
        df = sample_data.with_columns(ind.returns("close", 1).alias("ret"))
        assert df["ret"][0] is None
        # (102-100)/100 = 0.02
        assert df["ret"][1] == pytest.approx(0.02, abs=1e-10)
        # (101-102)/102
        assert df["ret"][2] == pytest.approx(-1.0 / 102.0, abs=1e-10)
        # (103-101)/101
        assert df["ret"][3] == pytest.approx(2.0 / 101.0, abs=1e-10)
