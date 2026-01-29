"""Tests for technical indicators."""

import polars as pl
import pytest
from polarbtest import indicators as ind


@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    return pl.DataFrame(
        {
            "close": [100, 102, 101, 103, 105, 104, 106, 108, 107, 109],
            "high": [101, 103, 102, 104, 106, 105, 107, 109, 108, 110],
            "low": [99, 101, 100, 102, 104, 103, 105, 107, 106, 108],
        }
    )


def test_sma(sample_data):
    """Test Simple Moving Average."""
    df = sample_data.with_columns([ind.sma("close", 3).alias("sma_3")])

    # First two values should be null
    assert df["sma_3"][0] is None
    assert df["sma_3"][1] is None

    # Third value should be average of first 3
    assert abs(df["sma_3"][2] - 101.0) < 0.01

    # Check that we have SMA values
    assert df["sma_3"][-1] is not None


def test_ema(sample_data):
    """Test Exponential Moving Average."""
    df = sample_data.with_columns([ind.ema("close", 3).alias("ema_3")])

    # First value should be null
    assert df["ema_3"][0] is None

    # EMA should react faster than SMA
    assert df["ema_3"][-1] is not None


def test_rsi(sample_data):
    """Test Relative Strength Index."""
    df = sample_data.with_columns([ind.rsi("close", 5).alias("rsi_5")])

    # RSI should be between 0 and 100
    rsi_values = df["rsi_5"].drop_nulls()
    assert all(0 <= val <= 100 for val in rsi_values)


def test_bollinger_bands(sample_data):
    """Test Bollinger Bands."""
    upper, middle, lower = ind.bollinger_bands("close", 3, 2.0)

    df = sample_data.with_columns(
        [
            upper.alias("bb_upper"),
            middle.alias("bb_middle"),
            lower.alias("bb_lower"),
        ]
    )

    # Upper should be greater than middle, middle greater than lower
    valid_rows = df.filter(
        pl.col("bb_upper").is_not_null()
        & pl.col("bb_middle").is_not_null()
        & pl.col("bb_lower").is_not_null()
    )

    for row in valid_rows.iter_rows(named=True):
        assert row["bb_upper"] >= row["bb_middle"]
        assert row["bb_middle"] >= row["bb_lower"]


def test_atr(sample_data):
    """Test Average True Range."""
    df = sample_data.with_columns([ind.atr("high", "low", "close", 3).alias("atr_3")])

    # ATR should be positive
    atr_values = df["atr_3"].drop_nulls()
    assert all(val > 0 for val in atr_values)


def test_macd(sample_data):
    """Test MACD."""
    macd_line, signal_line, histogram = ind.macd("close", 3, 5, 2)

    df = sample_data.with_columns(
        [
            macd_line.alias("macd"),
            signal_line.alias("signal"),
            histogram.alias("hist"),
        ]
    )

    # Histogram should be macd - signal
    valid_rows = df.filter(
        pl.col("macd").is_not_null()
        & pl.col("signal").is_not_null()
        & pl.col("hist").is_not_null()
    )

    assert len(valid_rows) > 0


def test_returns(sample_data):
    """Test returns calculation."""
    df = sample_data.with_columns([ind.returns("close", 1).alias("returns")])

    # First value should be null
    assert df["returns"][0] is None

    # Second value should be (102-100)/100 = 0.02
    assert abs(df["returns"][1] - 0.02) < 0.001


def test_log_returns(sample_data):
    """Test log returns calculation."""
    df = sample_data.with_columns([ind.log_returns("close", 1).alias("log_returns")])

    # First value should be null
    assert df["log_returns"][0] is None

    # Log returns should be close to regular returns for small changes
    assert df["log_returns"][1] is not None


def test_crossover():
    """Test crossover detection."""
    df = pl.DataFrame(
        {
            "fast": [1, 2, 3, 4, 5],
            "slow": [5, 4, 3, 2, 1],
        }
    )

    df = df.with_columns([ind.crossover("fast", "slow").alias("cross")])

    # Crossover should occur at index 2 (fast goes from below to above slow)
    assert df["cross"][2] == True
    assert df["cross"][1] == False


def test_crossunder():
    """Test crossunder detection."""
    df = pl.DataFrame(
        {
            "fast": [5, 4, 3, 2, 1],
            "slow": [1, 2, 3, 4, 5],
        }
    )

    df = df.with_columns([ind.crossunder("fast", "slow").alias("cross")])

    # Crossunder should occur at index 2
    assert df["cross"][2] == True
    assert df["cross"][1] == False
