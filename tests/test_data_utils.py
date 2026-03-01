"""Tests for polarbtest.data — validation, cleaning, and resampling utilities."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from polarbtest.data import (
    adjust_splits,
    clip_outliers,
    drop_zero_volume,
    fill_gaps,
    resample_ohlcv,
    validate,
    validate_columns,
    validate_dtypes,
    validate_no_negative_prices,
    validate_no_nulls,
    validate_ohlc_integrity,
    validate_timestamps,
)

# --- Fixtures ---


@pytest.fixture
def ohlcv_df() -> pl.DataFrame:
    """Valid OHLCV DataFrame with hourly timestamps."""
    base = datetime(2024, 1, 1)
    n = 10
    return pl.DataFrame(
        {
            "timestamp": [base + timedelta(hours=i) for i in range(n)],
            "open": [100.0 + i for i in range(n)],
            "high": [102.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [101.0 + i for i in range(n)],
            "volume": [1000.0 + i * 100 for i in range(n)],
        }
    )


# ========== Validation Tests ==========


class TestValidateColumns:
    def test_valid_ohlcv(self, ohlcv_df: pl.DataFrame) -> None:
        result = validate_columns(ohlcv_df, ohlcv=True)
        assert result.valid
        assert result.errors == []

    def test_missing_columns(self) -> None:
        df = pl.DataFrame({"close": [1.0, 2.0]})
        result = validate_columns(df, required={"close", "volume"})
        assert not result.valid
        assert any("volume" in e for e in result.errors)

    def test_missing_ohlcv(self) -> None:
        df = pl.DataFrame({"close": [1.0], "open": [1.0]})
        result = validate_columns(df, ohlcv=True)
        assert not result.valid

    def test_no_requirements(self) -> None:
        df = pl.DataFrame({"x": [1]})
        result = validate_columns(df)
        assert result.valid


class TestValidateDtypes:
    def test_valid_dtypes(self, ohlcv_df: pl.DataFrame) -> None:
        result = validate_dtypes(ohlcv_df)
        assert result.valid

    def test_string_price_column(self) -> None:
        df = pl.DataFrame({"close": ["100", "200"]})
        result = validate_dtypes(df)
        assert not result.valid
        assert any("close" in e for e in result.errors)

    def test_timestamp_warning(self) -> None:
        df = pl.DataFrame({"timestamp": [1, 2, 3]})
        result = validate_dtypes(df)
        assert len(result.warnings) > 0


class TestValidateTimestamps:
    def test_sorted_timestamps(self, ohlcv_df: pl.DataFrame) -> None:
        result = validate_timestamps(ohlcv_df)
        assert result.valid

    def test_unsorted(self) -> None:
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 3), datetime(2024, 1, 1), datetime(2024, 1, 2)]})
        result = validate_timestamps(df)
        assert not result.valid
        assert any("not sorted" in e for e in result.errors)

    def test_duplicates(self) -> None:
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 1), datetime(2024, 1, 2)]})
        result = validate_timestamps(df)
        assert not result.valid
        assert any("duplicate" in e for e in result.errors)

    def test_missing_column(self) -> None:
        df = pl.DataFrame({"close": [1.0]})
        result = validate_timestamps(df)
        assert result.valid  # No error, just warning
        assert len(result.warnings) > 0


class TestValidateOhlcIntegrity:
    def test_valid(self, ohlcv_df: pl.DataFrame) -> None:
        result = validate_ohlc_integrity(ohlcv_df)
        assert result.valid

    def test_high_less_than_low(self) -> None:
        df = pl.DataFrame({"open": [100.0], "high": [90.0], "low": [95.0], "close": [92.0]})
        result = validate_ohlc_integrity(df)
        assert not result.valid
        assert any("high < low" in e for e in result.errors)

    def test_high_less_than_open(self) -> None:
        df = pl.DataFrame({"open": [100.0], "high": [99.0], "low": [95.0], "close": [98.0]})
        result = validate_ohlc_integrity(df)
        assert not result.valid

    def test_low_greater_than_close(self) -> None:
        df = pl.DataFrame({"open": [100.0], "high": [105.0], "low": [102.0], "close": [101.0]})
        result = validate_ohlc_integrity(df)
        assert not result.valid

    def test_missing_columns_warning(self) -> None:
        df = pl.DataFrame({"close": [100.0]})
        result = validate_ohlc_integrity(df)
        assert result.valid
        assert len(result.warnings) > 0


class TestValidateNoNulls:
    def test_no_nulls(self, ohlcv_df: pl.DataFrame) -> None:
        result = validate_no_nulls(ohlcv_df)
        assert result.valid

    def test_with_nulls(self) -> None:
        df = pl.DataFrame({"close": [1.0, None, 3.0]})
        result = validate_no_nulls(df, columns=["close"])
        assert not result.valid
        assert any("1 null" in e for e in result.errors)

    def test_missing_column_skipped(self) -> None:
        df = pl.DataFrame({"close": [1.0]})
        result = validate_no_nulls(df, columns=["nonexistent"])
        assert result.valid


class TestValidateNoNegativePrices:
    def test_no_negatives(self, ohlcv_df: pl.DataFrame) -> None:
        result = validate_no_negative_prices(ohlcv_df)
        assert result.valid

    def test_negative_close(self) -> None:
        df = pl.DataFrame({"close": [-1.0, 2.0]})
        result = validate_no_negative_prices(df)
        assert not result.valid


class TestValidateCombined:
    def test_valid_full(self, ohlcv_df: pl.DataFrame) -> None:
        result = validate(ohlcv_df, ohlcv=True)
        assert result.valid
        assert result.errors == []

    def test_multiple_issues(self) -> None:
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 2), datetime(2024, 1, 1)],
                "close": ["bad", "data"],
            }
        )
        result = validate(df)
        assert not result.valid
        assert len(result.errors) >= 2


# ========== Cleaning Tests ==========


class TestFillGaps:
    def test_fills_missing_hours(self) -> None:
        base = datetime(2024, 1, 1)
        df = pl.DataFrame(
            {
                "timestamp": [base, base + timedelta(hours=1), base + timedelta(hours=3)],
                "close": [100.0, 101.0, 103.0],
            }
        )
        result = fill_gaps(df, interval="1h")
        assert result.height == 4
        # Hour 2 should be forward-filled from hour 1
        assert result["close"][2] == 101.0

    def test_backward_fill(self) -> None:
        base = datetime(2024, 1, 1)
        df = pl.DataFrame(
            {
                "timestamp": [base, base + timedelta(hours=2)],
                "close": [100.0, 102.0],
            }
        )
        result = fill_gaps(df, interval="1h", method="backward")
        assert result.height == 3
        assert result["close"][1] == 102.0

    def test_null_fill(self) -> None:
        base = datetime(2024, 1, 1)
        df = pl.DataFrame(
            {
                "timestamp": [base, base + timedelta(hours=2)],
                "close": [100.0, 102.0],
            }
        )
        result = fill_gaps(df, interval="1h", method="null")
        assert result.height == 3
        assert result["close"][1] is None

    def test_infer_interval(self) -> None:
        base = datetime(2024, 1, 1)
        df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(hours=i) for i in [0, 1, 2, 4, 5]],
                "close": [100.0, 101.0, 102.0, 104.0, 105.0],
            }
        )
        result = fill_gaps(df)
        assert result.height == 6  # 0-5 inclusive

    def test_no_gaps(self, ohlcv_df: pl.DataFrame) -> None:
        result = fill_gaps(ohlcv_df, interval="1h")
        assert result.height == ohlcv_df.height

    def test_missing_column_raises(self) -> None:
        df = pl.DataFrame({"close": [1.0]})
        with pytest.raises(ValueError, match="not found"):
            fill_gaps(df)

    def test_invalid_method_raises(self) -> None:
        base = datetime(2024, 1, 1)
        df = pl.DataFrame({"timestamp": [base], "close": [1.0]})
        with pytest.raises(ValueError, match="Invalid fill method"):
            fill_gaps(df, method="invalid")

    def test_single_row(self) -> None:
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "close": [100.0]})
        result = fill_gaps(df)
        assert result.height == 1

    def test_timedelta_interval(self) -> None:
        base = datetime(2024, 1, 1)
        df = pl.DataFrame(
            {
                "timestamp": [base, base + timedelta(minutes=10)],
                "close": [100.0, 101.0],
            }
        )
        result = fill_gaps(df, interval=timedelta(minutes=5))
        assert result.height == 3


class TestAdjustSplits:
    def test_2_for_1_split(self) -> None:
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i) for i in range(1, 6)],
                "open": [200.0, 200.0, 200.0, 100.0, 100.0],
                "high": [210.0, 210.0, 210.0, 105.0, 105.0],
                "low": [190.0, 190.0, 190.0, 95.0, 95.0],
                "close": [200.0, 200.0, 200.0, 100.0, 100.0],
                "volume": [1000.0, 1000.0, 1000.0, 2000.0, 2000.0],
            }
        )
        result = adjust_splits(df, splits=[("2024-01-04", 2.0)])
        # Prices before split should be halved
        assert result["close"][0] == pytest.approx(100.0)
        assert result["close"][1] == pytest.approx(100.0)
        assert result["close"][2] == pytest.approx(100.0)
        # Prices after split unchanged
        assert result["close"][3] == pytest.approx(100.0)
        # Volume before split should be doubled
        assert result["volume"][0] == pytest.approx(2000.0)
        # Volume after split unchanged
        assert result["volume"][3] == pytest.approx(2000.0)

    def test_multiple_splits(self) -> None:
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i) for i in range(1, 4)],
                "close": [400.0, 200.0, 100.0],
            }
        )
        result = adjust_splits(df, splits=[("2024-01-02", 2.0), ("2024-01-03", 2.0)])
        # First row: divided by both splits (400 / 2 / 2 = 100)
        assert result["close"][0] == pytest.approx(100.0)
        # Second row: divided by second split only (200 / 2 = 100)
        assert result["close"][1] == pytest.approx(100.0)
        # Third row: unchanged
        assert result["close"][2] == pytest.approx(100.0)

    def test_missing_timestamp_raises(self) -> None:
        df = pl.DataFrame({"close": [100.0]})
        with pytest.raises(ValueError, match="not found"):
            adjust_splits(df, splits=[("2024-01-01", 2.0)])


class TestDropZeroVolume:
    def test_drops_zeros(self) -> None:
        df = pl.DataFrame({"close": [1.0, 2.0, 3.0], "volume": [100.0, 0.0, 200.0]})
        result = drop_zero_volume(df)
        assert result.height == 2

    def test_drops_nulls(self) -> None:
        df = pl.DataFrame({"close": [1.0, 2.0], "volume": [100.0, None]})
        result = drop_zero_volume(df)
        assert result.height == 1

    def test_no_volume_column(self) -> None:
        df = pl.DataFrame({"close": [1.0, 2.0]})
        result = drop_zero_volume(df)
        assert result.height == 2


class TestClipOutliers:
    def test_clips_extremes(self) -> None:
        df = pl.DataFrame({"close": [1.0] * 100 + [1000.0] + [-500.0]})
        result = clip_outliers(df, columns=["close"])
        assert result["close"].max() <= 1000.0
        assert result["close"].min() >= -500.0
        # With tighter quantiles the extremes should be clipped
        result2 = clip_outliers(df, columns=["close"], lower_quantile=0.05, upper_quantile=0.95)
        assert result2["close"].max() < 1000.0
        assert result2["close"].min() > -500.0

    def test_no_columns(self) -> None:
        df = pl.DataFrame({"x": [1, 2, 3]})
        result = clip_outliers(df)
        assert result.height == 3


# ========== Resampling Tests ==========


class TestResampleOhlcv:
    def test_1h_to_4h(self) -> None:
        base = datetime(2024, 1, 1)
        n = 8
        df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(hours=i) for i in range(n)],
                "open": [100.0 + i for i in range(n)],
                "high": [110.0 + i for i in range(n)],
                "low": [90.0 + i for i in range(n)],
                "close": [105.0 + i for i in range(n)],
                "volume": [1000.0] * n,
            }
        )
        result = resample_ohlcv(df, interval="4h")
        assert result.height == 2
        # First candle: open=100, high=max(110..113)=113, low=min(90..93)=90, close=108, volume=4000
        assert result["open"][0] == pytest.approx(100.0)
        assert result["high"][0] == pytest.approx(113.0)
        assert result["low"][0] == pytest.approx(90.0)
        assert result["close"][0] == pytest.approx(108.0)
        assert result["volume"][0] == pytest.approx(4000.0)

    def test_5m_to_1h(self) -> None:
        base = datetime(2024, 1, 1)
        n = 24  # 2 hours of 5-min bars
        df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(minutes=5 * i) for i in range(n)],
                "close": [100.0 + i * 0.1 for i in range(n)],
                "volume": [500.0] * n,
            }
        )
        result = resample_ohlcv(df, interval="1h")
        assert result.height == 2
        assert result["volume"][0] == pytest.approx(6000.0)

    def test_timedelta_interval(self) -> None:
        base = datetime(2024, 1, 1)
        n = 6
        df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(hours=i) for i in range(n)],
                "close": [100.0 + i for i in range(n)],
            }
        )
        result = resample_ohlcv(df, interval=timedelta(hours=3))
        assert result.height == 2

    def test_extra_numeric_columns_aggregated(self) -> None:
        base = datetime(2024, 1, 1)
        df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(hours=i) for i in range(4)],
                "close": [100.0, 101.0, 102.0, 103.0],
                "sma_20": [99.0, 99.5, 100.0, 100.5],
            }
        )
        result = resample_ohlcv(df, interval="2h")
        assert "sma_20" in result.columns
        assert result.height == 2

    def test_missing_timestamp_raises(self) -> None:
        df = pl.DataFrame({"close": [1.0]})
        with pytest.raises(ValueError, match="not found"):
            resample_ohlcv(df, interval="1h")

    def test_1d_resampling(self) -> None:
        base = datetime(2024, 1, 1)
        n = 48  # 2 days of hourly data
        df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(hours=i) for i in range(n)],
                "open": [100.0] * n,
                "high": [110.0] * n,
                "low": [90.0] * n,
                "close": [105.0] * n,
                "volume": [100.0] * n,
            }
        )
        result = resample_ohlcv(df, interval="1d")
        assert result.height == 2
        assert result["volume"][0] == pytest.approx(2400.0)
