"""Tests that exercise the real TA-Lib library (not mocked).

Every test in this file calls TA-Lib directly to compute expected values, then
verifies that the polarbt integration wrapper produces identical results.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import talib

from polarbt.integrations.talib import ta, talib_available, talib_expr, talib_multi_expr, talib_series


@pytest.fixture
def ohlcv_data() -> pl.DataFrame:
    """Deterministic OHLCV data with enough bars for all indicators."""
    np.random.seed(123)
    n = 200
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.4) + 0.1
    low = close - np.abs(np.random.randn(n) * 0.4) - 0.1
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 50000, size=n).astype(float)
    return pl.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _np_close(df: pl.DataFrame) -> np.ndarray:
    return df["close"].to_numpy().astype(np.float64)


def _np(df: pl.DataFrame, col: str) -> np.ndarray:
    return df[col].to_numpy().astype(np.float64)


def _assert_arrays_equal(actual: pl.Series, expected: np.ndarray, tol: float = 1e-8) -> None:
    """Compare a Polars Series against a numpy array, ignoring NaN positions."""
    a = actual.to_numpy().astype(np.float64)
    mask = ~np.isnan(expected) & ~np.isnan(a)
    assert mask.sum() > 0, "No valid (non-NaN) values to compare"
    np.testing.assert_allclose(a[mask], expected[mask], atol=tol, rtol=1e-10)


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_talib_is_available(self):
        assert talib_available() is True


# ---------------------------------------------------------------------------
# Generic helpers with real TA-Lib
# ---------------------------------------------------------------------------


class TestTalibExprReal:
    def test_sma_via_talib_expr(self, ohlcv_data: pl.DataFrame):
        expected = talib.SMA(_np_close(ohlcv_data), timeperiod=20)
        df = ohlcv_data.with_columns(talib_expr(talib.SMA, "close", timeperiod=20).alias("sma"))
        _assert_arrays_equal(df["sma"], expected)

    def test_atr_via_talib_expr(self, ohlcv_data: pl.DataFrame):
        expected = talib.ATR(_np(ohlcv_data, "high"), _np(ohlcv_data, "low"), _np(ohlcv_data, "close"), timeperiod=14)
        df = ohlcv_data.with_columns(talib_expr(talib.ATR, ["high", "low", "close"], timeperiod=14).alias("atr"))
        _assert_arrays_equal(df["atr"], expected)

    def test_bbands_first_output_via_talib_expr(self, ohlcv_data: pl.DataFrame):
        upper, _, _ = talib.BBANDS(_np_close(ohlcv_data), timeperiod=20)
        df = ohlcv_data.with_columns(talib_expr(talib.BBANDS, "close", timeperiod=20).alias("bb_upper"))
        _assert_arrays_equal(df["bb_upper"], upper)


class TestTalibMultiExprReal:
    def test_bbands(self, ohlcv_data: pl.DataFrame):
        upper_exp, middle_exp, lower_exp = talib.BBANDS(_np_close(ohlcv_data), timeperiod=20)
        exprs = talib_multi_expr(talib.BBANDS, "close", timeperiod=20, output_names=["upper", "middle", "lower"])
        df = ohlcv_data.with_columns([v.alias(k) for k, v in exprs.items()])
        _assert_arrays_equal(df["upper"], upper_exp)
        _assert_arrays_equal(df["middle"], middle_exp)
        _assert_arrays_equal(df["lower"], lower_exp)

    def test_macd(self, ohlcv_data: pl.DataFrame):
        macd_exp, signal_exp, hist_exp = talib.MACD(_np_close(ohlcv_data))
        exprs = talib_multi_expr(talib.MACD, "close", output_names=["macd", "signal", "histogram"])
        df = ohlcv_data.with_columns([v.alias(k) for k, v in exprs.items()])
        _assert_arrays_equal(df["macd"], macd_exp)
        _assert_arrays_equal(df["signal"], signal_exp)
        _assert_arrays_equal(df["histogram"], hist_exp)

    def test_stoch(self, ohlcv_data: pl.DataFrame):
        slowk_exp, slowd_exp = talib.STOCH(_np(ohlcv_data, "high"), _np(ohlcv_data, "low"), _np(ohlcv_data, "close"))
        exprs = talib_multi_expr(talib.STOCH, ["high", "low", "close"], output_names=["slowk", "slowd"])
        df = ohlcv_data.with_columns([v.alias(k) for k, v in exprs.items()])
        _assert_arrays_equal(df["slowk"], slowk_exp)
        _assert_arrays_equal(df["slowd"], slowd_exp)

    def test_aroon(self, ohlcv_data: pl.DataFrame):
        down_exp, up_exp = talib.AROON(_np(ohlcv_data, "high"), _np(ohlcv_data, "low"), timeperiod=14)
        exprs = talib_multi_expr(talib.AROON, ["high", "low"], timeperiod=14, output_names=["aroon_down", "aroon_up"])
        df = ohlcv_data.with_columns([v.alias(k) for k, v in exprs.items()])
        _assert_arrays_equal(df["aroon_down"], down_exp)
        _assert_arrays_equal(df["aroon_up"], up_exp)


class TestTalibSeriesReal:
    def test_sma(self, ohlcv_data: pl.DataFrame):
        expected = talib.SMA(_np_close(ohlcv_data), timeperiod=10)
        result = talib_series(talib.SMA, ohlcv_data["close"], timeperiod=10)
        _assert_arrays_equal(result, expected)

    def test_rsi(self, ohlcv_data: pl.DataFrame):
        expected = talib.RSI(_np_close(ohlcv_data), timeperiod=14)
        result = talib_series(talib.RSI, ohlcv_data["close"], timeperiod=14)
        _assert_arrays_equal(result, expected)

    def test_multi_output_returns_first(self, ohlcv_data: pl.DataFrame):
        upper, _, _ = talib.BBANDS(_np_close(ohlcv_data), timeperiod=20)
        result = talib_series(talib.BBANDS, ohlcv_data["close"], timeperiod=20)
        _assert_arrays_equal(result, upper)


# ---------------------------------------------------------------------------
# TALibIndicators (ta namespace) — overlap / trend
# ---------------------------------------------------------------------------


class TestTaOverlap:
    def test_sma(self, ohlcv_data: pl.DataFrame):
        expected = talib.SMA(_np_close(ohlcv_data), timeperiod=20)
        df = ohlcv_data.with_columns(ta.sma("close", 20).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_ema(self, ohlcv_data: pl.DataFrame):
        expected = talib.EMA(_np_close(ohlcv_data), timeperiod=12)
        df = ohlcv_data.with_columns(ta.ema("close", 12).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_wma(self, ohlcv_data: pl.DataFrame):
        expected = talib.WMA(_np_close(ohlcv_data), timeperiod=15)
        df = ohlcv_data.with_columns(ta.wma("close", 15).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_dema(self, ohlcv_data: pl.DataFrame):
        expected = talib.DEMA(_np_close(ohlcv_data), timeperiod=20)
        df = ohlcv_data.with_columns(ta.dema("close", 20).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_tema(self, ohlcv_data: pl.DataFrame):
        expected = talib.TEMA(_np_close(ohlcv_data), timeperiod=20)
        df = ohlcv_data.with_columns(ta.tema("close", 20).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_kama(self, ohlcv_data: pl.DataFrame):
        expected = talib.KAMA(_np_close(ohlcv_data), timeperiod=30)
        df = ohlcv_data.with_columns(ta.kama("close", 30).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_trima(self, ohlcv_data: pl.DataFrame):
        expected = talib.TRIMA(_np_close(ohlcv_data), timeperiod=20)
        df = ohlcv_data.with_columns(ta.trima("close", 20).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_t3(self, ohlcv_data: pl.DataFrame):
        expected = talib.T3(_np_close(ohlcv_data), timeperiod=5, vfactor=0.7)
        df = ohlcv_data.with_columns(ta.t3("close", 5, 0.7).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_bollinger_bands(self, ohlcv_data: pl.DataFrame):
        upper_exp, middle_exp, lower_exp = talib.BBANDS(_np_close(ohlcv_data), timeperiod=20)
        result = ta.bollinger_bands("close", 20)
        df = ohlcv_data.with_columns([v.alias(k) for k, v in result.items()])
        _assert_arrays_equal(df["upper"], upper_exp)
        _assert_arrays_equal(df["middle"], middle_exp)
        _assert_arrays_equal(df["lower"], lower_exp)

    def test_sar(self, ohlcv_data: pl.DataFrame):
        expected = talib.SAR(_np(ohlcv_data, "high"), _np(ohlcv_data, "low"))
        df = ohlcv_data.with_columns(ta.sar("high", "low").alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_midpoint(self, ohlcv_data: pl.DataFrame):
        expected = talib.MIDPOINT(_np_close(ohlcv_data), timeperiod=14)
        df = ohlcv_data.with_columns(ta.midpoint("close", 14).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_midprice(self, ohlcv_data: pl.DataFrame):
        expected = talib.MIDPRICE(_np(ohlcv_data, "high"), _np(ohlcv_data, "low"), timeperiod=14)
        df = ohlcv_data.with_columns(ta.midprice("high", "low", 14).alias("v"))
        _assert_arrays_equal(df["v"], expected)


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


class TestTaMomentum:
    def test_rsi(self, ohlcv_data: pl.DataFrame):
        expected = talib.RSI(_np_close(ohlcv_data), timeperiod=14)
        df = ohlcv_data.with_columns(ta.rsi("close", 14).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_rsi_range(self, ohlcv_data: pl.DataFrame):
        """RSI values must be in [0, 100]."""
        df = ohlcv_data.with_columns(ta.rsi("close", 14).alias("rsi"))
        valid = df.filter(pl.col("rsi").is_not_null() & pl.col("rsi").is_not_nan())["rsi"]
        assert all(0 <= v <= 100 for v in valid)

    def test_macd(self, ohlcv_data: pl.DataFrame):
        macd_exp, signal_exp, hist_exp = talib.MACD(_np_close(ohlcv_data))
        result = ta.macd("close")
        df = ohlcv_data.with_columns([v.alias(k) for k, v in result.items()])
        _assert_arrays_equal(df["macd"], macd_exp)
        _assert_arrays_equal(df["signal"], signal_exp)
        _assert_arrays_equal(df["histogram"], hist_exp)

    def test_stochastic(self, ohlcv_data: pl.DataFrame):
        slowk_exp, slowd_exp = talib.STOCH(_np(ohlcv_data, "high"), _np(ohlcv_data, "low"), _np(ohlcv_data, "close"))
        result = ta.stochastic("high", "low", "close")
        df = ohlcv_data.with_columns([v.alias(k) for k, v in result.items()])
        _assert_arrays_equal(df["slowk"], slowk_exp)
        _assert_arrays_equal(df["slowd"], slowd_exp)

    def test_stochastic_fast(self, ohlcv_data: pl.DataFrame):
        fastk_exp, fastd_exp = talib.STOCHF(_np(ohlcv_data, "high"), _np(ohlcv_data, "low"), _np(ohlcv_data, "close"))
        result = ta.stochastic_fast("high", "low", "close")
        df = ohlcv_data.with_columns([v.alias(k) for k, v in result.items()])
        _assert_arrays_equal(df["fastk"], fastk_exp)
        _assert_arrays_equal(df["fastd"], fastd_exp)

    def test_williams_r(self, ohlcv_data: pl.DataFrame):
        expected = talib.WILLR(_np(ohlcv_data, "high"), _np(ohlcv_data, "low"), _np(ohlcv_data, "close"), timeperiod=14)
        df = ohlcv_data.with_columns(ta.williams_r("high", "low", "close", 14).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_williams_r_range(self, ohlcv_data: pl.DataFrame):
        """Williams %R must be in [-100, 0]."""
        df = ohlcv_data.with_columns(ta.williams_r("high", "low", "close").alias("wr"))
        valid = df.filter(pl.col("wr").is_not_null() & pl.col("wr").is_not_nan())["wr"]
        assert all(-100 <= v <= 0 for v in valid)

    def test_cci(self, ohlcv_data: pl.DataFrame):
        expected = talib.CCI(_np(ohlcv_data, "high"), _np(ohlcv_data, "low"), _np(ohlcv_data, "close"), timeperiod=20)
        df = ohlcv_data.with_columns(ta.cci("high", "low", "close", 20).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_mfi(self, ohlcv_data: pl.DataFrame):
        expected = talib.MFI(
            _np(ohlcv_data, "high"),
            _np(ohlcv_data, "low"),
            _np(ohlcv_data, "close"),
            _np(ohlcv_data, "volume"),
            timeperiod=14,
        )
        df = ohlcv_data.with_columns(ta.mfi("high", "low", "close", "volume", 14).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_mfi_range(self, ohlcv_data: pl.DataFrame):
        """MFI must be in [0, 100]."""
        df = ohlcv_data.with_columns(ta.mfi("high", "low", "close", "volume").alias("mfi"))
        valid = df.filter(pl.col("mfi").is_not_null() & pl.col("mfi").is_not_nan())["mfi"]
        assert all(0 <= v <= 100 for v in valid)

    def test_roc(self, ohlcv_data: pl.DataFrame):
        expected = talib.ROC(_np_close(ohlcv_data), timeperiod=12)
        df = ohlcv_data.with_columns(ta.roc("close", 12).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_mom(self, ohlcv_data: pl.DataFrame):
        expected = talib.MOM(_np_close(ohlcv_data), timeperiod=10)
        df = ohlcv_data.with_columns(ta.mom("close", 10).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_adx(self, ohlcv_data: pl.DataFrame):
        expected = talib.ADX(_np(ohlcv_data, "high"), _np(ohlcv_data, "low"), _np(ohlcv_data, "close"), timeperiod=14)
        df = ohlcv_data.with_columns(ta.adx("high", "low", "close", 14).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_adxr(self, ohlcv_data: pl.DataFrame):
        expected = talib.ADXR(_np(ohlcv_data, "high"), _np(ohlcv_data, "low"), _np(ohlcv_data, "close"), timeperiod=14)
        df = ohlcv_data.with_columns(ta.adxr("high", "low", "close", 14).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_apo(self, ohlcv_data: pl.DataFrame):
        expected = talib.APO(_np_close(ohlcv_data), fastperiod=12, slowperiod=26)
        df = ohlcv_data.with_columns(ta.apo("close", 12, 26).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_ppo(self, ohlcv_data: pl.DataFrame):
        expected = talib.PPO(_np_close(ohlcv_data), fastperiod=12, slowperiod=26)
        df = ohlcv_data.with_columns(ta.ppo("close", 12, 26).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_ultosc(self, ohlcv_data: pl.DataFrame):
        expected = talib.ULTOSC(_np(ohlcv_data, "high"), _np(ohlcv_data, "low"), _np(ohlcv_data, "close"))
        df = ohlcv_data.with_columns(ta.ultosc("high", "low", "close").alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_aroon(self, ohlcv_data: pl.DataFrame):
        down_exp, up_exp = talib.AROON(_np(ohlcv_data, "high"), _np(ohlcv_data, "low"), timeperiod=14)
        result = ta.aroon("high", "low", 14)
        df = ohlcv_data.with_columns([v.alias(k) for k, v in result.items()])
        _assert_arrays_equal(df["aroon_down"], down_exp)
        _assert_arrays_equal(df["aroon_up"], up_exp)

    def test_aroonosc(self, ohlcv_data: pl.DataFrame):
        expected = talib.AROONOSC(_np(ohlcv_data, "high"), _np(ohlcv_data, "low"), timeperiod=14)
        df = ohlcv_data.with_columns(ta.aroonosc("high", "low", 14).alias("v"))
        _assert_arrays_equal(df["v"], expected)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


class TestTaVolatility:
    def test_atr(self, ohlcv_data: pl.DataFrame):
        expected = talib.ATR(_np(ohlcv_data, "high"), _np(ohlcv_data, "low"), _np(ohlcv_data, "close"), timeperiod=14)
        df = ohlcv_data.with_columns(ta.atr("high", "low", "close", 14).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_atr_positive(self, ohlcv_data: pl.DataFrame):
        """ATR must be non-negative."""
        df = ohlcv_data.with_columns(ta.atr("high", "low", "close").alias("atr"))
        valid = df.filter(pl.col("atr").is_not_null() & pl.col("atr").is_not_nan())["atr"]
        assert all(v >= 0 for v in valid)

    def test_natr(self, ohlcv_data: pl.DataFrame):
        expected = talib.NATR(_np(ohlcv_data, "high"), _np(ohlcv_data, "low"), _np(ohlcv_data, "close"), timeperiod=14)
        df = ohlcv_data.with_columns(ta.natr("high", "low", "close", 14).alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_trange(self, ohlcv_data: pl.DataFrame):
        expected = talib.TRANGE(_np(ohlcv_data, "high"), _np(ohlcv_data, "low"), _np(ohlcv_data, "close"))
        df = ohlcv_data.with_columns(ta.trange("high", "low", "close").alias("v"))
        _assert_arrays_equal(df["v"], expected)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


class TestTaVolume:
    def test_obv(self, ohlcv_data: pl.DataFrame):
        expected = talib.OBV(_np_close(ohlcv_data), _np(ohlcv_data, "volume"))
        df = ohlcv_data.with_columns(ta.obv("close", "volume").alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_ad(self, ohlcv_data: pl.DataFrame):
        expected = talib.AD(
            _np(ohlcv_data, "high"),
            _np(ohlcv_data, "low"),
            _np(ohlcv_data, "close"),
            _np(ohlcv_data, "volume"),
        )
        df = ohlcv_data.with_columns(ta.ad("high", "low", "close", "volume").alias("v"))
        _assert_arrays_equal(df["v"], expected)

    def test_adosc(self, ohlcv_data: pl.DataFrame):
        expected = talib.ADOSC(
            _np(ohlcv_data, "high"),
            _np(ohlcv_data, "low"),
            _np(ohlcv_data, "close"),
            _np(ohlcv_data, "volume"),
            fastperiod=3,
            slowperiod=10,
        )
        df = ohlcv_data.with_columns(ta.adosc("high", "low", "close", "volume", 3, 10).alias("v"))
        _assert_arrays_equal(df["v"], expected)


# ---------------------------------------------------------------------------
# Candlestick Patterns
# ---------------------------------------------------------------------------


class TestTaCandlestick:
    def test_cdl_doji(self, ohlcv_data: pl.DataFrame):
        expected = talib.CDLDOJI(
            _np(ohlcv_data, "open"),
            _np(ohlcv_data, "high"),
            _np(ohlcv_data, "low"),
            _np(ohlcv_data, "close"),
        )
        df = ohlcv_data.with_columns(ta.cdl_doji("open", "high", "low", "close").alias("v"))
        _assert_arrays_equal(df["v"], expected.astype(np.float64))

    def test_cdl_hammer(self, ohlcv_data: pl.DataFrame):
        expected = talib.CDLHAMMER(
            _np(ohlcv_data, "open"),
            _np(ohlcv_data, "high"),
            _np(ohlcv_data, "low"),
            _np(ohlcv_data, "close"),
        )
        df = ohlcv_data.with_columns(ta.cdl_hammer("open", "high", "low", "close").alias("v"))
        _assert_arrays_equal(df["v"], expected.astype(np.float64))

    def test_cdl_engulfing(self, ohlcv_data: pl.DataFrame):
        expected = talib.CDLENGULFING(
            _np(ohlcv_data, "open"),
            _np(ohlcv_data, "high"),
            _np(ohlcv_data, "low"),
            _np(ohlcv_data, "close"),
        )
        df = ohlcv_data.with_columns(ta.cdl_engulfing("open", "high", "low", "close").alias("v"))
        _assert_arrays_equal(df["v"], expected.astype(np.float64))

    def test_cdl_morning_star(self, ohlcv_data: pl.DataFrame):
        expected = talib.CDLMORNINGSTAR(
            _np(ohlcv_data, "open"),
            _np(ohlcv_data, "high"),
            _np(ohlcv_data, "low"),
            _np(ohlcv_data, "close"),
        )
        df = ohlcv_data.with_columns(ta.cdl_morning_star("open", "high", "low", "close").alias("v"))
        _assert_arrays_equal(df["v"], expected.astype(np.float64))

    def test_cdl_evening_star(self, ohlcv_data: pl.DataFrame):
        expected = talib.CDLEVENINGSTAR(
            _np(ohlcv_data, "open"),
            _np(ohlcv_data, "high"),
            _np(ohlcv_data, "low"),
            _np(ohlcv_data, "close"),
        )
        df = ohlcv_data.with_columns(ta.cdl_evening_star("open", "high", "low", "close").alias("v"))
        _assert_arrays_equal(df["v"], expected.astype(np.float64))

    def test_cdl_values_are_integers(self, ohlcv_data: pl.DataFrame):
        """Candlestick pattern outputs should be integer-like (0, 100, -100)."""
        df = ohlcv_data.with_columns(ta.cdl_engulfing("open", "high", "low", "close").alias("v"))
        valid = df.filter(pl.col("v").is_not_null() & pl.col("v").is_not_nan())["v"]
        for v in valid:
            assert v in (0.0, 100.0, -100.0), f"Unexpected candlestick value: {v}"


# ---------------------------------------------------------------------------
# Custom period variations (verify parameter forwarding)
# ---------------------------------------------------------------------------


class TestCustomParameters:
    def test_sma_different_periods(self, ohlcv_data: pl.DataFrame):
        for period in [5, 10, 50]:
            expected = talib.SMA(_np_close(ohlcv_data), timeperiod=period)
            df = ohlcv_data.with_columns(ta.sma("close", period).alias("v"))
            _assert_arrays_equal(df["v"], expected)

    def test_ema_different_periods(self, ohlcv_data: pl.DataFrame):
        for period in [5, 20, 50]:
            expected = talib.EMA(_np_close(ohlcv_data), timeperiod=period)
            df = ohlcv_data.with_columns(ta.ema("close", period).alias("v"))
            _assert_arrays_equal(df["v"], expected)

    def test_rsi_different_periods(self, ohlcv_data: pl.DataFrame):
        for period in [7, 14, 21]:
            expected = talib.RSI(_np_close(ohlcv_data), timeperiod=period)
            df = ohlcv_data.with_columns(ta.rsi("close", period).alias("v"))
            _assert_arrays_equal(df["v"], expected)

    def test_bollinger_asymmetric_bands(self, ohlcv_data: pl.DataFrame):
        upper_exp, middle_exp, lower_exp = talib.BBANDS(_np_close(ohlcv_data), timeperiod=20, nbdevup=2.5, nbdevdn=1.5)
        result = ta.bollinger_bands("close", 20, nbdevup=2.5, nbdevdn=1.5)
        df = ohlcv_data.with_columns([v.alias(k) for k, v in result.items()])
        _assert_arrays_equal(df["upper"], upper_exp)
        _assert_arrays_equal(df["lower"], lower_exp)

    def test_macd_custom_periods(self, ohlcv_data: pl.DataFrame):
        macd_exp, signal_exp, hist_exp = talib.MACD(_np_close(ohlcv_data), fastperiod=8, slowperiod=21, signalperiod=5)
        result = ta.macd("close", fast=8, slow=21, signal=5)
        df = ohlcv_data.with_columns([v.alias(k) for k, v in result.items()])
        _assert_arrays_equal(df["macd"], macd_exp)
        _assert_arrays_equal(df["signal"], signal_exp)
        _assert_arrays_equal(df["histogram"], hist_exp)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_short_data(self):
        """Indicators on very short data should produce NaN-filled results without errors."""
        df = pl.DataFrame({"close": [100.0, 101.0, 102.0]})
        result = df.with_columns(ta.sma("close", 20).alias("v"))
        assert result["v"].is_nan().sum() == 3

    def test_constant_prices(self):
        """Constant prices should produce zero ATR, zero std, etc."""
        n = 50
        df = pl.DataFrame(
            {
                "high": [100.0] * n,
                "low": [100.0] * n,
                "close": [100.0] * n,
            }
        )
        result = df.with_columns(ta.atr("high", "low", "close", 14).alias("atr"))
        valid = result.filter(pl.col("atr").is_not_null() & pl.col("atr").is_not_nan())["atr"]
        assert all(abs(v) < 1e-10 for v in valid)

    def test_multiple_indicators_at_once(self, ohlcv_data: pl.DataFrame):
        """Apply multiple TA-Lib indicators in a single with_columns call."""
        df = ohlcv_data.with_columns(
            [
                ta.sma("close", 20).alias("sma_20"),
                ta.ema("close", 12).alias("ema_12"),
                ta.rsi("close", 14).alias("rsi_14"),
                ta.atr("high", "low", "close", 14).alias("atr_14"),
            ]
        )
        assert "sma_20" in df.columns
        assert "ema_12" in df.columns
        assert "rsi_14" in df.columns
        assert "atr_14" in df.columns
        # All should have values (after warmup)
        for col in ["sma_20", "ema_12", "rsi_14", "atr_14"]:
            valid = df.filter(pl.col(col).is_not_null() & pl.col(col).is_not_nan())[col]
            assert len(valid) > 100, f"{col} has too few valid values"
