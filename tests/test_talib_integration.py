"""Tests for TA-Lib integration module.

Tests use mocking so they run without TA-Lib installed. Tests marked with
``@pytest.mark.talib`` additionally run against real TA-Lib when available.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def ohlcv_data() -> pl.DataFrame:
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
# Availability check
# ---------------------------------------------------------------------------


class TestTalibAvailable:
    def test_talib_available_returns_bool(self):
        from polarbt.integrations.talib import talib_available

        assert isinstance(talib_available(), bool)

    def test_import_error_when_not_installed(self):
        """When TA-Lib is missing, calling functions raises ImportError."""
        with patch.dict("sys.modules", {"talib": None}):
            # Force reimport to pick up the mock

            import polarbt.integrations.talib as talib_mod

            original = talib_mod.TALIB_AVAILABLE
            talib_mod.TALIB_AVAILABLE = False
            try:
                with pytest.raises(ImportError, match="TA-Lib is not installed"):
                    talib_mod.ta.sma("close", 20)
            finally:
                talib_mod.TALIB_AVAILABLE = original


# ---------------------------------------------------------------------------
# Mock-based tests (always run)
# ---------------------------------------------------------------------------


def _make_mock_talib() -> MagicMock:
    """Create a mock talib module with common functions."""
    mock = MagicMock()
    # Single-output functions return numpy arrays
    for name in [
        "SMA",
        "EMA",
        "WMA",
        "DEMA",
        "TEMA",
        "KAMA",
        "TRIMA",
        "T3",
        "RSI",
        "WILLR",
        "CCI",
        "MFI",
        "ROC",
        "MOM",
        "ADX",
        "ADXR",
        "APO",
        "PPO",
        "ULTOSC",
        "AROONOSC",
        "ATR",
        "NATR",
        "TRANGE",
        "OBV",
        "AD",
        "ADOSC",
        "SAR",
        "MIDPOINT",
        "MIDPRICE",
        "CDLDOJI",
        "CDLHAMMER",
        "CDLENGULFING",
        "CDLMORNINGSTAR",
        "CDLEVENINGSTAR",
    ]:
        getattr(mock, name).side_effect = lambda *a, **kw: np.full(len(a[0]), 42.0)

    # Multi-output functions return tuples
    mock.BBANDS.side_effect = lambda *a, **kw: (
        np.full(len(a[0]), 110.0),
        np.full(len(a[0]), 100.0),
        np.full(len(a[0]), 90.0),
    )
    mock.MACD.side_effect = lambda *a, **kw: (
        np.full(len(a[0]), 1.0),
        np.full(len(a[0]), 0.5),
        np.full(len(a[0]), 0.5),
    )
    mock.STOCH.side_effect = lambda *a, **kw: (
        np.full(len(a[0]), 80.0),
        np.full(len(a[0]), 75.0),
    )
    mock.STOCHF.side_effect = lambda *a, **kw: (
        np.full(len(a[0]), 85.0),
        np.full(len(a[0]), 80.0),
    )
    mock.AROON.side_effect = lambda *a, **kw: (
        np.full(len(a[0]), 50.0),
        np.full(len(a[0]), 60.0),
    )
    return mock


@pytest.fixture
def mock_talib():
    """Patch talib module with mocks and re-enable TALIB_AVAILABLE."""
    mock = _make_mock_talib()
    import polarbt.integrations.talib as talib_mod

    original_talib = talib_mod._talib
    original_available = talib_mod.TALIB_AVAILABLE
    talib_mod._talib = mock
    talib_mod.TALIB_AVAILABLE = True
    yield mock
    talib_mod._talib = original_talib
    talib_mod.TALIB_AVAILABLE = original_available


class TestTalibExpr:
    def test_single_column(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import talib_expr

        df = ohlcv_data.with_columns(talib_expr(mock_talib.SMA, "close", timeperiod=20).alias("sma"))
        assert df["sma"].null_count() == 0
        assert abs(df["sma"][0] - 42.0) < 1e-9

    def test_multi_column(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import talib_expr

        df = ohlcv_data.with_columns(talib_expr(mock_talib.ATR, ["high", "low", "close"], timeperiod=14).alias("atr"))
        assert abs(df["atr"][0] - 42.0) < 1e-9


class TestTalibMultiExpr:
    def test_bollinger_bands(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import talib_multi_expr

        exprs = talib_multi_expr(
            mock_talib.BBANDS,
            "close",
            timeperiod=20,
            output_names=["upper", "middle", "lower"],
        )
        assert set(exprs.keys()) == {"upper", "middle", "lower"}
        df = ohlcv_data.with_columns([v.alias(k) for k, v in exprs.items()])
        assert abs(df["upper"][0] - 110.0) < 1e-9
        assert abs(df["middle"][0] - 100.0) < 1e-9
        assert abs(df["lower"][0] - 90.0) < 1e-9

    def test_default_output_names(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import talib_multi_expr

        exprs = talib_multi_expr(mock_talib.MACD, "close", fastperiod=12, slowperiod=26, signalperiod=9)
        assert set(exprs.keys()) == {"output_0", "output_1", "output_2"}


class TestTalibSeries:
    def test_basic(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import talib_series

        result = talib_series(mock_talib.SMA, ohlcv_data["close"], timeperiod=20)
        assert isinstance(result, pl.Series)
        assert len(result) == len(ohlcv_data)
        assert abs(result[0] - 42.0) < 1e-9

    def test_multi_output_returns_first(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import talib_series

        result = talib_series(mock_talib.BBANDS, ohlcv_data["close"], timeperiod=20)
        assert abs(result[0] - 110.0) < 1e-9  # first output (upper)


class TestTALibIndicatorsClass:
    """Test the convenience ``ta`` namespace with mocked TA-Lib."""

    def test_sma(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.sma("close", 20).alias("sma"))
        assert abs(df["sma"][0] - 42.0) < 1e-9
        mock_talib.SMA.assert_called()

    def test_ema(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.ema("close", 12).alias("ema"))
        assert abs(df["ema"][0] - 42.0) < 1e-9

    def test_wma(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.wma("close", 10).alias("wma"))
        assert abs(df["wma"][0] - 42.0) < 1e-9

    def test_dema(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.dema("close", 20).alias("dema"))
        assert abs(df["dema"][0] - 42.0) < 1e-9

    def test_tema(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.tema("close", 20).alias("tema"))
        assert abs(df["tema"][0] - 42.0) < 1e-9

    def test_kama(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.kama("close", 20).alias("kama"))
        assert abs(df["kama"][0] - 42.0) < 1e-9

    def test_trima(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.trima("close", 20).alias("trima"))
        assert abs(df["trima"][0] - 42.0) < 1e-9

    def test_t3(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.t3("close", 5, 0.7).alias("t3"))
        assert abs(df["t3"][0] - 42.0) < 1e-9

    def test_bollinger_bands(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        result = ta.bollinger_bands("close", 20)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"upper", "middle", "lower"}
        df = ohlcv_data.with_columns([v.alias(k) for k, v in result.items()])
        assert abs(df["upper"][0] - 110.0) < 1e-9

    def test_sar(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.sar("high", "low").alias("sar"))
        assert abs(df["sar"][0] - 42.0) < 1e-9

    def test_rsi(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.rsi("close", 14).alias("rsi"))
        assert abs(df["rsi"][0] - 42.0) < 1e-9

    def test_macd(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        result = ta.macd("close")
        assert set(result.keys()) == {"macd", "signal", "histogram"}
        df = ohlcv_data.with_columns([v.alias(k) for k, v in result.items()])
        assert abs(df["macd"][0] - 1.0) < 1e-9

    def test_stochastic(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        result = ta.stochastic("high", "low", "close")
        assert set(result.keys()) == {"slowk", "slowd"}

    def test_stochastic_fast(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        result = ta.stochastic_fast("high", "low", "close")
        assert set(result.keys()) == {"fastk", "fastd"}

    def test_williams_r(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.williams_r("high", "low", "close").alias("wr"))
        assert abs(df["wr"][0] - 42.0) < 1e-9

    def test_cci(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.cci("high", "low", "close").alias("cci"))
        assert abs(df["cci"][0] - 42.0) < 1e-9

    def test_mfi(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.mfi("high", "low", "close", "volume").alias("mfi"))
        assert abs(df["mfi"][0] - 42.0) < 1e-9

    def test_roc(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.roc("close", 12).alias("roc"))
        assert abs(df["roc"][0] - 42.0) < 1e-9

    def test_mom(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.mom("close", 10).alias("mom"))
        assert abs(df["mom"][0] - 42.0) < 1e-9

    def test_adx(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.adx("high", "low", "close").alias("adx"))
        assert abs(df["adx"][0] - 42.0) < 1e-9

    def test_apo(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.apo("close").alias("apo"))
        assert abs(df["apo"][0] - 42.0) < 1e-9

    def test_ppo(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.ppo("close").alias("ppo"))
        assert abs(df["ppo"][0] - 42.0) < 1e-9

    def test_ultosc(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.ultosc("high", "low", "close").alias("ultosc"))
        assert abs(df["ultosc"][0] - 42.0) < 1e-9

    def test_aroon(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        result = ta.aroon("high", "low")
        assert set(result.keys()) == {"aroon_down", "aroon_up"}

    def test_aroonosc(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.aroonosc("high", "low").alias("aroonosc"))
        assert abs(df["aroonosc"][0] - 42.0) < 1e-9

    def test_atr(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.atr("high", "low", "close").alias("atr"))
        assert abs(df["atr"][0] - 42.0) < 1e-9

    def test_natr(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.natr("high", "low", "close").alias("natr"))
        assert abs(df["natr"][0] - 42.0) < 1e-9

    def test_trange(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.trange("high", "low", "close").alias("tr"))
        assert abs(df["tr"][0] - 42.0) < 1e-9

    def test_obv(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.obv("close", "volume").alias("obv"))
        assert abs(df["obv"][0] - 42.0) < 1e-9

    def test_ad(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.ad("high", "low", "close", "volume").alias("ad"))
        assert abs(df["ad"][0] - 42.0) < 1e-9

    def test_adosc(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.adosc("high", "low", "close", "volume").alias("adosc"))
        assert abs(df["adosc"][0] - 42.0) < 1e-9

    def test_cdl_doji(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.cdl_doji("open", "high", "low", "close").alias("doji"))
        assert abs(df["doji"][0] - 42.0) < 1e-9

    def test_cdl_hammer(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.cdl_hammer("open", "high", "low", "close").alias("hammer"))
        assert abs(df["hammer"][0] - 42.0) < 1e-9

    def test_cdl_engulfing(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.cdl_engulfing("open", "high", "low", "close").alias("engulfing"))
        assert abs(df["engulfing"][0] - 42.0) < 1e-9

    def test_cdl_morning_star(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.cdl_morning_star("open", "high", "low", "close").alias("ms"))
        assert abs(df["ms"][0] - 42.0) < 1e-9

    def test_cdl_evening_star(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.cdl_evening_star("open", "high", "low", "close").alias("es"))
        assert abs(df["es"][0] - 42.0) < 1e-9

    def test_midpoint(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.midpoint("close").alias("mp"))
        assert abs(df["mp"][0] - 42.0) < 1e-9

    def test_midprice(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.midprice("high", "low").alias("mp"))
        assert abs(df["mp"][0] - 42.0) < 1e-9

    def test_adxr(self, ohlcv_data: pl.DataFrame, mock_talib: MagicMock):
        from polarbt.integrations.talib import ta

        df = ohlcv_data.with_columns(ta.adxr("high", "low", "close").alias("adxr"))
        assert abs(df["adxr"][0] - 42.0) < 1e-9
