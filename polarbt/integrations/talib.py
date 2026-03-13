"""Optional TA-Lib integration for PolarBT.

Provides convenience functions that call TA-Lib under the hood and return
Polars expressions or Series, making it easy to use TA-Lib indicators inside
``Strategy.preprocess()``.

Install TA-Lib separately::

    pip install TA-Lib

Usage::

    from polarbt.integrations.talib import ta

    class MyStrategy(Strategy):
        def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns([
                ta.sma("close", 20).alias("sma_20"),
                ta.rsi("close", 14).alias("rsi"),
            ])

The module also exposes a lower-level :func:`talib_expr` helper that wraps
*any* TA-Lib function into a Polars expression, and :func:`talib_series` that
operates on concrete Polars Series.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import polars as pl

try:
    import talib as _talib

    TALIB_AVAILABLE = True
except ImportError:
    _talib = None  # type: ignore[assignment,unused-ignore]
    TALIB_AVAILABLE = False


def talib_available() -> bool:
    """Return *True* if TA-Lib is installed and importable."""
    return TALIB_AVAILABLE


def _require_talib() -> None:
    if not TALIB_AVAILABLE:
        raise ImportError(
            "TA-Lib is not installed. Install it with: pip install TA-Lib\n"
            "See https://github.com/TA-Lib/ta-lib-python for installation instructions."
        )


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def talib_expr(
    func: Callable[..., np.ndarray[Any, Any]],
    columns: str | Sequence[str],
    *args: Any,
    **kwargs: Any,
) -> pl.Expr:
    """Wrap an arbitrary TA-Lib function into a Polars expression.

    The returned expression can be used inside ``df.with_columns()``.

    Args:
        func: A TA-Lib function (e.g. ``talib.SMA``).
        columns: Column name(s) to pass to *func* as positional numpy arrays.
        *args: Extra positional arguments forwarded to *func* after the arrays.
        **kwargs: Keyword arguments forwarded to *func*.

    Returns:
        A Polars expression that applies *func* via ``map_batches``.

    Example::

        import talib
        df.with_columns([
            talib_expr(talib.SMA, "close", timeperiod=20).alias("sma_20"),
            talib_expr(talib.BBANDS, ["close"], timeperiod=20).alias("bbands"),
        ])
    """
    _require_talib()

    if isinstance(columns, str):
        columns = [columns]
    col_list: list[str] = list(columns)

    def _apply(struct_series: pl.Series) -> pl.Series:
        df = struct_series.struct.unnest()
        arrays = [df[c].to_numpy().astype(np.float64) for c in col_list]
        result = func(*arrays, *args, **kwargs)
        if isinstance(result, tuple):
            # Multi-output: return first output (users should call talib_multi_expr instead)
            return pl.Series(np.asarray(result[0], dtype=np.float64))
        return pl.Series(np.asarray(result, dtype=np.float64))

    return pl.struct(col_list).map_batches(_apply, return_dtype=pl.Float64)


def talib_multi_expr(
    func: Callable[..., tuple[np.ndarray[Any, Any], ...]],
    columns: str | Sequence[str],
    *args: Any,
    output_names: Sequence[str] | None = None,
    **kwargs: Any,
) -> dict[str, pl.Expr]:
    """Wrap a multi-output TA-Lib function into multiple Polars expressions.

    Args:
        func: A TA-Lib function that returns a tuple of arrays.
        columns: Column name(s) to pass as positional numpy arrays.
        *args: Extra positional arguments forwarded to *func*.
        output_names: Names for each output. If ``None``, defaults to
            ``"output_0"``, ``"output_1"``, …
        **kwargs: Keyword arguments forwarded to *func*.

    Returns:
        Dict mapping output names to Polars expressions.

    Example::

        exprs = talib_multi_expr(
            talib.BBANDS, "close",
            timeperiod=20,
            output_names=["upper", "middle", "lower"],
        )
        df.with_columns([v.alias(k) for k, v in exprs.items()])
    """
    _require_talib()

    if isinstance(columns, str):
        columns = [columns]
    col_list: list[str] = list(columns)

    result_exprs: dict[str, pl.Expr] = {}

    def _make_extractor(idx: int) -> Callable[[pl.Series], pl.Series]:
        def _apply(struct_series: pl.Series) -> pl.Series:
            df = struct_series.struct.unnest()
            arrays = [df[c].to_numpy().astype(np.float64) for c in col_list]
            result = func(*arrays, *args, **kwargs)
            return pl.Series(np.asarray(result[idx], dtype=np.float64))

        return _apply

    # Determine how many outputs the function produces by inspecting its info
    # We do a dry run with small arrays to count outputs
    n_outputs = _count_outputs(func, len(col_list), *args, **kwargs)

    if output_names is None:
        output_names = [f"output_{i}" for i in range(n_outputs)]

    for i, name in enumerate(output_names):
        result_exprs[name] = pl.struct(col_list).map_batches(_make_extractor(i), return_dtype=pl.Float64)

    return result_exprs


def _count_outputs(func: Callable[..., Any], n_inputs: int, *args: Any, **kwargs: Any) -> int:
    """Determine how many arrays a TA-Lib function returns."""
    dummy = [np.ones(100, dtype=np.float64) for _ in range(n_inputs)]
    result = func(*dummy, *args, **kwargs)
    if isinstance(result, tuple):
        return len(result)
    return 1


def talib_series(
    func: Callable[..., np.ndarray[Any, Any]],
    *series: pl.Series,
    **kwargs: Any,
) -> pl.Series:
    """Apply a TA-Lib function to Polars Series directly.

    Useful outside of a DataFrame context.

    Args:
        func: A TA-Lib function.
        *series: Polars Series to pass as numpy arrays.
        **kwargs: Keyword arguments forwarded to *func*.

    Returns:
        Polars Series with the result.
    """
    _require_talib()
    arrays = [s.to_numpy().astype(np.float64) for s in series]
    result = func(*arrays, **kwargs)
    if isinstance(result, tuple):
        return pl.Series(np.asarray(result[0], dtype=np.float64))
    return pl.Series(np.asarray(result, dtype=np.float64))


# ---------------------------------------------------------------------------
# Convenience wrappers — mirror the polarbt.indicators API
# ---------------------------------------------------------------------------


class TALibIndicators:
    """Namespace of TA-Lib backed indicator functions.

    Each method returns a Polars expression (or dict of expressions for
    multi-output indicators) that can be used in ``df.with_columns()``.

    .. note::

       Multi-output functions return ``dict[str, pl.Expr]`` while
       ``polarbt.indicators`` returns tuples.  Single-output functions
       share the same signature.

    Access via the module-level ``ta`` instance::

        from polarbt.integrations.talib import ta

        df.with_columns([ta.sma("close", 20).alias("sma_20")])
    """

    # -- Overlap Studies (Trend) --

    @staticmethod
    def sma(column: str, period: int = 30) -> pl.Expr:
        """Simple Moving Average via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.SMA, column, timeperiod=period)

    @staticmethod
    def ema(column: str, period: int = 30) -> pl.Expr:
        """Exponential Moving Average via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.EMA, column, timeperiod=period)

    @staticmethod
    def wma(column: str, period: int = 30) -> pl.Expr:
        """Weighted Moving Average via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.WMA, column, timeperiod=period)

    @staticmethod
    def dema(column: str, period: int = 30) -> pl.Expr:
        """Double Exponential Moving Average via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.DEMA, column, timeperiod=period)

    @staticmethod
    def tema(column: str, period: int = 30) -> pl.Expr:
        """Triple Exponential Moving Average via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.TEMA, column, timeperiod=period)

    @staticmethod
    def kama(column: str, period: int = 30) -> pl.Expr:
        """Kaufman Adaptive Moving Average via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.KAMA, column, timeperiod=period)

    @staticmethod
    def trima(column: str, period: int = 30) -> pl.Expr:
        """Triangular Moving Average via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.TRIMA, column, timeperiod=period)

    @staticmethod
    def t3(column: str, period: int = 5, vfactor: float = 0.7) -> pl.Expr:
        """Triple Exponential Moving Average (T3) via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.T3, column, timeperiod=period, vfactor=vfactor)

    @staticmethod
    def bollinger_bands(
        column: str,
        period: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
    ) -> dict[str, pl.Expr]:
        """Bollinger Bands via TA-Lib.

        Returns dict with keys: ``upper``, ``middle``, ``lower``.
        """
        _require_talib()
        return talib_multi_expr(
            _talib.BBANDS,
            column,
            timeperiod=period,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn,
            output_names=["upper", "middle", "lower"],
        )

    @staticmethod
    def sar(high: str = "high", low: str = "low", acceleration: float = 0.02, maximum: float = 0.2) -> pl.Expr:
        """Parabolic SAR via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.SAR, [high, low], acceleration=acceleration, maximum=maximum)

    @staticmethod
    def midpoint(column: str, period: int = 14) -> pl.Expr:
        """MidPoint over period via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.MIDPOINT, column, timeperiod=period)

    @staticmethod
    def midprice(high: str = "high", low: str = "low", period: int = 14) -> pl.Expr:
        """Midpoint Price over period via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.MIDPRICE, [high, low], timeperiod=period)

    # -- Momentum --

    @staticmethod
    def rsi(column: str, period: int = 14) -> pl.Expr:
        """Relative Strength Index via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.RSI, column, timeperiod=period)

    @staticmethod
    def macd(
        column: str,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> dict[str, pl.Expr]:
        """MACD via TA-Lib.

        Returns dict with keys: ``macd``, ``signal``, ``histogram``.
        """
        _require_talib()
        return talib_multi_expr(
            _talib.MACD,
            column,
            fastperiod=fast,
            slowperiod=slow,
            signalperiod=signal,
            output_names=["macd", "signal", "histogram"],
        )

    @staticmethod
    def stochastic(
        high: str = "high",
        low: str = "low",
        close: str = "close",
        fastk_period: int = 5,
        slowk_period: int = 3,
        slowd_period: int = 3,
    ) -> dict[str, pl.Expr]:
        """Stochastic Oscillator via TA-Lib.

        Returns dict with keys: ``slowk``, ``slowd``.
        """
        _require_talib()
        return talib_multi_expr(
            _talib.STOCH,
            [high, low, close],
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowd_period=slowd_period,
            output_names=["slowk", "slowd"],
        )

    @staticmethod
    def stochastic_fast(
        high: str = "high",
        low: str = "low",
        close: str = "close",
        fastk_period: int = 5,
        fastd_period: int = 3,
    ) -> dict[str, pl.Expr]:
        """Fast Stochastic Oscillator via TA-Lib.

        Returns dict with keys: ``fastk``, ``fastd``.
        """
        _require_talib()
        return talib_multi_expr(
            _talib.STOCHF,
            [high, low, close],
            fastk_period=fastk_period,
            fastd_period=fastd_period,
            output_names=["fastk", "fastd"],
        )

    @staticmethod
    def williams_r(
        high: str = "high",
        low: str = "low",
        close: str = "close",
        period: int = 14,
    ) -> pl.Expr:
        """Williams %R via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.WILLR, [high, low, close], timeperiod=period)

    @staticmethod
    def cci(
        high: str = "high",
        low: str = "low",
        close: str = "close",
        period: int = 14,
    ) -> pl.Expr:
        """Commodity Channel Index via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.CCI, [high, low, close], timeperiod=period)

    @staticmethod
    def mfi(
        high: str = "high",
        low: str = "low",
        close: str = "close",
        volume: str = "volume",
        period: int = 14,
    ) -> pl.Expr:
        """Money Flow Index via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.MFI, [high, low, close, volume], timeperiod=period)

    @staticmethod
    def roc(column: str, period: int = 10) -> pl.Expr:
        """Rate of Change via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.ROC, column, timeperiod=period)

    @staticmethod
    def mom(column: str, period: int = 10) -> pl.Expr:
        """Momentum via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.MOM, column, timeperiod=period)

    @staticmethod
    def adx(
        high: str = "high",
        low: str = "low",
        close: str = "close",
        period: int = 14,
    ) -> pl.Expr:
        """Average Directional Movement Index via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.ADX, [high, low, close], timeperiod=period)

    @staticmethod
    def adxr(
        high: str = "high",
        low: str = "low",
        close: str = "close",
        period: int = 14,
    ) -> pl.Expr:
        """Average Directional Movement Index Rating via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.ADXR, [high, low, close], timeperiod=period)

    @staticmethod
    def apo(column: str, fast: int = 12, slow: int = 26) -> pl.Expr:
        """Absolute Price Oscillator via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.APO, column, fastperiod=fast, slowperiod=slow)

    @staticmethod
    def ppo(column: str, fast: int = 12, slow: int = 26) -> pl.Expr:
        """Percentage Price Oscillator via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.PPO, column, fastperiod=fast, slowperiod=slow)

    @staticmethod
    def ultosc(
        high: str = "high",
        low: str = "low",
        close: str = "close",
        period1: int = 7,
        period2: int = 14,
        period3: int = 28,
    ) -> pl.Expr:
        """Ultimate Oscillator via TA-Lib."""
        _require_talib()
        return talib_expr(
            _talib.ULTOSC, [high, low, close], timeperiod1=period1, timeperiod2=period2, timeperiod3=period3
        )

    @staticmethod
    def aroon(high: str = "high", low: str = "low", period: int = 14) -> dict[str, pl.Expr]:
        """Aroon via TA-Lib.

        Returns dict with keys: ``aroon_down``, ``aroon_up``.
        """
        _require_talib()
        return talib_multi_expr(
            _talib.AROON,
            [high, low],
            timeperiod=period,
            output_names=["aroon_down", "aroon_up"],
        )

    @staticmethod
    def aroonosc(high: str = "high", low: str = "low", period: int = 14) -> pl.Expr:
        """Aroon Oscillator via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.AROONOSC, [high, low], timeperiod=period)

    # -- Volatility --

    @staticmethod
    def atr(
        high: str = "high",
        low: str = "low",
        close: str = "close",
        period: int = 14,
    ) -> pl.Expr:
        """Average True Range via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.ATR, [high, low, close], timeperiod=period)

    @staticmethod
    def natr(
        high: str = "high",
        low: str = "low",
        close: str = "close",
        period: int = 14,
    ) -> pl.Expr:
        """Normalized Average True Range via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.NATR, [high, low, close], timeperiod=period)

    @staticmethod
    def trange(high: str = "high", low: str = "low", close: str = "close") -> pl.Expr:
        """True Range via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.TRANGE, [high, low, close])

    # -- Volume --

    @staticmethod
    def obv(close: str = "close", volume: str = "volume") -> pl.Expr:
        """On-Balance Volume via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.OBV, [close, volume])

    @staticmethod
    def ad(
        high: str = "high",
        low: str = "low",
        close: str = "close",
        volume: str = "volume",
    ) -> pl.Expr:
        """Chaikin A/D Line via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.AD, [high, low, close, volume])

    @staticmethod
    def adosc(
        high: str = "high",
        low: str = "low",
        close: str = "close",
        volume: str = "volume",
        fast: int = 3,
        slow: int = 10,
    ) -> pl.Expr:
        """Chaikin A/D Oscillator via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.ADOSC, [high, low, close, volume], fastperiod=fast, slowperiod=slow)

    # -- Pattern Recognition (candlestick patterns) --

    @staticmethod
    def cdl_doji(
        open: str = "open",  # noqa: A002
        high: str = "high",
        low: str = "low",
        close: str = "close",
    ) -> pl.Expr:
        """Doji candlestick pattern via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.CDLDOJI, [open, high, low, close])

    @staticmethod
    def cdl_hammer(
        open: str = "open",  # noqa: A002
        high: str = "high",
        low: str = "low",
        close: str = "close",
    ) -> pl.Expr:
        """Hammer candlestick pattern via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.CDLHAMMER, [open, high, low, close])

    @staticmethod
    def cdl_engulfing(
        open: str = "open",  # noqa: A002
        high: str = "high",
        low: str = "low",
        close: str = "close",
    ) -> pl.Expr:
        """Engulfing candlestick pattern via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.CDLENGULFING, [open, high, low, close])

    @staticmethod
    def cdl_morning_star(
        open: str = "open",  # noqa: A002
        high: str = "high",
        low: str = "low",
        close: str = "close",
        penetration: float = 0.3,
    ) -> pl.Expr:
        """Morning Star candlestick pattern via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.CDLMORNINGSTAR, [open, high, low, close], penetration=penetration)

    @staticmethod
    def cdl_evening_star(
        open: str = "open",  # noqa: A002
        high: str = "high",
        low: str = "low",
        close: str = "close",
        penetration: float = 0.3,
    ) -> pl.Expr:
        """Evening Star candlestick pattern via TA-Lib."""
        _require_talib()
        return talib_expr(_talib.CDLEVENINGSTAR, [open, high, low, close], penetration=penetration)


# Module-level convenience instance
ta = TALibIndicators()
