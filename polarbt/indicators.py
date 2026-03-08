"""
Technical indicators implemented as Polars expressions for maximum performance.

All functions return Polars expressions that can be used in .with_columns() calls.
"""

import numpy as np
import polars as pl


def sma(column: str, period: int) -> pl.Expr:
    """
    Simple Moving Average.

    Args:
        column: Name of the column to calculate SMA on
        period: Number of periods for the moving average

    Returns:
        Polars expression for SMA

    Example:
        df.with_columns([
            sma("close", 20).alias("sma_20")
        ])
    """
    return pl.col(column).rolling_mean(window_size=period)


def ema(column: str, period: int, adjust: bool = False) -> pl.Expr:
    """
    Exponential Moving Average.

    Args:
        column: Name of the column to calculate EMA on
        period: Number of periods for the moving average
        adjust: Whether to use adjusted weights (default False)

    Returns:
        Polars expression for EMA

    Example:
        df.with_columns([
            ema("close", 12).alias("ema_12")
        ])
    """
    alpha = 2.0 / (period + 1)
    # Make first value null to match SMA behavior
    return (
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col(column).ewm_mean(alpha=alpha, adjust=adjust))
    )


def rsi(column: str, period: int = 14) -> pl.Expr:
    """
    Relative Strength Index.

    Args:
        column: Name of the column to calculate RSI on
        period: Number of periods for RSI calculation (default 14)

    Returns:
        Polars expression for RSI

    Example:
        df.with_columns([
            rsi("close", 14).alias("rsi_14")
        ])
    """
    # Calculate price changes
    delta = pl.col(column).diff()

    # Separate gains and losses
    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)

    # Calculate average gains and losses using EMA
    avg_gain = gain.ewm_mean(alpha=1.0 / period, adjust=False)
    avg_loss = loss.ewm_mean(alpha=1.0 / period, adjust=False)

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi_value = 100 - (100 / (1 + rs))

    return rsi_value


def bollinger_bands(column: str, period: int = 20, std_dev: float = 2.0) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """
    Bollinger Bands.

    Args:
        column: Name of the column to calculate Bollinger Bands on
        period: Number of periods for the moving average (default 20)
        std_dev: Number of standard deviations for the bands (default 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band) Polars expressions

    Example:
        upper, middle, lower = bollinger_bands("close", 20, 2.0)
        df.with_columns([
            upper.alias("bb_upper"),
            middle.alias("bb_middle"),
            lower.alias("bb_lower")
        ])
    """
    middle = pl.col(column).rolling_mean(window_size=period)
    std = pl.col(column).rolling_std(window_size=period)
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return upper, middle, lower


def atr(high: str = "high", low: str = "low", close: str = "close", period: int = 14) -> pl.Expr:
    """
    Average True Range.

    Args:
        high: Name of the high price column
        low: Name of the low price column
        close: Name of the close price column
        period: Number of periods for ATR calculation (default 14)

    Returns:
        Polars expression for ATR

    Example:
        df.with_columns([
            atr("high", "low", "close", 14).alias("atr_14")
        ])
    """
    # True Range is the maximum of:
    # 1. Current high - current low
    # 2. Abs(current high - previous close)
    # 3. Abs(current low - previous close)

    prev_close = pl.col(close).shift(1)

    tr1 = pl.col(high) - pl.col(low)
    tr2 = (pl.col(high) - prev_close).abs()
    tr3 = (pl.col(low) - prev_close).abs()

    tr = pl.max_horizontal(tr1, tr2, tr3)

    # ATR is EMA of True Range
    atr_value = tr.ewm_mean(alpha=1.0 / period, adjust=False)

    return atr_value


def macd(column: str, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """
    Moving Average Convergence Divergence.

    Args:
        column: Name of the column to calculate MACD on
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram) Polars expressions

    Example:
        macd_line, signal_line, histogram = macd("close")
        df.with_columns([
            macd_line.alias("macd"),
            signal_line.alias("macd_signal"),
            histogram.alias("macd_hist")
        ])
    """
    fast_ema = ema(column, fast)
    slow_ema = ema(column, slow)

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm_mean(alpha=2.0 / (signal + 1), adjust=False)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def returns(column: str, periods: int = 1) -> pl.Expr:
    """
    Calculate percentage returns.

    Args:
        column: Name of the price column
        periods: Number of periods for return calculation (default 1)

    Returns:
        Polars expression for returns

    Example:
        df.with_columns([
            returns("close", 1).alias("returns")
        ])
    """
    return pl.col(column).pct_change(n=periods)


def log_returns(column: str, periods: int = 1) -> pl.Expr:
    """
    Calculate logarithmic returns.

    Args:
        column: Name of the price column
        periods: Number of periods for return calculation (default 1)

    Returns:
        Polars expression for log returns

    Example:
        df.with_columns([
            log_returns("close", 1).alias("log_returns")
        ])
    """
    return (pl.col(column) / pl.col(column).shift(periods)).log()


def crossover(fast_column: str, slow_column: str) -> pl.Expr:
    """
    Detect when fast crosses above slow.

    Args:
        fast_column: Name of the fast moving indicator
        slow_column: Name of the slow moving indicator

    Returns:
        Polars expression that is True when crossover occurs

    Example:
        df.with_columns([
            sma("close", 10).alias("sma_10"),
            sma("close", 20).alias("sma_20")
        ]).with_columns([
            crossover("sma_10", "sma_20").alias("golden_cross")
        ])
    """
    curr_above_or_equal = pl.col(fast_column) >= pl.col(slow_column)
    prev_below = pl.col(fast_column).shift(1) < pl.col(slow_column).shift(1)
    return curr_above_or_equal & prev_below


def crossunder(fast_column: str, slow_column: str) -> pl.Expr:
    """
    Detect when fast crosses below slow.

    Args:
        fast_column: Name of the fast moving indicator
        slow_column: Name of the slow moving indicator

    Returns:
        Polars expression that is True when crossunder occurs

    Example:
        df.with_columns([
            sma("close", 10).alias("sma_10"),
            sma("close", 20).alias("sma_20")
        ]).with_columns([
            crossunder("sma_10", "sma_20").alias("death_cross")
        ])
    """
    curr_below_or_equal = pl.col(fast_column) <= pl.col(slow_column)
    prev_above = pl.col(fast_column).shift(1) > pl.col(slow_column).shift(1)
    return curr_below_or_equal & prev_above


# ---------------------------------------------------------------------------
# Trend Indicators
# ---------------------------------------------------------------------------


def wma(column: str, period: int) -> pl.Expr:
    """Weighted Moving Average.

    Args:
        column: Name of the column to calculate WMA on.
        period: Number of periods for the moving average.

    Returns:
        Polars expression for WMA.
    """
    weights = np.arange(1, period + 1, dtype=np.float64)
    weight_sum = weights.sum()

    def _wma(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        out = np.full(len(arr), np.nan)
        for i in range(period - 1, len(arr)):
            window = arr[i - period + 1 : i + 1]
            if np.isnan(window).any():
                continue
            out[i] = np.dot(window, weights) / weight_sum
        return pl.Series(out)

    return pl.col(column).map_batches(_wma, return_dtype=pl.Float64)


def hma(column: str, period: int) -> pl.Expr:
    """Hull Moving Average.

    HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))

    Since HMA depends on intermediate WMA columns, this function operates
    on the full series via map_batches.

    Args:
        column: Name of the column to calculate HMA on.
        period: Number of periods.

    Returns:
        Polars expression for HMA.
    """
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)

    weights_half = np.arange(1, half_period + 1, dtype=np.float64)
    wsum_half = weights_half.sum()
    weights_full = np.arange(1, period + 1, dtype=np.float64)
    wsum_full = weights_full.sum()
    weights_sqrt = np.arange(1, sqrt_period + 1, dtype=np.float64)
    wsum_sqrt = weights_sqrt.sum()

    def _hma(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)

        def _calc_wma(data: np.ndarray, w: np.ndarray, ws: float, p: int) -> np.ndarray:
            out = np.full(len(data), np.nan)
            for i in range(p - 1, len(data)):
                window = data[i - p + 1 : i + 1]
                if np.isnan(window).any():
                    continue
                out[i] = np.dot(window, w) / ws
            return out

        wma_half = _calc_wma(arr, weights_half, wsum_half, half_period)
        wma_full = _calc_wma(arr, weights_full, wsum_full, period)
        diff = 2.0 * wma_half - wma_full
        result = _calc_wma(diff, weights_sqrt, wsum_sqrt, sqrt_period)
        result[np.isnan(result)] = np.nan
        return pl.Series(result)

    return pl.col(column).map_batches(_hma, return_dtype=pl.Float64)


def vwap(close: str = "close", volume: str = "volume", high: str | None = None, low: str | None = None) -> pl.Expr:
    """Volume Weighted Average Price.

    If *high* and *low* are provided the typical price ``(high + low + close) / 3``
    is used; otherwise the close price is used directly.

    Args:
        close: Name of the close price column.
        volume: Name of the volume column.
        high: Optional high price column for typical price.
        low: Optional low price column for typical price.

    Returns:
        Polars expression for VWAP.
    """
    if high is not None and low is not None:
        typical = (pl.col(high) + pl.col(low) + pl.col(close)) / 3.0
    else:
        typical = pl.col(close)
    return (typical * pl.col(volume)).cum_sum() / pl.col(volume).cum_sum()


def supertrend(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pl.Expr, pl.Expr]:
    """SuperTrend indicator.

    Returns two expressions: the SuperTrend line and a direction signal
    (1.0 for uptrend, -1.0 for downtrend).

    Args:
        high: Name of the high price column.
        low: Name of the low price column.
        close: Name of the close price column.
        period: ATR period (default 10).
        multiplier: ATR multiplier (default 3.0).

    Returns:
        Tuple of (supertrend_line, direction) Polars expressions.
    """

    def _supertrend(df: pl.DataFrame) -> pl.DataFrame:
        h = df[high].to_numpy().astype(np.float64)
        l = df[low].to_numpy().astype(np.float64)  # noqa: E741
        c = df[close].to_numpy().astype(np.float64)
        n = len(c)

        # True Range
        prev_c = np.empty(n)
        prev_c[0] = np.nan
        prev_c[1:] = c[:-1]
        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))

        # ATR via EMA
        atr_arr = np.full(n, np.nan)
        alpha = 1.0 / period
        first_valid = 1  # first bar with valid TR
        atr_arr[first_valid] = tr[first_valid]
        for i in range(first_valid + 1, n):
            atr_arr[i] = alpha * tr[i] + (1.0 - alpha) * atr_arr[i - 1]

        hl2 = (h + l) / 2.0
        upper_basic = hl2 + multiplier * atr_arr
        lower_basic = hl2 - multiplier * atr_arr

        upper_band = np.copy(upper_basic)
        lower_band = np.copy(lower_basic)
        direction = np.ones(n)
        st = np.full(n, np.nan)

        for i in range(1, n):
            if lower_basic[i] > lower_band[i - 1] or c[i - 1] < lower_band[i - 1]:
                lower_band[i] = lower_basic[i]
            else:
                lower_band[i] = lower_band[i - 1]

            if upper_basic[i] < upper_band[i - 1] or c[i - 1] > upper_band[i - 1]:
                upper_band[i] = upper_basic[i]
            else:
                upper_band[i] = upper_band[i - 1]

            if direction[i - 1] == 1.0:
                if c[i] < lower_band[i]:
                    direction[i] = -1.0
                else:
                    direction[i] = 1.0
            else:
                if c[i] > upper_band[i]:
                    direction[i] = 1.0
                else:
                    direction[i] = -1.0

            st[i] = lower_band[i] if direction[i] == 1.0 else upper_band[i]

        return pl.DataFrame({"_st_line": st, "_st_dir": direction})

    st_line = pl.struct([high, low, close]).map_batches(
        lambda s: _supertrend(s.struct.unnest())["_st_line"], return_dtype=pl.Float64
    )
    st_dir = pl.struct([high, low, close]).map_batches(
        lambda s: _supertrend(s.struct.unnest())["_st_dir"], return_dtype=pl.Float64
    )
    return st_line, st_dir


def adx(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    period: int = 14,
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Average Directional Index.

    Returns three expressions: ADX, +DI, -DI.

    Args:
        high: Name of the high price column.
        low: Name of the low price column.
        close: Name of the close price column.
        period: Smoothing period (default 14).

    Returns:
        Tuple of (adx, plus_di, minus_di) Polars expressions.
    """

    def _adx_calc(df: pl.DataFrame) -> pl.DataFrame:
        h = df[high].to_numpy().astype(np.float64)
        l = df[low].to_numpy().astype(np.float64)  # noqa: E741
        c = df[close].to_numpy().astype(np.float64)
        n = len(c)

        prev_h = np.empty(n)
        prev_h[0] = np.nan
        prev_h[1:] = h[:-1]
        prev_l = np.empty(n)
        prev_l[0] = np.nan
        prev_l[1:] = l[:-1]
        prev_c = np.empty(n)
        prev_c[0] = np.nan
        prev_c[1:] = c[:-1]

        plus_dm = np.where((h - prev_h) > (prev_l - l), np.maximum(h - prev_h, 0), 0.0)
        minus_dm = np.where((prev_l - l) > (h - prev_h), np.maximum(prev_l - l, 0), 0.0)
        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))

        alpha = 1.0 / period
        atr_s = np.full(n, np.nan)
        plus_dm_s = np.full(n, np.nan)
        minus_dm_s = np.full(n, np.nan)

        atr_s[1] = tr[1]
        plus_dm_s[1] = plus_dm[1]
        minus_dm_s[1] = minus_dm[1]

        for i in range(2, n):
            atr_s[i] = alpha * tr[i] + (1 - alpha) * atr_s[i - 1]
            plus_dm_s[i] = alpha * plus_dm[i] + (1 - alpha) * plus_dm_s[i - 1]
            minus_dm_s[i] = alpha * minus_dm[i] + (1 - alpha) * minus_dm_s[i - 1]

        plus_di = 100.0 * plus_dm_s / atr_s
        minus_di = 100.0 * minus_dm_s / atr_s
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di)

        adx_arr = np.full(n, np.nan)
        first_valid = np.argmax(~np.isnan(dx))
        if first_valid < n:
            adx_arr[first_valid] = dx[first_valid]
            for i in range(first_valid + 1, n):
                if np.isnan(dx[i]):
                    continue
                adx_arr[i] = alpha * dx[i] + (1 - alpha) * adx_arr[i - 1]

        return pl.DataFrame({"_adx": adx_arr, "_plus_di": plus_di, "_minus_di": minus_di})

    adx_expr = pl.struct([high, low, close]).map_batches(
        lambda s: _adx_calc(s.struct.unnest())["_adx"], return_dtype=pl.Float64
    )
    plus_di = pl.struct([high, low, close]).map_batches(
        lambda s: _adx_calc(s.struct.unnest())["_plus_di"], return_dtype=pl.Float64
    )
    minus_di = pl.struct([high, low, close]).map_batches(
        lambda s: _adx_calc(s.struct.unnest())["_minus_di"], return_dtype=pl.Float64
    )
    return adx_expr, plus_di, minus_di


# ---------------------------------------------------------------------------
# Momentum Indicators
# ---------------------------------------------------------------------------


def stochastic(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pl.Expr, pl.Expr]:
    """Stochastic Oscillator (%K and %D).

    Args:
        high: Name of the high price column.
        low: Name of the low price column.
        close: Name of the close price column.
        k_period: Look-back period for %K (default 14).
        d_period: Smoothing period for %D (default 3).

    Returns:
        Tuple of (%K, %D) Polars expressions.
    """
    lowest_low = pl.col(low).rolling_min(window_size=k_period)
    highest_high = pl.col(high).rolling_max(window_size=k_period)
    k = 100.0 * (pl.col(close) - lowest_low) / (highest_high - lowest_low)
    d = k.rolling_mean(window_size=d_period)
    return k, d


def williams_r(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    period: int = 14,
) -> pl.Expr:
    """Williams %R.

    Args:
        high: Name of the high price column.
        low: Name of the low price column.
        close: Name of the close price column.
        period: Look-back period (default 14).

    Returns:
        Polars expression for Williams %R (range -100 to 0).
    """
    highest_high = pl.col(high).rolling_max(window_size=period)
    lowest_low = pl.col(low).rolling_min(window_size=period)
    return -100.0 * (highest_high - pl.col(close)) / (highest_high - lowest_low)


def cci(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    period: int = 20,
) -> pl.Expr:
    """Commodity Channel Index.

    Args:
        high: Name of the high price column.
        low: Name of the low price column.
        close: Name of the close price column.
        period: Look-back period (default 20).

    Returns:
        Polars expression for CCI.
    """

    def _mean_dev(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        out = np.full(len(arr), np.nan)
        for i in range(period - 1, len(arr)):
            window = arr[i - period + 1 : i + 1]
            out[i] = np.mean(np.abs(window - np.mean(window)))
        return pl.Series(out)

    # CCI = (TP - SMA(TP)) / (0.015 * mean_deviation)
    # We need to compute mean deviation via map_batches on the typical price
    # but typical is an expression. We use a struct to pass all needed columns.
    def _cci_calc(df: pl.DataFrame) -> pl.DataFrame:
        tp = ((df[high].to_numpy() + df[low].to_numpy() + df[close].to_numpy()) / 3.0).astype(np.float64)
        n = len(tp)
        tp_mean = np.full(n, np.nan)
        mean_dev = np.full(n, np.nan)
        for i in range(period - 1, n):
            window = tp[i - period + 1 : i + 1]
            m = np.mean(window)
            tp_mean[i] = m
            mean_dev[i] = np.mean(np.abs(window - m))
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(mean_dev != 0, (tp - tp_mean) / (0.015 * mean_dev), 0.0)
        result[: period - 1] = np.nan
        return pl.DataFrame({"_cci": result})

    return pl.struct([high, low, close]).map_batches(
        lambda s: _cci_calc(s.struct.unnest())["_cci"], return_dtype=pl.Float64
    )


def mfi(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume",
    period: int = 14,
) -> pl.Expr:
    """Money Flow Index.

    Args:
        high: Name of the high price column.
        low: Name of the low price column.
        close: Name of the close price column.
        volume: Name of the volume column.
        period: Look-back period (default 14).

    Returns:
        Polars expression for MFI (range 0 to 100).
    """

    def _mfi_calc(df: pl.DataFrame) -> pl.DataFrame:
        h = df[high].to_numpy().astype(np.float64)
        l = df[low].to_numpy().astype(np.float64)  # noqa: E741
        c = df[close].to_numpy().astype(np.float64)
        v = df[volume].to_numpy().astype(np.float64)
        n = len(c)

        tp = (h + l + c) / 3.0
        raw_mf = tp * v

        result = np.full(n, np.nan)
        for i in range(period, n):
            pos_flow = 0.0
            neg_flow = 0.0
            for j in range(i - period + 1, i + 1):
                if tp[j] > tp[j - 1]:
                    pos_flow += raw_mf[j]
                elif tp[j] < tp[j - 1]:
                    neg_flow += raw_mf[j]
            if neg_flow == 0:
                result[i] = 100.0
            else:
                mfr = pos_flow / neg_flow
                result[i] = 100.0 - 100.0 / (1.0 + mfr)
        return pl.DataFrame({"_mfi": result})

    return pl.struct([high, low, close, volume]).map_batches(
        lambda s: _mfi_calc(s.struct.unnest())["_mfi"], return_dtype=pl.Float64
    )


def roc(column: str, period: int = 12) -> pl.Expr:
    """Rate of Change.

    Args:
        column: Name of the column.
        period: Look-back period (default 12).

    Returns:
        Polars expression for ROC (percentage).
    """
    return (pl.col(column) - pl.col(column).shift(period)) / pl.col(column).shift(period) * 100.0


# ---------------------------------------------------------------------------
# Volatility Indicators
# ---------------------------------------------------------------------------


def keltner_channels(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Keltner Channels.

    Args:
        high: Name of the high price column.
        low: Name of the low price column.
        close: Name of the close price column.
        ema_period: EMA period for the middle band (default 20).
        atr_period: ATR period (default 10).
        multiplier: ATR multiplier for band width (default 2.0).

    Returns:
        Tuple of (upper, middle, lower) Polars expressions.
    """
    middle = ema(close, ema_period)
    atr_val = atr(high, low, close, atr_period)
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val
    return upper, middle, lower


def donchian_channels(
    high: str = "high",
    low: str = "low",
    period: int = 20,
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Donchian Channels.

    Args:
        high: Name of the high price column.
        low: Name of the low price column.
        period: Look-back period (default 20).

    Returns:
        Tuple of (upper, middle, lower) Polars expressions.
    """
    upper = pl.col(high).rolling_max(window_size=period)
    lower = pl.col(low).rolling_min(window_size=period)
    middle = (upper + lower) / 2.0
    return upper, middle, lower


# ---------------------------------------------------------------------------
# Volume Indicators
# ---------------------------------------------------------------------------


def obv(close: str = "close", volume: str = "volume") -> pl.Expr:
    """On-Balance Volume.

    Args:
        close: Name of the close price column.
        volume: Name of the volume column.

    Returns:
        Polars expression for OBV.
    """
    direction = (
        pl.when(pl.col(close) > pl.col(close).shift(1))
        .then(pl.col(volume))
        .when(pl.col(close) < pl.col(close).shift(1))
        .then(-pl.col(volume))
        .otherwise(pl.lit(0))
    )
    return direction.cum_sum()


def ad_line(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume",
) -> pl.Expr:
    """Accumulation/Distribution Line.

    Args:
        high: Name of the high price column.
        low: Name of the low price column.
        close: Name of the close price column.
        volume: Name of the volume column.

    Returns:
        Polars expression for A/D Line.
    """
    hl_range = pl.col(high) - pl.col(low)
    clv = (
        pl.when(hl_range != 0)
        .then(((pl.col(close) - pl.col(low)) - (pl.col(high) - pl.col(close))) / hl_range)
        .otherwise(0.0)
    )
    return (clv * pl.col(volume)).cum_sum()


# ---------------------------------------------------------------------------
# Support / Resistance
# ---------------------------------------------------------------------------


def pivot_points(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    method: str = "standard",
) -> dict[str, pl.Expr]:
    """Pivot Points calculated from previous bar's high, low, close.

    Args:
        high: Name of the high price column.
        low: Name of the low price column.
        close: Name of the close price column.
        method: One of "standard", "fibonacci", "woodie", "camarilla".

    Returns:
        Dict mapping names (pp, r1, r2, r3, s1, s2, s3) to Polars expressions.
        Use previous bar values (shifted by 1).
    """
    prev_h = pl.col(high).shift(1)
    prev_l = pl.col(low).shift(1)
    prev_c = pl.col(close).shift(1)

    if method == "standard":
        pp = (prev_h + prev_l + prev_c) / 3.0
        return {
            "pp": pp,
            "r1": 2.0 * pp - prev_l,
            "s1": 2.0 * pp - prev_h,
            "r2": pp + (prev_h - prev_l),
            "s2": pp - (prev_h - prev_l),
            "r3": prev_h + 2.0 * (pp - prev_l),
            "s3": prev_l - 2.0 * (prev_h - pp),
        }
    elif method == "fibonacci":
        pp = (prev_h + prev_l + prev_c) / 3.0
        diff = prev_h - prev_l
        return {
            "pp": pp,
            "r1": pp + 0.382 * diff,
            "s1": pp - 0.382 * diff,
            "r2": pp + 0.618 * diff,
            "s2": pp - 0.618 * diff,
            "r3": pp + 1.000 * diff,
            "s3": pp - 1.000 * diff,
        }
    elif method == "woodie":
        pp = (prev_h + prev_l + 2.0 * prev_c) / 4.0
        return {
            "pp": pp,
            "r1": 2.0 * pp - prev_l,
            "s1": 2.0 * pp - prev_h,
            "r2": pp + (prev_h - prev_l),
            "s2": pp - (prev_h - prev_l),
        }
    elif method == "camarilla":
        pp = (prev_h + prev_l + prev_c) / 3.0
        diff = prev_h - prev_l
        return {
            "pp": pp,
            "r1": prev_c + diff * 1.1 / 12.0,
            "s1": prev_c - diff * 1.1 / 12.0,
            "r2": prev_c + diff * 1.1 / 6.0,
            "s2": prev_c - diff * 1.1 / 6.0,
            "r3": prev_c + diff * 1.1 / 4.0,
            "s3": prev_c - diff * 1.1 / 4.0,
        }
    else:
        raise ValueError(f"Unknown pivot point method: {method!r}. Use 'standard', 'fibonacci', 'woodie', 'camarilla'.")
