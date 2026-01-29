"""
Technical indicators implemented as Polars expressions for maximum performance.

All functions return Polars expressions that can be used in .with_columns() calls.
"""

import polars as pl
from typing import Union


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
    return pl.col(column).ewm_mean(alpha=alpha, adjust=adjust)


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


def bollinger_bands(
    column: str, period: int = 20, std_dev: float = 2.0
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
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


def atr(
    high: str = "high", low: str = "low", close: str = "close", period: int = 14
) -> pl.Expr:
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


def macd(
    column: str, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
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
    curr_above = pl.col(fast_column) > pl.col(slow_column)
    prev_below = pl.col(fast_column).shift(1) <= pl.col(slow_column).shift(1)
    return curr_above & prev_below


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
    curr_below = pl.col(fast_column) < pl.col(slow_column)
    prev_above = pl.col(fast_column).shift(1) >= pl.col(slow_column).shift(1)
    return curr_below & prev_above
