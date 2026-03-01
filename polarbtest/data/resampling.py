"""OHLCV resampling functions."""

from datetime import timedelta
from typing import Literal

import polars as pl


def resample_ohlcv(
    df: pl.DataFrame,
    interval: str | timedelta,
    timestamp_column: str = "timestamp",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    label: Literal["left", "right", "datapoint"] = "left",
) -> pl.DataFrame:
    """Resample OHLCV data to a larger timeframe.

    Aggregates bars using standard OHLCV logic: first open, max high, min low,
    last close, sum volume.

    Args:
        df: The DataFrame with OHLCV data and a timestamp column.
        interval: Target interval (e.g., "1h", "4h", "1d", or timedelta).
        timestamp_column: Name of the timestamp column.
        open_col: Name of the open column.
        high_col: Name of the high column.
        low_col: Name of the low column.
        close_col: Name of the close column.
        volume_col: Name of the volume column.
        label: Which edge of the interval to use as label — "left" (default) or "right".

    Returns:
        Resampled DataFrame with OHLCV columns plus any extra numeric columns (aggregated as last).

    Raises:
        ValueError: If timestamp column is missing.
    """
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")

    if isinstance(interval, timedelta):
        interval = _timedelta_to_polars_interval(interval)

    df = df.sort(timestamp_column)

    # Build aggregation expressions
    agg_exprs: list[pl.Expr] = []

    ohlcv_cols = {open_col, high_col, low_col, close_col, volume_col}

    if open_col in df.columns:
        agg_exprs.append(pl.col(open_col).first().alias(open_col))
    if high_col in df.columns:
        agg_exprs.append(pl.col(high_col).max().alias(high_col))
    if low_col in df.columns:
        agg_exprs.append(pl.col(low_col).min().alias(low_col))
    if close_col in df.columns:
        agg_exprs.append(pl.col(close_col).last().alias(close_col))
    if volume_col in df.columns:
        agg_exprs.append(pl.col(volume_col).sum().alias(volume_col))

    # Aggregate other numeric columns as last value
    for col in df.columns:
        if col == timestamp_column or col in ohlcv_cols:
            continue
        if df[col].dtype.is_numeric():
            agg_exprs.append(pl.col(col).last().alias(col))

    result = df.group_by_dynamic(
        timestamp_column,
        every=interval,
        label=label,
    ).agg(agg_exprs)

    return result


def _timedelta_to_polars_interval(td: timedelta) -> str:
    """Convert a timedelta to a Polars interval string."""
    total_seconds = int(td.total_seconds())
    if total_seconds % 86400 == 0:
        return f"{total_seconds // 86400}d"
    if total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}m"
    return f"{total_seconds}s"
