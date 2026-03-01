"""Data cleaning functions for OHLCV DataFrames."""

from datetime import timedelta

import polars as pl


def fill_gaps(
    df: pl.DataFrame,
    timestamp_column: str = "timestamp",
    interval: str | timedelta | None = None,
    method: str = "forward",
) -> pl.DataFrame:
    """Fill timestamp gaps by inserting missing rows and forward/backward filling values.

    Args:
        df: The DataFrame with a timestamp column.
        timestamp_column: Name of the timestamp column.
        interval: Expected interval between rows (e.g., "1h", "1d", timedelta(minutes=5)).
            If None, the interval is inferred from the most common difference.
        method: Fill method — "forward" (default), "backward", or "null" (insert nulls).

    Returns:
        DataFrame with gaps filled.

    Raises:
        ValueError: If timestamp column is missing or method is invalid.
    """
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")

    if method not in ("forward", "backward", "null"):
        raise ValueError(f"Invalid fill method '{method}', must be 'forward', 'backward', or 'null'")

    if df.height < 2:
        return df

    df = df.sort(timestamp_column)

    if interval is None:
        interval = _infer_interval(df, timestamp_column)

    if isinstance(interval, str):
        interval = _parse_interval_string(interval)

    # Generate complete range
    ts_min = df[timestamp_column].min()
    ts_max = df[timestamp_column].max()

    full_range = pl.DataFrame({timestamp_column: pl.datetime_range(ts_min, ts_max, interval, eager=True)})

    # Left join original data onto full range
    result = full_range.join(df, on=timestamp_column, how="left")

    # Fill nulls
    if method == "forward":
        value_cols = [c for c in result.columns if c != timestamp_column]
        result = result.with_columns([pl.col(c).forward_fill() for c in value_cols])
    elif method == "backward":
        value_cols = [c for c in result.columns if c != timestamp_column]
        result = result.with_columns([pl.col(c).backward_fill() for c in value_cols])

    return result


def adjust_splits(
    df: pl.DataFrame,
    splits: list[tuple[str, float]],
    timestamp_column: str = "timestamp",
    price_columns: list[str] | None = None,
    volume_column: str = "volume",
) -> pl.DataFrame:
    """Adjust OHLCV data for stock splits.

    Applies split ratios retroactively: prices before the split date are divided
    by the ratio, volume is multiplied by the ratio.

    Args:
        df: The DataFrame to adjust.
        splits: List of (date_string, ratio) tuples. A 2:1 split has ratio=2.0,
            meaning each share becomes 2 shares and price halves.
        timestamp_column: Name of the timestamp column.
        price_columns: Columns to adjust for price. Defaults to open/high/low/close.
        volume_column: Column to adjust for volume.

    Returns:
        DataFrame with split-adjusted prices and volume.

    Raises:
        ValueError: If timestamp column is missing.
    """
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")

    if price_columns is None:
        price_columns = [c for c in ["open", "high", "low", "close"] if c in df.columns]

    result = df.clone()

    for split_date_str, ratio in splits:
        split_date = _parse_date(split_date_str, result[timestamp_column].dtype)
        before_split = pl.col(timestamp_column) < split_date

        # Divide prices before split by ratio
        price_adjustments = [
            pl.when(before_split).then(pl.col(c) / ratio).otherwise(pl.col(c)).alias(c) for c in price_columns
        ]

        # Multiply volume before split by ratio
        vol_adjustments = []
        if volume_column in result.columns:
            vol_adjustments = [
                pl.when(before_split)
                .then(pl.col(volume_column) * ratio)
                .otherwise(pl.col(volume_column))
                .alias(volume_column)
            ]

        result = result.with_columns(price_adjustments + vol_adjustments)

    return result


def drop_zero_volume(df: pl.DataFrame, volume_column: str = "volume") -> pl.DataFrame:
    """Remove rows where volume is zero or null.

    Args:
        df: The DataFrame to filter.
        volume_column: Name of the volume column.

    Returns:
        Filtered DataFrame.
    """
    if volume_column not in df.columns:
        return df
    return df.filter((pl.col(volume_column) > 0) & pl.col(volume_column).is_not_null())


def clip_outliers(
    df: pl.DataFrame,
    columns: list[str] | None = None,
    lower_quantile: float = 0.001,
    upper_quantile: float = 0.999,
) -> pl.DataFrame:
    """Clip extreme values in specified columns to quantile bounds.

    Args:
        df: The DataFrame to process.
        columns: Columns to clip. Defaults to open/high/low/close.
        lower_quantile: Lower quantile bound (default 0.1%).
        upper_quantile: Upper quantile bound (default 99.9%).

    Returns:
        DataFrame with outliers clipped.
    """
    if columns is None:
        columns = [c for c in ["open", "high", "low", "close"] if c in df.columns]

    exprs = []
    for col in columns:
        if col not in df.columns:
            continue
        lower = df[col].quantile(lower_quantile)
        upper = df[col].quantile(upper_quantile)
        exprs.append(pl.col(col).clip(lower, upper).alias(col))

    if exprs:
        return df.with_columns(exprs)
    return df


def _infer_interval(df: pl.DataFrame, timestamp_column: str) -> timedelta:
    """Infer the most common interval between consecutive timestamps."""
    diffs = df[timestamp_column].diff().drop_nulls()
    # Use mode (most common difference)
    mode_val = diffs.mode().to_list()[0]
    if isinstance(mode_val, timedelta):
        return mode_val
    raise ValueError(f"Could not infer interval from timestamp column '{timestamp_column}'")


def _parse_interval_string(interval: str) -> timedelta:
    """Parse interval strings like '1h', '5m', '1d' into timedelta."""
    units = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
    if not interval:
        raise ValueError("Empty interval string")
    unit = interval[-1].lower()
    if unit not in units:
        raise ValueError(f"Unknown interval unit '{unit}', use one of {list(units.keys())}")
    value = int(interval[:-1])
    return timedelta(**{units[unit]: value})


def _parse_date(date_str: str, dtype: pl.DataType) -> object:
    """Parse a date string into a type compatible with the timestamp column."""
    from datetime import datetime

    dt = datetime.fromisoformat(date_str)
    if dtype == pl.Date:
        return dt.date()
    return dt
