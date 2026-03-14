"""Trade-level data validation and aggregation for DEX/AMM sources.

Provides utilities to validate raw trade data and aggregate it into OHLCV bars
suitable for backtesting. Supports both time-based and trade-count-based aggregation.
"""

from typing import Any

import polars as pl

from polarbt.data.validation import ValidationResult

REQUIRED_TRADE_COLUMNS = {"timestamp", "symbol", "price", "amount", "side"}
OPTIONAL_TRADE_COLUMNS = {"base_amount", "pool_reserve", "trader", "tx_id"}
VALID_SIDES = {"buy", "sell"}


def validate_trades(df: pl.DataFrame) -> ValidationResult:
    """Validate a trade-level DataFrame against the trade data standard.

    Checks required columns, dtypes, side values, positive price/amount,
    no nulls in required columns, and per-symbol timestamp ordering.
    Warns on duplicate tx_id values if the column is present.

    Args:
        df: The DataFrame to validate.

    Returns:
        ValidationResult with any errors and warnings.
    """
    result = ValidationResult()

    # Check required columns
    cols = set(df.columns)
    missing = REQUIRED_TRADE_COLUMNS - cols
    if missing:
        result.valid = False
        result.errors.append(f"Missing required columns: {sorted(missing)}")
        return result

    # Check dtypes
    _validate_trade_dtypes(df, result)

    # Check no nulls in required columns
    for col in REQUIRED_TRADE_COLUMNS:
        null_count = df[col].null_count()
        if null_count > 0:
            result.valid = False
            result.errors.append(f"Column '{col}' contains {null_count} null values")

    # Check side values
    unique_sides = set(df["side"].unique().to_list())
    invalid_sides = unique_sides - VALID_SIDES
    if invalid_sides:
        result.valid = False
        result.errors.append(f"Column 'side' contains invalid values: {sorted(invalid_sides)}")

    # Check price and amount are positive
    for col in ("price", "amount"):
        if df[col].dtype in _NUMERIC_DTYPES:
            non_positive = df.filter(pl.col(col) <= 0).height
            if non_positive > 0:
                result.valid = False
                result.errors.append(f"Column '{col}' contains {non_positive} non-positive values")

    # Check per-symbol timestamp ordering
    _validate_per_symbol_timestamps(df, result)

    # Warn on duplicate tx_id
    if "tx_id" in cols:
        n_dupes = df["tx_id"].len() - df["tx_id"].n_unique()
        if n_dupes > 0:
            result.warnings.append(f"Column 'tx_id' contains {n_dupes} duplicate values")

    return result


def aggregate_trades(
    df: pl.DataFrame,
    interval: str,
    *,
    exchange_rate: pl.DataFrame | None = None,
    min_trades: int = 1,
    extra_aggs: dict[str, pl.Expr] | None = None,
) -> pl.DataFrame:
    """Aggregate trade-level data into OHLCV bars per symbol using time buckets.

    Converts raw trades into fixed-interval bars with OHLCV prices, volume
    breakdown, VWAP, and optional USD-denominated volumes via exchange rate.

    Args:
        df: Validated trade DataFrame with required columns.
        interval: Bar duration string (e.g. "1m", "5m", "1h", "1d").
        exchange_rate: Optional DataFrame with (timestamp, rate) columns for
            quote-to-USD conversion. Rate is forward-filled to bar timestamps.
        min_trades: Minimum trades per bar to emit. Bars with fewer trades
            are dropped.
        extra_aggs: Custom aggregation expressions for pass-through columns.
            Keys are output column names, values are Polars expressions.

    Returns:
        Long-format DataFrame with OHLCV bars per symbol.
    """
    agg_exprs: list[pl.Expr] = [
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("amount").sum().alias("volume"),
        pl.col("price").count().alias("trades"),
        pl.col("amount").filter(pl.col("side") == "buy").sum().alias("buy_volume"),
        pl.col("amount").filter(pl.col("side") == "sell").sum().alias("sell_volume"),
        (pl.col("amount") * pl.col("price")).sum().alias("_vwap_num"),
    ]

    # Optional columns
    has_trader = "trader" in df.columns
    has_pool_reserve = "pool_reserve" in df.columns

    if has_trader:
        agg_exprs.append(pl.col("trader").n_unique().alias("unique_traders"))
    if has_pool_reserve:
        agg_exprs.append(pl.col("pool_reserve").last().alias("pool_reserve_last"))

    if extra_aggs:
        agg_exprs.extend(expr.alias(name) for name, expr in extra_aggs.items())

    result = (
        df.sort("timestamp")
        .group_by_dynamic("timestamp", every=interval, group_by="symbol")
        .agg(agg_exprs)
        .with_columns(
            (pl.col("_vwap_num") / pl.col("volume")).alias("vwap"),
        )
        .drop("_vwap_num")
    )

    # Filter by min_trades
    if min_trades > 1:
        result = result.filter(pl.col("trades") >= min_trades)

    # Exchange rate conversion
    if exchange_rate is not None:
        result = _apply_exchange_rate(result, exchange_rate)

    return result.sort("symbol", "timestamp")


def aggregate_trades_by_count(
    df: pl.DataFrame,
    n_trades: int,
    *,
    extra_aggs: dict[str, pl.Expr] | None = None,
) -> pl.DataFrame:
    """Aggregate trade-level data into bars of fixed trade count per symbol.

    Groups every N trades into one bar instead of using fixed time intervals.
    Useful for tokens with highly irregular trading activity.

    Args:
        df: Validated trade DataFrame with required columns.
        n_trades: Number of trades per bar.
        extra_aggs: Custom aggregation expressions for pass-through columns.

    Returns:
        Long-format DataFrame with OHLCV bars per symbol.
    """
    if n_trades < 1:
        raise ValueError(f"n_trades must be >= 1, got {n_trades}")

    df_sorted = df.sort("symbol", "timestamp")

    # Assign bar index per symbol: integer division of row number within each symbol group
    df_with_bar = df_sorted.with_columns(
        (pl.col("timestamp").cum_count().over("symbol") - 1).floordiv(n_trades).alias("_bar_idx"),
    )

    agg_exprs: list[pl.Expr] = [
        pl.col("timestamp").first().alias("timestamp"),
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("amount").sum().alias("volume"),
        pl.col("price").count().alias("trades"),
        pl.col("amount").filter(pl.col("side") == "buy").sum().alias("buy_volume"),
        pl.col("amount").filter(pl.col("side") == "sell").sum().alias("sell_volume"),
        (pl.col("amount") * pl.col("price")).sum().alias("_vwap_num"),
    ]

    has_trader = "trader" in df.columns
    has_pool_reserve = "pool_reserve" in df.columns

    if has_trader:
        agg_exprs.append(pl.col("trader").n_unique().alias("unique_traders"))
    if has_pool_reserve:
        agg_exprs.append(pl.col("pool_reserve").last().alias("pool_reserve_last"))

    if extra_aggs:
        agg_exprs.extend(expr.alias(name) for name, expr in extra_aggs.items())

    result = (
        df_with_bar.group_by("symbol", "_bar_idx")
        .agg(agg_exprs)
        .with_columns(
            (pl.col("_vwap_num") / pl.col("volume")).alias("vwap"),
        )
        .drop("_bar_idx", "_vwap_num")
        .sort("symbol", "timestamp")
    )

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_NUMERIC_DTYPES = {
    pl.Float32,
    pl.Float64,
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
}

_TIMESTAMP_DTYPES = {pl.Date, pl.Datetime}

_EXPECTED_DTYPES: dict[str, tuple[str, set[Any]]] = {
    "timestamp": ("temporal", _TIMESTAMP_DTYPES),
    "symbol": ("string", {pl.Utf8, pl.String}),
    "price": ("numeric", _NUMERIC_DTYPES),
    "amount": ("numeric", _NUMERIC_DTYPES),
    "side": ("string", {pl.Utf8, pl.String}),
}


def _validate_trade_dtypes(df: pl.DataFrame, result: ValidationResult) -> None:
    """Check that trade columns have appropriate data types."""
    for col, (expected_kind, valid_dtypes) in _EXPECTED_DTYPES.items():
        if col not in df.columns:
            continue
        dtype = df[col].dtype
        # For Datetime with parameters, check the base type
        if isinstance(dtype, pl.Datetime):
            if pl.Datetime not in valid_dtypes:
                result.valid = False
                result.errors.append(f"Column '{col}' has dtype {dtype}, expected {expected_kind}")
        elif dtype not in valid_dtypes:
            result.valid = False
            result.errors.append(f"Column '{col}' has dtype {dtype}, expected {expected_kind}")


def _validate_per_symbol_timestamps(df: pl.DataFrame, result: ValidationResult) -> None:
    """Check that timestamps are sorted ascending within each symbol."""
    # Detect symbols where timestamp decreases between consecutive rows
    unsorted_symbols = (
        df.with_columns((pl.col("timestamp") < pl.col("timestamp").shift(1).over("symbol")).alias("_out_of_order"))
        .group_by("symbol")
        .agg(pl.col("_out_of_order").sum().alias("_n_unsorted"))
        .filter(pl.col("_n_unsorted") > 0)
    )
    if unsorted_symbols.height > 0:
        count = unsorted_symbols.height
        examples = unsorted_symbols["symbol"].head(5).to_list()
        result.valid = False
        result.errors.append(f"Timestamps not sorted ascending for {count} symbol(s), e.g.: {examples}")


def _apply_exchange_rate(result: pl.DataFrame, exchange_rate: pl.DataFrame) -> pl.DataFrame:
    """Join exchange rate and compute USD-denominated volume columns."""
    rate_df = exchange_rate.sort("timestamp").select(
        pl.col("timestamp").alias("_rate_ts"),
        pl.col("rate"),
    )

    result = result.sort("timestamp").join_asof(
        rate_df,
        left_on="timestamp",
        right_on="_rate_ts",
        strategy="backward",
    )

    result = result.with_columns(
        (pl.col("volume") * pl.col("rate")).alias("volume_usd"),
        (pl.col("buy_volume") * pl.col("rate")).alias("buy_volume_usd"),
        (pl.col("sell_volume") * pl.col("rate")).alias("sell_volume_usd"),
    ).drop("rate")

    return result
