"""DeFi and memecoin indicators as Polars expressions.

Designed for aggregated bar data produced by ``data.aggregate_trades()``.
All functions return Polars expressions for use in ``.with_columns()`` calls,
typically with ``.over("symbol")`` for per-token partitioning.

Example::

    df = df.with_columns(
        defi.buy_sell_ratio().over("symbol").alias("bs_ratio"),
        defi.trade_intensity(window=10).over("symbol").alias("trade_accel"),
        defi.pump_detector(price_std=2.0, volume_std=2.0, window=20).over("symbol").alias("is_pump"),
    )
"""

import polars as pl

# ---------------------------------------------------------------------------
# Token activity indicators
# ---------------------------------------------------------------------------


def token_age(symbol_col: str = "symbol") -> pl.Expr:
    """Bars since first appearance of each symbol.

    Returns a 1-based count: first bar = 1, second bar = 2, etc.
    Use with ``.over(symbol_col)`` for per-symbol partitioning.

    Args:
        symbol_col: Name of the symbol column for partitioning context.
    """
    return pl.int_range(1, pl.len() + 1).over(symbol_col)


def buy_sell_ratio(
    buy_vol_col: str = "buy_volume",
    sell_vol_col: str = "sell_volume",
) -> pl.Expr:
    """Buy volume as fraction of total volume per bar.

    Returns values in [0, 1]. A value of 0.5 means equal buy and sell volume.
    Returns null when total volume is zero.

    Args:
        buy_vol_col: Name of the buy volume column.
        sell_vol_col: Name of the sell volume column.
    """
    total = pl.col(buy_vol_col) + pl.col(sell_vol_col)
    return pl.when(total > 0).then(pl.col(buy_vol_col) / total).otherwise(None)


def net_flow(
    buy_vol_col: str = "buy_volume",
    sell_vol_col: str = "sell_volume",
) -> pl.Expr:
    """Net volume flow: buy_volume - sell_volume.

    Positive values indicate net buying pressure; negative indicates selling.

    Args:
        buy_vol_col: Name of the buy volume column.
        sell_vol_col: Name of the sell volume column.
    """
    return pl.col(buy_vol_col) - pl.col(sell_vol_col)


def trade_intensity(trades_col: str = "trades", window: int = 10) -> pl.Expr:
    """Rolling trade count acceleration.

    Computes the difference between current trade count and its rolling mean,
    normalized by the rolling standard deviation (z-score). High values indicate
    unusual trade activity.

    Args:
        trades_col: Name of the trade count column.
        window: Rolling window size.
    """
    col = pl.col(trades_col).cast(pl.Float64)
    mean = col.rolling_mean(window_size=window)
    std = col.rolling_std(window_size=window)
    return pl.when(std > 0).then((col - mean) / std).otherwise(pl.lit(0.0))


def unique_trader_growth(traders_col: str = "unique_traders", window: int = 5) -> pl.Expr:
    """Rate of change in unique trader count over a rolling window.

    Computed as percentage change of the rolling mean of unique traders.
    Positive values indicate growing trader participation.

    Args:
        traders_col: Name of the unique traders column.
        window: Rolling window size for smoothing.
    """
    smoothed = pl.col(traders_col).cast(pl.Float64).rolling_mean(window_size=window)
    return smoothed.pct_change()


# ---------------------------------------------------------------------------
# Liquidity indicators
# ---------------------------------------------------------------------------


def pool_depth(reserve_col: str = "pool_reserve_last") -> pl.Expr:
    """Raw pool reserve value as a liquidity proxy.

    Simply passes through the reserve column for clarity and composability.

    Args:
        reserve_col: Name of the pool reserve column.
    """
    return pl.col(reserve_col)


def price_impact_estimate(amount: float, reserve_col: str = "pool_reserve_last") -> pl.Expr:
    """Estimated slippage for a given trade size using constant-product AMM formula.

    For a constant-product AMM (x * y = k), the price impact of buying ``amount``
    of quote currency from a pool with reserve ``R`` is::

        impact = amount / (R + amount)

    Returns values in [0, 1). A value of 0.01 means ~1% slippage.

    Args:
        amount: Trade size in quote currency.
        reserve_col: Name of the pool reserve column.
    """
    reserve = pl.col(reserve_col)
    return pl.lit(amount) / (reserve + pl.lit(amount))


def liquidity_ratio(volume_col: str = "volume", reserve_col: str = "pool_reserve_last") -> pl.Expr:
    """Volume relative to pool depth.

    High values indicate the pool is being traded heavily relative to its size,
    suggesting higher slippage risk. Returns null when reserve is zero.

    Args:
        volume_col: Name of the volume column.
        reserve_col: Name of the pool reserve column.
    """
    reserve = pl.col(reserve_col)
    return pl.when(reserve > 0).then(pl.col(volume_col) / reserve).otherwise(None)


# ---------------------------------------------------------------------------
# Momentum indicators (memecoin-tuned)
# ---------------------------------------------------------------------------


def launch_velocity(price_col: str = "close", window: int = 5) -> pl.Expr:
    """Price change rate in the first N bars after a token appears.

    Computes percentage change over ``window`` bars. After the initial window,
    continues to return the rolling pct_change (which remains useful but is no
    longer specific to launch). Use with ``.over("symbol")``.

    Args:
        price_col: Name of the price column.
        window: Number of bars for the change calculation.
    """
    return pl.col(price_col).pct_change(n=window)


def pump_detector(
    price_col: str = "close",
    volume_col: str = "volume",
    price_std: float = 2.0,
    volume_std: float = 2.0,
    window: int = 20,
) -> pl.Expr:
    """Flag bars with simultaneous price and volume spikes.

    A bar is flagged when both the price return and volume exceed their
    respective rolling mean by the specified number of standard deviations.

    Args:
        price_col: Name of the price column.
        volume_col: Name of the volume column.
        price_std: Threshold in standard deviations for price spike.
        volume_std: Threshold in standard deviations for volume spike.
        window: Rolling window size for mean and std calculation.
    """
    ret = pl.col(price_col).pct_change()
    ret_mean = ret.rolling_mean(window_size=window)
    ret_std_val = ret.rolling_std(window_size=window)
    price_spike = (ret - ret_mean) > (ret_std_val * price_std)

    vol = pl.col(volume_col)
    vol_mean = vol.rolling_mean(window_size=window)
    vol_std_val = vol.rolling_std(window_size=window)
    vol_spike = (vol - vol_mean) > (vol_std_val * volume_std)

    return price_spike & vol_spike


def rug_pull_detector(
    price_col: str = "close",
    volume_col: str = "volume",
    sell_vol_col: str = "sell_volume",
    price_drop: float = -0.3,
    sell_ratio: float = 0.8,
    window: int = 1,
) -> pl.Expr:
    """Flag bars with sharp price drops accompanied by high sell volume.

    A bar is flagged when the price return over ``window`` bars is below
    ``price_drop`` AND sell volume exceeds ``sell_ratio`` of total volume.

    Args:
        price_col: Name of the price column.
        volume_col: Name of the volume column.
        sell_vol_col: Name of the sell volume column.
        price_drop: Price return threshold (e.g. -0.3 = -30%).
        sell_ratio: Minimum sell_volume / total_volume ratio (e.g. 0.8 = 80%).
        window: Number of bars for price change calculation.
    """
    ret = pl.col(price_col).pct_change(n=window)
    sell_frac = pl.when(pl.col(volume_col) > 0).then(pl.col(sell_vol_col) / pl.col(volume_col)).otherwise(pl.lit(0.0))
    return (ret < price_drop) & (sell_frac >= sell_ratio)
