"""DeFi trade data pipeline: validate, aggregate, enrich, and backtest.

Demonstrates the full pipeline from raw DEX/AMM trade data to backtest results:
1. Load and validate raw trades (Pump.fun format)
2. Aggregate into time-based OHLCV bars
3. Apply DeFi indicators (buy/sell ratio, trade intensity, rug-pull detection)
4. Backtest with AMM-aware slippage and dynamic universe filtering
"""

import polars as pl

from polarbt import Engine, Strategy
from polarbt import indicators_defi as defi
from polarbt.core import BacktestContext
from polarbt.data.trades import aggregate_trades, validate_trades
from polarbt.slippage import AMMSlippage
from polarbt.universe import AgeFilter, CompositeFilter, VolumeFilter

# ---------------------------------------------------------------------------
# 1. Load and validate raw trade data
# ---------------------------------------------------------------------------

DATA_PATH = "/mnt/d/downloads/02_datasets/pumpfun_standard.parquet"

print("Loading raw trades...")
raw = pl.read_parquet(DATA_PATH)
print(f"  {raw.shape[0]:,} trades across {raw['symbol'].n_unique():,} tokens")

# Take a manageable subset: top 50 tokens by trade count
top_symbols = raw.group_by("symbol").len().sort("len", descending=True).head(50)["symbol"].to_list()
trades = raw.filter(pl.col("symbol").is_in(top_symbols)).sort("symbol", "timestamp")
print(f"  Subset: {trades.shape[0]:,} trades across {trades['symbol'].n_unique()} tokens")

# Clean: drop rows with non-positive price/amount (common in raw DEX data)
trades = trades.filter((pl.col("price") > 0) & (pl.col("amount") > 0))

# Validate
result = validate_trades(trades)
print(f"  Validation: {'PASS' if result.valid else 'FAIL'}")
if result.errors:
    for e in result.errors[:3]:
        print(f"    Error: {e}")
if result.warnings:
    for w in result.warnings[:3]:
        print(f"    Warning: {w}")

# ---------------------------------------------------------------------------
# 2. Aggregate into 5-minute bars
# ---------------------------------------------------------------------------

print("\nAggregating to 5-minute bars...")
bars = aggregate_trades(trades, "5m", min_trades=3)
print(f"  {bars.height:,} bars across {bars['symbol'].n_unique()} tokens")
print(f"  Columns: {bars.columns}")


# ---------------------------------------------------------------------------
# 3. Define a strategy using DeFi indicators
# ---------------------------------------------------------------------------


class BuySellMomentum(Strategy):
    """Buy tokens with strong buying pressure, exit on selling pressure or rug signals."""

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            defi.buy_sell_ratio().over("symbol").alias("bs_ratio"),
            defi.trade_intensity(window=10).over("symbol").alias("intensity"),
            defi.net_flow().alias("net_flow"),
            defi.rug_pull_detector(price_drop=-0.3, sell_ratio=0.8, window=1).over("symbol").alias("is_rug"),
        )

    def next(self, ctx: BacktestContext) -> None:
        for sym in ctx.symbols:
            row = ctx.row(sym)
            position = ctx.portfolio.get_position(sym)

            # Exit on rug-pull signal
            if row.get("is_rug") and position > 0:
                ctx.portfolio.close_position(sym)
                continue

            bs = row.get("bs_ratio")
            intensity = row.get("intensity")

            if bs is None or intensity is None:
                continue

            # Enter: strong buying pressure + elevated trade activity
            if bs > 0.65 and intensity > 1.5 and position == 0:
                ctx.portfolio.order_target_percent(sym, 0.05)

            # Exit: selling pressure dominates
            elif bs < 0.35 and position > 0:
                ctx.portfolio.close_position(sym)


# ---------------------------------------------------------------------------
# 4. Run backtest with AMM slippage and universe filtering
# ---------------------------------------------------------------------------

print("\nRunning backtest...")
engine = Engine(
    BuySellMomentum(),
    bars,
    initial_cash=100.0,
    commission=0.01,
    slippage=AMMSlippage(),
    universe_provider=CompositeFilter(
        AgeFilter(min_bars=5),
        VolumeFilter(min_volume=0.5),
    ),
    warmup=15,
)

results = engine.run()
print(results)

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

print(f"\nTrade count: {results.trade_stats.total_trades}")
print(f"Win rate: {results.trade_stats.win_rate:.1%}")
print(f"Final equity: {results.final_equity:.4f} SOL")
print(f"Total return: {results.total_return:.2%}")
