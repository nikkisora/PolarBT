"""DeFi trade data pipeline: validate, aggregate, enrich, and backtest.

Demonstrates the full pipeline from raw DEX/AMM trade data to backtest results:
1. Generate synthetic trade data (mimics Pump.fun-style memecoin activity)
2. Validate and aggregate into time-based OHLCV bars
3. Apply DeFi indicators (buy/sell ratio, trade intensity, rug-pull detection)
4. Backtest with AMM-aware slippage and dynamic universe filtering
"""

import random
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from polarbt import Engine, Strategy
from polarbt import indicators_defi as defi
from polarbt.core import BacktestContext
from polarbt.data.trades import aggregate_trades, validate_trades
from polarbt.slippage import AMMSlippage
from polarbt.universe import AgeFilter, CompositeFilter, VolumeFilter

# ---------------------------------------------------------------------------
# 1. Generate synthetic trade data
# ---------------------------------------------------------------------------


def generate_synthetic_trades(
    n_tokens: int = 10,
    duration_hours: int = 24,
    base_trades_per_hour: int = 50,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate realistic-looking DEX trade data for multiple tokens.

    Each token gets a random price trajectory (geometric Brownian motion),
    varying trade sizes, and a mix of buy/sell activity. Some tokens include
    a "pump" phase with elevated buying, and one token simulates a rug-pull.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    rows: list[dict] = []
    start = datetime(2025, 3, 1, 0, 0, 0)

    for token_idx in range(n_tokens):
        symbol = f"TOKEN_{token_idx:03d}"
        # Stagger token launches: each token starts a few hours after the previous
        token_start = start + timedelta(hours=token_idx * 2)

        # Random initial price and volatility
        price = rng.uniform(0.00001, 0.001)
        volatility = rng.uniform(0.02, 0.08)
        pool_reserve = rng.uniform(5000.0, 50000.0)

        # Decide token behavior
        is_pump = token_idx in (2, 5, 7)
        is_rug = token_idx == 9
        pump_start_hour = rng.integers(8, 16)

        n_hours = duration_hours - token_idx * 2
        if n_hours <= 0:
            continue

        for hour in range(n_hours):
            ts_base = token_start + timedelta(hours=hour)

            # Adjust activity and buy bias based on token phase
            if is_pump and pump_start_hour <= hour < pump_start_hour + 3:
                trades_this_hour = base_trades_per_hour * 3
                buy_prob = 0.8
                volatility_mult = 2.0
            elif is_rug and hour == n_hours - 1:
                trades_this_hour = base_trades_per_hour * 5
                buy_prob = 0.1
                volatility_mult = 3.0
            else:
                trades_this_hour = max(5, base_trades_per_hour + rng.integers(-20, 20))
                buy_prob = 0.5 + rng.uniform(-0.1, 0.1)
                volatility_mult = 1.0

            for t in range(trades_this_hour):
                # Price follows GBM
                price *= np.exp(rng.normal(0, volatility * volatility_mult / np.sqrt(trades_this_hour)))
                price = max(price, 1e-10)

                # Trade properties
                side = "buy" if rng.random() < buy_prob else "sell"
                amount = float(rng.exponential(0.5))
                pool_reserve += amount if side == "buy" else -min(amount, pool_reserve * 0.01)
                pool_reserve = max(pool_reserve, 100.0)

                rows.append(
                    {
                        "timestamp": ts_base + timedelta(seconds=int(3600 * t / trades_this_hour)),
                        "symbol": symbol,
                        "price": float(price),
                        "amount": float(amount),
                        "side": side,
                        "trader": f"wallet_{rng.integers(0, 200):04d}",
                        "pool_reserve": float(pool_reserve),
                    }
                )

    return pl.DataFrame(
        rows,
        schema={
            "timestamp": pl.Datetime("us"),
            "symbol": pl.String,
            "price": pl.Float64,
            "amount": pl.Float64,
            "side": pl.String,
            "trader": pl.String,
            "pool_reserve": pl.Float64,
        },
    )


print("Generating synthetic trade data...")
trades = generate_synthetic_trades(n_tokens=10, duration_hours=24, base_trades_per_hour=60)
print(f"  {trades.shape[0]:,} trades across {trades['symbol'].n_unique()} tokens")

# Validate
result = validate_trades(trades)
print(f"  Validation: {'PASS' if result.valid else 'FAIL'}")
if result.errors:
    for e in result.errors[:3]:
        print(f"    Error: {e}")

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
