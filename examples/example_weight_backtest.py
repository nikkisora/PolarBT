"""
Example: Weight-Based Portfolio Backtest

Demonstrates:
- Declarative weight allocation via backtest_weights()
- Monthly rebalancing with offset
- Stop-loss and take-profit per position
- Next-actions output for forward-looking operations
"""

from datetime import date, timedelta

import numpy as np
import polars as pl

from polarbt import backtest_weights

# Generate synthetic multi-asset data in long format
rng = np.random.default_rng(42)
N_DAYS = 252
START = date(2024, 1, 2)

symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
rows: list[dict] = []

prices = dict.fromkeys(symbols, 100.0)
drifts = {"AAPL": 0.0004, "GOOGL": 0.0002, "MSFT": 0.0003, "AMZN": -0.0001}

for i in range(N_DAYS):
    d = START + timedelta(days=i)
    # Simple momentum signal: top-2 by trailing return get equal weight
    for sym in symbols:
        ret = rng.normal(drifts[sym], 0.015)
        prices[sym] *= np.exp(ret)
        rows.append(
            {
                "date": d,
                "symbol": sym,
                "open": round(prices[sym] * (1 + rng.normal(0, 0.002)), 4),
                "high": round(prices[sym] * (1 + abs(rng.normal(0, 0.008))), 4),
                "low": round(prices[sym] * (1 - abs(rng.normal(0, 0.008))), 4),
                "close": round(prices[sym], 4),
                "volume": int(rng.uniform(100_000, 500_000)),
                "weight": 0.0,  # filled below
            }
        )

data = pl.DataFrame(rows)

# Assign equal-weight to top-2 momentum symbols each month
data = data.with_columns(pl.col("close").pct_change().rolling_sum(window_size=20).over("symbol").alias("momentum"))
data = data.with_columns(pl.col("momentum").rank(descending=True).over("date").alias("rank"))
data = data.with_columns(pl.when(pl.col("rank") <= 2).then(0.5).otherwise(0.0).alias("weight"))

if __name__ == "__main__":
    print("Weight-Based Portfolio Backtest")
    print("=" * 60)

    result = backtest_weights(
        data,
        resample="M",
        resample_offset="2d",
        fee_ratio=0.001,
        stop_loss=0.10,
        take_profit=0.25,
        position_limit=0.5,
        initial_capital=100_000,
    )

    print(result.metrics)

    print("\nEquity curve (first 5 rows):")
    print(result.equity.head(5))

    print(f"\nTotal trades: {len(result.trades)}")
    if len(result.trades) > 0:
        print(result.trades.head(5))

    if result.next_actions is not None:
        print("\nNext actions:")
        print(result.next_actions)
