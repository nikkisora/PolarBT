# Getting Started with PolarBT

This guide walks you through installing PolarBT, writing your first strategy, running a backtest, and interpreting the results.

## Installation

```bash
pip install polarbt

# Optional: plotting support
pip install polarbt[plotting]

# Optional: TA-Lib integration
pip install polarbt[talib]
```

## Your First Strategy

Every PolarBT strategy has two methods:

1. **`preprocess(df)`** — called once before the backtest. Add indicator columns using vectorized Polars operations.
2. **`next(ctx)`** — called on every bar after warmup. Place orders through `ctx.portfolio`.

```python
import polars as pl

from polarbt import Strategy, backtest, param
from polarbt import indicators as ind
from polarbt.core import BacktestContext


class SMACross(Strategy):
    fast_period = param(10)
    slow_period = param(30)

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            ind.sma("close", self.fast_period).alias("sma_fast"),
            ind.sma("close", self.slow_period).alias("sma_slow"),
            ind.crossover("sma_fast", "sma_slow").alias("buy_signal"),
            ind.crossunder("sma_fast", "sma_slow").alias("sell_signal"),
        ])

    def next(self, ctx: BacktestContext) -> None:
        if ctx.row.get("buy_signal"):
            ctx.portfolio.order_target_percent("asset", 1.0)
        elif ctx.row.get("sell_signal"):
            ctx.portfolio.close_position("asset")
```

## Running a Backtest

### Prepare Data

PolarBT accepts Polars DataFrames. The minimum required column is `close`. For full OHLCV support, include `open`, `high`, `low`, `close`, and `volume`.

```python
import numpy as np

np.random.seed(42)
n = 500
prices = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))

df = pl.DataFrame({
    "close": prices.tolist(),
})
```

### Run

```python
results = backtest(
    SMACross,
    df,
    params={"fast_period": 10, "slow_period": 30},
    initial_cash=100_000,
    commission=0.001,
    slippage=0.0005,
)
```

### Interpret Results

```python
print(f"Total Return:  {results.total_return:.2%}")
print(f"Sharpe Ratio:  {results.sharpe_ratio:.2f}")
print(f"Max Drawdown:  {results.max_drawdown:.2%}")
print(f"Win Rate:      {results.win_rate:.1f}%")
print(f"Final Equity:  ${results.final_equity:,.2f}")
```

The `results` object is a `BacktestMetrics` dataclass with all metrics from `calculate_metrics()` plus trade-level stats.

## Adding Risk Management

```python
class SMACrossWithRisk(Strategy):
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            ind.sma("close", 10).alias("sma_fast"),
            ind.sma("close", 30).alias("sma_slow"),
            ind.crossover("sma_fast", "sma_slow").alias("buy_signal"),
            ind.crossunder("sma_fast", "sma_slow").alias("sell_signal"),
            ind.atr("high", "low", "close", 14).alias("atr"),
        ])

    def next(self, ctx: BacktestContext) -> None:
        if ctx.row.get("buy_signal"):
            ctx.portfolio.order_target_percent("asset", 0.95)
            # ATR-based stop-loss
            atr = ctx.row.get("atr")
            if atr:
                ctx.portfolio.set_stop_loss("asset", stop_price=ctx.row["close"] - 2.0 * atr)
                ctx.portfolio.set_take_profit("asset", target_price=ctx.row["close"] + 3.0 * atr)
        elif ctx.row.get("sell_signal"):
            ctx.portfolio.close_position("asset")
```

## Using the Engine Directly

For more control (e.g., accessing the engine for plotting), use `Engine` directly:

```python
from polarbt import Engine

engine = Engine(
    strategy=SMACross(fast_period=10, slow_period=30),
    data=df,
    initial_cash=100_000,
    commission=0.001,
    slippage=0.0005,
)
results = engine.run()

# Access trades
trades_df = results.trades
print(trades_df)
```

## Visualization

```python
from polarbt.plotting import plot_backtest, plot_returns_distribution

fig = plot_backtest(
    engine,
    indicators=["sma_fast", "sma_slow"],
    title="SMA Crossover Strategy",
    save_html="backtest.html",
)

fig = plot_returns_distribution(engine, save_html="returns.html")
```

## Parameter Optimization

```python
from polarbt import optimize

best = optimize(
    SMACross,
    df,
    param_grid={
        "fast_period": [5, 10, 15, 20],
        "slow_period": [20, 30, 40, 50],
    },
    objective="sharpe_ratio",
    constraint=lambda p: p["fast_period"] < p["slow_period"],
    n_jobs=4,
)

# OptimizeResult separates params from metrics
print(f"Best params: {best.params}")
print(f"Best Sharpe: {best.metrics.sharpe_ratio:.3f}")

# Dict-style access still works
print(f"Best fast_period: {best['fast_period']}")
print(f"Best slow_period: {best['slow_period']}")
```

## Multi-Asset Strategies

Pass a dict of DataFrames for multi-asset backtesting:

```python
data = {
    "BTC": pl.DataFrame({"close": btc_prices}),
    "ETH": pl.DataFrame({"close": eth_prices}),
}

results = backtest(MyMultiAssetStrategy, data, params={...})
```

Columns are automatically prefixed (e.g., `BTC_close`, `ETH_close`).

## Weight-Based Backtesting

For portfolio allocation strategies (momentum rotation, factor models, equal-weight baskets), PolarBT provides a declarative `backtest_weights()` function. Instead of writing `preprocess()` + `next()`, you supply a long-format DataFrame with target weights per (date, symbol):

```python
import polars as pl
from polarbt import backtest_weights

# data: long-format DataFrame with columns date, symbol, close, weight
result = backtest_weights(
    data,
    resample="M",           # rebalance monthly
    resample_offset="2d",   # delay rebalance by 2 trading days
    fee_ratio=0.001,
    stop_loss=0.10,          # 10% per-position stop-loss
    take_profit=0.25,        # 25% per-position take-profit
    position_limit=0.5,      # max 50% in any single position
    initial_capital=100_000,
)

print(result.metrics)
print(result.equity.head())
print(result.trades.head())

# Forward-looking rebalance actions
if result.next_actions is not None:
    print(result.next_actions)
```

The result is a `WeightBacktestResult` with:
- `equity` — equity curve DataFrame
- `trades` — per-trade details
- `metrics` — standard `BacktestMetrics`
- `next_actions` — DataFrame of enter/exit/hold actions for the next rebalance

See `examples/example_weight_backtest.py` for a complete runnable example.

## Next Steps

- See `examples/` for complete runnable strategies
- See `docs/api-reference.md` for the full API reference
- See `docs/complete-reference.md` for detailed feature documentation
