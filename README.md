<p align="center">
  <img src="assets/cover.png" alt="PolarBtest" width="600">
</p>

# PolarBtest

A lightweight, high-performance backtesting library for trading strategy development and optimization. Built on [Polars](https://pola.rs/) for fast vectorized data processing with an event-driven execution loop for flexible strategy logic.

## Features

- **Hybrid architecture** — vectorized preprocessing (Polars) + event-driven execution loop
- **30+ built-in indicators** — SMA, EMA, RSI, MACD, Bollinger Bands, ATR, SuperTrend, ADX, and more
- **Complete order system** — market, limit, stop, stop-limit, bracket orders, day/GTC orders
- **Risk management** — stop-loss, take-profit, trailing stops, position size limits, drawdown stops
- **Short selling** — negative positions, borrow costs, position reversals
- **Margin & leverage** — configurable leverage, margin tracking, margin calls
- **Commission models** — percentage, fixed, maker/taker, volume-tiered, custom
- **Position sizing** — fixed, percent, fixed-risk, Kelly, volatility-based
- **Multi-asset** — pass a dict of DataFrames for portfolio strategies
- **Parallel optimization** — grid search, multi-objective Pareto, Bayesian optimization
- **Walk-forward analysis** — rolling and anchored train/test splits
- **Advanced analysis** — Monte Carlo simulation, look-ahead bias detection, permutation testing
- **Visualization** — interactive Plotly charts (price, equity, drawdown, trade markers, heatmaps)
- **Data utilities** — validation, cleaning, OHLCV resampling
- **Optional TA-Lib integration** — wrap any TA-Lib function into Polars expressions

## Installation

```bash
pip install polarbtest

# Optional extras
pip install polarbtest[plotting]   # Plotly charts
pip install polarbtest[talib]      # TA-Lib integration
```

## Quick Start

```python
import polars as pl
from polarbtest import Strategy, backtest
from polarbtest import indicators as ind
from polarbtest.core import BacktestContext


class SMACross(Strategy):
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            ind.sma("close", 10).alias("sma_fast"),
            ind.sma("close", 30).alias("sma_slow"),
            ind.crossover("sma_fast", "sma_slow").alias("buy"),
            ind.crossunder("sma_fast", "sma_slow").alias("sell"),
        ])

    def next(self, ctx: BacktestContext) -> None:
        if ctx.row.get("buy"):
            ctx.portfolio.order_target_percent("asset", 1.0)
        elif ctx.row.get("sell"):
            ctx.portfolio.close_position("asset")


results = backtest(
    SMACross,
    pl.DataFrame({"close": [100 + i * 0.5 for i in range(200)]}),
    params={},
    initial_cash=100_000,
)

print(f"Return: {results['total_return']:.2%}")
print(f"Sharpe: {results['sharpe_ratio']:.2f}")
```

## Examples

| Example | Description |
|---|---|
| [`example.py`](examples/example.py) | Basic SMA crossover |
| [`example_sma_crossover_stoploss.py`](examples/example_sma_crossover_stoploss.py) | SMA crossover with ATR stop-loss and trailing stop |
| [`example_rsi_bracket_orders.py`](examples/example_rsi_bracket_orders.py) | RSI mean reversion with bracket orders |
| [`example_momentum_rotation.py`](examples/example_momentum_rotation.py) | Multi-asset momentum rotation |
| [`example_ml_strategy.py`](examples/example_ml_strategy.py) | ML model integration |
| [`example_walk_forward.py`](examples/example_walk_forward.py) | Walk-forward analysis workflow |
| [`example_advanced_analysis.py`](examples/example_advanced_analysis.py) | Full workflow: optimization, heatmaps, Monte Carlo, permutation test |
| [`example_limit_orders.py`](examples/example_limit_orders.py) | Limit orders and stop-loss |
| [`example_trade_analysis.py`](examples/example_trade_analysis.py) | Trade-level analysis |
| [`example_plotting.py`](examples/example_plotting.py) | Interactive chart generation |
| [`example_commission.py`](examples/example_commission.py) | Commission model comparison |
| [`example_multi_asset.py`](examples/example_multi_asset.py) | Multi-asset dict input |

## Documentation

- [Getting Started Guide](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [Full Feature Documentation](DESCRIPTION.md)

## License

[MIT](LICENSE)
