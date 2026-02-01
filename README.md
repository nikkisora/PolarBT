# PolarBtest

A lightweight, high-performance backtesting library designed for evolutionary strategy search and ML-integrated trading systems.

## Features

- **Polars-Based**: Leverages Polars for blazing-fast data operations
- **Hybrid Architecture**: Vectorized preprocessing + event-driven execution
- **Multi-Asset Support**: Handle multiple assets with fractional positions
- **ML-Ready**: Seamlessly integrate machine learning models into strategies
- **Parallel Execution**: Multi-core backtesting for rapid parameter optimization
- **Evolutionary Search Friendly**: Clean API designed for LLM-driven strategy discovery

## Installation

```bash
pip install polars numpy
```

Then add the `polarbtest` directory to your Python path, or install in development mode:

```bash
pip install -e .
```

## Quick Start

### Basic Strategy

```python
import polars as pl
from polarbtest import Strategy, backtest, indicators as ind

# Define your strategy
class SMACrossStrategy(Strategy):
    def preprocess(self, df):
        """Calculate indicators using vectorized Polars operations"""
        fast = self.params.get("fast_period", 10)
        slow = self.params.get("slow_period", 20)
        
        return df.with_columns([
            ind.sma("close", fast).alias("sma_fast"),
            ind.sma("close", slow).alias("sma_slow"),
        ])
    
    def next(self, ctx):
        """Execute strategy logic on each bar"""
        # Golden cross: go long
        if ctx.row["sma_fast"] > ctx.row["sma_slow"]:
            ctx.portfolio.order_target_percent("asset", 1.0)
        # Death cross: close position
        else:
            ctx.portfolio.close_position("asset")

# Load your data
data = pl.read_csv("price_data.csv")

# Run backtest
results = backtest(
    SMACrossStrategy,
    data,
    params={"fast_period": 10, "slow_period": 20},
    initial_cash=100_000
)

print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

### Multi-Asset Strategy

```python
class MultiAssetMomentumStrategy(Strategy):
    def preprocess(self, df):
        lookback = self.params.get("lookback", 20)
        
        # Calculate momentum for each asset
        return df.with_columns([
            ind.returns("btc_close", lookback).alias("btc_momentum"),
            ind.returns("eth_close", lookback).alias("eth_momentum"),
            ind.returns("sol_close", lookback).alias("sol_momentum"),
        ])
    
    def next(self, ctx):
        # Allocate to top 2 assets by momentum
        assets = [
            ("BTC", ctx.row.get("btc_momentum", 0)),
            ("ETH", ctx.row.get("eth_momentum", 0)),
            ("SOL", ctx.row.get("sol_momentum", 0)),
        ]
        
        # Sort by momentum
        assets.sort(key=lambda x: x[1], reverse=True)
        
        # Equal weight to top 2
        for asset, _ in assets[:2]:
            ctx.portfolio.order_target_percent(asset, 0.5)
        
        # Close bottom 1
        ctx.portfolio.close_position(assets[2][0])

# Run with multiple price columns
results = backtest(
    MultiAssetMomentumStrategy,
    data,
    params={"lookback": 30},
    price_columns={
        "BTC": "btc_close",
        "ETH": "eth_close",
        "SOL": "sol_close",
    }
)
```

### ML-Integrated Strategy

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class MLStrategy(Strategy):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = None
    
    def preprocess(self, df):
        """Generate features and train model"""
        # Create features
        df = df.with_columns([
            ind.returns("close", 1).alias("returns_1d"),
            ind.returns("close", 5).alias("returns_5d"),
            ind.rsi("close", 14).alias("rsi"),
            ind.sma("close", 20).alias("sma_20"),
        ])
        
        # Prepare training data (first 70% of data)
        train_size = int(len(df) * 0.7)
        train_df = df[:train_size].drop_nulls()
        
        # Train model
        X = train_df.select(["returns_1d", "returns_5d", "rsi"]).to_numpy()
        y = (train_df["close"].shift(-1) > train_df["close"]).to_numpy()[:-1]
        X = X[:-1]
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Generate predictions for full dataset
        X_full = df.select(["returns_1d", "returns_5d", "rsi"]).to_numpy()
        predictions = np.zeros(len(df))
        
        # Only predict where we have features
        valid_idx = ~np.isnan(X_full).any(axis=1)
        predictions[valid_idx] = self.model.predict_proba(X_full[valid_idx])[:, 1]
        
        return df.with_columns([
            pl.Series("ml_signal", predictions)
        ])
    
    def next(self, ctx):
        signal = ctx.row.get("ml_signal", 0)
        
        # Trade based on ML prediction
        if signal > 0.6:
            ctx.portfolio.order_target_percent("asset", 1.0)
        elif signal < 0.4:
            ctx.portfolio.close_position("asset")
```

## Parallel Optimization

```python
from polarbtest import backtest_batch, optimize

# Test multiple parameter combinations
param_sets = [
    {"fast_period": 5, "slow_period": 20},
    {"fast_period": 10, "slow_period": 30},
    {"fast_period": 20, "slow_period": 50},
]

results_df = backtest_batch(
    SMACrossStrategy,
    data,
    param_sets,
    n_jobs=4  # Use 4 CPU cores
)

# Sort by Sharpe ratio
best_results = results_df.sort("sharpe_ratio", descending=True)
print(best_results.head())

# Or use grid search
param_grid = {
    "fast_period": [5, 10, 15, 20],
    "slow_period": [20, 30, 50, 100],
}

best = optimize(
    SMACrossStrategy,
    data,
    param_grid,
    objective="sharpe_ratio"
)

print(f"Best params: {best['params']}")
print(f"Best Sharpe: {best['sharpe_ratio']:.2f}")
```

## Available Indicators

All indicators return Polars expressions for use in `preprocess()`:

```python
from polarbtest import indicators as ind

# Trend indicators
ind.sma(column, period)              # Simple Moving Average
ind.ema(column, period)              # Exponential Moving Average
ind.macd(column, fast, slow, signal) # MACD

# Momentum indicators
ind.rsi(column, period)              # Relative Strength Index
ind.returns(column, periods)         # Percentage returns
ind.log_returns(column, periods)     # Log returns

# Volatility indicators
ind.bollinger_bands(column, period, std_dev)  # Bollinger Bands
ind.atr(high, low, close, period)            # Average True Range

# Signal detection
ind.crossover(fast, slow)   # Detect bullish crossover
ind.crossunder(fast, slow)  # Detect bearish crossover
```

## Portfolio Management

```python
# In your strategy's next() method:

# Market orders
ctx.portfolio.order("BTC", 0.5)           # Buy 0.5 BTC
ctx.portfolio.order("BTC", -0.25)         # Sell 0.25 BTC

# Target positions
ctx.portfolio.order_target("BTC", 1.0)    # Set position to 1.0 BTC
ctx.portfolio.order_target_percent("BTC", 0.5)  # Allocate 50% to BTC
ctx.portfolio.order_target_value("BTC", 50000)  # Set BTC value to $50k

# Close positions
ctx.portfolio.close_position("BTC")       # Close BTC position
ctx.portfolio.close_all_positions()       # Close all positions

# Check positions
position = ctx.portfolio.get_position("BTC")
total_value = ctx.portfolio.get_value()
```

## Automatic Warmup

By default, PolarBtest uses **automatic warmup** to handle indicator initialization:

```python
# Default behavior - auto warmup (recommended)
results = backtest(MyStrategy, data, params={...})

# Explicitly set auto warmup
results = backtest(MyStrategy, data, params={...}, warmup="auto")

# Manual warmup - skip first 20 bars
results = backtest(MyStrategy, data, params={...}, warmup=20)

# No warmup - start trading immediately
results = backtest(MyStrategy, data, params={...}, warmup=0)
```

### How Auto Warmup Works

When `warmup="auto"` (the default):
1. The engine preprocesses your data to calculate all indicators
2. It finds the first bar where **all columns** have non-null values
3. Strategy execution begins at that bar automatically


### When to Use Manual Warmup

Use an explicit integer for `warmup` when:
- You want to skip additional bars beyond indicator warmup
- You're implementing a custom warmup strategy
- You need consistent warmup across different parameter sets

```python
# Force 50 bars of warmup regardless of indicators
results = backtest(MyStrategy, data, warmup=50)
```

## Performance Metrics

Results include comprehensive performance metrics:

```python
results = backtest(MyStrategy, data, params={...})

# Returns
results["total_return"]     # Total return
results["cagr"]            # Compound Annual Growth Rate

# Risk-adjusted returns
results["sharpe_ratio"]    # Sharpe ratio (annualized)
results["sortino_ratio"]   # Sortino ratio (annualized)
results["calmar_ratio"]    # Calmar ratio

# Risk metrics
results["max_drawdown"]    # Maximum drawdown
results["volatility"]      # Volatility (annualized)

# Trade statistics
results["win_rate"]        # Percentage of winning periods
results["profit_factor"]   # Gross profit / gross loss
```

## Walk-Forward Analysis

```python
from polarbtest.runner import walk_forward_analysis

# Perform walk-forward optimization
wf_results = walk_forward_analysis(
    SMACrossStrategy,
    data,
    param_grid={"fast_period": [5, 10, 20], "slow_period": [20, 50, 100]},
    train_periods=252,  # 1 year training
    test_periods=63,    # 1 quarter testing
    objective="sharpe_ratio"
)

# Analyze out-of-sample performance
print(wf_results.select(["fold", "train_objective", "test_objective"]))
```

## For LLM-Driven Optimization

The library is designed for easy integration with LLM-based strategy search:

```python
def evaluate_strategy_code(strategy_code: str, data: pl.DataFrame) -> dict:
    """
    Evaluate a strategy from LLM-generated code.
    
    Args:
        strategy_code: Python code defining a Strategy class
        data: Price data
        
    Returns:
        Performance metrics
    """
    # Execute the strategy code
    namespace = {}
    exec(strategy_code, namespace)
    
    # Find the Strategy class
    strategy_class = None
    for name, obj in namespace.items():
        if isinstance(obj, type) and issubclass(obj, Strategy) and obj != Strategy:
            strategy_class = obj
            break
    
    if strategy_class is None:
        return {"error": "No Strategy class found"}
    
    # Run backtest
    return backtest(strategy_class, data, initial_cash=100_000)

# Example usage with LLM
llm_generated_code = """
class MyStrategy(Strategy):
    def preprocess(self, df):
        return df.with_columns([ind.rsi("close", 14).alias("rsi")])
    
    def next(self, ctx):
        if ctx.row.get("rsi"):
            if ctx.row["rsi"] < 30:
                ctx.portfolio.order_target_percent("asset", 1.0)
            elif ctx.row["rsi"] > 70:
                ctx.portfolio.close_position("asset")
"""

results = evaluate_strategy_code(llm_generated_code, data)
print(f"Fitness: {results['sharpe_ratio']}")
```

## Architecture

```
┌─────────────────────────────────────────┐
│           Strategy Definition           │
│  • preprocess(): Vectorized features    │
│  • next(): Event-driven logic           │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         Polars Data Pipeline            │
│  • Calculate all indicators at once     │
│  • ML model inference (batch)           │
│  • Zero-copy operations where possible  │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│           Backtesting Engine            │
│  • Iterate through bars                 │
│  • Update portfolio state               │
│  • Track equity curve                   │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│        Performance Metrics              │
│  • Sharpe, Sortino, Calmar             │
│  • Drawdown analysis                    │
│  • Trade statistics                     │
└─────────────────────────────────────────┘
```

## Requirements

- Python 3.8+
- polars
- numpy (for metrics calculation)

## License

MIT

## Contributing

Contributions welcome! This library is designed to be lightweight and extensible.

Key areas for contribution:
- Additional technical indicators
- Advanced order types (stop-loss, take-profit)
- More sophisticated portfolio management
- Additional performance metrics
- Visualization utilities
