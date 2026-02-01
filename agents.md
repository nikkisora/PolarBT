# PolarBtest - Agent Documentation

This document provides detailed information about the PolarBtest library architecture, components, and best practices for AI agents working with this codebase.

## Architecture Overview

PolarBtest is a hybrid backtesting engine that combines:
1. **Vectorized preprocessing** using Polars for performance
2. **Event-driven execution** for strategy flexibility
3. **Parallel execution** for evolutionary optimization

### Core Philosophy

- **Performance**: All heavy computation is vectorized using Polars
- **Flexibility**: Event-driven loop allows complex conditional logic and ML models
- **Scalability**: Multi-core execution for testing thousands of strategies
- **Simplicity**: Clean API designed for LLM-generated code

## Module Structure

```
polarbtest/
├── __init__.py       # Package exports
├── core.py           # Portfolio, Strategy, Engine, BacktestContext
├── indicators.py     # Technical indicators as Polars expressions
├── metrics.py        # Performance metrics calculation
└── runner.py         # Parallel execution and optimization
```

## Core Components

### 1. Portfolio (`core.py`)

Manages cash and positions with support for:
- Fractional shares (e.g., 0.001 BTC)
- Multiple assets simultaneously
- Transaction costs and slippage
- Various order types

**Key Methods:**
```python
portfolio.order(asset, quantity)                    # Market order
portfolio.order_target(asset, target_quantity)      # Target position
portfolio.order_target_percent(asset, percent)      # Target % of portfolio
portfolio.order_target_value(asset, value)          # Target $ value
portfolio.close_position(asset)                     # Close position
portfolio.get_position(asset)                       # Get current position
portfolio.get_value()                               # Total portfolio value
```

**Internal State:**
- `cash`: Current cash balance
- `positions`: Dict mapping asset names to quantities
- `equity_curve`: Historical portfolio values
- `_current_prices`: Current market prices for valuation

### 2. Strategy (`core.py`)

Base class for all trading strategies. Subclasses must implement:

**Required Methods:**
```python
def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
    """
    Vectorized feature engineering.
    Called ONCE before backtest starts.
    Use Polars expressions to calculate indicators, ML features, etc.
    """
    pass

def next(self, ctx: BacktestContext) -> None:
    """
    Event-driven logic.
    Called on EVERY bar.
    Use ctx.portfolio to place orders.
    """
    pass
```

**Optional Methods:**
```python
def on_start(self, portfolio: Portfolio):
    """Called once before backtest starts"""
    pass

def on_finish(self, portfolio: Portfolio):
    """Called once after backtest completes"""
    pass
```

**Parameters:**
Strategies accept parameters via `__init__(**params)`:
```python
class MyStrategy(Strategy):
    def __init__(self, **params):
        super().__init__(**params)
        self.custom_param = params.get("custom_param", default_value)
```

### 3. BacktestContext (`core.py`)

Data container passed to `strategy.next()`:

```python
@dataclass
class BacktestContext:
    timestamp: Any              # Current timestamp
    row: Dict[str, Any]        # Current bar data (prices, indicators)
    portfolio: Portfolio       # Reference to portfolio
    bar_index: int            # Current bar index
```

Access current data:
```python
def next(self, ctx):
    price = ctx.row["close"]
    sma = ctx.row["sma_20"]
    
    if ctx.bar_index < 20:
        return  # Skip first 20 bars
    
    # Place orders
    ctx.portfolio.order("BTC", 0.5)
```

### 4. Engine (`core.py`)

Executes the backtest simulation:

```python
engine = Engine(
    strategy=strategy_instance,
    data=polars_dataframe,
    initial_cash=100_000,
    commission=0.001,          # 0.1%
    slippage=0.0005,          # 0.05%
    price_columns={"BTC": "btc_close", "ETH": "eth_close"},
    warmup="auto",            # Default: auto-detect warmup period
    order_delay=0,            # Default: execute immediately
)

results = engine.run()
```

**Warmup Parameter:**
- `warmup="auto"` (default): Automatically skip bars until all indicators are ready
- `warmup=N` (integer): Manually skip first N bars
- `warmup=0`: No warmup, start immediately

The auto warmup feature finds the first bar where all columns (except timestamp) are non-null,
ensuring your strategy only executes when all indicators have valid values.

**Execution Flow:**
1. Call `strategy.preprocess(data)` to generate features
2. Calculate warmup period (if warmup="auto")
3. Call `strategy.on_start(portfolio)`
4. For each bar:
   - Update portfolio prices
   - Create BacktestContext
   - Call `strategy.next(ctx)` (only after warmup period)
   - Record equity
5. Call `strategy.on_finish(portfolio)`
6. Calculate and return metrics

### 5. Indicators (`indicators.py`)

All indicators return Polars expressions for use in `preprocess()`:

```python
# Trend
sma(column, period) -> pl.Expr
ema(column, period) -> pl.Expr
macd(column, fast, slow, signal) -> tuple[pl.Expr, pl.Expr, pl.Expr]

# Momentum
rsi(column, period) -> pl.Expr
returns(column, periods) -> pl.Expr
log_returns(column, periods) -> pl.Expr

# Volatility
bollinger_bands(column, period, std_dev) -> tuple[pl.Expr, pl.Expr, pl.Expr]
atr(high, low, close, period) -> pl.Expr

# Signals
crossover(fast_column, slow_column) -> pl.Expr
crossunder(fast_column, slow_column) -> pl.Expr
```

### 6. Runner (`runner.py`)

High-level functions for backtesting and optimization:

```python
# Single backtest
backtest(strategy_class, data, params, initial_cash, commission, slippage)

# Parallel batch
backtest_batch(strategy_class, data, param_sets, n_jobs)

# Grid search
optimize(strategy_class, data, param_grid, objective, maximize)

# Walk-forward
walk_forward_analysis(strategy_class, data, param_grid, train_periods, test_periods)
```

## Data Format

### Input Data

Expected Polars DataFrame format:

```python
pl.DataFrame({
    "timestamp": [...],  # Optional, but recommended
    "close": [...],      # Required for single asset
    "high": [...],       # Optional
    "low": [...],        # Optional
    "volume": [...],     # Optional
})
```

For multiple assets:
```python
pl.DataFrame({
    "timestamp": [...],
    "btc_close": [...],
    "eth_close": [...],
    "sol_close": [...],
})
```

### Output Metrics

Results dictionary contains:

```python
{
    # Returns
    "total_return": float,
    "cagr": float,
    
    # Risk-adjusted
    "sharpe_ratio": float,
    "sortino_ratio": float,
    "calmar_ratio": float,
    
    # Risk
    "max_drawdown": float,
    "volatility": float,
    "volatility_annualized": float,
    
    # Trade stats
    "win_rate": float,
    "profit_factor": float,
    
    # Portfolio
    "final_equity": float,
    "final_cash": float,
    "final_positions": dict,
    
    # Meta
    "params": dict,
    "success": bool,
}
```

## Best Practices for Strategy Development

### 1. Vectorize Heavy Computation

Do this in `preprocess()`:
```python
def preprocess(self, df):
    # Good: Vectorized
    return df.with_columns([
        ind.sma("close", 20).alias("sma_20"),
        ind.rsi("close", 14).alias("rsi"),
    ])
```

Not this in `next()`:
```python
def next(self, ctx):
    # Bad: Calculating on every bar
    last_20_prices = self.get_last_n_prices(20)
    sma = sum(last_20_prices) / len(last_20_prices)
```

### 2. Automatic Warmup (Default)

The engine automatically skips bars until all indicators are ready:
```python
# Use default auto warmup - no manual checks needed
results = backtest(MyStrategy, data, params={...})

# Or explicitly set it
results = backtest(MyStrategy, data, warmup="auto")

# Manual warmup if needed
results = backtest(MyStrategy, data, warmup=20)
```

With auto warmup enabled (default), you don't need to check for None values in `next()`:
```python
def next(self, ctx):
    # Auto warmup ensures indicators are ready
    if ctx.row["close"] > ctx.row["sma"]:
        ctx.portfolio.order_target_percent("asset", 1.0)
```

If you disable auto warmup (`warmup=0`), you should add defensive checks:
```python
def next(self, ctx):
    if ctx.row.get("sma") is None:
        return  # Skip this bar
    
    # Safe to use sma now
    if ctx.row["close"] > ctx.row["sma"]:
        ...
```

### 3. ML Model Integration

Train models in `preprocess()`, not `next()`:
```python
def preprocess(self, df):
    # Calculate features
    df = df.with_columns([
        ind.returns("close", 5).alias("ret_5d"),
        ind.rsi("close", 14).alias("rsi"),
    ])
    
    # Train model on historical data
    train_df = df[:int(len(df) * 0.7)].drop_nulls()
    X_train = train_df.select(["ret_5d", "rsi"]).to_numpy()
    y_train = (train_df["close"].shift(-1) > train_df["close"]).to_numpy()[:-1]
    
    self.model = RandomForestClassifier()
    self.model.fit(X_train[:-1], y_train)
    
    # Generate predictions for entire dataset
    X_full = df.select(["ret_5d", "rsi"]).to_numpy()
    predictions = self.model.predict_proba(X_full)[:, 1]
    
    return df.with_columns([
        pl.Series("prediction", predictions)
    ])

def next(self, ctx):
    # Just use the pre-computed prediction
    if ctx.row["prediction"] > 0.6:
        ctx.portfolio.order_target_percent("asset", 1.0)
```

### 4. Position Sizing

Use percentage-based sizing for robustness:
```python
# Good: Scales with portfolio
ctx.portfolio.order_target_percent("BTC", 0.5)  # 50% allocation

# Bad: Fixed quantity
ctx.portfolio.order("BTC", 1.0)  # Always 1 BTC
```

### 5. Multi-Asset Strategies

Explicitly specify price columns:
```python
engine = Engine(
    strategy,
    data,
    price_columns={
        "BTC": "btc_close",
        "ETH": "eth_close",
        "SOL": "sol_close",
    }
)
```

Then in `next()`:
```python
def next(self, ctx):
    ctx.portfolio.order_target_percent("BTC", 0.33)
    ctx.portfolio.order_target_percent("ETH", 0.33)
    ctx.portfolio.order_target_percent("SOL", 0.34)
```

## Performance Optimization

### For Strategy Execution

1. **Minimize work in `next()`**: Move everything possible to `preprocess()`
2. **Use Polars expressions**: Much faster than Python loops
3. **Avoid DataFrame operations in `next()`**: Only access pre-computed values

### For Parameter Search

1. **Use `backtest_batch()`**: Parallelizes across CPU cores
2. **Set `n_jobs` appropriately**: Usually `n_jobs=cpu_count()`
3. **Use LazyFrames**: For very large datasets, consider Polars LazyFrames

### Memory Management

1. **Don't store full history**: Only track what's needed for metrics
2. **Use `.select()` to drop unused columns**: After preprocessing
3. **Clear large objects**: In `on_finish()` if needed

## Common Patterns

### Trend Following

```python
class TrendFollower(Strategy):
    def preprocess(self, df):
        fast = self.params.get("fast", 10)
        slow = self.params.get("slow", 30)
        return df.with_columns([
            ind.ema("close", fast).alias("ema_fast"),
            ind.ema("close", slow).alias("ema_slow"),
        ])
    
    def next(self, ctx):
        if ctx.row.get("ema_fast") is None:
            return
        
        if ctx.row["ema_fast"] > ctx.row["ema_slow"]:
            ctx.portfolio.order_target_percent("asset", 1.0)
        else:
            ctx.portfolio.close_position("asset")
```

### Mean Reversion

```python
class MeanReversion(Strategy):
    def preprocess(self, df):
        upper, middle, lower = ind.bollinger_bands("close", 20, 2.0)
        return df.with_columns([
            upper.alias("bb_upper"),
            middle.alias("bb_middle"),
            lower.alias("bb_lower"),
        ])
    
    def next(self, ctx):
        if ctx.row.get("bb_lower") is None:
            return
        
        if ctx.row["close"] < ctx.row["bb_lower"]:
            ctx.portfolio.order_target_percent("asset", 1.0)
        elif ctx.row["close"] > ctx.row["bb_upper"]:
            ctx.portfolio.close_position("asset")
```

### Portfolio Rebalancing

```python
class PortfolioRebalancer(Strategy):
    def preprocess(self, df):
        lookback = self.params.get("lookback", 30)
        return df.with_columns([
            ind.returns("btc_close", lookback).alias("btc_momentum"),
            ind.returns("eth_close", lookback).alias("eth_momentum"),
        ])
    
    def next(self, ctx):
        if ctx.bar_index % 30 != 0:  # Rebalance monthly
            return
        
        # Equal weight
        ctx.portfolio.order_target_percent("BTC", 0.5)
        ctx.portfolio.order_target_percent("ETH", 0.5)
```

## Testing

Run tests with pytest:
```bash
pytest tests/
```

Test structure:
- `tests/test_indicators.py`: Indicator calculations
- `tests/test_core.py`: Portfolio and Engine
- `tests/test_runner.py`: Parallel execution

## Troubleshooting

### Common Issues

**"No price columns found in data"**
- Solution: Ensure your DataFrame has a "close" column or specify `price_columns`

**"Not enough cash"**
- Check commission and slippage settings
- Reduce position sizes
- Increase `initial_cash`

**Slow performance**
- Move calculations from `next()` to `preprocess()`
- Use Polars expressions instead of Python loops
- Enable parallel execution with `backtest_batch()`

**Type errors with Polars**
- Ensure your data types are correct (Float64 for prices)
- Handle None values properly
- Use `.drop_nulls()` or `.fill_null()` as needed

## Future Extensions

Potential areas for expansion:

1. **Advanced order types**: Stop-loss, take-profit, trailing stops
2. **Slippage models**: Volume-based, volatility-based
3. **Transaction costs**: Maker/taker fees, exchange-specific
4. **Risk management**: Position limits, exposure limits
5. **Visualization**: Equity curves, drawdown plots
6. **Live trading interface**: Paper trading, real execution

## Contributing

When adding new features:

1. Add tests to appropriate test file
2. Update docstrings with examples
3. Keep the hybrid architecture (vectorized + event-driven)
4. Maintain compatibility with the LLM-friendly API
5. Document in this agents.md file

## Questions?

This library is designed to be simple and transparent. If something is unclear:
1. Check the docstrings in the source code
2. Look at the test files for usage examples
3. Review the README.md for user-facing documentation
