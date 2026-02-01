# PolarBtest - Implementation Guide

This document provides detailed implementation information for working with PolarBtest.

## Essential APIs

### Strategy Base Class
```python
class MyStrategy(Strategy):
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        # Vectorized feature engineering (called ONCE)
        pass
    
    def next(self, ctx: BacktestContext) -> None:
        # Event-driven logic (called EVERY bar after warmup)
        pass
```

### Portfolio Methods
```python
portfolio.order_target_percent(asset, percent)  # Most common
portfolio.order_target_value(asset, value)
portfolio.close_position(asset)
portfolio.get_position(asset)
portfolio.get_value()
```

### Running Backtests
```python
from polarbtest.runner import backtest, backtest_batch, optimize

# Single backtest
results = backtest(MyStrategy, data, params={...})

# Parallel batch
results = backtest_batch(MyStrategy, data, param_sets, n_jobs=-1)

# Grid search optimization
best = optimize(MyStrategy, data, param_grid, objective="sharpe_ratio")
```

## Detailed Component Reference

### Portfolio Class

**Internal State:**
- `cash`: Current cash balance (float)
- `positions`: Dict mapping asset names to quantities (dict)
- `equity_curve`: Historical portfolio values (list)
- `_current_prices`: Current market prices for valuation (dict)

**All Methods:**
```python
# Market orders
portfolio.order(asset, quantity)                    # Buy/sell quantity
portfolio.order_target(asset, target_quantity)      # Target specific quantity
portfolio.order_target_percent(asset, percent)      # Target % of portfolio (0.0-1.0)
portfolio.order_target_value(asset, value)          # Target dollar value

# Position management
portfolio.close_position(asset)                     # Close entire position
portfolio.get_position(asset)                       # Get current quantity
portfolio.get_value()                               # Total portfolio value (cash + positions)
```

**Transaction Costs:**
Orders are subject to commission and slippage configured in Engine:
```python
# Commission: percentage fee on order value
# Slippage: adverse price movement on execution
cost = order_value * commission + order_value * slippage
```

### Strategy Class

**Required Methods:**
```python
def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
    """
    Vectorized feature engineering.
    Called ONCE before backtest starts.
    Use Polars expressions to calculate indicators, ML features, etc.
    Must return DataFrame with all features needed for next().
    """
    pass

def next(self, ctx: BacktestContext) -> None:
    """
    Event-driven logic.
    Called on EVERY bar after warmup period.
    Use ctx.portfolio to place orders.
    Access current data via ctx.row dictionary.
    """
    pass
```

**Optional Methods:**
```python
def on_start(self, portfolio: Portfolio):
    """Called once before backtest starts (after preprocess)"""
    pass

def on_finish(self, portfolio: Portfolio):
    """Called once after backtest completes"""
    pass
```

**Parameter Handling:**
```python
class MyStrategy(Strategy):
    def __init__(self, **params):
        super().__init__(**params)
        # Extract parameters with defaults
        self.fast_period = params.get("fast", 10)
        self.slow_period = params.get("slow", 30)
        self.threshold = params.get("threshold", 0.02)
```

### BacktestContext

**Structure:**
```python
@dataclass
class BacktestContext:
    timestamp: Any              # Current bar timestamp
    row: Dict[str, Any]        # Current bar data (all columns as dict)
    portfolio: Portfolio       # Reference to portfolio for orders
    bar_index: int            # Current bar index (0-based)
```

**Usage Example:**
```python
def next(self, ctx):
    # Access current prices and indicators
    price = ctx.row["close"]
    sma = ctx.row["sma_20"]
    volume = ctx.row["volume"]
    
    # Check bar index for time-based logic
    if ctx.bar_index < 20:
        return  # Skip first 20 bars
    
    # Place orders
    if price > sma:
        ctx.portfolio.order_target_percent("asset", 1.0)
```

### Engine Class

**Full Constructor:**
```python
engine = Engine(
    strategy=strategy_instance,        # Instantiated strategy
    data=polars_dataframe,            # Input data
    initial_cash=100_000,             # Starting capital
    commission=0.001,                 # 0.1% per trade
    slippage=0.0005,                  # 0.05% adverse price movement
    price_columns=None,               # Dict mapping assets to price columns
    warmup="auto",                    # "auto", integer, or 0
    order_delay=0,                    # Bars to delay order execution
)

results = engine.run()
```

**Warmup Parameter Details:**
- `warmup="auto"` (default): Finds first bar where all columns (except timestamp) are non-null
- `warmup=N` (integer): Skip first N bars manually
- `warmup=0`: No warmup, start immediately (requires None checking in next())

**Price Columns:**
For single asset (default):
```python
# Automatically uses "close" column
engine = Engine(strategy, data, initial_cash=100_000)
```

For multiple assets:
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

**Execution Flow:**
1. Call `strategy.preprocess(data)` to generate features
2. Calculate warmup period (if warmup="auto")
3. Call `strategy.on_start(portfolio)`
4. For each bar (starting after warmup):
   - Update portfolio prices from current bar
   - Create BacktestContext with current data
   - Call `strategy.next(ctx)`
   - Record portfolio equity
5. Call `strategy.on_finish(portfolio)`
6. Calculate performance metrics
7. Return results dictionary

### Indicators Module

All indicators return Polars expressions for lazy evaluation:

**Trend Indicators:**
```python
ind.sma(column: str, period: int) -> pl.Expr
ind.ema(column: str, period: int) -> pl.Expr
ind.macd(column: str, fast: int, slow: int, signal: int) -> tuple[pl.Expr, pl.Expr, pl.Expr]
# macd returns (macd_line, signal_line, histogram)
```

**Momentum Indicators:**
```python
ind.rsi(column: str, period: int) -> pl.Expr
ind.returns(column: str, periods: int = 1) -> pl.Expr
ind.log_returns(column: str, periods: int = 1) -> pl.Expr
```

**Volatility Indicators:**
```python
ind.bollinger_bands(column: str, period: int, std_dev: float) -> tuple[pl.Expr, pl.Expr, pl.Expr]
# Returns (upper_band, middle_band, lower_band)

ind.atr(high: str, low: str, close: str, period: int) -> pl.Expr
```

**Signal Indicators:**
```python
ind.crossover(fast_column: str, slow_column: str) -> pl.Expr  # Returns bool
ind.crossunder(fast_column: str, slow_column: str) -> pl.Expr  # Returns bool
```

**Usage Example:**
```python
def preprocess(self, df):
    # Get MACD components
    macd_line, signal_line, histogram = ind.macd("close", 12, 26, 9)
    
    # Get Bollinger Bands
    bb_upper, bb_middle, bb_lower = ind.bollinger_bands("close", 20, 2.0)
    
    return df.with_columns([
        macd_line.alias("macd"),
        signal_line.alias("macd_signal"),
        histogram.alias("macd_hist"),
        bb_upper.alias("bb_upper"),
        bb_middle.alias("bb_middle"),
        bb_lower.alias("bb_lower"),
        ind.rsi("close", 14).alias("rsi"),
        ind.atr("high", "low", "close", 14).alias("atr"),
    ])
```

### Runner Module

**Single Backtest:**
```python
from polarbtest.runner import backtest

results = backtest(
    strategy_class=MyStrategy,
    data=df,
    params={"fast": 10, "slow": 30},
    initial_cash=100_000,
    commission=0.001,
    slippage=0.0005,
)
```

**Parallel Batch:**
```python
from polarbtest.runner import backtest_batch

param_sets = [
    {"fast": 5, "slow": 20},
    {"fast": 10, "slow": 30},
    {"fast": 20, "slow": 50},
]

results_list = backtest_batch(
    strategy_class=MyStrategy,
    data=df,
    param_sets=param_sets,
    n_jobs=-1,  # Use all CPU cores
)
```

**Grid Search Optimization:**
```python
from polarbtest.runner import optimize

param_grid = {
    "fast": [5, 10, 15, 20],
    "slow": [20, 30, 40, 50],
    "threshold": [0.01, 0.02, 0.03],
}

best_results = optimize(
    strategy_class=MyStrategy,
    data=df,
    param_grid=param_grid,
    objective="sharpe_ratio",  # Metric to maximize
    maximize=True,
    n_jobs=-1,
)
```

**Walk-Forward Analysis:**
```python
from polarbtest.runner import walk_forward_analysis

wf_results = walk_forward_analysis(
    strategy_class=MyStrategy,
    data=df,
    param_grid=param_grid,
    train_periods=252,  # 1 year training
    test_periods=63,    # 3 months testing
    objective="sharpe_ratio",
)
```

## Data Formats

### Input Data Requirements

**Single Asset (Minimum):**
```python
import polars as pl

df = pl.DataFrame({
    "timestamp": [...],  # Optional but recommended
    "close": [...],      # Required
})
```

**Single Asset (Full OHLCV):**
```python
df = pl.DataFrame({
    "timestamp": [...],
    "open": [...],
    "high": [...],
    "low": [...],
    "close": [...],
    "volume": [...],
})
```

**Multiple Assets:**
```python
df = pl.DataFrame({
    "timestamp": [...],
    "btc_open": [...],
    "btc_high": [...],
    "btc_low": [...],
    "btc_close": [...],
    "btc_volume": [...],
    "eth_open": [...],
    "eth_high": [...],
    "eth_low": [...],
    "eth_close": [...],
    "eth_volume": [...],
})
```

### Output Metrics Dictionary

**Complete Structure:**
```python
{
    # Returns
    "total_return": float,              # Total % return
    "cagr": float,                      # Compound annual growth rate
    
    # Risk-adjusted returns
    "sharpe_ratio": float,              # Return/risk ratio
    "sortino_ratio": float,             # Return/downside risk ratio
    "calmar_ratio": float,              # CAGR/max drawdown ratio
    
    # Risk metrics
    "max_drawdown": float,              # Maximum peak-to-trough decline
    "volatility": float,                # Daily volatility
    "volatility_annualized": float,     # Annualized volatility
    
    # Trade statistics
    "win_rate": float,                  # % of winning trades
    "profit_factor": float,             # Gross profit / gross loss
    
    # Portfolio state
    "final_equity": float,              # Final portfolio value
    "final_cash": float,                # Final cash balance
    "final_positions": dict,            # Final positions {asset: quantity}
    
    # Metadata
    "params": dict,                     # Strategy parameters used
    "success": bool,                    # Whether backtest completed
}
```

## Advanced Patterns

### ML Model with Walk-Forward Validation

```python
class MLStrategy(Strategy):
    def preprocess(self, df):
        # Calculate features
        df = df.with_columns([
            ind.returns("close", 5).alias("ret_5d"),
            ind.returns("close", 20).alias("ret_20d"),
            ind.rsi("close", 14).alias("rsi"),
            ind.ema("close", 20).alias("ema_20"),
        ])
        
        # Use walk-forward approach for training
        train_size = int(len(df) * 0.7)
        train_df = df[:train_size].drop_nulls()
        
        # Prepare training data
        feature_cols = ["ret_5d", "ret_20d", "rsi"]
        X_train = train_df.select(feature_cols).to_numpy()
        y_train = (train_df["close"].shift(-1) > train_df["close"]).to_numpy()[:-1]
        
        # Train model
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5)
        self.model.fit(X_train[:-1], y_train)
        
        # Generate predictions for entire dataset
        X_full = df.select(feature_cols).fill_null(0).to_numpy()
        predictions = self.model.predict_proba(X_full)[:, 1]
        
        return df.with_columns([
            pl.Series("prediction", predictions)
        ])
    
    def next(self, ctx):
        prediction = ctx.row["prediction"]
        
        if prediction > 0.6:
            ctx.portfolio.order_target_percent("asset", 1.0)
        elif prediction < 0.4:
            ctx.portfolio.close_position("asset")
```

### Multi-Asset Momentum Rotation

```python
class MomentumRotation(Strategy):
    def preprocess(self, df):
        lookback = self.params.get("lookback", 30)
        
        return df.with_columns([
            ind.returns("btc_close", lookback).alias("btc_momentum"),
            ind.returns("eth_close", lookback).alias("eth_momentum"),
            ind.returns("sol_close", lookback).alias("sol_momentum"),
        ])
    
    def next(self, ctx):
        # Only rebalance monthly
        if ctx.bar_index % 30 != 0:
            return
        
        # Calculate momentum scores
        momentums = {
            "BTC": ctx.row["btc_momentum"],
            "ETH": ctx.row["eth_momentum"],
            "SOL": ctx.row["sol_momentum"],
        }
        
        # Rank by momentum
        sorted_assets = sorted(momentums.items(), key=lambda x: x[1], reverse=True)
        
        # Invest only in top 2 performers
        top_assets = [asset for asset, _ in sorted_assets[:2]]
        
        for asset in ["BTC", "ETH", "SOL"]:
            if asset in top_assets:
                ctx.portfolio.order_target_percent(asset, 0.5)
            else:
                ctx.portfolio.close_position(asset)
```

### Risk-Adjusted Position Sizing

```python
class RiskParity(Strategy):
    def preprocess(self, df):
        return df.with_columns([
            ind.sma("close", 20).alias("sma_20"),
            ind.atr("high", "low", "close", 14).alias("atr"),
        ])
    
    def next(self, ctx):
        price = ctx.row["close"]
        sma = ctx.row["sma_20"]
        atr = ctx.row["atr"]
        
        # Don't trade if ATR is too high (too risky)
        if atr / price > 0.05:  # 5% volatility threshold
            ctx.portfolio.close_position("asset")
            return
        
        # Position size inversely proportional to volatility
        base_allocation = 1.0
        risk_adjusted_allocation = base_allocation * (0.02 / (atr / price))
        risk_adjusted_allocation = min(risk_adjusted_allocation, 1.0)
        
        if price > sma:
            ctx.portfolio.order_target_percent("asset", risk_adjusted_allocation)
        else:
            ctx.portfolio.close_position("asset")
```

## Performance Optimization

### Strategy Execution Performance

**Bad - Slow:**
```python
def next(self, ctx):
    # Recalculating on every bar
    prices = []
    for i in range(20):
        prices.append(self.historical_data[ctx.bar_index - i]["close"])
    sma = sum(prices) / len(prices)
```

**Good - Fast:**
```python
def preprocess(self, df):
    # Calculate once using Polars
    return df.with_columns([
        ind.sma("close", 20).alias("sma_20")
    ])

def next(self, ctx):
    # Just access pre-computed value
    sma = ctx.row["sma_20"]
```

### Parameter Search Performance

**Sequential (Slow):**
```python
results = []
for params in param_sets:
    result = backtest(MyStrategy, data, params)
    results.append(result)
```

**Parallel (Fast):**
```python
results = backtest_batch(MyStrategy, data, param_sets, n_jobs=-1)
```

### Memory Management

**For Large Datasets:**
```python
def preprocess(self, df):
    # Calculate indicators
    df = df.with_columns([
        ind.sma("close", 20).alias("sma_20"),
        ind.rsi("close", 14).alias("rsi"),
    ])
    
    # Drop unused columns to save memory
    df = df.select(["timestamp", "close", "sma_20", "rsi"])
    
    return df
```

## Troubleshooting Guide

### Common Issues and Solutions

**Issue: "No price columns found in data"**
```python
# Solution: Ensure DataFrame has "close" column or specify price_columns
engine = Engine(
    strategy,
    data,
    price_columns={"BTC": "btc_close"}  # Explicitly specify
)
```

**Issue: "Not enough cash" errors**
```python
# Solutions:
# 1. Increase initial cash
engine = Engine(strategy, data, initial_cash=1_000_000)

# 2. Reduce commission/slippage
engine = Engine(strategy, data, commission=0.0001, slippage=0.0001)

# 3. Use percentage-based sizing instead of fixed quantities
ctx.portfolio.order_target_percent("asset", 0.5)  # Not order(asset, 100)
```

**Issue: Slow backtest performance**
```python
# Solutions:
# 1. Move calculations to preprocess()
def preprocess(self, df):
    return df.with_columns([ind.sma("close", 20).alias("sma")])

# 2. Use Polars expressions, not Python loops
# Bad: df.apply(lambda x: custom_function(x))
# Good: df.with_columns([pl.col("close").rolling_mean(20)])

# 3. Enable parallel execution for parameter search
results = backtest_batch(strategy, data, params, n_jobs=-1)
```

**Issue: Type errors with Polars**
```python
# Ensure correct dtypes
df = df.with_columns([
    pl.col("close").cast(pl.Float64),
    pl.col("volume").cast(pl.Float64),
])

# Handle null values
df = df.drop_nulls()  # or
df = df.fill_null(0)  # or
df = df.fill_null(strategy="forward")
```

**Issue: Indicators return None in next()**
```python
# Solution 1: Use auto warmup (default)
engine = Engine(strategy, data, warmup="auto")

# Solution 2: Manual warmup period
engine = Engine(strategy, data, warmup=30)

# Solution 3: Check for None if warmup=0
def next(self, ctx):
    if ctx.row.get("sma") is None:
        return
    # Safe to use sma now
```

### Debug Strategies

**Print current state:**
```python
def next(self, ctx):
    if ctx.bar_index % 100 == 0:  # Every 100 bars
        print(f"Bar {ctx.bar_index}: Value={ctx.portfolio.get_value():.2f}")
```

**Track trades:**
```python
class DebugStrategy(Strategy):
    def on_start(self, portfolio):
        self.trades = []
    
    def next(self, ctx):
        prev_pos = ctx.portfolio.get_position("asset")
        
        # Trading logic
        if ctx.row["close"] > ctx.row["sma"]:
            ctx.portfolio.order_target_percent("asset", 1.0)
        
        new_pos = ctx.portfolio.get_position("asset")
        
        # Log trade
        if prev_pos != new_pos:
            self.trades.append({
                "timestamp": ctx.timestamp,
                "action": "buy" if new_pos > prev_pos else "sell",
                "price": ctx.row["close"],
                "position": new_pos,
            })
    
    def on_finish(self, portfolio):
        print(f"Total trades: {len(self.trades)}")
        for trade in self.trades[:5]:  # Show first 5
            print(trade)
```

## Strategy Pattern Library

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
        if ctx.row["close"] < ctx.row["bb_lower"]:
            ctx.portfolio.order_target_percent("asset", 1.0)
        elif ctx.row["close"] > ctx.row["bb_upper"]:
            ctx.portfolio.close_position("asset")
```

### Breakout Strategy
```python
class Breakout(Strategy):
    def preprocess(self, df):
        lookback = self.params.get("lookback", 20)
        return df.with_columns([
            pl.col("high").rolling_max(lookback).alias("resistance"),
            pl.col("low").rolling_min(lookback).alias("support"),
        ])
    
    def next(self, ctx):
        if ctx.row["close"] > ctx.row["resistance"]:
            ctx.portfolio.order_target_percent("asset", 1.0)
        elif ctx.row["close"] < ctx.row["support"]:
            ctx.portfolio.close_position("asset")
```

### RSI Strategy
```python
class RSIStrategy(Strategy):
    def preprocess(self, df):
        return df.with_columns([
            ind.rsi("close", 14).alias("rsi")
        ])
    
    def next(self, ctx):
        rsi = ctx.row["rsi"]
        oversold = self.params.get("oversold", 30)
        overbought = self.params.get("overbought", 70)
        
        if rsi < oversold:
            ctx.portfolio.order_target_percent("asset", 1.0)
        elif rsi > overbought:
            ctx.portfolio.close_position("asset")
```

### Dual Momentum
```python
class DualMomentum(Strategy):
    def preprocess(self, df):
        lookback = self.params.get("lookback", 60)
        return df.with_columns([
            ind.returns("close", lookback).alias("absolute_momentum"),
            (pl.col("close") / ind.sma("close", lookback) - 1).alias("relative_momentum"),
        ])
    
    def next(self, ctx):
        # Both absolute and relative momentum must be positive
        if ctx.row["absolute_momentum"] > 0 and ctx.row["relative_momentum"] > 0:
            ctx.portfolio.order_target_percent("asset", 1.0)
        else:
            ctx.portfolio.close_position("asset")
```
