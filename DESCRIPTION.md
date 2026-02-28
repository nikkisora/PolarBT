# PolarBtest

A lightweight, high-performance backtesting library for trading strategy development and optimization. Built on Polars for fast vectorized data processing with an event-driven execution loop for flexible strategy logic.

**Version**: 0.1.0
**Python**: >= 3.10
**Dependencies**: polars >= 0.19.0, numpy >= 1.24.0

## Architecture

Hybrid design: vectorized preprocessing (Polars) + event-driven execution loop.

```
Strategy.preprocess(df)     →  Calculate all indicators at once (vectorized)
         ↓
Engine event loop           →  Iterate bars, update prices, call strategy.next()
         ↓
Portfolio                   →  Manage positions, orders, cash, equity curve
         ↓
Metrics                     →  Calculate performance metrics from equity curve
```

## Module Structure

```
polarbtest/
├── __init__.py       # Public API exports
├── core.py           # Portfolio, Strategy, Engine, BacktestContext
├── orders.py         # Order, OrderType, OrderStatus
├── trades.py         # Trade, TradeTracker
├── indicators.py     # Technical indicators as Polars expressions
├── metrics.py        # Performance metrics
└── runner.py         # backtest(), backtest_batch(), optimize(), walk_forward_analysis()
```

## Core Components

### Strategy

Base class with two required methods:

```python
class MyStrategy(Strategy):
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Called once. Add indicator columns using vectorized Polars ops."""
        return df.with_columns([ind.sma("close", 20).alias("sma_20")])

    def next(self, ctx: BacktestContext) -> None:
        """Called every bar after warmup. Place orders via ctx.portfolio."""
        if ctx.row["close"] > ctx.row["sma_20"]:
            ctx.portfolio.order_target_percent("asset", 1.0)
        else:
            ctx.portfolio.close_position("asset")
```

Optional lifecycle hooks: `on_start(portfolio)`, `on_finish(portfolio)`.

Parameters passed via `__init__(**params)` and accessible as `self.params`.

### BacktestContext

Passed to `strategy.next()` each bar:

- `ctx.row` — dict of all column values for current bar
- `ctx.portfolio` — Portfolio instance for placing orders
- `ctx.bar_index` — current bar index (0-based)
- `ctx.timestamp` — current timestamp

### Portfolio

Manages cash, positions, orders, and risk management.

**Order Methods:**
- `order(asset, quantity, limit_price=None)` — place market or limit order
- `order_target(asset, target_quantity)` — order to reach target position
- `order_target_percent(asset, percent)` — target % of portfolio value
- `order_target_value(asset, value)` — target dollar value
- `close_position(asset)` / `close_all_positions()`
- `order_day(asset, quantity, limit_price=None, bars_valid=None)` — day order with auto-expiry
- `order_gtc(asset, quantity, limit_price=None)` — good-till-cancelled
- `order_bracket(asset, quantity, stop_loss=None, take_profit=None, ...)` — entry + SL + TP (OCO)

**Risk Management:**
- `set_stop_loss(asset, stop_price=None, stop_pct=None)`
- `set_take_profit(asset, target_price=None, target_pct=None)`
- `set_trailing_stop(asset, trail_pct=None, trail_amount=None)`
- `remove_stop_loss/take_profit/trailing_stop(asset)`
- `get_stop_loss/take_profit/trailing_stop(asset)`

**Order Management:**
- `get_orders(status=None, asset=None)` — filter orders
- `get_order(order_id)` / `cancel_order(order_id)`

**Position & Value:**
- `get_position(asset)` — current quantity
- `get_value()` — total portfolio value (cash + positions)
- `get_trades()` — completed trades as DataFrame
- `get_trade_stats()` — aggregate trade statistics

**Commission:**
- Percentage only: `commission=0.001` (0.1% per trade)
- Fixed + percentage: `commission=(5.0, 0.001)` ($5 + 0.1% per trade)

**Day Order Auto-Expiry** (priority order):
1. Timestamp-based: expires when date changes (if timestamps available)
2. Explicit `bars_valid` parameter
3. `bars_per_day` Portfolio config (smart end-of-day calculation)
4. Default: 1 bar

### Order System

**Order Types** (OrderType enum):
- `MARKET` — immediate execution at current price
- `LIMIT` — execute at limit price or better (uses OHLC for fill detection)
- `STOP` / `STOP_LIMIT` — defined in enum but execution not fully wired

**Order Statuses**: PENDING, FILLED, PARTIAL, CANCELLED, REJECTED, EXPIRED

**Order Delay**: configurable via `order_delay` parameter (0 = immediate, 1 = next bar).

### Trade Tracking

Automatic trade lifecycle tracking via `TradeTracker`:
- Records entry/exit prices, sizes, timestamps, commissions
- Calculates P&L, return %, bars held
- Tracks MAE (Maximum Adverse Excursion) and MFE (Maximum Favorable Excursion)
- Exports to Polars DataFrame
- Aggregate stats: win_rate, avg_win, avg_loss, profit_factor, total_pnl

### Engine

Orchestrates the backtest:

```python
engine = Engine(
    strategy=strategy_instance,
    data=df,                          # pl.DataFrame or dict[str, pl.DataFrame]
    initial_cash=100_000,
    commission=0.001,                 # or (5.0, 0.001)
    slippage=0.0005,
    price_columns=None,               # auto-detected
    warmup="auto",                    # "auto", int, or 0
    order_delay=0,
)
results = engine.run()
```

**Warmup:**
- `"auto"` (default): finds first bar where all columns are non-null
- Integer: skip that many bars
- `0`: no warmup

**Multi-asset input:** pass `dict[str, pl.DataFrame]` — automatically merged with prefixed columns.

### Indicators

All return Polars expressions for use in `preprocess()`:

| Function | Description |
|---|---|
| `sma(column, period)` | Simple Moving Average |
| `ema(column, period)` | Exponential Moving Average |
| `rsi(column, period)` | Relative Strength Index |
| `macd(column, fast, slow, signal)` | MACD (returns tuple of 3 exprs) |
| `bollinger_bands(column, period, std_dev)` | Bollinger Bands (returns tuple of 3 exprs) |
| `atr(high, low, close, period)` | Average True Range |
| `returns(column, periods)` | Percentage returns |
| `log_returns(column, periods)` | Logarithmic returns |
| `crossover(fast_col, slow_col)` | Bullish crossover detection |
| `crossunder(fast_col, slow_col)` | Bearish crossover detection |

### Metrics

**`calculate_metrics(equity_df, initial_capital)`** returns:
- total_return, cagr
- sharpe_ratio, sortino_ratio, calmar_ratio
- max_drawdown, volatility, volatility_annualized
- win_rate, avg_win, avg_loss, profit_factor
- initial_equity, final_equity

**Standalone functions:**
- `sharpe_ratio()`, `sortino_ratio()`, `max_drawdown()`, `calmar_ratio()`
- `omega_ratio()`, `value_at_risk()`, `conditional_value_at_risk()`
- `rolling_sharpe()`, `underwater_plot_data()`

### Runner

**`backtest(strategy_class, data, params=None, ...)`** — run single backtest, returns metrics dict.

**`backtest_batch(strategy_class, data, param_sets, n_jobs=None, ...)`** — parallel execution across CPU cores, returns DataFrame of results.

**`optimize(strategy_class, data, param_grid, objective="sharpe_ratio", ...)`** — grid search, returns best result dict.

**`walk_forward_analysis(strategy_class, data, param_grid, train_periods, test_periods, ...)`** — walk-forward optimization with train/test splits, returns DataFrame of fold results.

## Data Format

**Single asset (minimum):**
```python
df = pl.DataFrame({"close": [100.0, 101.0, ...]})
```

**Single asset (full OHLCV):**
```python
df = pl.DataFrame({
    "timestamp": [...], "open": [...], "high": [...],
    "low": [...], "close": [...], "volume": [...]
})
```

**Multi-asset (dict input):**
```python
data = {
    "BTC": pl.DataFrame({"timestamp": [...], "close": [...]}),
    "ETH": pl.DataFrame({"timestamp": [...], "close": [...]}),
}
results = backtest(MyStrategy, data, params={...})
```

## Test Coverage

152 tests passing, 2 skipped (short selling edge cases). Test files:
- test_core.py, test_indicators.py, test_orders.py, test_limit_orders.py
- test_trades.py, test_runner.py, test_warmup.py
- test_take_profit.py, test_trailing_stop.py, test_bracket_orders.py
- test_order_expiry.py, test_mae_mfe.py
