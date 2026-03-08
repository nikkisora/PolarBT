# PolarBT Agent Reference

Complete reference for PolarBT — a hybrid backtesting engine combining vectorized preprocessing (Polars) with event-driven execution for trading strategies.

## Architecture

PolarBT uses a two-phase architecture:

1. **Preprocess phase** — Strategy computes all indicators using vectorized Polars expressions (runs once, fast)
2. **Event loop phase** — Engine iterates bar-by-bar, calling `strategy.next()` with a `BacktestContext` containing the current row and portfolio

Data flows through: `raw DataFrame → Strategy.preprocess() → Engine loop → Strategy.next() → Portfolio orders → BacktestMetrics`

## Quick Start

```python
import polars as pl
from polarbt import Engine, Strategy
from polarbt import indicators as ind
from polarbt.core import BacktestContext


class SMACross(Strategy):
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            ind.sma("close", 10).alias("sma_fast"),
            ind.sma("close", 30).alias("sma_slow"),
        ).with_columns(
            ind.crossover("sma_fast", "sma_slow").alias("buy"),
            ind.crossunder("sma_fast", "sma_slow").alias("sell"),
        )

    def next(self, ctx: BacktestContext) -> None:
        if ctx.row.get("buy"):
            ctx.portfolio.order_target_percent("asset", 1.0)
        elif ctx.row.get("sell"):
            ctx.portfolio.close_position("asset")


engine = Engine(SMACross(), data, commission=0.005, initial_cash=100_000)
results = engine.run()
print(results)
```

## Core Classes

### Strategy

Abstract base class. Subclass and implement `preprocess()` and `next()`.

```python
class Strategy(ABC):
    def __init__(self, **params: Any) -> None
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame  # REQUIRED
    def next(self, ctx: BacktestContext) -> None             # REQUIRED
    def on_start(self, portfolio: Portfolio) -> None          # optional hook
    def on_finish(self, portfolio: Portfolio) -> None         # optional hook
```

- Declare parameters as class attributes using `param()`:
  ```python
  class MyStrategy(Strategy):
      fast = param(10)
      slow = param(30)
  ```
- Access them as `self.fast`, `self.slow` — they read from `self.params` with the declared default
- `self.params` — dict of all `**params` passed to `__init__` (also accessible directly)
- Strategy parameters are passed as keyword arguments: `MyStrategy(fast=10, slow=30)`
- `preprocess()` must return a DataFrame with the same rows, plus any added indicator columns
- `next()` is called every bar after warmup; use `ctx.portfolio` to place orders

### BacktestContext

Dataclass passed to `Strategy.next()` on each bar:

| Attribute | Type | Description |
|---|---|---|
| `timestamp` | `Any` | Current bar timestamp |
| `row` | `dict[str, Any]` | Current bar data (all columns as key-value pairs) |
| `portfolio` | `Portfolio` | Reference to the portfolio for placing orders |
| `bar_index` | `int` | Current bar index (0-based) |

Access indicator values: `ctx.row["sma_fast"]`, `ctx.row.get("buy")`

### Engine

Orchestrates the backtest simulation.

```python
Engine(
    strategy: Strategy,
    data: pl.DataFrame | dict[str, pl.DataFrame],
    initial_cash: float = 100_000.0,
    commission: float | tuple[float, float] | CommissionModel = 0.0,
    slippage: float = 0.0,
    price_columns: dict[str, str] | None = None,
    warmup: int | str = "auto",
    order_delay: int = 0,
    borrow_rate: float = 0.0,
    bars_per_day: float | None = None,
    max_position_size: float | None = None,
    max_total_exposure: float | None = None,
    max_drawdown_stop: float | None = None,
    daily_loss_limit: float | None = None,
    leverage: float = 1.0,
    maintenance_margin: float | None = None,
    fractional_shares: bool = True,
    factor_column: str | None = None,
)
```

**Key parameters:**

| Parameter | Description |
|---|---|
| `data` | Single `pl.DataFrame` with OHLCV columns, or `dict[str, pl.DataFrame]` for multi-asset |
| `commission` | `float` (percentage, e.g. 0.001=0.1%), `tuple` (fixed+percent, e.g. (5.0, 0.001)), or `CommissionModel` |
| `slippage` | Fraction applied to execution price (e.g. 0.0005=0.05%) |
| `warmup` | `"auto"` (default, detects first bar where all columns are non-null) or `int` |
| `order_delay` | Bars before order executes. 0=same bar, 1=next bar (more realistic) |
| `leverage` | Max leverage multiplier. Default 1.0 (no leverage). Cash can go negative when >1 |
| `maintenance_margin` | Min margin ratio before auto-liquidation (margin call). Only relevant when leverage>1 |
| `max_position_size` | Max single position as fraction of portfolio value (e.g. 0.5=50%) |
| `max_total_exposure` | Max sum of abs position values / portfolio value (e.g. 1.5=150%) |
| `max_drawdown_stop` | Halt new risk-increasing orders when drawdown exceeds this (e.g. 0.2=20%) |
| `daily_loss_limit` | Halt trading for the day when intraday loss exceeds this (e.g. 0.05=5%) |
| `borrow_rate` | Annual rate for short positions (e.g. 0.02=2%/year), charged daily/252 |
| `bars_per_day` | For intraday data: bars in a trading day (e.g. 390 for 1-min bars) |
| `fractional_shares` | When `False`, order quantities are truncated to whole numbers |
| `factor_column` | Column with price adjustment factors. When present, commissions are calculated on raw (unadjusted) prices |

**Data format:** The engine auto-detects common column names (Date→timestamp, Open→open, Close→close, etc.). For single-asset data, the default price column is `"close"` with implicit asset name `"asset"`.

**Multi-asset input:**
```python
engine = Engine(strategy, {"BTC": btc_df, "ETH": eth_df})
# Merges into wide format: timestamp, BTC_close, ETH_close, ...
# price_columns auto-detected: {"BTC": "BTC_close", "ETH": "ETH_close"}
```

**`engine.run()` returns** a `BacktestMetrics` dataclass with fields:

| Field | Type | Description |
|---|---|---|
| `total_return` | `float` | Total return as fraction |
| `cagr` | `float` | Compound annual growth rate |
| `sharpe_ratio` | `float` | Annualized Sharpe ratio |
| `sortino_ratio` | `float` | Annualized Sortino ratio |
| `max_drawdown` | `float` | Max drawdown as positive fraction |
| `calmar_ratio` | `float` | CAGR / max drawdown |
| `volatility_annualized` | `float` | Annualized volatility |
| `final_equity` | `float` | Final portfolio value |
| `equity_peak` | `float` | Peak portfolio value |
| `buy_hold_return` | `float` | Buy & hold return for comparison |
| `trade_stats` | `TradeStats` | Aggregate trade stats (total_trades, win_rate, profit_factor, etc.) |
| `trades` | `pl.DataFrame` | All completed trades |
| `best_trade_pct` | `float` | Best trade return % |
| `worst_trade_pct` | `float` | Worst trade return % |
| `expectancy` | `float` | Average P&L per trade |
| `sqn` | `float` | System Quality Number |
| `kelly_criterion` | `float` | Kelly criterion |
| `ulcer_index` | `float` | Ulcer Index |
| `tail_ratio` | `float` | Right tail / left tail ratio |
| `max_drawdown_duration` | `float` | Max drawdown duration in bars |
| `avg_drawdown_duration` | `float` | Avg drawdown duration in bars |

Post-run attributes: `engine.portfolio`, `engine.processed_data`, `engine.results`

### Portfolio

Manages cash, positions, orders, and risk. Accessed via `ctx.portfolio` in `next()`.

**Ordering methods:**

```python
# Basic order (positive=buy, negative=sell)
portfolio.order(asset, quantity, limit_price=None, stop_price=None, order_type=None, tags=None) -> str | None

# Target-based orders
portfolio.order_target(asset, target_quantity)                 # order to reach exact position size
portfolio.order_target_value(asset, target_value)              # order to reach $ value
portfolio.order_target_percent(asset, target_percent)          # order to reach % of portfolio (fee-aware)
portfolio.close_position(asset)                                # close entire position
portfolio.close_all_positions()                                # close all positions

# Order types
portfolio.order_day(asset, quantity, limit_price=None, bars_valid=None)   # expires end of day
portfolio.order_gtc(asset, quantity, limit_price=None)                     # good till cancelled

# Bracket orders (entry + SL + TP)
portfolio.order_bracket(asset, quantity, stop_loss=None, take_profit=None,
                        stop_loss_pct=None, take_profit_pct=None, limit_price=None)
    -> {"entry": str|None, "stop_loss": str|None, "take_profit": str|None}

# Sizer-based orders
portfolio.order_with_sizer(asset, sizer, direction, price=None, **kwargs) -> str | None
```

**Risk management:**

```python
portfolio.set_stop_loss(asset, stop_price=None, stop_pct=None) -> str | None
portfolio.set_take_profit(asset, target_price=None, target_pct=None) -> str | None
portfolio.set_trailing_stop(asset, trail_pct=None, trail_amount=None) -> str | None

portfolio.remove_stop_loss(asset) -> bool
portfolio.remove_take_profit(asset) -> bool
portfolio.remove_trailing_stop(asset) -> bool

portfolio.get_stop_loss(asset) -> float | None
portfolio.get_take_profit(asset) -> float | None
portfolio.get_trailing_stop(asset) -> float | None
```

**Querying state:**

```python
portfolio.get_position(asset) -> float          # current position (can be negative for short)
portfolio.get_value() -> float                   # total portfolio value (cash + positions)
portfolio.get_buying_power() -> float            # available buying power (with leverage)
portfolio.get_margin_used() -> float             # margin committed as collateral
portfolio.get_margin_available() -> float        # remaining margin
portfolio.get_margin_ratio() -> float | None     # equity / total position value
portfolio.trading_halted -> bool                 # whether risk limits are active

portfolio.get_orders(status=None, asset=None) -> list[Order]
portfolio.get_order(order_id) -> Order | None
portfolio.cancel_order(order_id) -> bool

portfolio.get_trades() -> pl.DataFrame           # completed trades
portfolio.get_trade_stats() -> TradeStats         # aggregate trade stats
```

**Order types** (`OrderType` enum): `MARKET`, `LIMIT`, `STOP`, `STOP_LIMIT`

Order type is auto-detected from parameters:
- No prices → `MARKET`
- `limit_price` only → `LIMIT`
- `stop_price` only → `STOP`
- Both → `STOP_LIMIT`

**Order statuses** (`OrderStatus` enum): `PENDING`, `FILLED`, `PARTIAL`, `CANCELLED`, `REJECTED`, `EXPIRED`

## Indicators

All indicators are Polars expressions used in `preprocess()`. Import as `from polarbt import indicators as ind`.

### Trend
| Function | Signature | Returns |
|---|---|---|
| `sma` | `(column, period)` | `Expr` |
| `ema` | `(column, period, adjust=False)` | `Expr` |
| `wma` | `(column, period)` | `Expr` — Weighted Moving Average |
| `hma` | `(column, period)` | `Expr` — Hull Moving Average |
| `vwap` | `(close, volume, high=None, low=None)` | `Expr` — Volume Weighted Average Price |
| `supertrend` | `(high, low, close, period=10, multiplier=3.0)` | `(line, direction)` — direction: 1.0=up, -1.0=down |
| `adx` | `(high, low, close, period=14)` | `(adx, plus_di, minus_di)` |
| `macd` | `(column, fast=12, slow=26, signal=9)` | `(macd_line, signal_line, histogram)` |

### Momentum
| Function | Signature | Returns |
|---|---|---|
| `rsi` | `(column, period=14)` | `Expr` |
| `stochastic` | `(high, low, close, k_period=14, d_period=3)` | `(%K, %D)` |
| `williams_r` | `(high, low, close, period=14)` | `Expr` — range -100 to 0 |
| `cci` | `(high, low, close, period=20)` | `Expr` — Commodity Channel Index |
| `mfi` | `(high, low, close, volume, period=14)` | `Expr` — Money Flow Index (0-100) |
| `roc` | `(column, period=12)` | `Expr` — Rate of Change (%) |

### Volatility
| Function | Signature | Returns |
|---|---|---|
| `atr` | `(high, low, close, period=14)` | `Expr` — Average True Range |
| `bollinger_bands` | `(column, period=20, std_dev=2.0)` | `(upper, middle, lower)` |
| `keltner_channels` | `(high, low, close, ema_period=20, atr_period=10, multiplier=2.0)` | `(upper, middle, lower)` |
| `donchian_channels` | `(high, low, period=20)` | `(upper, middle, lower)` |

### Volume
| Function | Signature | Returns |
|---|---|---|
| `obv` | `(close, volume)` | `Expr` — On-Balance Volume |
| `ad_line` | `(high, low, close, volume)` | `Expr` — Accumulation/Distribution |

### Signals
| Function | Signature | Returns |
|---|---|---|
| `crossover` | `(fast_column, slow_column)` | `Expr` (bool) — fast crosses above slow |
| `crossunder` | `(fast_column, slow_column)` | `Expr` (bool) — fast crosses below slow |

### Returns
| Function | Signature | Returns |
|---|---|---|
| `returns` | `(column, periods=1)` | `Expr` — percentage returns |
| `log_returns` | `(column, periods=1)` | `Expr` — logarithmic returns |

### Support/Resistance
| Function | Signature | Returns |
|---|---|---|
| `pivot_points` | `(high, low, close, method="standard")` | `dict[str, Expr]` — keys vary by method |

Methods: `"standard"` (pp, r1-r3, s1-s3), `"fibonacci"` (pp, r1-r3, s1-s3), `"woodie"` (pp, r1-r2, s1-s2), `"camarilla"` (pp, r1-r3, s1-s3)

### Usage pattern

Single-return indicators:
```python
df.with_columns(ind.sma("close", 20).alias("sma_20"))
```

Multi-return indicators:
```python
upper, middle, lower = ind.bollinger_bands("close", 20, 2.0)
df.with_columns(upper.alias("bb_upper"), middle.alias("bb_middle"), lower.alias("bb_lower"))
```

Crossover signals (must be in a second `.with_columns()` since they reference columns added in the first):
```python
df.with_columns(
    ind.sma("close", 10).alias("sma_fast"),
    ind.sma("close", 30).alias("sma_slow"),
).with_columns(
    ind.crossover("sma_fast", "sma_slow").alias("buy"),
)
```

## Position Sizers

Import: `from polarbt.sizers import ...`

All sizers extend `Sizer` and implement `size(portfolio, asset, price, **kwargs) -> float` (unsigned quantity).

| Sizer | Constructor | Notes |
|---|---|---|
| `FixedSizer` | `(quantity)` | Always returns fixed units |
| `PercentSizer` | `(percent)` | Fraction of portfolio value (e.g. 0.1=10%) |
| `FixedRiskSizer` | `(risk_percent)` | Requires `stop_distance` kwarg |
| `KellySizer` | `(win_rate, avg_win, avg_loss, max_fraction=0.25)` | Kelly criterion sizing |
| `VolatilitySizer` | `(target_risk_percent)` | Requires `atr` kwarg |
| `MaxPositionSizer` | `(sizer, max_quantity=None, max_percent=None)` | Wraps another sizer with caps |

Usage:
```python
sizer = FixedRiskSizer(risk_percent=0.02)
ctx.portfolio.order_with_sizer("asset", sizer, direction=1.0, stop_distance=atr_value)
```

## Commission Models

Import: `from polarbt.commissions import ...`

| Model | Constructor | Description |
|---|---|---|
| `PercentCommission` | `(rate)` | Percentage only (e.g. 0.001=0.1%) |
| `FixedPlusPercentCommission` | `(fixed, percent)` | Fixed fee + percentage |
| `MakerTakerCommission` | `(maker_rate, taker_rate, is_maker=False, fixed=0.0)` | Different rates for maker/taker |
| `TieredCommission` | `(tiers, fixed=0.0)` | Volume-based tiers: `[(0, 0.001), (100_000, 0.0008)]` |
| `CustomCommission` | `(func)` | Callable: `(size, price, is_reversal) -> float` |

Shorthand in Engine: `commission=0.001` (PercentCommission), `commission=(5.0, 0.001)` (FixedPlusPercent).

## Runner & Optimization

Import: `from polarbt import backtest, backtest_batch, optimize, optimize_multi, optimize_bayesian, walk_forward_analysis`

### Single backtest

```python
results = backtest(
    strategy_class=MyStrategy,    # class, not instance
    data=df,
    params={"fast": 10, "slow": 30},
    initial_cash=100_000,
    commission=0.001,
    slippage=0.0005,
)
```

Returns the same `BacktestMetrics` as `engine.run()`, with `params` and `success` fields populated.

### Batch execution

```python
results_df = backtest_batch(
    MyStrategy, data,
    param_sets=[{"fast": 10}, {"fast": 20}, {"fast": 50}],
    n_jobs=None,  # all CPUs (uses spawn context for Polars safety)
)
```

Returns `pl.DataFrame` with one row per param set, all metrics as columns.

### Grid search optimization

```python
best = optimize(
    MyStrategy, data,
    param_grid={"fast": [5, 10, 20], "slow": [20, 50, 100]},
    objective="sharpe_ratio",
    maximize=True,
    constraint=lambda p: p["fast"] < p["slow"],
)
```

### Multi-objective Pareto optimization

```python
pareto_df = optimize_multi(
    MyStrategy, data,
    param_grid={"period": [5, 10, 20, 50]},
    objectives=["sharpe_ratio", "max_drawdown"],
    maximize=[True, False],
)
```

### Bayesian optimization

Requires `scikit-optimize`. Uses continuous parameter spaces.

```python
best = optimize_bayesian(
    MyStrategy, data,
    param_space={"sma_period": (5, 50), "rsi_period": (7, 28)},
    n_calls=50,
    n_initial_points=10,
)
```

### Walk-forward analysis

```python
wf_results = walk_forward_analysis(
    MyStrategy, data,
    param_grid={"fast": [5, 10, 20], "slow": [20, 50]},
    train_periods=252,
    test_periods=63,
    anchored=False,   # True = expanding window
)
```

Returns `pl.DataFrame` with per-fold results: train/test metrics, best_params, fold indices.

## Advanced Analysis

Import: `from polarbt import monte_carlo, detect_look_ahead_bias, permutation_test`

### Monte Carlo simulation

```python
mc = monte_carlo(
    trades=engine.portfolio.trade_tracker.trades,
    initial_capital=100_000,
    n_simulations=1000,
    confidence_level=0.95,
    seed=42,
)
# mc.confidence_intervals["final_equity"]  -> (lower, upper)
# mc.confidence_intervals["max_drawdown"]  -> (lower, upper)
# mc.final_equities                        -> np.ndarray
# mc.simulated_equities                    -> np.ndarray (n_sims × n_trades+1)
```

### Look-ahead bias detection

```python
result = detect_look_ahead_bias(MyStrategy(), raw_df, sample_bars=5)
if result.biased_columns:
    print(f"WARNING: {result.biased_columns}")
# result.clean_columns, result.details
```

### Permutation testing

```python
perm = permutation_test(
    MyStrategy, df,
    metric="sharpe_ratio",
    n_permutations=100,
    seed=42,
)
# perm.p_value          -> statistical significance
# perm.original_metric  -> strategy's actual metric
# perm.null_distribution -> np.ndarray of shuffled results
```

## Data Utilities

Import: `from polarbt.data import ...`

### Validation

```python
from polarbt.data import validate, ValidationResult

result = validate(df, ohlcv=True)
# result.valid -> bool
# result.errors -> list[str]
# result.warnings -> list[str]
```

Individual checks: `validate_columns`, `validate_dtypes`, `validate_timestamps`, `validate_ohlc_integrity`, `validate_no_nulls`, `validate_no_negative_prices`

### Cleaning

```python
from polarbt.data import fill_gaps, adjust_splits, drop_zero_volume, clip_outliers

df = fill_gaps(df, interval="1h", method="forward")  # fill timestamp gaps
df = adjust_splits(df, splits=[("2024-06-01", 4.0)])  # adjust for stock splits
df = drop_zero_volume(df)                              # remove zero/null volume rows
df = clip_outliers(df, lower_quantile=0.001, upper_quantile=0.999)
```

### Resampling

```python
from polarbt.data import resample_ohlcv

daily = resample_ohlcv(df, interval="1d")       # OHLCV resampling
hourly = resample_ohlcv(df, interval="4h")
```

## Metrics

Import: `from polarbt.metrics import ...`

Standalone metric functions (take equity DataFrames with `equity` column):

| Function | Signature | Returns |
|---|---|---|
| `calculate_metrics` | `(equity_df, initial_capital)` | `dict` — comprehensive metrics |
| `sharpe_ratio` | `(equity_df, risk_free_rate=0.0)` | `float` |
| `sortino_ratio` | `(equity_df, risk_free_rate=0.0, target_return=0.0)` | `float` |
| `max_drawdown` | `(equity_df)` | `float` |
| `calmar_ratio` | `(equity_df, initial_capital)` | `float` |
| `omega_ratio` | `(equity_df, threshold=0.0)` | `float` |
| `rolling_sharpe` | `(equity_df, window=252)` | `pl.DataFrame` |
| `underwater_plot_data` | `(equity_df)` | `pl.DataFrame` with drawdown series |
| `value_at_risk` | `(equity_df, confidence=0.95)` | `float` |
| `conditional_value_at_risk` | `(equity_df, confidence=0.95)` | `float` |
| `ulcer_index` | `(equity_df, period=14)` | `float` |
| `tail_ratio` | `(equity_df, confidence=0.95)` | `float` |
| `information_ratio` | `(equity_df, benchmark_df)` | `float` |
| `alpha_beta` | `(equity_df, benchmark_df, risk_free_rate=0.0)` | `{"alpha": float, "beta": float}` |
| `drawdown_duration_stats` | `(equity_df)` | `dict` |
| `monthly_returns` | `(equity_df)` | `pl.DataFrame` with year, month, return |
| `trade_level_metrics` | `(trades)` | `dict` with expectancy, sqn, kelly |

## Plotting

Requires `pip install plotly`. Import: `from polarbt.plotting import ...`

| Function | Description |
|---|---|
| `plot_backtest(engine, indicators=None, bands=None, ...)` | Multi-panel chart: equity, P/L, price with candles/trades, volume |
| `plot_returns_distribution(engine)` | Histogram of trade returns |
| `plot_sensitivity(results_df, param, metric)` | Parameter sensitivity line chart |
| `plot_param_heatmap(results_df, x_param, y_param, metric)` | 2D parameter heatmap |
| `plot_permutation_test(perm_result)` | Null distribution with original metric line |

All return `plotly.graph_objects.Figure`. Save with `fig.write_html("chart.html")`.

```python
from polarbt.plotting import plot_backtest

fig = plot_backtest(
    engine,
    indicators=["sma_fast", "sma_slow"],
    bands=[("bb_upper", "bb_lower")],
    title="My Strategy",
)
fig.write_html("backtest.html")
```

## Trades

`Trade` dataclass attributes: `trade_id`, `asset`, `direction` ("long"/"short"), `entry_bar`, `entry_timestamp`, `entry_price`, `entry_size`, `entry_value`, `exit_bar`, `exit_timestamp`, `exit_price`, `exit_size`, `exit_value`, `pnl`, `pnl_pct`, `return_pct`, `bars_held`, `mae` (max adverse excursion, percentage), `mfe` (max favorable excursion, percentage), `bmfe` (best MFE before MAE), `trade_mdd` (max drawdown during trade, percentage), `pdays` (profitable days ratio).

Access trades:
```python
trades_df = engine.portfolio.get_trades()          # pl.DataFrame
trades_list = engine.portfolio.trade_tracker.trades # list[Trade]
stats = engine.portfolio.get_trade_stats()          # TradeStats
```

## TA-Lib Integration

Optional. Requires TA-Lib C library + Python wrapper.

```python
from polarbt.integrations.talib import talib_expr, talib_multi_expr, talib_series, TALibIndicators

# Single output (pass the TA-Lib function object, not a string)
import talib
df.with_columns(talib_expr(talib.RSI, "close", timeperiod=14).alias("ta_rsi"))

# Multiple outputs (returns dict[str, pl.Expr])
exprs = talib_multi_expr(talib.MACD, "close", fastperiod=12, slowperiod=26, signalperiod=9,
                         output_names=["macd", "signal", "hist"])
df.with_columns([v.alias(k) for k, v in exprs.items()])

# Series helper (operates on numpy, returns Series)
rsi_series = talib_series(talib.RSI, df["close"], timeperiod=14)

# Class-based (static methods, no DataFrame needed)
ta = TALibIndicators()
df.with_columns(ta.sma("close", 20).alias("ta_sma"))
df.with_columns(ta.rsi("close", 14).alias("ta_rsi"))
```

## Common Patterns

### Strategy with parameters for optimization

```python
class MyStrategy(Strategy):
    fast = param(10)
    slow = param(30)

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            ind.sma("close", self.fast).alias("sma_fast"),
            ind.sma("close", self.slow).alias("sma_slow"),
        )
```

### Short selling

```python
def next(self, ctx: BacktestContext) -> None:
    if ctx.row.get("sell_signal"):
        ctx.portfolio.order("asset", -100)  # short 100 units
    elif ctx.row.get("cover_signal"):
        ctx.portfolio.close_position("asset")
```

### Multi-asset rotation

```python
engine = Engine(strategy, {"SPY": spy_df, "TLT": tlt_df, "GLD": gld_df})

# In next():
ctx.portfolio.order_target_percent("SPY", 0.5)
ctx.portfolio.order_target_percent("TLT", 0.3)
ctx.portfolio.order_target_percent("GLD", 0.2)
```

### Bracket orders with ATR-based stops

```python
def next(self, ctx: BacktestContext) -> None:
    atr_val = ctx.row["atr_14"]
    price = ctx.row["close"]
    if ctx.row.get("buy"):
        ctx.portfolio.order_bracket(
            "asset", 100,
            stop_loss=price - 2 * atr_val,
            take_profit=price + 3 * atr_val,
        )
```

### Limit and stop orders

```python
# Buy limit below market
ctx.portfolio.order("asset", 100, limit_price=95.0)

# Sell stop (breakdown entry)
ctx.portfolio.order("asset", -100, stop_price=90.0)

# Stop-limit
ctx.portfolio.order("asset", 100, stop_price=105.0, limit_price=106.0)

# Day order (expires end of day)
ctx.portfolio.order_day("asset", 100, limit_price=95.0)
```

## Weight-Based Backtesting

For portfolio allocation strategies that work with target weights rather than event-driven order logic, use `backtest_weights()`. It accepts a long-format DataFrame with one row per (date, symbol) and a weight column.

Import: `from polarbt import backtest_weights, WeightBacktestResult`

### Basic usage

```python
import polars as pl
from polarbt import backtest_weights

# data has columns: date, symbol, close, weight (and optionally open, high, low, volume)
result = backtest_weights(data, resample="M", fee_ratio=0.001, initial_capital=100_000)

print(result.metrics)          # BacktestMetrics
print(result.equity.head())    # equity curve
print(result.trades.head())    # trade log
```

### Parameters

```python
backtest_weights(
    data: pl.DataFrame,
    date_col: str = "date",
    symbol_col: str = "symbol",
    price_col: str = "close",
    weight_col: str = "weight",
    open_col: str | None = "open",
    high_col: str | None = "high",
    low_col: str | None = "low",
    resample: str | None = "M",        # rebalance frequency
    resample_offset: str | None = None, # delay rebalance (e.g. "2d", "1W")
    fee_ratio: float = 0.001,
    tax_ratio: float = 0.0,
    stop_loss: float | None = None,     # per-position stop-loss (0.10 = 10%)
    take_profit: float | None = None,
    trail_stop: float | None = None,
    position_limit: float = 1.0,        # max absolute weight per symbol
    touched_exit: bool = False,         # OHLC intraday stop detection
    t_plus: int = 1,                    # execution delay (0 = same bar)
    initial_capital: float = 100_000.0,
    factor_col: str | None = None,      # price adjustment factor column
) -> WeightBacktestResult
```

| Parameter | Description |
|---|---|
| `resample` | `"D"` (daily), `"W"` / `"W-FRI"` (weekly), `"M"` (monthly), `"Q"` (quarterly), `"Y"` (yearly), or `None` (rebalance only when weights change) |
| `resample_offset` | Delay rebalance by N trading days after the period boundary. Format: `"<N>d"` or `"<N>W"` |
| `stop_loss` / `take_profit` / `trail_stop` | Per-position risk management. Expressed as fractions (e.g. 0.10 = 10%) |
| `touched_exit` | When `True` and OHLC data is available, uses open/high/low to detect intraday stop triggers with priority logic (open gap → high → low) |
| `position_limit` | Clips individual weights to `[-limit, +limit]`. Boolean weights (all 0/1) are auto-converted to equal-weight |
| `t_plus` | Execution delay. `0` = execute on signal bar, `1` = execute next bar (more realistic) |
| `factor_col` | Column with price adjustment factors. When present, commissions are calculated on raw (unadjusted) prices |

### Weight normalization

- **Boolean signals**: If all non-zero weights are `1.0`, they are converted to equal-weight (e.g. 3 symbols with weight 1.0 → 0.333 each)
- **Over-allocation**: If `sum(|weights|) > 1`, weights are scaled proportionally to sum to 1
- **Position limit**: Individual weights are clipped to `[-position_limit, position_limit]`

### Result object

`WeightBacktestResult` contains:

| Attribute | Type | Description |
|---|---|---|
| `equity` | `pl.DataFrame` | Columns: `date`, `cumulative_return` |
| `trades` | `pl.DataFrame` | Columns: `symbol`, `entry_date`, `exit_date`, `entry_price`, `exit_price`, `weight`, `return_pct`, `bars_held` |
| `metrics` | `BacktestMetrics` | Standard performance metrics (same as `Engine.run()`) |
| `next_actions` | `pl.DataFrame \| None` | Forward-looking actions: `symbol`, `action` (enter/exit/hold), `current_weight`, `target_weight` |

### Next actions

The `next_actions` attribute shows what trades would be needed to transition from the current portfolio state to the latest target weights. This is useful for live trading integration.

```python
if result.next_actions is not None:
    print(result.next_actions)
    # ┌────────┬────────┬────────────────┬───────────────┐
    # │ symbol ┆ action ┆ current_weight ┆ target_weight │
    # ╞════════╪════════╪════════════════╪═══════════════╡
    # │ AAPL   ┆ hold   ┆ 0.25           ┆ 0.25          │
    # │ GOOGL  ┆ exit   ┆ 0.25           ┆ 0.0           │
    # │ MSFT   ┆ enter  ┆ 0.0            ┆ 0.5           │
    # └────────┴────────┴────────────────┴───────────────┘
```

A standalone version is also available:

```python
from polarbt import compute_next_actions

actions = compute_next_actions(
    current_positions={"AAPL": 100, "GOOGL": 50},
    target_weights={"AAPL": 0.5, "MSFT": 0.5},
    portfolio_value=100_000,
    current_prices={"AAPL": 150.0, "GOOGL": 120.0, "MSFT": 300.0},
)
# Returns DataFrame with: symbol, action, current_value, target_value, delta_shares
```

### Example

```python
from datetime import date, timedelta
import numpy as np
import polars as pl
from polarbt import backtest_weights

# Build long-format data with momentum-based weights
rng = np.random.default_rng(42)
symbols = ["AAPL", "GOOGL", "MSFT"]
rows = []
prices = {s: 100.0 for s in symbols}

for i in range(252):
    d = date(2024, 1, 2) + timedelta(days=i)
    for sym in symbols:
        prices[sym] *= np.exp(rng.normal(0.0003, 0.015))
        rows.append({"date": d, "symbol": sym, "close": prices[sym], "weight": 0.0})

data = pl.DataFrame(rows)

# Assign equal-weight to top-2 by momentum
data = data.with_columns(
    pl.col("close").pct_change().rolling_sum(window_size=20).over("symbol").alias("mom")
)
data = data.with_columns(pl.col("mom").rank(descending=True).over("date").alias("rank"))
data = data.with_columns(pl.when(pl.col("rank") <= 2).then(0.5).otherwise(0.0).alias("weight"))

result = backtest_weights(
    data,
    resample="M",
    fee_ratio=0.001,
    stop_loss=0.10,
    initial_capital=100_000,
)
print(result)
```

## DataFrame Column Conventions

Engine expects lowercase OHLCV columns: `open`, `high`, `low`, `close`, `volume`, `timestamp`. Common variants are auto-detected (e.g. `Open`, `CLOSE`, `Date`, `Adj Close`).

Use `standardize_dataframe(df)` explicitly if needed. For multi-asset, `merge_asset_dataframes({"BTC": df1, "ETH": df2})` creates prefixed columns (`BTC_close`, `ETH_close`, etc.).
