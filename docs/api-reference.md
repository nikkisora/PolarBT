# API Reference

Auto-generated summary of PolarBT's public API. For detailed docstrings, see the source code.

## Core (`polarbt.core`)

### `Strategy`

Base class for all strategies.

| Method | Description |
|---|---|
| `__init__(**params)` | Initialize with strategy parameters (accessible via `self.params`) |
| `preprocess(df) -> pl.DataFrame` | **Required.** Add indicator columns using vectorized Polars ops. Called once. |
| `next(ctx: BacktestContext)` | **Required.** Called every bar after warmup. Place orders via `ctx.portfolio`. |
| `on_start(portfolio)` | Optional. Called before first bar. |
| `on_finish(portfolio)` | Optional. Called after last bar. |

### `BacktestContext`

Passed to `strategy.next()` each bar.

| Attribute | Type | Description |
|---|---|---|
| `row` | `dict[str, Any]` | All column values for the current bar |
| `portfolio` | `Portfolio` | Portfolio instance for placing orders |
| `bar_index` | `int` | Current bar index (0-based) |
| `timestamp` | `Any` | Current timestamp (if available) |

### `Portfolio`

Manages cash, positions, orders, and risk management.

**Order Methods:**

| Method | Description |
|---|---|
| `order(asset, quantity, limit_price=None, ...)` | Place market or limit order |
| `order_target(asset, target_quantity)` | Order to reach target position |
| `order_target_percent(asset, percent)` | Target % of portfolio value |
| `order_target_value(asset, value)` | Target dollar value |
| `close_position(asset)` | Close position for asset |
| `close_all_positions()` | Close all positions |
| `order_day(asset, quantity, ...)` | Day order with auto-expiry |
| `order_gtc(asset, quantity, ...)` | Good-till-cancelled order |
| `order_bracket(asset, quantity, stop_loss, take_profit, ...)` | Entry + SL + TP (OCO) |
| `order_with_sizer(asset, sizer, direction, ...)` | Compute quantity via Sizer, then place order |

**Risk Management:**

| Method | Description |
|---|---|
| `set_stop_loss(asset, stop_price=None, stop_pct=None)` | Set stop-loss |
| `set_take_profit(asset, target_price=None, target_pct=None)` | Set take-profit |
| `set_trailing_stop(asset, trail_pct=None, trail_amount=None)` | Set trailing stop |
| `remove_stop_loss(asset)` | Remove stop-loss |
| `remove_take_profit(asset)` | Remove take-profit |
| `remove_trailing_stop(asset)` | Remove trailing stop |

**Position & Value:**

| Method | Description |
|---|---|
| `get_position(asset) -> float` | Current position quantity |
| `get_value() -> float` | Total portfolio value |
| `get_trades() -> pl.DataFrame` | Completed trades |
| `get_trade_stats() -> dict` | Aggregate trade statistics |
| `get_orders(status=None, asset=None)` | Filter orders |
| `get_buying_power() -> float` | Available buying power (with leverage) |
| `get_margin_used() -> float` | Current margin used |
| `get_margin_available() -> float` | Available margin |
| `get_margin_ratio() -> float` | Current margin ratio |

### `Engine`

```python
Engine(
    strategy,
    data,
    initial_cash=100_000,
    commission=0.001,
    slippage=0.0005,
    price_columns=None,
    warmup="auto",
    order_delay=0,
    borrow_rate=0.0,
    bars_per_day=None,
    max_position_size=None,
    max_total_exposure=None,
    max_drawdown_stop=None,
    daily_loss_limit=None,
    leverage=1.0,
    maintenance_margin=None,
)
```

| Method | Description |
|---|---|
| `run() -> dict` | Run backtest, return results dict |

## Orders (`polarbt.orders`)

### `OrderType` (enum)

`MARKET`, `LIMIT`, `STOP`, `STOP_LIMIT`

### `OrderStatus` (enum)

`PENDING`, `FILLED`, `PARTIAL`, `CANCELLED`, `REJECTED`, `EXPIRED`

### `Order`

Dataclass representing an order with fields: `order_id`, `asset`, `quantity`, `order_type`, `limit_price`, `stop_price`, `status`, `filled_price`, `filled_bar`, `tags`, etc.

## Trades (`polarbt.trades`)

### `Trade`

Dataclass with fields: `trade_id`, `asset`, `direction`, `entry_price`, `exit_price`, `size`, `pnl`, `pnl_pct`, `entry_bar`, `exit_bar`, `bars_held`, `commission`, `mae`, `mfe`.

### `TradeTracker`

Automatic trade lifecycle tracking. Exports to DataFrame via `to_dataframe()`.

## Indicators (`polarbt.indicators`)

All return `pl.Expr` for use in `preprocess()`.

### Trend

| Function | Signature |
|---|---|
| `sma` | `sma(column, period) -> Expr` |
| `ema` | `ema(column, period) -> Expr` |
| `wma` | `wma(column, period) -> Expr` |
| `hma` | `hma(column, period) -> Expr` |
| `vwap` | `vwap(close, volume, high, low) -> Expr` |
| `supertrend` | `supertrend(high, low, close, period, multiplier) -> tuple[Expr, Expr]` |
| `adx` | `adx(high, low, close, period) -> tuple[Expr, Expr, Expr]` |

### Momentum

| Function | Signature |
|---|---|
| `rsi` | `rsi(column, period) -> Expr` |
| `macd` | `macd(column, fast, slow, signal) -> tuple[Expr, Expr, Expr]` |
| `stochastic` | `stochastic(high, low, close, k, d) -> tuple[Expr, Expr]` |
| `williams_r` | `williams_r(high, low, close, period) -> Expr` |
| `cci` | `cci(high, low, close, period) -> Expr` |
| `mfi` | `mfi(high, low, close, volume, period) -> Expr` |
| `roc` | `roc(column, period) -> Expr` |
| `returns` | `returns(column, periods) -> Expr` |
| `log_returns` | `log_returns(column, periods) -> Expr` |

### Volatility

| Function | Signature |
|---|---|
| `bollinger_bands` | `bollinger_bands(column, period, std_dev) -> tuple[Expr, Expr, Expr]` |
| `atr` | `atr(high, low, close, period) -> Expr` |
| `keltner_channels` | `keltner_channels(high, low, close, ema, atr, mult) -> tuple[Expr, Expr, Expr]` |
| `donchian_channels` | `donchian_channels(high, low, period) -> tuple[Expr, Expr, Expr]` |

### Volume

| Function | Signature |
|---|---|
| `obv` | `obv(close, volume) -> Expr` |
| `ad_line` | `ad_line(high, low, close, volume) -> Expr` |

### Signals

| Function | Signature |
|---|---|
| `crossover` | `crossover(fast_col, slow_col) -> Expr` |
| `crossunder` | `crossunder(fast_col, slow_col) -> Expr` |

### Support/Resistance

| Function | Signature |
|---|---|
| `pivot_points` | `pivot_points(high, low, close, method) -> dict[str, Expr]` |

## Metrics (`polarbt.metrics`)

### `calculate_metrics(equity_df, initial_capital) -> dict`

Returns: `total_return`, `cagr`, `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `max_drawdown`, `volatility`, `volatility_annualized`, `ulcer_index`, `tail_ratio`, `max_drawdown_duration`, `avg_drawdown_duration`, `drawdown_count`, `profit_factor`, `initial_equity`, `final_equity`.

### Standalone Functions

| Function | Description |
|---|---|
| `sharpe_ratio(returns, risk_free)` | Sharpe ratio |
| `sortino_ratio(returns, risk_free)` | Sortino ratio |
| `max_drawdown(equity)` | Maximum drawdown |
| `calmar_ratio(returns, equity)` | Calmar ratio |
| `omega_ratio(returns, threshold)` | Omega ratio |
| `value_at_risk(returns, confidence)` | VaR |
| `conditional_value_at_risk(returns, confidence)` | CVaR |
| `rolling_sharpe(returns, window)` | Rolling Sharpe ratio |
| `underwater_plot_data(equity)` | Drawdown series |
| `ulcer_index(equity)` | Ulcer Index |
| `tail_ratio(returns)` | Tail ratio |
| `information_ratio(returns, benchmark_returns)` | Information ratio |
| `alpha_beta(returns, benchmark_returns)` | Alpha and beta |
| `drawdown_duration_stats(equity)` | Max/avg duration, count |
| `monthly_returns(equity, timestamps)` | Monthly returns table |
| `trade_level_metrics(trades_df)` | Expectancy, SQN, Kelly, streaks |

## Commission Models (`polarbt.commissions`)

| Class | Constructor |
|---|---|
| `PercentCommission` | `PercentCommission(rate)` |
| `FixedPlusPercentCommission` | `FixedPlusPercentCommission(fixed, percent)` |
| `MakerTakerCommission` | `MakerTakerCommission(maker_rate, taker_rate, is_maker, fixed)` |
| `TieredCommission` | `TieredCommission(tiers, fixed)` |
| `CustomCommission` | `CustomCommission(func)` |

## Position Sizing (`polarbt.sizers`)

| Class | Constructor |
|---|---|
| `FixedSizer` | `FixedSizer(quantity)` |
| `PercentSizer` | `PercentSizer(percent)` |
| `FixedRiskSizer` | `FixedRiskSizer(risk_percent)` |
| `KellySizer` | `KellySizer(win_rate, avg_win, avg_loss, max_fraction)` |
| `VolatilitySizer` | `VolatilitySizer(target_risk_percent)` |
| `MaxPositionSizer` | `MaxPositionSizer(sizer, max_quantity, max_percent)` |

## Runner (`polarbt.runner`)

| Function | Description |
|---|---|
| `backtest(strategy_class, data, params, ...)` | Run single backtest |
| `backtest_batch(strategy_class, data, param_sets, n_jobs, ...)` | Parallel batch execution |
| `optimize(strategy_class, data, param_grid, objective, constraint, ...)` | Grid search optimization |
| `optimize_multi(strategy_class, data, param_grid, objectives, ...)` | Multi-objective Pareto optimization |
| `optimize_bayesian(strategy_class, data, param_space, objective, n_calls, ...)` | Bayesian optimization (requires scikit-optimize) |
| `walk_forward_analysis(strategy_class, data, param_grid, train_periods, test_periods, ...)` | Walk-forward optimization |

## Visualization (`polarbt.plotting`)

Requires `plotly`: `pip install polarbt[plotting]`

| Function | Description |
|---|---|
| `plot_backtest(engine, indicators, bands, ...)` | Multi-panel backtest chart |
| `plot_returns_distribution(engine, ...)` | Returns histogram |
| `plot_sensitivity(results_df, param, metric, ...)` | Parameter sensitivity plot |
| `plot_param_heatmap(results_df, param_x, param_y, metric, ...)` | 2D parameter heatmap |
| `plot_permutation_test(perm_result, ...)` | Permutation test histogram |

All return `plotly.graph_objects.Figure`. Use `save_html="file.html"` to export.

## Data Utilities (`polarbt.data`)

### Validation

| Function | Description |
|---|---|
| `validate(df, ohlcv)` | Run all validation checks |
| `validate_columns(df, required, ohlcv)` | Check required columns |
| `validate_dtypes(df)` | Check numeric dtypes |
| `validate_timestamps(df, column)` | Check sorted, no duplicates |
| `validate_ohlc_integrity(df)` | Check OHLC relationships |
| `validate_no_nulls(df, columns)` | Check for nulls |
| `validate_no_negative_prices(df)` | Check non-negative prices |

### Cleaning

| Function | Description |
|---|---|
| `fill_gaps(df, interval, method)` | Fill timestamp gaps |
| `adjust_splits(df, splits)` | Adjust for stock splits |
| `drop_zero_volume(df)` | Remove zero-volume rows |
| `clip_outliers(df, columns, lower_quantile, upper_quantile)` | Clip extreme values |

### Resampling

| Function | Description |
|---|---|
| `resample_ohlcv(df, interval)` | Resample OHLCV to larger timeframe |

## Advanced Analysis (`polarbt.analysis`)

| Function | Description |
|---|---|
| `monte_carlo(trades, initial_capital, n_simulations, confidence_level, seed)` | Monte Carlo simulation on trade P&Ls |
| `detect_look_ahead_bias(strategy, data, sample_bars, tolerance)` | Detect future data leaks in preprocess() |
| `permutation_test(strategy_class, data, metric, n_permutations, seed, ...)` | Statistical significance test |

## TA-Lib Integration (`polarbt.integrations.talib`)

Optional. Requires TA-Lib: `pip install polarbt[talib]`

| Function | Description |
|---|---|
| `ta.sma(column, period)` | TA-Lib SMA |
| `ta.ema(column, period)` | TA-Lib EMA |
| `ta.rsi(column, period)` | TA-Lib RSI |
| `ta.macd(column, fast, slow, signal)` | TA-Lib MACD |
| `ta.bollinger_bands(column, period)` | TA-Lib Bollinger Bands |
| `ta.atr(high, low, close, period)` | TA-Lib ATR |
| `talib_expr(func, columns, ...)` | Wrap any single-output TA-Lib function |
| `talib_multi_expr(func, columns, ..., output_names)` | Wrap multi-output functions |
| `talib_series(func, *series, ...)` | Apply to Polars Series |
| `talib_available()` | Check if TA-Lib is installed |
