# Changelog

## [Unreleased]

### Added
- Unified long-format data model: Engine accepts single DF, dict of DFs, or long-format DF with `symbol` column
- `BacktestContext.symbols`, `BacktestContext.data`, and `BacktestContext.row()` method for multi-asset access
- `Portfolio.rebalance(weights)` for atomic weight-based portfolio rebalancing
- `WeightStrategy` base class for weight-driven strategies via `get_weights()`
- 28 new tests for long-format engine, rebalance, and WeightStrategy

### Fixed
- Multi-asset OHLC data preserved (previously `merge_asset_dataframes()` dropped open/high/low)

### Changed
- Engine internally normalizes all input to long-format; strategies receive long-format in `preprocess()`
- Multi-asset indicators now use standard column names with `.over("symbol")` instead of `BTC_close` convention

## [0.1.11] - 2026-03-09

### Fixed
- `Engine` now eagerly clears previous portfolio on re-run, preventing memory accumulation across sequential backtests
- `Engine.__del__` added as safety net for callers that skip explicit `cleanup()`
- `prices` possibly unbound in `weight_backtest.py` when data is empty
- Win rate formatting in examples (was showing `0.6%` instead of `60.0%`)

### Changed
- `_calculate_auto_warmup` uses vectorized `arg_max()` instead of row-by-row iteration
- `standardize_dataframe` no longer clones input (Polars DataFrames are immutable)

### Removed
- Dict-style access on `OptimizeResult` — use `.params` and `.metrics` attributes instead

## [0.1.10] - 2026-03-08

### Changed
- Permutation test ~7.7x faster on large datasets via active order tracking and precomputed arrays

## [0.1.9] - 2026-03-08

### Fixed
- `permutation_test()` OOM crash: workers now shuffle data independently instead of pre-generating all permutations, and `n_jobs` capped to 4

## [0.1.8] - 2026-03-08

### Added
- `OptimizeResult` return type for `optimize()` — separates `.params` from `.metrics`, prevents name collisions
- `n_jobs` parameter for `optimize_bayesian()` — parallel batch evaluation via `ProcessPoolExecutor`

### Fixed
- `optimize_bayesian()` constraint handling no longer pollutes GP surrogate model with penalty values
- `optimize()` now filters out failed backtests before selecting best result
- `optimize_bayesian()` no longer redundantly re-runs the best parameter set

## [0.1.7] - 2026-03-08

### Fixed
- Supertrend indicator always returning direction 1.0 due to NaN band propagation

## [0.1.6] - 2026-03-08

### Fixed
- Order dict corruption causing segfaults during sequential backtests
- Concurrent stop-loss mutation causing inconsistent exit prices

## [0.1.5] - 2026-03-08

### Added
- Weight-based backtesting (`backtest_weights()`)
- OHLC priority exits for stop-loss/take-profit (open > high > low)
- Liquidity metrics: buy-high ratio, sell-low ratio, capacity
- Factor-based pricing for adjusted data (splits, dividends)
- Avg. Trade MDD metric and pretty-print output

### Changed
- `format_results()` moved into `BacktestMetrics.__str__()`

## [0.1.4] - 2026-03-04

### Added
- `param()` descriptor for declarative strategy parameters

## [0.1.3] - 2026-03-04

### Added
- `BacktestMetrics` and `TradeStats` typed dataclasses replacing untyped result dicts
- Inline speed tracker for `optimize()` and `backtest_batch()` progress
- `constraint` parameter for `walk_forward_analysis()`

### Fixed
- Trade-level percentages normalized to 0–1 float format
- Documentation inconsistencies with actual code

## [0.1.2] - 2026-03-04

### Fixed
- Example code corrections

## [0.1.1] - 2026-03-04

### Fixed
- PyPI publish workflow

## [0.1.0] - 2026-03-04

Initial public release.

### Added
- Core backtesting engine with event-driven loop and vectorized preprocessing (Polars)
- Multi-asset support with fractional shares
- Auto-warmup detection for indicators
- Order delay simulation
- Trade tracking with MAE/MFE
- 30+ technical indicators (SMA, EMA, RSI, MACD, Bollinger, Supertrend, etc.)
- Optional TA-Lib integration
- 6 position sizers (Fixed, Percent, FixedRisk, Kelly, Volatility, MaxPosition)
- Pluggable commission models (Percent, FixedPlusPercent, MakerTaker, Tiered, Custom)
- Risk limits: max position size, max exposure, drawdown stop, daily loss limit
- Margin and leverage support with margin calls
- Grid search, multi-objective Pareto, and Bayesian optimization
- Walk-forward analysis
- Parallel batch execution with spawn context (Polars-safe)
- Monte Carlo simulation, look-ahead bias detection, permutation testing
- Interactive plotting with Plotly (equity curve, drawdown, candlestick, sensitivity, heatmap)
- Data utilities: validation, cleaning, OHLCV resampling, column standardization
- Enhanced metrics: Sharpe, Sortino, Calmar, ulcer index, tail ratio, SQN, Kelly criterion
- Pretty-printed backtest results
- OHLCV column name auto-detection and standardization
- PyPI publish workflow
