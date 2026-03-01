# PolarBtest — Development Plan

Remaining work organized by priority. Each item is a self-contained task.

---

## 1. Core Gaps (fix what's half-done) ✓ COMPLETE

- [x] Wire STOP order execution in `_try_execute_order()` — trigger when price crosses stop_price, then execute as market
- [x] Wire STOP_LIMIT order execution — trigger at stop_price, then place limit order at limit_price
- [x] Implement automatic SL/TP placement when a pending bracket entry order fills (bracket metadata stored on Order, applied on fill)
- [x] Add short selling support to Portfolio — negative positions with proper cash handling, position reversals
- [x] Add borrow costs for short positions (daily rate deducted from cash, configurable via `borrow_rate` parameter)

## 2. Visualization ✓ COMPLETE

- [x] Create `polarbtest/plotting/` module with plotly as optional dependency
- [x] Implement `plot_backtest()` — price chart + equity curve + drawdown subplot
- [x] Add trade entry/exit markers (arrows) on price chart
- [x] Add indicator overlay helpers (lines, bands)
- [x] Add volume bars subplot
- [x] Support saving charts to HTML
- [x] Add returns distribution histogram

## 3. Enhanced Metrics ✓ COMPLETE

- [x] Add Ulcer Index to `metrics.py`
- [x] Add Tail Ratio
- [x] Add Information Ratio (requires benchmark input)
- [x] Add Alpha/Beta vs benchmark
- [x] Add drawdown duration statistics (max, avg, count)
- [x] Add monthly returns table
- [x] Add trade-level metrics — expectancy, SQN, Kelly criterion, consecutive wins/losses
- [x] Integrate ulcer_index, tail_ratio, drawdown duration stats into `calculate_metrics()` output

## 4. Position Sizing ✓ COMPLETE

- [x] Create `polarbtest/sizers.py` with `Sizer` base class
- [x] Implement `FixedSizer` — fixed unit count
- [x] Implement `PercentSizer` — percentage of portfolio
- [x] Implement `FixedRiskSizer` — risk X% per trade based on stop distance
- [x] Implement `KellySizer` — Kelly criterion sizing
- [x] Implement `VolatilitySizer` — ATR-based sizing for constant volatility
- [x] Implement `MaxPositionSizer` — wrapper that caps position size
- [x] Add `order_with_sizer()` method to Portfolio

## 5. Risk Limits ✓ COMPLETE

- [x] Add `max_position_size` parameter to Portfolio (cap single position as % of portfolio)
- [x] Add `max_total_exposure` parameter (cap total exposure as % of portfolio)
- [x] Add `max_drawdown_stop` parameter (halt trading at X% drawdown)
- [x] Add `daily_loss_limit` parameter (halt trading at X% daily loss)
- [x] Enforce limits in order execution path

## 6. Commission Models ✓ COMPLETE

- [x] Create `CommissionModel` base class
- [x] Implement `MakerTakerCommission` — different rates for maker/taker
- [x] Implement `TieredCommission` — volume-based tiers
- [x] Implement `CustomCommission` — user-provided callable
- [x] Update Portfolio to accept `CommissionModel` instances alongside current float/tuple

## 7. Margin & Leverage ✓ COMPLETE

- [x] Add `leverage` parameter to Portfolio
- [x] Implement `get_buying_power()` with leverage
- [x] Add margin requirement tracking (`get_margin_used()`, `get_margin_available()`, `get_margin_ratio()`)
- [x] Implement margin call handling (auto-close positions when margin ratio < maintenance_margin)

## 8. Additional Indicators ✓ COMPLETE

### Trend
- [x] WMA (Weighted Moving Average)
- [x] HMA (Hull Moving Average)
- [x] VWAP
- [x] SuperTrend
- [x] ADX (Average Directional Index)

### Momentum
- [x] Stochastic Oscillator
- [x] Williams %R
- [x] CCI (Commodity Channel Index)
- [x] MFI (Money Flow Index)
- [x] ROC (Rate of Change)

### Volatility
- [x] Keltner Channels
- [x] Donchian Channels

### Volume
- [x] OBV (On-Balance Volume)
- [x] A/D Line (Accumulation/Distribution)

### Support/Resistance
- [x] Pivot Points (Standard, Fibonacci, Woodie, Camarilla)

### TA-Lib Integration
- [x] Create optional TA-Lib wrapper in `polarbtest/integrations/talib.py` (optional dependency)
- [x] Add convenience functions that call TA-Lib and return Polars expressions/Series
- [x] Fall back gracefully when TA-Lib is not installed

## 9. Data Utilities ✓ COMPLETE

- [x] Create `polarbtest/data/` module
- [x] Add data validation functions (check required columns, dtypes, sorted timestamps)
- [x] Add data cleaning functions (fill gaps, handle splits)
- [x] Implement OHLCV resampling (e.g., 1-min to 1-hour)

## 10. Optimization Enhancements ✓ COMPLETE

- [x] Add constraint functions to `optimize()` (e.g., reject params where fast >= slow)
- [x] Add multi-objective optimization support
- [x] Add Bayesian optimization (optional, requires scikit-optimize)
- [x] Create parameter sensitivity plots
- [x] Create 2D parameter heatmaps

## 11. Advanced Analysis ✓ COMPLETE

- [x] Implement Monte Carlo simulation on trade results
- [x] Implement look-ahead bias detection (scan for future data leaks in preprocess)
- [x] Implement permutation test by shuffling market data

## 12. Documentation & Examples ✓ COMPLETE

- [x] Write getting started guide with end-to-end example
- [x] Add example: SMA crossover with stop-loss
- [x] Add example: RSI mean reversion with bracket orders
- [x] Add example: Multi-asset momentum rotation
- [x] Add example: ML-integrated strategy
- [x] Add example: Walk-forward analysis workflow
- [x] Verify that examples cover all available functionality, fill the gaps if needed
- [x] Write API reference (can be auto-generated from docstrings)
- [x] Write a short readme, add cover image from `assets/cover.png`
- [x] Add MIT license

---

## Priority Order

1. **Core Gaps** — STOP/STOP_LIMIT execution, short selling ✓
2. **Visualization** — essential for strategy development ✓
3. **Enhanced Metrics** — quick wins ✓
4. **Documentation & Examples** — usability
5. **Position Sizing** — important for realistic backtesting
6. **Risk Limits** — important for realistic backtesting
7. **Commission Models** — nice to have
8. **Additional Indicators** — nice to have
9. **Data Utilities** — nice to have
10. **Optimization Enhancements** — nice to have
11. **Margin & Leverage** — advanced
12. **Advanced Analysis** — advanced
