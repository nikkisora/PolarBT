# PolarBtest — Development Plan

Remaining work organized by priority. Each item is a self-contained task.

---

## 1. Core Gaps (fix what's half-done) ✓ COMPLETE

- [x] Wire STOP order execution in `_try_execute_order()` — trigger when price crosses stop_price, then execute as market
- [x] Wire STOP_LIMIT order execution — trigger at stop_price, then place limit order at limit_price
- [x] Implement automatic SL/TP placement when a pending bracket entry order fills (bracket metadata stored on Order, applied on fill)
- [x] Add short selling support to Portfolio — negative positions with proper cash handling, position reversals
- [x] Add borrow costs for short positions (daily rate deducted from cash, configurable via `borrow_rate` parameter)

## 2. Visualization

- [ ] Create `polarbtest/plotting/` module with plotly as optional dependency
- [ ] Implement `plot_backtest()` — price chart + equity curve + drawdown subplot
- [ ] Add trade entry/exit markers (arrows) on price chart
- [ ] Add indicator overlay helpers (lines, bands)
- [ ] Add volume bars subplot
- [ ] Support saving charts to HTML
- [ ] Add returns distribution histogram

## 3. Enhanced Metrics

- [ ] Add Ulcer Index to `metrics.py`
- [ ] Add Tail Ratio
- [ ] Add Information Ratio (requires benchmark input)
- [ ] Add Alpha/Beta vs benchmark
- [ ] Add drawdown duration statistics (max, avg, count)
- [ ] Add monthly returns table
- [ ] Add trade-level metrics to `calculate_metrics()` output — expectancy, SQN, Kelly criterion, consecutive wins/losses

## 4. Position Sizing

- [ ] Create `polarbtest/sizers.py` with `Sizer` base class
- [ ] Implement `FixedSizer` — fixed unit count
- [ ] Implement `PercentSizer` — percentage of portfolio
- [ ] Implement `FixedRiskSizer` — risk X% per trade based on stop distance
- [ ] Implement `KellySizer` — Kelly criterion sizing
- [ ] Implement `VolatilitySizer` — ATR-based sizing for constant volatility
- [ ] Implement `MaxPositionSizer` — wrapper that caps position size
- [ ] Add `order_with_sizer()` method to Portfolio

## 5. Risk Limits

- [ ] Add `max_position_size` parameter to Portfolio (cap single position as % of portfolio)
- [ ] Add `max_total_exposure` parameter (cap total exposure as % of portfolio)
- [ ] Add `max_drawdown_stop` parameter (halt trading at X% drawdown)
- [ ] Add `daily_loss_limit` parameter (halt trading at X% daily loss)
- [ ] Enforce limits in order execution path

## 6. Commission Models

- [ ] Create `CommissionModel` base class
- [ ] Implement `MakerTakerCommission` — different rates for maker/taker
- [ ] Implement `TieredCommission` — volume-based tiers
- [ ] Implement `CustomCommission` — user-provided callable
- [ ] Update Portfolio to accept `CommissionModel` instances alongside current float/tuple

## 7. Margin & Leverage

- [ ] Add `leverage` parameter to Portfolio
- [ ] Implement `get_buying_power()` with leverage
- [ ] Add margin requirement tracking
- [ ] Implement margin call handling (auto-close positions)

## 8. Additional Indicators

### Trend
- [ ] WMA (Weighted Moving Average)
- [ ] HMA (Hull Moving Average)
- [ ] VWAP
- [ ] SuperTrend
- [ ] ADX (Average Directional Index)

### Momentum
- [ ] Stochastic Oscillator
- [ ] Williams %R
- [ ] CCI (Commodity Channel Index)
- [ ] MFI (Money Flow Index)
- [ ] ROC (Rate of Change)

### Volatility
- [ ] Keltner Channels
- [ ] Donchian Channels

### Volume
- [ ] OBV (On-Balance Volume)
- [ ] A/D Line (Accumulation/Distribution)

### Support/Resistance
- [ ] Pivot Points (Standard, Fibonacci, Woodie, Camarilla)

### TA-Lib Integration
- [ ] Create optional TA-Lib wrapper in `polarbtest/integrations/talib.py` (optional dependency)
- [ ] Add convenience functions that call TA-Lib and return Polars expressions/Series
- [ ] Fall back gracefully when TA-Lib is not installed

## 9. Data Utilities

- [ ] Create `polarbtest/data/` module
- [ ] Add data validation functions (check required columns, dtypes, sorted timestamps)
- [ ] Add data cleaning functions (fill gaps, handle splits)
- [ ] Implement OHLCV resampling (e.g., 1-min to 1-hour)

## 10. Optimization Enhancements

- [ ] Add constraint functions to `optimize()` (e.g., reject params where fast >= slow)
- [ ] Add multi-objective optimization support
- [ ] Add Bayesian optimization (optional, requires scikit-optimize)
- [ ] Create parameter sensitivity plots
- [ ] Create 2D parameter heatmaps

## 11. Advanced Analysis

- [ ] Implement Monte Carlo simulation on trade results
- [ ] Implement look-ahead bias detection (scan for future data leaks in preprocess)

## 12. Documentation & Examples

- [ ] Write getting started guide with end-to-end example
- [ ] Add example: SMA crossover with stop-loss
- [ ] Add example: RSI mean reversion with bracket orders
- [ ] Add example: Multi-asset momentum rotation
- [ ] Add example: ML-integrated strategy
- [ ] Add example: Walk-forward analysis workflow
- [ ] Write API reference (can be auto-generated from docstrings)

---

## Priority Order

1. **Core Gaps** — STOP/STOP_LIMIT execution, short selling
2. **Visualization** — essential for strategy development
3. **Documentation & Examples** — usability
4. **Enhanced Metrics** — quick wins
5. **Position Sizing** — important for realistic backtesting
6. **Risk Limits** — important for realistic backtesting
7. **Commission Models** — nice to have
8. **Additional Indicators** — nice to have
9. **Data Utilities** — nice to have
10. **Optimization Enhancements** — nice to have
11. **Margin & Leverage** — advanced
12. **Advanced Analysis** — advanced
