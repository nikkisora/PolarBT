# PolarBtest TODO

> Simplified task list distilled from ENHANCEMENT_ROADMAP.md

**Last Updated**: 2026-02-03

---

## Phase 1: Core Trading Functionality 🔴 CRITICAL

### Order System ✅ COMPLETE (Phase 1.5)
- [x] Create `Order` dataclass with order types (MARKET, LIMIT, STOP, STOP_LIMIT)
- [x] Implement limit orders
- [x] Implement stop-loss orders
- [x] Implement take-profit orders ✅ DONE (2026-02-03)
- [x] Implement trailing stop-loss ✅ DONE (2026-02-03)
- [x] Implement OCO (One-Cancels-Other) orders via bracket orders ✅ DONE (2026-02-03)
- [x] Add GTC vs Day orders support ✅ DONE (2026-02-03)
- [x] Use OHLC data for realistic order fills
- [x] Add order management methods (cancel, modify, get_orders)

### Trade Tracking ✅ COMPLETE (Phase 1.5)
- [x] Create `Trade` dataclass to track complete trades
- [x] Track entry/exit prices, sizes, timestamps
- [x] Calculate MAE (Max Adverse Excursion) and MFE (Max Favorable Excursion) ✅ DONE (2026-02-03)
- [x] Export trades to Polars DataFrame
- [x] Add trade-level metrics (win rate, profit factor, expectancy, SQN, Kelly) - basic stats done

### Short Selling ✅ SUPPORTED
- [x] Add short selling support methods (already supported via negative quantities)
- [ ] Implement borrow costs for short positions (planned for Phase 3)
- [x] Add position type helpers (is_long, is_short, etc.) - available in Trade class

---

## Phase 2: Visualization & Analysis 🔴 HIGH

### Plotting Module
- [ ] Create `polarbtest/plotting/` module structure
- [ ] Implement main `plot_backtest()` function
- [ ] Add candlestick/OHLC price charts
- [ ] Add equity curve visualization
- [ ] Add drawdown chart (underwater plot)
- [ ] Add trade markers (entry/exit arrows)
- [ ] Add volume bars
- [ ] Add indicator overlay helpers
- [ ] Support saving to HTML

### Enhanced Metrics
- [ ] Implement Omega ratio
- [ ] Implement Ulcer index
- [ ] Implement Tail ratio
- [ ] Implement Information ratio
- [ ] Calculate Alpha and Beta vs benchmark
- [ ] Add drawdown duration statistics
- [ ] Calculate monthly returns table

---

## Phase 3: Risk Management & Realism 🟡 MEDIUM

### Position Sizing
- [ ] Create `Sizer` base class
- [ ] Implement `FixedSizer`
- [ ] Implement `PercentSizer`
- [ ] Implement `FixedRiskSizer`
- [ ] Implement `KellySizer`
- [ ] Implement `VolatilitySizer` (ATR-based)
- [ ] Implement `MaxPositionSizer` wrapper
- [ ] Add `order_with_sizer()` method to Portfolio

### Risk Limits
- [ ] Add max position size limit
- [ ] Add max total exposure limit
- [ ] Add max drawdown stop
- [ ] Add daily loss limit
- [ ] Implement risk limit enforcement in order execution

### Commission Models
- [ ] Create `CommissionModel` base class
- [ ] Implement `PercentCommission`
- [ ] Implement `MakerTakerCommission`
- [ ] Implement `TieredCommission`
- [ ] Implement `CustomCommission`

### Margin & Leverage
- [ ] Add leverage support to Portfolio
- [ ] Calculate buying power with leverage
- [ ] Implement margin requirements
- [ ] Handle margin calls (auto-close positions)

---

## Phase 4: Extended Indicators & Utilities 🟡 MEDIUM

### Trend Indicators
- [ ] Implement WMA (Weighted Moving Average)
- [ ] Implement HMA (Hull Moving Average)
- [ ] Implement VWAP
- [ ] Implement SuperTrend
- [ ] Implement ADX (Average Directional Index)

### Momentum Indicators
- [ ] Implement Stochastic Oscillator
- [ ] Implement Williams %R
- [ ] Implement CCI (Commodity Channel Index)
- [ ] Implement MFI (Money Flow Index)
- [ ] Implement ROC (Rate of Change)

### Volatility Indicators
- [ ] Implement Keltner Channels
- [ ] Implement Donchian Channels

### Volume Indicators
- [ ] Implement OBV (On-Balance Volume)
- [ ] Implement A/D Line (Accumulation/Distribution)

### Support/Resistance
- [ ] Implement pivot points (Standard, Fibonacci, Woodie, Camarilla)

### Data Utilities
- [ ] Create data validation functions
- [ ] Create data cleaning functions
- [ ] Implement OHLCV resampling

### TA-Lib Integration
- [ ] Create TA-Lib wrapper functions
- [ ] Add convenience methods for common indicators

---

## Phase 5: Optimization Enhancements 🟡 MEDIUM

### Core Optimization
- [ ] Add constraint functions to `optimize()`
- [ ] Implement multi-objective optimization
- [ ] Add Bayesian optimization (optional, requires scikit-optimize)

### Optimization Visualization
- [ ] Create 2D parameter heatmaps
- [ ] Create parameter sensitivity plots
- [ ] Create 3D optimization surface plots

---

## Phase 6: Documentation & Examples 🔴 HIGH

### Documentation
- [ ] Create comprehensive README
- [ ] Write getting started guide
- [ ] Write user guide sections (strategies, indicators, risk management, etc.)
- [ ] Write API reference documentation
- [ ] Create best practices guide

### Example Strategies
- [ ] Simple SMA Cross strategy
- [ ] RSI Mean Reversion strategy
- [ ] Bollinger Band Breakout strategy
- [ ] Trend Following with ATR stop-loss
- [ ] Multi-Asset Momentum strategy

---

## Phase 7: Advanced Features 🟢 LOW

### Advanced Analysis
- [ ] Implement Monte Carlo simulation
- [ ] Implement look-ahead bias detection

---

## Project Structure Updates

### New Modules to Create
- [ ] `polarbtest/orders.py` - Order types and management
- [ ] `polarbtest/trades.py` - Trade tracking
- [ ] `polarbtest/sizers.py` - Position sizing strategies
- [ ] `polarbtest/plotting/` - Visualization module
- [ ] `polarbtest/data/` - Data utilities
- [ ] `polarbtest/integrations/` - Optional integrations

### Testing
- [ ] Add unit tests for order system
- [ ] Add unit tests for trade tracking
- [ ] Add unit tests for position sizers
- [ ] Add integration tests
- [ ] Add performance benchmarks

---

## Dependencies

### Core (Required)
- polars >= 0.19.0
- numpy >= 1.20.0

### Optional Extras
- plotting: plotly >= 5.0.0
- optimization: scikit-optimize >= 0.9.0
- talib: TA-Lib >= 0.4.0
- data: yfinance >= 0.2.0
- reports: jinja2 >= 3.0.0

---

## Priority Order

1. **Phase 1** - Core trading functionality (order types, trade tracking)
2. **Phase 2** - Visualization and metrics
3. **Phase 6** - Documentation and examples
4. **Phase 3** - Risk management enhancements
5. **Phase 4** - Extended indicators
6. **Phase 5** - Optimization enhancements
7. **Phase 7** - Advanced features

**Estimated Timeline**: 10-14 weeks for Phases 1-6
