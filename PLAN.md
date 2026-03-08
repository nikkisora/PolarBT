# Feature Plan: Polars Backtest Extension Inspired Enhancements

Inspired by analysis of [Yvictor/polars_backtest_extension](https://github.com/Yvictor/polars_backtest_extension), a Rust-based weight-driven portfolio backtester for Polars. These 7 features fill gaps in PolarBT without duplicating what the event-driven engine already does well.

---

## Feature 1: Vectorized Weight-Based Backtest Mode

**Priority:** High
**Effort:** Medium
**Files:** `polarbt/weight_backtest.py` (new), `polarbt/__init__.py`, `tests/test_weight_backtest.py` (new)

### Motivation

Many real-world strategies are simple portfolio weight allocations (momentum rotation, factor models, equal-weight baskets). Currently users must write a full `Strategy` subclass with `next()` even for "buy everything where signal > 0 with equal weight." A declarative weight-based API is faster to write, easier to optimize, and can run significantly faster by skipping the Python per-bar loop.

### API Design

```python
from polarbt import backtest_weights, WeightBacktestResult

# Long format: one row per (date, symbol)
result = backtest_weights(
    data=df,                         # pl.DataFrame with date, symbol, price, weight columns
    date_col="date",                 # str, column name for dates
    symbol_col="symbol",             # str, column name for symbols
    price_col="close",               # str, column name for execution price
    weight_col="weight",             # str, column name for target weights
    # Optional OHLC for touched exit
    open_col="open",                 # str | None
    high_col="high",                 # str | None
    low_col="low",                   # str | None
    # Rebalancing
    resample="M",                    # str | None: "D", "W", "W-FRI", "M", "Q", "Y", None
    resample_offset=None,            # str | None: "1d", "2d", "1W" (Feature 2)
    # Costs
    fee_ratio=0.001,                 # float, transaction fee ratio
    tax_ratio=0.0,                   # float, transaction tax ratio (e.g. stamp duty)
    # Risk management
    stop_loss=None,                  # float | None, e.g. 0.10 = 10%
    take_profit=None,                # float | None
    trail_stop=None,                 # float | None
    position_limit=1.0,              # float, max weight per stock
    touched_exit=False,              # bool, use OHLC for intraday stop detection
    # Execution
    t_plus=1,                        # int, execution delay in bars (0=same bar, 1=next bar)
)
```

### Return type: `WeightBacktestResult`

```python
@dataclass
class WeightBacktestResult:
    # Core results
    equity: pl.DataFrame              # columns: date, cumulative_return
    trades: pl.DataFrame              # columns: symbol, entry_date, exit_date, entry_price,
                                      #   exit_price, weight, return_pct, bars_held, mae, mfe,
                                      #   bmfe, mdd, pdays
    # Metrics (reuse existing)
    metrics: BacktestMetrics
    # Next actions (Feature 6)
    next_actions: pl.DataFrame | None # columns: symbol, action, current_weight, target_weight

    def __str__(self) -> str: ...     # Delegate to metrics.__str__()
```

### Implementation Plan

1. **Data validation** — Require sorted `date_col`, non-null prices. Validate column existence.
2. **Weight normalization** — Port `normalize_weights_finlab` logic:
   - If `sum(|weights|) > 1.0`: divide by sum.
   - If `sum(|weights|) <= 1.0`: keep as-is (allows partial investment).
   - Apply `position_limit` clipping.
   - Boolean weight columns → equal-weight conversion.
3. **Resample boundary detection** — Detect period boundaries from date transitions:
   - Monthly: `date.dt.month()` changes.
   - Weekly: ISO week changes, or specific weekday.
   - Quarterly/Yearly: similar.
   - `None`: rebalance only when weights change.
4. **Core simulation loop** — Vectorized in Polars where possible, with a thin Python loop over unique dates:
   ```
   For each date:
     1. Update positions: cr *= (price_today / price_yesterday) per symbol
     2. If touched_exit: check OHLC stops (Feature 5 logic), exit at touched price
     3. Else: execute pending T+1 stops, then detect new stops for tomorrow
     4. If rebalance boundary (accounting for offset):
        Execute rebalance: calculate target values from weights * portfolio_value,
        compute deltas, apply fees/taxes
     5. Record equity = cash + sum(position_values)
   ```
5. **Trade tracking** — Record entry/exit for each position lifecycle.
6. **Metrics calculation** — Reuse `metrics.calculate_metrics()` on the equity DataFrame. Reuse `trade_level_metrics()` for trade stats.
7. **Integration with runner** — Add `backtest_weights_batch()` for parallel parameter sweeps over weight-based strategies (e.g., different lookback periods for signal generation).

### Key differences from event-driven engine

| Aspect | `Engine.run()` | `backtest_weights()` |
|--------|---------------|---------------------|
| Signal source | `strategy.next()` calls `order()` | Pre-computed weight column |
| Iteration | `iter_rows(named=True)` per bar | Loop over unique dates |
| Order types | MARKET, LIMIT, STOP, STOP_LIMIT | Implicit market orders on rebalance |
| Position mgmt | Manual via portfolio methods | Automatic from weight deltas |
| Speed | Slower (full Python per-bar) | Faster (vectorized where possible) |

### Tests

- Basic equal-weight portfolio: 3 stocks, verify equity matches manual calculation.
- Weight normalization: sum > 1, sum < 1, sum = 1, with zeros, all zeros.
- Resample modes: daily, weekly, monthly, quarterly, yearly, position-change-only.
- Stop loss / take profit / trailing stop.
- Fee and tax deduction.
- T+0 vs T+1 execution.
- Boolean weight column → equal-weight conversion.
- Single stock (degenerate case).
- Empty DataFrame, single row.

---

## Feature 2: Resample Offset for Rebalancing

**Priority:** Medium
**Effort:** Low
**Files:** `polarbt/weight_backtest.py`, `polarbt/core.py`, `tests/test_weight_backtest.py`, `tests/test_core.py`

### Motivation

Real portfolios often can't trade on the exact period boundary — month-end may be a holiday, end-of-month data arrives with delay, or trading desks need time to prepare orders. An offset delays rebalancing by N trading days after the boundary.

### API

```python
# Weight-based API (Feature 1)
result = backtest_weights(df, resample="M", resample_offset="2d")

# Event-driven API (new parameter on Engine)
engine = Engine(
    strategy=my_strategy,
    data=df,
    rebalance_period="M",        # existing concept
    rebalance_offset="2d",       # NEW: delay rebalance by 2 trading days
)
```

For the event-driven engine, `rebalance_offset` only applies if the strategy uses periodic rebalancing signaled through a new `Engine` parameter (or through `BacktestContext`). This is a lighter integration than in the weight-based path.

### Implementation

1. **Parse offset string** — Support `"Nd"` (days) and `"NW"` (weeks → N*7 days). Store as `int` days.
2. **Delayed rebalance queue** — When a period boundary is detected, push a `DelayedRebalance(target_date=boundary_date + offset_days)` onto a queue.
3. **Trigger check** — On each date, check if `current_date > target_date` for any queued rebalance. If yes, pop and execute.
4. **Multi-period gap handling** — If the market is closed for multiple periods (holiday gap), queue one rebalance per boundary. When multiple delayed rebalances trigger on the same day, collapse them into one (de-duplication, matching polars_backtest_extension behavior).

### Tests

- Monthly + 0d offset = standard monthly.
- Monthly + 1d offset: verify rebalance happens on 2nd trading day of new month.
- Monthly + offset with holiday gap: verify correct trigger date.
- Weekly + offset.
- Multiple rebalances collapsing into one during long gaps.
- Offset with no rebalance pending (no-op).

---

## Feature 3: Enhanced MAE/MFE Trade Metrics — BMFE, Per-Trade MDD, pdays

**Priority:** Medium
**Effort:** Low
**Files:** `polarbt/trades.py`, `polarbt/results.py`, `tests/test_trades.py`

### Motivation

PolarBT already tracks MAE (max adverse excursion) and MFE (max favorable excursion) per trade. Adding BMFE, per-trade MDD, and pdays gives deeper trade quality analysis:

- **BMFE** (Before-MAE MFE): "How good was the trade before the worst point?" High BMFE + high MAE = good trade with poor exit timing.
- **Per-trade MDD**: Maximum drawdown within a single trade's lifetime. Different from portfolio-level MDD.
- **pdays**: Number of bars where the trade was in profit. Identifies consistently profitable trades vs. oscillating ones.

### Changes

#### `Trade` dataclass — Add 3 fields

```python
@dataclass
class Trade:
    # ... existing fields ...
    mae: float | None = None
    mfe: float | None = None
    bmfe: float | None = None      # NEW: MFE at the time MAE occurred
    trade_mdd: float | None = None  # NEW: max drawdown within this trade
    pdays: int | None = None        # NEW: number of profitable bars
```

#### `TradeTracker` — Extend tracking state

Current `open_positions` dict entry gains 3 new keys:

```python
# In on_position_opened():
self.open_positions[asset] = {
    # ... existing keys ...
    "mae": 0.0,
    "mfe": 0.0,
    "bmfe": 0.0,         # NEW: MFE when MAE last worsened
    "trade_mdd": 0.0,    # NEW: max drawdown in this trade
    "peak_pnl": 0.0,     # NEW: running peak PnL for MDD calculation
    "pdays": 0,          # NEW: count of profitable bars
}
```

#### `TradeTracker.update_mae_mfe()` — Extend update logic

```python
def update_mae_mfe(self, asset: str, current_price: float) -> None:
    info = self.open_positions.get(asset)
    if not info:
        return

    entry_price = info["entry_price"]
    direction = info["direction"]

    if direction == "long":
        unrealized_pnl = current_price - entry_price
    else:
        unrealized_pnl = entry_price - current_price

    unrealized_pct = unrealized_pnl / entry_price

    # Existing MAE/MFE update
    old_mae = info["mae"]
    if unrealized_pct < info["mae"]:
        info["mae"] = unrealized_pct
        info["bmfe"] = info["mfe"]       # NEW: snapshot MFE at the moment MAE worsens
    if unrealized_pct > info["mfe"]:
        info["mfe"] = unrealized_pct

    # NEW: Per-trade MDD
    if unrealized_pct > info["peak_pnl"]:
        info["peak_pnl"] = unrealized_pct
    drawdown = unrealized_pct - info["peak_pnl"]
    if drawdown < info["trade_mdd"]:
        info["trade_mdd"] = drawdown

    # NEW: pdays
    if unrealized_pct > 0:
        info["pdays"] += 1
```

#### `TradeTracker.on_position_closed()` — Pass new fields to Trade

```python
# In the Trade constructor call:
Trade(
    ...,
    bmfe=entry_info.get("bmfe"),
    trade_mdd=entry_info.get("trade_mdd"),
    pdays=entry_info.get("pdays"),
)
```

#### `TradeTracker.get_trades_df()` — Add columns

Add `"bmfe"`, `"trade_mdd"`, `"pdays"` to the DataFrame columns.

#### `TradeStats` — Add aggregate metrics

```python
@dataclass
class TradeStats:
    # ... existing fields ...
    avg_bmfe: float = 0.0       # NEW: average BMFE across trades
    avg_trade_mdd: float = 0.0  # NEW: average per-trade max drawdown
    avg_pdays: float = 0.0      # NEW: average profitable days per trade
```

### Tests

- Long trade going up then down: verify BMFE = MFE at the bar before the crash.
- Trade that never goes negative: BMFE = 0.0, pdays = bars_held.
- Trade that never goes positive: pdays = 0, trade_mdd = MAE.
- Partial close: verify both the closed and remaining portions track independently.
- Short trade: verify direction-aware calculation.
- Multiple trades: verify `TradeStats` aggregates correctly.

---

## Feature 4: Liquidity Metrics

**Priority:** Medium
**Effort:** Low-Medium
**Files:** `polarbt/metrics.py`, `polarbt/results.py`, `tests/test_metrics.py`

### Motivation

Three metrics that answer critical deployment questions:

1. **buy_high_ratio**: What fraction of entries occurred at limit-up prices? (Can't actually buy at auction-halted prices.)
2. **sell_low_ratio**: What fraction of exits occurred at limit-down prices? (Can't actually sell.)
3. **capacity**: How much capital can this strategy manage? Based on 10th percentile of money flow through traded stocks.

### API

Liquidity metrics are calculated from optional data columns, and added to `BacktestMetrics`.

```python
# Users provide optional columns in their data
df = df.with_columns([
    pl.col("close").mul(1.1).alias("limit_up"),     # 10% limit-up price
    pl.col("close").mul(0.9).alias("limit_down"),    # 10% limit-down price
    (pl.col("close") * pl.col("volume")).alias("trading_value"),  # daily money flow
])

# Metrics are automatically calculated when columns are present
result = backtest(MyStrategy, df)
print(result.buy_high_ratio)   # 0.02 = 2% of entries at limit-up
print(result.sell_low_ratio)   # 0.01 = 1% of exits at limit-down
print(result.capacity)         # 5_000_000.0 = strategy can handle ~$5M
```

### Implementation

#### New fields on `BacktestMetrics`

```python
@dataclass
class BacktestMetrics:
    # ... existing fields ...
    buy_high_ratio: float | None = None   # NEW
    sell_low_ratio: float | None = None   # NEW
    capacity: float | None = None         # NEW
```

#### New function in `metrics.py`

```python
def liquidity_metrics(
    trades_df: pl.DataFrame,
    data: pl.DataFrame,
    limit_up_col: str = "limit_up",
    limit_down_col: str = "limit_down",
    trading_value_col: str = "trading_value",
) -> dict[str, float | None]:
    """Calculate liquidity metrics from trades and market data.

    Args:
        trades_df: Trades DataFrame from TradeTracker.
        data: Original market data DataFrame.
        limit_up_col: Column with limit-up prices.
        limit_down_col: Column with limit-down prices.
        trading_value_col: Column with daily money flow (price * volume).

    Returns:
        Dict with buy_high_ratio, sell_low_ratio, capacity. None if columns missing.
    """
```

**buy_high_ratio calculation:**
- For each entry trade, check if `entry_price >= limit_up` on the entry date.
- `buy_high_ratio = count(entries at limit-up) / total_entries`.

**sell_low_ratio calculation:**
- For each exit trade, check if `exit_price <= limit_down` on the exit date.
- `sell_low_ratio = count(exits at limit-down) / total_exits`.

**capacity calculation:**
- For each traded symbol on each active trading day, get `trading_value`.
- Capacity = 10th percentile of these values.
- Interpretation: the strategy can handle at least this much capital flow on 90% of trading days.

#### Integration

- Call `liquidity_metrics()` inside `Engine._calculate_results()` if the relevant columns exist in the data.
- Pass results into `BacktestMetrics` constructor.
- Also usable in `backtest_weights()` (Feature 1).

### Tests

- DataFrame with limit_up/limit_down columns: verify ratio calculation.
- All entries at limit-up: ratio = 1.0.
- No entries at limit-up: ratio = 0.0.
- Missing columns: returns None gracefully.
- Capacity calculation with known trading_value distribution.

---

## Feature 5: Touched Exit — OHLC Priority Logic

**Priority:** Low-Medium
**Effort:** Low
**Files:** `polarbt/core.py`, `tests/test_core.py`

### Motivation

PolarBT already checks OHLC for intrabar stop/TP triggers. However, when multiple conditions trigger in the same bar (e.g., both high touches TP and low touches SL), the execution price matters. The polars_backtest_extension uses a clear priority:

1. **Open** price is checked first — if open already breaches, exit at open (gap scenario).
2. **High** — if high breaches take-profit, exit at the take-profit threshold price.
3. **Low** — if low breaches stop-loss, exit at the stop-loss threshold price.

This is more realistic because gaps that breach stops on open should execute at the gap price, not the stop price.

### Current behavior

In `Portfolio._check_stop_losses()` (line ~812):
- Checks `low <= stop_price` for longs → closes at `stop_price`.
- Checks `high >= stop_price` for shorts → closes at `stop_price`.

Separate `_check_take_profits()` runs independently. No priority between them.

### Proposed changes

#### Add priority-based stop checking in `Portfolio.update_prices()`

Replace the three independent stop-check calls with a unified priority-based check:

```python
def _check_stops_with_priority(self) -> None:
    """Check all stop conditions with OHLC priority: open > high > low.

    For each asset with active stops:
    1. If open breaches any stop → exit at open price (gap scenario)
    2. If high breaches take-profit (long) or stop-loss (short) → exit at threshold
    3. If low breaches stop-loss (long) or take-profit (short) → exit at threshold

    Only one exit per asset per bar. First match wins.
    """
    assets_to_check = set()
    assets_to_check.update(self._stop_losses.keys())
    assets_to_check.update(self._take_profits.keys())
    assets_to_check.update(self._trailing_stops.keys())

    for asset in list(assets_to_check):
        if asset not in self.positions:
            continue

        ohlc = self._current_ohlc.get(asset, {})
        open_price = ohlc.get("open", self._current_prices.get(asset, 0.0))
        high_price = ohlc.get("high", open_price)
        low_price = ohlc.get("low", open_price)

        position_size = self.positions[asset]
        is_long = position_size > 0

        stop_price = self._stop_losses.get(asset)
        tp_price = self._take_profits.get(asset)
        trail_info = self._trailing_stops.get(asset)
        trail_stop_price = trail_info.get("stop_price") if trail_info else None

        # Effective stop = tightest of fixed stop and trailing stop
        effective_stop = None
        if is_long:
            candidates = [p for p in [stop_price, trail_stop_price] if p is not None]
            effective_stop = max(candidates) if candidates else None
        else:
            candidates = [p for p in [stop_price, trail_stop_price] if p is not None]
            effective_stop = min(candidates) if candidates else None

        # Priority 1: Open breaches
        if effective_stop is not None:
            if is_long and open_price <= effective_stop:
                self._close_at_price(asset, open_price, is_risk_order=True)
                continue
            if not is_long and open_price >= effective_stop:
                self._close_at_price(asset, open_price, is_risk_order=True)
                continue

        if tp_price is not None:
            if is_long and open_price >= tp_price:
                self._close_at_price(asset, open_price, is_risk_order=True)
                continue
            if not is_long and open_price <= tp_price:
                self._close_at_price(asset, open_price, is_risk_order=True)
                continue

        # Priority 2: High breaches (TP for long, SL for short)
        if is_long and tp_price is not None and high_price >= tp_price:
            self._close_at_price(asset, tp_price, is_risk_order=True)
            continue
        if not is_long and effective_stop is not None and high_price >= effective_stop:
            self._close_at_price(asset, effective_stop, is_risk_order=True)
            continue

        # Priority 3: Low breaches (SL for long, TP for short)
        if is_long and effective_stop is not None and low_price <= effective_stop:
            self._close_at_price(asset, effective_stop, is_risk_order=True)
            continue
        if not is_long and tp_price is not None and low_price <= tp_price:
            self._close_at_price(asset, tp_price, is_risk_order=True)
            continue

        # No exit triggered — update trailing stop high-water mark
        if trail_info is not None:
            # ... existing trailing stop update logic ...
```

#### Backward compatibility

The existing `_check_stop_losses()`, `_check_take_profits()`, `_update_and_check_trailing_stops()` methods remain as-is but `update_prices()` calls the new unified method instead when OHLC data is available. When OHLC is not available, the existing behavior (close-price checks) is unchanged.

### Tests

- Gap down through stop-loss: open < stop → exit at open, not stop price.
- Gap up through take-profit: open > TP → exit at open, not TP price.
- Normal stop-loss trigger: low touches stop but open is above → exit at stop price.
- Normal take-profit trigger: high touches TP but open is below → exit at TP price.
- Both SL and TP touched in same bar: verify priority determines which fires.
- No OHLC data: existing behavior unchanged.
- Trailing stop with gap: verify open-price exit.
- Short position: verify reversed direction logic.

---

## Feature 6: Forward-Looking Stock Operations (Next Actions)

**Priority:** Low
**Effort:** Medium
**Files:** `polarbt/weight_backtest.py`, `polarbt/analysis.py`, `tests/test_weight_backtest.py`

### Motivation

After running a backtest up to today, "what do I do next?" is the natural question for live trading. The polars_backtest_extension computes `stock_operations` at the end of the backtest: which symbols to enter, exit, or hold, plus target weights for the next period.

This feature is primarily useful for the weight-based backtest (Feature 1) where the answer is deterministic from the weight column.

### API

```python
result = backtest_weights(df)

# Already included in WeightBacktestResult.next_actions
print(result.next_actions)
# shape: (5, 4)
# ┌────────┬────────┬────────────────┬───────────────┐
# │ symbol │ action │ current_weight │ target_weight │
# │ ---    │ ---    │ ---            │ ---           │
# │ str    │ str    │ f64            │ f64           │
# ╞════════╪════════╪════════════════╪═══════════════╡
# │ AAPL   │ hold   │ 0.20           │ 0.25          │
# │ GOOGL  │ exit   │ 0.15           │ 0.00          │
# │ MSFT   │ enter  │ 0.00           │ 0.20          │
# │ TSLA   │ hold   │ 0.10           │ 0.10          │
# └────────┴────────┴────────────────┴───────────────┘
```

Standalone helper for the event-driven path:

```python
from polarbt.analysis import compute_next_actions

actions = compute_next_actions(
    current_positions={"AAPL": 100, "GOOGL": 50},
    target_weights={"AAPL": 0.25, "MSFT": 0.20, "TSLA": 0.10},
    portfolio_value=100_000.0,
    current_prices={"AAPL": 150.0, "GOOGL": 100.0, "MSFT": 300.0, "TSLA": 200.0},
)
# Returns DataFrame with: symbol, action, current_value, target_value, delta_shares
```

### Implementation

1. At the end of `backtest_weights()`, compare the last active positions against the latest signal weights from the weight column.
2. Classify each symbol:
   - `"enter"`: not currently held, has positive target weight.
   - `"exit"`: currently held, has zero (or absent) target weight.
   - `"hold"`: currently held, has positive target weight.
3. Calculate next rebalance date using the resample pattern + offset.
4. Build a `pl.DataFrame` with the results.

### Tests

- Portfolio with 3 held + 2 new signals: verify enter/exit/hold classification.
- All positions exiting: verify all marked "exit".
- No changes: all marked "hold".
- Empty portfolio with signals: all marked "enter".
- Integration with `backtest_weights()`: verify `result.next_actions` is populated.

---

## Feature 7: Factor-Based Price Adjustment

**Priority:** Low
**Effort:** Low
**Files:** `polarbt/core.py`, `polarbt/weight_backtest.py`, `tests/test_core.py`

### Motivation

When backtesting with split/dividend-adjusted prices, returns should use adjusted prices but commissions/fees should be calculated on raw (unadjusted) prices. The `factor` column provides this mapping: `raw_price = adjusted_price / factor`.

### API

```python
# Weight-based API
result = backtest_weights(
    df,
    price_col="adj_close",
    factor_col="factor",    # NEW: optional, raw_price = adj_close / factor
)

# Event-driven API — new parameter on Engine
engine = Engine(
    strategy=my_strategy,
    data=df,
    factor_column="factor",  # NEW: optional
)
```

### Implementation

#### `standardize_dataframe()` — Recognize factor column

Add `"factor"` to the set of known columns. If present, keep it through standardization unchanged.

#### Commission calculation — Use raw price

In `Portfolio._try_execute_order()`, when calculating commission:

```python
# Current: commission on fill_price
commission = self._commission_model.calculate(fill_price * abs(quantity), ...)

# New: commission on raw price if factor available
raw_price = fill_price / self._factors.get(asset, 1.0)
commission = self._commission_model.calculate(raw_price * abs(quantity), ...)
```

#### `Portfolio.update_prices()` — Store factors

Add a `_factors: dict[str, float]` that's updated each bar from the data row.

#### Weight-based backtest — Same principle

In `backtest_weights()`, when calculating rebalance costs:
```python
raw_price = price / factor  # factor defaults to 1.0
fee = abs(delta_value / price * raw_price) * fee_ratio
```

### Tests

- Without factor column: behavior unchanged.
- With factor=2.0: commissions calculated on half the price, returns on full price.
- Factor of 1.0: identical to no-factor behavior.
- Factor changes mid-backtest (simulating a 2:1 split): verify returns are smooth but commission reflects raw price.

---

## Implementation Order

```
Phase 1 (Foundation):
  Feature 3: Enhanced MAE/MFE          — standalone, no dependencies
  Feature 5: Touched Exit Priority     — standalone, improves existing code

Phase 2 (Weight-Based Core):
  Feature 1: Weight-Based Backtest     — the big one, unlocks Features 2, 6, 7
  Feature 2: Resample Offset           — integrates into Feature 1

Phase 3 (Analysis & Polish):
  Feature 4: Liquidity Metrics         — works with both engines
  Feature 6: Next Actions              — builds on Feature 1
  Feature 7: Factor-Based Prices       — works with both engines
```

## Verification

After all features are implemented:
1. Run `pytest tests/ -q` — all tests pass.
2. Run `ruff check --fix . && ruff format .` — no lint issues.
3. Run `mypy polarbt/` — no type errors.
4. Update `polarbt/__init__.py` exports.
5. Archive summary in `archive/2026-03-08_polars_backtest_features.md`.
