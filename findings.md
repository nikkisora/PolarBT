# Findings: Phase 1.5 Implementation

**Session**: 2026-02-03
**Task**: Implement Phase 1.5 - Advanced Order Types

---

## Key Discoveries

### Existing Infrastructure

**Order System (polarbtest/orders.py)**:
- ✅ Order class exists with all needed fields (order_id, asset, size, order_type, status)
- ✅ Already has `valid_until`, `parent_order`, `child_orders` fields for Phase 1.5
- ✅ OrderType enum has MARKET, LIMIT, STOP, STOP_LIMIT
- ✅ OrderStatus enum has PENDING, FILLED, PARTIAL, CANCELLED, REJECTED, EXPIRED
- ✅ Helper methods: is_filled(), is_active(), can_be_cancelled(), is_buy(), is_sell()
- ✅ mark_filled(), mark_cancelled(), mark_rejected(), mark_expired() methods

**Trade Tracking (polarbtest/trades.py)**:
- ✅ Trade class exists with trade_id, asset, direction, entry/exit data
- ✅ **MAE and MFE fields already exist** (mae, mfe) but are currently None
- ✅ TradeTracker class handles position open/close/reverse
- ✅ Exports trades to DataFrame
- ⚠️ MAE/MFE tracking not yet implemented (need to update during trade lifecycle)

**Portfolio (polarbtest/core.py)**:
- ✅ Order management: order(), order_target(), close_position()
- ✅ get_orders(), get_order(), cancel_order() methods exist
- ✅ set_stop_loss() method exists (lines 679-709)
- ✅ _check_stop_losses() called in update_prices() (line 237)
- ✅ _execute_pending_orders() and _check_order_expiry() exist
- ✅ _current_ohlc dict for high/low price data
- ✅ TradeTracker integrated via trade_tracker attribute
- ❌ No take-profit functionality yet
- ❌ No trailing stop functionality yet
- ❌ No OCO order functionality yet

---

## Code Patterns

### Order Execution Flow:
1. update_prices() called each bar with prices, ohlc_data, bar_index, timestamp
2. _check_stop_losses() checks if stops triggered
3. _execute_pending_orders() processes orders based on order_delay
4. _try_execute_order() attempts execution:
   - Checks limit order fillability via _can_fill_limit_order()
   - Uses OHLC data (low for buy limits, high for sell limits)
   - Applies slippage and commission
   - Updates positions and trade_tracker

### Stop-Loss Pattern:
- Stored in `_stop_losses` dict: {asset: {"stop_price": float, "order_id": str}}
- Checked against OHLC low price for long positions
- Triggers close_position() when hit
- set_stop_loss() supports stop_price or stop_pct

### Trade Tracking Pattern:
- on_position_opened() when entering new position
- on_position_closed() when reducing/closing position
- on_position_reversed() when flipping from long to short
- Automatic P&L calculation in Trade.__post_init__()

### Testing Pattern:
- Pytest with class-based organization
- Simple test methods with clear names
- Direct instantiation of classes
- Minimal fixtures (tests are self-contained)

---

## Design Decisions

### Take-Profit Implementation:
- Mirror stop-loss structure with _take_profits dict
- Check in _check_take_profits() called after stop-loss check
- Use OHLC high for long positions (opposite of stop-loss)
- set_take_profit(), remove_take_profit(), get_take_profit() methods

### Trailing Stop Implementation:
- Store in _trailing_stops dict with: {asset: {"trail_pct": float, "trail_amount": float, "highest_price": float, "stop_price": float}}
- Update in _update_trailing_stops() called each bar
- Track highest price (for longs) / lowest price (for shorts)
- Recalculate stop_price as price moves favorably
- Execute when stop_price breached

### OCO Implementation:
- Use existing parent_order and child_orders fields
- order_bracket() creates entry order with linked SL/TP child orders
- link_oco() manually links existing orders
- _cancel_oco_siblings() cancels related orders when one fills
- Call from _try_execute_order() after successful fill

### Order Expiry Implementation:
- Use existing valid_until field (bar index)
- _check_order_expiry() already exists (line 294)
- Call in update_prices() before executing orders
- mark_expired() for expired orders

### MAE/MFE Implementation:
- Update in _update_mae_mfe() called during update_prices()
- Track for each open position in trade_tracker
- Calculate unrealized P&L vs entry price
- Update Trade.mae (minimum) and Trade.mfe (maximum)

---

## Open Questions

1. ~~Does Order class already support take-profit orders?~~ ✅ No, need to implement
2. ~~Is there existing trailing stop infrastructure?~~ ✅ No, need to implement
3. ~~How are orders currently executed in the event loop?~~ ✅ Documented above
4. ~~What's the current test coverage for order system?~~ ✅ Basic tests exist in tests/test_orders.py
5. ~~Are MAE/MFE fields already in Trade class?~~ ✅ Yes, but not populated

**New Questions**:
6. ~~Should trailing stops support both long and short positions?~~ ✅ Yes, implemented for both
7. Should OCO orders support more than 2 linked orders?
8. Should MAE/MFE be calculated as absolute $ or percentage?

**Trailing Stop Implementation Notes**:
- Critical bug discovered: Intra-bar OHLC handling can cause false triggers
- Solution: Check OLD stop price before updating based on new high/low
- Implementation: Combined update and check into `_update_and_check_trailing_stops()`
- Trailing stops use HIGH price for updates (long) or LOW price (shorts) for accuracy

---

## Phase 9: Day Order Auto-Expiry Analysis

### Current Implementation
- `order_day()` requires manual `bars_valid` parameter (default=1)
- Expiry calculated as: `order.valid_until = self._current_bar + bars_valid`
- No awareness of actual trading session/day boundaries

### Problem
- Users must manually specify how many bars constitute a "day"
- Different timeframes (1min vs 1hour vs daily) require different `bars_valid` values
- Not intuitive - "day order" should automatically understand day boundaries

### Available Context
- `update_prices()` receives `timestamp` parameter (can be int, datetime, pd.Timestamp, etc.)
- `_current_timestamp` stored in Portfolio
- Data typically has timestamp/date column

### Design Options

**Option 1: Timestamp-based day detection**
- Extract date from timestamp (handle various timestamp types)
- Order expires at end of the current trading day (when date changes)
- Pros: Most accurate, matches real trading behavior
- Cons: Requires timestamp handling logic, needs to support multiple timestamp types

**Option 2: Market session configuration**
- Allow user to specify bars_per_day in Portfolio initialization
- Auto-calculate expiry: `bars_valid = bars_per_day`
- Pros: Simple, deterministic
- Cons: Less automatic, user still needs to configure

**Option 3: Hybrid approach**
- Try timestamp-based detection first (if timestamps available)
- Fall back to bars_valid parameter
- Pros: Best of both worlds, backwards compatible
- Cons: More complex implementation

### Recommended Approach: Option 3 (Hybrid) - REVISED

**Implementation plan**:
1. Add `bars_per_day` optional parameter to Portfolio.__init__() for manual override
2. Enhance `order_day()` to auto-detect day boundaries from timestamp if available
3. Fall back to `bars_valid` or `bars_per_day` if timestamps unavailable
4. Support timestamp formats: yyyy-mm-dd hh:mm:ss, yyyy-mm-dd, unix timestamp (int/float), datetime objects
5. Remove pandas dependency - use only Python stdlib and datetime

**Revised Requirements (from user)**:
- No pandas import - keep it simple with stdlib
- Timestamp standardization: yyyy-mm-dd hh:mm:ss (or yyyy-mm-dd, or unix)
- bars_per_day must calculate actual end-of-day bar considering current bar position within day
  - Example: If bars_per_day=390 (1-min bars, 6.5hr trading day) and order placed at bar 100 (within day 0),
    it should expire at bar 389 (end of day 0), not bar 490 (100+390)

**Improved bars_per_day logic**:
```python
# Calculate which bar within the day we're currently on
current_bar_in_day = self._current_bar % self.bars_per_day
# Calculate bars remaining until end of current day
bars_until_eod = self.bars_per_day - current_bar_in_day - 1
# Order expires at end of current day
order.valid_until = self._current_bar + bars_until_eod
```

**Algorithm**:
```python
def order_day(self, asset, quantity, limit_price=None, bars_valid=None):
    # 1. Explicit bars_valid (highest priority)
    if bars_valid is not None:
        order.valid_until = self._current_bar + bars_valid
    
    # 2. Try timestamp-based detection
    elif self._current_timestamp is not None:
        current_day = extract_date(self._current_timestamp)
        order.expiry_date = current_day
    
    # 3. Fall back to bars_per_day (with smart end-of-day calculation)
    elif self.bars_per_day is not None:
        current_bar_in_day = self._current_bar % self.bars_per_day
        bars_until_eod = self.bars_per_day - current_bar_in_day - 1
        order.valid_until = self._current_bar + bars_until_eod
    
    # 4. Default to 1 bar
    else:
        order.valid_until = self._current_bar + 1
```

---

## Phase 9 Completion Summary

### What Changed

1. **Order class** (`polarbtest/orders.py`):
   - Added `expiry_date` field for timestamp-based expiry

2. **Portfolio class** (`polarbtest/core.py`):
   - Added `bars_per_day` parameter to `__init__()`
   - Added `_extract_date()` helper (stdlib only, no pandas)
   - Enhanced `order_day()` with 4-level priority system
   - Updated `_check_order_expiry()` for date-based expiry

3. **Tests** (`tests/test_order_expiry.py`):
   - 16 tests total (7 original + 9 new)
   - Added 3 bars_per_day tests
   - All passing

### Key Features

✅ **Automatic day detection** from timestamps  
✅ **Smart bars_per_day** calculation (end-of-day, not just +N bars)  
✅ **No pandas dependency** (stdlib only)  
✅ **Backwards compatible** (all existing code works)  
✅ **Flexible** (4 priority levels: explicit, timestamp, config, default)  

### Priority System

1. Explicit `bars_valid` parameter (highest)
2. Timestamp-based day detection
3. `bars_per_day` configuration
4. Default to 1 bar (lowest)

### Test Results

```
pytest: 152 passed, 2 skipped
mypy: Success (0 errors)
ruff: No errors
```

---

## References

- ENHANCEMENT_ROADMAP.md - Phase 1.1 Order System Redesign (lines 63-227)
- TODO.md - Phase 1 Order System (lines 9-33)
- polarbtest/orders.py - Order implementation
- polarbtest/trades.py - Trade tracking
- polarbtest/core.py - Portfolio order management
- PHASE_1_5_DAY_ORDER_ENHANCEMENT.md - Day order enhancement summary
