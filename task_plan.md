# Task Plan: Phase 1.5 - Advanced Order Types

**Goal**: Implement Phase 1.5 from TODO.md - Advanced order types including take-profit, trailing stop-loss, OCO orders, and GTC/Day order support.

**Date Started**: 2026-02-03
**Status**: in_progress

---

## Context

From TODO.md, Phase 1.5 includes:
- [ ] Implement take-profit orders (similar to stop-loss)
- [ ] Implement trailing stop-loss
- [ ] Implement OCO (One-Cancels-Other) orders
- [ ] Add GTC vs Day orders support (foundation exists)
- [ ] Calculate MAE (Max Adverse Excursion) and MFE (Max Favorable Excursion) (foundation exists)

From ENHANCEMENT_ROADMAP.md Phase 1.1 Order System Redesign section:
- Order class already exists in polarbtest/orders.py
- Need to add take-profit, trailing stop-loss, OCO functionality
- Need to implement order validity (GTC vs Day)
- Trade tracking exists in polarbtest/trades.py - need to add MAE/MFE tracking

---

## Phases

### Phase 1: Analyze Existing Code ✅ complete
**Status**: complete
**Estimated time**: 15 minutes

Read and understand:
- ✅ polarbtest/orders.py - Order class and types
- ✅ polarbtest/trades.py - TradeTracker implementation  
- ✅ polarbtest/core.py - Portfolio order management
- ✅ Understand current order execution flow

**Files examined**:
- polarbtest/orders.py - Complete Order infrastructure exists
- polarbtest/trades.py - MAE/MFE fields exist but not populated
- polarbtest/core.py (order-related methods) - Good foundation
- tests/test_orders.py - Test patterns understood

**Key findings**: See findings.md for detailed analysis

---

### Phase 2: Implement Take-Profit Orders ✅ complete
**Status**: complete
**Estimated time**: 30 minutes

Tasks:
1. ✅ Add take-profit order support to Order class (if needed) - Used existing dict structure
2. ✅ Implement `set_take_profit()` method in Portfolio
3. ✅ Implement `remove_take_profit()` method in Portfolio
4. ✅ Implement `get_take_profit()` helper method
5. ✅ Add take-profit order execution logic via `_check_take_profits()`
6. ✅ Test take-profit functionality - All tests pass (8 passed, 1 skipped)
7. ✅ Added cleanup logic to remove SL/TP when positions close
8. ✅ Enhanced stop-loss to support short positions

**Success criteria**:
- ✅ Can set take-profit on positions
- ✅ Take-profit triggers when price reaches target  
- ✅ Positions close automatically at target price
- ✅ Both absolute price and percentage targets work
- ✅ SL/TP cleanup when position closes

---

### Phase 3: Implement Trailing Stop-Loss ✅ complete
**Status**: complete
**Estimated time**: 45 minutes (actual: 90 minutes including debugging)

Tasks:
1. ✅ Add trailing stop data structure to Portfolio
2. ✅ Implement `set_trailing_stop()` method with support for:
   - ✅ Percentage-based trailing
   - ✅ Absolute value trailing
   - ⏸️ ATR-based trailing (deferred for later)
3. ✅ Add trailing stop update logic in event loop (`_update_and_check_trailing_stops`)
4. ✅ Add trailing stop execution when triggered (combined with update)
5. ✅ Test trailing stop functionality (10/10 passing)
6. ✅ Fixed intra-bar OHLC bug (check old stop before updating based on new high/low)

**Success criteria**:
- ✅ Trailing stop follows price movements upward
- ✅ Triggers when price retraces by specified amount
- ✅ Multiple trailing stop types work correctly (percentage and absolute)
- ✅ All tests passing

**Bug fixed**: Combined update and check logic to avoid triggering stop on same bar where it was updated due to intra-bar OHLC range.

---

### Phase 4: Implement OCO Orders ✅ complete
**Status**: complete
**Estimated time**: 40 minutes (actual: 50 minutes)

Tasks:
1. ✅ OCO linking fields already exist in Order class (parent_order, child_orders)
2. ✅ Implement `order_bracket()` method (entry + SL + TP)
3. ⏸️ Manual OCO linking deferred (not needed - automatic via bracket orders)
4. ✅ OCO cancellation logic already exists in `_update_trade_tracker()` cleanup
5. ✅ Test OCO functionality (6 tests, all passing, 1 skipped)

**Success criteria**:
- ✅ Bracket orders create entry with SL and TP
- ✅ When SL triggers, TP is cancelled (and vice versa)
- ✅ Supports both absolute prices and percentages

**Note**: OCO behavior is automatic - when a position closes (either from SL or TP triggering), both stops are removed via the existing cleanup logic in `_update_trade_tracker()`.

---

### Phase 5: Implement GTC vs Day Orders ✅ complete
**Status**: complete
**Estimated time**: 30 minutes (actual: 25 minutes)

Tasks:
1. ✅ `valid_until` field already exists in Order class
2. ✅ Order expiry logic already exists in `_check_order_expiry()`, just needed to call it
3. ✅ Added helper methods `order_gtc()` and `order_day()`
4. ✅ Test order expiry (7 tests, all passing)

**Success criteria**:
- ✅ GTC orders stay active until filled or manually cancelled (valid_until=None)
- ✅ Day orders expire after specified bars (valid_until=bar+days)
- ✅ Expired orders are automatically marked EXPIRED

**Note**: The order expiry infrastructure already existed, just needed to be integrated into the event loop and documented with helper methods.

---

### Phase 6: Implement MAE/MFE Tracking ✅ complete
**Status**: complete
**Estimated time**: 30 minutes (actual: 35 minutes)

Tasks:
1. ✅ MAE/MFE fields already exist in Trade class
2. ✅ Implemented `update_mae_mfe()` in TradeTracker
3. ✅ Call MAE/MFE update in Portfolio.update_prices() for all open positions
4. ✅ Added MAE/MFE to trade DataFrame export
5. ✅ Test MAE/MFE calculation (8 tests, all passing)

**Success criteria**:
- ✅ MAE tracks worst unrealized loss during trade (most negative P&L)
- ✅ MFE tracks best unrealized profit during trade (most positive P&L)
- ✅ Values export correctly to trades DataFrame
- ✅ Works across multiple trades independently

**Implementation notes**:
- MAE/MFE initialized to 0.0 when position opens
- Updated on every price update for open positions
- Copied to Trade object when position closes
- Calculated as absolute price difference (not percentage)

---

### Phase 7: Testing & Integration ✅ complete
**Status**: complete
**Estimated time**: 45 minutes (actual: 20 minutes)

Tasks:
1. ✅ Comprehensive tests written for all new features (29 new tests total)
2. ✅ Ran existing test suite - no regressions
3. ✅ All tests passing
4. ✅ Ran ruff and mypy - fixed all issues
5. ⏸️ Update IMPLEMENTATION.md (deferred to Phase 8)

**Success criteria**:
- ✅ All tests pass (143 passed, 2 skipped)
- ✅ No mypy errors (all type checking passes)
- ✅ Ruff warnings minimal (only style suggestions, no errors)
- ⏸️ Documentation update in Phase 8

**Quality metrics**:
- mypy: Success (0 errors)
- ruff: 6 style suggestions (SIM rules - optional improvements)
- pytest: 143 passed, 2 skipped, 0 failed

---

### Phase 8: Update Documentation ⏸️ pending  
**Status**: pending
**Estimated time**: 20 minutes

Tasks:
1. ✅ Update TODO.md to mark Phase 1.5 as complete
2. ⏸️ Update IMPLEMENTATION.md with API examples
3. ⏸️ Add usage examples for new features (optional)
4. ✅ Update archive with implementation summary

**Success criteria**:
- TODO.md reflects completion
- IMPLEMENTATION.md has clear examples  
- Archive contains summary

---

### Phase 9: Enhance Day Order Expiry Logic ✅ complete
**Status**: complete
**Estimated time**: 60 minutes (actual: 55 minutes)

**Context**: Currently `order_day()` requires manual `bars_valid` parameter. Need to automatically calculate expiry based on trading session/day logic.

Tasks:
1. ✅ Analyze current day order implementation
2. ✅ Design automatic expiry calculation (understand what "day" means in backtesting context)
3. ✅ Implement automatic expiry calculation (initial version)
   - ✅ Add `expiry_date` field to Order class
   - ✅ Add `bars_per_day` parameter to Portfolio
   - ✅ Add `_extract_date()` helper function
   - ✅ Update `order_day()` with hybrid expiry logic
   - ✅ Update `_check_order_expiry()` to handle date-based expiry
4. ✅ Refine implementation based on feedback
   - ✅ Remove pandas dependency (stdlib only)
   - ✅ Standardize timestamp parsing (yyyy-mm-dd hh:mm:ss, date only, UTC timestamps, datetime objects)
   - ✅ Fix bars_per_day logic to calculate end-of-day properly (modulo arithmetic)
5. ✅ Update tests to reflect new behavior
   - ✅ Added 3 new tests for bars_per_day logic
   - ✅ All 16 order expiry tests passing
6. ⏸️ Update documentation (IMPLEMENTATION.md, README.md)

**Success criteria**:
- ✅ Day orders automatically expire at end of trading day
- ✅ No manual bars_valid needed (or made optional with smart default)
- ✅ All tests passing (154 total, 152 passed, 2 skipped)
- ✅ No pandas dependency
- ✅ bars_per_day uses correct end-of-day calculation
- ⏸️ Documentation updated

**User Feedback (2026-02-03)**:
- Remove pandas import
- Standardize timestamp to yyyy-mm-dd hh:mm:ss (can be date only, UTC timestamp, or separate date+time columns)
- bars_per_day should calculate actual end-of-day bar, considering when during the day order was placed

---

## Dependencies

**Required**:
- Existing Order class in polarbtest/orders.py
- Existing TradeTracker in polarbtest/trades.py
- Portfolio class in polarbtest/core.py

**Optional**:
- None

---

## Risks & Challenges

1. **Order execution complexity**: Managing multiple order types and their interactions
2. **Trailing stop logic**: Ensuring trailing stops update correctly without bugs
3. **OCO implementation**: Proper linking and cancellation of related orders
4. **Test coverage**: Need comprehensive tests for edge cases
5. **Performance**: Ensure order checking doesn't slow down backtests

---

## Success Metrics

- All Phase 1.5 items checked off in TODO.md
- Test coverage for new features
- No regressions in existing tests
- Clean code (passes ruff and mypy)
- Clear documentation and examples

---

## Notes

- Follow existing code style in the project
- Maintain backwards compatibility
- Prioritize correctness over optimization initially
- Use ENHANCEMENT_ROADMAP.md as reference for implementation details

---

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| - | - | - |
