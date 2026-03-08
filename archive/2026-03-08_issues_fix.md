# Fix: MOEX M5 Integration Issues

Fixed three critical issues discovered during integration testing with MOEX M5 data (249 stocks, ~100K bars each).

## Changes

### Issue 1: Order dict corruption on micro-price stocks (Critical)

- `_check_order_expiry()`: snapshot `self.orders` via `list(self.orders.values())` before iterating to prevent dict-during-iteration corruption
- Added `_purge_inactive_orders()` to Portfolio: removes filled/cancelled/rejected/expired orders when dict exceeds 1000 entries, preventing unbounded growth
- Called automatically from `update_prices()` when threshold is exceeded

### Issue 2: Segfault on sequential backtests (High)

- Added `Engine.cleanup()` method: explicitly releases portfolio, processed data, trade history, and calls `gc.collect()` to free native Arrow buffers
- `Engine.run()` now clears previous run state at start to handle re-runs
- `runner.backtest()` calls `engine.cleanup()` after extracting results to prevent memory accumulation across sequential calls

### Issue 3: Concurrent dict mutation in stop-loss fill path (Medium)

- Refactored `_check_stops_with_priority()` to two-phase approach:
  - Phase 1: Detect all stop/TP/trailing stop triggers (read-only on `self.orders`)
  - Phase 2: Execute all closes (mutates `self.orders`)
- This cleanly separates read from write, eliminating mutation during detection

### Misc

- Fixed pre-existing mypy errors for skopt imports in `runner.py`

## Files Changed

- `polarbt/core.py` — Portfolio purging, dict snapshot, two-phase stops, Engine.cleanup
- `polarbt/runner.py` — cleanup in backtest(), mypy fix
- `tests/test_issues_fix.py` — 17 new tests covering all three issues
