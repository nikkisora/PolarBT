# Phase 1.5 Enhancement: Automatic Day Order Expiry

**Date**: 2026-02-03  
**Status**: Complete  
**Time**: 55 minutes

---

## Overview

Enhanced the day order functionality to automatically calculate expiration times based on trading session boundaries, eliminating the need to manually specify `bars_valid` in most cases.

---

## Problem Statement

Previously, `order_day()` required users to manually specify `bars_valid`, which was:
- **Not intuitive**: Users had to know how many bars constitute a "trading day"
- **Timeframe dependent**: Different for 1-min (390 bars), 1-hour (7.5 bars), daily (1 bar) data
- **Error-prone**: Easy to miscalculate, especially mid-day

**Old behavior**:
```python
# User had to manually calculate bars per day
portfolio.order_day("BTC", 0.1, limit_price=50000, bars_valid=390)
```

---

## Solution

Implemented automatic day detection using three strategies (in priority order):

### 1. Timestamp-based Detection (Automatic)
When timestamps are available, orders automatically expire when the date changes.

```python
# Timestamps provided - auto-detects day boundary
portfolio.order_day("BTC", 0.1, limit_price=50000)
# Order expires at end of current day (when date changes)
```

**Supported timestamp formats**:
- Python `datetime` objects: `datetime(2024, 1, 15, 9, 30)`
- Unix timestamps: `1705305000` (seconds or milliseconds)
- String formats: `"2024-01-15 09:30:00"`, `"2024-01-15"`, `"2024-01-15T09:30:00"`

### 2. bars_per_day Configuration (Semi-automatic)
Portfolio can be initialized with `bars_per_day` for consistent day length.

```python
# Configure portfolio with bars per day
portfolio = Portfolio(initial_cash=10000, bars_per_day=390)  # 1-min bars

# Day orders automatically expire at end of current day
portfolio.order_day("BTC", 0.1, limit_price=50000)
```

**Smart end-of-day calculation**:
- If order placed at bar 0 → expires at bar 389 (end of day 0)
- If order placed at bar 100 → expires at bar 389 (end of same day)
- If order placed at bar 450 → expires at bar 779 (end of day 1)

### 3. Manual Override (Explicit)
Users can still explicitly specify `bars_valid` for full control.

```python
# Explicit control
portfolio.order_day("BTC", 0.1, limit_price=50000, bars_valid=5)
```

---

## Implementation Details

### Modified Files

**polarbtest/orders.py**:
- Added `expiry_date: Any` field to Order class for timestamp-based expiry

**polarbtest/core.py**:
- Added `_extract_date()` helper function (stdlib only, no pandas)
- Added `bars_per_day: int | None` parameter to Portfolio.__init__()
- Enhanced `order_day()` with hybrid expiry logic
- Updated `_check_order_expiry()` to handle both date-based and bar-based expiry

**tests/test_order_expiry.py**:
- Added 9 new comprehensive tests
- All 16 tests passing (7 original + 9 new)

### Key Algorithm: Smart bars_per_day Calculation

```python
# Calculate which bar within the current trading day
current_bar_in_day = self._current_bar % self.bars_per_day

# Calculate bars remaining until end of current day
bars_until_eod = self.bars_per_day - current_bar_in_day - 1

# Order expires at end of current trading day
order.valid_until = self._current_bar + bars_until_eod
```

**Example** (bars_per_day=390):
- Bar 0: expires at bar 389 (0 + 389 = 389)
- Bar 100: expires at bar 389 (100 + 289 = 389)
- Bar 450: expires at bar 779 (450 + 329 = 779)

---

## Testing

### Test Coverage

**16 total order expiry tests** (all passing):
1. Original tests (7) - backwards compatibility
2. Datetime-based expiry (2)
3. Unix timestamp expiry (1)
4. bars_per_day logic (3) ← NEW
5. Override and edge cases (3)

### Code Quality

```
pytest: 152 passed, 2 skipped
mypy: Success (0 errors)
ruff: No errors (8 minor style suggestions)
```

---

## Usage Examples

### Example 1: Timestamp-based (Recommended)

```python
from datetime import datetime
from polarbtest import Portfolio

portfolio = Portfolio(initial_cash=10000)

# Day 1
timestamp = datetime(2024, 1, 15, 9, 30)
portfolio.update_prices({"BTC": 50000}, bar_index=0, timestamp=timestamp)
portfolio.order_day("BTC", 0.1, limit_price=49000)
# ✓ Automatically expires when date changes to 2024-01-16
```

### Example 2: bars_per_day Configuration

```python
# For 1-minute bars (6.5 hour session = 390 bars)
portfolio = Portfolio(initial_cash=10000, bars_per_day=390)

# No timestamps needed
portfolio.update_prices({"BTC": 50000}, bar_index=100)
portfolio.order_day("BTC", 0.1, limit_price=49000)
# ✓ Automatically expires at bar 389 (end of day 0)
```

### Example 3: Manual Override

```python
portfolio = Portfolio(initial_cash=10000, bars_per_day=390)

# Override automatic calculation
portfolio.order_day("BTC", 0.1, limit_price=49000, bars_valid=5)
# ✓ Expires after exactly 5 bars (manual control)
```

---

## Benefits

1. **Intuitive**: "Day order" behaves like a real day order
2. **Flexible**: Works with timestamps, configuration, or manual override
3. **Correct**: Smart calculation ensures orders expire at actual day boundaries
4. **Compatible**: All existing code continues to work (backwards compatible)
5. **Simple**: No external dependencies (stdlib only)

---

## Migration Guide

### No changes needed for existing code

All existing code continues to work unchanged:

```python
# Old code (still works)
portfolio.order_day("BTC", 0.1, limit_price=50000, bars_valid=1)
```

### Optional: Enable automatic expiry

**Option 1**: Add timestamps to your data (recommended)
```python
# Just add timestamp parameter - that's it!
portfolio.update_prices(prices, bar_index=i, timestamp=df["timestamp"][i])
```

**Option 2**: Configure bars_per_day
```python
# One-time configuration
portfolio = Portfolio(initial_cash=10000, bars_per_day=390)
```

---

## Technical Notes

### Timestamp Parsing

The `_extract_date()` function handles:
- Python datetime objects (standard library)
- Unix timestamps (int/float, auto-detects seconds vs milliseconds)
- String formats: "yyyy-mm-dd hh:mm:ss", "yyyy-mm-dd", ISO with 'T'
- Other types via str() conversion (Polars datetime, etc.)
- No pandas dependency (removed for simplicity)

### Day Boundary Detection

**Timestamp-based**: Date comparison
```python
if current_date > order.expiry_date:
    order.mark_expired()
```

**Bar-based**: Bar index comparison
```python
if current_bar > order.valid_until:
    order.mark_expired()
```

### Priority Order

1. `bars_valid` parameter (if specified)
2. Timestamp-based day detection (if timestamp available)
3. `bars_per_day` configuration (if set)
4. Default to 1 bar (fallback)

---

## Future Enhancements

Potential improvements for future phases:
- Support for extended trading hours
- Different day definitions (calendar day vs trading session)
- Multi-session support (pre-market, regular, after-hours)
- Week/month order expiry similar to day orders

---

## Summary

Successfully enhanced day orders to automatically calculate expiration based on trading session boundaries. The implementation is:
- ✅ Automatic (timestamp-based or configured)
- ✅ Correct (smart end-of-day calculation)
- ✅ Simple (no external dependencies)
- ✅ Compatible (backwards compatible)
- ✅ Tested (16 comprehensive tests)

Phase 1.5 day order enhancement is complete and ready for production use.
