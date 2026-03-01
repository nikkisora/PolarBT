# 09 — Data Utilities

## Summary

Implemented `polarbt/data/` module with validation, cleaning, and resampling utilities for OHLCV DataFrames.

## Changes

### New Files
- `polarbt/data/__init__.py` — public exports
- `polarbt/data/validation.py` — `validate()`, `validate_columns()`, `validate_dtypes()`, `validate_timestamps()`, `validate_ohlc_integrity()`, `validate_no_nulls()`, `validate_no_negative_prices()`, `ValidationResult` dataclass
- `polarbt/data/cleaning.py` — `fill_gaps()`, `adjust_splits()`, `drop_zero_volume()`, `clip_outliers()`
- `polarbt/data/resampling.py` — `resample_ohlcv()` with standard OHLCV aggregation logic
- `tests/test_data_utils.py` — 46 tests covering all functions

### Modified Files
- `polarbt/__init__.py` — added `data` module export
- `PLAN.md` — marked section 9 complete
- `DESCRIPTION.md` — added Data Utilities documentation, updated module structure and test count
