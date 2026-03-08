# 07 - Optimizer Fixes

## Changes

### `optimize_bayesian` constraint handling (bug fix)
Replaced penalty-value approach (`999.0`/`-999.0`) with skopt's `Optimizer.ask/tell` pattern. Constrained points are now skipped entirely and never fed to the Gaussian Process surrogate model.

### `optimize_bayesian` parallelism (enhancement)
Added `n_jobs` parameter. Candidates are asked in batches and evaluated in parallel using `ProcessPoolExecutor` with spawn context, matching `backtest_batch()` behavior.

### `optimize()` return type (enhancement)
Returns `OptimizeResult` instead of flat dict. Separates `.params` from `.metrics` to prevent name collisions. Supports dict-style `result["key"]` access for backward compatibility.

### `optimize()` failed backtest filtering (bug fix)
Filters out failed backtests (`success=False`) before selecting the best result. Prevents sentinel `sharpe_ratio=-999.0` from being selected when minimizing.

### `optimize_bayesian` redundant re-run (fix)
Removed the redundant final `backtest()` call. Best result is now looked up from the already-evaluated results list.

## Files Modified
- `polarbt/results.py` - Added `OptimizeResult` class
- `polarbt/runner.py` - All five fixes
- `polarbt/__init__.py` - Export `OptimizeResult`
- `tests/test_runner.py` - Added tests for new behavior
