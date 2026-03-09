# Cleanup: OptimizeResult, potential issues, examples

**Date:** 2026-03-09

## Changes

1. **Fixed `prices` unbound variable** in `weight_backtest.py` — added default `prices: dict[str, float] = {}` before the loop so the variable is always bound even if `dates_in_order` is empty.

2. **Removed dict-like methods from `OptimizeResult`** — dropped `__getitem__`, `__contains__`, `get`, `keys`, `values`, `items`, and `_scalar_dict`. Access now exclusively via `.params` and `.metrics` attributes.

3. **Updated internal code** — `walk_forward_analysis()` in `runner.py` now accesses `OptimizeResult.params` and `OptimizeResult.metrics` directly instead of dict-style.

4. **Updated all examples** to use `best.params[...]` and `best.metrics.sharpe_ratio` instead of `best[...]`.

5. **Fixed win_rate formatting** in 4 examples — changed `{win_rate:.1f}%` to `{win_rate:.1%}` so it correctly displays e.g. `50.0%` instead of `0.5%`.

6. **Updated tests** that relied on dict-style `OptimizeResult` access.

## Files Modified

- `polarbt/results.py` — removed dict-like methods from `OptimizeResult`
- `polarbt/runner.py` — updated `walk_forward_analysis` and `optimize` docstring
- `polarbt/weight_backtest.py` — added `prices` default before loop
- `tests/test_optimization.py` — updated assertions
- `tests/test_runner.py` — updated assertions
- `examples/example.py` — use `.params` / `.metrics`
- `examples/example_walk_forward.py` — use `.params` / `.metrics`
- `examples/example_advanced_analysis.py` — use `.params` / `.metrics`
- `examples/example_sma_crossover_stoploss.py` — fix win_rate format
- `examples/example_rsi_bracket_orders.py` — fix win_rate format
- `examples/example_momentum_rotation.py` — fix win_rate format
- `examples/example_ml_strategy.py` — fix win_rate format
- `POTENTIAL_ISSUES.md` — cleared resolved issue
