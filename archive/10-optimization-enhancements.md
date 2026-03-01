# Part 10: Optimization Enhancements

## Summary

Added five optimization enhancements to the backtesting engine.

## Changes

### runner.py
- **Constraint functions**: `optimize()` accepts `constraint` callable to filter invalid parameter combinations before running backtests (e.g., `constraint=lambda p: p["fast"] < p["slow"]`)
- **Multi-objective optimization**: `optimize_multi()` runs grid search and returns Pareto-optimal (non-dominated) parameter combinations across multiple objectives with configurable maximize/minimize per objective
- **Bayesian optimization**: `optimize_bayesian()` uses scikit-optimize's Gaussian Process to efficiently explore continuous parameter spaces with far fewer evaluations than grid search (optional dependency)
- Internal refactoring: extracted `_collect_engine_kwargs()` and `_generate_param_sets()` helpers to reduce duplication

### plotting/charts.py
- **`plot_sensitivity()`**: Parameter sensitivity chart showing metric vs single parameter with individual data points and mean line
- **`plot_param_heatmap()`**: 2D heatmap showing metric values across two parameters with configurable aggregation (mean/max/min)

### __init__.py, plotting/__init__.py
- Exported new functions: `optimize`, `optimize_multi`, `optimize_bayesian`, `plot_sensitivity`, `plot_param_heatmap`

### Tests
- 23 new tests in `test_optimization.py` covering constraints, Pareto front computation, multi-objective optimization, sensitivity plots, and heatmaps

## Files Changed
- `polarbt/runner.py`
- `polarbt/plotting/charts.py`
- `polarbt/plotting/__init__.py`
- `polarbt/__init__.py`
- `tests/test_optimization.py` (new)
- `DESCRIPTION.md`
- `PLAN.md`
