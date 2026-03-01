"""
Runner utilities for backtesting and parallel execution.

This module provides high-level functions for running backtests,
optimized for evolutionary search and parameter optimization.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import polars as pl

from polarbtest.commissions import CommissionModel
from polarbtest.core import Engine, Strategy


@dataclass
class BacktestResult:
    """
    Container for backtest results.

    Attributes:
        params: Strategy parameters used
        metrics: Performance metrics dictionary
        success: Whether backtest completed successfully
        error: Error message if backtest failed
    """

    params: dict[str, Any]
    metrics: dict[str, Any]
    success: bool = True
    error: str | None = None


def backtest(
    strategy_class: type[Strategy],
    data: pl.DataFrame | dict[str, pl.DataFrame],
    params: dict[str, Any] | None = None,
    initial_cash: float = 100_000.0,
    commission: float | tuple[float, float] | CommissionModel = 0.001,
    slippage: float = 0.0005,
    price_columns: dict[str, str] | None = None,
    warmup: int | str = "auto",
    order_delay: int = 0,
    borrow_rate: float = 0.0,
    bars_per_day: float | None = None,
    max_position_size: float | None = None,
    max_total_exposure: float | None = None,
    max_drawdown_stop: float | None = None,
    daily_loss_limit: float | None = None,
    leverage: float = 1.0,
    maintenance_margin: float | None = None,
) -> dict[str, Any]:
    """
    Run a single backtest.

    This is the main entry point for backtesting a strategy with given parameters.
    Designed for easy integration with LLM-driven optimization.

    Args:
        strategy_class: Strategy class (not instance)
        data: Polars DataFrame with price data OR dict mapping asset names to DataFrames
        params: Dictionary of strategy parameters (default None)
        initial_cash: Starting capital (default 100,000)
        commission: Commission as a percentage (e.g., 0.001 = 0.1%) or tuple of (fixed_commission, percent_commission)
                   For example: 0.001 means 0.1% per trade, (5.0, 0.001) means $5 + 0.1% per trade
        slippage: Slippage rate as fraction (default 0.0005 = 0.05%)
        price_columns: Dict mapping asset names to price columns
        warmup: Number of bars to skip before executing strategy, or "auto" to automatically
               detect when all indicators are ready (default "auto")
        order_delay: Number of bars to delay order execution (default 0)
        borrow_rate: Annual borrow rate for short positions (default 0.0)
        bars_per_day: Number of bars in a trading day (default None)
        max_position_size: Maximum single position size as fraction of portfolio value (default None)
        max_total_exposure: Maximum total exposure as fraction of portfolio value (default None)
        max_drawdown_stop: Maximum drawdown before halting trading (default None)
        daily_loss_limit: Maximum daily loss before halting trading for the day (default None)
        leverage: Maximum leverage multiplier (default 1.0)
        maintenance_margin: Minimum margin ratio before margin call (default None)

    Returns:
        Dictionary containing backtest results and metrics

    Example:
        # Single asset with auto warmup (default)
        results = backtest(
            MyStrategy,
            data,
            params={"sma_period": 20, "rsi_period": 14},
            initial_cash=100000
        )

        # Multi-asset with manual warmup
        results = backtest(
            MyStrategy,
            {"BTC": btc_df, "ETH": eth_df},
            params={"fast": 10, "slow": 20},
            warmup=20  # or warmup="auto" (default)
        )
        print(f"Sharpe Ratio: {results['sharpe_ratio']}")
    """
    if params is None:
        params = {}

    try:
        # Instantiate strategy with parameters
        strategy = strategy_class(**params)

        # Create and run engine
        engine = Engine(
            strategy=strategy,
            data=data,
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
            price_columns=price_columns,
            warmup=warmup,
            order_delay=order_delay,
            borrow_rate=borrow_rate,
            bars_per_day=bars_per_day,
            max_position_size=max_position_size,
            max_total_exposure=max_total_exposure,
            max_drawdown_stop=max_drawdown_stop,
            daily_loss_limit=daily_loss_limit,
            leverage=leverage,
            maintenance_margin=maintenance_margin,
        )

        results = engine.run()
        results["params"] = params
        results["success"] = True

        return results

    except Exception as e:
        import traceback

        return {
            "params": params,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "sharpe_ratio": -999.0,  # Penalty for failed backtests
            "total_return": -1.0,
        }


def _run_backtest_worker(
    args: tuple[
        type[Strategy],
        pl.DataFrame | dict[str, pl.DataFrame],
        dict[str, Any],
        float,
        float | tuple[float, float] | CommissionModel,
        float,
        dict[str, str] | None,
        int | str,
        int,
        float,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        float,
        float | None,
    ],
) -> BacktestResult:
    """
    Worker function for parallel backtest execution.

    Args:
        args: Tuple of (strategy_class, data, params, initial_cash, commission, slippage,
              price_columns, warmup, order_delay, borrow_rate, bars_per_day,
              max_position_size, max_total_exposure, max_drawdown_stop, daily_loss_limit,
              leverage, maintenance_margin)

    Returns:
        BacktestResult object
    """
    (
        strategy_class,
        data,
        params,
        initial_cash,
        commission,
        slippage,
        price_columns,
        warmup,
        order_delay,
        borrow_rate,
        bars_per_day,
        max_position_size,
        max_total_exposure,
        max_drawdown_stop,
        daily_loss_limit,
        leverage,
        maintenance_margin,
    ) = args

    try:
        results = backtest(
            strategy_class=strategy_class,
            data=data,
            params=params,
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
            price_columns=price_columns,
            warmup=warmup,
            order_delay=order_delay,
            borrow_rate=borrow_rate,
            bars_per_day=bars_per_day,
            max_position_size=max_position_size,
            max_total_exposure=max_total_exposure,
            max_drawdown_stop=max_drawdown_stop,
            daily_loss_limit=daily_loss_limit,
            leverage=leverage,
            maintenance_margin=maintenance_margin,
        )

        return BacktestResult(
            params=params,
            metrics=results,
            success=True,
        )

    except Exception as e:
        return BacktestResult(
            params=params,
            metrics={},
            success=False,
            error=str(e),
        )


def backtest_batch(
    strategy_class: type[Strategy],
    data: pl.DataFrame | dict[str, pl.DataFrame],
    param_sets: list[dict[str, Any]],
    initial_cash: float = 100_000.0,
    commission: float | tuple[float, float] | CommissionModel = 0.001,
    slippage: float = 0.0005,
    price_columns: dict[str, str] | None = None,
    warmup: int | str = "auto",
    order_delay: int = 0,
    n_jobs: int | None = None,
    verbose: bool = True,
    borrow_rate: float = 0.0,
    bars_per_day: float | None = None,
    max_position_size: float | None = None,
    max_total_exposure: float | None = None,
    max_drawdown_stop: float | None = None,
    daily_loss_limit: float | None = None,
    leverage: float = 1.0,
    maintenance_margin: float | None = None,
) -> pl.DataFrame:
    """
    Run multiple backtests in parallel.

    This function is optimized for evolutionary search, allowing you to test
    many parameter combinations across all CPU cores.

    Args:
        strategy_class: Strategy class (not instance)
        data: Polars DataFrame with price data
        param_sets: List of parameter dictionaries to test
        initial_cash: Starting capital
        commission: Commission as a percentage or tuple of (fixed_commission, percent_commission)
        slippage: Slippage rate
        price_columns: Dict mapping asset names to price columns
        warmup: Number of bars to skip or "auto" (default "auto")
        order_delay: Number of bars to delay order execution (default 0)
        n_jobs: Number of parallel jobs (default: all CPUs)
        verbose: Print progress (default True)
        borrow_rate: Annual borrow rate for short positions (default 0.0)
        bars_per_day: Number of bars in a trading day (default None)
        max_position_size: Maximum single position size as fraction of portfolio value (default None)
        max_total_exposure: Maximum total exposure as fraction of portfolio value (default None)
        max_drawdown_stop: Maximum drawdown before halting trading (default None)
        daily_loss_limit: Maximum daily loss before halting for the day (default None)
        leverage: Maximum leverage multiplier (default 1.0)
        maintenance_margin: Minimum margin ratio before margin call (default None)

    Returns:
        Polars DataFrame with results for each parameter set

    Example:
        param_sets = [
            {"sma_period": 10, "rsi_period": 14},
            {"sma_period": 20, "rsi_period": 14},
            {"sma_period": 50, "rsi_period": 21},
        ]

        results_df = backtest_batch(MyStrategy, data, param_sets)
        best = results_df.sort("sharpe_ratio", descending=True).head(1)
        print(best)
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()

    if verbose:
        print(f"Running {len(param_sets)} backtests on {n_jobs} cores...")

    # Prepare arguments for workers
    args_list = [
        (
            strategy_class,
            data,
            params,
            initial_cash,
            commission,
            slippage,
            price_columns,
            warmup,
            order_delay,
            borrow_rate,
            bars_per_day,
            max_position_size,
            max_total_exposure,
            max_drawdown_stop,
            daily_loss_limit,
            leverage,
            maintenance_margin,
        )
        for params in param_sets
    ]

    results = []

    if n_jobs == 1:
        # Sequential execution — avoids fork-safety issues with Polars on Linux
        for i, args in enumerate(args_list, 1):
            result = _run_backtest_worker(args)
            results.append(result)

            if verbose and i % max(1, len(param_sets) // 10) == 0:
                print(f"  Progress: {i}/{len(param_sets)} ({100 * i // len(param_sets)}%)")
    else:
        # Use spawn context to avoid Polars fork deadlocks on Linux
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as executor:
            futures = {executor.submit(_run_backtest_worker, args): i for i, args in enumerate(args_list)}

            for completed, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)

                if verbose and completed % max(1, len(param_sets) // 10) == 0:
                    print(f"  Progress: {completed}/{len(param_sets)} ({100 * completed // len(param_sets)}%)")

    if verbose:
        print(f"Completed {len(results)} backtests")

    # Convert results to DataFrame
    rows = []
    for result in results:
        row = {**result.params}
        if result.success:
            row.update(result.metrics)
        else:
            row["error"] = result.error
            row["success"] = False
        rows.append(row)

    return pl.DataFrame(rows)


def _collect_engine_kwargs(
    initial_cash: float,
    commission: float | tuple[float, float] | CommissionModel,
    slippage: float,
    price_columns: dict[str, str] | None,
    warmup: int | str,
    order_delay: int,
    borrow_rate: float,
    bars_per_day: float | None,
    max_position_size: float | None,
    max_total_exposure: float | None,
    max_drawdown_stop: float | None,
    daily_loss_limit: float | None,
    leverage: float,
    maintenance_margin: float | None,
) -> dict[str, Any]:
    """Collect common engine keyword arguments into a dictionary."""
    return {
        "initial_cash": initial_cash,
        "commission": commission,
        "slippage": slippage,
        "price_columns": price_columns,
        "warmup": warmup,
        "order_delay": order_delay,
        "borrow_rate": borrow_rate,
        "bars_per_day": bars_per_day,
        "max_position_size": max_position_size,
        "max_total_exposure": max_total_exposure,
        "max_drawdown_stop": max_drawdown_stop,
        "daily_loss_limit": daily_loss_limit,
        "leverage": leverage,
        "maintenance_margin": maintenance_margin,
    }


def _generate_param_sets(
    param_grid: dict[str, list[Any]],
    constraint: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Generate parameter combinations from a grid, optionally filtered by a constraint.

    Args:
        param_grid: Dictionary mapping parameter names to lists of values.
        constraint: Optional callable that takes a params dict and returns True if valid.

    Returns:
        List of parameter dictionaries.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    param_sets = [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]

    if constraint is not None:
        param_sets = [p for p in param_sets if constraint(p)]

    return param_sets


def optimize(
    strategy_class: type[Strategy],
    data: pl.DataFrame | dict[str, pl.DataFrame],
    param_grid: dict[str, list[Any]],
    objective: str = "sharpe_ratio",
    maximize: bool = True,
    constraint: Callable[[dict[str, Any]], bool] | None = None,
    initial_cash: float = 100_000.0,
    commission: float | tuple[float, float] | CommissionModel = 0.001,
    slippage: float = 0.0005,
    price_columns: dict[str, str] | None = None,
    warmup: int | str = "auto",
    order_delay: int = 0,
    n_jobs: int | None = None,
    verbose: bool = True,
    borrow_rate: float = 0.0,
    bars_per_day: float | None = None,
    max_position_size: float | None = None,
    max_total_exposure: float | None = None,
    max_drawdown_stop: float | None = None,
    daily_loss_limit: float | None = None,
    leverage: float = 1.0,
    maintenance_margin: float | None = None,
) -> dict[str, Any]:
    """Grid search optimization for strategy parameters.

    Args:
        strategy_class: Strategy class.
        data: Price data.
        param_grid: Dictionary mapping parameter names to lists of values.
        objective: Metric to optimize (default "sharpe_ratio").
        maximize: Whether to maximize objective (default True).
        constraint: Optional callable that takes a params dict and returns True
            if the combination is valid. Invalid combinations are skipped.
            Example: ``lambda p: p["fast"] < p["slow"]``
        initial_cash: Starting capital.
        commission: Commission as a percentage or tuple of (fixed_commission, percent_commission).
        slippage: Slippage rate.
        price_columns: Asset price columns.
        warmup: Number of bars to skip or "auto" (default "auto").
        order_delay: Number of bars to delay order execution (default 0).
        n_jobs: Number of parallel jobs.
        verbose: Print progress.
        borrow_rate: Annual borrow rate for short positions (default 0.0).
        bars_per_day: Number of bars in a trading day (default None).
        max_position_size: Maximum single position size as fraction of portfolio value (default None).
        max_total_exposure: Maximum total exposure as fraction of portfolio value (default None).
        max_drawdown_stop: Maximum drawdown before halting trading (default None).
        daily_loss_limit: Maximum daily loss before halting for the day (default None).
        leverage: Maximum leverage multiplier (default 1.0).
        maintenance_margin: Minimum margin ratio before margin call (default None).

    Returns:
        Dictionary with best parameters and results.

    Example:
        >>> param_grid = {"fast": [5, 10, 20], "slow": [20, 50, 100]}
        >>> best = optimize(
        ...     MyStrategy, data, param_grid,
        ...     constraint=lambda p: p["fast"] < p["slow"],
        ... )
    """
    param_sets = _generate_param_sets(param_grid, constraint)

    if not param_sets:
        raise ValueError("No parameter combinations remain after applying constraint")

    if verbose:
        total = 1
        for v in param_grid.values():
            total *= len(v)
        skipped = total - len(param_sets)
        msg = f"Testing {len(param_sets)} parameter combinations"
        if skipped > 0:
            msg += f" ({skipped} filtered by constraint)"
        print(msg + "...")

    results_df = backtest_batch(
        strategy_class=strategy_class,
        data=data,
        param_sets=param_sets,
        n_jobs=n_jobs,
        verbose=verbose,
        **_collect_engine_kwargs(
            initial_cash,
            commission,
            slippage,
            price_columns,
            warmup,
            order_delay,
            borrow_rate,
            bars_per_day,
            max_position_size,
            max_total_exposure,
            max_drawdown_stop,
            daily_loss_limit,
            leverage,
            maintenance_margin,
        ),
    )

    if objective not in results_df.columns:
        raise ValueError(f"Objective '{objective}' not found in results")

    best_row = results_df.sort(objective, descending=maximize).head(1)

    return best_row.to_dicts()[0]


def optimize_multi(
    strategy_class: type[Strategy],
    data: pl.DataFrame | dict[str, pl.DataFrame],
    param_grid: dict[str, list[Any]],
    objectives: list[str],
    maximize: list[bool] | None = None,
    constraint: Callable[[dict[str, Any]], bool] | None = None,
    initial_cash: float = 100_000.0,
    commission: float | tuple[float, float] | CommissionModel = 0.001,
    slippage: float = 0.0005,
    price_columns: dict[str, str] | None = None,
    warmup: int | str = "auto",
    order_delay: int = 0,
    n_jobs: int | None = None,
    verbose: bool = True,
    borrow_rate: float = 0.0,
    bars_per_day: float | None = None,
    max_position_size: float | None = None,
    max_total_exposure: float | None = None,
    max_drawdown_stop: float | None = None,
    daily_loss_limit: float | None = None,
    leverage: float = 1.0,
    maintenance_margin: float | None = None,
) -> pl.DataFrame:
    """Multi-objective optimization returning the Pareto front.

    Runs a grid search and returns only the non-dominated (Pareto-optimal)
    parameter combinations across all specified objectives.

    Args:
        strategy_class: Strategy class.
        data: Price data.
        param_grid: Dictionary mapping parameter names to lists of values.
        objectives: List of metric names to optimize simultaneously.
        maximize: Per-objective direction (default: all True). Must match length of objectives.
        constraint: Optional callable to filter parameter combinations.
        initial_cash: Starting capital.
        commission: Commission rate.
        slippage: Slippage rate.
        price_columns: Asset price columns.
        warmup: Warmup setting.
        order_delay: Order delay in bars.
        n_jobs: Number of parallel jobs.
        verbose: Print progress.
        borrow_rate: Annual borrow rate for short positions.
        bars_per_day: Bars per trading day.
        max_position_size: Max single position size.
        max_total_exposure: Max total exposure.
        max_drawdown_stop: Max drawdown stop.
        daily_loss_limit: Daily loss limit.
        leverage: Leverage multiplier.
        maintenance_margin: Maintenance margin ratio.

    Returns:
        DataFrame containing only Pareto-optimal rows with all metrics.

    Example:
        >>> pareto = optimize_multi(
        ...     MyStrategy, data,
        ...     param_grid={"sma_period": [5, 10, 20]},
        ...     objectives=["sharpe_ratio", "max_drawdown"],
        ...     maximize=[True, False],
        ... )
    """
    if len(objectives) < 2:
        raise ValueError("optimize_multi requires at least 2 objectives")

    if maximize is None:
        maximize = [True] * len(objectives)

    if len(maximize) != len(objectives):
        raise ValueError("maximize list must match length of objectives")

    param_sets = _generate_param_sets(param_grid, constraint)

    if not param_sets:
        raise ValueError("No parameter combinations remain after applying constraint")

    if verbose:
        print(f"Testing {len(param_sets)} parameter combinations for {len(objectives)} objectives...")

    results_df = backtest_batch(
        strategy_class=strategy_class,
        data=data,
        param_sets=param_sets,
        n_jobs=n_jobs,
        verbose=verbose,
        **_collect_engine_kwargs(
            initial_cash,
            commission,
            slippage,
            price_columns,
            warmup,
            order_delay,
            borrow_rate,
            bars_per_day,
            max_position_size,
            max_total_exposure,
            max_drawdown_stop,
            daily_loss_limit,
            leverage,
            maintenance_margin,
        ),
    )

    for obj in objectives:
        if obj not in results_df.columns:
            raise ValueError(f"Objective '{obj}' not found in results")

    # Compute Pareto front
    pareto_mask = _pareto_front(results_df, objectives, maximize)
    return results_df.filter(pl.Series(pareto_mask))


def _pareto_front(
    df: pl.DataFrame,
    objectives: list[str],
    maximize: list[bool],
) -> list[bool]:
    """Compute Pareto-optimal mask for a DataFrame.

    A row is Pareto-optimal if no other row is strictly better in all objectives.
    """
    n = len(df)
    if n == 0:
        return []

    # Extract objective values, flipping sign for minimization so we always maximize
    values: list[list[float]] = []
    for i, obj in enumerate(objectives):
        col = df[obj].to_list()
        sign = 1.0 if maximize[i] else -1.0
        values.append([float(v) * sign if v is not None else float("-inf") for v in col])

    mask = [True] * n
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            # Check if j dominates i (j >= i in all, j > i in at least one)
            all_ge = True
            any_gt = False
            for k in range(len(objectives)):
                if values[k][j] < values[k][i]:
                    all_ge = False
                    break
                if values[k][j] > values[k][i]:
                    any_gt = True
            if all_ge and any_gt:
                mask[i] = False
                break

    return mask


def optimize_bayesian(
    strategy_class: type[Strategy],
    data: pl.DataFrame | dict[str, pl.DataFrame],
    param_space: dict[str, tuple[float, float]],
    objective: str = "sharpe_ratio",
    maximize: bool = True,
    n_calls: int = 50,
    n_initial_points: int = 10,
    constraint: Callable[[dict[str, Any]], bool] | None = None,
    initial_cash: float = 100_000.0,
    commission: float | tuple[float, float] | CommissionModel = 0.001,
    slippage: float = 0.0005,
    price_columns: dict[str, str] | None = None,
    warmup: int | str = "auto",
    order_delay: int = 0,
    verbose: bool = True,
    borrow_rate: float = 0.0,
    bars_per_day: float | None = None,
    max_position_size: float | None = None,
    max_total_exposure: float | None = None,
    max_drawdown_stop: float | None = None,
    daily_loss_limit: float | None = None,
    leverage: float = 1.0,
    maintenance_margin: float | None = None,
) -> dict[str, Any]:
    """Bayesian optimization for strategy parameters using scikit-optimize.

    Requires scikit-optimize: ``pip install scikit-optimize``

    Unlike grid search, this explores the parameter space efficiently using
    a Gaussian Process surrogate model, requiring far fewer evaluations.

    Args:
        strategy_class: Strategy class.
        data: Price data.
        param_space: Dictionary mapping parameter names to (min, max) tuples.
            Integer ranges are inferred when both bounds are ints.
        objective: Metric to optimize (default "sharpe_ratio").
        maximize: Whether to maximize objective (default True).
        n_calls: Total number of evaluations (default 50).
        n_initial_points: Number of random initial points (default 10).
        constraint: Optional callable to reject parameter combinations.
        initial_cash: Starting capital.
        commission: Commission rate.
        slippage: Slippage rate.
        price_columns: Asset price columns.
        warmup: Warmup setting.
        order_delay: Order delay in bars.
        verbose: Print progress.
        borrow_rate: Annual borrow rate for short positions.
        bars_per_day: Bars per trading day.
        max_position_size: Max single position size.
        max_total_exposure: Max total exposure.
        max_drawdown_stop: Max drawdown stop.
        daily_loss_limit: Daily loss limit.
        leverage: Leverage multiplier.
        maintenance_margin: Maintenance margin ratio.

    Returns:
        Dictionary with best parameters and results, plus ``all_results`` DataFrame.

    Example:
        >>> best = optimize_bayesian(
        ...     MyStrategy, data,
        ...     param_space={"sma_period": (5, 50), "rsi_period": (7, 28)},
        ...     n_calls=30,
        ... )
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Integer, Real
    except ImportError:
        raise ImportError(
            "scikit-optimize is required for Bayesian optimization. Install it with: pip install scikit-optimize"
        ) from None

    keys = list(param_space.keys())
    dimensions = []
    for key in keys:
        low, high = param_space[key]
        if isinstance(low, int) and isinstance(high, int):
            dimensions.append(Integer(low, high, name=key))
        else:
            dimensions.append(Real(float(low), float(high), name=key))

    engine_kwargs = _collect_engine_kwargs(
        initial_cash,
        commission,
        slippage,
        price_columns,
        warmup,
        order_delay,
        borrow_rate,
        bars_per_day,
        max_position_size,
        max_total_exposure,
        max_drawdown_stop,
        daily_loss_limit,
        leverage,
        maintenance_margin,
    )

    all_results: list[dict[str, Any]] = []

    def objective_func(x: list[Any]) -> float:
        params = dict(zip(keys, x, strict=True))

        if constraint is not None and not constraint(params):
            return 999.0 if maximize else -999.0

        result = backtest(
            strategy_class=strategy_class,
            data=data,
            params=params,
            **engine_kwargs,
        )
        all_results.append(result)

        val = result.get(objective, -999.0 if maximize else 999.0)
        if not isinstance(val, (int, float)):
            val = -999.0 if maximize else 999.0

        return -float(val) if maximize else float(val)

    opt_result = gp_minimize(
        objective_func,
        dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        verbose=verbose,
    )

    best_params = dict(zip(keys, opt_result.x, strict=True))

    # Run final backtest with best params to get full results
    best_result = backtest(
        strategy_class=strategy_class,
        data=data,
        params=best_params,
        **engine_kwargs,
    )
    best_result["all_results"] = pl.DataFrame(all_results) if all_results else pl.DataFrame()

    return best_result


def walk_forward_analysis(
    strategy_class: type[Strategy],
    data: pl.DataFrame | dict[str, pl.DataFrame],
    param_grid: dict[str, list[Any]],
    train_periods: int,
    test_periods: int,
    objective: str = "sharpe_ratio",
    maximize: bool = True,
    initial_cash: float = 100_000.0,
    commission: float | tuple[float, float] | CommissionModel = 0.001,
    slippage: float = 0.0005,
    price_columns: dict[str, str] | None = None,
    warmup: int | str = "auto",
    order_delay: int = 0,
    anchored: bool = False,
    verbose: bool = True,
    n_jobs: int | None = None,
    borrow_rate: float = 0.0,
    bars_per_day: float | None = None,
    max_position_size: float | None = None,
    max_total_exposure: float | None = None,
    max_drawdown_stop: float | None = None,
    daily_loss_limit: float | None = None,
    leverage: float = 1.0,
    maintenance_margin: float | None = None,
) -> pl.DataFrame:
    """
    Perform walk-forward analysis.

    Args:
        strategy_class: Strategy class
        data: Price data (must have timestamp column)
        param_grid: Parameter grid for optimization
        train_periods: Number of periods for training
        test_periods: Number of periods for testing
        objective: Metric to optimize
        maximize: Whether to maximize objective
        initial_cash: Starting capital
        commission: Commission as a percentage or tuple of (fixed_commission, percent_commission)
        slippage: Slippage rate
        price_columns: Asset price columns
        warmup: Number of bars to skip or "auto" (default "auto")
        order_delay: Number of bars to delay order execution (default 0)
        anchored: Use anchored walk-forward (default False)
        verbose: Print progress
        n_jobs: Number of parallel jobs for optimization (default None = all CPUs)
        borrow_rate: Annual borrow rate for short positions (default 0.0)
        bars_per_day: Number of bars in a trading day (default None)

    Returns:
        DataFrame with walk-forward results

    Example:
        # Train on 252 days, test on 63 days
        wf_results = walk_forward_analysis(
            MyStrategy,
            data,
            param_grid={"sma_period": [10, 20, 50]},
            train_periods=252,
            test_periods=63
        )
    """
    # Get total periods based on data type
    total_periods = len(next(iter(data.values()))) if isinstance(data, dict) else len(data)

    results = []

    start_idx = 0
    fold = 0

    while start_idx + train_periods + test_periods <= total_periods:
        fold += 1

        train_start = 0 if anchored else start_idx

        train_end = start_idx + train_periods
        test_start = train_end
        test_end = test_start + test_periods

        if verbose:
            print(f"\nFold {fold}: Train [{train_start}:{train_end}], Test [{test_start}:{test_end}]")

        # Split data
        train_data: pl.DataFrame | dict[str, pl.DataFrame]
        test_data: pl.DataFrame | dict[str, pl.DataFrame]
        if isinstance(data, dict):
            train_data = {asset: df[train_start:train_end] for asset, df in data.items()}
            test_data = {asset: df[test_start:test_end] for asset, df in data.items()}
        else:
            train_data = data[train_start:train_end]
            test_data = data[test_start:test_end]

        # Optimize on training data
        best_params = optimize(
            strategy_class=strategy_class,
            data=train_data,
            param_grid=param_grid,
            objective=objective,
            maximize=maximize,
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
            price_columns=price_columns,
            warmup=warmup,
            order_delay=order_delay,
            n_jobs=n_jobs,
            verbose=False,
            borrow_rate=borrow_rate,
            bars_per_day=bars_per_day,
            max_position_size=max_position_size,
            max_total_exposure=max_total_exposure,
            max_drawdown_stop=max_drawdown_stop,
            daily_loss_limit=daily_loss_limit,
            leverage=leverage,
            maintenance_margin=maintenance_margin,
        )

        # Test on out-of-sample data
        test_result = backtest(
            strategy_class=strategy_class,
            data=test_data,
            params=best_params["params"],
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
            price_columns=price_columns,
            warmup=warmup,
            order_delay=order_delay,
            borrow_rate=borrow_rate,
            bars_per_day=bars_per_day,
            max_position_size=max_position_size,
            max_total_exposure=max_total_exposure,
            max_drawdown_stop=max_drawdown_stop,
            daily_loss_limit=daily_loss_limit,
            leverage=leverage,
            maintenance_margin=maintenance_margin,
        )

        results.append(
            {
                "fold": fold,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "best_params": best_params["params"],
                "train_objective": best_params.get(objective, 0.0),
                "test_objective": test_result.get(objective, 0.0),
                **{f"test_{k}": v for k, v in test_result.items() if k not in ["params", "final_positions"]},
            }
        )

        if verbose:
            print(f"  Train {objective}: {best_params.get(objective, 0.0):.4f}")
            print(f"  Test {objective}: {test_result.get(objective, 0.0):.4f}")

        # Move to next window
        start_idx += test_periods

    return pl.DataFrame(results)
