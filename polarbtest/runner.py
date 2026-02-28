"""
Runner utilities for backtesting and parallel execution.

This module provides high-level functions for running backtests,
optimized for evolutionary search and parameter optimization.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import polars as pl

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
    commission: float | tuple[float, float] = 0.001,
    slippage: float = 0.0005,
    price_columns: dict[str, str] | None = None,
    warmup: int | str = "auto",
    order_delay: int = 0,
    borrow_rate: float = 0.0,
    bars_per_day: float | None = None,
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
        float | tuple[float, float],
        float,
        dict[str, str] | None,
        int | str,
        int,
        float,
        float | None,
    ],
) -> BacktestResult:
    """
    Worker function for parallel backtest execution.

    Args:
        args: Tuple of (strategy_class, data, params, initial_cash, commission, slippage,
              price_columns, warmup, order_delay, borrow_rate, bars_per_day)

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
    commission: float | tuple[float, float] = 0.001,
    slippage: float = 0.0005,
    price_columns: dict[str, str] | None = None,
    warmup: int | str = "auto",
    order_delay: int = 0,
    n_jobs: int | None = None,
    verbose: bool = True,
    borrow_rate: float = 0.0,
    bars_per_day: float | None = None,
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
        )
        for params in param_sets
    ]

    results = []

    # Run backtests in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
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


def optimize(
    strategy_class: type[Strategy],
    data: pl.DataFrame | dict[str, pl.DataFrame],
    param_grid: dict[str, list[Any]],
    objective: str = "sharpe_ratio",
    maximize: bool = True,
    initial_cash: float = 100_000.0,
    commission: float | tuple[float, float] = 0.001,
    slippage: float = 0.0005,
    price_columns: dict[str, str] | None = None,
    warmup: int | str = "auto",
    order_delay: int = 0,
    n_jobs: int | None = None,
    verbose: bool = True,
    borrow_rate: float = 0.0,
    bars_per_day: float | None = None,
) -> dict[str, Any]:
    """
    Grid search optimization for strategy parameters.

    Args:
        strategy_class: Strategy class
        data: Price data
        param_grid: Dictionary mapping parameter names to lists of values
        objective: Metric to optimize (default "sharpe_ratio")
        maximize: Whether to maximize objective (default True)
        initial_cash: Starting capital
        commission: Commission as a percentage or tuple of (fixed_commission, percent_commission)
        slippage: Slippage rate
        price_columns: Asset price columns
        warmup: Number of bars to skip or "auto" (default "auto")
        order_delay: Number of bars to delay order execution (default 0)
        n_jobs: Number of parallel jobs
        verbose: Print progress
        borrow_rate: Annual borrow rate for short positions (default 0.0)
        bars_per_day: Number of bars in a trading day (default None)

    Returns:
        Dictionary with best parameters and results

    Example:
        param_grid = {
            "sma_period": [10, 20, 50, 100],
            "rsi_period": [7, 14, 21],
        }

        best = optimize(MyStrategy, data, param_grid)
        print(f"Best params: {best['params']}")
        print(f"Best Sharpe: {best['sharpe_ratio']}")
    """
    # Generate all parameter combinations
    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    param_sets = [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]

    if verbose:
        print(f"Testing {len(param_sets)} parameter combinations...")

    # Run batch backtest
    results_df = backtest_batch(
        strategy_class=strategy_class,
        data=data,
        param_sets=param_sets,
        initial_cash=initial_cash,
        commission=commission,
        slippage=slippage,
        price_columns=price_columns,
        warmup=warmup,
        order_delay=order_delay,
        n_jobs=n_jobs,
        verbose=verbose,
        borrow_rate=borrow_rate,
        bars_per_day=bars_per_day,
    )

    # Find best result
    if objective not in results_df.columns:
        raise ValueError(f"Objective '{objective}' not found in results")

    best_row = results_df.sort(objective, descending=maximize).head(1)

    return best_row.to_dicts()[0]


def walk_forward_analysis(
    strategy_class: type[Strategy],
    data: pl.DataFrame | dict[str, pl.DataFrame],
    param_grid: dict[str, list[Any]],
    train_periods: int,
    test_periods: int,
    objective: str = "sharpe_ratio",
    maximize: bool = True,
    initial_cash: float = 100_000.0,
    commission: float | tuple[float, float] = 0.001,
    slippage: float = 0.0005,
    price_columns: dict[str, str] | None = None,
    warmup: int | str = "auto",
    order_delay: int = 0,
    anchored: bool = False,
    verbose: bool = True,
    borrow_rate: float = 0.0,
    bars_per_day: float | None = None,
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
            verbose=False,
            borrow_rate=borrow_rate,
            bars_per_day=bars_per_day,
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
