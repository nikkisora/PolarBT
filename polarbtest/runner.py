"""
Runner utilities for backtesting and parallel execution.

This module provides high-level functions for running backtests,
optimized for evolutionary search and parameter optimization.
"""

import polars as pl
from typing import Dict, Any, List, Type, Optional, Callable, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass

from polarbtest.core import Strategy, Engine


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

    params: Dict[str, Any]
    metrics: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None


def backtest(
    strategy_class: Type[Strategy],
    data: Union[pl.DataFrame, Dict[str, pl.DataFrame]],
    params: Optional[Dict[str, Any]] = None,
    initial_cash: float = 100_000.0,
    commission: Union[float, tuple[float, float]] = 0.001,
    slippage: float = 0.0005,
    price_columns: Optional[Dict[str, str]] = None,
    warmup: Union[int, str] = "auto",
    order_delay: int = 0,
) -> Dict[str, Any]:
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


def _run_backtest_worker(args: tuple) -> BacktestResult:
    """
    Worker function for parallel backtest execution.

    Args:
        args: Tuple of (strategy_class, data, params, initial_cash, commission, slippage, price_columns, warmup, order_delay)

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
    strategy_class: Type[Strategy],
    data: Union[pl.DataFrame, Dict[str, pl.DataFrame]],
    param_sets: List[Dict[str, Any]],
    initial_cash: float = 100_000.0,
    commission: Union[float, tuple[float, float]] = 0.001,
    slippage: float = 0.0005,
    price_columns: Optional[Dict[str, str]] = None,
    warmup: Union[int, str] = "auto",
    order_delay: int = 0,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
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
        )
        for params in param_sets
    ]

    results = []

    # Run backtests in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(_run_backtest_worker, args): i
            for i, args in enumerate(args_list)
        }

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            completed += 1
            if verbose and completed % max(1, len(param_sets) // 10) == 0:
                print(
                    f"  Progress: {completed}/{len(param_sets)} ({100 * completed // len(param_sets)}%)"
                )

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
    strategy_class: Type[Strategy],
    data: Union[pl.DataFrame, Dict[str, pl.DataFrame]],
    param_grid: Dict[str, List[Any]],
    objective: str = "sharpe_ratio",
    maximize: bool = True,
    initial_cash: float = 100_000.0,
    commission: Union[float, tuple[float, float]] = 0.001,
    slippage: float = 0.0005,
    price_columns: Optional[Dict[str, str]] = None,
    warmup: Union[int, str] = "auto",
    order_delay: int = 0,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
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

    param_sets = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

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
    )

    # Find best result
    if objective not in results_df.columns:
        raise ValueError(f"Objective '{objective}' not found in results")

    best_row = results_df.sort(objective, descending=maximize).head(1)

    return best_row.to_dicts()[0]


def walk_forward_analysis(
    strategy_class: Type[Strategy],
    data: Union[pl.DataFrame, Dict[str, pl.DataFrame]],
    param_grid: Dict[str, List[Any]],
    train_periods: int,
    test_periods: int,
    objective: str = "sharpe_ratio",
    maximize: bool = True,
    initial_cash: float = 100_000.0,
    commission: Union[float, tuple[float, float]] = 0.001,
    slippage: float = 0.0005,
    price_columns: Optional[Dict[str, str]] = None,
    warmup: Union[int, str] = "auto",
    order_delay: int = 0,
    anchored: bool = False,
    verbose: bool = True,
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
    total_periods = len(data)
    results = []

    start_idx = 0
    fold = 0

    while start_idx + train_periods + test_periods <= total_periods:
        fold += 1

        if anchored:
            train_start = 0
        else:
            train_start = start_idx

        train_end = start_idx + train_periods
        test_start = train_end
        test_end = test_start + test_periods

        if verbose:
            print(
                f"\nFold {fold}: Train [{train_start}:{train_end}], Test [{test_start}:{test_end}]"
            )

        # Split data
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
                **{
                    f"test_{k}": v
                    for k, v in test_result.items()
                    if k not in ["params", "final_positions"]
                },
            }
        )

        if verbose:
            print(f"  Train {objective}: {best_params.get(objective, 0.0):.4f}")
            print(f"  Test {objective}: {test_result.get(objective, 0.0):.4f}")

        # Move to next window
        start_idx += test_periods

    return pl.DataFrame(results)
