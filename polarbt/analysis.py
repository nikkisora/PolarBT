"""
Advanced analysis tools for backtesting validation.

Provides Monte Carlo simulation, look-ahead bias detection, and permutation testing
for statistical validation of trading strategy results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from polarbt.commissions import CommissionModel
from polarbt.core import Strategy


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation on trade results.

    Attributes:
        simulated_equities: Array of shape (n_simulations, n_trades+1) with equity curves.
        final_equities: Array of final equity values for each simulation.
        max_drawdowns: Array of max drawdown for each simulation.
        confidence_intervals: Dictionary of metric -> (lower, upper) at the specified confidence level.
        percentiles: Dictionary of metric -> array of percentile values.
        initial_capital: Initial capital used.
        n_simulations: Number of simulations run.
    """

    simulated_equities: np.ndarray
    final_equities: np.ndarray
    max_drawdowns: np.ndarray
    confidence_intervals: dict[str, tuple[float, float]]
    percentiles: dict[str, np.ndarray]
    initial_capital: float
    n_simulations: int


def monte_carlo(
    trades: list[Any],
    initial_capital: float = 100_000.0,
    n_simulations: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation by resampling trade P&L sequences.

    Randomly resamples (with replacement) the sequence of trade P&Ls to generate
    simulated equity curves, providing confidence intervals on key metrics.

    Args:
        trades: List of Trade objects with pnl attribute.
        initial_capital: Starting capital for simulations.
        n_simulations: Number of Monte Carlo simulations to run.
        confidence_level: Confidence level for intervals (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        MonteCarloResult with simulated equity curves and statistics.

    Raises:
        ValueError: If trades list is empty.

    Example:
        >>> result = monte_carlo(engine.portfolio.trade_tracker.trades, initial_capital=100_000)
        >>> print(f"95% CI on final equity: {result.confidence_intervals['final_equity']}")
    """
    if not trades:
        raise ValueError("trades list must not be empty")

    rng = np.random.default_rng(seed)
    pnls = np.array([t.pnl for t in trades])
    n_trades = len(pnls)

    # Resample trade indices with replacement
    indices = rng.integers(0, n_trades, size=(n_simulations, n_trades))
    resampled_pnls = pnls[indices]  # shape: (n_simulations, n_trades)

    # Build equity curves: cumulative sum of P&Ls starting from initial_capital
    cumulative = np.cumsum(resampled_pnls, axis=1)
    equities = np.column_stack([np.full(n_simulations, initial_capital), initial_capital + cumulative])

    final_equities = equities[:, -1]

    # Max drawdown per simulation
    running_max = np.maximum.accumulate(equities, axis=1)
    drawdowns = (equities - running_max) / running_max
    max_drawdowns = np.abs(drawdowns.min(axis=1))

    # Confidence intervals
    alpha = 1 - confidence_level
    lower_pct = alpha / 2 * 100
    upper_pct = (1 - alpha / 2) * 100

    ci: dict[str, tuple[float, float]] = {
        "final_equity": (
            float(np.percentile(final_equities, lower_pct)),
            float(np.percentile(final_equities, upper_pct)),
        ),
        "max_drawdown": (
            float(np.percentile(max_drawdowns, lower_pct)),
            float(np.percentile(max_drawdowns, upper_pct)),
        ),
        "total_return": (
            float(np.percentile((final_equities - initial_capital) / initial_capital, lower_pct)),
            float(np.percentile((final_equities - initial_capital) / initial_capital, upper_pct)),
        ),
    }

    standard_pcts = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99], dtype=float)
    percentiles: dict[str, np.ndarray] = {
        "final_equity": np.percentile(final_equities, standard_pcts),
        "max_drawdown": np.percentile(max_drawdowns, standard_pcts),
    }

    return MonteCarloResult(
        simulated_equities=equities,
        final_equities=final_equities,
        max_drawdowns=max_drawdowns,
        confidence_intervals=ci,
        percentiles=percentiles,
        initial_capital=initial_capital,
        n_simulations=n_simulations,
    )


@dataclass
class LookAheadResult:
    """Results from look-ahead bias detection.

    Attributes:
        biased_columns: List of column names that use future data.
        clean_columns: List of column names that do not use future data.
        details: Dictionary mapping biased column names to details about the bias.
    """

    biased_columns: list[str]
    clean_columns: list[str]
    details: dict[str, str]


def detect_look_ahead_bias(
    strategy: Strategy,
    data: pl.DataFrame,
    sample_bars: int = 5,
    tolerance: float = 1e-10,
) -> LookAheadResult:
    """Detect look-ahead bias by checking if preprocess uses future data.

    For each column added by preprocess(), computes values using truncated data
    (up to bar i) and compares to values computed on the full dataset. If they
    differ, the column leaks future information.

    Args:
        strategy: Strategy instance with preprocess() method.
        data: Raw OHLCV DataFrame before preprocessing.
        sample_bars: Number of bars to sample for checking (from the middle of the data).
        tolerance: Numerical tolerance for float comparison.

    Returns:
        LookAheadResult with lists of biased and clean columns.

    Example:
        >>> result = detect_look_ahead_bias(MyStrategy(), df, sample_bars=5)
        >>> if result.biased_columns:
        ...     print(f"WARNING: Look-ahead bias detected in: {result.biased_columns}")
    """
    original_columns = set(data.columns)

    # Preprocess full dataset
    full_processed = strategy.preprocess(data)
    new_columns = [c for c in full_processed.columns if c not in original_columns]

    if not new_columns:
        return LookAheadResult(biased_columns=[], clean_columns=[], details={})

    n = len(data)
    # Sample bars from the second half to give indicators enough warmup
    mid = n // 2
    end = n - 1
    if sample_bars >= (end - mid):
        sample_indices = list(range(mid, end))
    else:
        step = max(1, (end - mid) // sample_bars)
        sample_indices = list(range(mid, end, step))[:sample_bars]

    biased: list[str] = []
    clean: list[str] = []
    details: dict[str, str] = {}

    for col in new_columns:
        is_biased = False
        for bar_idx in sample_indices:
            # Preprocess truncated data (up to and including bar_idx)
            truncated = data[: bar_idx + 1]
            try:
                truncated_processed = strategy.preprocess(truncated)
            except Exception:
                # If preprocess fails on truncated data, skip this bar
                continue

            if col not in truncated_processed.columns:
                continue

            full_val = full_processed[col][bar_idx]
            trunc_val = truncated_processed[col][-1]

            # Handle nulls
            if full_val is None and trunc_val is None:
                continue
            if full_val is None or trunc_val is None:
                is_biased = True
                details[col] = f"Value mismatch at bar {bar_idx}: full={full_val}, truncated={trunc_val}"
                break

            # Compare values
            try:
                if isinstance(full_val, float) and isinstance(trunc_val, float):
                    if abs(full_val - trunc_val) > tolerance:
                        is_biased = True
                        details[col] = (
                            f"Value mismatch at bar {bar_idx}: full={full_val:.6f}, truncated={trunc_val:.6f}"
                        )
                        break
                elif full_val != trunc_val:
                    is_biased = True
                    details[col] = f"Value mismatch at bar {bar_idx}: full={full_val}, truncated={trunc_val}"
                    break
            except (TypeError, ValueError):
                continue

        if is_biased:
            biased.append(col)
        else:
            clean.append(col)

    return LookAheadResult(biased_columns=biased, clean_columns=clean, details=details)


@dataclass
class PermutationTestResult:
    """Results from permutation test.

    Attributes:
        original_metric: The metric value from the original (unshuffled) backtest.
        null_distribution: Array of metric values from shuffled backtests.
        p_value: Fraction of shuffled results >= original (for maximize=True).
        mean_null: Mean of the null distribution.
        std_null: Standard deviation of the null distribution.
        n_permutations: Number of permutations run.
    """

    original_metric: float
    null_distribution: np.ndarray
    p_value: float
    mean_null: float
    std_null: float
    n_permutations: int


def permutation_test(
    strategy_class: type[Strategy],
    data: pl.DataFrame,
    original_metric: float | None = None,
    metric: str = "sharpe_ratio",
    n_permutations: int = 100,
    seed: int | None = None,
    initial_cash: float = 100_000.0,
    commission: float | tuple[float, float] | CommissionModel = 0.001,
    slippage: float = 0.0005,
    warmup: int | str = "auto",
    order_delay: int = 0,
    params: dict[str, Any] | None = None,
    **engine_kwargs: Any,
) -> PermutationTestResult:
    """Test strategy significance by shuffling market returns.

    Generates a null distribution by shuffling the daily returns of the price
    data and re-running the strategy. The p-value indicates the probability
    that the strategy's performance could be achieved by chance.

    Args:
        strategy_class: Strategy class to test.
        data: Original OHLCV DataFrame.
        original_metric: Pre-computed metric value. If None, runs the strategy on original data.
        metric: Metric to evaluate (default "sharpe_ratio").
        n_permutations: Number of shuffled backtests to run.
        seed: Random seed for reproducibility.
        initial_cash: Starting capital.
        commission: Commission rate.
        slippage: Slippage rate.
        warmup: Warmup setting.
        order_delay: Order delay.
        params: Strategy parameters.
        **engine_kwargs: Additional Engine keyword arguments.

    Returns:
        PermutationTestResult with p-value and null distribution.

    Example:
        >>> result = permutation_test(MyStrategy, df, metric="sharpe_ratio", n_permutations=100)
        >>> print(f"p-value: {result.p_value:.4f}")
    """
    from polarbt.runner import backtest

    if params is None:
        params = {}

    # Run original backtest if metric not provided
    if original_metric is None:
        orig_result = backtest(
            strategy_class=strategy_class,
            data=data,
            params=params,
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
            warmup=warmup,
            order_delay=order_delay,
            **engine_kwargs,
        )
        original_metric = float(orig_result.get(metric, 0.0))

    rng = np.random.default_rng(seed)
    null_metrics: list[float] = []

    # Identify price columns to shuffle
    price_cols = [c for c in ["open", "high", "low", "close"] if c in data.columns]
    if not price_cols:
        raise ValueError("Data must contain at least one of: open, high, low, close")

    # Compute returns from close (or first available price col)
    base_col = "close" if "close" in data.columns else price_cols[0]
    prices = data[base_col].to_numpy().astype(float)

    for _ in range(n_permutations):
        # Shuffle returns and reconstruct prices
        shuffled_data = _shuffle_returns(data, price_cols, prices, rng)

        try:
            result = backtest(
                strategy_class=strategy_class,
                data=shuffled_data,
                params=params,
                initial_cash=initial_cash,
                commission=commission,
                slippage=slippage,
                warmup=warmup,
                order_delay=order_delay,
                **engine_kwargs,
            )
            null_metrics.append(float(result.get(metric, 0.0)))
        except Exception:
            null_metrics.append(0.0)

    null_array = np.array(null_metrics)

    # p-value with standard correction: (count + 1) / (n + 1)
    # This avoids reporting an exact zero p-value, which would imply
    # impossible certainty given a finite number of permutations.
    count_ge = int(np.sum(null_array >= original_metric))
    p_value = (count_ge + 1) / (n_permutations + 1)

    return PermutationTestResult(
        original_metric=original_metric,
        null_distribution=null_array,
        p_value=p_value,
        mean_null=float(np.mean(null_array)),
        std_null=float(np.std(null_array)),
        n_permutations=n_permutations,
    )


def _shuffle_returns(
    data: pl.DataFrame,
    price_cols: list[str],
    base_prices: np.ndarray,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Shuffle returns and reconstruct OHLCV data preserving intra-bar relationships.

    Computes bar-to-bar returns from the close column, shuffles them, then
    reconstructs all price columns maintaining the ratios between OHLC values
    within each bar.
    """
    n = len(data)
    if n < 2:
        return data

    # Compute log returns of base prices
    log_returns = np.diff(np.log(base_prices))

    # Shuffle the returns
    shuffled_log_returns = log_returns.copy()
    rng.shuffle(shuffled_log_returns)

    # Reconstruct base prices from shuffled returns
    new_base = np.empty(n)
    new_base[0] = base_prices[0]
    new_base[1:] = base_prices[0] * np.exp(np.cumsum(shuffled_log_returns))

    # Compute ratio of new to old prices for scaling OHLC
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(base_prices > 0, new_base / base_prices, 1.0)

    # Build new DataFrame
    new_cols: dict[str, Any] = {}
    for col in data.columns:
        if col in price_cols:
            old_vals = data[col].to_numpy().astype(float)
            new_cols[col] = (old_vals * scale).tolist()
        else:
            new_cols[col] = data[col]

    return pl.DataFrame(new_cols)
