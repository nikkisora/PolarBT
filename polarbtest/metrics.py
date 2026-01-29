"""
Performance metrics for backtesting results.

All metrics are calculated using vectorized Polars operations for maximum performance.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, Optional


def calculate_metrics(
    equity_df: pl.DataFrame, initial_capital: float
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics from an equity curve.

    Args:
        equity_df: DataFrame with 'timestamp' and 'equity' columns
        initial_capital: Initial portfolio value

    Returns:
        Dictionary containing performance metrics
    """
    if len(equity_df) == 0:
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "num_trades": 0,
        }

    # Calculate returns
    equity_df = equity_df.with_columns(
        [(pl.col("equity") / pl.col("equity").shift(1) - 1).alias("returns")]
    )

    final_equity = equity_df["equity"][-1]
    total_return = (final_equity - initial_capital) / initial_capital

    # Annualized return (assuming daily data)
    num_periods = len(equity_df)
    if num_periods > 1:
        cagr = (final_equity / initial_capital) ** (252 / num_periods) - 1
    else:
        cagr = 0.0

    # Sharpe ratio (annualized, assuming daily returns)
    returns = equity_df["returns"].drop_nulls()
    if len(returns) > 0:
        mean_return = returns.mean()
        std_return = returns.std()
        if std_return is not None and std_return > 0:
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    # Sortino ratio (annualized)
    downside_returns = returns.filter(returns < 0)
    if len(downside_returns) > 0:
        downside_std = downside_returns.std()
        if downside_std is not None and downside_std > 0:
            sortino_ratio = (mean_return / downside_std) * np.sqrt(252)
        else:
            sortino_ratio = 0.0
    else:
        sortino_ratio = sharpe_ratio  # No downside, use Sharpe

    # Maximum drawdown
    equity_df = equity_df.with_columns(
        [pl.col("equity").cum_max().alias("running_max")]
    )
    equity_df = equity_df.with_columns(
        [
            ((pl.col("equity") - pl.col("running_max")) / pl.col("running_max")).alias(
                "drawdown"
            )
        ]
    )

    max_drawdown = abs(equity_df["drawdown"].min())

    # Calmar ratio
    if max_drawdown > 0:
        calmar_ratio = cagr / max_drawdown
    else:
        calmar_ratio = 0.0

    # Win rate and other stats
    num_positive = (returns > 0).sum()
    num_negative = (returns < 0).sum()
    total_trades = num_positive + num_negative

    if total_trades > 0:
        win_rate = num_positive / total_trades
    else:
        win_rate = 0.0

    # Average win/loss
    positive_returns = returns.filter(returns > 0)
    negative_returns = returns.filter(returns < 0)

    avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0.0
    avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0.0

    # Profit factor
    total_wins = positive_returns.sum() if len(positive_returns) > 0 else 0.0
    total_losses = abs(negative_returns.sum()) if len(negative_returns) > 0 else 0.0

    if total_losses > 0:
        profit_factor = total_wins / total_losses
    else:
        profit_factor = float("inf") if total_wins > 0 else 0.0

    return {
        # Returns
        "total_return": float(total_return),
        "cagr": float(cagr),
        # Risk metrics
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar_ratio),
        # Volatility
        "volatility": float(std_return) if len(returns) > 0 else 0.0,
        "volatility_annualized": float(std_return * np.sqrt(252))
        if len(returns) > 0
        else 0.0,
        # Trade statistics
        "num_periods": int(num_periods),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor)
        if profit_factor != float("inf")
        else 999.0,
        # Equity curve stats
        "initial_equity": float(initial_capital),
        "final_equity": float(final_equity),
    }


def sharpe_ratio(equity_df: pl.DataFrame, risk_free_rate: float = 0.0) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        equity_df: DataFrame with 'equity' column
        risk_free_rate: Annual risk-free rate (default 0.0)

    Returns:
        Annualized Sharpe ratio
    """
    returns = equity_df["equity"].pct_change().drop_nulls()

    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    if std_excess > 0:
        return float(mean_excess / std_excess * np.sqrt(252))
    return 0.0


def sortino_ratio(
    equity_df: pl.DataFrame, risk_free_rate: float = 0.0, target_return: float = 0.0
) -> float:
    """
    Calculate annualized Sortino ratio.

    Args:
        equity_df: DataFrame with 'equity' column
        risk_free_rate: Annual risk-free rate (default 0.0)
        target_return: Target return threshold (default 0.0)

    Returns:
        Annualized Sortino ratio
    """
    returns = equity_df["equity"].pct_change().drop_nulls()

    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / 252)
    downside_returns = returns.filter(returns < target_return)

    if len(downside_returns) == 0:
        return 0.0

    mean_excess = excess_returns.mean()
    downside_std = downside_returns.std()

    if downside_std > 0:
        return float(mean_excess / downside_std * np.sqrt(252))
    return 0.0


def max_drawdown(equity_df: pl.DataFrame) -> float:
    """
    Calculate maximum drawdown.

    Args:
        equity_df: DataFrame with 'equity' column

    Returns:
        Maximum drawdown as a positive fraction
    """
    equity = equity_df["equity"]
    running_max = equity.cum_max()
    drawdown = (equity - running_max) / running_max
    return float(abs(drawdown.min()))


def calmar_ratio(equity_df: pl.DataFrame, initial_capital: float) -> float:
    """
    Calculate Calmar ratio (CAGR / Max Drawdown).

    Args:
        equity_df: DataFrame with 'equity' column
        initial_capital: Initial portfolio value

    Returns:
        Calmar ratio
    """
    num_periods = len(equity_df)
    if num_periods <= 1:
        return 0.0

    final_equity = equity_df["equity"][-1]
    cagr = (final_equity / initial_capital) ** (252 / num_periods) - 1

    max_dd = max_drawdown(equity_df)

    if max_dd > 0:
        return float(cagr / max_dd)
    return 0.0


def omega_ratio(equity_df: pl.DataFrame, threshold: float = 0.0) -> float:
    """
    Calculate Omega ratio.

    Args:
        equity_df: DataFrame with 'equity' column
        threshold: Return threshold (default 0.0)

    Returns:
        Omega ratio
    """
    returns = equity_df["equity"].pct_change().drop_nulls()

    if len(returns) == 0:
        return 0.0

    gains = returns.filter(returns > threshold) - threshold
    losses = threshold - returns.filter(returns < threshold)

    total_gains = gains.sum() if len(gains) > 0 else 0.0
    total_losses = losses.sum() if len(losses) > 0 else 0.0

    if total_losses > 0:
        return float(total_gains / total_losses)
    return float("inf") if total_gains > 0 else 0.0


def rolling_sharpe(equity_df: pl.DataFrame, window: int = 252) -> pl.DataFrame:
    """
    Calculate rolling Sharpe ratio.

    Args:
        equity_df: DataFrame with 'timestamp' and 'equity' columns
        window: Rolling window size (default 252 for 1 year)

    Returns:
        DataFrame with added 'rolling_sharpe' column
    """
    df = equity_df.with_columns([pl.col("equity").pct_change().alias("returns")])

    df = df.with_columns(
        [
            (
                pl.col("returns").rolling_mean(window_size=window)
                / pl.col("returns").rolling_std(window_size=window)
                * np.sqrt(252)
            ).alias("rolling_sharpe")
        ]
    )

    return df


def underwater_plot_data(equity_df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate drawdown series for underwater plot.

    Args:
        equity_df: DataFrame with 'timestamp' and 'equity' columns

    Returns:
        DataFrame with 'timestamp' and 'drawdown' columns
    """
    df = equity_df.with_columns([pl.col("equity").cum_max().alias("running_max")])

    df = df.with_columns(
        [
            ((pl.col("equity") - pl.col("running_max")) / pl.col("running_max")).alias(
                "drawdown"
            )
        ]
    )

    return df.select(["timestamp", "drawdown"])


def value_at_risk(equity_df: pl.DataFrame, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        equity_df: DataFrame with 'equity' column
        confidence: Confidence level (default 0.95 for 95% VaR)

    Returns:
        VaR as a positive value
    """
    returns = equity_df["equity"].pct_change().drop_nulls()

    if len(returns) == 0:
        return 0.0

    var = abs(float(returns.quantile(1 - confidence)))
    return var


def conditional_value_at_risk(
    equity_df: pl.DataFrame, confidence: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.

    Args:
        equity_df: DataFrame with 'equity' column
        confidence: Confidence level (default 0.95)

    Returns:
        CVaR as a positive value
    """
    returns = equity_df["equity"].pct_change().drop_nulls()

    if len(returns) == 0:
        return 0.0

    var_threshold = returns.quantile(1 - confidence)
    tail_losses = returns.filter(returns <= var_threshold)

    if len(tail_losses) > 0:
        return abs(float(tail_losses.mean()))
    return 0.0
