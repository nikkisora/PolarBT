"""
Performance metrics for backtesting results.

All metrics are calculated using vectorized Polars operations for maximum performance.
"""

from typing import Any

import numpy as np
import polars as pl


def calculate_metrics(equity_df: pl.DataFrame, initial_capital: float) -> dict[str, Any]:
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
    equity_df = equity_df.with_columns([(pl.col("equity") / pl.col("equity").shift(1) - 1).alias("returns")])

    final_equity = equity_df["equity"][-1]
    total_return = (final_equity - initial_capital) / initial_capital

    # Annualized return (assuming daily data)
    num_periods = len(equity_df)
    cagr = (final_equity / initial_capital) ** (252 / num_periods) - 1 if num_periods > 1 else 0.0

    # Sharpe ratio (annualized, assuming daily returns)
    returns = equity_df["returns"].drop_nulls()
    mean_return = 0.0
    std_return = 0.0
    if len(returns) > 0:
        mean_return_val = returns.mean()
        std_return_val = returns.std()
        # Cast to float for type safety - Polars returns numeric types
        mean_return = float(mean_return_val) if mean_return_val is not None else 0.0  # type: ignore[arg-type]
        std_return = float(std_return_val) if std_return_val is not None else 0.0  # type: ignore[arg-type]
        sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    # Sortino ratio (annualized)
    downside_returns = returns.filter(returns < 0)
    if len(downside_returns) > 0:
        downside_std_val = downside_returns.std()
        downside_std = float(downside_std_val) if downside_std_val is not None else 0.0  # type: ignore[arg-type]
        sortino_ratio = mean_return / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
    else:
        sortino_ratio = sharpe_ratio  # No downside, use Sharpe

    # Maximum drawdown
    equity_df = equity_df.with_columns([pl.col("equity").cum_max().alias("running_max")])
    equity_df = equity_df.with_columns(
        [((pl.col("equity") - pl.col("running_max")) / pl.col("running_max")).alias("drawdown")]
    )

    max_dd_val = equity_df["drawdown"].min()
    max_drawdown = float(abs(max_dd_val)) if max_dd_val is not None else 0.0  # type: ignore[arg-type]

    # Calmar ratio
    calmar_ratio = cagr / max_drawdown if max_drawdown > 0 else 0.0

    # Win rate and other stats
    num_positive = (returns > 0).sum()
    num_negative = (returns < 0).sum()
    total_trades = num_positive + num_negative

    win_rate = num_positive / total_trades if total_trades > 0 else 0.0

    # Average win/loss
    positive_returns = returns.filter(returns > 0)
    negative_returns = returns.filter(returns < 0)

    avg_win_val = positive_returns.mean() if len(positive_returns) > 0 else 0.0
    avg_loss_val = negative_returns.mean() if len(negative_returns) > 0 else 0.0
    avg_win = float(avg_win_val) if avg_win_val is not None else 0.0  # type: ignore[arg-type]
    avg_loss = float(abs(avg_loss_val)) if avg_loss_val is not None else 0.0  # type: ignore[arg-type]

    # Profit factor
    total_wins_val = positive_returns.sum() if len(positive_returns) > 0 else 0.0
    total_losses_val = negative_returns.sum() if len(negative_returns) > 0 else 0.0
    total_wins = float(total_wins_val) if total_wins_val is not None else 0.0
    total_losses = float(abs(total_losses_val)) if total_losses_val is not None else 0.0

    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf") if total_wins > 0 else 0.0

    # Ulcer Index
    ui = ulcer_index(equity_df)

    # Tail Ratio
    tr = tail_ratio(equity_df)

    # Drawdown duration stats
    dd_stats = drawdown_duration_stats(equity_df)

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
        "volatility_annualized": float(std_return * np.sqrt(252)) if len(returns) > 0 else 0.0,
        # Enhanced metrics
        "ulcer_index": ui,
        "tail_ratio": tr,
        "max_drawdown_duration": dd_stats["max_drawdown_duration"],
        "avg_drawdown_duration": dd_stats["avg_drawdown_duration"],
        "drawdown_count": dd_stats["drawdown_count"],
        # Daily return statistics
        "num_periods": int(num_periods),
        "daily_win_rate": float(win_rate),
        "daily_avg_win": float(avg_win),
        "daily_avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor) if profit_factor != float("inf") else 999.0,
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

    if std_excess is not None and std_excess > 0:  # type: ignore[operator]
        return float(mean_excess / std_excess * np.sqrt(252))  # type: ignore[operator]
    return 0.0


def sortino_ratio(equity_df: pl.DataFrame, risk_free_rate: float = 0.0, target_return: float = 0.0) -> float:
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

    if downside_std is not None and downside_std > 0:  # type: ignore[operator]
        return float(mean_excess / downside_std * np.sqrt(252))  # type: ignore[operator]
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
    min_dd = drawdown.min()
    return float(abs(min_dd)) if min_dd is not None else 0.0  # type: ignore[arg-type]


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

    df = df.with_columns([((pl.col("equity") - pl.col("running_max")) / pl.col("running_max")).alias("drawdown")])

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

    quantile_val = returns.quantile(1 - confidence)
    var = abs(float(quantile_val)) if quantile_val is not None else 0.0  # type: ignore[arg-type]
    return var


def conditional_value_at_risk(equity_df: pl.DataFrame, confidence: float = 0.95) -> float:
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
        mean_val = tail_losses.mean()
        return abs(float(mean_val)) if mean_val is not None else 0.0  # type: ignore[arg-type]
    return 0.0


def ulcer_index(equity_df: pl.DataFrame, period: int = 14) -> float:
    """Calculate Ulcer Index — measures downside volatility via drawdown depth.

    Args:
        equity_df: DataFrame with 'equity' column.
        period: Lookback period for rolling max (default 14).

    Returns:
        Ulcer Index value.
    """
    equity = equity_df["equity"]
    if len(equity) < period:
        return 0.0

    running_max = equity.rolling_max(window_size=period)
    pct_drawdown = ((equity - running_max) / running_max * 100).drop_nulls()

    if len(pct_drawdown) == 0:
        return 0.0

    squared_mean = (pct_drawdown * pct_drawdown).mean()
    if squared_mean is None:
        return 0.0
    return float(np.sqrt(float(squared_mean)))  # type: ignore[arg-type]


def tail_ratio(equity_df: pl.DataFrame, confidence: float = 0.95) -> float:
    """Calculate Tail Ratio — ratio of right tail to left tail.

    A ratio > 1 indicates the strategy has fatter right tails (larger gains than losses).

    Args:
        equity_df: DataFrame with 'equity' column.
        confidence: Confidence level for quantile calculation (default 0.95).

    Returns:
        Tail ratio.
    """
    returns = equity_df["equity"].pct_change().drop_nulls()

    if len(returns) == 0:
        return 0.0

    right_tail = returns.quantile(confidence)
    left_tail = returns.quantile(1 - confidence)

    if left_tail is None or right_tail is None:
        return 0.0

    left_val = abs(float(left_tail))  # type: ignore[arg-type]
    if left_val == 0:
        return float("inf") if float(right_tail) > 0 else 0.0  # type: ignore[arg-type]
    return float(right_tail) / left_val  # type: ignore[arg-type]


def information_ratio(equity_df: pl.DataFrame, benchmark_df: pl.DataFrame) -> float:
    """Calculate Information Ratio — excess return per unit of tracking error.

    Args:
        equity_df: DataFrame with 'equity' column.
        benchmark_df: DataFrame with 'equity' column for benchmark.

    Returns:
        Annualized Information Ratio.
    """
    returns = equity_df["equity"].pct_change().drop_nulls()
    bench_returns = benchmark_df["equity"].pct_change().drop_nulls()

    min_len = min(len(returns), len(bench_returns))
    if min_len == 0:
        return 0.0

    returns = returns.head(min_len)
    bench_returns = bench_returns.head(min_len)

    active_returns = returns - bench_returns
    mean_active = active_returns.mean()
    std_active = active_returns.std()

    if mean_active is None or std_active is None or float(std_active) == 0:  # type: ignore[arg-type]
        return 0.0

    return float(float(mean_active) / float(std_active) * np.sqrt(252))  # type: ignore[arg-type]


def alpha_beta(equity_df: pl.DataFrame, benchmark_df: pl.DataFrame, risk_free_rate: float = 0.0) -> dict[str, float]:
    """Calculate Alpha and Beta vs a benchmark using CAPM regression.

    Args:
        equity_df: DataFrame with 'equity' column.
        benchmark_df: DataFrame with 'equity' column for benchmark.
        risk_free_rate: Annual risk-free rate (default 0.0).

    Returns:
        Dictionary with 'alpha' (annualized) and 'beta' keys.
    """
    returns = equity_df["equity"].pct_change().drop_nulls().to_numpy()
    bench_returns = benchmark_df["equity"].pct_change().drop_nulls().to_numpy()

    min_len = min(len(returns), len(bench_returns))
    if min_len < 2:
        return {"alpha": 0.0, "beta": 0.0}

    returns = returns[:min_len]
    bench_returns = bench_returns[:min_len]

    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    excess_bench = bench_returns - daily_rf

    # OLS: beta = cov(r, b) / var(b)
    cov = np.cov(excess_returns, excess_bench, ddof=1)
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 0.0
    daily_alpha = float(np.mean(excess_returns) - beta * np.mean(excess_bench))
    annualized_alpha = daily_alpha * 252

    return {"alpha": annualized_alpha, "beta": beta}


def drawdown_duration_stats(equity_df: pl.DataFrame) -> dict[str, float]:
    """Calculate drawdown duration statistics.

    Args:
        equity_df: DataFrame with 'equity' column.

    Returns:
        Dictionary with max_drawdown_duration, avg_drawdown_duration, and drawdown_count.
        Durations are in number of bars.
    """
    equity = equity_df["equity"].to_numpy()

    if len(equity) < 2:
        return {"max_drawdown_duration": 0.0, "avg_drawdown_duration": 0.0, "drawdown_count": 0}

    running_max = np.maximum.accumulate(equity)
    in_drawdown = equity < running_max

    durations: list[int] = []
    current_duration = 0

    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
        elif current_duration > 0:
            durations.append(current_duration)
            current_duration = 0

    if current_duration > 0:
        durations.append(current_duration)

    if not durations:
        return {"max_drawdown_duration": 0.0, "avg_drawdown_duration": 0.0, "drawdown_count": 0}

    return {
        "max_drawdown_duration": float(max(durations)),
        "avg_drawdown_duration": float(np.mean(durations)),
        "drawdown_count": len(durations),
    }


def monthly_returns(equity_df: pl.DataFrame) -> pl.DataFrame:
    """Calculate monthly returns table.

    Args:
        equity_df: DataFrame with 'timestamp' and 'equity' columns.
            timestamp must be a Date or Datetime type.

    Returns:
        DataFrame with 'year', 'month', and 'return' columns.
    """
    if "timestamp" not in equity_df.columns or len(equity_df) == 0:
        return pl.DataFrame(schema={"year": pl.Int32, "month": pl.Int32, "return": pl.Float64})

    df = equity_df.select(["timestamp", "equity"])

    # Extract year/month
    ts_col = df["timestamp"]
    if ts_col.dtype == pl.Date or ts_col.dtype == pl.Datetime or str(ts_col.dtype).startswith("Datetime"):
        df = df.with_columns(
            [
                pl.col("timestamp").dt.year().alias("year"),
                pl.col("timestamp").dt.month().alias("month"),
            ]
        )
    else:
        return pl.DataFrame(schema={"year": pl.Int32, "month": pl.Int32, "return": pl.Float64})

    # Get last equity per month (end-of-month value)
    monthly = df.group_by(["year", "month"]).agg([pl.col("equity").last().alias("end_equity")]).sort(["year", "month"])

    # Use previous month's end equity as the start equity for each month.
    # For the first month, use the first equity value from the original data.
    first_equity = float(df["equity"][0])
    monthly = monthly.with_columns([pl.col("end_equity").shift(1).fill_null(first_equity).alias("start_equity")])

    monthly = monthly.with_columns(
        [((pl.col("end_equity") - pl.col("start_equity")) / pl.col("start_equity")).alias("return")]
    )

    return monthly.select(["year", "month", "return"])


def trade_level_metrics(trades: list[Any]) -> dict[str, float]:
    """Calculate trade-level metrics from a list of Trade objects.

    Args:
        trades: List of Trade objects with pnl attribute.

    Returns:
        Dictionary with expectancy, sqn, kelly_criterion,
        max_consecutive_wins, and max_consecutive_losses.
    """
    if not trades:
        return {
            "expectancy": 0.0,
            "sqn": 0.0,
            "kelly_criterion": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        }

    pnls = [t.pnl for t in trades]
    n = len(pnls)

    # Expectancy = average P&L per trade
    expectancy = float(np.mean(pnls))

    # SQN = sqrt(n) * mean(pnl) / std(pnl)
    std_pnl = float(np.std(pnls, ddof=1)) if n > 1 else 0.0
    sqn = np.sqrt(n) * expectancy / std_pnl if std_pnl > 0 else 0.0

    # Kelly criterion = W - (1-W)/R where W=win_rate, R=avg_win/avg_loss
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    win_rate = len(winners) / n if n > 0 else 0.0

    if losers and winners:
        avg_win = float(np.mean(winners))
        avg_loss = abs(float(np.mean(losers)))
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        kelly = win_rate - (1 - win_rate) / win_loss_ratio if win_loss_ratio > 0 else 0.0
    else:
        kelly = 1.0 if winners else 0.0

    # Consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0

    for pnl in pnls:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        elif pnl < 0:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0

    return {
        "expectancy": expectancy,
        "sqn": float(sqn),
        "kelly_criterion": float(kelly),
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
    }
