"""
Example: Walk-Forward Analysis Workflow

Demonstrates:
- Walk-forward optimization (train/test split)
- Anchored vs rolling windows
- Interpreting fold-level results
- Comparing walk-forward performance to in-sample optimization
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from polarbt import Strategy, optimize, param
from polarbt import indicators as ind
from polarbt.core import BacktestContext
from polarbt.runner import walk_forward_analysis

N = 600
BASE_DATE = datetime(2022, 1, 1)
rng = np.random.default_rng(42)

# Clear regime changes to make walk-forward meaningful
log_returns = rng.normal(0.001, 0.012, N)
log_returns[0:150] += 0.001  # mild bull
log_returns[150:300] -= 0.001  # mild bear
log_returns[300:450] += 0.002  # strong bull
log_returns[450:600] -= 0.001  # mild bear

close = 100.0 * np.exp(np.cumsum(log_returns))
high = close * (1 + rng.uniform(0.002, 0.015, N))
low = close * (1 - rng.uniform(0.002, 0.015, N))
opn = close * (1 + rng.normal(0, 0.003, N))

data = pl.DataFrame(
    {
        "timestamp": [BASE_DATE + timedelta(days=i) for i in range(N)],
        "open": opn.tolist(),
        "high": high.tolist(),
        "low": low.tolist(),
        "close": close.tolist(),
        "volume": rng.uniform(1_000, 50_000, N).tolist(),
    }
)


class TrendFollower(Strategy):
    """Simple trend-following strategy for walk-forward testing.

    Parameters:
        fast_period: Fast SMA period
        slow_period: Slow SMA period
    """

    fast_period = param(10)
    slow_period = param(30)

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            [
                ind.sma("close", self.fast_period).alias("sma_fast"),
                ind.sma("close", self.slow_period).alias("sma_slow"),
            ]
        ).with_columns(
            [
                ind.crossover("sma_fast", "sma_slow").alias("buy"),
                ind.crossunder("sma_fast", "sma_slow").alias("sell"),
            ]
        )

    def next(self, ctx: BacktestContext) -> None:
        if ctx.row.get("buy"):
            ctx.portfolio.order_target_percent("asset", 0.95)
        elif ctx.row.get("sell"):
            ctx.portfolio.close_position("asset")


param_grid = {
    "fast_period": [5, 10, 15],
    "slow_period": [20, 30, 40],
}


def valid_params(p: dict) -> bool:  # type: ignore[type-arg]
    return p["fast_period"] < p["slow_period"]  # type: ignore[no-any-return]


if __name__ == "__main__":
    # -------------------------------------------------------------------
    # 1. Walk-Forward Analysis (rolling window)
    # -------------------------------------------------------------------

    print("=" * 70)
    print("WALK-FORWARD ANALYSIS (Rolling Window)")
    print("=" * 70)

    wf_results = walk_forward_analysis(
        TrendFollower,
        data,
        param_grid=param_grid,
        train_periods=150,  # train on 150 bars
        test_periods=75,  # test on 75 bars
        objective="sharpe_ratio",
        constraint=valid_params,
        initial_cash=100_000,
        commission=0.001,
        slippage=0.0005,
        anchored=False,  # rolling window (not expanding)
        n_jobs=1,  # sequential for script compatibility
        verbose=True,
    )

    print("\nFold Results:")
    print(
        wf_results.select(
            [
                "fold",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "best_params",
                "train_objective",
                "test_objective",
                "test_total_return",
            ]
        )
    )

    # Aggregate out-of-sample performance
    avg_oos_sharpe = wf_results["test_objective"].mean()
    avg_oos_return = wf_results["test_total_return"].mean()
    print(f"\nAverage OOS Sharpe:  {avg_oos_sharpe:.3f}")
    print(f"Average OOS Return:  {avg_oos_return:.2%}")

    # -------------------------------------------------------------------
    # 2. Compare to in-sample optimization
    # -------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("IN-SAMPLE OPTIMIZATION (for comparison)")
    print("=" * 70)

    best_is = optimize(
        TrendFollower,
        data,
        param_grid=param_grid,
        objective="sharpe_ratio",
        constraint=valid_params,
        initial_cash=100_000,
        commission=0.001,
        slippage=0.0005,
        n_jobs=1,
        verbose=False,
    )

    best_is_params = {k: best_is[k] for k in param_grid}
    print(f"Best In-Sample Params:  {best_is_params}")
    print(f"In-Sample Sharpe:       {best_is['sharpe_ratio']:.3f}")
    print(f"In-Sample Return:       {best_is['total_return']:.2%}")

    # -------------------------------------------------------------------
    # 3. Anchored walk-forward (expanding training window)
    # -------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("WALK-FORWARD ANALYSIS (Anchored / Expanding Window)")
    print("=" * 70)

    wf_anchored = walk_forward_analysis(
        TrendFollower,
        data,
        param_grid=param_grid,
        train_periods=150,
        test_periods=75,
        objective="sharpe_ratio",
        initial_cash=100_000,
        commission=0.001,
        slippage=0.0005,
        anchored=True,  # expanding window from bar 0
        n_jobs=1,
        verbose=True,
    )

    print("\nAnchored Fold Results:")
    print(
        wf_anchored.select(
            [
                "fold",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "test_objective",
                "test_total_return",
            ]
        )
    )

    avg_anchored_sharpe = wf_anchored["test_objective"].mean()
    print(f"\nAverage Anchored OOS Sharpe: {avg_anchored_sharpe:.3f}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"In-Sample Sharpe:               {best_is['sharpe_ratio']:.3f}")
    print(f"Walk-Forward OOS Sharpe (Roll):  {avg_oos_sharpe:.3f}")
    print(f"Walk-Forward OOS Sharpe (Anch):  {avg_anchored_sharpe:.3f}")
    print("\nA large gap between in-sample and OOS suggests overfitting.")
