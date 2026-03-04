"""
Example: Multi-Asset Momentum Rotation

Demonstrates:
- Multi-asset dict input
- Momentum-based asset ranking and rotation
- Position sizing across multiple assets
- Risk limits (max_position_size, max_total_exposure)
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from polarbt import Engine, Strategy, param
from polarbt import indicators as ind
from polarbt.core import BacktestContext

N = 300
BASE_DATE = datetime(2022, 1, 1)
rng = np.random.default_rng(42)


def generate_asset(drift: float, vol: float) -> pl.DataFrame:
    """Generate synthetic OHLCV data for one asset."""
    log_ret = rng.normal(drift, vol, N)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    high = close * (1 + rng.uniform(0.002, 0.015, N))
    low = close * (1 - rng.uniform(0.002, 0.015, N))
    return pl.DataFrame(
        {
            "timestamp": [BASE_DATE + timedelta(days=i) for i in range(N)],
            "open": (close * (1 + rng.normal(0, 0.003, N))).tolist(),
            "high": high.tolist(),
            "low": low.tolist(),
            "close": close.tolist(),
            "volume": rng.uniform(1_000, 30_000, N).tolist(),
        }
    )


# Create assets with distinct characteristics for clear rotation
assets = {
    "BTC": generate_asset(0.002, 0.015),  # strong uptrend, moderate vol
    "ETH": generate_asset(-0.001, 0.020),  # downtrend, higher vol
    "SOL": generate_asset(0.0015, 0.018),  # uptrend, high vol
    "BNB": generate_asset(0.0, 0.012),  # flat, low vol
}


class MomentumRotation(Strategy):
    """Rank assets by momentum and allocate to the top N performers.

    Parameters:
        lookback: Momentum lookback period (bars)
        top_n: Number of top assets to hold
    """

    lookback = param(20)
    top_n = param(2)
    asset_names = ["BTC", "ETH", "SOL", "BNB"]

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        cols = []
        for name in self.asset_names:
            cols.append(ind.returns(f"{name}_close", self.lookback).alias(f"{name}_momentum"))
        return df.with_columns(cols)

    def next(self, ctx: BacktestContext) -> None:
        # Collect momentum scores
        scores: dict[str, float] = {}
        for name in self.asset_names:
            mom = ctx.row.get(f"{name}_momentum")
            if mom is not None:
                scores[name] = mom

        if len(scores) < len(self.asset_names):
            return

        # Rank by momentum (descending)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_assets = {name for name, _ in ranked[: self.top_n]}

        # Equal-weight allocation to top assets (0.95 to leave room for commission)
        weight = 0.95 / self.top_n

        for name in self.asset_names:
            if name in top_assets:
                ctx.portfolio.order_target_percent(name, weight)
            else:
                ctx.portfolio.order_target_percent(name, 0.0)


if __name__ == "__main__":
    print("Multi-Asset Momentum Rotation")
    print("=" * 60)

    engine = Engine(
        strategy=MomentumRotation(lookback=20, top_n=2),
        data=assets,
        initial_cash=100_000,
        commission=0.001,
        slippage=0.0005,
        max_position_size=0.6,  # cap single position at 60%
        max_total_exposure=1.0,  # no leverage
    )
    results = engine.run()

    print(f"Total Return:    {results.total_return:+.2%}")
    print(f"Sharpe Ratio:    {results.sharpe_ratio:.3f}")
    print(f"Max Drawdown:    {results.max_drawdown:.2%}")
    print(f"Calmar Ratio:    {results.calmar_ratio:.3f}")
    print(f"Final Equity:    ${results.final_equity:,.2f}")

    trade_stats = results.trade_stats
    print(f"Trades:          {trade_stats.total_trades}")
    print(f"Win Rate:        {trade_stats.win_rate:.1f}%")

    # Show final positions
    print("\nFinal Positions:")
    for name in ["BTC", "ETH", "SOL", "BNB"]:
        qty = engine.portfolio.get_position(name)
        if qty != 0:
            print(f"  {name}: {qty:.6f}")
