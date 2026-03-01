"""
Example: SMA Crossover with Stop-Loss

Demonstrates:
- SMA crossover entry signals
- ATR-based stop-loss placement
- Trailing stop for profit protection
- Position sizing with PercentSizer
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from polarbtest import Engine, Strategy
from polarbtest import indicators as ind
from polarbtest.core import BacktestContext
from polarbtest.sizers import PercentSizer

# Generate data with clear trending regimes: bull → bear → bull
N = 500
BASE_DATE = datetime(2022, 1, 1)
rng = np.random.default_rng(42)

log_returns = rng.normal(0.001, 0.012, N)
log_returns[0:200] += 0.001  # bull run
log_returns[200:300] -= 0.002  # bear market
log_returns[300:500] += 0.001  # recovery

close = 100.0 * np.exp(np.cumsum(log_returns))
high = close * (1 + rng.uniform(0.002, 0.015, N))
low = close * (1 - rng.uniform(0.002, 0.015, N))
opn = close * (1 + rng.normal(0, 0.003, N))
volume = rng.uniform(1_000, 50_000, N)

data = pl.DataFrame(
    {
        "timestamp": [BASE_DATE + timedelta(days=i) for i in range(N)],
        "open": opn.tolist(),
        "high": high.tolist(),
        "low": low.tolist(),
        "close": close.tolist(),
        "volume": volume.tolist(),
    }
)


class SMACrossoverStopLoss(Strategy):
    """SMA crossover with ATR-based stop-loss and trailing stop.

    Parameters:
        fast_period: Fast SMA period
        slow_period: Slow SMA period
        atr_period: ATR period for stop-loss calculation
        atr_sl_mult: ATR multiplier for initial stop-loss distance
        trail_pct: Trailing stop percentage (e.g., 0.08 = 8%)
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        atr_period: int = 14,
        atr_sl_mult: float = 3.0,
        trail_pct: float = 0.08,
        **kwargs: object,
    ) -> None:
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            atr_period=atr_period,
            atr_sl_mult=atr_sl_mult,
            trail_pct=trail_pct,
            **kwargs,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.trail_pct = trail_pct

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            [
                ind.sma("close", self.fast_period).alias("sma_fast"),
                ind.sma("close", self.slow_period).alias("sma_slow"),
                ind.atr("high", "low", "close", self.atr_period).alias("atr"),
            ]
        ).with_columns(
            [
                ind.crossover("sma_fast", "sma_slow").alias("golden_cross"),
                ind.crossunder("sma_fast", "sma_slow").alias("death_cross"),
            ]
        )

    def next(self, ctx: BacktestContext) -> None:
        pos = ctx.portfolio.get_position("asset")
        price = ctx.row["close"]
        atr = ctx.row.get("atr")

        if ctx.row.get("golden_cross") and pos == 0 and atr:
            # Enter long with percent sizer
            sizer = PercentSizer(percent=0.95)
            ctx.portfolio.order_with_sizer("asset", sizer, direction=1.0, price=price)

            # Set ATR-based stop-loss
            sl_price = price - self.atr_sl_mult * atr
            ctx.portfolio.set_stop_loss("asset", stop_price=sl_price)

            # Set trailing stop for profit protection
            ctx.portfolio.set_trailing_stop("asset", trail_pct=self.trail_pct)

        elif ctx.row.get("death_cross") and pos > 0:
            ctx.portfolio.close_position("asset")


if __name__ == "__main__":
    print("SMA Crossover with Stop-Loss")
    print("=" * 60)

    engine = Engine(
        strategy=SMACrossoverStopLoss(fast_period=10, slow_period=30, atr_sl_mult=3.0, trail_pct=0.08),
        data=data,
        initial_cash=100_000,
        commission=0.001,
        slippage=0.0005,
    )
    results = engine.run()

    print(f"Total Return:    {results['total_return']:+.2%}")
    print(f"Sharpe Ratio:    {results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown:    {results['max_drawdown']:.2%}")
    print(f"Calmar Ratio:    {results['calmar_ratio']:.3f}")

    trade_stats = results["trade_stats"]
    print(
        f"Trades:          {trade_stats['total_trades']} ({trade_stats['winning_trades']}W / {trade_stats['losing_trades']}L)"
    )
    print(f"Win Rate:        {trade_stats['win_rate']:.1f}%")
    print(f"Profit Factor:   {trade_stats['profit_factor']:.2f}")

    trades_df = results["trades"]
    if len(trades_df) > 0:
        print("\nTrades:")
        for row in trades_df.iter_rows(named=True):
            pnl = row["pnl"]
            tag = "WIN" if pnl > 0 else "LOSS"
            print(
                f"  {row['direction']:5s} | Entry: ${row['entry_price']:.2f} → Exit: ${row['exit_price']:.2f}"
                f" | PnL: {pnl:+.2f} ({row['pnl_pct']:+.1f}%) | {row['bars_held']} bars | {tag}"
            )
