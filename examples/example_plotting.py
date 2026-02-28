"""
Example: Generating interactive HTML charts from backtest results.

Demonstrates plot_backtest() with indicators, bands, trade markers,
and plot_returns_distribution(). Both charts are saved to HTML files.

The strategy alternates between long and short positions to showcase
all four trade outcomes: profitable long, losing long, profitable short, losing short.
"""

import math as _math
from datetime import datetime, timedelta

import polars as pl

from polarbtest import Engine, Strategy
from polarbtest import indicators as ind
from polarbtest.core import BacktestContext
from polarbtest.plotting import plot_backtest, plot_returns_distribution

# Generate synthetic OHLCV data with real timestamps and oscillating prices.
# Pattern: rise → fall → rise → fall to create clear long and short opportunities.
N = 200
BASE_DATE = datetime(2024, 1, 1)
timestamps = [BASE_DATE + timedelta(days=i) for i in range(N)]

# Price pattern designed to produce winning and losing trades on both sides:
# 0-40:   rise (long entry, will profit)
# 40-55:  brief dip then recovery (whipsaw: losing short)
# 55-80:  continued rise (long profits more)
# 80-100: sharp decline (long exit at loss, short entry profits)
# 100-140: steady decline (short wins)
# 140-160: brief spike then resume decline (whipsaw: losing long)
# 160-200: rise (long wins)
close: list[float] = []
for i in range(N):
    noise = 0.4 * _math.sin(i * 0.8)
    if i < 40:
        close.append(100.0 + i * 1.0 + noise)
    elif i < 48:
        close.append(140.0 - (i - 40) * 2.0 + noise)  # dip
    elif i < 55:
        close.append(124.0 + (i - 48) * 3.0 + noise)  # recover fast
    elif i < 80:
        close.append(145.0 + (i - 55) * 0.4 + noise)  # slow rise
    elif i < 100:
        close.append(155.0 - (i - 80) * 2.5 + noise)  # sharp drop
    elif i < 140:
        close.append(105.0 - (i - 100) * 0.6 + noise)  # steady fall
    elif i < 148:
        close.append(81.0 + (i - 140) * 2.5 + noise)  # spike up
    elif i < 160:
        close.append(101.0 - (i - 148) * 2.0 + noise)  # drop back
    else:
        close.append(77.0 + (i - 160) * 1.2 + noise)  # final rise

data = pl.DataFrame(
    {
        "timestamp": timestamps,
        "open": [close[0]] + [close[i - 1] + 0.2 * ((-1) ** i) for i in range(1, N)],
        "high": [c + 1.5 + 0.3 * (i % 5) for i, c in enumerate(close)],
        "low": [c - 1.5 - 0.3 * (i % 5) for i, c in enumerate(close)],
        "close": close,
        "volume": [int(5000 + 2000 * abs((i % 20 - 10) / 10)) for i in range(N)],
    }
)


class LongShortStrategy(Strategy):
    """Strategy that goes long in uptrends and short in downtrends.

    Uses SMA crossover to detect trend direction, RSI for timing.
    Produces profitable and losing trades on both sides.
    """

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        bb = ind.bollinger_bands("close", 20, 2.0)
        return df.with_columns(
            [
                ind.sma("close", 5).alias("sma_fast"),
                ind.sma("close", 15).alias("sma_slow"),
                ind.rsi("close", 14).alias("rsi"),
                bb[0].alias("bb_upper"),
                bb[2].alias("bb_lower"),
            ]
        )

    def next(self, ctx: BacktestContext) -> None:
        sma_fast = ctx.row["sma_fast"]
        sma_slow = ctx.row["sma_slow"]
        pos = ctx.portfolio.get_position("asset")

        # SMA crossover: go long when fast > slow, short when fast < slow
        if sma_fast > sma_slow and pos <= 0:
            ctx.portfolio.order_target_percent("asset", 1.0)
        elif sma_fast < sma_slow and pos >= 0:
            ctx.portfolio.order_target_percent("asset", -1.0)


engine = Engine(
    strategy=LongShortStrategy(),
    data=data,
    initial_cash=100_000,
    commission=0.001,
)
results = engine.run()

trade_stats = results["trade_stats"]
print(f"Total return:  {results['total_return']:+.2%}")
print(f"Sharpe ratio:  {results['sharpe_ratio']:.2f}")
print(f"Max drawdown:  {results['max_drawdown']:.2%}")
print(
    f"Trades:        {trade_stats['total_trades']} ({trade_stats['winning_trades']}W / {trade_stats['losing_trades']}L)"
)
print(f"Win rate:      {trade_stats['win_rate']:.1f}%")

# Print individual trades
trades_df = results["trades"]
if len(trades_df) > 0:
    print("\nTrades:")
    for row in trades_df.iter_rows(named=True):
        d = row["direction"]
        pnl = row["pnl"]
        result = "WIN" if pnl > 0 else "LOSS"
        print(f"  {d:5s} | PnL: {pnl:+10.2f} ({row['pnl_pct']:+.1f}%) | {row['bars_held']} bars | {result}")

# Generate charts
fig_backtest = plot_backtest(
    engine,
    indicators=["sma_fast", "sma_slow"],
    bands=[("bb_upper", "bb_lower")],
    title="Long/Short Strategy — Backtest Results",
    save_html="backtest_chart.html",
)

fig_returns = plot_returns_distribution(
    engine,
    title="Daily Returns Distribution",
    save_html="returns_chart.html",
)

print("\nCharts saved to backtest_chart.html and returns_chart.html")
