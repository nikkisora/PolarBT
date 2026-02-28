"""
Example: Generating interactive HTML charts from backtest results.

Demonstrates plot_backtest() with indicators, bands, trade markers,
and plot_returns_distribution(). Both charts are saved to HTML files.
"""

import math

import polars as pl

from polarbtest import Engine, Strategy
from polarbtest import indicators as ind
from polarbtest.core import BacktestContext
from polarbtest.plotting import plot_backtest, plot_returns_distribution


# Generate synthetic OHLCV data with a trend + oscillation
N = 300
close = [100 + i * 0.1 + 15 * math.sin(i / 10) for i in range(N)]
data = pl.DataFrame(
    {
        "timestamp": list(range(N)),
        "open": [c - 0.5 + 0.3 * math.sin(i) for i, c in enumerate(close)],
        "high": [c + abs(2 * math.sin(i / 3)) + 0.5 for i, c in enumerate(close)],
        "low": [c - abs(2 * math.sin(i / 3)) - 0.5 for i, c in enumerate(close)],
        "close": close,
        "volume": [int(1000 + 500 * abs(math.sin(i / 5))) for i in range(N)],
    }
)


class BollingerRSIStrategy(Strategy):
    """Buy when RSI < 30 and price near lower band, sell when RSI > 70."""

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        period = self.params.get("bb_period", 20)
        bb = ind.bollinger_bands("close", period, 2.0)
        return df.with_columns(
            [
                ind.sma("close", period).alias("sma_20"),
                ind.rsi("close", 14).alias("rsi"),
                bb[0].alias("bb_upper"),
                bb[1].alias("bb_middle"),
                bb[2].alias("bb_lower"),
            ]
        )

    def next(self, ctx: BacktestContext) -> None:
        rsi = ctx.row["rsi"]
        price = ctx.row["close"]
        bb_lower = ctx.row["bb_lower"]
        bb_upper = ctx.row["bb_upper"]

        if rsi < 40 and price < bb_lower * 1.02:
            ctx.portfolio.order_target_percent("asset", 1.0)
        elif rsi > 60 and price > bb_upper * 0.98:
            ctx.portfolio.close_position("asset")


# Run backtest
engine = Engine(
    strategy=BollingerRSIStrategy(bb_period=20),
    data=data,
    initial_cash=100_000,
    commission=0.001,
)
results = engine.run()

print(f"Total return: {results['total_return']:.2%}")
print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
print(f"Max drawdown: {results['max_drawdown']:.2%}")
print(f"Trades: {results['trade_stats']['total_trades']}")

# Generate charts
fig_backtest = plot_backtest(
    engine,
    indicators=["sma_20"],
    bands=[("bb_upper", "bb_lower")],
    title="Bollinger + RSI Strategy",
    save_html="backtest_chart.html",
)

fig_returns = plot_returns_distribution(
    engine,
    title="Daily Returns Distribution",
    save_html="returns_chart.html",
)

print("\nCharts saved to backtest_chart.html and returns_chart.html")
