import polars as pl
import yfinance as yf

from polarbt import Engine, Strategy, format_results
from polarbt import indicators as ind
from polarbt.core import BacktestContext
from polarbt.plotting import plot_backtest


class SMACross(Strategy):
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            ind.sma("close", 10).alias("sma_fast"),
            ind.sma("close", 30).alias("sma_slow"),
        ).with_columns(
            ind.crossover("sma_fast", "sma_slow").alias("buy"),
            ind.crossunder("sma_fast", "sma_slow").alias("sell"),
        )

    def next(self, ctx: BacktestContext) -> None:
        if ctx.row.get("buy"):
            ctx.portfolio.order_target_percent("asset", 1.0)
        elif ctx.row.get("sell"):
            ctx.portfolio.close_position("asset")


# Download data from Yahoo Finance
ticker = yf.download("AAPL", start="2016-01-01", end="2026-01-01", auto_adjust=True)
ticker = ticker.droplevel("Ticker", axis=1).reset_index()
data = pl.from_pandas(ticker)

# Run backtest
engine = Engine(SMACross(), data, commission=0.005, initial_cash=100_000)
results = engine.run()

# Pretty-print results
print(format_results(results))

# Interactive chart saved to HTML
fig = plot_backtest(engine, title="SMA Crossover — AAPL", indicators=["sma_fast", "sma_slow"])
fig.write_html("backtest.html")
