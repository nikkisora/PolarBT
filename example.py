"""
Simple example demonstrating PolarBtest usage.
"""

import polars as pl
from polarbtest import Strategy, backtest, indicators as ind


# Create sample data
data = pl.DataFrame(
    {
        "timestamp": range(100),
        "close": [100 + i * 0.5 + (i % 10 - 5) for i in range(100)],
    }
)


# Define a simple moving average crossover strategy
class SMACrossStrategy(Strategy):
    def preprocess(self, df):
        """Calculate moving averages using vectorized Polars operations"""
        fast_period = self.params.get("fast_period", 10)
        slow_period = self.params.get("slow_period", 20)

        return df.with_columns(
            [
                ind.sma("close", fast_period).alias("sma_fast"),
                ind.sma("close", slow_period).alias("sma_slow"),
            ]
        )

    def next(self, ctx):
        """Execute strategy logic on each bar"""
        # Wait for indicators to warm up
        if ctx.row.get("sma_fast") is None or ctx.row.get("sma_slow") is None:
            return

        # Golden cross: go long
        if ctx.row["sma_fast"] > ctx.row["sma_slow"]:
            ctx.portfolio.order_target_percent("asset", 1.0)
        # Death cross: close position
        else:
            ctx.portfolio.close_position("asset")


if __name__ == "__main__":
    print("Running PolarBtest Example\n" + "=" * 50)

    # Run single backtest
    results = backtest(
        SMACrossStrategy,
        data,
        params={"fast_period": 10, "slow_period": 20},
        initial_cash=100_000,
    )

    print("\nBacktest Results:")
    print(f"  Total Return:    {results['total_return']:.2%}")
    print(f"  Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:    {results['max_drawdown']:.2%}")
    print(f"  Final Equity:    ${results['final_equity']:,.2f}")
    print(f"  Win Rate:        {results['win_rate']:.2%}")

    print("\n" + "=" * 50)
    print("Example completed successfully!")
