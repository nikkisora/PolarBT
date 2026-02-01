"""
Example demonstrating multi-asset strategy with new features:
- Multiple dataframes passed as dict
- Automatic warmup period
- Order delay for realistic execution
- Standardized column names
"""

import polars as pl
from polarbtest import Strategy, backtest, indicators as ind


# Create sample data for multiple assets
btc_data = pl.DataFrame(
    {
        "date": range(100),  # Will be auto-standardized to "timestamp"
        "close": [40000 + i * 100 + (i % 10 - 5) * 200 for i in range(100)],
    }
)

eth_data = pl.DataFrame(
    {
        "date": range(100),  # Will be auto-standardized to "timestamp"
        "close": [2500 + i * 5 + (i % 8 - 4) * 50 for i in range(100)],
    }
)


# Define a momentum-based portfolio strategy
class MomentumPortfolio(Strategy):
    def preprocess(self, df):
        """Calculate momentum indicators for all assets"""
        lookback = self.params.get("lookback", 20)

        return df.with_columns(
            [
                ind.returns("BTC_close", lookback).alias("btc_momentum"),
                ind.returns("ETH_close", lookback).alias("eth_momentum"),
            ]
        )

    def next(self, ctx):
        """Allocate to asset with strongest momentum"""
        btc_mom = ctx.row.get("btc_momentum")
        eth_mom = ctx.row.get("eth_momentum")

        # Skip if indicators not ready
        if btc_mom is None or eth_mom is None:
            return

        # Allocate 100% to strongest momentum asset
        if btc_mom > eth_mom:
            ctx.portfolio.order_target_percent("BTC", 1.0)
            ctx.portfolio.order_target_percent("ETH", 0.0)
        else:
            ctx.portfolio.order_target_percent("BTC", 0.0)
            ctx.portfolio.order_target_percent("ETH", 1.0)


if __name__ == "__main__":
    print("Multi-Asset Strategy Example")
    print("=" * 60)

    # Run backtest with dict of dataframes
    results = backtest(
        MomentumPortfolio,
        data={
            "BTC": btc_data,
            "ETH": eth_data,
        },
        params={"lookback": 20},
        initial_cash=100_000,
        warmup=20,  # Auto-skip first 20 bars
        order_delay=1,  # Orders execute next bar (more realistic)
    )

    print("\nBacktest Results:")
    if results.get("success", True):
        print(f"  Total Return:     {results.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio:     {results.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown:     {results.get('max_drawdown', 0):.2%}")
        print(f"  Final Equity:     ${results.get('final_equity', 0):,.2f}")
        print(f"  Win Rate:         {results.get('win_rate', 0):.2%}")
        print(f"\n  Final Positions:")
        for asset, qty in results.get("final_positions", {}).items():
            print(f"    {asset}: {qty:.6f}")
    else:
        print(f"  ERROR: {results.get('error', 'Unknown error')}")
        if "traceback" in results:
            print(f"\n{results['traceback']}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
