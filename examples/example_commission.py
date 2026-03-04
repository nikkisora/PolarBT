"""
Example demonstrating different commission models in PolarBT.

This script shows how to use:
1. Percentage-only commission (backward compatible)
2. Fixed commission only
3. Mixed fixed + percentage commission
"""

import polars as pl

from polarbt import indicators as ind
from polarbt.core import BacktestContext, Strategy
from polarbt.runner import backtest


class SimpleStrategy(Strategy):
    """Simple SMA crossover strategy for testing commission impact."""

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        fast = self.params.get("fast", 10)
        slow = self.params.get("slow", 20)

        return df.with_columns(
            [
                ind.sma("close", fast).alias("sma_fast"),
                ind.sma("close", slow).alias("sma_slow"),
            ]
        )

    def next(self, ctx: BacktestContext) -> None:
        # Buy when fast SMA crosses above slow SMA
        if ctx.row["sma_fast"] > ctx.row["sma_slow"]:
            ctx.portfolio.order_target_percent("asset", 0.95)
        else:
            ctx.portfolio.close_position("asset")


def create_sample_data(n_bars: int = 252) -> pl.DataFrame:
    """Create sample price data with an upward trend."""
    import random

    price = 100.0
    prices = []

    for _ in range(n_bars):
        # Random walk with slight upward drift
        price *= 1 + random.gauss(0.001, 0.02)
        prices.append(max(price, 1.0))  # Prevent negative prices

    return pl.DataFrame(
        {
            "timestamp": range(n_bars),
            "close": prices,
        }
    )


def main() -> None:
    # Generate sample data
    data = create_sample_data(252)  # 1 year of daily data

    print("=" * 80)
    print("Commission Models Comparison")
    print("=" * 80)
    print()

    # Test 1: Percentage-only commission (backward compatible)
    print("1. Percentage-Only Commission (0.1%)")
    print("-" * 80)
    results1 = backtest(
        SimpleStrategy,
        data,
        params={"fast": 10, "slow": 20},
        initial_cash=100_000,
        commission=0.001,  # 0.1% per trade
    )
    print(f"Final Equity: ${results1.final_equity:,.2f}")
    print(f"Total Return: {results1.total_return:.2%}")
    print(f"Sharpe Ratio: {results1.sharpe_ratio:.2f}")
    print()

    # Test 2: Fixed commission only
    print("2. Fixed Commission Only ($5 per trade)")
    print("-" * 80)
    results2 = backtest(
        SimpleStrategy,
        data,
        params={"fast": 10, "slow": 20},
        initial_cash=100_000,
        commission=(5.0, 0.0),  # $5 fixed, 0% percentage
    )
    print(f"Final Equity: ${results2.final_equity:,.2f}")
    print(f"Total Return: {results2.total_return:.2%}")
    print(f"Sharpe Ratio: {results2.sharpe_ratio:.2f}")
    print()

    # Test 3: Mixed fixed + percentage commission
    print("3. Mixed Commission ($5 + 0.1%)")
    print("-" * 80)
    results3 = backtest(
        SimpleStrategy,
        data,
        params={"fast": 10, "slow": 20},
        initial_cash=100_000,
        commission=(5.0, 0.001),  # $5 fixed + 0.1%
    )
    print(f"Final Equity: ${results3.final_equity:,.2f}")
    print(f"Total Return: {results3.total_return:.2%}")
    print(f"Sharpe Ratio: {results3.sharpe_ratio:.2f}")
    print()

    # Test 4: Commission-free (for comparison)
    print("4. Commission-Free (for comparison)")
    print("-" * 80)
    results4 = backtest(
        SimpleStrategy,
        data,
        params={"fast": 10, "slow": 20},
        initial_cash=100_000,
        commission=0.0,  # No commission
    )
    print(f"Final Equity: ${results4.final_equity:,.2f}")
    print(f"Total Return: {results4.total_return:.2%}")
    print(f"Sharpe Ratio: {results4.sharpe_ratio:.2f}")
    print()

    # Summary comparison
    print("=" * 80)
    print("Summary: Commission Impact on Returns")
    print("=" * 80)
    print(f"{'Commission Model':<30} {'Final Equity':<15} {'Total Return':<15}")
    print("-" * 80)
    print(f"{'Commission-Free':<30} ${results4.final_equity:>13,.2f} {results4.total_return:>13.2%}")
    print(f"{'0.1% Percentage':<30} ${results1.final_equity:>13,.2f} {results1.total_return:>13.2%}")
    print(f"{'$5 Fixed':<30} ${results2.final_equity:>13,.2f} {results2.total_return:>13.2%}")
    print(f"{'$5 + 0.1% Mixed':<30} ${results3.final_equity:>13,.2f} {results3.total_return:>13.2%}")
    print()

    # Calculate commission impact
    print("Commission Impact (compared to commission-free):")
    print(f"  0.1% Percentage: {(results1.total_return - results4.total_return):.2%}")
    print(f"  $5 Fixed:        {(results2.total_return - results4.total_return):.2%}")
    print(f"  $5 + 0.1% Mixed: {(results3.total_return - results4.total_return):.2%}")
    print()


if __name__ == "__main__":
    main()
