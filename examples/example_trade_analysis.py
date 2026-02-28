"""
Example: Trade-Level Analysis

This example demonstrates:
1. Accessing individual trade data
2. Calculating trade-level metrics
3. Analyzing trade performance by direction
4. Identifying best and worst trades
"""

from typing import cast

import numpy as np
import polars as pl

from polarbtest import Strategy, backtest
from polarbtest import indicators as ind
from polarbtest.core import BacktestContext


class SimpleRSIStrategy(Strategy):
    """Simple RSI mean-reversion strategy for generating trades."""

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate RSI indicator."""
        return df.with_columns([ind.rsi("close", 14).alias("rsi")])

    def next(self, ctx: BacktestContext) -> None:
        """Execute strategy: buy when RSI < 30, sell when RSI > 70."""
        rsi = ctx.row.get("rsi")
        if rsi is None:
            return

        position = ctx.portfolio.get_position("asset")

        # Enter long on oversold
        if rsi < 30 and position == 0:
            ctx.portfolio.order_target_percent("asset", 1.0)

        # Exit on overbought
        elif rsi > 70 and position > 0:
            ctx.portfolio.close_position("asset")


def analyze_trades(trades_df: pl.DataFrame) -> None:
    """Perform detailed trade analysis."""

    if len(trades_df) == 0:
        print("No trades to analyze")
        return

    print("\n" + "=" * 80)
    print("TRADE ANALYSIS")
    print("=" * 80)

    # Overall statistics
    total_trades = len(trades_df)
    winning_trades = trades_df.filter(pl.col("pnl") > 0)
    losing_trades = trades_df.filter(pl.col("pnl") < 0)

    print("\nOverall Statistics:")
    print(f"  Total Trades:     {total_trades}")
    print(f"  Winning Trades:   {len(winning_trades)} ({len(winning_trades) / total_trades * 100:.1f}%)")
    print(f"  Losing Trades:    {len(losing_trades)} ({len(losing_trades) / total_trades * 100:.1f}%)")

    # P&L statistics
    total_pnl = trades_df["pnl"].sum()
    gross_wins = winning_trades["pnl"].sum() if len(winning_trades) > 0 else 0
    gross_losses = abs(losing_trades["pnl"].sum()) if len(losing_trades) > 0 else 0

    print("\nP&L Statistics:")
    print(f"  Total P&L:        ${total_pnl:.2f}")
    print(f"  Gross Wins:       ${gross_wins:.2f}")
    print(f"  Gross Losses:     ${gross_losses:.2f}")
    print(f"  Profit Factor:    {gross_wins / gross_losses if gross_losses > 0 else float('inf'):.2f}")

    if len(winning_trades) > 0:
        avg_win = cast(float, winning_trades["pnl"].mean())
        avg_win_pct = cast(float, winning_trades["pnl_pct"].mean())
        max_win = cast(float, winning_trades["pnl"].max())
        max_win_pct = cast(float, winning_trades["pnl_pct"].max())
        print(f"  Avg Win:          ${avg_win:.2f} ({avg_win_pct:.2f}%)")
        print(f"  Largest Win:      ${max_win:.2f} ({max_win_pct:.2f}%)")

    if len(losing_trades) > 0:
        avg_loss = cast(float, losing_trades["pnl"].mean())
        avg_loss_pct = cast(float, losing_trades["pnl_pct"].mean())
        min_loss = cast(float, losing_trades["pnl"].min())
        min_loss_pct = cast(float, losing_trades["pnl_pct"].min())
        print(f"  Avg Loss:         ${avg_loss:.2f} ({avg_loss_pct:.2f}%)")
        print(f"  Largest Loss:     ${min_loss:.2f} ({min_loss_pct:.2f}%)")

    # Holding period analysis
    avg_bars = cast(float, trades_df["bars_held"].mean())
    min_bars = cast(int, trades_df["bars_held"].min())
    max_bars = cast(int, trades_df["bars_held"].max())
    print("\nHolding Period:")
    print(f"  Avg Bars Held:    {avg_bars:.1f}")
    print(f"  Min Bars Held:    {min_bars}")
    print(f"  Max Bars Held:    {max_bars}")

    # Best and worst trades
    print("\nTop 3 Winning Trades:")
    best_trades = trades_df.sort("pnl", descending=True).head(3)
    for i, row in enumerate(best_trades.iter_rows(named=True), 1):
        print(
            f"  {i}. Entry: ${row['entry_price']:.2f} → Exit: ${row['exit_price']:.2f} | "
            f"P&L: ${row['pnl']:.2f} ({row['pnl_pct']:.2f}%) | Bars: {row['bars_held']}"
        )

    print("\nTop 3 Losing Trades:")
    worst_trades = trades_df.sort("pnl").head(3)
    for i, row in enumerate(worst_trades.iter_rows(named=True), 1):
        print(
            f"  {i}. Entry: ${row['entry_price']:.2f} → Exit: ${row['exit_price']:.2f} | "
            f"P&L: ${row['pnl']:.2f} ({row['pnl_pct']:.2f}%) | Bars: {row['bars_held']}"
        )

    # Consecutive wins/losses
    print("\nStreak Analysis:")
    pnl_signs = (trades_df["pnl"] > 0).to_list()

    current_streak = 1
    max_win_streak = 0
    max_loss_streak = 0

    for i in range(1, len(pnl_signs)):
        if pnl_signs[i] == pnl_signs[i - 1]:
            current_streak += 1
        else:
            if pnl_signs[i - 1]:
                max_win_streak = max(max_win_streak, current_streak)
            else:
                max_loss_streak = max(max_loss_streak, current_streak)
            current_streak = 1

    # Check last streak
    if pnl_signs[-1]:
        max_win_streak = max(max_win_streak, current_streak)
    else:
        max_loss_streak = max(max_loss_streak, current_streak)

    print(f"  Max Winning Streak:  {max_win_streak}")
    print(f"  Max Losing Streak:   {max_loss_streak}")

    # Return distribution
    print("\nReturn Distribution:")
    pnl_pcts = trades_df["pnl_pct"].to_numpy()
    print(f"  Mean:             {np.mean(pnl_pcts):.2f}%")
    print(f"  Median:           {np.median(pnl_pcts):.2f}%")
    print(f"  Std Dev:          {np.std(pnl_pcts):.2f}%")
    print(f"  Skewness:         {np.mean(((pnl_pcts - np.mean(pnl_pcts)) / np.std(pnl_pcts)) ** 3):.2f}")

    print("\n" + "=" * 80)


def main() -> None:
    """Run trade analysis example."""
    # Generate sample OHLCV data
    np.random.seed(42)
    bars = 500
    base_price = 100

    # Generate more volatile price action for more trades
    returns = np.random.normal(0.0005, 0.025, bars)
    close_prices = base_price * np.exp(np.cumsum(returns))

    data = pl.DataFrame(
        {
            "time": range(bars),
            "open": close_prices * (1 + np.random.uniform(-0.005, 0.005, bars)),
            "high": close_prices * (1 + np.random.uniform(0.005, 0.02, bars)),
            "low": close_prices * (1 + np.random.uniform(-0.02, -0.005, bars)),
            "close": close_prices,
            "volume": np.random.randint(1000, 10000, bars),
        }
    )

    print("=" * 80)
    print("Running RSI Mean Reversion Strategy")
    print("=" * 80)

    # Run backtest
    results = backtest(SimpleRSIStrategy, data, params={}, initial_cash=10_000, commission=0.001)

    print("\nStrategy Performance:")
    print(f"  Total Return:    {results['total_return']:.2%}")
    print(f"  Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:    {results['max_drawdown']:.2%}")

    # Detailed trade analysis
    if "trades" in results and len(results["trades"]) > 0:
        analyze_trades(results["trades"])

        # Show complete trade log
        print("\n" + "=" * 80)
        print("COMPLETE TRADE LOG")
        print("=" * 80)
        print(
            results["trades"].select(
                ["trade_id", "entry_bar", "entry_price", "exit_bar", "exit_price", "pnl", "pnl_pct", "bars_held"]
            )
        )
    else:
        print("\nNo trades executed")

    print()


if __name__ == "__main__":
    main()
