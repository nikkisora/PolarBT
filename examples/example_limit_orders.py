"""
Example: Using Limit Orders and Stop-Loss Orders

This example demonstrates:
1. Placing limit orders for entry
2. Using stop-loss orders for risk management
3. Tracking order statuses
4. Analyzing trade results
"""

from typing import cast

import polars as pl

from polarbtest import Strategy, backtest
from polarbtest import indicators as ind
from polarbtest.core import BacktestContext
from polarbtest.orders import OrderStatus, OrderType


class LimitOrderStrategy(Strategy):
    """
    Strategy that uses limit orders to enter positions and stop-loss for exits.

    - Enter long with limit order 1% below current price
    - Set stop-loss at 2% below entry
    - Exit with limit order 3% above entry (take profit)
    """

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate RSI for signal generation."""
        return df.with_columns([ind.rsi("close", 14).alias("rsi")])

    def next(self, ctx: BacktestContext) -> None:
        """Execute strategy logic with limit orders and stop-loss."""
        position = ctx.portfolio.get_position("asset")
        current_price = ctx.row.get("close")
        rsi = ctx.row.get("rsi")

        if rsi is None or current_price is None:
            return

        # No position - look for entry
        if position == 0:
            # RSI oversold - place buy limit order 1% below market
            if rsi < 30:
                limit_price = current_price * 0.99  # 1% below market
                order_id = ctx.portfolio.order(
                    asset="asset",
                    quantity=1.0,
                    limit_price=limit_price,
                    order_type=OrderType.LIMIT,
                    tags=["entry", "oversold"],
                )

                if order_id:
                    print(
                        f"Bar {ctx.bar_index}: Placed buy limit order at ${limit_price:.2f} (market: ${current_price:.2f})"
                    )

        # Have position - check if we need to set stop-loss
        elif position > 0:
            # Get all filled entry orders
            filled_orders = ctx.portfolio.get_orders(status=OrderStatus.FILLED, asset="asset")

            # If we just entered, set stop-loss
            if filled_orders:
                last_fill = filled_orders[-1]
                if last_fill.filled_bar == ctx.bar_index - 1 and last_fill.filled_price is not None:
                    entry_price = last_fill.filled_price
                    stop_price = entry_price * 0.98  # 2% stop-loss
                    take_profit = entry_price * 1.03  # 3% take-profit

                    # Set stop-loss
                    ctx.portfolio.set_stop_loss(asset="asset", stop_price=stop_price)

                    # Set take-profit limit order
                    ctx.portfolio.order(
                        asset="asset",
                        quantity=-1.0,
                        limit_price=take_profit,
                        order_type=OrderType.LIMIT,
                        tags=["exit", "take_profit"],
                    )

                    print(f"Bar {ctx.bar_index}: Set SL at ${stop_price:.2f} and TP at ${take_profit:.2f}")

        # RSI overbought - close position if we have one
        if position > 0 and rsi > 70:
            ctx.portfolio.close_position("asset")
            print(f"Bar {ctx.bar_index}: Closed position (RSI overbought: {rsi:.1f})")


def main() -> None:
    """Run limit order strategy example."""
    # Generate sample OHLCV data with volatility
    import numpy as np

    np.random.seed(42)
    bars = 200
    base_price = 100

    # Generate price with trend and volatility
    returns = np.random.normal(0.001, 0.02, bars)
    close_prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices
    data = pl.DataFrame(
        {
            "time": range(bars),
            "open": close_prices * (1 + np.random.uniform(-0.005, 0.005, bars)),
            "high": close_prices * (1 + np.random.uniform(0.005, 0.015, bars)),
            "low": close_prices * (1 + np.random.uniform(-0.015, -0.005, bars)),
            "close": close_prices,
            "volume": np.random.randint(1000, 10000, bars),
        }
    )

    print("=" * 80)
    print("Limit Order Strategy Backtest")
    print("=" * 80)
    print(f"Data: {len(data)} bars")
    price_min = cast(float, data["close"].min())
    price_max = cast(float, data["close"].max())
    print(f"Price range: ${price_min:.2f} - ${price_max:.2f}")
    print()

    # Run backtest
    results = backtest(
        LimitOrderStrategy,
        data,
        params={},
        initial_cash=10_000,
        commission=0.001,  # 0.1%
        slippage=0.0005,  # 0.05%
    )

    print("\nStrategy Performance:")
    print("-" * 80)
    print(f"Total Return:    {results['total_return']:.2%}")
    print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:    {results['max_drawdown']:.2%}")
    print(f"Win Rate:        {results['win_rate']:.2%}")
    print(f"Profit Factor:   {results.get('profit_factor', 0):.2f}")

    # Show trade-level analysis
    print("\nTrade Analysis:")
    print("-" * 80)

    # Get trades DataFrame
    if "trades" in results and len(results["trades"]) > 0:
        trades_df = results["trades"]

        print(f"Total Trades:    {len(trades_df)}")
        print(f"Winning Trades:  {(trades_df['pnl'] > 0).sum()}")
        print(f"Losing Trades:   {(trades_df['pnl'] < 0).sum()}")
        print()

        # Show first few trades
        print("First 5 Trades:")
        print(
            trades_df.head(5).select(
                [
                    "asset",
                    "direction",
                    "entry_bar",
                    "entry_price",
                    "exit_bar",
                    "exit_price",
                    "pnl",
                    "pnl_pct",
                    "bars_held",
                ]
            )
        )

        # Calculate additional stats
        winning_trades = trades_df.filter(pl.col("pnl") > 0)
        losing_trades = trades_df.filter(pl.col("pnl") < 0)

        if len(winning_trades) > 0:
            avg_win = cast(float, winning_trades["pnl"].mean())
            avg_win_pct = cast(float, winning_trades["pnl_pct"].mean())
            print(f"\nAvg Winning Trade: ${avg_win:.2f} ({avg_win_pct:.2f}%)")
        if len(losing_trades) > 0:
            avg_loss = cast(float, losing_trades["pnl"].mean())
            avg_loss_pct = cast(float, losing_trades["pnl_pct"].mean())
            print(f"Avg Losing Trade:  ${avg_loss:.2f} ({avg_loss_pct:.2f}%)")

        avg_bars = cast(float, trades_df["bars_held"].mean())
        print(f"Avg Bars Held:     {avg_bars:.1f}")
    else:
        print("No trades executed")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
