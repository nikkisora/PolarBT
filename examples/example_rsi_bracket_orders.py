"""
Example: RSI Mean Reversion with Bracket Orders

Demonstrates:
- RSI-based mean reversion signals
- Bracket orders (entry + stop-loss + take-profit as OCO)
- Bollinger Bands for entry confirmation
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from polarbt import Engine, Strategy
from polarbt import indicators as ind
from polarbt.core import BacktestContext

N = 500
BASE_DATE = datetime(2022, 1, 1)
rng = np.random.default_rng(42)

log_returns = rng.normal(0.0001, 0.018, N)
close = 100.0 * np.exp(np.cumsum(log_returns))
high = close * (1 + rng.uniform(0.003, 0.02, N))
low = close * (1 - rng.uniform(0.003, 0.02, N))
opn = close * (1 + rng.normal(0, 0.005, N))
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


class RSIBracketStrategy(Strategy):
    """RSI mean reversion with bracket orders for automatic risk management.

    When RSI is oversold and price is near the lower Bollinger Band, enter a
    long position using a bracket order that automatically sets stop-loss and
    take-profit levels.

    Parameters:
        rsi_period: RSI lookback period
        bb_period: Bollinger Band lookback period
        rsi_buy: RSI threshold for buy signal (oversold)
        rsi_sell: RSI threshold for sell signal (overbought)
        sl_pct: Stop-loss percentage below entry
        tp_pct: Take-profit percentage above entry
    """

    def __init__(
        self,
        rsi_period: int = 14,
        bb_period: int = 20,
        rsi_buy: int = 30,
        rsi_sell: int = 70,
        sl_pct: float = 0.03,
        tp_pct: float = 0.05,
        **kwargs: object,
    ) -> None:
        super().__init__(
            rsi_period=rsi_period,
            bb_period=bb_period,
            rsi_buy=rsi_buy,
            rsi_sell=rsi_sell,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            **kwargs,
        )
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        bb = ind.bollinger_bands("close", self.bb_period)
        return df.with_columns(
            [
                ind.rsi("close", self.rsi_period).alias("rsi"),
                bb[0].alias("bb_upper"),
                bb[1].alias("bb_mid"),
                bb[2].alias("bb_lower"),
            ]
        )

    def next(self, ctx: BacktestContext) -> None:
        rsi = ctx.row.get("rsi")
        bb_lower = ctx.row.get("bb_lower")
        price = ctx.row["close"]

        if rsi is None or bb_lower is None:
            return

        pos = ctx.portfolio.get_position("asset")

        # Entry: RSI oversold + price near lower BB
        if rsi < self.rsi_buy and price < bb_lower * 1.02 and pos == 0:
            # Calculate position size (95% of portfolio)
            portfolio_value = ctx.portfolio.get_value()
            quantity = (portfolio_value * 0.95) / price

            # Place bracket order: entry + SL + TP
            ctx.portfolio.order_bracket(
                asset="asset",
                quantity=quantity,
                stop_loss=price * (1 - self.sl_pct),
                take_profit=price * (1 + self.tp_pct),
            )

        # Manual exit: RSI overbought (overrides TP if hit first)
        elif pos > 0 and rsi > self.rsi_sell:
            ctx.portfolio.close_position("asset")


if __name__ == "__main__":
    print("RSI Mean Reversion with Bracket Orders")
    print("=" * 60)

    engine = Engine(
        strategy=RSIBracketStrategy(rsi_period=14, bb_period=20, rsi_buy=30, rsi_sell=70, sl_pct=0.03, tp_pct=0.05),
        data=data,
        initial_cash=100_000,
        commission=0.001,
        slippage=0.0005,
    )
    results = engine.run()

    print(f"Total Return:    {results.total_return:+.2%}")
    print(f"Sharpe Ratio:    {results.sharpe_ratio:.3f}")
    print(f"Max Drawdown:    {results.max_drawdown:.2%}")

    trade_stats = results.trade_stats
    print(f"Trades:          {trade_stats.total_trades} ({trade_stats.winning_trades}W / {trade_stats.losing_trades}L)")
    print(f"Win Rate:        {trade_stats.win_rate:.1f}%")
    print(f"Profit Factor:   {trade_stats.profit_factor:.2f}")

    trades_df = results.trades
    if len(trades_df) > 0:
        print("\nTrades:")
        for row in trades_df.iter_rows(named=True):
            pnl = row["pnl"]
            tag = "WIN" if pnl > 0 else "LOSS"
            print(
                f"  Entry: ${row['entry_price']:.2f} → Exit: ${row['exit_price']:.2f} | PnL: {pnl:+.2f} ({row['pnl_pct']:+.1f}%) | {row['bars_held']} bars | {tag}"
            )
