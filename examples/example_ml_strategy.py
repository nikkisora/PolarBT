"""
Example: ML-Integrated Strategy

Demonstrates how to integrate a machine learning model with PolarBT.
Uses a simple linear regression as a stand-in for any sklearn/torch model.

The pattern:
1. Train a model on historical features in preprocess()
2. Add predictions as a column
3. Use predictions in next() for trade decisions

Note: This example uses numpy for a minimal linear regression to avoid
requiring sklearn as a dependency. Replace with any ML framework.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from polarbt import Engine, Strategy
from polarbt import indicators as ind
from polarbt.core import BacktestContext

# Generate data with clear trending regimes
N = 500
BASE_DATE = datetime(2022, 1, 1)
rng = np.random.default_rng(42)

log_returns = rng.normal(0.001, 0.012, N)
log_returns[0:200] += 0.001  # bull
log_returns[200:300] -= 0.002  # bear
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


def simple_linear_predict(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Minimal OLS prediction — replace with any ML model.

    Fits on the entire feature matrix and returns predictions.
    In a real workflow you would train on a rolling/expanding window
    to avoid look-ahead bias.
    """
    X = np.column_stack([np.ones(len(features)), features])
    beta = np.linalg.lstsq(X, target, rcond=None)[0]
    return X @ beta


class MLStrategy(Strategy):
    """Strategy that uses ML predictions for trade signals.

    Features used:
    - RSI (momentum)
    - SMA ratio (trend)
    - Volatility (risk)
    - Volume ratio (activity)

    The model predicts next-bar returns. Positions are taken when the
    predicted return exceeds a threshold.

    Parameters:
        rsi_period: RSI lookback
        sma_period: SMA lookback for trend feature
        vol_period: Volatility lookback
        threshold: Minimum predicted return to enter (e.g., 0.0002 = 0.02%)
    """

    def __init__(
        self,
        rsi_period: int = 14,
        sma_period: int = 20,
        vol_period: int = 20,
        threshold: float = 0.0002,
        **kwargs: object,
    ) -> None:
        super().__init__(
            rsi_period=rsi_period, sma_period=sma_period, vol_period=vol_period, threshold=threshold, **kwargs
        )
        self.rsi_period = rsi_period
        self.sma_period = sma_period
        self.vol_period = vol_period
        self.threshold = threshold

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        # Step 1: compute features using vectorized indicators
        df = df.with_columns(
            [
                ind.rsi("close", self.rsi_period).alias("rsi"),
                (pl.col("close") / ind.sma("close", self.sma_period)).alias("sma_ratio"),
                pl.col("close").pct_change().rolling_std(self.vol_period).alias("volatility"),
                (pl.col("volume") / pl.col("volume").rolling_mean(self.vol_period)).alias("vol_ratio"),
                pl.col("close").pct_change().shift(-1).alias("future_return"),
            ]
        )

        # Step 2: build feature matrix from non-null rows
        feature_cols = ["rsi", "sma_ratio", "volatility", "vol_ratio"]
        target_col = "future_return"

        # Extract numpy arrays for ML
        valid_mask = df.select(feature_cols + [target_col]).drop_nulls()
        if len(valid_mask) < 50:
            return df.with_columns(pl.lit(None).cast(pl.Float64).alias("ml_prediction"))

        features = valid_mask.select(feature_cols).to_numpy()
        target = valid_mask[target_col].to_numpy()

        # Step 3: generate predictions (in-sample for simplicity)
        predictions = simple_linear_predict(features, target)

        # Step 4: map predictions back to the DataFrame
        pred_values: list[float | None] = [None] * len(df)
        valid_indices = df.select(feature_cols + [target_col]).drop_nulls().with_row_index("__idx")
        idx_list = valid_indices["__idx"].to_list()

        for i, idx in enumerate(idx_list):
            pred_values[idx] = float(predictions[i])

        df = df.with_columns(pl.Series("ml_prediction", pred_values, dtype=pl.Float64))

        # Drop future_return to avoid look-ahead bias in next()
        return df.drop("future_return")

    def next(self, ctx: BacktestContext) -> None:
        pred = ctx.row.get("ml_prediction")
        if pred is None:
            return

        pos = ctx.portfolio.get_position("asset")

        if pred > self.threshold and pos == 0:
            ctx.portfolio.order_target_percent("asset", 0.95)
        elif pred < -self.threshold and pos > 0:
            ctx.portfolio.close_position("asset")


if __name__ == "__main__":
    print("ML-Integrated Strategy")
    print("=" * 60)

    engine = Engine(
        strategy=MLStrategy(rsi_period=14, sma_period=20, vol_period=20, threshold=0.0002),
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

    trades_df = results.trades
    if len(trades_df) > 0:
        print("\nTrades:")
        for row in trades_df.iter_rows(named=True):
            pnl = row["pnl"]
            tag = "WIN" if pnl > 0 else "LOSS"
            print(
                f"  Entry: ${row['entry_price']:.2f} → Exit: ${row['exit_price']:.2f}"
                f" | PnL: {pnl:+.2f} ({row['pnl_pct']:+.1f}%) | {row['bars_held']} bars | {tag}"
            )

    print("\nNote: This example uses in-sample predictions for simplicity.")
    print("In production, use rolling/expanding window training or walk-forward analysis")
    print("to avoid look-ahead bias. See example_walk_forward.py.")
