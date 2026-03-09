"""
Example: Full workflow — optimization, heatmap, TA-Lib indicators, advanced analysis.

Demonstrates:
1. Strategy using TA-Lib integration (RSI, Bollinger Bands, ATR) with built-in fallback
2. Parameter grid optimization with constraint filtering
3. 2D parameter heatmap and sensitivity plot
4. Backtest of the best parameters with trade visualization
5. Monte Carlo simulation on trade results
6. Look-ahead bias detection
7. Permutation test for statistical significance
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from polarbt import (
    Engine,
    Strategy,
    backtest_batch,
    detect_look_ahead_bias,
    monte_carlo,
    optimize,
    param,
    permutation_test,
)
from polarbt import indicators as ind
from polarbt.core import BacktestContext
from polarbt.plotting import plot_backtest, plot_param_heatmap, plot_permutation_test, plot_sensitivity

# Try TA-Lib; fall back to built-in indicators if not installed.
try:
    from polarbt.integrations.talib import ta, talib_available

    USE_TALIB = talib_available()
except ImportError:
    USE_TALIB = False

# ---------------------------------------------------------------------------
# 1. Generate synthetic OHLCV data (500 daily bars with realistic patterns)
# ---------------------------------------------------------------------------

N = 500
BASE_DATE = datetime(2022, 1, 1)
timestamps = [BASE_DATE + timedelta(days=i) for i in range(N)]

rng = np.random.default_rng(42)
log_returns = rng.normal(0.0003, 0.015, N)  # slight upward drift
# Inject a trending regime (bars 100-250) and a mean-reverting regime (bars 300-450)
log_returns[100:250] += 0.002  # bull run
log_returns[300:400] -= 0.003  # bear market
log_returns[400:450] += 0.004  # recovery

close = 100.0 * np.exp(np.cumsum(log_returns))
high = close * (1 + rng.uniform(0.002, 0.02, N))
low = close * (1 - rng.uniform(0.002, 0.02, N))
opn = close * (1 + rng.normal(0, 0.005, N))
volume = rng.uniform(1_000, 50_000, N) * (1 + 0.5 * np.abs(log_returns) / 0.015)

data = pl.DataFrame(
    {
        "timestamp": timestamps,
        "open": opn.tolist(),
        "high": high.tolist(),
        "low": low.tolist(),
        "close": close.tolist(),
        "volume": volume.tolist(),
    }
)

# ---------------------------------------------------------------------------
# 2. Define strategy — RSI mean reversion with Bollinger Band filter
# ---------------------------------------------------------------------------


class RSIBollingerStrategy(Strategy):
    """Mean reversion: buy when RSI is oversold AND price near lower BB, sell when overbought.

    Parameters:
        rsi_period: RSI lookback period
        bb_period: Bollinger Band lookback period
        rsi_buy: RSI threshold for buy signal
        rsi_sell: RSI threshold for sell signal
        atr_sl_mult: ATR multiplier for stop-loss distance
    """

    rsi_period = param(14)
    bb_period = param(20)
    rsi_buy = param(30)
    rsi_sell = param(70)
    atr_sl_mult = param(2.0)

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        if USE_TALIB:
            # Use TA-Lib for maximum accuracy
            bb = ta.bollinger_bands("close", self.bb_period)
            cols = [
                ta.rsi("close", self.rsi_period).alias("rsi"),
                bb["upper"].alias("bb_upper"),
                bb["middle"].alias("bb_mid"),
                bb["lower"].alias("bb_lower"),
                ta.atr("high", "low", "close", 14).alias("atr"),
            ]
        else:
            # Built-in Polars indicators (no external dependency)
            bb = ind.bollinger_bands("close", self.bb_period)
            cols = [
                ind.rsi("close", self.rsi_period).alias("rsi"),
                bb[0].alias("bb_upper"),
                bb[1].alias("bb_mid"),
                bb[2].alias("bb_lower"),
                ind.atr("high", "low", "close", 14).alias("atr"),
            ]

        return df.with_columns(cols)

    def next(self, ctx: BacktestContext) -> None:
        rsi = ctx.row.get("rsi")
        bb_lower = ctx.row.get("bb_lower")
        bb_upper = ctx.row.get("bb_upper")
        atr_val = ctx.row.get("atr")
        price = ctx.row["close"]

        if rsi is None or bb_lower is None or atr_val is None:
            return

        pos = ctx.portfolio.get_position("asset")

        # Buy signal: RSI oversold + price near lower Bollinger Band
        if rsi < self.rsi_buy and price < bb_lower * 1.01 and pos == 0:
            ctx.portfolio.order_target_percent("asset", 0.95)
            # Set ATR-based stop-loss
            sl_price = price - self.atr_sl_mult * atr_val
            ctx.portfolio.set_stop_loss("asset", stop_price=sl_price)

        # Sell signal: RSI overbought or price above upper BB
        elif pos > 0 and (rsi > self.rsi_sell or price > bb_upper):
            ctx.portfolio.close_position("asset")


def valid_params(p: dict) -> bool:  # type: ignore[type-arg]
    """Ensure RSI buy threshold is strictly below sell threshold."""
    return p["rsi_buy"] < p["rsi_sell"]  # type: ignore[no-any-return]


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # 3. Look-ahead bias detection (run BEFORE optimization to catch bugs early)
    # -----------------------------------------------------------------------

    print("=" * 70)
    print("STEP 1: Look-Ahead Bias Detection")
    print("=" * 70)

    bias_result = detect_look_ahead_bias(
        RSIBollingerStrategy(rsi_period=14, bb_period=20, rsi_buy=30, rsi_sell=70),
        data,
        sample_bars=5,
    )

    if bias_result.biased_columns:
        print(f"  WARNING — biased columns: {bias_result.biased_columns}")
        for col, detail in bias_result.details.items():
            print(f"    {col}: {detail}")
    else:
        print(f"  All {len(bias_result.clean_columns)} indicator columns are clean.")
        print(f"  Columns checked: {bias_result.clean_columns}")

    # -----------------------------------------------------------------------
    # 4. Parameter optimization (grid search)
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("STEP 2: Parameter Optimization")
    print("=" * 70)

    param_grid = {
        "rsi_period": [7, 10, 14, 21],
        "bb_period": [15, 20, 25, 30],
        "rsi_buy": [25, 30, 35],
        "rsi_sell": [65, 70, 75],
    }

    best = optimize(
        RSIBollingerStrategy,
        data,
        param_grid=param_grid,
        objective="sharpe_ratio",
        constraint=valid_params,
        initial_cash=100_000,
        commission=0.001,
        slippage=0.0005,
        n_jobs=8,
        verbose=True,
    )

    print("\n  Best parameters:")
    best_params = best.params
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"  Sharpe Ratio:  {best.metrics.sharpe_ratio:.3f}")
    print(f"  Total Return:  {best.metrics.total_return:.2%}")
    print(f"  Max Drawdown:  {best.metrics.max_drawdown:.2%}")

    # -----------------------------------------------------------------------
    # 5. Generate heatmap and sensitivity plots
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("STEP 3: Parameter Heatmaps & Sensitivity Plots")
    print("=" * 70)

    # Run batch with a 2D slice: fix rsi_buy/rsi_sell, vary rsi_period & bb_period
    heatmap_params = [
        {
            "rsi_period": rp,
            "bb_period": bp,
            "rsi_buy": best_params["rsi_buy"],
            "rsi_sell": best_params["rsi_sell"],
        }
        for rp in param_grid["rsi_period"]
        for bp in param_grid["bb_period"]
    ]

    results_df = backtest_batch(
        RSIBollingerStrategy,
        data,
        param_sets=heatmap_params,
        initial_cash=100_000,
        commission=0.001,
        slippage=0.0005,
        n_jobs=1,
        verbose=False,
    )

    fig_heatmap = plot_param_heatmap(
        results_df,
        param_x="rsi_period",
        param_y="bb_period",
        metric="sharpe_ratio",
        title="Sharpe Ratio — RSI Period vs BB Period",
        save_html="heatmap_sharpe.html",
    )
    print("  Saved: heatmap_sharpe.html")

    fig_sensitivity = plot_sensitivity(
        results_df,
        param="rsi_period",
        metric="sharpe_ratio",
        title="Sensitivity: Sharpe vs RSI Period",
        save_html="sensitivity_rsi.html",
    )
    print("  Saved: sensitivity_rsi.html")

    # -----------------------------------------------------------------------
    # 6. Run best strategy and plot backtest
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("STEP 4: Backtest Best Strategy")
    print("=" * 70)

    engine = Engine(
        strategy=RSIBollingerStrategy(**best_params),
        data=data,
        initial_cash=100_000,
        commission=0.001,
        slippage=0.0005,
    )
    results = engine.run()

    trade_stats = results.trade_stats
    print(f"  Total Return:  {results.total_return:+.2%}")
    print(f"  Sharpe Ratio:  {results.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio: {results.sortino_ratio:.3f}")
    print(f"  Max Drawdown:  {results.max_drawdown:.2%}")
    print(f"  Calmar Ratio:  {results.calmar_ratio:.3f}")
    print(f"  Trades:        {trade_stats.total_trades} ({trade_stats.winning_trades}W / {trade_stats.losing_trades}L)")
    print(f"  Win Rate:      {trade_stats.win_rate:.1%}")
    print(f"  Profit Factor: {trade_stats.profit_factor:.2f}")

    fig_bt = plot_backtest(
        engine,
        indicators=["bb_mid"],
        bands=[("bb_upper", "bb_lower")],
        title=f"RSI-Bollinger Strategy (RSI={best_params['rsi_period']}, BB={best_params['bb_period']})",
        save_html="backtest_best.html",
    )
    print("  Saved: backtest_best.html")

    # -----------------------------------------------------------------------
    # 7. Monte Carlo simulation
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("STEP 5: Monte Carlo Simulation (1,000 resamples)")
    print("=" * 70)

    trades = engine.portfolio.trade_tracker.trades

    if len(trades) >= 2:
        mc = monte_carlo(
            trades,
            initial_capital=100_000,
            n_simulations=1_000,
            confidence_level=0.95,
            seed=42,
        )

        ci_equity = mc.confidence_intervals["final_equity"]
        ci_dd = mc.confidence_intervals["max_drawdown"]
        ci_ret = mc.confidence_intervals["total_return"]

        print(f"  Simulations:          {mc.n_simulations}")
        print(f"  Median Final Equity:  ${float(mc.percentiles['final_equity'][4]):,.0f}")
        print(f"  95% CI Final Equity:  ${ci_equity[0]:,.0f} — ${ci_equity[1]:,.0f}")
        print(f"  95% CI Total Return:  {ci_ret[0]:.2%} — {ci_ret[1]:.2%}")
        print(f"  95% CI Max Drawdown:  {ci_dd[0]:.2%} — {ci_dd[1]:.2%}")

        pct_labels = ["1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%"]
        print("\n  Final Equity Percentiles:")
        for label, val in zip(pct_labels, mc.percentiles["final_equity"], strict=True):
            print(f"    {label:>4s}: ${float(val):>12,.0f}")
    else:
        print(f"  Skipped — only {len(trades)} trade(s), need at least 2.")

    # -----------------------------------------------------------------------
    # 8. Permutation test
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("STEP 6: Permutation Test (50 shuffles)")
    print("=" * 70)

    perm = permutation_test(
        RSIBollingerStrategy,
        data,
        original_metric=results.sharpe_ratio,
        metric="sharpe_ratio",
        n_permutations=50,
        seed=42,
        initial_cash=100_000,
        commission=0.001,
        slippage=0.0005,
        params=best_params,
    )

    print(f"  Original Sharpe:   {perm.original_metric:.3f}")
    print(f"  Null Mean±Std:     {perm.mean_null:.3f} ± {perm.std_null:.3f}")
    print(f"  p-value:           {perm.p_value:.4f}")

    if perm.p_value < 0.05:
        print("  → Statistically significant at 5% level (reject null hypothesis)")
    elif perm.p_value < 0.10:
        print("  → Marginally significant at 10% level")
    else:
        print("  → Not statistically significant — performance may be due to chance")

    fig_perm = plot_permutation_test(
        perm,
        title=f"Permutation Test — Sharpe Ratio (p = {perm.p_value:.4f})",
        save_html="permutation_test.html",
    )
    print("  Saved: permutation_test.html")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print("  Strategy:       RSI-Bollinger Mean Reversion")
    print(f"  TA-Lib Used:    {'Yes' if USE_TALIB else 'No (built-in indicators)'}")
    print(f"  Best Params:    {best_params}")
    print(f"  Sharpe Ratio:   {results.sharpe_ratio:.3f}")
    print(f"  p-value:        {perm.p_value:.4f}")
    if len(trades) >= 2:
        print(f"  MC 95% CI:      ${ci_equity[0]:,.0f} — ${ci_equity[1]:,.0f}")
    print(f"  Bias Check:     {'CLEAN' if not bias_result.biased_columns else 'BIASED'}")
    print("\n  Charts: heatmap_sharpe.html, sensitivity_rsi.html, backtest_best.html, permutation_test.html")
