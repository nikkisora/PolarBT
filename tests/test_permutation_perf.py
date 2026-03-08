"""Performance benchmark for permutation test on large datasets."""

import time

import numpy as np
import polars as pl

from polarbt import indicators as ind
from polarbt.analysis import PermutationTestResult, _shuffle_returns, permutation_test
from polarbt.core import BacktestContext, Strategy


class SimpleSMAStrategy(Strategy):
    """Simple SMA crossover for benchmarking."""

    def __init__(self, fast: int = 10, slow: int = 30, **kwargs: object) -> None:
        super().__init__(fast=fast, slow=slow, **kwargs)
        self.fast = fast
        self.slow = slow

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            [
                ind.sma("close", self.fast).alias("sma_fast"),
                ind.sma("close", self.slow).alias("sma_slow"),
            ]
        )

    def next(self, ctx: BacktestContext) -> None:
        fast = ctx.row.get("sma_fast")
        slow = ctx.row.get("sma_slow")
        if fast is None or slow is None:
            return
        if fast > slow and ctx.portfolio.get_position("asset") == 0:
            ctx.portfolio.order("asset", 1.0)
        elif fast < slow and ctx.portfolio.get_position("asset") > 0:
            ctx.portfolio.close_position("asset")


def _make_ohlcv_data(n: int = 15000, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 10.0)
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    low = np.maximum(low, 1.0)
    opn = close + rng.normal(0, 0.5, n)
    opn = np.maximum(opn, 1.0)
    volume = rng.uniform(100, 1000, n)

    return pl.DataFrame(
        {
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class TestShuffleReturnsPerf:
    """Benchmark _shuffle_returns in isolation."""

    def test_shuffle_returns_speed(self) -> None:
        data = _make_ohlcv_data(15_000)
        price_cols = ["open", "high", "low", "close"]
        base_prices = data["close"].to_numpy().astype(float)
        log_returns = np.diff(np.log(base_prices))
        price_arrays = {col: data[col].to_numpy().astype(float) for col in price_cols}
        rng = np.random.default_rng(42)

        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            _shuffle_returns(data, price_cols, base_prices, log_returns, price_arrays, rng)
        elapsed = time.perf_counter() - start

        per_call_ms = (elapsed / n_iters) * 1000
        print(f"\n_shuffle_returns: {per_call_ms:.2f} ms/call ({n_iters} iters, 15k rows)")
        # This is a benchmark, not a strict assertion — but flag if very slow
        assert per_call_ms < 50, f"_shuffle_returns too slow: {per_call_ms:.2f} ms"


class TestPermutationTestPerf:
    """Benchmark full permutation_test."""

    def test_permutation_test_15k_candles(self) -> None:
        """Benchmark: 15k candles, 20 permutations."""
        data = _make_ohlcv_data(15_000)

        start = time.perf_counter()
        result = permutation_test(
            SimpleSMAStrategy,
            data,
            original_metric=0.5,
            metric="sharpe_ratio",
            n_permutations=20,
            seed=42,
            params={"fast": 10, "slow": 30},
        )
        elapsed = time.perf_counter() - start

        per_perm_ms = (elapsed / 20) * 1000
        print(f"\npermutation_test (15k candles, 20 perms): {elapsed:.2f}s total, {per_perm_ms:.0f} ms/perm")
        assert isinstance(result, PermutationTestResult)
        assert result.n_permutations == 20
