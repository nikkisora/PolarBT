"""Tests for trade-level data validation and aggregation (Phase 1)."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from polarbt.data.trades import (
    aggregate_trades,
    aggregate_trades_by_count,
    validate_trades,
)

REAL_DATA_PATH = Path("/mnt/d/downloads/02_datasets/pumpfun_standard.parquet")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_trades() -> pl.DataFrame:
    """Minimal valid trade DataFrame with two symbols."""
    base = datetime(2025, 1, 1, 12, 0, 0)
    return pl.DataFrame(
        {
            "timestamp": [base + timedelta(seconds=i) for i in range(10)],
            "symbol": ["AAA"] * 5 + ["BBB"] * 5,
            "price": [1.0, 1.1, 1.2, 1.15, 1.05, 2.0, 2.1, 2.3, 2.2, 2.0],
            "amount": [10.0] * 10,
            "side": ["buy", "sell", "buy", "buy", "sell", "sell", "buy", "buy", "sell", "sell"],
        },
        schema={
            "timestamp": pl.Datetime("us"),
            "symbol": pl.String,
            "price": pl.Float64,
            "amount": pl.Float64,
            "side": pl.String,
        },
    )


@pytest.fixture
def trades_with_optional(simple_trades: pl.DataFrame) -> pl.DataFrame:
    """Trade DataFrame with optional trader and pool_reserve columns."""
    return simple_trades.with_columns(
        pl.lit("wallet_a").alias("trader"),
        pl.lit(1000.0).alias("pool_reserve"),
    )


@pytest.fixture
def real_trades() -> pl.DataFrame | None:
    """Load a small slice of the real pumpfun dataset if available."""
    if not REAL_DATA_PATH.exists():
        pytest.skip("Real parquet data not available")
    df = pl.read_parquet(REAL_DATA_PATH)
    # Take a manageable slice: one symbol with most trades
    top_symbol = df.group_by("symbol").len().sort("len", descending=True).head(1)["symbol"][0]
    return df.filter(pl.col("symbol") == top_symbol).head(10_000)


# ===========================================================================
# 1.1 Validation tests
# ===========================================================================


class TestValidateTrades:
    def test_valid_data(self, simple_trades: pl.DataFrame) -> None:
        result = validate_trades(simple_trades)
        assert result.valid
        assert result.errors == []

    def test_missing_columns(self) -> None:
        df = pl.DataFrame({"timestamp": [datetime.now()], "price": [1.0]})
        result = validate_trades(df)
        assert not result.valid
        assert any("Missing required columns" in e for e in result.errors)

    def test_invalid_side_values(self, simple_trades: pl.DataFrame) -> None:
        df = simple_trades.with_columns(pl.lit("unknown").alias("side"))
        result = validate_trades(df)
        assert not result.valid
        assert any("invalid values" in e for e in result.errors)

    def test_negative_price(self, simple_trades: pl.DataFrame) -> None:
        df = simple_trades.with_columns(
            pl.when(pl.col("price") == pl.col("price").first())
            .then(pl.lit(-1.0))
            .otherwise(pl.col("price"))
            .alias("price")
        )
        result = validate_trades(df)
        assert not result.valid
        assert any("non-positive" in e for e in result.errors)

    def test_zero_amount(self, simple_trades: pl.DataFrame) -> None:
        df = simple_trades.with_columns(pl.lit(0.0).alias("amount"))
        result = validate_trades(df)
        assert not result.valid
        assert any("non-positive" in e and "amount" in e for e in result.errors)

    def test_null_in_required_column(self, simple_trades: pl.DataFrame) -> None:
        df = simple_trades.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("price"))
            .alias("price")
        )
        result = validate_trades(df)
        assert not result.valid
        assert any("null" in e for e in result.errors)

    def test_unsorted_timestamps(self) -> None:
        base = datetime(2025, 1, 1)
        df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(seconds=s) for s in [3, 1, 2]],
                "symbol": ["X"] * 3,
                "price": [1.0, 2.0, 3.0],
                "amount": [1.0, 1.0, 1.0],
                "side": ["buy", "sell", "buy"],
            },
            schema={
                "timestamp": pl.Datetime("us"),
                "symbol": pl.String,
                "price": pl.Float64,
                "amount": pl.Float64,
                "side": pl.String,
            },
        )
        result = validate_trades(df)
        assert not result.valid
        assert any("not sorted" in e for e in result.errors)

    def test_duplicate_tx_id_warns(self, simple_trades: pl.DataFrame) -> None:
        df = simple_trades.with_columns(pl.lit("same_tx").alias("tx_id"))
        result = validate_trades(df)
        assert result.valid  # duplicates are warnings, not errors
        assert any("duplicate" in w for w in result.warnings)

    def test_wrong_dtype(self) -> None:
        base = datetime(2025, 1, 1)
        df = pl.DataFrame(
            {
                "timestamp": [base],
                "symbol": ["X"],
                "price": ["not_a_number"],
                "amount": [1.0],
                "side": ["buy"],
            }
        )
        result = validate_trades(df)
        assert not result.valid
        assert any("dtype" in e for e in result.errors)

    def test_real_data_validates(self, real_trades: pl.DataFrame) -> None:
        # Real data may not be pre-sorted per symbol; sort first
        sorted_df = real_trades.sort("symbol", "timestamp")
        result = validate_trades(sorted_df)
        assert result.valid, f"Validation errors: {result.errors}"


# ===========================================================================
# 1.2 Time-based aggregation tests
# ===========================================================================


class TestAggregateTradesTime:
    def test_basic_ohlcv_output(self, simple_trades: pl.DataFrame) -> None:
        bars = aggregate_trades(simple_trades, "5s")
        expected_cols = {
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trades",
            "buy_volume",
            "sell_volume",
            "vwap",
        }
        assert expected_cols.issubset(set(bars.columns))

    def test_ohlc_correctness(self) -> None:
        base = datetime(2025, 1, 1)
        df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(seconds=i) for i in range(4)],
                "symbol": ["X"] * 4,
                "price": [10.0, 15.0, 5.0, 12.0],
                "amount": [1.0, 2.0, 3.0, 4.0],
                "side": ["buy", "buy", "sell", "buy"],
            },
            schema={
                "timestamp": pl.Datetime("us"),
                "symbol": pl.String,
                "price": pl.Float64,
                "amount": pl.Float64,
                "side": pl.String,
            },
        )
        bars = aggregate_trades(df, "1h")
        assert bars.height == 1
        row = bars.row(0, named=True)
        assert row["open"] == 10.0
        assert row["high"] == 15.0
        assert row["low"] == 5.0
        assert row["close"] == 12.0
        assert row["volume"] == 10.0
        assert row["trades"] == 4
        assert row["buy_volume"] == 7.0  # 1+2+4
        assert row["sell_volume"] == 3.0

    def test_vwap_calculation(self) -> None:
        base = datetime(2025, 1, 1)
        df = pl.DataFrame(
            {
                "timestamp": [base, base + timedelta(seconds=1)],
                "symbol": ["X", "X"],
                "price": [10.0, 20.0],
                "amount": [3.0, 1.0],
                "side": ["buy", "sell"],
            },
            schema={
                "timestamp": pl.Datetime("us"),
                "symbol": pl.String,
                "price": pl.Float64,
                "amount": pl.Float64,
                "side": pl.String,
            },
        )
        bars = aggregate_trades(df, "1h")
        expected_vwap = (10.0 * 3.0 + 20.0 * 1.0) / (3.0 + 1.0)
        assert bars["vwap"][0] == pytest.approx(expected_vwap)

    def test_multi_symbol_separation(self, simple_trades: pl.DataFrame) -> None:
        bars = aggregate_trades(simple_trades, "1h")
        symbols = set(bars["symbol"].to_list())
        assert symbols == {"AAA", "BBB"}

    def test_min_trades_filter(self, simple_trades: pl.DataFrame) -> None:
        bars_all = aggregate_trades(simple_trades, "1s")
        bars_filtered = aggregate_trades(simple_trades, "1s", min_trades=2)
        assert bars_filtered.height <= bars_all.height

    def test_optional_trader_column(self, trades_with_optional: pl.DataFrame) -> None:
        bars = aggregate_trades(trades_with_optional, "1h")
        assert "unique_traders" in bars.columns

    def test_optional_pool_reserve_column(self, trades_with_optional: pl.DataFrame) -> None:
        bars = aggregate_trades(trades_with_optional, "1h")
        assert "pool_reserve_last" in bars.columns

    def test_extra_aggs(self, simple_trades: pl.DataFrame) -> None:
        bars = aggregate_trades(
            simple_trades,
            "1h",
            extra_aggs={"price_std": pl.col("price").std()},
        )
        assert "price_std" in bars.columns

    def test_exchange_rate_conversion(self, simple_trades: pl.DataFrame) -> None:
        rate_df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1)],
                "rate": [150.0],
            },
            schema={"timestamp": pl.Datetime("us"), "rate": pl.Float64},
        )
        bars = aggregate_trades(simple_trades, "1h", exchange_rate=rate_df)
        assert "volume_usd" in bars.columns
        assert "buy_volume_usd" in bars.columns
        assert "sell_volume_usd" in bars.columns
        # USD volume should be quote volume * rate
        for row in bars.iter_rows(named=True):
            assert row["volume_usd"] == pytest.approx(row["volume"] * 150.0)

    def test_high_gte_low(self, simple_trades: pl.DataFrame) -> None:
        bars = aggregate_trades(simple_trades, "2s")
        violations = bars.filter(pl.col("high") < pl.col("low"))
        assert violations.height == 0

    def test_real_data_aggregation(self, real_trades: pl.DataFrame) -> None:
        bars = aggregate_trades(real_trades, "1m")
        assert bars.height > 0
        assert set(bars.columns) >= {"timestamp", "symbol", "open", "high", "low", "close", "volume", "trades", "vwap"}
        # OHLC integrity
        assert bars.filter(pl.col("high") < pl.col("low")).height == 0
        assert bars.filter(pl.col("high") < pl.col("open")).height == 0
        assert bars.filter(pl.col("low") > pl.col("close")).height == 0
        # Volume is positive
        assert bars.filter(pl.col("volume") <= 0).height == 0
        # buy_volume + sell_volume == volume
        vol_check = bars.with_columns((pl.col("buy_volume") + pl.col("sell_volume")).alias("_total"))
        assert (vol_check["_total"] - vol_check["volume"]).abs().max() < 1e-10  # type: ignore[operator]
        # Has trader column from source
        if "trader" in real_trades.columns:
            assert "unique_traders" in bars.columns


# ===========================================================================
# 1.3 Trade-count aggregation tests
# ===========================================================================


class TestAggregateTradesByCount:
    def test_basic_output(self, simple_trades: pl.DataFrame) -> None:
        bars = aggregate_trades_by_count(simple_trades, 3)
        expected_cols = {
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trades",
            "buy_volume",
            "sell_volume",
            "vwap",
        }
        assert expected_cols.issubset(set(bars.columns))

    def test_trade_count_per_bar(self) -> None:
        base = datetime(2025, 1, 1)
        df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(seconds=i) for i in range(9)],
                "symbol": ["X"] * 9,
                "price": [float(i + 1) for i in range(9)],
                "amount": [1.0] * 9,
                "side": ["buy"] * 9,
            },
            schema={
                "timestamp": pl.Datetime("us"),
                "symbol": pl.String,
                "price": pl.Float64,
                "amount": pl.Float64,
                "side": pl.String,
            },
        )
        bars = aggregate_trades_by_count(df, 3)
        assert bars.height == 3
        assert all(t == 3 for t in bars["trades"].to_list())

    def test_remainder_bar(self) -> None:
        """Last bar can have fewer than n_trades."""
        base = datetime(2025, 1, 1)
        df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(seconds=i) for i in range(7)],
                "symbol": ["X"] * 7,
                "price": [1.0] * 7,
                "amount": [1.0] * 7,
                "side": ["buy"] * 7,
            },
            schema={
                "timestamp": pl.Datetime("us"),
                "symbol": pl.String,
                "price": pl.Float64,
                "amount": pl.Float64,
                "side": pl.String,
            },
        )
        bars = aggregate_trades_by_count(df, 3)
        assert bars.height == 3  # 3 + 3 + 1
        assert bars["trades"].to_list() == [3, 3, 1]

    def test_invalid_n_trades(self, simple_trades: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="n_trades must be >= 1"):
            aggregate_trades_by_count(simple_trades, 0)

    def test_multi_symbol_separation(self, simple_trades: pl.DataFrame) -> None:
        bars = aggregate_trades_by_count(simple_trades, 2)
        for sym in bars["symbol"].unique().to_list():
            sym_bars = bars.filter(pl.col("symbol") == sym)
            # Each full bar should have exactly 2 trades (except possibly last)
            full_bars = sym_bars.filter(pl.col("trades") == 2)
            assert full_bars.height >= 1

    def test_ohlc_correctness(self) -> None:
        base = datetime(2025, 1, 1)
        df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(seconds=i) for i in range(3)],
                "symbol": ["X"] * 3,
                "price": [10.0, 15.0, 5.0],
                "amount": [1.0, 2.0, 3.0],
                "side": ["buy", "sell", "buy"],
            },
            schema={
                "timestamp": pl.Datetime("us"),
                "symbol": pl.String,
                "price": pl.Float64,
                "amount": pl.Float64,
                "side": pl.String,
            },
        )
        bars = aggregate_trades_by_count(df, 3)
        assert bars.height == 1
        row = bars.row(0, named=True)
        assert row["open"] == 10.0
        assert row["high"] == 15.0
        assert row["low"] == 5.0
        assert row["close"] == 5.0

    def test_real_data_count_bars(self, real_trades: pl.DataFrame) -> None:
        bars = aggregate_trades_by_count(real_trades, 50)
        assert bars.height > 0
        # All full bars should have exactly 50 trades
        full_bars = bars.filter(pl.col("trades") == 50)
        assert full_bars.height > 0
        # OHLC integrity
        assert bars.filter(pl.col("high") < pl.col("low")).height == 0
