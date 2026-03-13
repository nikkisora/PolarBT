"""Tests for Phase 0: Unified long-format engine, BacktestContext, rebalance, WeightStrategy."""

import polars as pl
import pytest

from polarbt import indicators as ind
from polarbt.core import (
    BacktestContext,
    Engine,
    Portfolio,
    Strategy,
    WeightStrategy,
    _RowAccessor,
)

# ---------------------------------------------------------------------------
# _RowAccessor
# ---------------------------------------------------------------------------


class TestRowAccessor:
    def test_single_symbol_getitem(self):
        data = {"asset": {"close": 100.0, "sma": 95.0}}
        acc = _RowAccessor(data, ["asset"])
        assert acc["close"] == 100.0
        assert acc["sma"] == 95.0

    def test_single_symbol_get(self):
        data = {"asset": {"close": 100.0}}
        acc = _RowAccessor(data, ["asset"])
        assert acc.get("close") == 100.0
        assert acc.get("missing", 42) == 42

    def test_single_symbol_contains(self):
        data = {"asset": {"close": 100.0}}
        acc = _RowAccessor(data, ["asset"])
        assert "close" in acc
        assert "missing" not in acc

    def test_single_symbol_keys_values_items(self):
        data = {"asset": {"close": 100.0, "volume": 500}}
        acc = _RowAccessor(data, ["asset"])
        assert set(acc.keys()) == {"close", "volume"}
        assert set(acc.values()) == {100.0, 500}
        assert dict(acc.items()) == {"close": 100.0, "volume": 500}

    def test_single_symbol_call_no_arg(self):
        data = {"asset": {"close": 100.0}}
        acc = _RowAccessor(data, ["asset"])
        assert acc()["close"] == 100.0

    def test_single_symbol_call_with_arg(self):
        data = {"asset": {"close": 100.0}}
        acc = _RowAccessor(data, ["asset"])
        assert acc("asset")["close"] == 100.0

    def test_multi_symbol_getitem_raises(self):
        data = {"BTC": {"close": 50000}, "ETH": {"close": 3000}}
        acc = _RowAccessor(data, ["BTC", "ETH"])
        with pytest.raises(KeyError, match="Ambiguous"):
            acc["close"]

    def test_multi_symbol_call_specific(self):
        data = {"BTC": {"close": 50000}, "ETH": {"close": 3000}}
        acc = _RowAccessor(data, ["BTC", "ETH"])
        assert acc("BTC")["close"] == 50000
        assert acc("ETH")["close"] == 3000

    def test_multi_symbol_call_no_arg_raises(self):
        data = {"BTC": {"close": 50000}, "ETH": {"close": 3000}}
        acc = _RowAccessor(data, ["BTC", "ETH"])
        with pytest.raises(ValueError, match="Must specify symbol"):
            acc()

    def test_call_missing_symbol_raises(self):
        data = {"BTC": {"close": 50000}}
        acc = _RowAccessor(data, ["BTC"])
        with pytest.raises(KeyError, match="not available"):
            acc("ETH")


# ---------------------------------------------------------------------------
# BacktestContext
# ---------------------------------------------------------------------------


class TestBacktestContextNew:
    def test_single_asset_backward_compat(self):
        """ctx.row['close'] works for single-asset mode."""
        portfolio = Portfolio()
        portfolio.update_prices({"asset": 100})
        bar_data = {"asset": {"close": 100, "sma": 95}}
        ctx = BacktestContext(
            timestamp=0,
            bar_index=5,
            portfolio=portfolio,
            symbols=["asset"],
            data=bar_data,
        )
        assert ctx.row["close"] == 100
        assert ctx.row["sma"] == 95

    def test_multi_asset_ctx_row_call(self):
        """ctx.row('BTC')['close'] works for multi-asset mode."""
        portfolio = Portfolio()
        portfolio.update_prices({"BTC": 50000, "ETH": 3000})
        bar_data = {
            "BTC": {"close": 50000, "volume": 100},
            "ETH": {"close": 3000, "volume": 200},
        }
        ctx = BacktestContext(
            timestamp=0,
            bar_index=0,
            portfolio=portfolio,
            symbols=["BTC", "ETH"],
            data=bar_data,
        )
        assert ctx.row("BTC")["close"] == 50000
        assert ctx.row("ETH")["volume"] == 200

    def test_symbols_populated(self):
        portfolio = Portfolio()
        bar_data = {"A": {"close": 10}, "B": {"close": 20}}
        ctx = BacktestContext(
            timestamp=0,
            bar_index=0,
            portfolio=portfolio,
            symbols=["A", "B"],
            data=bar_data,
        )
        assert ctx.symbols == ["A", "B"]

    def test_data_dict_access(self):
        portfolio = Portfolio()
        bar_data = {"X": {"close": 42, "open": 40}}
        ctx = BacktestContext(
            timestamp=0,
            bar_index=0,
            portfolio=portfolio,
            symbols=["X"],
            data=bar_data,
        )
        assert ctx.data["X"]["close"] == 42
        assert ctx.data["X"]["open"] == 40


# ---------------------------------------------------------------------------
# Engine: Form A (single DataFrame, no symbol column)
# ---------------------------------------------------------------------------


class TestEngineFormA:
    def test_single_asset_unchanged(self):
        """Existing single-asset API works unchanged."""

        class SMAStrategy(Strategy):
            def preprocess(self, df):
                return df.with_columns(ind.sma("close", 5).over("symbol").alias("sma"))

            def next(self, ctx):
                if ctx.row["close"] > ctx.row["sma"]:
                    ctx.portfolio.order_target_percent("asset", 1.0)
                else:
                    ctx.portfolio.close_position("asset")

        df = pl.DataFrame(
            {
                "timestamp": range(50),
                "close": [100.0 + i * 2.0 for i in range(50)],
            }
        )
        engine = Engine(SMAStrategy(), df, initial_cash=100_000)
        results = engine.run()
        assert results.total_return > 0


# ---------------------------------------------------------------------------
# Engine: Form B (dict of DataFrames)
# ---------------------------------------------------------------------------


class TestEngineFormB:
    def test_dict_input_multi_asset(self):
        """dict input is converted to long format; OHLC preserved."""
        btc_df = pl.DataFrame(
            {
                "timestamp": range(30),
                "open": [49000.0 + i * 100 for i in range(30)],
                "high": [51000.0 + i * 100 for i in range(30)],
                "low": [48000.0 + i * 100 for i in range(30)],
                "close": [50000.0 + i * 100 for i in range(30)],
                "volume": [1000] * 30,
            }
        )
        eth_df = pl.DataFrame(
            {
                "timestamp": range(30),
                "open": [2900.0 + i * 10 for i in range(30)],
                "high": [3100.0 + i * 10 for i in range(30)],
                "low": [2800.0 + i * 10 for i in range(30)],
                "close": [3000.0 + i * 10 for i in range(30)],
                "volume": [2000] * 30,
            }
        )

        class AllocStrategy(Strategy):
            def preprocess(self, df):
                return df

            def next(self, ctx):
                assert "BTC" in ctx.symbols or "ETH" in ctx.symbols
                ctx.portfolio.order_target_percent("BTC", 0.5)
                ctx.portfolio.order_target_percent("ETH", 0.5)

        engine = Engine(AllocStrategy(), {"BTC": btc_df, "ETH": eth_df}, initial_cash=100_000)
        results = engine.run()
        assert results.final_equity > 0
        # Should have positions in both assets
        assert "BTC" in results.final_positions or "ETH" in results.final_positions

    def test_dict_input_ohlc_preserved(self):
        """Form B preserves OHLC for stop-loss/take-profit."""
        btc_df = pl.DataFrame(
            {
                "timestamp": range(10),
                "open": [100.0] * 10,
                "high": [110.0] * 10,
                "low": [90.0] * 10,
                "close": [100.0] * 10,
            }
        )

        class BuyAndSetStop(Strategy):
            def preprocess(self, df):
                return df

            def next(self, ctx):
                if ctx.bar_index == 0:
                    ctx.portfolio.order("BTC", 10)
                    ctx.portfolio.set_stop_loss("BTC", stop_price=85.0)

        engine = Engine(BuyAndSetStop(), {"BTC": btc_df}, initial_cash=100_000)
        results = engine.run()
        # Stop at 85 should not trigger since low is 90
        assert "BTC" in results.final_positions


# ---------------------------------------------------------------------------
# Engine: Form C (single DataFrame with symbol column)
# ---------------------------------------------------------------------------


class TestEngineFormC:
    def test_long_format_input(self):
        """DataFrame with 'symbol' column is used directly as long format."""
        rows = []
        for i in range(30):
            for sym, base_price in [("A", 100.0), ("B", 200.0)]:
                rows.append(
                    {
                        "timestamp": i,
                        "symbol": sym,
                        "open": base_price + i - 1,
                        "high": base_price + i + 5,
                        "low": base_price + i - 5,
                        "close": base_price + i,
                        "volume": 1000,
                    }
                )
        df = pl.DataFrame(rows)

        class LongFormatStrategy(Strategy):
            def preprocess(self, df):
                return df.with_columns(
                    ind.sma("close", 5).over("symbol").alias("sma"),
                )

            def next(self, ctx):
                for sym in ctx.symbols:
                    row = ctx.row(sym)
                    if row["sma"] is not None and row["close"] > row["sma"]:
                        ctx.portfolio.order_target_percent(sym, 0.3)

        engine = Engine(LongFormatStrategy(), df, initial_cash=100_000)
        results = engine.run()
        assert results.final_equity > 0


# ---------------------------------------------------------------------------
# Preprocess with .over("symbol")
# ---------------------------------------------------------------------------


class TestPreprocessOverSymbol:
    def test_indicators_with_over_symbol(self):
        """Indicators using .over('symbol') work correctly in long format."""
        btc_df = pl.DataFrame(
            {
                "timestamp": range(30),
                "close": [50000.0 + i * 100 for i in range(30)],
            }
        )
        eth_df = pl.DataFrame(
            {
                "timestamp": range(30),
                "close": [3000.0 + i * 10 for i in range(30)],
            }
        )

        class IndicatorStrategy(Strategy):
            def preprocess(self, df):
                return df.with_columns(
                    ind.sma("close", 5).over("symbol").alias("sma_5"),
                    ind.rsi("close", 14).over("symbol").alias("rsi"),
                )

            def next(self, ctx):
                for sym in ctx.symbols:
                    row = ctx.row(sym)
                    if row.get("sma_5") is not None:
                        ctx.portfolio.order_target_percent(sym, 0.3)

        engine = Engine(IndicatorStrategy(), {"BTC": btc_df, "ETH": eth_df}, initial_cash=100_000)
        results = engine.run()
        assert results.final_equity > 0


# ---------------------------------------------------------------------------
# Portfolio.rebalance()
# ---------------------------------------------------------------------------


class TestPortfolioRebalance:
    def test_basic_rebalance(self):
        """Rebalance allocates to target weights."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0)
        portfolio.update_prices({"A": 100.0, "B": 200.0})

        portfolio.rebalance({"A": 0.5, "B": 0.5})

        value = portfolio.get_value()
        a_value = portfolio.get_position("A") * 100.0
        b_value = portfolio.get_position("B") * 200.0
        assert abs(a_value / value - 0.5) < 0.02
        assert abs(b_value / value - 0.5) < 0.02

    def test_rebalance_closes_unlisted_positions(self):
        """Positions not in target weights are closed."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0)
        portfolio.update_prices({"A": 100.0, "B": 200.0})
        portfolio.order("A", 100)  # 10k in A

        portfolio.rebalance({"B": 1.0})

        assert portfolio.get_position("A") == 0.0
        assert portfolio.get_position("B") > 0

    def test_rebalance_atomic_snapshot(self):
        """All orders are sized from a single portfolio snapshot."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.0, commission=0.0)
        portfolio.update_prices({"A": 100.0, "B": 100.0})

        portfolio.rebalance({"A": 0.5, "B": 0.5})

        a_val = portfolio.get_position("A") * 100.0
        b_val = portfolio.get_position("B") * 100.0

        # Both should be close to 50%
        assert abs(a_val - b_val) < 1.0  # nearly equal

    def test_rebalance_with_fees(self):
        """Rebalance with fees still doesn't exceed cash."""
        portfolio = Portfolio(initial_cash=100_000, slippage=0.001, commission=0.001)
        portfolio.update_prices({"A": 50.0, "B": 100.0})

        portfolio.rebalance({"A": 0.5, "B": 0.5})
        assert portfolio.cash >= -0.01  # Cash should not go negative

    def test_rebalance_sells_before_buys(self):
        """Rebalancing sells first to free cash for buys."""
        portfolio = Portfolio(initial_cash=10_000, slippage=0.0, commission=0.0)
        portfolio.update_prices({"A": 100.0, "B": 100.0})
        portfolio.order("A", 100)  # Spend all cash on A
        assert portfolio.cash == 0.0

        # Rebalance: move 50% from A to B
        portfolio.rebalance({"A": 0.5, "B": 0.5})
        assert portfolio.get_position("B") > 0


# ---------------------------------------------------------------------------
# WeightStrategy
# ---------------------------------------------------------------------------


class TestWeightStrategy:
    def test_basic_weight_strategy(self):
        """WeightStrategy base class works end-to-end."""

        class EqualWeight(WeightStrategy):
            def preprocess(self, df):
                return df

            def get_weights(self, ctx):
                n = len(ctx.symbols)
                return dict.fromkeys(ctx.symbols, 1.0 / n) if n > 0 else {}

        btc_df = pl.DataFrame(
            {
                "timestamp": range(20),
                "close": [50000.0 + i * 100 for i in range(20)],
            }
        )
        eth_df = pl.DataFrame(
            {
                "timestamp": range(20),
                "close": [3000.0 + i * 10 for i in range(20)],
            }
        )

        engine = Engine(EqualWeight(), {"BTC": btc_df, "ETH": eth_df}, initial_cash=100_000)
        results = engine.run()
        assert results.final_equity > 0
        assert len(results.final_positions) > 0

    def test_weight_strategy_single_asset(self):
        """WeightStrategy works with single-asset data."""

        class AllIn(WeightStrategy):
            def preprocess(self, df):
                return df

            def get_weights(self, ctx):
                return {"asset": 1.0}

        df = pl.DataFrame(
            {
                "timestamp": range(20),
                "close": [100.0 + i for i in range(20)],
            }
        )
        engine = Engine(AllIn(), df, initial_cash=100_000, commission=0.0, slippage=0.0)
        results = engine.run()
        assert results.total_return > 0

    def test_weight_strategy_conditional(self):
        """WeightStrategy can use indicator data."""

        class MomentumWeight(WeightStrategy):
            def preprocess(self, df):
                return df.with_columns(
                    ind.sma("close", 5).over("symbol").alias("sma_5"),
                )

            def get_weights(self, ctx):
                weights = {}
                for sym in ctx.symbols:
                    row = ctx.row(sym)
                    if row.get("sma_5") is not None and row["close"] > row["sma_5"]:
                        weights[sym] = 0.5
                return weights

        btc_df = pl.DataFrame(
            {
                "timestamp": range(30),
                "close": [50000.0 + i * 100 for i in range(30)],
            }
        )
        engine = Engine(MomentumWeight(), {"BTC": btc_df}, initial_cash=100_000)
        results = engine.run()
        assert results.final_equity > 0


# ---------------------------------------------------------------------------
# Multi-asset stop-loss/take-profit with OHLC (verifying the fix)
# ---------------------------------------------------------------------------


class TestMultiAssetOHLC:
    def test_multi_asset_stop_loss_uses_ohlc(self):
        """Multi-asset mode: stop-loss uses OHLC data (previously broken)."""

        class BuyAndStop(Strategy):
            def preprocess(self, df):
                return df

            def next(self, ctx):
                if ctx.bar_index == 0:
                    ctx.portfolio.order("BTC", 1)
                    ctx.portfolio.set_stop_loss("BTC", stop_price=49000.0)

        btc_df = pl.DataFrame(
            {
                "timestamp": range(5),
                "open": [50000.0, 50000.0, 50000.0, 50000.0, 50000.0],
                "high": [51000.0, 51000.0, 51000.0, 51000.0, 51000.0],
                "low": [49500.0, 49500.0, 48000.0, 49500.0, 49500.0],
                "close": [50000.0, 50000.0, 49000.0, 50000.0, 50000.0],
            }
        )

        engine = Engine(BuyAndStop(), {"BTC": btc_df}, initial_cash=100_000, slippage=0.0)
        results = engine.run()

        # Stop at 49000 should trigger on bar 2 where low=48000
        trades = results.trades
        assert len(trades) == 1
        # Exit should be at 49000 (stop price), not close
        assert trades["exit_price"][0] == pytest.approx(49000.0, abs=1.0)
