"""Tests for dynamic universe support (Phase 2)."""

from datetime import datetime, timedelta
from typing import Any

import polars as pl
import pytest

from polarbt.core import BacktestContext, Engine, Strategy
from polarbt.universe import (
    AgeFilter,
    AllSymbols,
    CompositeFilter,
    TopN,
    UniverseContext,
    UniverseProvider,
    VolumeFilter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_universe_ctx(
    available_symbols: list[str],
    bar_data: dict[str, dict[str, Any]] | None = None,
    first_seen_bar: dict[str, int] | None = None,
    bar_count: dict[str, int] | None = None,
    bar_index: int = 10,
) -> UniverseContext:
    """Helper to build a UniverseContext for unit tests."""
    if bar_data is None:
        bar_data = {s: {"close": 1.0, "volume": 100.0} for s in available_symbols}
    return UniverseContext(
        timestamp=datetime(2025, 6, 1),
        bar_index=bar_index,
        available_symbols=available_symbols,
        bar_data=bar_data,
        first_seen_bar=first_seen_bar or dict.fromkeys(available_symbols, 0),
        bar_count=bar_count or dict.fromkeys(available_symbols, bar_index + 1),
    )


def _make_multi_asset_df(
    symbols: list[str],
    bars_per_symbol: int = 10,
    volume_per_symbol: dict[str, float] | None = None,
    start_bar: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Build a long-format DataFrame with multiple symbols for engine tests."""
    rows: list[dict[str, Any]] = []
    base = datetime(2025, 1, 1)
    start_bar = start_bar or {}
    volume_per_symbol = volume_per_symbol or {}

    for bar_idx in range(bars_per_symbol):
        ts = base + timedelta(hours=bar_idx)
        for sym in symbols:
            sym_start = start_bar.get(sym, 0)
            if bar_idx < sym_start:
                continue
            vol = volume_per_symbol.get(sym, 100.0)
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "open": 10.0 + bar_idx,
                    "high": 11.0 + bar_idx,
                    "low": 9.0 + bar_idx,
                    "close": 10.5 + bar_idx,
                    "volume": vol,
                }
            )

    return pl.DataFrame(
        rows,
        schema={
            "timestamp": pl.Datetime("us"),
            "symbol": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )


class RecordingStrategy(Strategy):
    """Strategy that records ctx.symbols and lifecycle data each bar."""

    def __init__(self) -> None:
        self.recorded_symbols: list[list[str]] = []
        self.recorded_available: list[list[str]] = []
        self.recorded_first_seen: list[dict[str, int]] = []
        self.recorded_bar_count: list[dict[str, int]] = []

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def next(self, ctx: BacktestContext) -> None:
        self.recorded_symbols.append(list(ctx.symbols))
        self.recorded_available.append(list(ctx.available_symbols))
        self.recorded_first_seen.append(dict(ctx.first_seen_bar))
        self.recorded_bar_count.append(dict(ctx.bar_count))


# ===========================================================================
# 2.1 Universe Provider unit tests
# ===========================================================================


class TestAllSymbols:
    def test_returns_all(self) -> None:
        provider = AllSymbols()
        ctx = _make_universe_ctx(["A", "B", "C"])
        assert provider.get_universe(ctx) == ["A", "B", "C"]

    def test_empty(self) -> None:
        provider = AllSymbols()
        ctx = _make_universe_ctx([])
        assert provider.get_universe(ctx) == []


class TestAgeFilter:
    def test_filters_young_symbols(self) -> None:
        provider = AgeFilter(min_bars=5)
        ctx = _make_universe_ctx(
            ["OLD", "NEW"],
            bar_count={"OLD": 10, "NEW": 3},
        )
        result = provider.get_universe(ctx)
        assert result == ["OLD"]

    def test_includes_at_threshold(self) -> None:
        provider = AgeFilter(min_bars=5)
        ctx = _make_universe_ctx(["X"], bar_count={"X": 5})
        assert provider.get_universe(ctx) == ["X"]

    def test_excludes_below_threshold(self) -> None:
        provider = AgeFilter(min_bars=5)
        ctx = _make_universe_ctx(["X"], bar_count={"X": 4})
        assert provider.get_universe(ctx) == []

    def test_invalid_min_bars(self) -> None:
        with pytest.raises(ValueError, match="min_bars must be >= 1"):
            AgeFilter(min_bars=0)


class TestVolumeFilter:
    def test_filters_low_volume(self) -> None:
        provider = VolumeFilter(min_volume=50.0)
        ctx = _make_universe_ctx(
            ["HIGH", "LOW"],
            bar_data={"HIGH": {"volume": 100.0}, "LOW": {"volume": 10.0}},
        )
        result = provider.get_universe(ctx)
        assert result == ["HIGH"]

    def test_custom_volume_column(self) -> None:
        provider = VolumeFilter(min_volume=50.0, volume_col="dollar_volume")
        ctx = _make_universe_ctx(
            ["A"],
            bar_data={"A": {"dollar_volume": 100.0}},
        )
        assert provider.get_universe(ctx) == ["A"]

    def test_missing_volume_excludes(self) -> None:
        provider = VolumeFilter(min_volume=50.0)
        ctx = _make_universe_ctx(
            ["A"],
            bar_data={"A": {"close": 1.0}},
        )
        assert provider.get_universe(ctx) == []

    def test_invalid_min_volume(self) -> None:
        with pytest.raises(ValueError, match="min_volume must be >= 0"):
            VolumeFilter(min_volume=-1.0)


class TestTopN:
    def test_top_n_by_volume(self) -> None:
        provider = TopN(n=2, sort_by="volume")
        ctx = _make_universe_ctx(
            ["A", "B", "C"],
            bar_data={
                "A": {"volume": 300.0},
                "B": {"volume": 100.0},
                "C": {"volume": 200.0},
            },
        )
        result = provider.get_universe(ctx)
        assert result == ["A", "C"]

    def test_n_larger_than_available(self) -> None:
        provider = TopN(n=10, sort_by="volume")
        ctx = _make_universe_ctx(
            ["A", "B"],
            bar_data={"A": {"volume": 100.0}, "B": {"volume": 200.0}},
        )
        result = provider.get_universe(ctx)
        assert len(result) == 2

    def test_custom_sort_column(self) -> None:
        provider = TopN(n=1, sort_by="market_cap")
        ctx = _make_universe_ctx(
            ["A", "B"],
            bar_data={"A": {"market_cap": 500.0}, "B": {"market_cap": 1000.0}},
        )
        assert provider.get_universe(ctx) == ["B"]

    def test_invalid_n(self) -> None:
        with pytest.raises(ValueError, match="n must be >= 1"):
            TopN(n=0)


class TestCompositeFilter:
    def test_chains_filters(self) -> None:
        provider = CompositeFilter(
            AgeFilter(min_bars=5),
            VolumeFilter(min_volume=50.0),
        )
        ctx = _make_universe_ctx(
            ["OLD_HIGH", "OLD_LOW", "NEW_HIGH"],
            bar_data={
                "OLD_HIGH": {"volume": 100.0},
                "OLD_LOW": {"volume": 10.0},
                "NEW_HIGH": {"volume": 100.0},
            },
            bar_count={"OLD_HIGH": 10, "OLD_LOW": 10, "NEW_HIGH": 2},
        )
        result = provider.get_universe(ctx)
        assert result == ["OLD_HIGH"]

    def test_empty_filters_passes_all(self) -> None:
        provider = CompositeFilter()
        ctx = _make_universe_ctx(["A", "B"])
        assert provider.get_universe(ctx) == ["A", "B"]


class TestProtocolCompliance:
    def test_all_implementations_are_universe_providers(self) -> None:
        assert isinstance(AllSymbols(), UniverseProvider)
        assert isinstance(AgeFilter(1), UniverseProvider)
        assert isinstance(VolumeFilter(0), UniverseProvider)
        assert isinstance(TopN(1), UniverseProvider)
        assert isinstance(CompositeFilter(), UniverseProvider)


# ===========================================================================
# 2.2 Token lifecycle tracking & engine integration tests
# ===========================================================================


class TestLifecycleTracking:
    def test_first_seen_bar_tracks_appearance(self) -> None:
        # Symbol B appears only from bar 3 onward
        df = _make_multi_asset_df(["A", "B"], bars_per_symbol=6, start_bar={"B": 3})
        strategy = RecordingStrategy()
        engine = Engine(strategy, df, warmup=0)
        engine.run()

        # After all bars, first_seen_bar should show B appeared at bar 3
        last_first_seen = strategy.recorded_first_seen[-1]
        assert last_first_seen["A"] == 0
        assert last_first_seen["B"] == 3

    def test_bar_count_increments(self) -> None:
        df = _make_multi_asset_df(["A", "B"], bars_per_symbol=5, start_bar={"B": 2})
        strategy = RecordingStrategy()
        engine = Engine(strategy, df, warmup=0)
        engine.run()

        last_bar_count = strategy.recorded_bar_count[-1]
        assert last_bar_count["A"] == 5
        assert last_bar_count["B"] == 3  # appears on bars 2,3,4

    def test_available_symbols_populated(self) -> None:
        df = _make_multi_asset_df(["A", "B"], bars_per_symbol=5, start_bar={"B": 3})
        strategy = RecordingStrategy()
        engine = Engine(strategy, df, warmup=0)
        engine.run()

        # Bar 0,1,2: only A available
        assert strategy.recorded_available[0] == ["A"]
        assert strategy.recorded_available[1] == ["A"]
        assert strategy.recorded_available[2] == ["A"]
        # Bar 3,4: both available
        assert set(strategy.recorded_available[3]) == {"A", "B"}
        assert set(strategy.recorded_available[4]) == {"A", "B"}


class TestEngineUniverseIntegration:
    def test_no_provider_passes_all_symbols(self) -> None:
        df = _make_multi_asset_df(["A", "B", "C"], bars_per_symbol=5)
        strategy = RecordingStrategy()
        engine = Engine(strategy, df, warmup=0)
        engine.run()

        for symbols in strategy.recorded_symbols:
            assert set(symbols) == {"A", "B", "C"}

    def test_age_filter_excludes_young_symbols(self) -> None:
        # B appears at bar 3, filter requires min_bars=2
        df = _make_multi_asset_df(["A", "B"], bars_per_symbol=6, start_bar={"B": 3})
        strategy = RecordingStrategy()
        engine = Engine(strategy, df, warmup=0, universe_provider=AgeFilter(min_bars=2))
        engine.run()

        # Bar 3: B has bar_count=1, filtered out
        assert strategy.recorded_symbols[3] == ["A"]
        # Bar 4: B has bar_count=2, included
        assert set(strategy.recorded_symbols[4]) == {"A", "B"}

    def test_volume_filter_in_engine(self) -> None:
        df = _make_multi_asset_df(
            ["HIGH", "LOW"],
            bars_per_symbol=5,
            volume_per_symbol={"HIGH": 200.0, "LOW": 10.0},
        )
        strategy = RecordingStrategy()
        engine = Engine(strategy, df, warmup=0, universe_provider=VolumeFilter(min_volume=50.0))
        engine.run()

        for symbols in strategy.recorded_symbols:
            assert symbols == ["HIGH"]

    def test_top_n_in_engine(self) -> None:
        df = _make_multi_asset_df(
            ["A", "B", "C"],
            bars_per_symbol=5,
            volume_per_symbol={"A": 300.0, "B": 100.0, "C": 200.0},
        )
        strategy = RecordingStrategy()
        engine = Engine(strategy, df, warmup=0, universe_provider=TopN(n=2, sort_by="volume"))
        engine.run()

        for symbols in strategy.recorded_symbols:
            assert set(symbols) == {"A", "C"}

    def test_composite_filter_in_engine(self) -> None:
        df = _make_multi_asset_df(
            ["A", "B", "C"],
            bars_per_symbol=6,
            volume_per_symbol={"A": 200.0, "B": 200.0, "C": 10.0},
            start_bar={"B": 3},
        )
        strategy = RecordingStrategy()
        provider = CompositeFilter(AgeFilter(min_bars=2), VolumeFilter(min_volume=50.0))
        engine = Engine(strategy, df, warmup=0, universe_provider=provider)
        engine.run()

        # Bar 0: A has bar_count=1 (excluded by age), C filtered by volume, B absent
        assert strategy.recorded_symbols[0] == []
        # Bar 1: A has bar_count=2 (included), C still filtered by volume
        assert strategy.recorded_symbols[1] == ["A"]
        # Bar 3: B appears (bar_count=1), filtered by age
        assert strategy.recorded_symbols[3] == ["A"]
        # Bar 4: B has bar_count=2, passes both filters
        assert set(strategy.recorded_symbols[4]) == {"A", "B"}

    def test_filtered_symbol_positions_still_valued(self) -> None:
        """Portfolio should still value positions in filtered-out symbols."""
        df = _make_multi_asset_df(
            ["A", "B"],
            bars_per_symbol=5,
            volume_per_symbol={"A": 200.0, "B": 200.0},
        )

        class BuyThenFilterStrategy(Strategy):
            """Buy B on bar 0, then filter B out from bar 2 onward."""

            def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

            def next(self, ctx: BacktestContext) -> None:
                if ctx.bar_index == 0:
                    ctx.portfolio.order("B", 1.0)

        # Use a provider that filters B out after bar 1
        class LateFilter:
            def get_universe(self, ctx: UniverseContext) -> list[str]:
                if ctx.bar_index >= 2:
                    return [s for s in ctx.available_symbols if s != "B"]
                return ctx.available_symbols

        strategy = BuyThenFilterStrategy()
        engine = Engine(strategy, df, warmup=0, universe_provider=LateFilter())
        engine.run()

        # Position in B should still exist and be valued
        assert engine.portfolio is not None
        assert engine.portfolio.get_position("B") == 1.0

    def test_single_asset_backward_compat(self) -> None:
        """Single-asset strategies should still work with lifecycle fields."""
        base = datetime(2025, 1, 1)
        df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(hours=i) for i in range(5)],
                "open": [10.0] * 5,
                "high": [11.0] * 5,
                "low": [9.0] * 5,
                "close": [10.0] * 5,
                "volume": [100.0] * 5,
            },
            schema={
                "timestamp": pl.Datetime("us"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            },
        )

        strategy = RecordingStrategy()
        engine = Engine(strategy, df, warmup=0)
        engine.run()

        # Lifecycle data should be populated for the default "asset" symbol
        assert strategy.recorded_first_seen[-1]["asset"] == 0
        assert strategy.recorded_bar_count[-1]["asset"] == 5
        assert strategy.recorded_available[0] == ["asset"]
