"""Dynamic universe providers for filtering tradeable symbols per bar.

Provides a protocol and built-in implementations for controlling which symbols
are available to a strategy on each bar. Useful for large/dynamic universes
where tokens appear and disappear over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class UniverseContext:
    """Per-bar context passed to universe providers.

    Attributes:
        timestamp: Current bar timestamp.
        bar_index: Current bar index (0-based).
        available_symbols: All symbols with data on this bar.
        bar_data: Per-symbol bar data ``{symbol: {col: value, ...}}``.
        first_seen_bar: Bar index when each symbol first appeared.
        bar_count: Number of bars each symbol has been active.
    """

    timestamp: Any
    bar_index: int
    available_symbols: list[str]
    bar_data: dict[str, dict[str, Any]]
    first_seen_bar: dict[str, int]
    bar_count: dict[str, int]


@runtime_checkable
class UniverseProvider(Protocol):
    """Protocol for filtering tradeable symbols each bar."""

    def get_universe(self, ctx: UniverseContext) -> list[str]:
        """Return the subset of symbols the strategy is allowed to trade this bar.

        Args:
            ctx: Universe context with current bar data and symbol lifecycle info.

        Returns:
            Filtered list of tradeable symbols.
        """
        ...


class AllSymbols:
    """Pass through all available symbols."""

    def get_universe(self, ctx: UniverseContext) -> list[str]:
        """Return all available symbols."""
        return ctx.available_symbols


class AgeFilter:
    """Only include symbols that have existed for at least ``min_bars`` bars.

    Args:
        min_bars: Minimum number of bars a symbol must have been active.
    """

    def __init__(self, min_bars: int) -> None:
        if min_bars < 1:
            raise ValueError(f"min_bars must be >= 1, got {min_bars}")
        self.min_bars = min_bars

    def get_universe(self, ctx: UniverseContext) -> list[str]:
        """Return symbols that have been active for at least min_bars."""
        return [s for s in ctx.available_symbols if ctx.bar_count.get(s, 0) >= self.min_bars]


class VolumeFilter:
    """Only include symbols whose recent volume exceeds a threshold.

    Evaluates the current bar's volume column value against ``min_volume``.
    For rolling lookback filtering, use in ``preprocess()`` to create a
    rolling volume column, then reference it via ``volume_col``.

    Args:
        min_volume: Minimum volume threshold.
        volume_col: Column name to read volume from. Defaults to ``"volume"``.
    """

    def __init__(self, min_volume: float, volume_col: str = "volume") -> None:
        if min_volume < 0:
            raise ValueError(f"min_volume must be >= 0, got {min_volume}")
        self.min_volume = min_volume
        self.volume_col = volume_col

    def get_universe(self, ctx: UniverseContext) -> list[str]:
        """Return symbols whose volume meets the threshold."""
        result: list[str] = []
        for sym in ctx.available_symbols:
            row = ctx.bar_data.get(sym, {})
            vol = row.get(self.volume_col)
            if vol is not None and float(vol) >= self.min_volume:
                result.append(sym)
        return result


class TopN:
    """Keep only the top N symbols ranked by a metric column.

    Args:
        n: Number of symbols to keep.
        sort_by: Column name to rank by (descending). Defaults to ``"volume"``.
    """

    def __init__(self, n: int, sort_by: str = "volume") -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n
        self.sort_by = sort_by

    def get_universe(self, ctx: UniverseContext) -> list[str]:
        """Return the top N symbols by the sort column."""
        scored: list[tuple[str, float]] = []
        for sym in ctx.available_symbols:
            row = ctx.bar_data.get(sym, {})
            val = row.get(self.sort_by)
            if val is not None:
                scored.append((sym, float(val)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[: self.n]]


@dataclass
class CompositeFilter:
    """Chain multiple universe providers. Each filter narrows the previous result.

    Args:
        filters: Sequence of UniverseProvider instances applied in order.
    """

    filters: list[UniverseProvider] = field(default_factory=list)

    def __init__(self, *filters: UniverseProvider) -> None:
        self.filters = list(filters)

    def get_universe(self, ctx: UniverseContext) -> list[str]:
        """Apply all filters in sequence, each narrowing the universe."""
        symbols = ctx.available_symbols
        for f in self.filters:
            filtered_ctx = UniverseContext(
                timestamp=ctx.timestamp,
                bar_index=ctx.bar_index,
                available_symbols=symbols,
                bar_data=ctx.bar_data,
                first_seen_bar=ctx.first_seen_bar,
                bar_count=ctx.bar_count,
            )
            symbols = f.get_universe(filtered_ctx)
        return symbols
