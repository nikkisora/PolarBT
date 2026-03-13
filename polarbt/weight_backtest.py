"""Weight-based portfolio backtesting.

Provides a declarative backtester for portfolio weight allocation strategies
(momentum rotation, factor models, equal-weight baskets).  Internally uses
the unified Engine + Portfolio so all features (commission models, stop-loss
with OHLC priority, leverage, order delay, etc.) are available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import polars as pl

from polarbt.core import BacktestContext, Engine, Strategy
from polarbt.results import BacktestMetrics


@dataclass
class WeightBacktestResult:
    """Results from a weight-based backtest.

    Attributes:
        equity: DataFrame with columns ``date`` and ``cumulative_return``.
        trades: DataFrame with per-trade details.
        metrics: Standard backtest performance metrics.
        next_actions: Forward-looking stock operations.
    """

    equity: pl.DataFrame
    trades: pl.DataFrame
    metrics: BacktestMetrics
    next_actions: pl.DataFrame | None = None

    def __str__(self) -> str:
        return str(self.metrics)


# ---------------------------------------------------------------------------
# Helpers (kept as public for tests and reuse)
# ---------------------------------------------------------------------------


def _parse_offset(offset: str | None) -> int:
    """Parse an offset string like ``"2d"`` or ``"1W"`` into integer days.

    Args:
        offset: Offset string (e.g. ``"2d"``, ``"1W"``). None returns 0.

    Returns:
        Number of calendar days.
    """
    if offset is None:
        return 0
    m = re.fullmatch(r"(\d+)\s*([dDwW])", offset.strip())
    if not m:
        raise ValueError(f"Invalid offset format: {offset!r}. Use '<N>d' or '<N>W'.")
    value = int(m.group(1))
    unit = m.group(2).upper()
    return value * 7 if unit == "W" else value


def _detect_rebalance_dates(
    dates: pl.Series,
    resample: str | None,
    offset_days: int,
) -> set[Any]:
    """Return the set of dates on which rebalancing should occur.

    When *resample* is ``None`` the caller rebalances whenever weights change
    (handled externally). Otherwise period boundaries (month, week, etc.) are
    detected and optionally shifted by *offset_days* trading days.

    Args:
        dates: Sorted series of unique dates.
        resample: One of ``"D"``, ``"W"``, ``"W-FRI"``, ``"M"``, ``"Q"``, ``"Y"`` or ``None``.
        offset_days: Shift rebalance by this many trading days after the boundary.

    Returns:
        Set of dates on which rebalancing should execute.
    """
    if resample is None or resample == "D":
        return set(dates.to_list())

    date_list = dates.to_list()
    if not date_list:
        return set()

    boundary_indices: list[int] = [0]

    for i in range(1, len(date_list)):
        prev = date_list[i - 1]
        curr = date_list[i]
        is_boundary = False

        if resample == "M":
            is_boundary = curr.month != prev.month
        elif resample == "Q":
            is_boundary = (curr.month - 1) // 3 != (prev.month - 1) // 3
        elif resample == "Y":
            is_boundary = curr.year != prev.year
        elif resample in ("W", "W-FRI"):
            is_boundary = curr.isocalendar()[1] != prev.isocalendar()[1]

        if is_boundary:
            boundary_indices.append(i)

    rebalance_indices: set[int] = set()
    for idx in boundary_indices:
        shifted = idx + offset_days
        if shifted < len(date_list):
            rebalance_indices.add(shifted)

    return {date_list[i] for i in rebalance_indices}


def _normalize_weights(
    weights: dict[str, float],
    position_limit: float,
) -> dict[str, float]:
    """Normalize a weight vector.

    - Boolean-like (0/1) weights are treated as equal-weight signals.
    - If ``sum(|w|) > 1`` the weights are scaled to sum to 1.
    - Individual weights are clipped to ``[-position_limit, position_limit]``.

    Args:
        weights: Mapping of symbol to raw weight.
        position_limit: Maximum absolute weight per symbol.

    Returns:
        Normalized weight dict.
    """
    if not weights:
        return {}

    clipped = {s: max(-position_limit, min(position_limit, w)) for s, w in weights.items()}

    nonzero = {s: w for s, w in clipped.items() if w != 0}
    if nonzero and all(v == 1.0 for v in nonzero.values()):
        eq_w = 1.0 / len(nonzero)
        clipped = {s: (eq_w if w != 0 else 0.0) for s, w in clipped.items()}

    total_abs = sum(abs(w) for w in clipped.values())
    if total_abs > 1.0:
        clipped = {s: w / total_abs for s, w in clipped.items()}

    return clipped


# ---------------------------------------------------------------------------
# Internal strategy that reads weights from DataFrame columns
# ---------------------------------------------------------------------------


class _DataFrameWeightStrategy(Strategy):
    """Internal strategy that reads target weights from a DataFrame column and rebalances on schedule."""

    def __init__(
        self,
        *,
        weight_col: str,
        rebalance_dates: set[Any],
        resample: str | None,
        position_limit: float,
        stop_loss: float | None,
        take_profit: float | None,
        trail_stop: float | None,
    ) -> None:
        super().__init__()
        self._weight_col = weight_col
        self._rebalance_dates = rebalance_dates
        self._resample = resample
        self._position_limit = position_limit
        self._stop_loss = stop_loss
        self._take_profit = take_profit
        self._trail_stop = trail_stop
        self._prev_weights: dict[str, float] = {}
        self._latest_weights: dict[str, float] = {}
        # Track positions from previous bar to detect stop-outs
        self._prev_position_syms: set[str] = set()

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def next(self, ctx: BacktestContext) -> None:
        # 1. Detect symbols stopped out this bar: had position last bar, now flat
        closed_by_stop: set[str] = set()
        has_stops = self._stop_loss is not None or self._take_profit is not None or self._trail_stop is not None
        if has_stops:
            for sym in self._prev_position_syms:
                if ctx.portfolio.get_position(sym) == 0:
                    closed_by_stop.add(sym)

        # 2. Set stops on positions that just appeared (handles order_delay gracefully)
        self._apply_stops(ctx)

        # 3. Read current weights from bar data
        current_weights: dict[str, float] = {}
        for sym in ctx.symbols:
            w = ctx.row(sym).get(self._weight_col)
            if w is not None:
                current_weights[sym] = float(w)

        # 4. Determine if we should rebalance
        should_rebalance = False
        if self._resample is None:
            if current_weights != self._prev_weights:
                should_rebalance = True
        elif ctx.timestamp in self._rebalance_dates:
            should_rebalance = True

        if should_rebalance:
            # Remove symbols that were stopped out this bar to prevent re-entry
            rebalance_weights = {s: w for s, w in current_weights.items() if s not in closed_by_stop}
            normalized = _normalize_weights(rebalance_weights, self._position_limit)
            ctx.portfolio.rebalance(normalized)
            # For order_delay=0, positions exist now — set stops immediately
            self._apply_stops(ctx)

        self._prev_weights = current_weights
        self._latest_weights = current_weights

        # 5. Track current positions for next bar's stop-out detection
        self._prev_position_syms = {sym for sym in ctx.symbols if ctx.portfolio.get_position(sym) != 0}

    def _apply_stops(self, ctx: BacktestContext) -> None:
        """Set stop-loss/take-profit/trailing-stop on positions that don't have them yet."""
        has_stops = self._stop_loss is not None or self._take_profit is not None or self._trail_stop is not None
        if not has_stops:
            return

        for sym in ctx.symbols:
            pos = ctx.portfolio.get_position(sym)
            if pos == 0:
                continue
            price = ctx.row(sym).get("close")
            if price is None:
                continue
            price = float(price)
            is_long = pos > 0

            if self._stop_loss is not None and ctx.portfolio.get_stop_loss(sym) is None:
                sl_price = price * (1 - self._stop_loss) if is_long else price * (1 + self._stop_loss)
                ctx.portfolio.set_stop_loss(sym, stop_price=sl_price)

            if self._take_profit is not None and ctx.portfolio.get_take_profit(sym) is None:
                tp_price = price * (1 + self._take_profit) if is_long else price * (1 - self._take_profit)
                ctx.portfolio.set_take_profit(sym, target_price=tp_price)

            if self._trail_stop is not None and ctx.portfolio.get_trailing_stop(sym) is None:
                ctx.portfolio.set_trailing_stop(sym, trail_pct=self._trail_stop)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def backtest_weights(
    data: pl.DataFrame,
    date_col: str = "date",
    symbol_col: str = "symbol",
    price_col: str = "close",
    weight_col: str = "weight",
    open_col: str | None = "open",
    high_col: str | None = "high",
    low_col: str | None = "low",
    resample: str | None = "M",
    resample_offset: str | None = None,
    fee_ratio: float = 0.001,
    tax_ratio: float = 0.0,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trail_stop: float | None = None,
    position_limit: float = 1.0,
    touched_exit: bool = False,
    t_plus: int = 1,
    initial_capital: float = 100_000.0,
    factor_col: str | None = None,
) -> WeightBacktestResult:
    """Run a weight-based portfolio backtest.

    Internally uses the unified Engine and Portfolio, so all core features
    (commission models, OHLC stop priority, leverage, etc.) apply.

    Args:
        data: Long-format DataFrame with one row per (date, symbol).
        date_col: Column name for dates.
        symbol_col: Column name for symbols.
        price_col: Column name for execution price.
        weight_col: Column name for target weights.
        open_col: Column with open prices (for touched exit).
        high_col: Column with high prices (for touched exit).
        low_col: Column with low prices (for touched exit).
        resample: Rebalance frequency (``"D"``, ``"W"``, ``"W-FRI"``, ``"M"``,
            ``"Q"``, ``"Y"``, or ``None`` for weight-change-only).
        resample_offset: Delay rebalance by N trading days (e.g. ``"2d"``).
        fee_ratio: Transaction fee as fraction of trade value.
        tax_ratio: Transaction tax as fraction of trade value.
        stop_loss: Per-position stop-loss threshold (e.g. 0.10 = 10%).
        take_profit: Per-position take-profit threshold.
        trail_stop: Trailing stop distance as fraction.
        position_limit: Maximum absolute weight per symbol.
        touched_exit: Use OHLC for intraday stop detection.
        t_plus: Execution delay in bars (0 = same bar, 1 = next bar).
        initial_capital: Starting portfolio value.
        factor_col: Optional column for price adjustment factor.

    Returns:
        WeightBacktestResult with equity curve, trades, and metrics.
    """
    # --- Validation ---
    required_cols = [date_col, symbol_col, price_col, weight_col]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in data. Available: {data.columns}")

    if len(data) == 0:
        empty_equity = pl.DataFrame(
            {"date": [], "cumulative_return": []}, schema={"date": pl.Date, "cumulative_return": pl.Float64}
        )
        empty_trades = pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "entry_date": pl.Date,
                "exit_date": pl.Date,
                "entry_price": pl.Float64,
                "exit_price": pl.Float64,
                "weight": pl.Float64,
                "return_pct": pl.Float64,
                "bars_held": pl.Int64,
            }
        )
        return WeightBacktestResult(
            equity=empty_equity,
            trades=empty_trades,
            metrics=BacktestMetrics(initial_equity=initial_capital, final_equity=initial_capital),
        )

    # --- Prepare data for Engine ---
    # Rename columns to canonical names expected by Engine
    renames: dict[str, str] = {}
    if date_col != "timestamp":
        renames[date_col] = "timestamp"
    if symbol_col != "symbol":
        renames[symbol_col] = "symbol"
    if price_col != "close":
        renames[price_col] = "close"

    engine_data = data
    if renames:
        engine_data = engine_data.rename(renames)

    # Rename OHLC columns if they exist and aren't already canonical
    ohlc_renames: dict[str, str] = {}
    if open_col and open_col in data.columns and open_col != "open" and open_col not in renames:
        ohlc_renames[open_col] = "open"
    if high_col and high_col in data.columns and high_col != "high" and high_col not in renames:
        ohlc_renames[high_col] = "high"
    if low_col and low_col in data.columns and low_col != "low" and low_col not in renames:
        ohlc_renames[low_col] = "low"
    if ohlc_renames:
        engine_data = engine_data.rename(ohlc_renames)

    # Strip OHLC columns when touched_exit=False so Engine uses close-only for stops
    if not touched_exit:
        drop_cols = [c for c in ("open", "high", "low") if c in engine_data.columns]
        if drop_cols:
            engine_data = engine_data.drop(drop_cols)

    engine_data = engine_data.sort(["timestamp", "symbol"])

    # --- Compute rebalance schedule ---
    offset_days = _parse_offset(resample_offset)
    # Use the original date column values (now renamed to 'timestamp')
    unique_dates = engine_data["timestamp"].unique().sort()
    rebalance_dates = _detect_rebalance_dates(unique_dates, resample, offset_days)

    # --- Build and run Engine ---
    strategy = _DataFrameWeightStrategy(
        weight_col=weight_col,
        rebalance_dates=rebalance_dates,
        resample=resample,
        position_limit=position_limit,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trail_stop=trail_stop,
    )

    commission = fee_ratio + tax_ratio

    engine = Engine(
        strategy=strategy,
        data=engine_data,  # Form C: long-format DataFrame with 'symbol' column
        initial_cash=initial_capital,
        commission=commission,
        slippage=0.0,
        warmup=0,
        order_delay=t_plus,
        factor_column=factor_col,
    )
    engine_results = engine.run()

    # --- Build WeightBacktestResult ---
    portfolio = engine.portfolio
    if portfolio is None:
        raise RuntimeError("Engine did not initialize a portfolio; backtest produced no results")

    # Equity curve: (date, cumulative_return)
    equity_dates = portfolio.timestamps
    equity_values = portfolio.equity_curve
    equity_df = pl.DataFrame(
        {
            "date": equity_dates,
            "cumulative_return": [v / initial_capital - 1 for v in equity_values],
        }
    )

    # Trade mapping: Engine trades → weight backtest trade schema
    engine_trades = engine_results.trades
    if len(engine_trades) > 0:
        # Map column names: asset→symbol, entry_timestamp→entry_date, exit_timestamp→exit_date
        trade_cols: dict[str, list[Any]] = {
            "symbol": engine_trades["asset"].to_list(),
            "entry_date": engine_trades["entry_timestamp"].to_list(),
            "exit_date": engine_trades["exit_timestamp"].to_list(),
            "entry_price": engine_trades["entry_price"].to_list(),
            "exit_price": engine_trades["exit_price"].to_list(),
            "weight": [0.0] * len(engine_trades),  # weight at exit is not tracked per-trade
            "return_pct": engine_trades["return_pct"].to_list(),
            "bars_held": engine_trades["bars_held"].to_list(),
        }
        trades_df = pl.DataFrame(trade_cols)
    else:
        trades_df = pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "entry_date": pl.Date,
                "exit_date": pl.Date,
                "entry_price": pl.Float64,
                "exit_price": pl.Float64,
                "weight": pl.Float64,
                "return_pct": pl.Float64,
                "bars_held": pl.Int64,
            }
        )

    # Build next_actions from final portfolio state
    positions = dict(portfolio.positions)
    latest_weights = strategy._latest_weights
    current_prices = dict(portfolio._current_prices)
    next_actions = _compute_next_actions_from_state(positions, latest_weights, current_prices)

    return WeightBacktestResult(
        equity=equity_df,
        trades=trades_df,
        metrics=engine_results,
        next_actions=next_actions,
    )


def _compute_next_actions_from_state(
    current_positions: dict[str, float],
    latest_weights: dict[str, float],
    current_prices: dict[str, float],
) -> pl.DataFrame | None:
    """Compute next actions from current positions and latest signal weights.

    Args:
        current_positions: Current portfolio positions {symbol: qty}.
        latest_weights: Latest weight signals {symbol: weight}.
        current_prices: Current prices {symbol: price}.

    Returns:
        DataFrame with columns symbol, action, current_weight, target_weight.
        None if no positions and no weights.
    """
    all_symbols = set(current_positions.keys()) | set(latest_weights.keys())
    if not all_symbols:
        return None

    total_value = sum(abs(qty) * current_prices.get(s, 0.0) for s, qty in current_positions.items())
    if total_value == 0:
        total_value = 1.0

    records: list[dict[str, Any]] = []
    for sym in sorted(all_symbols):
        current_qty = current_positions.get(sym, 0.0)
        current_w = (current_qty * current_prices.get(sym, 0.0)) / total_value if current_qty != 0 else 0.0
        target_w = latest_weights.get(sym, 0.0)

        if current_qty == 0 and target_w != 0:
            action = "enter"
        elif current_qty != 0 and target_w == 0:
            action = "exit"
        else:
            action = "hold"

        records.append(
            {
                "symbol": sym,
                "action": action,
                "current_weight": round(current_w, 6),
                "target_weight": round(target_w, 6),
            }
        )

    return pl.DataFrame(records) if records else None
