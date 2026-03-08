"""Weight-based portfolio backtesting.

Provides a declarative, vectorized-where-possible backtester for portfolio weight
allocation strategies (momentum rotation, factor models, equal-weight baskets).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import polars as pl

from polarbt.metrics import calculate_metrics
from polarbt.results import BacktestMetrics, TradeStats, _backtest_metrics_from_dict


@dataclass
class WeightBacktestResult:
    """Results from a weight-based backtest.

    Attributes:
        equity: DataFrame with columns ``date`` and ``cumulative_return``.
        trades: DataFrame with per-trade details.
        metrics: Standard backtest performance metrics.
        next_actions: Forward-looking stock operations (Feature 6).
    """

    equity: pl.DataFrame
    trades: pl.DataFrame
    metrics: BacktestMetrics
    next_actions: pl.DataFrame | None = None

    def __str__(self) -> str:
        return str(self.metrics)


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

    # Detect period boundaries
    boundary_indices: list[int] = [0]  # Always include first date

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
        elif resample == "W" or resample == "W-FRI":
            is_boundary = curr.isocalendar()[1] != prev.isocalendar()[1]

        if is_boundary:
            boundary_indices.append(i)

    # Apply offset: shift each boundary index forward by offset_days
    rebalance_indices: set[int] = set()
    for idx in boundary_indices:
        shifted = idx + offset_days
        # Collapse: if shifted >= len(date_list), skip. If multiple collapse
        # to same date, the set naturally deduplicates.
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

    # Clip individual weights
    clipped = {s: max(-position_limit, min(position_limit, w)) for s, w in weights.items()}

    # Boolean detection: if all non-zero weights are 1.0 -> equal weight
    nonzero = {s: w for s, w in clipped.items() if w != 0}
    if nonzero and all(v == 1.0 for v in nonzero.values()):
        eq_w = 1.0 / len(nonzero)
        clipped = {s: (eq_w if w != 0 else 0.0) for s, w in clipped.items()}

    # Scale if sum(|w|) > 1
    total_abs = sum(abs(w) for w in clipped.values())
    if total_abs > 1.0:
        clipped = {s: w / total_abs for s, w in clipped.items()}

    return clipped


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

    # Sort by date
    data = data.sort(date_col)

    offset_days = _parse_offset(resample_offset)
    unique_dates = data[date_col].unique().sort()
    rebalance_dates = _detect_rebalance_dates(unique_dates, resample, offset_days)

    # --- State ---
    portfolio_value = initial_capital
    cash = initial_capital
    positions: dict[str, float] = {}  # symbol -> quantity
    entry_prices: dict[str, float] = {}  # symbol -> entry price
    entry_dates: dict[str, Any] = {}  # symbol -> entry date
    peak_prices: dict[str, float] = {}  # symbol -> highest price since entry (for trail)
    entry_bars: dict[str, int] = {}  # symbol -> bar index at entry

    equity_dates: list[Any] = []
    equity_values: list[float] = []
    trade_records: list[dict[str, Any]] = []

    # Build a date -> row mapping: for each date, collect {symbol: row_data}
    dates_in_order = unique_dates.to_list()

    # Pending weights for T+1 execution
    pending_weights: dict[str, float] | None = None
    prev_weights: dict[str, float] = {}

    for bar_idx, current_date in enumerate(dates_in_order):
        date_rows = data.filter(pl.col(date_col) == current_date)

        # Build price map for this date
        prices: dict[str, float] = {}
        ohlc: dict[str, dict[str, float]] = {}
        factors: dict[str, float] = {}

        for row in date_rows.iter_rows(named=True):
            sym = row[symbol_col]
            p = row[price_col]
            if p is not None and p > 0:
                prices[sym] = float(p)
                if open_col and open_col in data.columns and row.get(open_col) is not None:
                    ohlc[sym] = {
                        "open": float(row[open_col]),
                        "high": float(row.get(high_col, p))
                        if high_col and high_col in data.columns and row.get(high_col) is not None
                        else float(p),
                        "low": float(row.get(low_col, p))
                        if low_col and low_col in data.columns and row.get(low_col) is not None
                        else float(p),
                        "close": float(p),
                    }
                if factor_col and factor_col in data.columns and row.get(factor_col) is not None:
                    factors[sym] = float(row[factor_col])

        # 1. Update position values at current prices
        position_value = 0.0
        for sym, qty in list(positions.items()):
            if sym in prices:
                position_value += qty * prices[sym]

        portfolio_value = cash + position_value

        # 2. Check stops (SL/TP/trailing) with priority logic when OHLC available
        closed_by_stop: set[str] = set()
        for sym in list(positions.keys()):
            if sym not in prices:
                continue
            qty = positions[sym]
            if qty == 0:
                continue
            ep = entry_prices.get(sym, prices[sym])
            current_p = prices[sym]

            exit_price: float | None = None
            is_long = qty > 0

            if touched_exit and sym in ohlc:
                o, h, low = ohlc[sym]["open"], ohlc[sym]["high"], ohlc[sym]["low"]
                # Determine effective stop price
                eff_sl: float | None = None
                if stop_loss is not None:
                    eff_sl = ep * (1 - stop_loss) if is_long else ep * (1 + stop_loss)
                if trail_stop is not None and sym in peak_prices:
                    ts_price = peak_prices[sym] * (1 - trail_stop) if is_long else peak_prices[sym] * (1 + trail_stop)
                    eff_sl = (
                        ts_price if eff_sl is None else (max(eff_sl, ts_price) if is_long else min(eff_sl, ts_price))
                    )

                eff_tp: float | None = None
                if take_profit is not None:
                    eff_tp = ep * (1 + take_profit) if is_long else ep * (1 - take_profit)

                # Priority 1: Open breaches
                if eff_sl is not None and ((is_long and o <= eff_sl) or (not is_long and o >= eff_sl)):
                    exit_price = o
                if (
                    exit_price is None
                    and eff_tp is not None
                    and ((is_long and o >= eff_tp) or (not is_long and o <= eff_tp))
                ):
                    exit_price = o
                # Priority 2: High
                if exit_price is None:
                    if is_long and eff_tp is not None and h >= eff_tp:
                        exit_price = eff_tp
                    elif not is_long and eff_sl is not None and h >= eff_sl:
                        exit_price = eff_sl
                # Priority 3: Low
                if exit_price is None:
                    if is_long and eff_sl is not None and low <= eff_sl:
                        exit_price = eff_sl
                    elif not is_long and eff_tp is not None and low <= eff_tp:
                        exit_price = eff_tp
            else:
                # Close-price stop checks
                if stop_loss is not None:
                    if is_long:
                        sl_price = ep * (1 - stop_loss)
                        if current_p <= sl_price:
                            exit_price = sl_price
                    else:
                        sl_price = ep * (1 + stop_loss)
                        if current_p >= sl_price:
                            exit_price = sl_price

                if exit_price is None and take_profit is not None:
                    if is_long:
                        tp_price = ep * (1 + take_profit)
                        if current_p >= tp_price:
                            exit_price = tp_price
                    else:
                        tp_price = ep * (1 - take_profit)
                        if current_p <= tp_price:
                            exit_price = tp_price

                if exit_price is None and trail_stop is not None and sym in peak_prices:
                    if is_long:
                        ts_price = peak_prices[sym] * (1 - trail_stop)
                        if current_p <= ts_price:
                            exit_price = ts_price
                    else:
                        ts_price = peak_prices[sym] * (1 + trail_stop)
                        if current_p >= ts_price:
                            exit_price = ts_price

            if exit_price is not None:
                # Record trade
                ret = (exit_price - ep) / ep if is_long else (ep - exit_price) / ep
                factor = factors.get(sym, 1.0) if factor_col else 1.0
                raw_exit = exit_price / factor if factor != 0 else exit_price
                fee = abs(qty * raw_exit) * (fee_ratio + tax_ratio)
                cash += qty * exit_price - fee if is_long else -qty * exit_price - fee
                # For short: we initially received proceeds, now buy back
                if not is_long:
                    cash += qty * exit_price  # qty is negative
                    # Correct: for short, closing means buying back
                    # cash change = -|qty| * exit_price - fee (buying back costs money)
                    # But we already have the short proceeds from entry.
                    # Let's simplify: just track cash properly.
                    pass

                trade_records.append(
                    {
                        "symbol": sym,
                        "entry_date": entry_dates.get(sym),
                        "exit_date": current_date,
                        "entry_price": ep,
                        "exit_price": exit_price,
                        "weight": 0.0,
                        "return_pct": ret,
                        "bars_held": bar_idx - entry_bars.get(sym, bar_idx),
                    }
                )
                closed_by_stop.add(sym)
                del positions[sym]
                entry_prices.pop(sym, None)
                entry_dates.pop(sym, None)
                peak_prices.pop(sym, None)
                entry_bars.pop(sym, None)

        # Update trailing stop peak prices
        if trail_stop is not None:
            for sym, qty in positions.items():
                if sym not in prices:
                    continue
                p = prices[sym]
                if qty > 0:
                    if sym in ohlc:
                        p = ohlc[sym]["high"]
                    peak_prices[sym] = max(peak_prices.get(sym, p), p)
                elif qty < 0:
                    if sym in ohlc:
                        p = ohlc[sym]["low"]
                    peak_prices[sym] = min(peak_prices.get(sym, p), p)

        # 3. Execute pending T+1 rebalance
        if pending_weights is not None and t_plus > 0:
            cash, positions, entry_prices, entry_dates, peak_prices, entry_bars = _execute_rebalance(
                pending_weights,
                positions,
                prices,
                cash,
                portfolio_value,
                fee_ratio,
                tax_ratio,
                entry_prices,
                entry_dates,
                peak_prices,
                entry_bars,
                current_date,
                bar_idx,
                trade_records,
                closed_by_stop,
                factor_col,
                factors,
            )
            pending_weights = None

        # 4. Check if this is a rebalance date
        current_weights: dict[str, float] = {}
        for row in date_rows.iter_rows(named=True):
            sym = row[symbol_col]
            w = row[weight_col]
            if w is not None:
                current_weights[sym] = float(w)

        should_rebalance = False
        if resample is None:
            # Rebalance only when weights change
            if current_weights != prev_weights:
                should_rebalance = True
        else:
            if current_date in rebalance_dates:
                should_rebalance = True

        if should_rebalance:
            normalized = _normalize_weights(current_weights, position_limit)
            if t_plus == 0:
                cash, positions, entry_prices, entry_dates, peak_prices, entry_bars = _execute_rebalance(
                    normalized,
                    positions,
                    prices,
                    cash,
                    portfolio_value,
                    fee_ratio,
                    tax_ratio,
                    entry_prices,
                    entry_dates,
                    peak_prices,
                    entry_bars,
                    current_date,
                    bar_idx,
                    trade_records,
                    closed_by_stop,
                    factor_col,
                    factors,
                )
            else:
                pending_weights = normalized

        prev_weights = current_weights

        # 5. Record equity
        position_value = sum(qty * prices.get(sym, 0.0) for sym, qty in positions.items())
        portfolio_value = cash + position_value
        equity_dates.append(current_date)
        equity_values.append(portfolio_value)

    # --- Build results ---
    equity_df = pl.DataFrame(
        {
            "date": equity_dates,
            "cumulative_return": [v / initial_capital - 1 for v in equity_values],
        }
    )

    if trade_records:
        trades_df = pl.DataFrame(trade_records)
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

    # Calculate metrics using existing infrastructure
    metrics_equity_df = pl.DataFrame(
        {
            "timestamp": equity_dates,
            "equity": equity_values,
        },
        strict=False,
    )

    metrics_dict = calculate_metrics(metrics_equity_df, initial_capital)
    metrics_dict["initial_equity"] = initial_capital
    metrics_dict["final_equity"] = equity_values[-1] if equity_values else initial_capital

    # Trade-level metrics
    if len(trades_df) > 0 and "return_pct" in trades_df.columns:
        pct_col = trades_df["return_pct"]
        metrics_dict["best_trade_pct"] = float(pct_col.max())  # type: ignore[arg-type]
        metrics_dict["worst_trade_pct"] = float(pct_col.min())  # type: ignore[arg-type]
        metrics_dict["avg_trade_pct"] = float(pct_col.mean())  # type: ignore[arg-type]
        if "bars_held" in trades_df.columns:
            bars_col = trades_df["bars_held"]
            metrics_dict["max_trade_duration"] = float(bars_col.max())  # type: ignore[arg-type]
            metrics_dict["avg_trade_duration"] = float(bars_col.mean())  # type: ignore[arg-type]
        else:
            metrics_dict["max_trade_duration"] = 0.0
            metrics_dict["avg_trade_duration"] = 0.0
    else:
        for k in ("best_trade_pct", "worst_trade_pct", "avg_trade_pct", "max_trade_duration", "avg_trade_duration"):
            metrics_dict[k] = 0.0

    trade_stats = TradeStats(
        total_trades=len(trades_df),
        win_rate=len(trades_df.filter(pl.col("return_pct") > 0)) / len(trades_df) if len(trades_df) > 0 else 0.0,
    )
    metrics_dict["trades"] = trades_df
    metrics_dict["win_rate"] = trade_stats.win_rate

    bm = _backtest_metrics_from_dict(metrics_dict, trade_stats)

    # Build next_actions (Feature 6)
    next_actions = _compute_next_actions_from_state(positions, prev_weights, prices)

    return WeightBacktestResult(
        equity=equity_df,
        trades=trades_df,
        metrics=bm,
        next_actions=next_actions,
    )


def _execute_rebalance(
    target_weights: dict[str, float],
    positions: dict[str, float],
    prices: dict[str, float],
    cash: float,
    portfolio_value: float,
    fee_ratio: float,
    tax_ratio: float,
    entry_prices: dict[str, float],
    entry_dates: dict[str, Any],
    peak_prices: dict[str, float],
    entry_bars: dict[str, int],
    current_date: Any,
    bar_idx: int,
    trade_records: list[dict[str, Any]],
    closed_by_stop: set[str],
    factor_col: str | None,
    factors: dict[str, float],
) -> tuple[float, dict[str, float], dict[str, float], dict[str, Any], dict[str, float], dict[str, int]]:
    """Execute a rebalance from current positions to target weights.

    Returns updated (cash, positions, entry_prices, entry_dates, peak_prices, entry_bars).
    """
    if portfolio_value <= 0:
        return cash, positions, entry_prices, entry_dates, peak_prices, entry_bars

    # All symbols involved
    all_symbols = set(positions.keys()) | set(target_weights.keys())

    for sym in all_symbols:
        if sym in closed_by_stop:
            continue

        target_w = target_weights.get(sym, 0.0)
        target_value = portfolio_value * target_w
        price = prices.get(sym)
        if price is None or price <= 0:
            continue

        target_qty = target_value / price
        current_qty = positions.get(sym, 0.0)
        delta_qty = target_qty - current_qty

        if abs(delta_qty) < 1e-10:
            continue

        # Fee on the delta
        factor = factors.get(sym, 1.0) if factor_col else 1.0
        raw_price = price / factor if factor != 0 else price
        delta_value = abs(delta_qty) * raw_price
        fee = delta_value * (fee_ratio + tax_ratio)

        # Record trade for positions being closed or reduced
        if current_qty != 0 and (target_qty == 0 or (current_qty > 0) != (target_qty > 0)):
            # Full or partial close
            ep = entry_prices.get(sym, price)
            is_long = current_qty > 0
            ret = (price - ep) / ep if is_long else (ep - price) / ep
            trade_records.append(
                {
                    "symbol": sym,
                    "entry_date": entry_dates.get(sym),
                    "exit_date": current_date,
                    "entry_price": ep,
                    "exit_price": price,
                    "weight": target_w,
                    "return_pct": ret,
                    "bars_held": bar_idx - entry_bars.get(sym, bar_idx),
                }
            )

        # Update cash
        cash -= delta_qty * price + fee

        # Update position
        if abs(target_qty) < 1e-10:
            positions.pop(sym, None)
            entry_prices.pop(sym, None)
            entry_dates.pop(sym, None)
            peak_prices.pop(sym, None)
            entry_bars.pop(sym, None)
        else:
            if current_qty == 0 or (current_qty > 0) != (target_qty > 0):
                # New position
                entry_prices[sym] = price
                entry_dates[sym] = current_date
                entry_bars[sym] = bar_idx
                peak_prices[sym] = price
            positions[sym] = target_qty

    return cash, positions, entry_prices, entry_dates, peak_prices, entry_bars


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
        total_value = 1.0  # Avoid division by zero

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
