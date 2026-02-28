"""
Chart generation for backtesting results using Plotly.

All functions return plotly Figure objects that can be displayed, customized, or saved to HTML.
Plotly is an optional dependency — import errors provide a clear installation message.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    import plotly.graph_objects as go

# Colors
GREEN = "#26a69a"
RED = "#ef5350"
GREEN_FILL = "rgba(38,166,154,0.15)"
RED_FILL = "rgba(239,83,80,0.15)"


def _require_plotly() -> None:
    """Raise ImportError with install instructions if plotly is not available."""
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise ImportError("plotly is required for visualization. Install it with: pip install plotly") from None


def _get_plotly_modules() -> tuple[Any, Any]:
    """Import and return (go, make_subplots) from plotly."""
    _require_plotly()
    import plotly.graph_objects as go_module
    from plotly.subplots import make_subplots as ms

    return go_module, ms


def plot_backtest(
    engine: Any,
    price_column: str | None = None,
    asset: str | None = None,
    show_trades: bool = True,
    show_volume: bool = True,
    indicators: list[str] | None = None,
    bands: list[tuple[str, str]] | None = None,
    title: str = "Backtest Results",
    height: int = 1000,
    save_html: str | None = None,
) -> go.Figure:
    """Create a multi-panel backtest chart: price, P/L, equity curve, and drawdown.

    Panels (top to bottom):
    - Equity curve with stats annotation (Peak, Final, Max DD, Max DD Duration)
    - Profit/Loss per trade (bubble size = bars held)
    - Price chart with OHLC candles, indicators, bands, and trade arrows
    - Volume bars (if available)

    Args:
        engine: Engine instance after calling .run()
        price_column: Price column to plot (auto-detected if None)
        asset: Asset name for multi-asset data (auto-detected if None)
        show_trades: Show trade markers and entry-to-exit arrows on price chart
        show_volume: Show volume subplot (if volume data available)
        indicators: List of column names to overlay on price chart as lines
        bands: List of (upper, lower) column name tuples to plot as filled bands
        title: Chart title
        height: Chart height in pixels
        save_html: If provided, save chart to this HTML file path

    Returns:
        Plotly Figure object

    Raises:
        ImportError: If plotly is not installed
        ValueError: If engine has not been run yet
    """
    go_mod, make_subplots = _get_plotly_modules()

    if engine.portfolio is None or engine.results is None:
        raise ValueError("Engine must be run before plotting. Call engine.run() first.")

    portfolio = engine.portfolio
    data = engine.processed_data

    # Resolve asset and price column
    if asset is None:
        asset = list(engine.price_columns.keys())[0]
    if price_column is None:
        price_column = engine.price_columns[asset]

    # Determine x-axis values
    timestamps = list(range(len(data)))
    if "timestamp" in data.columns:
        timestamps = data["timestamp"].to_list()

    prices = data[price_column].to_list()

    # Detect available subplots
    has_volume = show_volume and _has_volume_column(data, asset)
    has_ohlc = _has_ohlc_columns(data, asset)
    trades_df = _get_filtered_trades(engine, asset)
    num_trades = len(trades_df) if trades_df is not None else 0

    # Build subplot layout: Equity, P/L, Price, Volume
    subplot_specs: list[list[dict[str, Any]]] = []
    row_heights: list[float] = []
    subplot_titles: list[str] = []

    # Row 1: Equity
    subplot_specs.append([{"secondary_y": False}])
    row_heights.append(0.20)
    subplot_titles.append("Equity")

    # Row 2: Profit / Loss
    subplot_specs.append([{"secondary_y": False}])
    row_heights.append(0.12)
    subplot_titles.append("Profit / Loss")

    # Row 3: Price (OHLC)
    subplot_specs.append([{"secondary_y": False}])
    row_heights.append(0.56 if has_volume else 0.68)
    subplot_titles.append(f"OHLC  —  Trades ({num_trades})")

    # Row 4: Volume (optional)
    if has_volume:
        subplot_specs.append([{"secondary_y": False}])
        row_heights.append(0.12)
        subplot_titles.append("Volume")

    num_rows = len(subplot_specs)
    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
    )

    equity_row = 1
    pnl_row = 2
    price_row = 3
    volume_row = 4 if has_volume else None

    # === Row 1: Equity curve with stats ===
    equity_values = portfolio.equity_curve
    equity_timestamps = timestamps[: len(equity_values)]

    # Drawdown fill (behind equity line)
    drawdown = _calculate_drawdown(equity_values)
    running_max = _calculate_running_max(equity_values)

    # Running max (peak) as subtle fill
    fig.add_trace(
        go_mod.Scatter(
            x=equity_timestamps,
            y=running_max,
            name="Peak",
            line={"color": "rgba(0,0,0,0)", "width": 0},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=equity_row,
        col=1,
    )
    fig.add_trace(
        go_mod.Scatter(
            x=equity_timestamps,
            y=equity_values,
            name="Equity",
            line={"color": "#2196F3", "width": 1.5},
            fill="tonexty",
            fillcolor="rgba(255,235,59,0.25)",
        ),
        row=equity_row,
        col=1,
    )

    # Stats annotation
    _add_equity_stats(fig, equity_values, equity_timestamps, drawdown, go_mod, equity_row)

    # === Row 2: Profit / Loss per trade ===
    _add_pnl_chart(fig, trades_df, timestamps, go_mod, pnl_row)

    # === Row 3: Price chart ===
    if has_ohlc:
        ohlc_cols = _get_ohlc_columns(data, asset)
        open_vals = data[ohlc_cols["open"]].to_list()
        high_vals = data[ohlc_cols["high"]].to_list()
        low_vals = data[ohlc_cols["low"]].to_list()
        close_vals = data[ohlc_cols["close"]].to_list()

        _add_ohlc_bars(fig, timestamps, open_vals, high_vals, low_vals, close_vals, go_mod, price_row)
    else:
        fig.add_trace(
            go_mod.Scatter(
                x=timestamps,
                y=prices,
                name="Price",
                line={"color": "#2196F3", "width": 1.5},
            ),
            row=price_row,
            col=1,
        )

    # Indicator overlays
    if indicators:
        colors = ["#FF9800", "#9C27B0", "#4CAF50", "#F44336", "#00BCD4", "#795548"]
        for i, col_name in enumerate(indicators):
            if col_name in data.columns:
                fig.add_trace(
                    go_mod.Scatter(
                        x=timestamps,
                        y=data[col_name].to_list(),
                        name=col_name,
                        line={"color": colors[i % len(colors)], "width": 1},
                    ),
                    row=price_row,
                    col=1,
                )

    # Band overlays
    if bands:
        band_colors = [("rgba(33,150,243,0.1)", "#2196F3"), ("rgba(156,39,176,0.1)", "#9C27B0")]
        for i, (upper_col, lower_col) in enumerate(bands):
            if upper_col in data.columns and lower_col in data.columns:
                fill_color, line_color = band_colors[i % len(band_colors)]
                fig.add_trace(
                    go_mod.Scatter(
                        x=timestamps,
                        y=data[upper_col].to_list(),
                        name=upper_col,
                        line={"color": line_color, "width": 0.5},
                        showlegend=True,
                    ),
                    row=price_row,
                    col=1,
                )
                fig.add_trace(
                    go_mod.Scatter(
                        x=timestamps,
                        y=data[lower_col].to_list(),
                        name=lower_col,
                        line={"color": line_color, "width": 0.5},
                        fill="tonexty",
                        fillcolor=fill_color,
                        showlegend=True,
                    ),
                    row=price_row,
                    col=1,
                )

    # Trade markers and arrows
    if show_trades:
        _add_trade_markers(fig, trades_df, timestamps, go_mod, price_row)

    # === Row 4: Volume (optional) ===
    if has_volume and volume_row is not None:
        vol_col = _get_volume_column(data, asset)
        vol_values = data[vol_col].to_list()

        # Color volume bars by price direction
        if has_ohlc:
            ohlc_cols = _get_ohlc_columns(data, asset)
            opens = data[ohlc_cols["open"]].to_list()
            closes = data[ohlc_cols["close"]].to_list()
            vol_colors = [GREEN if c >= o else RED for o, c in zip(opens, closes, strict=True)]
        else:
            vol_colors = ["rgba(100,100,100,0.5)"] * len(vol_values)

        fig.add_trace(
            go_mod.Bar(
                x=timestamps,
                y=vol_values,
                name="Volume",
                marker_color=vol_colors,
                showlegend=False,
            ),
            row=volume_row,
            col=1,
        )

    # Layout
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        xaxis_rangeslider_visible=False,
    )

    # Y-axis labels and settings
    fig.update_yaxes(title_text="Equity", row=equity_row, col=1)
    fig.update_yaxes(title_text="P/L %", tickformat=".0%", row=pnl_row, col=1)
    fig.update_yaxes(title_text="Price", row=price_row, col=1)
    if volume_row is not None:
        fig.update_yaxes(title_text="Volume", fixedrange=True, row=volume_row, col=1)

    # Visual separation: borders on all subplots, tick labels on bottom
    for r in range(1, num_rows + 1):
        fig.update_xaxes(
            showline=True,
            linewidth=1,
            linecolor="lightgray",
            mirror=True,
            rangeslider_visible=False,
            row=r,
            col=1,
        )
        fig.update_yaxes(showline=True, linewidth=1, linecolor="lightgray", mirror=True, row=r, col=1)
    # Ensure the bottom subplot shows x-axis tick labels (timestamps)
    fig.update_xaxes(showticklabels=True, row=num_rows, col=1)

    if save_html is not None:
        fig.write_html(save_html)

    return fig


def plot_returns_distribution(
    engine: Any,
    bins: int = 50,
    title: str = "Returns Distribution",
    height: int = 400,
    save_html: str | None = None,
) -> go.Figure:
    """Plot histogram of daily returns.

    Args:
        engine: Engine instance after calling .run()
        bins: Number of histogram bins
        title: Chart title
        height: Chart height in pixels
        save_html: If provided, save chart to this HTML file path

    Returns:
        Plotly Figure object

    Raises:
        ImportError: If plotly is not installed
        ValueError: If engine has not been run yet
    """
    go_mod, _ = _get_plotly_modules()

    if engine.portfolio is None:
        raise ValueError("Engine must be run before plotting. Call engine.run() first.")

    equity = engine.portfolio.equity_curve
    if len(equity) < 2:
        raise ValueError("Not enough data points to calculate returns.")

    returns = [(equity[i] / equity[i - 1] - 1) for i in range(1, len(equity))]

    fig = go_mod.Figure()
    fig.add_trace(
        go_mod.Histogram(
            x=returns,
            nbinsx=bins,
            name="Returns",
            marker_color="#2196F3",
            opacity=0.75,
        )
    )

    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

    mean_ret = sum(returns) / len(returns) if returns else 0
    fig.add_vline(x=mean_ret, line_dash="dot", line_color="red", line_width=1, annotation_text="Mean")

    fig.update_layout(
        title=title,
        xaxis_title="Return",
        yaxis_title="Frequency",
        height=height,
        template="plotly_white",
        bargap=0.05,
    )

    if save_html is not None:
        fig.write_html(save_html)

    return fig


# --- Helper functions ---


def _has_volume_column(data: pl.DataFrame, asset: str) -> bool:
    """Check if volume data is available."""
    if asset == "asset":
        return "volume" in data.columns
    return f"{asset}_volume" in data.columns


def _get_volume_column(data: pl.DataFrame, asset: str) -> str:
    """Get the volume column name."""
    if asset == "asset":
        return "volume"
    return f"{asset}_volume"


def _has_ohlc_columns(data: pl.DataFrame, asset: str) -> bool:
    """Check if full OHLC data is available."""
    if asset == "asset":
        return all(col in data.columns for col in ("open", "high", "low", "close"))
    prefix = f"{asset}_"
    return all(f"{prefix}{col}" in data.columns for col in ("open", "high", "low", "close"))


def _get_ohlc_columns(data: pl.DataFrame, asset: str) -> dict[str, str]:
    """Get OHLC column name mapping."""
    if asset == "asset":
        return {"open": "open", "high": "high", "low": "low", "close": "close"}
    return {
        "open": f"{asset}_open",
        "high": f"{asset}_high",
        "low": f"{asset}_low",
        "close": f"{asset}_close",
    }


def _calculate_drawdown(equity: list[float]) -> list[float]:
    """Calculate drawdown series from equity curve."""
    if not equity:
        return []
    running_max = equity[0]
    drawdown = []
    for val in equity:
        if val > running_max:
            running_max = val
        dd = (val - running_max) / running_max if running_max > 0 else 0.0
        drawdown.append(dd)
    return drawdown


def _calculate_running_max(equity: list[float]) -> list[float]:
    """Calculate running maximum of equity curve."""
    if not equity:
        return []
    result = []
    current_max = equity[0]
    for val in equity:
        if val > current_max:
            current_max = val
        result.append(current_max)
    return result


def _calculate_max_drawdown_duration(equity: list[float]) -> int:
    """Calculate max drawdown duration in bars (time from peak to recovery or end)."""
    if not equity:
        return 0
    peak = equity[0]
    bars_since_peak = 0
    max_duration = 0
    for val in equity:
        if val >= peak:
            peak = val
            bars_since_peak = 0
        else:
            bars_since_peak += 1
            if bars_since_peak > max_duration:
                max_duration = bars_since_peak
    return max_duration


def _calculate_max_dd_duration_indices(equity: list[float]) -> tuple[int, int]:
    """Return (start_idx, end_idx) of the longest drawdown period."""
    if not equity:
        return (0, 0)
    peak = equity[0]
    bars_since_peak = 0
    best_start = 0
    best_end = 0
    best_duration = 0
    current_peak_idx = 0

    for i, val in enumerate(equity):
        if val >= peak:
            peak = val
            current_peak_idx = i
            bars_since_peak = 0
        else:
            bars_since_peak += 1
            if bars_since_peak > best_duration:
                best_duration = bars_since_peak
                best_start = current_peak_idx
                best_end = i

    return (best_start, best_end)


def _format_duration(timestamps: list[Any], start_idx: int, end_idx: int, bar_count: int) -> str:
    """Format drawdown duration as human-readable time or bar count.

    When timestamps are dates/datetimes, computes the actual time difference
    and displays it as days/months/years. Falls back to bar count otherwise.
    """
    if start_idx < len(timestamps) and end_idx < len(timestamps):
        t_start = timestamps[start_idx]
        t_end = timestamps[end_idx]

        delta: timedelta | None = None
        if isinstance(t_start, datetime) and isinstance(t_end, datetime):
            delta = t_end - t_start
        elif isinstance(t_start, date) and isinstance(t_end, date):
            delta = t_end - t_start  # noqa: SIM114

        if delta is not None:
            days = delta.days
            if days < 1:
                hours = int(delta.total_seconds() // 3600)
                if hours > 0:
                    return f"{hours} hours"
                minutes = int(delta.total_seconds() // 60)
                return f"{minutes} min"
            if days < 60:
                return f"{days} days"
            if days < 365:
                months = days // 30
                return f"{months} months"
            years = days // 365
            remaining_months = (days % 365) // 30
            if remaining_months > 0:
                return f"{years}y {remaining_months}m"
            return f"{years} years"

    return f"{bar_count} bars"


def _get_filtered_trades(engine: Any, asset: str) -> pl.DataFrame | None:
    """Get trades DataFrame filtered for the target asset."""
    trades_df = engine.results.get("trades")
    if trades_df is None or len(trades_df) == 0:
        return None

    if "asset" in trades_df.columns:
        asset_filter = asset if asset != "asset" else trades_df["asset"][0]
        trades_df = trades_df.filter(pl.col("asset") == asset_filter)

    if len(trades_df) == 0:
        return None

    return trades_df


def _add_equity_stats(
    fig: Any,
    equity: list[float],
    timestamps: list[Any],
    drawdown: list[float],
    go_mod: Any,
    row: int,
) -> None:
    """Add stats markers and annotation block to equity subplot."""
    if not equity:
        return

    initial = equity[0]
    final = equity[-1]
    peak = max(equity)
    peak_idx = equity.index(peak)
    max_dd = min(drawdown) if drawdown else 0.0
    max_dd_duration = _calculate_max_drawdown_duration(equity)

    peak_return = (peak / initial - 1) * 100
    final_return = (final / initial - 1) * 100

    # Peak marker (cyan)
    fig.add_trace(
        go_mod.Scatter(
            x=[timestamps[peak_idx]],
            y=[peak],
            mode="markers",
            name=f"Peak ({peak_return:+.1f}%)",
            marker={"size": 12, "color": "cyan", "symbol": "circle"},
            showlegend=True,
        ),
        row=row,
        col=1,
    )

    # Final marker (blue)
    fig.add_trace(
        go_mod.Scatter(
            x=[timestamps[-1]],
            y=[final],
            mode="markers",
            name=f"Final ({final_return:+.1f}%)",
            marker={"size": 12, "color": "#2196F3", "symbol": "circle"},
            showlegend=True,
        ),
        row=row,
        col=1,
    )

    # Max drawdown marker (red) — at the lowest drawdown point
    if drawdown:
        dd_min_idx = drawdown.index(min(drawdown))
        fig.add_trace(
            go_mod.Scatter(
                x=[timestamps[dd_min_idx]],
                y=[equity[dd_min_idx]],
                mode="markers",
                name=f"Max Drawdown ({max_dd:.1%})",
                marker={"size": 12, "color": RED, "symbol": "circle"},
                showlegend=True,
            ),
            row=row,
            col=1,
        )

    # Max DD duration line — find the actual duration window
    _add_dd_duration_line(fig, equity, timestamps, go_mod, row, max_dd_duration)


def _add_dd_duration_line(
    fig: Any,
    equity: list[float],
    timestamps: list[Any],
    go_mod: Any,
    row: int,
    max_dd_duration: int,
) -> None:
    """Draw a horizontal line spanning the longest drawdown period on the equity chart."""
    if not equity or max_dd_duration == 0:
        return

    best_start, best_end = _calculate_max_dd_duration_indices(equity)
    best_duration = best_end - best_start

    if best_duration > 0:
        duration_label = _format_duration(timestamps, best_start, best_end, max_dd_duration)
        y_level = equity[best_start]
        fig.add_trace(
            go_mod.Scatter(
                x=[timestamps[best_start], timestamps[best_end]],
                y=[y_level, y_level],
                mode="lines",
                name=f"Max Dd Dur. ({duration_label})",
                line={"color": RED, "width": 2, "dash": "solid"},
                showlegend=True,
            ),
            row=row,
            col=1,
        )


def _add_ohlc_bars(
    fig: Any,
    timestamps: list[Any],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    go_mod: Any,
    row: int,
) -> None:
    """Draw OHLC candles using Plotly's native Candlestick trace."""
    fig.add_trace(
        go_mod.Candlestick(
            x=timestamps,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="OHLC",
            increasing={"fillcolor": GREEN, "line": {"color": GREEN}},
            decreasing={"fillcolor": RED, "line": {"color": RED}},
            showlegend=True,
        ),
        row=row,
        col=1,
    )


def _add_pnl_chart(
    fig: Any,
    trades_df: pl.DataFrame | None,
    timestamps: list[Any],
    go_mod: Any,
    row: int,
) -> None:
    """Add Profit/Loss chart with lines from entry (0%) to exit (PnL%).

    Each trade is shown as a line spanning from entry_bar to exit_bar,
    starting at 0% and ending at the trade's PnL%. A bubble at the exit
    shows the final PnL.
    """
    # Dashed zero line
    if timestamps:
        fig.add_trace(
            go_mod.Scatter(
                x=[timestamps[0], timestamps[-1]],
                y=[0, 0],
                mode="lines",
                line={"color": "gray", "width": 1, "dash": "dash"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )

    if trades_df is None or len(trades_df) == 0:
        return

    entry_bars = trades_df["entry_bar"].to_list()
    exit_bars = trades_df["exit_bar"].to_list()
    pnl_pcts = trades_df["pnl_pct"].to_list()
    pnls = trades_df["pnl"].to_list()
    directions = trades_df["direction"].to_list()

    bars_held = [ex - en for en, ex in zip(entry_bars, exit_bars, strict=True)]
    max_bars = max(bars_held) if bars_held else 1
    sizes = [max(6, min(25, 6 + 19 * (bh / max_bars))) for bh in bars_held]

    # Draw a line + endpoint bubble for each trade
    for i in range(len(entry_bars)):
        eb = entry_bars[i]
        xb = exit_bars[i]
        pnl_frac = pnl_pcts[i] / 100.0
        color = GREEN if pnls[i] > 0 else RED
        x0 = timestamps[eb] if eb < len(timestamps) else eb
        x1 = timestamps[xb] if xb < len(timestamps) else xb
        d = directions[i]
        bh = bars_held[i]

        hover = f"{'Long' if d == 'long' else 'Short'}<br>PnL: {pnls[i]:+.2f} ({pnl_pcts[i]:+.1f}%)<br>Bars held: {bh}"

        # Line from entry at 0% to exit at PnL%
        fig.add_trace(
            go_mod.Scatter(
                x=[x0, x1],
                y=[0, pnl_frac],
                mode="lines+markers",
                line={"color": color, "width": 1.5},
                marker={"size": [0, sizes[i]], "color": color, "opacity": 0.8},
                text=["", hover],
                hoverinfo="text",
                showlegend=i == 0,
                name="Trades P/L" if i == 0 else None,
            ),
            row=row,
            col=1,
        )


def _add_trade_markers(
    fig: Any,
    trades_df: pl.DataFrame | None,
    timestamps: list[Any],
    go_mod: Any,
    row: int,
) -> None:
    """Add trade entry/exit markers and connecting arrows to the price chart."""
    if trades_df is None or len(trades_df) == 0:
        return

    entry_bars = trades_df["entry_bar"].to_list()
    entry_prices = trades_df["entry_price"].to_list()
    exit_bars = trades_df["exit_bar"].to_list()
    exit_prices = trades_df["exit_price"].to_list()
    directions = trades_df["direction"].to_list()
    pnls = trades_df["pnl"].to_list()

    # Entry markers
    entry_x = [timestamps[b] if b < len(timestamps) else b for b in entry_bars]
    entry_colors = [GREEN if d == "long" else RED for d in directions]
    entry_symbols = ["triangle-up" if d == "long" else "triangle-down" for d in directions]
    entry_text = [
        f"{'Long' if d == 'long' else 'Short'} @ {p:.2f}" for d, p in zip(directions, entry_prices, strict=True)
    ]

    fig.add_trace(
        go_mod.Scatter(
            x=entry_x,
            y=entry_prices,
            mode="markers",
            name="Entry",
            marker={"size": 10, "color": entry_colors, "symbol": entry_symbols, "line": {"width": 1, "color": "white"}},
            text=entry_text,
            hoverinfo="text",
        ),
        row=row,
        col=1,
    )

    # Exit markers
    exit_x = [timestamps[b] if b < len(timestamps) else b for b in exit_bars]
    exit_colors = [GREEN if pnl > 0 else RED for pnl in pnls]
    exit_text = [f"Exit @ {p:.2f} (PnL: {pnl:+.2f})" for p, pnl in zip(exit_prices, pnls, strict=True)]

    fig.add_trace(
        go_mod.Scatter(
            x=exit_x,
            y=exit_prices,
            mode="markers",
            name="Exit",
            marker={"size": 8, "color": exit_colors, "symbol": "x", "line": {"width": 1, "color": "white"}},
            text=exit_text,
            hoverinfo="text",
        ),
        row=row,
        col=1,
    )

    # Connecting arrows: entry → exit, colored by PnL, shaded region
    for i in range(len(entry_bars)):
        eb = entry_bars[i]
        xb = exit_bars[i]
        color = GREEN if pnls[i] > 0 else RED
        fill = GREEN_FILL if pnls[i] > 0 else RED_FILL

        # Get x-range for the shaded region
        x0 = timestamps[eb] if eb < len(timestamps) else eb
        x1 = timestamps[xb] if xb < len(timestamps) else xb

        # Shaded region between entry and exit
        ep = entry_prices[i]
        xp = exit_prices[i]
        y_top = max(ep, xp)
        y_bot = min(ep, xp)

        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=y_bot,
            y1=y_top,
            fillcolor=fill,
            line={"width": 0},
            row=row,
            col=1,
        )

        # Arrow line from entry to exit
        fig.add_annotation(
            x=x1,
            y=xp,
            ax=x0,
            ay=ep,
            xref=f"x{row}" if row > 1 else "x",
            yref=f"y{row}" if row > 1 else "y",
            axref=f"x{row}" if row > 1 else "x",
            ayref=f"y{row}" if row > 1 else "y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=1.5,
            arrowcolor=color,
            opacity=0.8,
        )
