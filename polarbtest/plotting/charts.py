"""
Chart generation for backtesting results using Plotly.

All functions return plotly Figure objects that can be displayed, customized, or saved to HTML.
Plotly is an optional dependency — import errors provide a clear installation message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    import plotly.graph_objects as go


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
    height: int = 900,
    save_html: str | None = None,
) -> go.Figure:
    """Create a multi-panel backtest chart: price, equity curve, and drawdown.

    Args:
        engine: Engine instance after calling .run()
        price_column: Price column to plot (auto-detected if None)
        asset: Asset name for multi-asset data (auto-detected if None)
        show_trades: Show trade entry/exit markers on price chart
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

    Example:
        engine = Engine(strategy=my_strat, data=df)
        results = engine.run()
        fig = plot_backtest(engine, indicators=["sma_20"], save_html="backtest.html")
        fig.show()
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

    # Build subplot layout
    subplot_specs: list[list[dict[str, Any]]] = []
    row_heights: list[float] = []
    subplot_titles: list[str] = []

    # Row 1: Price
    subplot_specs.append([{"secondary_y": False}])
    row_heights.append(0.45 if has_volume else 0.50)
    subplot_titles.append("Price")

    # Row 2: Volume (optional)
    if has_volume:
        subplot_specs.append([{"secondary_y": False}])
        row_heights.append(0.10)
        subplot_titles.append("Volume")

    # Row 3: Equity
    subplot_specs.append([{"secondary_y": False}])
    row_heights.append(0.25)
    subplot_titles.append("Equity")

    # Row 4: Drawdown
    subplot_specs.append([{"secondary_y": False}])
    row_heights.append(0.20)
    subplot_titles.append("Drawdown")

    num_rows = len(subplot_specs)
    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
    )

    current_row = 1

    # --- Row 1: Price chart ---
    if has_ohlc:
        ohlc_cols = _get_ohlc_columns(data, asset)
        fig.add_trace(
            go_mod.Candlestick(
                x=timestamps,
                open=data[ohlc_cols["open"]].to_list(),
                high=data[ohlc_cols["high"]].to_list(),
                low=data[ohlc_cols["low"]].to_list(),
                close=data[ohlc_cols["close"]].to_list(),
                name="Price",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ),
            row=current_row,
            col=1,
        )
    else:
        fig.add_trace(
            go_mod.Scatter(
                x=timestamps,
                y=prices,
                name="Price",
                line={"color": "#2196F3", "width": 1.5},
            ),
            row=current_row,
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
                    row=current_row,
                    col=1,
                )

    # Band overlays (e.g., Bollinger Bands)
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
                    row=current_row,
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
                    row=current_row,
                    col=1,
                )

    # Trade markers
    if show_trades:
        _add_trade_markers(fig, engine, timestamps, data, price_column, asset, go_mod, current_row)

    current_row += 1

    # --- Row 2: Volume (optional) ---
    if has_volume:
        vol_col = _get_volume_column(data, asset)
        fig.add_trace(
            go_mod.Bar(
                x=timestamps,
                y=data[vol_col].to_list(),
                name="Volume",
                marker_color="rgba(100,100,100,0.4)",
                showlegend=False,
            ),
            row=current_row,
            col=1,
        )
        current_row += 1

    # --- Row 3: Equity curve ---
    equity_values = portfolio.equity_curve
    equity_timestamps = timestamps[: len(equity_values)]
    fig.add_trace(
        go_mod.Scatter(
            x=equity_timestamps,
            y=equity_values,
            name="Equity",
            line={"color": "#4CAF50", "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(76,175,80,0.1)",
        ),
        row=current_row,
        col=1,
    )
    current_row += 1

    # --- Row 4: Drawdown ---
    drawdown = _calculate_drawdown(equity_values)
    fig.add_trace(
        go_mod.Scatter(
            x=equity_timestamps,
            y=drawdown,
            name="Drawdown",
            line={"color": "#F44336", "width": 1},
            fill="tozeroy",
            fillcolor="rgba(244,67,54,0.2)",
        ),
        row=current_row,
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

    # Format y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    dd_row = num_rows
    eq_row = num_rows - 1
    fig.update_yaxes(title_text="Equity", row=eq_row, col=1)
    fig.update_yaxes(title_text="Drawdown %", tickformat=".1%", row=dd_row, col=1)

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

    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

    # Add mean line
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


def _add_trade_markers(
    fig: Any,
    engine: Any,
    timestamps: list[Any],
    data: pl.DataFrame,
    price_column: str,
    asset: str,
    go_mod: Any,
    row: int,
) -> None:
    """Add trade entry/exit markers to the price chart."""
    trades_df = engine.results.get("trades")
    if trades_df is None or len(trades_df) == 0:
        return

    # Filter trades for the target asset
    if "asset" in trades_df.columns:
        asset_filter = asset if asset != "asset" else trades_df["asset"][0]
        trades_df = trades_df.filter(pl.col("asset") == asset_filter)

    if len(trades_df) == 0:
        return

    entry_bars = trades_df["entry_bar"].to_list()
    entry_prices = trades_df["entry_price"].to_list()
    exit_bars = trades_df["exit_bar"].to_list()
    exit_prices = trades_df["exit_price"].to_list()
    directions = trades_df["direction"].to_list()
    pnls = trades_df["pnl"].to_list()

    # Entry markers
    entry_x = [timestamps[b] if b < len(timestamps) else b for b in entry_bars]
    entry_colors = ["#26a69a" if d == "long" else "#ef5350" for d in directions]
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
    exit_colors = ["#26a69a" if pnl > 0 else "#ef5350" for pnl in pnls]
    exit_text = [f"Exit @ {p:.2f} (PnL: {pnl:.2f})" for p, pnl in zip(exit_prices, pnls, strict=True)]

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
