# PolarBtest Enhancement Roadmap

> Comprehensive feature enhancement plan based on analysis of backtrader and backtesting.py

**Last Updated**: 2026-01-31  
**Status**: Planning Phase

---

## Table of Contents

- [Philosophy & Design Principles](#philosophy--design-principles)
- [Phase 1: Core Trading Functionality](#phase-1-core-trading-functionality)
- [Phase 2: Visualization & Analysis](#phase-2-visualization--analysis)
- [Phase 3: Risk Management & Realism](#phase-3-risk-management--realism)
- [Phase 4: Extended Indicators & Utilities](#phase-4-extended-indicators--utilities)
- [Phase 5: Optimization Enhancements](#phase-5-optimization-enhancements)
- [Phase 6: Documentation & Examples](#phase-6-documentation--examples)
- [Phase 7: Advanced Features](#phase-7-advanced-features)
- [Implementation Guidelines](#implementation-guidelines)
- [Dependencies Strategy](#dependencies-strategy)

---

## Philosophy & Design Principles

### Core Principles to Maintain

1. **Minimal Core, Rich Ecosystem**
   - Keep `core.py` focused and simple
   - Advanced features in separate modules
   - Optional dependencies for advanced features

2. **Polars-First Architecture**
   - Vectorized preprocessing remains the heart
   - Event-driven execution for flexibility
   - Zero-copy operations where possible

3. **LLM-Friendly API**
   - Clean, predictable interfaces
   - Self-documenting code
   - Comprehensive docstrings

4. **Performance-Oriented**
   - Fast backtesting for evolutionary search
   - Parallel execution support
   - Efficient memory usage

5. **Extensibility Over Features**
   - Easy to extend (composition > inheritance)
   - Plugin architecture for custom features
   - Clear separation of concerns

---

## Phase 1: Core Trading Functionality

**Timeline**: 2-3 weeks  
**Priority**: 🔴 CRITICAL  
**Dependencies**: None

### 1.1 Order System Redesign

#### New `Order` Class
```python
@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    asset: str
    size: float  # Positive = buy, Negative = sell
    order_type: OrderType  # MARKET, LIMIT, STOP, STOP_LIMIT
    status: OrderStatus  # PENDING, FILLED, CANCELLED, REJECTED
    
    # Prices
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Execution details
    created_bar: int = 0
    filled_bar: Optional[int] = None
    filled_price: Optional[float] = None
    filled_size: float = 0.0
    
    # Order management
    valid_until: Optional[int] = None  # GTC if None
    parent_order: Optional[str] = None  # For OCO orders
    child_orders: List[str] = field(default_factory=list)  # SL/TP orders
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    commission_paid: float = 0.0
```

#### Order Types to Implement

**Market Orders** (✅ Already have)
```python
portfolio.order("BTC", 1.0)  # Buy 1 BTC at market
```

**Limit Orders** (NEW)
```python
portfolio.order("BTC", 1.0, limit=50000)  # Buy at $50k or lower
portfolio.order("BTC", -0.5, limit=52000)  # Sell at $52k or higher
```

**Stop-Loss Orders** (NEW)
```python
# Stop-loss on position
portfolio.set_stop_loss("BTC", stop_price=48000)
portfolio.set_stop_loss("BTC", stop_pct=-0.05)  # 5% below entry

# Stop-loss on order (bracket)
order = portfolio.order("BTC", 1.0)
portfolio.set_stop_loss_for_order(order, stop_price=48000)
```

**Take-Profit Orders** (NEW)
```python
portfolio.set_take_profit("BTC", target_price=55000)
portfolio.set_take_profit("BTC", target_pct=0.10)  # 10% profit
```

**Stop-Limit Orders** (NEW)
```python
# Triggered at stop, executed at limit
portfolio.order("BTC", -1.0, stop=48000, limit=47500)
```

**Trailing Stop-Loss** (NEW)
```python
# Percentage-based trailing
portfolio.set_trailing_stop("BTC", trail_pct=0.05)  # Trail by 5%

# ATR-based trailing
portfolio.set_trailing_stop("BTC", atr_multiplier=2.0, atr_period=14)

# Absolute value trailing
portfolio.set_trailing_stop("BTC", trail_amount=1000)  # Trail by $1000
```

**OCO Orders (One-Cancels-Other)** (NEW)
```python
# Bracket order: entry with SL and TP
portfolio.order_bracket(
    asset="BTC",
    size=1.0,
    stop_loss=48000,
    take_profit=55000
)

# Or manually create OCO
order1 = portfolio.order("BTC", -1.0, limit=55000)  # Take profit
order2 = portfolio.order("BTC", -1.0, stop=48000)   # Stop loss
portfolio.link_oco([order1, order2])
```

**Good-Till-Cancelled (GTC) vs Day Orders** (NEW)
```python
# GTC (default - stays until filled or cancelled)
portfolio.order("BTC", 1.0, limit=50000, valid_until=None)

# Day order (cancel at end of day/period)
portfolio.order("BTC", 1.0, limit=50000, valid_until="1D")

# Valid for specific number of bars
portfolio.order("BTC", 1.0, limit=50000, valid_until=10)  # 10 bars
```

#### Order Execution Logic

**Using OHLC for Realistic Fills**:
```python
# Current: Only uses close price
# New: Check if limit/stop hit during bar using high/low

def _check_order_fill(self, order: Order, bar: Dict) -> bool:
    """Check if order should fill based on bar's OHLC."""
    if order.order_type == OrderType.LIMIT:
        if order.size > 0:  # Buy limit
            # Fill if low <= limit price
            return bar.get("low", bar["close"]) <= order.limit_price
        else:  # Sell limit
            # Fill if high >= limit price
            return bar.get("high", bar["close"]) >= order.limit_price
    
    elif order.order_type == OrderType.STOP:
        if order.size > 0:  # Buy stop
            # Trigger if high >= stop price
            return bar.get("high", bar["close"]) >= order.stop_price
        else:  # Sell stop
            # Trigger if low <= stop price
            return bar.get("low", bar["close"]) <= order.stop_price
    
    # ... similar logic for stop-limit, etc.
```

**Order Priority**:
1. Stop-loss orders (highest priority)
2. Take-profit orders
3. Stop orders
4. Limit orders
5. Market orders

#### Portfolio Order Management

**New Methods**:
```python
# Order management
portfolio.get_orders(status=OrderStatus.PENDING)
portfolio.cancel_order(order_id)
portfolio.cancel_all_orders(asset=None)  # Cancel all or for specific asset
portfolio.modify_order(order_id, new_limit=price)

# Stop-loss / Take-profit management
portfolio.set_stop_loss(asset, stop_price=None, stop_pct=None)
portfolio.set_take_profit(asset, target_price=None, target_pct=None)
portfolio.set_trailing_stop(asset, trail_pct=None, trail_amount=None, atr_multiplier=None)
portfolio.remove_stop_loss(asset)
portfolio.remove_take_profit(asset)

# Get active risk management
portfolio.get_stop_loss(asset) -> Optional[float]
portfolio.get_take_profit(asset) -> Optional[float]
```

### 1.2 Trade Tracking System

#### `Trade` Class
```python
@dataclass
class Trade:
    """Represents a complete trade (entry to exit)."""
    trade_id: str
    asset: str
    
    # Entry
    entry_time: Any
    entry_bar: int
    entry_price: float
    entry_size: float
    entry_value: float
    entry_commission: float
    
    # Exit (None if still open)
    exit_time: Optional[Any] = None
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    exit_size: Optional[float] = None
    exit_value: Optional[float] = None
    exit_commission: Optional[float] = None
    
    # Performance
    pnl: Optional[float] = None  # Absolute P&L
    pnl_pct: Optional[float] = None  # Percentage P&L
    return_pct: Optional[float] = None  # Return on investment
    
    # Duration
    duration_bars: Optional[int] = None
    duration_time: Optional[Any] = None  # timedelta if timestamps available
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_adverse_excursion: float = 0.0  # MAE - worst drawdown during trade
    max_favorable_excursion: float = 0.0  # MFE - best profit during trade
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    @property
    def is_open(self) -> bool:
        return self.exit_time is None
    
    @property
    def is_winning(self) -> bool:
        return self.pnl is not None and self.pnl > 0
```

#### Trade Tracking in Portfolio

**New Portfolio Attributes**:
```python
class Portfolio:
    # ... existing attributes ...
    
    # Trade tracking
    trades: List[Trade] = field(default_factory=list)  # Active trades
    closed_trades: List[Trade] = field(default_factory=list)  # Completed trades
    _trade_counter: int = 0
```

**Trade Lifecycle Methods**:
```python
def _open_trade(self, asset: str, size: float, price: float, bar_index: int, timestamp: Any):
    """Create new trade on position entry."""
    self._trade_counter += 1
    trade = Trade(
        trade_id=f"T{self._trade_counter:06d}",
        asset=asset,
        entry_time=timestamp,
        entry_bar=bar_index,
        entry_price=price,
        entry_size=abs(size),
        entry_value=abs(size) * price,
        entry_commission=self._calculate_commission(abs(size) * price),
    )
    self.trades.append(trade)
    return trade

def _close_trade(self, trade: Trade, price: float, bar_index: int, timestamp: Any):
    """Close trade and calculate P&L."""
    trade.exit_time = timestamp
    trade.exit_bar = bar_index
    trade.exit_price = price
    trade.exit_size = trade.entry_size
    trade.exit_value = trade.entry_size * price
    trade.exit_commission = self._calculate_commission(trade.exit_value)
    
    # Calculate P&L
    if trade.entry_size > 0:  # Long trade
        trade.pnl = trade.exit_value - trade.entry_value - trade.entry_commission - trade.exit_commission
    else:  # Short trade
        trade.pnl = trade.entry_value - trade.exit_value - trade.entry_commission - trade.exit_commission
    
    trade.pnl_pct = (trade.pnl / trade.entry_value) * 100
    trade.return_pct = (trade.exit_price / trade.entry_price - 1) * 100
    trade.duration_bars = trade.exit_bar - trade.entry_bar
    
    # Move to closed trades
    self.trades.remove(trade)
    self.closed_trades.append(trade)
    return trade

def _update_trade_mae_mfe(self, current_price: float, bar_index: int):
    """Update MAE/MFE for active trades."""
    for trade in self.trades:
        unrealized_pnl_pct = ((current_price / trade.entry_price) - 1) * 100
        
        # Update MAE (worst point)
        if unrealized_pnl_pct < trade.max_adverse_excursion:
            trade.max_adverse_excursion = unrealized_pnl_pct
        
        # Update MFE (best point)
        if unrealized_pnl_pct > trade.max_favorable_excursion:
            trade.max_favorable_excursion = unrealized_pnl_pct
```

**Export Trades**:
```python
def get_trades_df(self) -> pl.DataFrame:
    """Export closed trades as Polars DataFrame."""
    if not self.closed_trades:
        return pl.DataFrame()
    
    return pl.DataFrame([
        {
            "trade_id": t.trade_id,
            "asset": t.asset,
            "entry_time": t.entry_time,
            "entry_price": t.entry_price,
            "entry_size": t.entry_size,
            "exit_time": t.exit_time,
            "exit_price": t.exit_price,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "duration_bars": t.duration_bars,
            "mae": t.max_adverse_excursion,
            "mfe": t.max_favorable_excursion,
            "stop_loss": t.stop_loss,
            "take_profit": t.take_profit,
        }
        for t in self.closed_trades
    ])
```

### 1.3 Trade-Level Metrics

**Add to `metrics.py`**:

```python
def calculate_trade_metrics(trades_df: pl.DataFrame) -> Dict[str, Any]:
    """Calculate trade-level statistics."""
    
    if len(trades_df) == 0:
        return {}
    
    winning_trades = trades_df.filter(pl.col("pnl") > 0)
    losing_trades = trades_df.filter(pl.col("pnl") < 0)
    
    num_trades = len(trades_df)
    num_winning = len(winning_trades)
    num_losing = len(losing_trades)
    
    metrics = {
        # Trade counts
        "num_trades": num_trades,
        "num_winning_trades": num_winning,
        "num_losing_trades": num_losing,
        "win_rate": num_winning / num_trades if num_trades > 0 else 0.0,
        
        # Trade performance
        "best_trade_pct": float(trades_df["pnl_pct"].max()) if num_trades > 0 else 0.0,
        "worst_trade_pct": float(trades_df["pnl_pct"].min()) if num_trades > 0 else 0.0,
        "avg_trade_pnl": float(trades_df["pnl"].mean()) if num_trades > 0 else 0.0,
        "avg_trade_pct": float(trades_df["pnl_pct"].mean()) if num_trades > 0 else 0.0,
        
        # Winning vs losing
        "avg_winning_trade": float(winning_trades["pnl"].mean()) if num_winning > 0 else 0.0,
        "avg_losing_trade": float(losing_trades["pnl"].mean()) if num_losing > 0 else 0.0,
        "avg_winning_trade_pct": float(winning_trades["pnl_pct"].mean()) if num_winning > 0 else 0.0,
        "avg_losing_trade_pct": float(losing_trades["pnl_pct"].mean()) if num_losing > 0 else 0.0,
        
        "largest_win": float(trades_df["pnl"].max()) if num_trades > 0 else 0.0,
        "largest_loss": float(trades_df["pnl"].min()) if num_trades > 0 else 0.0,
        
        # Duration
        "avg_trade_duration": float(trades_df["duration_bars"].mean()) if num_trades > 0 else 0.0,
        "max_trade_duration": int(trades_df["duration_bars"].max()) if num_trades > 0 else 0,
        "min_trade_duration": int(trades_df["duration_bars"].min()) if num_trades > 0 else 0,
        
        # Profit factor
        "gross_profit": float(winning_trades["pnl"].sum()) if num_winning > 0 else 0.0,
        "gross_loss": abs(float(losing_trades["pnl"].sum())) if num_losing > 0 else 0.0,
    }
    
    # Profit factor
    if metrics["gross_loss"] > 0:
        metrics["profit_factor"] = metrics["gross_profit"] / metrics["gross_loss"]
    else:
        metrics["profit_factor"] = float('inf') if metrics["gross_profit"] > 0 else 0.0
    
    # Expectancy
    metrics["expectancy"] = metrics["avg_trade_pnl"]
    
    # SQN (System Quality Number) - Van Tharp
    if num_trades > 0:
        avg_pnl = trades_df["pnl"].mean()
        std_pnl = trades_df["pnl"].std()
        if std_pnl and std_pnl > 0:
            metrics["sqn"] = (avg_pnl / std_pnl) * np.sqrt(num_trades)
        else:
            metrics["sqn"] = 0.0
    else:
        metrics["sqn"] = 0.0
    
    # Kelly Criterion
    if metrics["win_rate"] > 0 and metrics["win_rate"] < 1:
        avg_win = metrics["avg_winning_trade_pct"] / 100
        avg_loss = abs(metrics["avg_losing_trade_pct"]) / 100
        if avg_loss > 0:
            win_loss_ratio = avg_win / avg_loss
            kelly = metrics["win_rate"] - ((1 - metrics["win_rate"]) / win_loss_ratio)
            metrics["kelly_criterion"] = max(0.0, kelly)  # Don't go negative
        else:
            metrics["kelly_criterion"] = 0.0
    else:
        metrics["kelly_criterion"] = 0.0
    
    # Consecutive wins/losses
    pnl_series = trades_df["pnl"].to_list()
    metrics["max_consecutive_wins"] = _max_consecutive(pnl_series, lambda x: x > 0)
    metrics["max_consecutive_losses"] = _max_consecutive(pnl_series, lambda x: x < 0)
    
    # MAE/MFE analysis
    if "mae" in trades_df.columns:
        metrics["avg_mae"] = float(trades_df["mae"].mean())
        metrics["avg_mfe"] = float(trades_df["mfe"].mean())
    
    return metrics


def _max_consecutive(values: List[float], condition) -> int:
    """Calculate maximum consecutive occurrences."""
    max_count = 0
    current_count = 0
    
    for value in values:
        if condition(value):
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    
    return max_count
```

### 1.4 Short Selling Support

**Simple Implementation** (Phase 1):

```python
class Portfolio:
    # ... existing code ...
    
    def sell_short(self, asset: str, size: float) -> bool:
        """Explicitly sell short (same as negative order)."""
        return self.order(asset, -abs(size))
    
    def buy_to_cover(self, asset: str, size: float) -> bool:
        """Buy to cover short position."""
        current_pos = self.get_position(asset)
        if current_pos >= 0:
            return False  # Not short
        return self.order(asset, abs(size))
    
    def is_long(self, asset: str) -> bool:
        """Check if long position."""
        return self.get_position(asset) > 0
    
    def is_short(self, asset: str) -> bool:
        """Check if short position."""
        return self.get_position(asset) < 0
    
    def get_long_position(self, asset: str) -> float:
        """Get long position size (0 if short or flat)."""
        pos = self.get_position(asset)
        return pos if pos > 0 else 0.0
    
    def get_short_position(self, asset: str) -> float:
        """Get short position size (0 if long or flat)."""
        pos = self.get_position(asset)
        return abs(pos) if pos < 0 else 0.0
```

**Extended Implementation** (Phase 3 - with borrow costs):

```python
class Portfolio:
    def __init__(
        self,
        initial_cash: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        borrow_rate: float = 0.0,  # Annual borrow rate for shorts
        # ... other params ...
    ):
        # ... existing init ...
        self.borrow_rate = borrow_rate
        self._daily_borrow_rate = borrow_rate / 252  # Assuming 252 trading days
    
    def _calculate_borrow_costs(self) -> float:
        """Calculate daily borrow costs for short positions."""
        if self.borrow_rate == 0:
            return 0.0
        
        total_borrow_cost = 0.0
        for asset, qty in self.positions.items():
            if qty < 0:  # Short position
                short_value = abs(qty) * self._current_prices.get(asset, 0)
                daily_cost = short_value * self._daily_borrow_rate
                total_borrow_cost += daily_cost
        
        return total_borrow_cost
    
    def record_equity(self, timestamp: Any) -> None:
        """Record equity and deduct borrow costs."""
        # Deduct borrow costs for short positions
        borrow_costs = self._calculate_borrow_costs()
        self.cash -= borrow_costs
        
        # Record equity as before
        self.equity_curve.append(self.get_value())
        self.timestamps.append(timestamp)
```

---

## Phase 2: Visualization & Analysis

**Timeline**: 2-3 weeks  
**Priority**: 🔴 HIGH  
**Dependencies**: plotly (optional)

### 2.1 Plotting Module Architecture

**File Structure**:
```
polarbtest/
├── plotting/
│   ├── __init__.py
│   ├── charts.py         # Main chart generation
│   ├── equity.py         # Equity and performance plots
│   ├── trades.py         # Trade visualization
│   ├── indicators.py     # Indicator plotting helpers
│   └── themes.py         # Color schemes and styling
```

**Optional Import Pattern**:
```python
# In polarbtest/__init__.py
try:
    from polarbtest.plotting import plot_backtest
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    
    def plot_backtest(*args, **kwargs):
        raise ImportError(
            "Plotting requires plotly. Install with: pip install polarbtest[plotting]"
        )
```

### 2.2 Main Plotting Interface

**Primary API** (inspired by backtesting.py):

```python
def plot_backtest(
    data: pl.DataFrame,
    equity_curve: List[float],
    timestamps: List[Any],
    trades: List[Trade] = None,
    indicators: Dict[str, Union[str, List[str]]] = None,
    price_column: str = "close",
    volume_column: Optional[str] = "volume",
    show_positions: bool = True,
    show_drawdown: bool = True,
    plot_width: int = 1200,
    plot_height: int = 800,
    title: str = "Backtest Results",
    show: bool = True,
    save_path: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """
    Create interactive backtest visualization.
    
    Args:
        data: Price data DataFrame
        equity_curve: Portfolio equity over time
        timestamps: Timestamps for equity curve
        trades: List of Trade objects to visualize
        indicators: Dict mapping names to column names or list of column names
                   {"SMA 20": "sma_20", "Bollinger": ["bb_upper", "bb_mid", "bb_lower"]}
        price_column: Column name for price data
        volume_column: Column name for volume (None to hide)
        show_positions: Show position markers on chart
        show_drawdown: Include drawdown subplot
        plot_width: Plot width in pixels
        plot_height: Plot height in pixels
        title: Chart title
        show: Display plot in browser
        save_path: Save to HTML file
        
    Returns:
        Plotly Figure object
    """
```

### 2.3 Chart Components

**Main Price Chart**:
- Candlestick chart (OHLC data)
- Volume bars (optional subplot or overlay)
- Indicator overlays (moving averages, bands, etc.)
- Trade entry markers (green triangles)
- Trade exit markers (red triangles)
- Trade duration lines (connecting entry to exit)
- Active position highlighting

**Equity Curve Chart**:
- Portfolio value over time
- Buy & Hold comparison line (optional)
- Color-coded positive/negative regions
- Peak equity markers
- Drawdown shading

**Drawdown Chart**:
- Underwater plot
- Maximum drawdown marker
- Drawdown duration highlighting
- Recovery periods

**Returns Distribution**:
- Histogram of trade returns
- Normal distribution overlay
- Mean/median markers

**Example Layout**:
```python
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.5, 0.15, 0.2, 0.15],
    subplot_titles=(
        "Price & Trades",
        "Volume",
        "Equity Curve",
        "Drawdown"
    )
)

# Row 1: Price chart with trades
fig.add_trace(
    go.Candlestick(
        x=data["timestamp"],
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        name="Price"
    ),
    row=1, col=1
)

# Add trade markers
for trade in closed_trades:
    # Entry marker
    fig.add_trace(
        go.Scatter(
            x=[trade.entry_time],
            y=[trade.entry_price],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="green"),
            showlegend=False,
            hovertemplate=f"Entry: ${trade.entry_price:.2f}<br>Size: {trade.entry_size}"
        ),
        row=1, col=1
    )
    
    # Exit marker
    if trade.exit_time:
        fig.add_trace(
            go.Scatter(
                x=[trade.exit_time],
                y=[trade.exit_price],
                mode="markers",
                marker=dict(symbol="triangle-down", size=12, color="red"),
                showlegend=False,
                hovertemplate=f"Exit: ${trade.exit_price:.2f}<br>P/L: {trade.pnl_pct:.2f}%"
            ),
            row=1, col=1
        )
        
        # Trade duration line
        fig.add_trace(
            go.Scatter(
                x=[trade.entry_time, trade.exit_time],
                y=[trade.entry_price, trade.exit_price],
                mode="lines",
                line=dict(
                    color="green" if trade.pnl > 0 else "red",
                    width=1,
                    dash="dot"
                ),
                showlegend=False,
                hoverinfo="skip"
            ),
            row=1, col=1
        )

# Row 2: Volume
fig.add_trace(
    go.Bar(
        x=data["timestamp"],
        y=data["volume"],
        name="Volume",
        marker_color="rgba(128, 128, 128, 0.5)"
    ),
    row=2, col=1
)

# Row 3: Equity curve
fig.add_trace(
    go.Scatter(
        x=timestamps,
        y=equity_curve,
        mode="lines",
        name="Equity",
        line=dict(color="blue", width=2),
        fill="tonexty",
        fillcolor="rgba(0, 100, 200, 0.2)"
    ),
    row=3, col=1
)

# Row 4: Drawdown
drawdown_series = calculate_drawdown(equity_curve)
fig.add_trace(
    go.Scatter(
        x=timestamps,
        y=drawdown_series,
        mode="lines",
        name="Drawdown",
        line=dict(color="red", width=1),
        fill="tozeroy",
        fillcolor="rgba(255, 0, 0, 0.3)"
    ),
    row=4, col=1
)

# Update layout
fig.update_layout(
    title=title,
    width=plot_width,
    height=plot_height,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    template="plotly_white"
)

if save_path:
    fig.write_html(save_path)

if show:
    fig.show()

return fig
```

### 2.4 Indicator Visualization Helpers

```python
def add_indicator_overlay(
    fig: go.Figure,
    data: pl.DataFrame,
    column: str,
    name: str,
    row: int = 1,
    color: str = None,
    width: int = 1,
    dash: str = None
):
    """Add indicator line to chart."""
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data[column],
            mode="lines",
            name=name,
            line=dict(color=color, width=width, dash=dash)
        ),
        row=row, col=1
    )


def add_band_indicator(
    fig: go.Figure,
    data: pl.DataFrame,
    upper_col: str,
    lower_col: str,
    middle_col: str = None,
    name: str = "Bands",
    row: int = 1,
    color: str = "rgba(128, 128, 128, 0.2)"
):
    """Add band indicator (Bollinger, Keltner, etc.)."""
    # Upper band
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data[upper_col],
            mode="lines",
            name=f"{name} Upper",
            line=dict(color="gray", width=1, dash="dash"),
            showlegend=False
        ),
        row=row, col=1
    )
    
    # Lower band
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data[lower_col],
            mode="lines",
            name=f"{name} Lower",
            line=dict(color="gray", width=1, dash="dash"),
            fill="tonexty",
            fillcolor=color,
            showlegend=True
        ),
        row=row, col=1
    )
    
    # Middle line (optional)
    if middle_col:
        fig.add_trace(
            go.Scatter(
                x=data["timestamp"],
                y=data[middle_col],
                mode="lines",
                name=f"{name} Middle",
                line=dict(color="blue", width=1)
            ),
            row=row, col=1
        )


def add_oscillator_subplot(
    fig: go.Figure,
    data: pl.DataFrame,
    column: str,
    name: str,
    row: int,
    overbought: float = None,
    oversold: float = None
):
    """Add oscillator (RSI, Stochastic, etc.) as subplot."""
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data[column],
            mode="lines",
            name=name,
            line=dict(color="purple", width=2)
        ),
        row=row, col=1
    )
    
    # Add horizontal reference lines
    if overbought:
        fig.add_hline(
            y=overbought,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            row=row, col=1
        )
    
    if oversold:
        fig.add_hline(
            y=oversold,
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            row=row, col=1
        )
```

### 2.5 Enhanced Metrics

**Add to `metrics.py`**:

```python
def omega_ratio(equity_df: pl.DataFrame, threshold: float = 0.0) -> float:
    """
    Calculate Omega ratio.
    
    Omega = Probability weighted ratio of gains vs losses relative to threshold.
    """
    returns = equity_df["equity"].pct_change().drop_nulls()
    
    if len(returns) == 0:
        return 0.0
    
    gains = returns.filter(returns > threshold) - threshold
    losses = threshold - returns.filter(returns < threshold)
    
    total_gains = gains.sum() if len(gains) > 0 else 0.0
    total_losses = losses.sum() if len(losses) > 0 else 0.0
    
    if total_losses > 0:
        return float(total_gains / total_losses)
    return float("inf") if total_gains > 0 else 0.0


def ulcer_index(equity_df: pl.DataFrame) -> float:
    """
    Calculate Ulcer Index (downside volatility measure).
    
    Ulcer Index = sqrt(mean(squared percentage drawdowns))
    """
    equity = equity_df["equity"]
    running_max = equity.cum_max()
    drawdown_pct = ((equity - running_max) / running_max) * 100
    
    ulcer = np.sqrt((drawdown_pct ** 2).mean())
    return float(ulcer)


def tail_ratio(equity_df: pl.DataFrame, percentile: float = 0.95) -> float:
    """
    Calculate Tail Ratio (right tail / left tail).
    
    Measures asymmetry of return distribution tails.
    """
    returns = equity_df["equity"].pct_change().drop_nulls()
    
    if len(returns) == 0:
        return 0.0
    
    right_tail = abs(float(returns.quantile(percentile)))
    left_tail = abs(float(returns.quantile(1 - percentile)))
    
    if left_tail > 0:
        return right_tail / left_tail
    return 0.0


def information_ratio(
    equity_df: pl.DataFrame,
    benchmark_returns: pl.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Information Ratio (excess return / tracking error).
    """
    returns = equity_df["equity"].pct_change().drop_nulls()
    
    if len(returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    
    # Align lengths
    min_len = min(len(returns), len(benchmark_returns))
    returns = returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]
    
    # Excess returns
    excess_returns = returns - benchmark_returns
    
    # Tracking error (std of excess returns)
    tracking_error = excess_returns.std()
    
    if tracking_error and tracking_error > 0:
        ir = (excess_returns.mean() / tracking_error) * np.sqrt(periods_per_year)
        return float(ir)
    
    return 0.0


def calculate_alpha_beta(
    equity_df: pl.DataFrame,
    benchmark_returns: pl.Series,
    risk_free_rate: float = 0.0
) -> tuple[float, float]:
    """
    Calculate Alpha and Beta vs benchmark.
    
    Beta = Cov(returns, benchmark) / Var(benchmark)
    Alpha = Portfolio return - (Risk-free rate + Beta * (Benchmark return - Risk-free rate))
    """
    returns = equity_df["equity"].pct_change().drop_nulls()
    
    if len(returns) == 0 or len(benchmark_returns) == 0:
        return 0.0, 0.0
    
    # Align lengths
    min_len = min(len(returns), len(benchmark_returns))
    returns = returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]
    
    # Calculate beta
    covariance = np.cov(returns.to_numpy(), benchmark_returns.to_numpy())[0, 1]
    benchmark_variance = benchmark_returns.var()
    
    if benchmark_variance > 0:
        beta = covariance / benchmark_variance
    else:
        beta = 0.0
    
    # Calculate alpha (annualized)
    portfolio_return = (equity_df["equity"][-1] / equity_df["equity"][0]) - 1
    benchmark_return = (benchmark_returns.sum())
    
    alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
    
    return float(alpha), float(beta)


def calculate_drawdown_duration(equity_df: pl.DataFrame) -> Dict[str, int]:
    """
    Calculate drawdown duration statistics.
    
    Returns max and average drawdown duration in bars.
    """
    equity = equity_df["equity"]
    running_max = equity.cum_max()
    is_drawdown = equity < running_max
    
    # Find drawdown periods
    drawdown_periods = []
    current_dd_length = 0
    
    for in_dd in is_drawdown.to_list():
        if in_dd:
            current_dd_length += 1
        else:
            if current_dd_length > 0:
                drawdown_periods.append(current_dd_length)
                current_dd_length = 0
    
    # Handle if still in drawdown at end
    if current_dd_length > 0:
        drawdown_periods.append(current_dd_length)
    
    if drawdown_periods:
        return {
            "max_drawdown_duration": max(drawdown_periods),
            "avg_drawdown_duration": sum(drawdown_periods) / len(drawdown_periods),
            "num_drawdown_periods": len(drawdown_periods)
        }
    else:
        return {
            "max_drawdown_duration": 0,
            "avg_drawdown_duration": 0.0,
            "num_drawdown_periods": 0
        }


def calculate_monthly_returns(equity_df: pl.DataFrame) -> pl.DataFrame:
    """Calculate monthly returns table."""
    # Requires timestamp column
    if "timestamp" not in equity_df.columns:
        return pl.DataFrame()
    
    monthly = (
        equity_df
        .with_columns([
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.month().alias("month")
        ])
        .group_by(["year", "month"])
        .agg([
            pl.col("equity").first().alias("start_equity"),
            pl.col("equity").last().alias("end_equity")
        ])
        .with_columns([
            ((pl.col("end_equity") / pl.col("start_equity")) - 1).alias("return")
        ])
        .sort(["year", "month"])
    )
    
    return monthly
```

**Integration into main metrics**:

```python
def calculate_metrics(
    equity_df: pl.DataFrame,
    initial_capital: float,
    benchmark_returns: Optional[pl.Series] = None
) -> Dict[str, Any]:
    """Enhanced metrics calculation."""
    
    # ... existing metrics ...
    
    # Additional metrics
    metrics["omega_ratio"] = omega_ratio(equity_df)
    metrics["ulcer_index"] = ulcer_index(equity_df)
    metrics["tail_ratio"] = tail_ratio(equity_df)
    
    # Drawdown duration
    dd_stats = calculate_drawdown_duration(equity_df)
    metrics.update(dd_stats)
    
    # Benchmark comparison (if provided)
    if benchmark_returns is not None:
        alpha, beta = calculate_alpha_beta(equity_df, benchmark_returns)
        metrics["alpha"] = alpha
        metrics["beta"] = beta
        metrics["information_ratio"] = information_ratio(equity_df, benchmark_returns)
    
    return metrics
```

---

## Phase 3: Risk Management & Realism

**Timeline**: 2 weeks  
**Priority**: 🟡 MEDIUM  
**Dependencies**: Phase 1

### 3.1 Position Sizing Strategies

**File**: `polarbtest/sizers.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class Sizer(ABC):
    """Base class for position sizing strategies."""
    
    @abstractmethod
    def calculate(
        self,
        portfolio: "Portfolio",
        asset: str,
        signal_strength: float = 1.0,
        **kwargs
    ) -> float:
        """
        Calculate position size.
        
        Args:
            portfolio: Portfolio instance
            asset: Asset to size
            signal_strength: Signal strength (0-1)
            **kwargs: Additional parameters (price, stop_loss, etc.)
            
        Returns:
            Position size (number of units)
        """
        pass


class FixedSizer(Sizer):
    """Fixed position size."""
    
    def __init__(self, size: float):
        self.size = size
    
    def calculate(self, portfolio, asset, signal_strength=1.0, **kwargs):
        return self.size * signal_strength


class PercentSizer(Sizer):
    """Position size as percentage of portfolio."""
    
    def __init__(self, percent: float = 0.1):
        self.percent = percent
    
    def calculate(self, portfolio, asset, signal_strength=1.0, **kwargs):
        price = kwargs.get("price") or portfolio._current_prices.get(asset, 0)
        if price <= 0:
            return 0.0
        
        portfolio_value = portfolio.get_value()
        target_value = portfolio_value * self.percent * signal_strength
        
        return target_value / price


class FixedRiskSizer(Sizer):
    """
    Risk-based position sizing.
    
    Size position so that loss from stop-loss = X% of portfolio.
    """
    
    def __init__(self, risk_pct: float = 0.02):
        """
        Args:
            risk_pct: Percentage of portfolio to risk per trade (e.g., 0.02 = 2%)
        """
        self.risk_pct = risk_pct
    
    def calculate(self, portfolio, asset, signal_strength=1.0, **kwargs):
        price = kwargs.get("price") or portfolio._current_prices.get(asset, 0)
        stop_loss = kwargs.get("stop_loss")
        
        if not stop_loss or price <= 0:
            return 0.0
        
        # Calculate risk per unit
        risk_per_unit = abs(price - stop_loss)
        
        if risk_per_unit <= 0:
            return 0.0
        
        # Calculate position size
        portfolio_value = portfolio.get_value()
        risk_amount = portfolio_value * self.risk_pct * signal_strength
        
        size = risk_amount / risk_per_unit
        
        return size


class KellySizer(Sizer):
    """
    Kelly Criterion position sizing.
    
    Kelly % = (Win% * AvgWin - Loss% * AvgLoss) / AvgLoss
    """
    
    def __init__(
        self,
        win_rate: float = None,
        avg_win: float = None,
        avg_loss: float = None,
        fraction: float = 1.0
    ):
        """
        Args:
            win_rate: Historical win rate (if known)
            avg_win: Average winning trade %
            avg_loss: Average losing trade %
            fraction: Fraction of Kelly to use (0.5 = half Kelly, safer)
        """
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.fraction = fraction
    
    def calculate(self, portfolio, asset, signal_strength=1.0, **kwargs):
        # Use provided stats or calculate from portfolio history
        win_rate = kwargs.get("win_rate") or self.win_rate
        avg_win = kwargs.get("avg_win") or self.avg_win
        avg_loss = kwargs.get("avg_loss") or self.avg_loss
        
        if not all([win_rate, avg_win, avg_loss]):
            # Can't calculate Kelly, fall back to fixed %
            return PercentSizer(0.1).calculate(portfolio, asset, signal_strength, **kwargs)
        
        # Kelly formula
        kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
        kelly_pct = max(0.0, min(kelly_pct, 1.0))  # Clamp to [0, 1]
        
        # Apply fraction and signal strength
        final_pct = kelly_pct * self.fraction * signal_strength
        
        # Convert to position size
        price = kwargs.get("price") or portfolio._current_prices.get(asset, 0)
        if price <= 0:
            return 0.0
        
        portfolio_value = portfolio.get_value()
        target_value = portfolio_value * final_pct
        
        return target_value / price


class VolatilitySizer(Sizer):
    """
    Size position based on volatility (ATR).
    
    Target constant volatility across all positions.
    """
    
    def __init__(self, target_volatility: float = 0.15):
        """
        Args:
            target_volatility: Target portfolio volatility (e.g., 0.15 = 15%)
        """
        self.target_volatility = target_volatility
    
    def calculate(self, portfolio, asset, signal_strength=1.0, **kwargs):
        price = kwargs.get("price") or portfolio._current_prices.get(asset, 0)
        atr = kwargs.get("atr")
        
        if not atr or price <= 0:
            return 0.0
        
        # Asset volatility as fraction
        asset_volatility = atr / price
        
        if asset_volatility <= 0:
            return 0.0
        
        # Adjust position size to achieve target volatility
        portfolio_value = portfolio.get_value()
        volatility_adjusted_pct = (self.target_volatility / asset_volatility) * signal_strength
        
        # Clamp to reasonable range
        volatility_adjusted_pct = min(volatility_adjusted_pct, 1.0)
        
        target_value = portfolio_value * volatility_adjusted_pct
        return target_value / price


class MaxPositionSizer(Sizer):
    """
    Wrapper that enforces maximum position size.
    """
    
    def __init__(self, base_sizer: Sizer, max_pct: float = 0.25):
        """
        Args:
            base_sizer: Underlying sizer to wrap
            max_pct: Maximum position as % of portfolio
        """
        self.base_sizer = base_sizer
        self.max_pct = max_pct
    
    def calculate(self, portfolio, asset, signal_strength=1.0, **kwargs):
        # Get size from base sizer
        size = self.base_sizer.calculate(portfolio, asset, signal_strength, **kwargs)
        
        # Apply cap
        price = kwargs.get("price") or portfolio._current_prices.get(asset, 0)
        if price <= 0:
            return size
        
        portfolio_value = portfolio.get_value()
        max_value = portfolio_value * self.max_pct
        max_size = max_value / price
        
        return min(size, max_size)
```

**Integration with Portfolio**:

```python
# Add convenience methods to Portfolio
class Portfolio:
    # ... existing code ...
    
    def order_with_sizer(
        self,
        asset: str,
        sizer: Sizer,
        signal_strength: float = 1.0,
        **kwargs
    ) -> bool:
        """
        Place order using a sizer.
        
        Args:
            asset: Asset to trade
            sizer: Sizer instance to calculate position size
            signal_strength: Signal strength (0-1)
            **kwargs: Additional parameters (stop_loss, atr, etc.)
        """
        size = sizer.calculate(
            portfolio=self,
            asset=asset,
            signal_strength=signal_strength,
            price=self._current_prices.get(asset),
            **kwargs
        )
        
        return self.order_target(asset, size)
```

**Usage Example**:

```python
class RiskManagedStrategy(Strategy):
    def __init__(self, **params):
        super().__init__(**params)
        # Risk 2% per trade
        self.sizer = FixedRiskSizer(risk_pct=0.02)
    
    def preprocess(self, df):
        return df.with_columns([
            ind.atr("high", "low", "close", 14).alias("atr"),
            ind.sma("close", 20).alias("sma_20")
        ])
    
    def next(self, ctx):
        if ctx.bar_index < 20:
            return
        
        price = ctx.row["close"]
        atr = ctx.row.get("atr", 0)
        
        if ctx.row["close"] > ctx.row["sma_20"]:
            # Buy signal - calculate stop loss
            stop_loss = price - 2 * atr
            
            # Size position based on risk
            ctx.portfolio.order_with_sizer(
                "asset",
                self.sizer,
                signal_strength=1.0,
                price=price,
                stop_loss=stop_loss
            )
            
            # Set stop loss on position
            ctx.portfolio.set_stop_loss("asset", stop_price=stop_loss)
```

### 3.2 Risk Limits

**Add to Portfolio `__init__`**:

```python
class Portfolio:
    def __init__(
        self,
        initial_cash: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        order_delay: int = 0,
        
        # Risk limits (NEW)
        max_position_size: Optional[float] = None,  # Max % in single position
        max_total_exposure: Optional[float] = None,  # Max % total exposure
        max_drawdown_stop: Optional[float] = None,   # Stop trading at X% drawdown
        daily_loss_limit: Optional[float] = None,    # Max daily loss %
        
        # Short selling (NEW)
        borrow_rate: float = 0.0,  # Annual borrow rate for shorts
    ):
        # ... existing init ...
        
        # Risk limits
        self.max_position_size = max_position_size
        self.max_total_exposure = max_total_exposure
        self.max_drawdown_stop = max_drawdown_stop
        self.daily_loss_limit = daily_loss_limit
        
        # State tracking
        self._peak_equity = initial_cash
        self._daily_start_equity = initial_cash
        self._trading_halted = False
        self._halt_reason = None
```

**Risk Limit Enforcement**:

```python
def _check_risk_limits(self) -> bool:
    """
    Check if risk limits are violated.
    
    Returns False if trading should be halted.
    """
    current_equity = self.get_value()
    
    # Check max drawdown stop
    if self.max_drawdown_stop:
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        
        current_dd = (self._peak_equity - current_equity) / self._peak_equity
        if current_dd >= self.max_drawdown_stop:
            self._trading_halted = True
            self._halt_reason = f"Max drawdown limit reached: {current_dd:.2%}"
            return False
    
    # Check daily loss limit
    if self.daily_loss_limit:
        daily_loss = (self._daily_start_equity - current_equity) / self._daily_start_equity
        if daily_loss >= self.daily_loss_limit:
            self._trading_halted = True
            self._halt_reason = f"Daily loss limit reached: {daily_loss:.2%}"
            return False
    
    return True


def _check_exposure_limit(self, new_order_value: float) -> bool:
    """Check if new order would violate exposure limits."""
    if not self.max_total_exposure:
        return True
    
    # Calculate current exposure
    current_exposure = sum(
        abs(qty) * self._current_prices.get(asset, 0)
        for asset, qty in self.positions.items()
    )
    
    total_portfolio = self.get_value()
    new_exposure = (current_exposure + abs(new_order_value)) / total_portfolio
    
    return new_exposure <= self.max_total_exposure


def _check_position_size_limit(self, asset: str, size: float, price: float) -> float:
    """
    Cap position size to max limit.
    
    Returns adjusted size.
    """
    if not self.max_position_size:
        return size
    
    total_portfolio = self.get_value()
    max_value = total_portfolio * self.max_position_size
    max_size = max_value / price
    
    return min(abs(size), max_size) * (1 if size > 0 else -1)


# Modify order execution to check limits
def order(self, asset: str, quantity: float, limit_price: Optional[float] = None) -> bool:
    """Place order with risk limit checks."""
    
    # Check if trading is halted
    if self._trading_halted:
        return False
    
    # Check risk limits
    if not self._check_risk_limits():
        return False
    
    # Check and adjust position size
    price = limit_price or self._current_prices.get(asset, 0)
    quantity = self._check_position_size_limit(asset, quantity, price)
    
    # Check exposure limit
    order_value = abs(quantity) * price
    if not self._check_exposure_limit(order_value):
        return False
    
    # Proceed with order (existing logic)
    # ...
```

### 3.3 Advanced Commission Models

```python
from typing import Callable

class CommissionModel:
    """Base class for commission models."""
    
    def calculate(self, value: float, asset: str, is_maker: bool = False) -> float:
        """Calculate commission for trade."""
        raise NotImplementedError


class PercentCommission(CommissionModel):
    """Simple percentage commission."""
    
    def __init__(self, rate: float = 0.001):
        self.rate = rate
    
    def calculate(self, value: float, asset: str, is_maker: bool = False) -> float:
        return value * self.rate


class MakerTakerCommission(CommissionModel):
    """Maker/taker fee structure."""
    
    def __init__(self, maker_rate: float = -0.0002, taker_rate: float = 0.0004):
        self.maker_rate = maker_rate
        self.taker_rate = taker_rate
    
    def calculate(self, value: float, asset: str, is_maker: bool = False) -> float:
        rate = self.maker_rate if is_maker else self.taker_rate
        return value * rate


class TieredCommission(CommissionModel):
    """Volume-based tiered commission."""
    
    def __init__(self, tiers: Dict[float, float]):
        """
        Args:
            tiers: Dict mapping volume thresholds to rates
                   {0: 0.001, 100_000: 0.0008, 1_000_000: 0.0006}
        """
        self.tiers = sorted(tiers.items(), reverse=True)
    
    def calculate(self, value: float, asset: str, is_maker: bool = False) -> float:
        # This would require tracking total volume
        # For simplicity, just use value
        for threshold, rate in self.tiers:
            if value >= threshold:
                return value * rate
        return 0.0


class CustomCommission(CommissionModel):
    """Custom commission using callable."""
    
    def __init__(self, func: Callable[[float, str, bool], float]):
        self.func = func
    
    def calculate(self, value: float, asset: str, is_maker: bool = False) -> float:
        return self.func(value, asset, is_maker)


# Update Portfolio to use commission models
class Portfolio:
    def __init__(
        self,
        # ... other params ...
        commission: Union[float, CommissionModel] = 0.001,
    ):
        # Convert float to model
        if isinstance(commission, (int, float)):
            self.commission_model = PercentCommission(commission)
        else:
            self.commission_model = commission
    
    def _calculate_commission(self, value: float, asset: str = "", is_maker: bool = False) -> float:
        """Calculate commission using model."""
        return self.commission_model.calculate(value, asset, is_maker)
```

### 3.4 Margin & Leverage

```python
class Portfolio:
    def __init__(
        self,
        # ... existing params ...
        
        # Margin & Leverage (NEW)
        leverage: float = 1.0,              # Maximum leverage (1.0 = no leverage)
        margin_requirement: float = 1.0,    # Initial margin (0.5 = 2x leverage)
        maintenance_margin: float = 0.25,   # Maintenance margin
        margin_call_close_pct: float = 0.5, # Close X% of positions on margin call
    ):
        # ... existing init ...
        
        self.leverage = leverage
        self.margin_requirement = margin_requirement
        self.maintenance_margin = maintenance_margin
        self.margin_call_close_pct = margin_call_close_pct
        
        self._in_margin_call = False
    
    def get_buying_power(self) -> float:
        """Calculate available buying power with leverage."""
        # Available cash * leverage
        base_power = self.cash * self.leverage
        
        # Subtract used margin
        used_margin = self._calculate_used_margin()
        
        return max(0, base_power - used_margin)
    
    def _calculate_used_margin(self) -> float:
        """Calculate margin currently in use."""
        total_position_value = sum(
            abs(qty) * self._current_prices.get(asset, 0)
            for asset, qty in self.positions.items()
        )
        return total_position_value * self.margin_requirement
    
    def _check_margin_requirements(self) -> bool:
        """Check if account meets margin requirements."""
        equity = self.get_value()
        used_margin = self._calculate_used_margin()
        
        if used_margin == 0:
            return True
        
        # Margin ratio = Equity / Used Margin
        margin_ratio = equity / used_margin
        
        # Check maintenance margin
        if margin_ratio < self.maintenance_margin:
            self._handle_margin_call()
            return False
        
        return True
    
    def _handle_margin_call(self):
        """Handle margin call by closing positions."""
        if self._in_margin_call:
            return  # Already handling
        
        self._in_margin_call = True
        
        # Close percentage of positions (largest first)
        positions_by_size = sorted(
            self.positions.items(),
            key=lambda x: abs(x[1]) * self._current_prices.get(x[0], 0),
            reverse=True
        )
        
        # Close positions until margin is restored
        for asset, qty in positions_by_size:
            close_size = qty * self.margin_call_close_pct
            self.order(asset, -close_size)
            
            # Check if margin restored
            if self._check_margin_requirements():
                break
        
        self._in_margin_call = False
```

---

## Phase 4: Extended Indicators & Utilities

**Timeline**: 1-2 weeks  
**Priority**: 🟡 MEDIUM  
**Dependencies**: None

### 4.1 Additional Technical Indicators

**Add to `polarbtest/indicators.py`**:

#### Trend Indicators

```python
def wma(column: str, period: int) -> pl.Expr:
    """
    Weighted Moving Average.
    
    More weight to recent prices: weights = [1, 2, 3, ..., period]
    """
    weights = list(range(1, period + 1))
    weight_sum = sum(weights)
    
    # Create weighted average using rolling window
    return (
        pl.col(column)
        .rolling_map(
            lambda s: np.average(s, weights=weights[-len(s):]) if len(s) == period else None,
            window_size=period
        )
    )


def hma(column: str, period: int) -> pl.Expr:
    """
    Hull Moving Average - smoother and less lag.
    
    HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    """
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(column, half_period)
    wma_full = wma(column, period)
    
    # 2 * WMA(n/2) - WMA(n)
    raw_hma = 2 * wma_half - wma_full
    
    # Apply WMA with sqrt(n) period to the result
    # This is tricky in Polars - need to create intermediate column
    return wma(raw_hma, sqrt_period)


def vwap(price: str = "close", volume: str = "volume") -> pl.Expr:
    """
    Volume Weighted Average Price.
    
    VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
    """
    typical_price = pl.col(price)
    vol = pl.col(volume)
    
    return (typical_price * vol).cum_sum() / vol.cum_sum()


def supertrend(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    period: int = 10,
    multiplier: float = 3.0
) -> tuple[pl.Expr, pl.Expr]:
    """
    SuperTrend indicator.
    
    Returns:
        (supertrend_line, trend_direction)
        trend_direction: 1 = uptrend, -1 = downtrend
    """
    # Calculate ATR
    atr_val = atr(high, low, close, period)
    
    # Basic bands
    hl_avg = (pl.col(high) + pl.col(low)) / 2
    upper_band = hl_avg + (multiplier * atr_val)
    lower_band = hl_avg - (multiplier * atr_val)
    
    # This requires iterative logic - complex in Polars
    # Simplified version: just return the bands
    # Full implementation would need custom function
    
    return upper_band, lower_band


def adx(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    period: int = 14
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """
    Average Directional Index (ADX) - trend strength.
    
    Returns:
        (adx, plus_di, minus_di)
    """
    # True Range
    tr = atr(high, low, close, 1)  # TR without smoothing
    
    # Directional Movement
    high_diff = pl.col(high) - pl.col(high).shift(1)
    low_diff = pl.col(low).shift(1) - pl.col(low)
    
    plus_dm = pl.when((high_diff > low_diff) & (high_diff > 0)).then(high_diff).otherwise(0)
    minus_dm = pl.when((low_diff > high_diff) & (low_diff > 0)).then(low_diff).otherwise(0)
    
    # Smooth with EMA
    plus_dm_smooth = plus_dm.ewm_mean(alpha=1/period, adjust=False)
    minus_dm_smooth = minus_dm.ewm_mean(alpha=1/period, adjust=False)
    tr_smooth = tr.ewm_mean(alpha=1/period, adjust=False)
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # ADX
    dx = 100 * (pl.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx_line = dx.ewm_mean(alpha=1/period, adjust=False)
    
    return adx_line, plus_di, minus_di
```

#### Momentum Indicators

```python
def stochastic(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    k_period: int = 14,
    d_period: int = 3
) -> tuple[pl.Expr, pl.Expr]:
    """
    Stochastic Oscillator.
    
    %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
    %D = SMA(%K, d_period)
    
    Returns:
        (percent_k, percent_d)
    """
    lowest_low = pl.col(low).rolling_min(window_size=k_period)
    highest_high = pl.col(high).rolling_max(window_size=k_period)
    
    percent_k = 100 * (pl.col(close) - lowest_low) / (highest_high - lowest_low)
    percent_d = percent_k.rolling_mean(window_size=d_period)
    
    return percent_k, percent_d


def williams_r(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    period: int = 14
) -> pl.Expr:
    """
    Williams %R - similar to Stochastic but inverted scale (-100 to 0).
    
    %R = -100 * (Highest High - Close) / (Highest High - Lowest Low)
    """
    lowest_low = pl.col(low).rolling_min(window_size=period)
    highest_high = pl.col(high).rolling_max(window_size=period)
    
    return -100 * (highest_high - pl.col(close)) / (highest_high - lowest_low)


def cci(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    period: int = 20
) -> pl.Expr:
    """
    Commodity Channel Index.
    
    CCI = (Typical Price - SMA(Typical Price)) / (0.015 * Mean Deviation)
    """
    typical_price = (pl.col(high) + pl.col(low) + pl.col(close)) / 3
    tp_sma = typical_price.rolling_mean(window_size=period)
    
    # Mean deviation
    mean_deviation = (typical_price - tp_sma).abs().rolling_mean(window_size=period)
    
    return (typical_price - tp_sma) / (0.015 * mean_deviation)


def mfi(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume",
    period: int = 14
) -> pl.Expr:
    """
    Money Flow Index - volume-weighted RSI.
    
    MFI = 100 - (100 / (1 + Money Flow Ratio))
    """
    typical_price = (pl.col(high) + pl.col(low) + pl.col(close)) / 3
    money_flow = typical_price * pl.col(volume)
    
    # Positive and negative money flow
    price_change = typical_price.diff()
    positive_flow = pl.when(price_change > 0).then(money_flow).otherwise(0)
    negative_flow = pl.when(price_change < 0).then(money_flow).otherwise(0)
    
    # Sum over period
    positive_mf = positive_flow.rolling_sum(window_size=period)
    negative_mf = negative_flow.rolling_sum(window_size=period)
    
    # MFI
    mf_ratio = positive_mf / negative_mf
    mfi_value = 100 - (100 / (1 + mf_ratio))
    
    return mfi_value


def roc(column: str, period: int = 12) -> pl.Expr:
    """
    Rate of Change.
    
    ROC = ((Close - Close[n periods ago]) / Close[n periods ago]) * 100
    """
    return ((pl.col(column) - pl.col(column).shift(period)) / pl.col(column).shift(period)) * 100
```

#### Volatility Indicators

```python
def keltner_channels(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    period: int = 20,
    atr_multiplier: float = 2.0,
    atr_period: int = 10
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """
    Keltner Channels - ATR-based bands.
    
    Returns:
        (upper_band, middle_line, lower_band)
    """
    middle = ema(close, period)
    atr_val = atr(high, low, close, atr_period)
    
    upper = middle + (atr_multiplier * atr_val)
    lower = middle - (atr_multiplier * atr_val)
    
    return upper, middle, lower


def donchian_channels(
    high: str = "high",
    low: str = "low",
    period: int = 20
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """
    Donchian Channels - highest high and lowest low.
    
    Returns:
        (upper_band, middle_line, lower_band)
    """
    upper = pl.col(high).rolling_max(window_size=period)
    lower = pl.col(low).rolling_min(window_size=period)
    middle = (upper + lower) / 2
    
    return upper, middle, lower
```

#### Volume Indicators

```python
def obv(close: str = "close", volume: str = "volume") -> pl.Expr:
    """
    On-Balance Volume.
    
    OBV increases by volume when price rises, decreases when price falls.
    """
    price_change = pl.col(close).diff()
    
    signed_volume = pl.when(price_change > 0).then(pl.col(volume)) \
                     .when(price_change < 0).then(-pl.col(volume)) \
                     .otherwise(0)
    
    return signed_volume.cum_sum()


def ad_line(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume"
) -> pl.Expr:
    """
    Accumulation/Distribution Line.
    
    AD = ((Close - Low) - (High - Close)) / (High - Low) * Volume
    """
    clv = ((pl.col(close) - pl.col(low)) - (pl.col(high) - pl.col(close))) / \
          (pl.col(high) - pl.col(low))
    
    ad_value = clv * pl.col(volume)
    
    return ad_value.cum_sum()
```

#### Support/Resistance

```python
def pivot_points(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    method: str = "standard"
) -> Dict[str, pl.Expr]:
    """
    Calculate pivot points and support/resistance levels.
    
    Args:
        method: "standard", "fibonacci", "woodie", "camarilla"
    
    Returns:
        Dict with keys: "pivot", "r1", "r2", "r3", "s1", "s2", "s3"
    """
    h = pl.col(high).shift(1)  # Previous high
    l = pl.col(low).shift(1)   # Previous low
    c = pl.col(close).shift(1) # Previous close
    
    if method == "standard":
        pivot = (h + l + c) / 3
        r1 = 2 * pivot - l
        r2 = pivot + (h - l)
        r3 = h + 2 * (pivot - l)
        s1 = 2 * pivot - h
        s2 = pivot - (h - l)
        s3 = l - 2 * (h - pivot)
        
    elif method == "fibonacci":
        pivot = (h + l + c) / 3
        r1 = pivot + 0.382 * (h - l)
        r2 = pivot + 0.618 * (h - l)
        r3 = pivot + 1.000 * (h - l)
        s1 = pivot - 0.382 * (h - l)
        s2 = pivot - 0.618 * (h - l)
        s3 = pivot - 1.000 * (h - l)
        
    elif method == "woodie":
        pivot = (h + l + 2 * c) / 4
        r1 = 2 * pivot - l
        r2 = pivot + (h - l)
        r3 = h + 2 * (pivot - l)
        s1 = 2 * pivot - h
        s2 = pivot - (h - l)
        s3 = l - 2 * (h - pivot)
        
    elif method == "camarilla":
        pivot = (h + l + c) / 3
        range_hl = h - l
        r1 = c + range_hl * 1.1 / 12
        r2 = c + range_hl * 1.1 / 6
        r3 = c + range_hl * 1.1 / 4
        r4 = c + range_hl * 1.1 / 2
        s1 = c - range_hl * 1.1 / 12
        s2 = c - range_hl * 1.1 / 6
        s3 = c - range_hl * 1.1 / 4
        s4 = c - range_hl * 1.1 / 2
        
        return {
            "pivot": pivot,
            "r1": r1, "r2": r2, "r3": r3, "r4": r4,
            "s1": s1, "s2": s2, "s3": s3, "s4": s4
        }
    
    else:
        raise ValueError(f"Unknown pivot method: {method}")
    
    return {
        "pivot": pivot,
        "r1": r1, "r2": r2, "r3": r3,
        "s1": s1, "s2": s2, "s3": s3
    }
```

### 4.2 TA-Lib Integration (Optional)

**File**: `polarbtest/integrations/talib.py`

```python
"""
TA-Lib integration for PolarBtest.

Install with: pip install polarbtest[talib]
or: pip install TA-Lib
"""

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None

import polars as pl
import numpy as np
from typing import Dict, Any, List


def _check_talib():
    """Check if TA-Lib is available."""
    if not TALIB_AVAILABLE:
        raise ImportError(
            "TA-Lib not installed. Install with: pip install TA-Lib"
        )


def talib_indicator(
    df: pl.DataFrame,
    func_name: str,
    output_names: List[str] = None,
    **kwargs
) -> pl.DataFrame:
    """
    Apply TA-Lib indicator to DataFrame.
    
    Args:
        df: Polars DataFrame with OHLCV columns
        func_name: TA-Lib function name (e.g., "RSI", "MACD", "BBANDS")
        output_names: Names for output columns (auto-generated if None)
        **kwargs: Parameters for TA-Lib function
        
    Returns:
        DataFrame with indicator columns added
        
    Example:
        df = talib_indicator(df, "RSI", ["rsi"], timeperiod=14, price="close")
        df = talib_indicator(df, "MACD", ["macd", "signal", "hist"],
                            fastperiod=12, slowperiod=26, signalperiod=9)
    """
    _check_talib()
    
    # Get TA-Lib function
    func = getattr(talib, func_name.upper())
    
    # Convert to numpy for TA-Lib
    # Determine which columns to use
    input_arrays = []
    
    # Common input patterns
    if func_name.upper() in ["RSI", "SMA", "EMA", "WMA"]:
        # Single input
        price_col = kwargs.pop("price", "close")
        input_arrays = [df[price_col].to_numpy()]
        
    elif func_name.upper() in ["MACD", "STOCH", "BBANDS"]:
        # Standard indicators
        if "price" in kwargs:
            input_arrays = [df[kwargs.pop("price")].to_numpy()]
        else:
            input_arrays = [df["close"].to_numpy()]
            
    elif func_name.upper() in ["ATR", "ADX"]:
        # Need H, L, C
        input_arrays = [
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            df["close"].to_numpy()
        ]
    
    else:
        # Generic: try to infer from function
        # This is complex - user should specify
        raise ValueError(f"Please specify input columns for {func_name}")
    
    # Call TA-Lib function
    result = func(*input_arrays, **kwargs)
    
    # Handle single vs multiple outputs
    if not isinstance(result, tuple):
        result = (result,)
    
    # Generate output names if not provided
    if output_names is None:
        if len(result) == 1:
            output_names = [func_name.lower()]
        else:
            output_names = [f"{func_name.lower()}_{i}" for i in range(len(result))]
    
    # Add to DataFrame
    new_cols = {}
    for name, values in zip(output_names, result):
        new_cols[name] = pl.Series(name, values)
    
    return df.with_columns(new_cols.values())


# Convenience wrappers for common indicators
def talib_rsi(df: pl.DataFrame, period: int = 14, price: str = "close") -> pl.DataFrame:
    """Calculate RSI using TA-Lib."""
    return talib_indicator(df, "RSI", ["rsi"], timeperiod=period, price=price)


def talib_macd(
    df: pl.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    price: str = "close"
) -> pl.DataFrame:
    """Calculate MACD using TA-Lib."""
    return talib_indicator(
        df, "MACD",
        ["macd", "macd_signal", "macd_hist"],
        fastperiod=fast,
        slowperiod=slow,
        signalperiod=signal,
        price=price
    )


def talib_bbands(
    df: pl.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    price: str = "close"
) -> pl.DataFrame:
    """Calculate Bollinger Bands using TA-Lib."""
    return talib_indicator(
        df, "BBANDS",
        ["bb_upper", "bb_middle", "bb_lower"],
        timeperiod=period,
        nbdevup=std_dev,
        nbdevdn=std_dev,
        price=price
    )


# Export list of all available TA-Lib functions
def list_talib_indicators() -> Dict[str, List[str]]:
    """List all available TA-Lib indicators by group."""
    _check_talib()
    
    groups = {
        "Overlap Studies": talib.get_functions()[:30],  # Approximation
        "Momentum Indicators": [],
        "Volume Indicators": [],
        "Volatility Indicators": [],
        "Price Transform": [],
        "Cycle Indicators": [],
        "Pattern Recognition": [],
        "Statistic Functions": [],
    }
    
    # This would need proper categorization
    # For now, just return all functions
    return {"All Functions": talib.get_functions()}
```

### 4.3 Data Validation & Cleaning

**File**: `polarbtest/data/validation.py`

```python
"""Data validation and cleaning utilities."""

import polars as pl
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ValidationIssue:
    """Data validation issue."""
    severity: str  # "error", "warning", "info"
    category: str  # "missing", "invalid", "duplicate", etc.
    message: str
    rows: List[int] = None
    count: int = 0


def validate_ohlcv(
    df: pl.DataFrame,
    timestamp_col: str = "timestamp",
    require_volume: bool = False
) -> List[ValidationIssue]:
    """
    Validate OHLCV data for common issues.
    
    Args:
        df: DataFrame to validate
        timestamp_col: Name of timestamp column
        require_volume: Whether volume is required
        
    Returns:
        List of validation issues found
    """
    issues = []
    
    # Check required columns
    required = ["open", "high", "low", "close"]
    if require_volume:
        required.append("volume")
    
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        issues.append(ValidationIssue(
            severity="error",
            category="missing",
            message=f"Missing required columns: {missing_cols}"
        ))
        return issues  # Can't continue without required columns
    
    # Check for null values
    for col in required:
        null_count = df[col].null_count()
        if null_count > 0:
            issues.append(ValidationIssue(
                severity="warning",
                category="missing",
                message=f"Column '{col}' has {null_count} null values",
                count=null_count
            ))
    
    # Check OHLC relationships
    # High should be >= Low
    invalid_hl = df.filter(pl.col("high") < pl.col("low"))
    if len(invalid_hl) > 0:
        issues.append(ValidationIssue(
            severity="error",
            category="invalid",
            message=f"Found {len(invalid_hl)} bars where High < Low",
            rows=invalid_hl.select(pl.arange(0, pl.count()).alias("idx"))["idx"].to_list()[:10],
            count=len(invalid_hl)
        ))
    
    # Close should be between Low and High
    invalid_close = df.filter(
        (pl.col("close") > pl.col("high")) | (pl.col("close") < pl.col("low"))
    )
    if len(invalid_close) > 0:
        issues.append(ValidationIssue(
            severity="error",
            category="invalid",
            message=f"Found {len(invalid_close)} bars where Close is outside High/Low range",
            count=len(invalid_close)
        ))
    
    # Open should be between Low and High
    invalid_open = df.filter(
        (pl.col("open") > pl.col("high")) | (pl.col("open") < pl.col("low"))
    )
    if len(invalid_open) > 0:
        issues.append(ValidationIssue(
            severity="warning",
            category="invalid",
            message=f"Found {len(invalid_open)} bars where Open is outside High/Low range",
            count=len(invalid_open)
        ))
    
    # Check for negative prices
    for col in ["open", "high", "low", "close"]:
        negative = df.filter(pl.col(col) <= 0)
        if len(negative) > 0:
            issues.append(ValidationIssue(
                severity="error",
                category="invalid",
                message=f"Found {len(negative)} bars with non-positive {col}",
                count=len(negative)
            ))
    
    # Check for negative volume
    if "volume" in df.columns:
        negative_vol = df.filter(pl.col("volume") < 0)
        if len(negative_vol) > 0:
            issues.append(ValidationIssue(
                severity="error",
                category="invalid",
                message=f"Found {len(negative_vol)} bars with negative volume",
                count=len(negative_vol)
            ))
    
    # Check timestamp column if exists
    if timestamp_col in df.columns:
        # Check for duplicates
        duplicates = df.filter(pl.col(timestamp_col).is_duplicated())
        if len(duplicates) > 0:
            issues.append(ValidationIssue(
                severity="warning",
                category="duplicate",
                message=f"Found {len(duplicates)} duplicate timestamps",
                count=len(duplicates)
            ))
        
        # Check if sorted
        is_sorted = df[timestamp_col].is_sorted()
        if not is_sorted:
            issues.append(ValidationIssue(
                severity="warning",
                category="order",
                message="Timestamps are not sorted in ascending order"
            ))
        
        # Check for gaps (if timestamps are datetime)
        # This is complex and depends on expected frequency
        # Skip for now
    
    return issues


def clean_ohlcv(
    df: pl.DataFrame,
    fill_method: str = "forward",
    drop_invalid: bool = False,
    remove_duplicates: bool = False,
    sort_by_timestamp: bool = True,
    timestamp_col: str = "timestamp"
) -> pl.DataFrame:
    """
    Clean OHLCV data.
    
    Args:
        df: DataFrame to clean
        fill_method: "forward", "backward", "interpolate", or None
        drop_invalid: Drop rows with invalid OHLC relationships
        remove_duplicates: Remove duplicate timestamps
        sort_by_timestamp: Sort by timestamp
        timestamp_col: Name of timestamp column
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.clone()
    
    # Sort by timestamp
    if sort_by_timestamp and timestamp_col in df_clean.columns:
        df_clean = df_clean.sort(timestamp_col)
    
    # Remove duplicates
    if remove_duplicates and timestamp_col in df_clean.columns:
        df_clean = df_clean.unique(subset=[timestamp_col], keep="first")
    
    # Drop invalid OHLC bars
    if drop_invalid:
        # High >= Low
        df_clean = df_clean.filter(pl.col("high") >= pl.col("low"))
        
        # Close between High and Low
        df_clean = df_clean.filter(
            (pl.col("close") <= pl.col("high")) & (pl.col("close") >= pl.col("low"))
        )
        
        # Positive prices
        for col in ["open", "high", "low", "close"]:
            df_clean = df_clean.filter(pl.col(col) > 0)
    
    # Fill missing values
    if fill_method == "forward":
        df_clean = df_clean.fill_null(strategy="forward")
    elif fill_method == "backward":
        df_clean = df_clean.fill_null(strategy="backward")
    elif fill_method == "interpolate":
        # Polars doesn't have built-in interpolation
        # Would need custom implementation
        pass
    
    return df_clean


def detect_gaps(
    df: pl.DataFrame,
    timestamp_col: str = "timestamp",
    expected_freq: str = "1h"
) -> pl.DataFrame:
    """
    Detect gaps in time series data.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        expected_freq: Expected frequency (e.g., "1h", "1d", "5m")
        
    Returns:
        DataFrame with gaps (start_time, end_time, gap_duration)
    """
    # Calculate time differences
    df_gaps = df.with_columns([
        pl.col(timestamp_col).diff().alias("time_diff")
    ])
    
    # This needs proper frequency detection and comparison
    # Simplified version: just return time diffs
    return df_gaps.filter(pl.col("time_diff").is_not_null())
```

### 4.4 Resampling Utilities

**File**: `polarbtest/data/resampling.py`

```python
"""Data resampling utilities."""

import polars as pl
from typing import Dict, Any


# Standard OHLCV aggregation rules
OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_ohlcv(
    df: pl.DataFrame,
    timeframe: str,
    timestamp_col: str = "timestamp",
    agg_dict: Dict[str, str] = None
) -> pl.DataFrame:
    """
    Resample OHLCV data to different timeframe.
    
    Args:
        df: DataFrame with OHLCV data
        timeframe: Target timeframe (Polars duration string)
                   Examples: "5m", "1h", "1d", "1w"
        timestamp_col: Name of timestamp column
        agg_dict: Custom aggregation rules (default: OHLCV_AGG)
        
    Returns:
        Resampled DataFrame
        
    Example:
        # Resample 1-minute to 5-minute
        df_5m = resample_ohlcv(df_1m, "5m")
        
        # Resample to daily
        df_daily = resample_ohlcv(df_hourly, "1d")
    """
    if agg_dict is None:
        agg_dict = OHLCV_AGG.copy()
    
    # Build aggregation expressions
    agg_exprs = []
    for col, method in agg_dict.items():
        if col not in df.columns:
            continue
            
        if method == "first":
            agg_exprs.append(pl.col(col).first().alias(col))
        elif method == "last":
            agg_exprs.append(pl.col(col).last().alias(col))
        elif method == "max":
            agg_exprs.append(pl.col(col).max().alias(col))
        elif method == "min":
            agg_exprs.append(pl.col(col).min().alias(col))
        elif method == "sum":
            agg_exprs.append(pl.col(col).sum().alias(col))
        elif method == "mean":
            agg_exprs.append(pl.col(col).mean().alias(col))
    
    # Resample
    df_resampled = (
        df.sort(timestamp_col)
        .group_by_dynamic(timestamp_col, every=timeframe)
        .agg(agg_exprs)
    )
    
    return df_resampled
```

---

## Phase 5: Optimization Enhancements

**Timeline**: 1-2 weeks  
**Priority**: 🟡 MEDIUM  
**Dependencies**: Phase 2 (for visualization)

### 5.1 Constraint Functions

**Add to `runner.py`**:

```python
def optimize(
    strategy_class: Type[Strategy],
    data: Union[pl.DataFrame, Dict[str, pl.DataFrame]],
    param_grid: Dict[str, List[Any]],
    objective: str = "sharpe_ratio",
    maximize: bool = True,
    constraints: Optional[Callable[[Dict], bool]] = None,  # NEW
    initial_cash: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    price_columns: Optional[Dict[str, str]] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Grid search optimization with constraints.
    
    Args:
        constraints: Function that takes params dict and returns True if valid
                    Example: lambda p: p["fast"] < p["slow"]
    """
    # Generate all parameter combinations
    import itertools
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    # Apply constraints
    if constraints:
        param_sets = [p for p in all_combos if constraints(p)]
        if verbose:
            filtered = len(all_combos) - len(param_sets)
            print(f"Filtered {filtered} invalid parameter combinations")
    else:
        param_sets = all_combos
    
    # ... rest of existing optimize code ...
```

**Usage Example**:

```python
param_grid = {
    "fast_period": [5, 10, 15, 20],
    "slow_period": [20, 50, 100],
    "rsi_threshold": [30, 40, 50]
}

# Ensure fast < slow
best = optimize(
    MyStrategy,
    data,
    param_grid,
    constraints=lambda p: p["fast_period"] < p["slow_period"]
)
```

### 5.2 Multi-Objective Optimization

```python
def optimize_multi_objective(
    strategy_class: Type[Strategy],
    data: Union[pl.DataFrame, Dict[str, pl.DataFrame]],
    param_grid: Dict[str, List[Any]],
    objectives: Dict[str, str],  # {"sharpe_ratio": "maximize", "max_drawdown": "minimize"}
    weights: Optional[Dict[str, float]] = None,  # Importance weights
    **kwargs
) -> Dict[str, Any]:
    """
    Multi-objective optimization.
    
    Optimizes multiple metrics simultaneously using weighted sum.
    
    Args:
        objectives: Dict mapping metric names to "maximize" or "minimize"
        weights: Optional weights for each objective (default: equal weight)
        
    Example:
        best = optimize_multi_objective(
            MyStrategy,
            data,
            param_grid,
            objectives={
                "sharpe_ratio": "maximize",
                "max_drawdown": "minimize",
                "total_return": "maximize"
            },
            weights={
                "sharpe_ratio": 0.5,
                "max_drawdown": 0.3,
                "total_return": 0.2
            }
        )
    """
    # Run batch backtest
    results_df = backtest_batch(
        strategy_class=strategy_class,
        data=data,
        param_sets=[dict(zip(param_grid.keys(), combo)) 
                   for combo in itertools.product(*param_grid.values())],
        **kwargs
    )
    
    # Normalize metrics to [0, 1]
    normalized_df = results_df.clone()
    
    for metric, direction in objectives.items():
        if metric not in results_df.columns:
            raise ValueError(f"Metric '{metric}' not in results")
        
        col = results_df[metric]
        min_val = col.min()
        max_val = col.max()
        
        if max_val != min_val:
            # Normalize to [0, 1]
            normalized = (col - min_val) / (max_val - min_val)
            
            # Invert if minimizing
            if direction == "minimize":
                normalized = 1 - normalized
            
            normalized_df = normalized_df.with_columns([
                normalized.alias(f"norm_{metric}")
            ])
    
    # Calculate weighted score
    if weights is None:
        weights = {metric: 1.0 / len(objectives) for metric in objectives.keys()}
    
    # Ensure weights sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Compute composite score
    score_expr = sum(
        pl.col(f"norm_{metric}") * weight
        for metric, weight in weights.items()
    )
    
    normalized_df = normalized_df.with_columns([
        score_expr.alias("composite_score")
    ])
    
    # Get best
    best_row = normalized_df.sort("composite_score", descending=True).head(1)
    
    return best_row.to_dicts()[0]
```

### 5.3 Bayesian Optimization (Optional)

```python
"""
Bayesian optimization using scikit-optimize.

Install with: pip install polarbtest[optimization]
or: pip install scikit-optimize
"""

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


def bayesian_optimize(
    strategy_class: Type[Strategy],
    data: Union[pl.DataFrame, Dict[str, pl.DataFrame]],
    param_space: Dict[str, tuple],  # {param: (min, max, type)}
    objective: str = "sharpe_ratio",
    maximize: bool = True,
    n_calls: int = 100,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Bayesian optimization for parameter search.
    
    More efficient than grid search for large parameter spaces.
    
    Args:
        param_space: Dict mapping parameter names to (min, max, type) tuples
                    Example: {"fast": (5, 50, "int"), "slow": (20, 200, "int")}
        n_calls: Number of evaluations
        
    Example:
        best = bayesian_optimize(
            MyStrategy,
            data,
            param_space={
                "fast_period": (5, 50, "int"),
                "slow_period": (20, 200, "int"),
                "threshold": (0.01, 0.10, "float")
            },
            n_calls=100
        )
    """
    if not SKOPT_AVAILABLE:
        raise ImportError(
            "Bayesian optimization requires scikit-optimize. "
            "Install with: pip install scikit-optimize"
        )
    
    # Build search space
    dimensions = []
    param_names = []
    
    for name, (min_val, max_val, param_type) in param_space.items():
        param_names.append(name)
        
        if param_type == "int":
            dimensions.append(Integer(min_val, max_val, name=name))
        elif param_type == "float":
            dimensions.append(Real(min_val, max_val, name=name))
        elif param_type == "categorical":
            # min_val is actually a list of categories
            dimensions.append(Categorical(min_val, name=name))
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    # Objective function
    @use_named_args(dimensions)
    def objective_func(**params):
        # Run backtest
        results = backtest(
            strategy_class=strategy_class,
            data=data,
            params=params,
            **kwargs
        )
        
        # Extract objective value
        value = results.get(objective, 0.0)
        
        # Minimize (negate if maximizing)
        return -value if maximize else value
    
    # Run optimization
    result = gp_minimize(
        objective_func,
        dimensions,
        n_calls=n_calls,
        random_state=random_state
    )
    
    # Extract best parameters
    best_params = dict(zip(param_names, result.x))
    
    # Run final backtest with best params
    best_results = backtest(
        strategy_class=strategy_class,
        data=data,
        params=best_params,
        **kwargs
    )
    
    return best_results
```

### 5.4 Optimization Visualization

**File**: `polarbtest/plotting/optimization.py`

```python
"""Optimization result visualization."""

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def plot_optimization_heatmap(
    results_df: pl.DataFrame,
    x_param: str,
    y_param: str,
    z_metric: str = "sharpe_ratio",
    title: str = None,
    colorscale: str = "RdYlGn",
    show: bool = True,
    save_path: str = None
) -> go.Figure:
    """
    Plot 2D heatmap of optimization results.
    
    Args:
        results_df: Results from backtest_batch
        x_param: Parameter for x-axis
        y_param: Parameter for y-axis
        z_metric: Metric to visualize
        
    Example:
        results = backtest_batch(MyStrategy, data, param_sets)
        plot_optimization_heatmap(
            results,
            x_param="fast_period",
            y_param="slow_period",
            z_metric="sharpe_ratio"
        )
    """
    # Pivot data for heatmap
    pivot_df = results_df.pivot(
        values=z_metric,
        index=y_param,
        columns=x_param
    )
    
    # Convert to numpy
    x_vals = sorted(results_df[x_param].unique())
    y_vals = sorted(results_df[y_param].unique())
    z_vals = pivot_df.select(pl.exclude(y_param)).to_numpy()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        colorscale=colorscale,
        colorbar=dict(title=z_metric)
    ))
    
    fig.update_layout(
        title=title or f"{z_metric} by {x_param} and {y_param}",
        xaxis_title=x_param,
        yaxis_title=y_param,
        width=800,
        height=600
    )
    
    if save_path:
        fig.write_html(save_path)
    
    if show:
        fig.show()
    
    return fig


def plot_parameter_sensitivity(
    results_df: pl.DataFrame,
    param: str,
    metrics: List[str] = ["sharpe_ratio", "total_return", "max_drawdown"],
    title: str = None,
    show: bool = True,
    save_path: str = None
) -> go.Figure:
    """
    Plot how metrics change with single parameter.
    
    Args:
        results_df: Results from backtest_batch
        param: Parameter to analyze
        metrics: Metrics to plot
    """
    # Group by parameter and average metrics
    grouped = results_df.group_by(param).agg([
        pl.col(m).mean().alias(f"{m}_mean") for m in metrics
    ] + [
        pl.col(m).std().alias(f"{m}_std") for m in metrics
    ]).sort(param)
    
    # Create subplots
    fig = make_subplots(
        rows=len(metrics), cols=1,
        subplot_titles=metrics,
        shared_xaxes=True
    )
    
    for i, metric in enumerate(metrics, 1):
        # Mean line
        fig.add_trace(
            go.Scatter(
                x=grouped[param],
                y=grouped[f"{metric}_mean"],
                mode="lines+markers",
                name=metric,
                error_y=dict(
                    type="data",
                    array=grouped[f"{metric}_std"],
                    visible=True
                )
            ),
            row=i, col=1
        )
    
    fig.update_layout(
        title=title or f"Parameter Sensitivity: {param}",
        height=300 * len(metrics),
        showlegend=False
    )
    
    fig.update_xaxes(title_text=param, row=len(metrics), col=1)
    
    if save_path:
        fig.write_html(save_path)
    
    if show:
        fig.show()
    
    return fig


def plot_optimization_surface_3d(
    results_df: pl.DataFrame,
    x_param: str,
    y_param: str,
    z_metric: str = "sharpe_ratio",
    title: str = None,
    show: bool = True,
    save_path: str = None
) -> go.Figure:
    """
    Plot 3D surface of optimization results.
    
    Useful for visualizing the optimization landscape.
    """
    # Pivot data
    pivot_df = results_df.pivot(
        values=z_metric,
        index=y_param,
        columns=x_param
    )
    
    x_vals = sorted(results_df[x_param].unique())
    y_vals = sorted(results_df[y_param].unique())
    z_vals = pivot_df.select(pl.exclude(y_param)).to_numpy()
    
    # Create 3D surface
    fig = go.Figure(data=[go.Surface(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        colorscale="Viridis",
        colorbar=dict(title=z_metric)
    )])
    
    fig.update_layout(
        title=title or f"{z_metric} Optimization Surface",
        scene=dict(
            xaxis_title=x_param,
            yaxis_title=y_param,
            zaxis_title=z_metric
        ),
        width=900,
        height=700
    )
    
    if save_path:
        fig.write_html(save_path)
    
    if show:
        fig.show()
    
    return fig
```

---

## Phase 6: Documentation & Examples

**Timeline**: 1-2 weeks  
**Priority**: 🔴 HIGH  
**Dependencies**: All previous phases

### 6.1 Documentation Structure

```
docs/
├── README.md                 # Main documentation
├── getting-started.md        # Quick start guide
├── user-guide/
│   ├── installation.md
│   ├── basic-concepts.md
│   ├── writing-strategies.md
│   ├── indicators.md
│   ├── order-types.md
│   ├── risk-management.md
│   ├── optimization.md
│   ├── visualization.md
│   └── advanced-topics.md
├── api-reference/
│   ├── core.md              # Portfolio, Strategy, Engine
│   ├── indicators.md        # All indicators
│   ├── metrics.md           # Performance metrics
│   ├── runner.md            # Optimization functions
│   ├── plotting.md          # Visualization
│   └── sizers.md            # Position sizing
├── examples/
│   ├── 01_simple_sma.md
│   ├── 02_rsi_mean_reversion.md
│   ├── 03_multi_asset.md
│   ├── 04_ml_integration.md
│   ├── 05_risk_management.md
│   └── 06_optimization.md
├── tutorials/
│   ├── beginner/
│   ├── intermediate/
│   └── advanced/
└── best-practices.md
```

### 6.2 Example Strategies Library

**File**: `examples/strategies.py`

```python
"""
Collection of example trading strategies.

These demonstrate various features and best practices.
"""

from polarbtest import Strategy, indicators as ind, BacktestContext


class SimpleSMACross(Strategy):
    """
    Basic SMA crossover strategy.
    
    Buy when fast SMA crosses above slow SMA.
    Sell when fast SMA crosses below slow SMA.
    """
    
    def preprocess(self, df):
        fast = self.params.get("fast_period", 10)
        slow = self.params.get("slow_period", 30)
        
        return df.with_columns([
            ind.sma("close", fast).alias("sma_fast"),
            ind.sma("close", slow).alias("sma_slow"),
            ind.crossover("sma_fast", "sma_slow").alias("golden_cross"),
            ind.crossunder("sma_fast", "sma_slow").alias("death_cross"),
        ])
    
    def next(self, ctx):
        if ctx.row.get("sma_fast") is None:
            return
        
        if ctx.row["golden_cross"]:
            ctx.portfolio.order_target_percent("asset", 1.0)
        elif ctx.row["death_cross"]:
            ctx.portfolio.close_position("asset")


class RSIMeanReversion(Strategy):
    """
    RSI mean reversion strategy.
    
    Buy when RSI < oversold threshold.
    Sell when RSI > overbought threshold.
    """
    
    def preprocess(self, df):
        rsi_period = self.params.get("rsi_period", 14)
        
        return df.with_columns([
            ind.rsi("close", rsi_period).alias("rsi")
        ])
    
    def next(self, ctx):
        if ctx.row.get("rsi") is None:
            return
        
        oversold = self.params.get("oversold", 30)
        overbought = self.params.get("overbought", 70)
        
        if ctx.row["rsi"] < oversold:
            ctx.portfolio.order_target_percent("asset", 1.0)
        elif ctx.row["rsi"] > overbought:
            ctx.portfolio.close_position("asset")


class BollingerBandBreakout(Strategy):
    """
    Bollinger Band breakout strategy.
    
    Buy when price breaks above upper band.
    Sell when price breaks below lower band.
    """
    
    def preprocess(self, df):
        period = self.params.get("bb_period", 20)
        std_dev = self.params.get("bb_std", 2.0)
        
        upper, middle, lower = ind.bollinger_bands("close", period, std_dev)
        
        return df.with_columns([
            upper.alias("bb_upper"),
            middle.alias("bb_middle"),
            lower.alias("bb_lower"),
        ])
    
    def next(self, ctx):
        if ctx.row.get("bb_upper") is None:
            return
        
        price = ctx.row["close"]
        
        # Breakout above upper band
        if price > ctx.row["bb_upper"]:
            ctx.portfolio.order_target_percent("asset", 1.0)
        
        # Breakdown below lower band
        elif price < ctx.row["bb_lower"]:
            ctx.portfolio.close_position("asset")


class TrendFollowingWithATR(Strategy):
    """
    Trend following with ATR-based stop loss.
    
    Demonstrates risk management with position sizing.
    """
    
    def __init__(self, **params):
        super().__init__(**params)
        from polarbtest.sizers import FixedRiskSizer
        self.sizer = FixedRiskSizer(risk_pct=0.02)
    
    def preprocess(self, df):
        return df.with_columns([
            ind.ema("close", 50).alias("ema_50"),
            ind.atr("high", "low", "close", 14).alias("atr"),
        ])
    
    def next(self, ctx):
        if ctx.bar_index < 50:
            return
        
        price = ctx.row["close"]
        ema = ctx.row["ema_50"]
        atr = ctx.row.get("atr", 0)
        
        # Uptrend
        if price > ema:
            # Calculate stop loss
            stop_loss = price - 2 * atr
            
            # Size position based on risk
            ctx.portfolio.order_with_sizer(
                "asset",
                self.sizer,
                signal_strength=1.0,
                price=price,
                stop_loss=stop_loss
            )
            
            # Set stop loss
            ctx.portfolio.set_stop_loss("asset", stop_price=stop_loss)
            
            # Set take profit at 3x ATR
            take_profit = price + 3 * atr
            ctx.portfolio.set_take_profit("asset", target_price=take_profit)
        
        # Downtrend
        elif price < ema:
            ctx.portfolio.close_position("asset")


class MultiAssetMomentum(Strategy):
    """
    Multi-asset momentum rotation.
    
    Allocate to top N assets by momentum.
    """
    
    def preprocess(self, df):
        lookback = self.params.get("lookback", 20)
        
        # Assuming multi-asset data with columns: btc_close, eth_close, sol_close
        return df.with_columns([
            ind.returns("btc_close", lookback).alias("btc_momentum"),
            ind.returns("eth_close", lookback).alias("eth_momentum"),
            ind.returns("sol_close", lookback).alias("sol_momentum"),
        ])
    
    def next(self, ctx):
        if ctx.bar_index < self.params.get("lookback", 20):
            return
        
        # Get momentum for each asset
        assets = [
            ("BTC", ctx.row.get("btc_momentum", 0)),
            ("ETH", ctx.row.get("eth_momentum", 0)),
            ("SOL", ctx.row.get("sol_momentum", 0)),
        ]
        
        # Sort by momentum
        assets.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate to top N
        top_n = self.params.get("top_n", 2)
        weight = 1.0 / top_n
        
        for i, (asset, momentum) in enumerate(assets):
            if i < top_n:
                ctx.portfolio.order_target_percent(asset, weight)
            else:
                ctx.portfolio.close_position(asset)


# Export all strategies
__all__ = [
    "SimpleSMACross",
    "RSIMeanReversion",
    "BollingerBandBreakout",
    "TrendFollowingWithATR",
    "MultiAssetMomentum",
]
```

### 6.3 Jupyter Notebook Tutorials

Create notebooks for:
1. **Getting Started** - Basic walkthrough
2. **Custom Indicators** - Creating custom indicators
3. **Optimization Tutorial** - Parameter optimization
4. **ML Integration** - Using scikit-learn
5. **Multi-Asset Strategies** - Portfolio management

---

## Phase 7: Advanced Features (Optional)

**Timeline**: Ongoing  
**Priority**: 🟢 LOW  
**Dependencies**: Varies

### 7.1 Monte Carlo Simulation

```python
def monte_carlo_simulation(
    strategy_class: Type[Strategy],
    data: pl.DataFrame,
    params: Dict[str, Any],
    n_simulations: int = 1000,
    randomize: str = "order_fills",  # or "entry_timing", "data_order"
    seed: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation on strategy.
    
    Args:
        randomize: What to randomize
            - "order_fills": Randomize fill prices within bid-ask spread
            - "entry_timing": Randomly shift entry timing by ±1 bar
            - "data_order": Randomly permute data order (for robustness)
    """
    np.random.seed(seed)
    results = []
    
    for i in range(n_simulations):
        # Randomize data based on method
        if randomize == "data_order":
            # This doesn't make sense for time series
            # Skip for now
            sim_data = data.sample(fraction=1.0, seed=i)
        elif randomize == "entry_timing":
            # Add random delay to entries (complex)
            sim_data = data
        else:
            sim_data = data
        
        # Run backtest with modified slippage/randomness
        result = backtest(
            strategy_class,
            sim_data,
            params,
            slippage=np.random.uniform(0, 0.001),  # Random slippage
            **kwargs
        )
        
        results.append(result)
    
    # Aggregate results
    return {
        "simulations": results,
        "mean_return": np.mean([r["total_return"] for r in results]),
        "std_return": np.std([r["total_return"] for r in results]),
        "min_return": np.min([r["total_return"] for r in results]),
        "max_return": np.max([r["total_return"] for r in results]),
        "confidence_95": np.percentile([r["total_return"] for r in results], [2.5, 97.5]),
    }
```

### 7.2 Look-Ahead Bias Detection

```python
def detect_lookahead_bias(
    strategy_class: Type[Strategy],
    data: pl.DataFrame
) -> List[str]:
    """
    Attempt to detect look-ahead bias in strategy.
    
    This is heuristic-based and won't catch everything.
    """
    warnings = []
    
    # Check if strategy accesses future data in preprocess
    # This would require code analysis - complex
    
    # Check if indicators use future data
    # ...
    
    return warnings
```

---

## Implementation Guidelines

### Code Organization

```
polarbtest/
├── __init__.py              # Core exports
├── core.py                  # Portfolio, Strategy, Engine (KEEP MINIMAL)
├── indicators.py            # Technical indicators
├── metrics.py               # Performance metrics
├── runner.py                # Optimization functions
├── orders.py                # Order types and management (NEW)
├── trades.py                # Trade tracking (NEW)
├── sizers.py                # Position sizing (NEW)
├── plotting/                # Visualization (NEW)
│   ├── __init__.py
│   ├── charts.py
│   ├── equity.py
│   ├── trades.py
│   └── optimization.py
├── data/                    # Data utilities (NEW)
│   ├── __init__.py
│   ├── validation.py
│   ├── resampling.py
│   └── loaders.py
└── integrations/            # Optional integrations (NEW)
    ├── __init__.py
    └── talib.py
```

### Testing Strategy

- Unit tests for each module
- Integration tests for full workflows
- Example-based tests (run all examples)
- Performance benchmarks

### Backward Compatibility

- Keep existing API working
- Mark deprecated features
- Provide migration guide

---

## Dependencies Strategy

**Core Dependencies** (required):
```toml
[project]
dependencies = [
    "polars>=0.19.0",
    "numpy>=1.20.0",
]
```

**Optional Dependencies**:
```toml
[project.optional-dependencies]
plotting = [
    "plotly>=5.0.0",
]
optimization = [
    "scikit-optimize>=0.9.0",
]
talib = [
    "TA-Lib>=0.4.0",
]
data = [
    "yfinance>=0.2.0",  # For data loading
]
reports = [
    "jinja2>=3.0.0",    # For HTML reports
]
all = [
    "plotly>=5.0.0",
    "scikit-optimize>=0.9.0",
    "TA-Lib>=0.4.0",
    "yfinance>=0.2.0",
    "jinja2>=3.0.0",
]
```

**Installation**:
```bash
# Core only
pip install polarbtest

# With plotting
pip install polarbtest[plotting]

# Everything
pip install polarbtest[all]
```

---

## Summary

This roadmap provides a comprehensive enhancement plan for PolarBtest that:

1. **Maintains Core Philosophy**: Keeps the engine minimal and fast
2. **Adds Professional Features**: Order types, trade tracking, risk management
3. **Improves Usability**: Visualization, better metrics, examples
4. **Enables Research**: Advanced analytics, optimization tools
5. **Stays Extensible**: Optional features, clear APIs, plugin architecture

The phased approach allows incremental development while maintaining a working library at each stage.

**Estimated Total Timeline**: 10-14 weeks for Phases 1-6, with Phase 7 as ongoing enhancements.

**Priority Recommendation**: Focus on Phases 1-3 first (core trading functionality, visualization, risk management) as these provide the most value for professional trading and research use cases.
