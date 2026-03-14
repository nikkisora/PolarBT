"""
Core backtesting engine and components.

This module provides the fundamental building blocks for backtesting:
- Portfolio: Manages positions and cash
- Strategy: Base class for defining trading strategies
- Engine: Executes the backtest simulation
- BacktestContext: Data container passed to strategy.next()
"""

import gc
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import polars as pl

from polarbt.commissions import CommissionModel, make_commission_model
from polarbt.orders import Order, OrderStatus, OrderType
from polarbt.results import BacktestMetrics, TradeStats, _backtest_metrics_from_dict
from polarbt.slippage import SlippageModel, make_slippage_model
from polarbt.trades import TradeTracker
from polarbt.universe import UniverseContext, UniverseProvider

DEFAULT_ASSET_NAME = "asset"


def _extract_date(timestamp: Any) -> date | None:
    """
    Extract date from various timestamp types.

    Supported formats:
    - datetime objects (datetime.datetime or datetime.date)
    - Unix timestamps (int or float, seconds or milliseconds)
    - String formats: "yyyy-mm-dd hh:mm:ss", "yyyy-mm-dd", ISO format with 'T'

    Args:
        timestamp: Timestamp in various formats

    Returns:
        date object or None if extraction fails
    """
    if timestamp is None:
        return None

    # Python datetime
    if isinstance(timestamp, datetime):
        return timestamp.date()

    # Already a date
    if isinstance(timestamp, date):
        return timestamp

    # Unix timestamp (int or float)
    if isinstance(timestamp, (int, float)):
        # Handle both seconds and milliseconds
        ts_value = timestamp
        if ts_value > 1e10:  # Likely milliseconds
            ts_value = ts_value / 1000
        try:
            return datetime.fromtimestamp(ts_value).date()
        except (ValueError, OSError):
            return None

    # String parsing - try common formats
    if isinstance(timestamp, str):
        # Remove common separators and extract date part
        # "yyyy-mm-dd hh:mm:ss" -> "yyyy-mm-dd"
        # "yyyy-mm-ddThh:mm:ss" -> "yyyy-mm-dd"
        # "yyyy-mm-dd" -> "yyyy-mm-dd"
        try:
            # Split by space or 'T' to get date part
            if " " in timestamp:
                date_part = timestamp.split(" ")[0]
            elif "T" in timestamp:
                date_part = timestamp.split("T")[0]
            else:
                date_part = timestamp

            # Parse yyyy-mm-dd
            parts = date_part.split("-")
            if len(parts) == 3:
                return date(int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, IndexError):
            pass

    # Try converting to string and parsing (for other types like Polars datetime)
    try:
        dt_str = str(timestamp)
        if " " in dt_str:
            date_part = dt_str.split(" ")[0]
        elif "T" in dt_str:
            date_part = dt_str.split("T")[0]
        else:
            date_part = dt_str

        parts = date_part.split("-")
        if len(parts) == 3:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, AttributeError, IndexError):
        pass

    return None


_OHLCV_ALIASES: dict[str, list[str]] = {
    "open": ["Open", "OPEN", "open_price", "Open_Price"],
    "high": ["High", "HIGH", "high_price", "High_Price"],
    "low": ["Low", "LOW", "low_price", "Low_Price"],
    "close": ["Close", "CLOSE", "close_price", "Close_Price", "adj_close", "Adj_Close", "Adj Close"],
    "volume": ["Volume", "VOLUME", "vol", "Vol"],
    "factor": ["Factor", "FACTOR", "adj_factor", "Adj_Factor", "split_factor"],
}


def standardize_dataframe(
    df: pl.DataFrame,
    timestamp_col: str | None = None,
    auto_detect: bool = True,
) -> pl.DataFrame:
    """Standardize a DataFrame by renaming common timestamp and OHLCV column names.

    Detects common column name variants (e.g. ``"Date"`` -> ``"timestamp"``,
    ``"Open"`` -> ``"open"``) and renames them to the canonical lowercase names
    expected by the engine.

    Args:
        df: Input DataFrame.
        timestamp_col: Specific timestamp column to rename (if None, auto-detect).
        auto_detect: Auto-detect common column names (default True).

    Returns:
        DataFrame with standardized column names.

    Example:
        # Auto-detect and rename
        df = standardize_dataframe(df)

        # Specify timestamp column explicitly
        df = standardize_dataframe(df, timestamp_col="datetime")
    """
    renames: dict[str, str] = {}

    # --- Timestamp ---
    if "timestamp" not in df.columns:
        if timestamp_col and timestamp_col in df.columns:
            renames[timestamp_col] = "timestamp"
        elif auto_detect:
            common_names = ["date", "datetime", "time", "dt", "Date", "DateTime", "Time"]
            for name in common_names:
                if name in df.columns:
                    renames[name] = "timestamp"
                    break

    # --- OHLCV ---
    if auto_detect:
        for canonical, aliases in _OHLCV_ALIASES.items():
            if canonical in df.columns:
                continue
            for alias in aliases:
                if alias in df.columns and alias not in renames:
                    renames[alias] = canonical
                    break

    if renames:
        df = df.rename(renames)

    return df


def merge_asset_dataframes(
    data_dict: dict[str, pl.DataFrame],
    price_column: str = "close",
) -> tuple[pl.DataFrame, dict[str, str]]:
    """
    Merge multiple asset dataframes into a single wide-format dataframe.

    Args:
        data_dict: Dictionary mapping asset names to their dataframes
        price_column: Name of the price column in each dataframe (default "close")

    Returns:
        Tuple of (merged_dataframe, price_columns_mapping)

    Example:
        btc_df = pl.DataFrame({"timestamp": [...], "close": [...]})
        eth_df = pl.DataFrame({"timestamp": [...], "close": [...]})

        merged_df, price_cols = merge_asset_dataframes({
            "BTC": btc_df,
            "ETH": eth_df
        })
        # merged_df has columns: timestamp, BTC_close, ETH_close
        # price_cols = {"BTC": "BTC_close", "ETH": "ETH_close"}
    """
    if not data_dict:
        raise ValueError("data_dict cannot be empty")

    # Standardize all dataframes
    standardized = {asset: standardize_dataframe(df) for asset, df in data_dict.items()}

    # Start with the first dataframe's timestamp
    first_asset = list(standardized.keys())[0]
    merged = (
        standardized[first_asset].select(["timestamp"]) if "timestamp" in standardized[first_asset].columns else None
    )

    # Build price columns mapping
    price_columns = {}

    # Merge all dataframes
    for asset, df in standardized.items():
        # Rename price column to include asset name
        new_col_name = f"{asset}_{price_column}"
        price_columns[asset] = new_col_name

        # Select timestamp and price columns
        if "timestamp" in df.columns:
            df_subset = df.select(["timestamp", price_column]).rename({price_column: new_col_name})

            merged = df_subset if merged is None else merged.join(df_subset, on="timestamp", how="full", coalesce=True)
        else:
            # No timestamp - add index and merge
            df_subset = df.select([price_column]).rename({price_column: new_col_name})
            merged = df_subset if merged is None else pl.concat([merged, df_subset], how="horizontal")

    if merged is not None and "timestamp" in merged.columns:
        merged = merged.sort("timestamp")

    if merged is None:
        raise ValueError("Failed to merge dataframes")

    return merged, price_columns


class _RowAccessor:
    """Provides dual access to bar data: ``ctx.row["close"]`` (property) and ``ctx.row("BTC")`` (method).

    In single-asset mode ``ctx.row`` behaves like a plain dict (backward-compatible).
    In multi-asset mode ``ctx.row("BTC")`` returns the dict for a specific symbol,
    while ``ctx.row`` (no call) returns the single symbol's dict if there is exactly one.
    """

    __slots__ = ("_data", "_symbols")

    def __init__(self, data: dict[str, dict[str, Any]], symbols: list[str]) -> None:
        self._data = data
        self._symbols = symbols

    # --- dict-like access (backward compat for single-asset ``ctx.row["close"]``) ---

    def __getitem__(self, key: str) -> Any:
        if len(self._symbols) == 1:
            return self._data[self._symbols[0]][key]
        raise KeyError(
            f"Ambiguous row access with {len(self._symbols)} symbols. Use ctx.row('SYMBOL')['{key}'] instead."
        )

    def __contains__(self, key: object) -> bool:
        if len(self._symbols) == 1:
            return key in self._data[self._symbols[0]]
        return False

    def get(self, key: str, default: Any = None) -> Any:
        if len(self._symbols) == 1:
            return self._data[self._symbols[0]].get(key, default)
        return default

    def keys(self) -> Any:
        if len(self._symbols) == 1:
            return self._data[self._symbols[0]].keys()
        raise RuntimeError("Ambiguous: multiple symbols. Use ctx.row('SYMBOL').keys().")

    def values(self) -> Any:
        if len(self._symbols) == 1:
            return self._data[self._symbols[0]].values()
        raise RuntimeError("Ambiguous: multiple symbols. Use ctx.row('SYMBOL').values().")

    def items(self) -> Any:
        if len(self._symbols) == 1:
            return self._data[self._symbols[0]].items()
        raise RuntimeError("Ambiguous: multiple symbols. Use ctx.row('SYMBOL').items().")

    # --- callable access (multi-asset ``ctx.row("BTC")``) ---

    def __call__(self, symbol: str | None = None) -> dict[str, Any]:
        if symbol is None:
            if len(self._symbols) == 1:
                return self._data[self._symbols[0]]
            raise ValueError(
                f"Must specify symbol when {len(self._symbols)} symbols are present. Available: {self._symbols}"
            )
        if symbol not in self._data:
            raise KeyError(f"Symbol '{symbol}' not available this bar. Available: {self._symbols}")
        return self._data[symbol]

    def __repr__(self) -> str:
        if len(self._symbols) == 1:
            return repr(self._data[self._symbols[0]])
        return f"_RowAccessor(symbols={self._symbols})"


@dataclass
class BacktestContext:
    """Context object passed to Strategy.next() on each bar.

    Attributes:
        timestamp: Current timestamp.
        bar_index: Current bar index in the dataset.
        portfolio: Reference to the Portfolio instance.
        symbols: Tradeable symbols on this bar (after universe filtering).
        data: Per-symbol bar data ``{symbol: {col: value, ...}}``.
        row: Dual-access helper — use ``ctx.row["close"]`` in single-asset mode
            or ``ctx.row("BTC")["close"]`` in multi-asset mode.
        first_seen_bar: Bar index when each symbol first appeared in the data.
        bar_count: Number of bars each symbol has been active so far.
        available_symbols: All symbols with data on this bar (before universe filtering).
    """

    timestamp: Any
    bar_index: int
    portfolio: "Portfolio"
    symbols: list[str]
    data: dict[str, dict[str, Any]]
    row: _RowAccessor = field(default_factory=lambda: _RowAccessor({}, []))  # set by Engine
    first_seen_bar: dict[str, int] = field(default_factory=dict)
    bar_count: dict[str, int] = field(default_factory=dict)
    available_symbols: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Build _RowAccessor from data if not already set properly
        if not self.row._symbols and self.data:
            object.__setattr__(self, "row", _RowAccessor(self.data, self.symbols))


class Portfolio:
    """
    Manages cash and positions with support for fractional shares and multiple assets.

    The Portfolio tracks:
    - Cash balance
    - Asset positions (can be fractional)
    - Historical equity curve
    - Transaction costs and slippage
    - Orders and trades
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        commission: float | tuple[float, float] | CommissionModel = 0.0,
        slippage: float | SlippageModel = 0.0,
        order_delay: int = 0,
        bars_per_day: float | None = None,
        borrow_rate: float = 0.0,
        max_position_size: float | None = None,
        max_total_exposure: float | None = None,
        max_drawdown_stop: float | None = None,
        daily_loss_limit: float | None = None,
        leverage: float = 1.0,
        maintenance_margin: float | None = None,
        fractional_shares: bool = True,
        factor_column: str | None = None,
    ):
        """
        Initialize a new portfolio.

        Args:
            initial_cash: Starting cash balance
            commission: Commission specification. Accepts a percentage float (e.g., 0.001 = 0.1%),
                       a tuple of (fixed, percent) (e.g., (5.0, 0.001) = $5 + 0.1%), or a
                       CommissionModel instance for advanced models (MakerTakerCommission,
                       TieredCommission, CustomCommission, etc.)
            slippage: Slippage rate as a fraction (e.g., 0.0005 = 0.05%)
            order_delay: Number of bars to delay order execution (0 = immediate, 1 = next bar)
            bars_per_day: Number of bars in a trading day (used for day order expiry fallback).
                         For example: 390 for 1-min bars, 7.5 for 1-hour bars, 1 for daily bars.
                         If None, day orders will try to use timestamps to detect day boundaries,
                         or fall back to bars_valid parameter.
            borrow_rate: Annual borrow rate for short positions (e.g., 0.02 = 2% per year).
                        Cost is deducted from cash each bar based on the short position value.
                        Daily rate = borrow_rate / 252 (trading days). For intraday bars,
                        set bars_per_day so the cost is spread across bars within a day.
            max_position_size: Maximum single position size as fraction of portfolio value
                              (e.g., 0.5 = 50%). Orders that would exceed this are clamped.
                              None means no limit.
            max_total_exposure: Maximum total exposure as fraction of portfolio value
                               (e.g., 1.5 = 150%). Sum of absolute position values / portfolio value.
                               Orders that would exceed this are clamped. None means no limit.
            max_drawdown_stop: Maximum drawdown before halting trading (e.g., 0.2 = 20%).
                              When peak-to-trough drawdown exceeds this, new risk-increasing orders
                              are rejected. Risk-reducing orders (SL/TP/closes) still execute.
                              None means no limit.
            daily_loss_limit: Maximum daily loss before halting trading for the day (e.g., 0.05 = 5%).
                             When intraday loss exceeds this percentage of start-of-day equity,
                             new risk-increasing orders are rejected until the next day.
                             None means no limit.
            leverage: Maximum leverage multiplier (e.g., 2.0 = 2x leverage). Buying power
                     equals equity × leverage. Default 1.0 (no leverage). Cash can go negative
                     when leverage > 1, representing borrowed funds from the broker.
            maintenance_margin: Minimum margin ratio before margin call (e.g., 0.25 = 25%).
                               Margin ratio = equity / total_abs_position_value.
                               When margin ratio falls below this, all positions are auto-closed.
                                None means no margin calls (default). Only relevant when leverage > 1.
            fractional_shares: Whether to allow fractional share quantities (default True).
                              When False, order quantities are truncated to whole numbers
                              (toward zero). Useful for stock markets that don't support
                              fractional trading.
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash

        # Parse commission into a CommissionModel
        self.commission_model = make_commission_model(commission)

        # Keep legacy attributes for backward compatibility
        if isinstance(commission, tuple):
            self.commission_fixed = commission[0]
            self.commission_percent = commission[1]
        elif isinstance(commission, CommissionModel):
            self.commission_fixed = 0.0
            self.commission_percent = 0.0
        else:
            self.commission_fixed = 0.0
            self.commission_percent = commission

        self.slippage_model = make_slippage_model(slippage)
        # Keep legacy float attribute for backward compatibility
        self.slippage = slippage if isinstance(slippage, (int, float)) else 0.0
        self.order_delay = order_delay
        self.bars_per_day = bars_per_day
        self.borrow_rate = borrow_rate

        # Risk limits
        self.max_position_size = max_position_size
        self.max_total_exposure = max_total_exposure
        self.max_drawdown_stop = max_drawdown_stop
        self.daily_loss_limit = daily_loss_limit

        # Margin & Leverage
        self.leverage = leverage
        self.maintenance_margin = maintenance_margin
        self._margin_called: bool = False
        self.fractional_shares = fractional_shares
        self.factor_column = factor_column

        # Factor tracking for commission calculation on raw prices
        self._factors: dict[str, float] = {}

        # Risk limit state
        self._peak_equity: float = initial_cash
        self._trading_halted: bool = False
        self._daily_halted: bool = False
        self._day_start_equity: float = initial_cash
        self._last_date: Any = None

        # Asset positions: {asset_name: quantity}
        self.positions: dict[str, float] = defaultdict(float)

        # History tracking for metrics calculation
        self.equity_curve: list[float] = []
        self.timestamps: list[Any] = []

        # Current prices and OHLC data for portfolio valuation
        self._current_prices: dict[str, float] = {}
        self._current_ohlc: dict[str, dict[str, float]] = {}
        self._current_bar: int = 0
        self._current_timestamp: Any = None
        self._current_bar_data: dict[str, dict[str, Any]] = {}

        # Order management
        self.orders: dict[str, Order] = {}
        self._active_order_ids: set[str] = set()
        self._next_order_id: int = 0

        # Trade tracking
        self.trade_tracker = TradeTracker()

        # Stop-loss tracking: {asset: {"stop_price": float, "order_id": str}}
        self._stop_losses: dict[str, dict[str, Any]] = {}

        # Take-profit tracking: {asset: {"target_price": float, "order_id": str}}
        self._take_profits: dict[str, dict[str, Any]] = {}

        # Trailing stop tracking: {asset: {"trail_pct": float | None, "trail_amount": float | None,
        #                                   "highest_price": float, "stop_price": float, "order_id": str}}
        self._trailing_stops: dict[str, dict[str, Any]] = {}

    def update_prices(
        self,
        prices: dict[str, float],
        bar_index: int = 0,
        ohlc_data: dict[str, dict[str, float]] | None = None,
        timestamp: Any = None,
    ) -> None:
        """
        Update current market prices and OHLC data for all assets and execute pending orders.

        Args:
            prices: Dictionary mapping asset names to current close prices
            bar_index: Current bar index (used for order delay)
            ohlc_data: Optional OHLC data {asset: {"open": x, "high": x, "low": x, "close": x}}
            timestamp: Current timestamp
        """
        self._current_prices = prices
        self._current_ohlc = ohlc_data or {}
        self._current_bar = bar_index
        self._current_timestamp = timestamp

        # Deduct borrow costs for short positions
        if self.borrow_rate > 0:
            self._deduct_borrow_costs()

        # Update risk limit state (drawdown halt, daily loss halt)
        self._update_risk_limits()

        # Check for margin calls (auto-close positions if margin ratio too low)
        self._check_margin_call()

        # Update MAE/MFE for open positions
        for asset in self.trade_tracker.open_positions:
            if asset in prices:
                self.trade_tracker.update_mae_mfe(asset, prices[asset])

        # Use priority-based stop checking when OHLC data is available
        if self._current_ohlc:
            self._check_stops_with_priority()
        else:
            self._check_stop_losses()
            self._check_take_profits()
            self._update_and_check_trailing_stops()

        # Check and expire orders that have exceeded their valid_until time
        self._check_order_expiry()

        # Execute pending orders
        self._execute_pending_orders()

        # Purge inactive orders when dict grows too large to prevent
        # unbounded growth and potential memory corruption (Issue 1)
        if len(self.orders) > self._MAX_ORDERS_BEFORE_PURGE:
            self._purge_inactive_orders()

    def _deduct_borrow_costs(self) -> None:
        """Deduct borrow costs for short positions.

        Daily rate = borrow_rate / 252. For intraday bars, the daily rate is
        further divided by bars_per_day so the total daily cost is consistent.
        """
        daily_rate = self.borrow_rate / 252.0
        if self.bars_per_day is not None and self.bars_per_day > 0:
            bar_rate = daily_rate / self.bars_per_day
        else:
            bar_rate = daily_rate

        for asset, qty in self.positions.items():
            if qty < 0:
                price = self._current_prices.get(asset, 0.0)
                short_value = abs(qty) * price
                borrow_cost = short_value * bar_rate
                self.cash -= borrow_cost

    def _update_risk_limits(self) -> None:
        """Update risk limit state: peak equity, drawdown halt, daily loss halt."""
        current_equity = self.get_value()

        # Update peak equity for drawdown tracking
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        # Check max drawdown stop
        if self.max_drawdown_stop is not None and self._peak_equity > 0:
            drawdown = (self._peak_equity - current_equity) / self._peak_equity
            if drawdown >= self.max_drawdown_stop:
                self._trading_halted = True

        # Check daily loss limit
        if self.daily_loss_limit is not None:
            current_date = _extract_date(self._current_timestamp) if self._current_timestamp is not None else None
            if current_date is not None and current_date != self._last_date:
                # New day — reset daily halt and record start-of-day equity
                self._daily_halted = False
                self._day_start_equity = current_equity
                self._last_date = current_date

            if self._day_start_equity > 0:
                daily_loss = (self._day_start_equity - current_equity) / self._day_start_equity
                if daily_loss >= self.daily_loss_limit:
                    self._daily_halted = True

    def _check_margin_call(self) -> None:
        """Check if margin ratio has fallen below maintenance margin and close all positions.

        A margin call occurs when equity / total_abs_position_value < maintenance_margin.
        When triggered, all positions are immediately closed at current market prices.
        """
        if self.maintenance_margin is None or not self.positions:
            return

        margin_ratio = self.get_margin_ratio()
        if margin_ratio is not None and margin_ratio < self.maintenance_margin:
            self._margin_called = True
            # Close all positions — iterate over a copy since close_position modifies dict
            for asset in list(self.positions.keys()):
                position = self.positions.get(asset, 0.0)
                if position != 0:
                    quantity = -position
                    order_id = self._generate_order_id()
                    order = Order(
                        order_id=order_id,
                        asset=asset,
                        size=quantity,
                        order_type=OrderType.MARKET,
                        status=OrderStatus.PENDING,
                        created_bar=self._current_bar,
                        created_timestamp=self._current_timestamp,
                        _risk_order=True,
                    )
                    self.orders[order_id] = order
                    self._active_order_ids.add(order_id)
                    self._try_execute_order(order)

    def _is_order_risk_reducing(self, asset: str, quantity: float) -> bool:
        """Check if an order reduces risk (moves position toward zero).

        Args:
            asset: Asset symbol.
            quantity: Order quantity (positive=buy, negative=sell).

        Returns:
            True if the order reduces or closes the position.
        """
        current = self.get_position(asset)
        if current == 0:
            return False
        # Reducing: selling a long or buying back a short, and not exceeding zero
        if current > 0 and quantity < 0 and abs(quantity) <= abs(current):
            return True
        return current < 0 and quantity > 0 and abs(quantity) <= abs(current)

    def _get_total_exposure(self) -> float:
        """Get total exposure as fraction of portfolio value.

        Returns:
            Sum of absolute position values divided by portfolio value.
        """
        portfolio_value = self.get_value()
        if portfolio_value <= 0:
            return 0.0
        total_abs_value = sum(abs(qty) * self._current_prices.get(asset, 0.0) for asset, qty in self.positions.items())
        return total_abs_value / portfolio_value

    def _clamp_order_for_position_limit(self, asset: str, quantity: float, price: float) -> float:
        """Clamp order quantity so the resulting position doesn't exceed max_position_size.

        Args:
            asset: Asset symbol.
            quantity: Desired order quantity.
            price: Expected execution price.

        Returns:
            Clamped quantity (may be 0 if fully blocked).
        """
        if self.max_position_size is None:
            return quantity
        portfolio_value = self.get_value()
        if portfolio_value <= 0:
            return 0.0
        max_value = self.max_position_size * portfolio_value
        max_qty = max_value / price if price > 0 else 0.0

        current = self.get_position(asset)
        new_position = current + quantity
        if abs(new_position) * price <= max_value:
            return quantity

        # Clamp: allow up to max_qty in the direction of new_position
        allowed = max_qty - current if new_position > 0 else -(max_qty + current)

        # If clamping reverses direction (sign mismatch), return 0
        if quantity > 0:
            return max(0.0, allowed)
        else:
            return min(0.0, allowed)

    def _clamp_order_for_exposure_limit(self, asset: str, quantity: float, price: float) -> float:
        """Clamp order quantity so total exposure doesn't exceed max_total_exposure.

        Args:
            asset: Asset symbol.
            quantity: Desired order quantity.
            price: Expected execution price.

        Returns:
            Clamped quantity (may be 0 if fully blocked).
        """
        if self.max_total_exposure is None:
            return quantity
        portfolio_value = self.get_value()
        if portfolio_value <= 0:
            return 0.0

        # Calculate current total abs exposure excluding this asset
        current_other_exposure = sum(
            abs(qty) * self._current_prices.get(a, 0.0) for a, qty in self.positions.items() if a != asset
        )
        current_asset_position = self.get_position(asset)
        new_position = current_asset_position + quantity
        new_asset_exposure = abs(new_position) * price
        new_total_exposure = current_other_exposure + new_asset_exposure

        max_exposure_value = self.max_total_exposure * portfolio_value
        if new_total_exposure <= max_exposure_value:
            return quantity

        # How much room do we have?
        room = max_exposure_value - current_other_exposure
        if room <= 0:
            # If we're already reducing exposure, allow it
            if abs(new_position) < abs(current_asset_position):
                return quantity
            return 0.0

        # Max allowed abs position for this asset
        max_asset_qty = room / price if price > 0 else 0.0

        if new_position > 0:
            allowed_qty = max_asset_qty - current_asset_position
        elif new_position < 0:
            allowed_qty = -(max_asset_qty + current_asset_position)
        else:
            return quantity  # going to zero is always fine

        if quantity > 0:
            return max(0.0, min(quantity, allowed_qty))
        else:
            return min(0.0, max(quantity, allowed_qty))

    @property
    def trading_halted(self) -> bool:
        """Whether trading is halted due to risk limits."""
        return self._trading_halted or self._daily_halted

    def get_value(self) -> float:
        """
        Calculate total portfolio value (cash + positions).

        Returns:
            Total portfolio value
        """
        positions_value = sum(qty * self._current_prices.get(asset, 0.0) for asset, qty in self.positions.items())
        return self.cash + positions_value

    def get_buying_power(self) -> float:
        """Calculate available buying power with leverage.

        Buying power = equity × leverage − total absolute position value.
        When leverage is 1.0, this equals available cash (no borrowed funds).

        Returns:
            Available buying power for opening new positions.
        """
        equity = self.get_value()
        total_position_value = sum(
            abs(qty) * self._current_prices.get(asset, 0.0) for asset, qty in self.positions.items()
        )
        return max(0.0, equity * self.leverage - total_position_value)

    def get_margin_used(self) -> float:
        """Get margin currently used by open positions.

        Margin used = total absolute position value / leverage.
        This represents how much of your equity is committed as collateral.

        Returns:
            Margin used in currency units.
        """
        total_position_value = sum(
            abs(qty) * self._current_prices.get(asset, 0.0) for asset, qty in self.positions.items()
        )
        return total_position_value / self.leverage if self.leverage > 0 else 0.0

    def get_margin_available(self) -> float:
        """Get remaining margin available for new positions.

        Margin available = equity − margin used.

        Returns:
            Available margin in currency units.
        """
        return max(0.0, self.get_value() - self.get_margin_used())

    def get_margin_ratio(self) -> float | None:
        """Get current margin ratio (equity / total position value).

        A margin ratio of 1.0 means equity equals total position value (no leverage used).
        A margin ratio of 0.5 means you are using 2x leverage.
        A margin ratio below maintenance_margin triggers a margin call.

        Returns:
            Margin ratio as a float, or None if no positions are open.
        """
        total_position_value = sum(
            abs(qty) * self._current_prices.get(asset, 0.0) for asset, qty in self.positions.items()
        )
        if total_position_value == 0:
            return None
        return self.get_value() / total_position_value

    def get_position(self, asset: str) -> float:
        """
        Get current position size for an asset.

        Args:
            asset: Asset name

        Returns:
            Position quantity (can be fractional)
        """
        return self.positions.get(asset, 0.0)

    def _close_at_price(self, asset: str, fill_price: float) -> str | None:
        """Close position with a market order that fills at a specific price.

        Used by SL/TP/trailing stop to fill at the trigger price rather than
        the bar's close. Slippage is still applied by the execution engine.

        Args:
            asset: Asset symbol.
            fill_price: The pre-slippage price at which the order should fill.

        Returns:
            order_id if order was placed, None otherwise.
        """
        position = self.get_position(asset)
        if position == 0:
            return None
        quantity = -position  # reverse the position
        order_id = self.order(asset, quantity)
        if order_id is not None:
            order = self.orders[order_id]
            # If the order was filled immediately (order_delay=0), we need to
            # adjust the fill. For delayed orders, set _forced_price so the
            # execution engine uses it when the order eventually fills.
            if order.is_filled():
                # Already filled at market — need to correct the fill price.
                # Recalculate with the intended price.
                self._correct_fill_price(order, fill_price)
            else:
                order._forced_price = fill_price
        return order_id

    def _correct_fill_price(self, order: Order, intended_price: float) -> None:
        """Correct a filled order's price to the intended SL/TP/trailing stop price.

        Reverses the cash effect of the original fill and re-applies it at the
        correct price (with slippage).

        Args:
            order: The already-filled order to correct.
            intended_price: The pre-slippage trigger price.
        """
        old_exec_price = order.filled_price
        if old_exec_price is None:
            return

        # Calculate new execution price with slippage
        slippage_ctx = self._current_bar_data.get(order.asset) if self._current_bar_data else None
        new_exec_price = self.slippage_model.calculate(intended_price, abs(order.size), order.is_buy(), slippage_ctx)

        # Calculate cash difference
        size = abs(order.size)
        old_gross = size * old_exec_price
        new_gross = size * new_exec_price

        # Recalculate commission on new gross (use raw price if factor available)
        old_commission = order.commission_paid
        factor = self._factors.get(order.asset, 1.0)
        raw_new_exec_price = new_exec_price / factor if factor != 0 else new_exec_price
        new_commission = self.commission_model.calculate(size, raw_new_exec_price)

        if order.is_buy():
            # Buying: cash was decreased by old cost, needs to be decreased by new cost
            # old: cash -= old_gross + old_commission
            # new: cash -= new_gross + new_commission
            cash_adjustment = (old_gross + old_commission) - (new_gross + new_commission)
            self.cash += cash_adjustment
        else:
            # Selling: cash was increased by old proceeds
            # old: cash += old_gross - old_commission
            # new: cash += new_gross - new_commission
            cash_adjustment = (new_gross - new_commission) - (old_gross - old_commission)
            self.cash += cash_adjustment

        # Update order record
        slippage_cost = size * abs(new_exec_price - intended_price)
        order.filled_price = new_exec_price
        order.commission_paid = new_commission
        order.slippage_cost = slippage_cost

        # Update trade tracker — correct the exit price of the last trade
        if self.trade_tracker.trades:
            last_trade = self.trade_tracker.trades[-1]
            if last_trade.asset == order.asset:
                last_trade.exit_price = new_exec_price
                last_trade.exit_value = last_trade.exit_size * new_exec_price
                last_trade.exit_commission = new_commission
                last_trade._calculate_metrics()

    def _check_stop_losses(self) -> None:
        """Check if any stop-loss orders should be triggered."""
        for asset in list(self._stop_losses.keys()):
            stop_info = self._stop_losses[asset]
            stop_price = stop_info["stop_price"]

            # Get OHLC data for this asset
            ohlc = self._current_ohlc.get(asset, {})
            low = ohlc.get("low", self._current_prices.get(asset, float("inf")))
            high = ohlc.get("high", self._current_prices.get(asset, 0.0))

            # Check if stop was hit
            position_size = self.get_position(asset)
            if position_size > 0 and low <= stop_price:
                # Long position stop hit - close at stop price
                self._close_at_price(asset, stop_price)
                if asset in self._stop_losses:
                    del self._stop_losses[asset]
            elif position_size < 0 and high >= stop_price:
                # Short position stop hit - close at stop price
                self._close_at_price(asset, stop_price)
                if asset in self._stop_losses:
                    del self._stop_losses[asset]

    def _check_take_profits(self) -> None:
        """Check if any take-profit orders should be triggered."""
        for asset in list(self._take_profits.keys()):
            tp_info = self._take_profits[asset]
            target_price = tp_info["target_price"]

            # Get OHLC data for this asset
            ohlc = self._current_ohlc.get(asset, {})
            low = ohlc.get("low", self._current_prices.get(asset, float("inf")))
            high = ohlc.get("high", self._current_prices.get(asset, 0.0))

            # Check if take-profit was hit
            position_size = self.get_position(asset)
            if position_size > 0 and high >= target_price:
                # Long position take-profit hit - close at target price
                self._close_at_price(asset, target_price)
                if asset in self._take_profits:
                    del self._take_profits[asset]
            elif position_size < 0 and low <= target_price:
                # Short position take-profit hit - close at target price
                self._close_at_price(asset, target_price)
                if asset in self._take_profits:
                    del self._take_profits[asset]

    def _check_stops_with_priority(self) -> None:
        """Check all stop conditions with OHLC priority: open > high > low.

        Uses a two-phase approach to avoid mutating self.orders while other
        methods in the same update_prices() cycle may reference it:

        Phase 1 — Detect triggers (read-only on self.orders):
            1. If open breaches any stop -> exit at open price (gap scenario)
            2. If high breaches take-profit (long) or stop-loss (short) -> exit at threshold
            3. If low breaches stop-loss (long) or take-profit (short) -> exit at threshold

        Phase 2 — Execute all detected closes (mutates self.orders).

        Only one exit per asset per bar. First match wins.
        """
        assets_to_check: set[str] = set()
        assets_to_check.update(self._stop_losses.keys())
        assets_to_check.update(self._take_profits.keys())
        assets_to_check.update(self._trailing_stops.keys())

        # Phase 1: Detect triggers (does not mutate self.orders)
        pending_closes: list[tuple[str, float]] = []

        for asset in list(assets_to_check):
            if asset not in self.positions:
                continue

            ohlc = self._current_ohlc.get(asset, {})
            close_price = self._current_prices.get(asset, 0.0)
            open_price = ohlc.get("open", close_price)
            high_price = ohlc.get("high", open_price)
            low_price = ohlc.get("low", open_price)

            position_size = self.positions[asset]
            is_long = position_size > 0

            # Gather stop prices
            sl_info = self._stop_losses.get(asset)
            stop_price = sl_info["stop_price"] if sl_info else None

            tp_info = self._take_profits.get(asset)
            tp_price = tp_info["target_price"] if tp_info else None

            trail_info = self._trailing_stops.get(asset)
            trail_stop_price = trail_info["stop_price"] if trail_info else None

            # Effective stop = tightest of fixed stop and trailing stop
            effective_stop: float | None = None
            if is_long:
                candidates = [p for p in [stop_price, trail_stop_price] if p is not None]
                effective_stop = max(candidates) if candidates else None
            else:
                candidates = [p for p in [stop_price, trail_stop_price] if p is not None]
                effective_stop = min(candidates) if candidates else None

            fill_price: float | None = None

            # Priority 1: Open breaches
            if (
                effective_stop is not None
                and ((is_long and open_price <= effective_stop) or (not is_long and open_price >= effective_stop))
            ) or (
                tp_price is not None
                and ((is_long and open_price >= tp_price) or (not is_long and open_price <= tp_price))
            ):
                fill_price = open_price

            # Priority 2: High breaches (TP for long, SL for short)
            if fill_price is None:
                if is_long and tp_price is not None and high_price >= tp_price:
                    fill_price = tp_price
                elif not is_long and effective_stop is not None and high_price >= effective_stop:
                    fill_price = effective_stop

            # Priority 3: Low breaches (SL for long, TP for short)
            if fill_price is None:
                if is_long and effective_stop is not None and low_price <= effective_stop:
                    fill_price = effective_stop
                elif not is_long and tp_price is not None and low_price <= tp_price:
                    fill_price = tp_price

            if fill_price is not None:
                pending_closes.append((asset, fill_price))
            elif trail_info is not None:
                # No exit triggered — update trailing stop high-water mark
                trail_pct = trail_info["trail_pct"]
                trail_amount = trail_info["trail_amount"]
                if is_long and high_price > trail_info["highest_price"]:
                    trail_info["highest_price"] = high_price
                    if trail_pct is not None:
                        trail_info["stop_price"] = high_price * (1 - trail_pct)
                    elif trail_amount is not None:
                        trail_info["stop_price"] = high_price - trail_amount
                elif not is_long and low_price < trail_info["highest_price"]:
                    trail_info["highest_price"] = low_price
                    if trail_pct is not None:
                        trail_info["stop_price"] = low_price * (1 + trail_pct)
                    elif trail_amount is not None:
                        trail_info["stop_price"] = low_price + trail_amount

        # Phase 2: Execute closes (mutates self.orders)
        for asset, exit_price in pending_closes:
            self._close_at_price(asset, exit_price)
            self._cleanup_stops(asset)

    _MAX_ORDERS_BEFORE_PURGE = 1000

    def _purge_inactive_orders(self) -> None:
        """Remove filled, cancelled, rejected, and expired orders to prevent unbounded dict growth."""
        self.orders = {oid: order for oid, order in self.orders.items() if order.is_active()}

    def _cleanup_stops(self, asset: str) -> None:
        """Remove all stop-related state for an asset after exit."""
        self._stop_losses.pop(asset, None)
        self._take_profits.pop(asset, None)
        self._trailing_stops.pop(asset, None)

    def _execute_pending_orders(self) -> None:
        """Execute orders that are due."""
        if not self._active_order_ids:
            return
        # Get all pending orders that should execute this bar
        orders_to_execute = [
            self.orders[oid]
            for oid in self._active_order_ids
            if self.orders[oid].created_bar + self.order_delay <= self._current_bar
        ]

        for order in orders_to_execute:
            self._try_execute_order(order)

    def _check_order_expiry(self) -> None:
        """Check and expire orders that have passed their valid_until time or expiry_date."""
        if not self._active_order_ids:
            return
        current_date = _extract_date(self._current_timestamp)

        expired_ids: list[str] = []
        for oid in self._active_order_ids:
            order = self.orders[oid]

            # Check date-based expiry first (if order has expiry_date set)
            if order.expiry_date is not None and current_date is not None and current_date > order.expiry_date:
                order.mark_expired()
                expired_ids.append(oid)
                continue

            # Check bar-based expiry (skip if this order uses date-based expiry / valid_until was set to large number)
            if (
                order.valid_until is not None
                and self._current_bar > order.valid_until
                and (order.expiry_date is None or order.valid_until < 999999)
            ):
                order.mark_expired()
                expired_ids.append(oid)

        for oid in expired_ids:
            self._active_order_ids.discard(oid)

    def _can_fill_limit_order(self, order: Order) -> bool:
        """
        Check if a limit order can be filled based on current OHLC data.

        For buy limit: low must be <= limit_price
        For sell limit: high must be >= limit_price

        Works for both LIMIT orders and triggered STOP_LIMIT orders.

        Args:
            order: The limit order to check

        Returns:
            True if order can be filled, False otherwise
        """
        is_limit = order.order_type == OrderType.LIMIT
        is_triggered_stop_limit = order.order_type == OrderType.STOP_LIMIT and order.triggered

        if not is_limit and not is_triggered_stop_limit:
            return True

        if order.limit_price is None:
            return True

        ohlc = self._current_ohlc.get(order.asset)
        if not ohlc:
            return False

        if order.is_buy():
            return ohlc.get("low", float("inf")) <= order.limit_price
        else:
            return ohlc.get("high", 0) >= order.limit_price

    def _is_stop_triggered(self, order: Order) -> bool:
        """
        Check if a stop order's trigger price has been hit.

        For buy stops: triggered when high >= stop_price (breakout entry)
        For sell stops: triggered when low <= stop_price (breakdown entry)

        Args:
            order: The STOP or STOP_LIMIT order to check

        Returns:
            True if stop price was hit, False otherwise
        """
        if order.stop_price is None:
            return False

        ohlc = self._current_ohlc.get(order.asset, {})
        high = ohlc.get("high", self._current_prices.get(order.asset, 0.0))
        low = ohlc.get("low", self._current_prices.get(order.asset, float("inf")))

        if order.is_buy():
            return high >= order.stop_price
        else:
            return low <= order.stop_price

    def _can_afford_order(self, total_cost: float, asset: str, quantity: float, price: float) -> bool:
        """Check if an order can be afforded, considering leverage.

        Without leverage (leverage=1.0), this is a simple cash check.
        With leverage, checks if the resulting position fits within margin constraints:
        new total margin required must not exceed equity.

        Args:
            total_cost: Total cash cost of the order (including commission).
            asset: Asset symbol.
            quantity: Order quantity (signed).
            price: Execution price.

        Returns:
            True if the order can be afforded.
        """
        if self.leverage <= 1.0:
            return total_cost <= self.cash

        # With leverage: check if buying power covers the new position value
        new_position = self.positions.get(asset, 0.0) + quantity
        new_position_value = abs(new_position) * price

        # Calculate total position value after this order (excluding this asset's old value)
        other_position_value = sum(
            abs(qty) * self._current_prices.get(a, 0.0) for a, qty in self.positions.items() if a != asset
        )
        new_total_position_value = other_position_value + new_position_value

        # Margin required = total position value / leverage
        margin_required = new_total_position_value / self.leverage

        # Equity is reduced by commission only (position value changes are neutral to equity)
        commission_cost = total_cost - abs(quantity) * price if quantity > 0 else 0.0
        equity_after = self.get_value() - abs(commission_cost)

        return equity_after >= margin_required

    def _try_execute_order(self, order: Order) -> bool:
        """
        Try to execute an order.

        Handles MARKET, LIMIT, STOP, and STOP_LIMIT order types.
        Supports both long and short positions.

        Args:
            order: Order to execute

        Returns:
            True if order was executed, False otherwise
        """
        result = self._try_execute_order_inner(order)
        if not order.is_active():
            self._active_order_ids.discard(order.order_id)
        return result

    def _try_execute_order_inner(self, order: Order) -> bool:
        """Inner implementation of order execution logic."""
        if order.size == 0:
            order.mark_rejected()
            return False

        # Handle STOP orders: check if stop price is triggered, then execute as market
        if order.order_type == OrderType.STOP and not self._is_stop_triggered(order):
            return False

        # Handle STOP_LIMIT orders: two-phase execution
        if order.order_type == OrderType.STOP_LIMIT:
            if not order.triggered:
                # Phase 1: Check if stop price is triggered
                if not self._is_stop_triggered(order):
                    return False
                # Stop triggered — convert to limit order behavior
                order.triggered = True
                # Now check if the limit can also fill on this same bar
                # (fall through to limit check below)

            # Phase 2: Behave as a limit order
            if order.limit_price is None:
                order.mark_rejected()
                return False
            if not self._can_fill_limit_order(order):
                return False

        # Check if limit order can be filled
        if order.order_type == OrderType.LIMIT and not self._can_fill_limit_order(order):
            return False

        # Determine execution price
        if order._forced_price is not None:
            price: float = order._forced_price
        elif order.order_type == OrderType.STOP and order.stop_price is not None:
            price = order.stop_price
        elif order.order_type == OrderType.LIMIT or (order.order_type == OrderType.STOP_LIMIT and order.triggered):
            if order.limit_price is not None:
                price = order.limit_price
            else:
                price_value = self._current_prices.get(order.asset)
                if price_value is None:
                    order.mark_rejected()
                    return False
                price = price_value
        else:
            price_value = self._current_prices.get(order.asset)
            if price_value is None:
                order.mark_rejected()
                return False
            price = price_value

        if price <= 0:
            order.mark_rejected()
            return False

        # Apply slippage
        slippage_ctx = self._current_bar_data.get(order.asset) if self._current_bar_data else None
        execution_price = self.slippage_model.calculate(price, abs(order.size), order.is_buy(), slippage_ctx)

        # Enforce position size and exposure limits (skip for risk-reducing orders)
        if not order._risk_order and not self._is_order_risk_reducing(order.asset, order.size):
            clamped = self._clamp_order_for_position_limit(order.asset, order.size, execution_price)
            clamped = self._clamp_order_for_exposure_limit(order.asset, clamped, execution_price)
            if abs(clamped) < 1e-10:
                order.mark_rejected()
                return False
            if abs(clamped) != abs(order.size):
                order.size = clamped

        # Track old position for trade tracking
        old_position = self.get_position(order.asset)
        current_position = self.positions.get(order.asset, 0.0)

        # Determine if this order crosses zero (reversal) — needs two fixed commissions
        is_reversal = (order.is_buy() and current_position < 0 and abs(order.size) > abs(current_position)) or (
            order.is_sell() and current_position > 0 and abs(order.size) > current_position
        )

        # Calculate costs via commission model
        # When a factor column is configured, commissions are based on raw (unadjusted) prices
        factor = self._factors.get(order.asset, 1.0)
        raw_execution_price = execution_price / factor if factor != 0 else execution_price
        gross_cost = abs(order.size) * execution_price
        commission_cost = self.commission_model.calculate(abs(order.size), raw_execution_price, is_reversal)

        # Check if order can be afforded (margin-aware for leveraged accounts)
        if order.is_buy():
            if current_position < 0:
                # Covering a short position (partially or fully + possible new long)
                cover_size = min(abs(order.size), abs(current_position))
                new_long_size = abs(order.size) - cover_size

                cover_cost = cover_size * execution_price
                new_long_cost = new_long_size * execution_price
                total_cost = cover_cost + new_long_cost + commission_cost

                if not self._can_afford_order(total_cost, order.asset, order.size, execution_price):
                    order.mark_rejected()
                    return False
                self.cash -= total_cost
            else:
                # Opening or increasing a long position
                total_cost = gross_cost + commission_cost
                if not self._can_afford_order(total_cost, order.asset, order.size, execution_price):
                    order.mark_rejected()
                    return False
                self.cash -= total_cost

            self.positions[order.asset] += order.size
        else:
            # Selling: could be closing a long, or opening/increasing a short
            sell_size = abs(order.size)

            if current_position > 0 and sell_size <= current_position:
                # Closing or reducing a long position — receive proceeds
                self.cash += gross_cost - commission_cost
            elif current_position > 0 and sell_size > current_position:
                # Closing long AND opening short in one order
                long_close_proceeds = current_position * execution_price
                short_size = sell_size - current_position
                short_proceeds = short_size * execution_price
                self.cash += long_close_proceeds + short_proceeds - commission_cost
            else:
                # Opening or increasing a short position — receive sale proceeds
                self.cash += gross_cost - commission_cost

            self.positions[order.asset] += order.size

        # Clean up zero positions
        if order.asset in self.positions and abs(self.positions[order.asset]) < 1e-10:
            del self.positions[order.asset]

        # Mark order as filled
        slippage_cost = abs(order.size) * abs(execution_price - price)
        order.mark_filled(
            bar=self._current_bar,
            timestamp=self._current_timestamp,
            price=execution_price,
            commission=commission_cost,
            slippage=slippage_cost,
        )

        # Apply bracket order SL/TP if this entry order had bracket metadata
        if order.bracket_stop_loss is not None:
            self.set_stop_loss(order.asset, stop_price=order.bracket_stop_loss)
        if order.bracket_take_profit is not None:
            self.set_take_profit(order.asset, target_price=order.bracket_take_profit)

        # Update trade tracker
        new_position = self.get_position(order.asset)
        self._update_trade_tracker(order.asset, old_position, new_position, execution_price, commission_cost)

        return True

    def _update_trade_tracker(
        self, asset: str, old_position: float, new_position: float, price: float, commission: float
    ) -> None:
        """
        Update trade tracker based on position changes.

        Args:
            asset: Asset symbol
            old_position: Position before order
            new_position: Position after order
            price: Execution price
            commission: Commission paid
        """
        # Position opened
        if old_position == 0 and new_position != 0:
            self.trade_tracker.on_position_opened(
                asset=asset,
                size=new_position,
                price=price,
                bar=self._current_bar,
                timestamp=self._current_timestamp,
                commission=commission,
            )
        # Position closed completely
        elif old_position != 0 and new_position == 0:
            self.trade_tracker.on_position_closed(
                asset=asset,
                size_closed=abs(old_position),
                price=price,
                bar=self._current_bar,
                timestamp=self._current_timestamp,
                commission=commission,
            )
            # Clean up stop-loss, take-profit, and trailing stop when position closes
            self.remove_stop_loss(asset)
            self.remove_take_profit(asset)
            self.remove_trailing_stop(asset)
        # Position reduced (partial close)
        elif abs(new_position) < abs(old_position):
            size_closed = abs(old_position) - abs(new_position)
            self.trade_tracker.on_position_closed(
                asset=asset,
                size_closed=size_closed,
                price=price,
                bar=self._current_bar,
                timestamp=self._current_timestamp,
                commission=commission,
            )
        # Position increased (same direction, larger size)
        elif abs(new_position) > abs(old_position) and (
            (old_position > 0 and new_position > 0) or (old_position < 0 and new_position < 0)
        ):
            added_size = abs(new_position) - abs(old_position)
            self.trade_tracker.on_position_increased(
                asset=asset,
                added_size=added_size,
                price=price,
                commission=commission,
            )
        # Position reversed (e.g., long -> short)
        elif (old_position > 0 and new_position < 0) or (old_position < 0 and new_position > 0):
            self.trade_tracker.on_position_reversed(
                asset=asset,
                old_size=old_position,
                new_size=new_position,
                price=price,
                bar=self._current_bar,
                timestamp=self._current_timestamp,
                commission=commission,
            )

    def _round_quantity(self, quantity: float) -> float:
        """Round order quantity based on fractional_shares setting.

        When fractional_shares is False, truncates toward zero to whole numbers.
        When fractional_shares is True, returns quantity unchanged.
        """
        if self.fractional_shares:
            return quantity
        return float(math.trunc(quantity))

    def _generate_order_id(self) -> str:
        """Generate a unique order ID."""
        order_id = f"order_{self._next_order_id}"
        self._next_order_id += 1
        return order_id

    def order(
        self,
        asset: str,
        quantity: float,
        limit_price: float | None = None,
        stop_price: float | None = None,
        order_type: OrderType | None = None,
        tags: list[str] | None = None,
    ) -> str | None:
        """
        Place an order for an asset.

        Args:
            asset: Asset name
            quantity: Number of units to buy (positive) or sell (negative)
            limit_price: Optional limit price for LIMIT orders
            stop_price: Optional stop price for STOP orders
            order_type: Order type (defaults to MARKET if not specified)
            tags: Optional tags for categorization

        Returns:
            order_id if order was placed, None if rejected
        """
        quantity = self._round_quantity(quantity)
        if quantity == 0:
            return None

        # Reject risk-increasing orders when trading is halted
        if self.trading_halted and not self._is_order_risk_reducing(asset, quantity):
            return None

        # Reject STOP/STOP_LIMIT orders without a stop_price
        if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and stop_price is None:
            return None

        # Determine order type if not specified
        if order_type is None:
            if limit_price is not None and stop_price is not None:
                order_type = OrderType.STOP_LIMIT
            elif limit_price is not None:
                order_type = OrderType.LIMIT
            elif stop_price is not None:
                order_type = OrderType.STOP
            else:
                order_type = OrderType.MARKET

        # Create order
        order_id = self._generate_order_id()
        order = Order(
            order_id=order_id,
            asset=asset,
            size=quantity,
            order_type=order_type,
            status=OrderStatus.PENDING,
            limit_price=limit_price,
            stop_price=stop_price,
            created_bar=self._current_bar,
            created_timestamp=self._current_timestamp,
            tags=tags or [],
        )

        # Store order
        self.orders[order_id] = order
        self._active_order_ids.add(order_id)

        # Try to execute immediately if no delay
        if self.order_delay == 0:
            self._try_execute_order(order)

        return order_id

    def order_target(self, asset: str, target_quantity: float, limit_price: float | None = None) -> str | None:
        """
        Order to reach a target position size.

        Args:
            asset: Asset name
            target_quantity: Desired final position quantity
            limit_price: Optional limit price

        Returns:
            order_id if order was placed, None otherwise
        """
        current_position = self.get_position(asset)
        delta = target_quantity - current_position
        return self.order(asset, delta, limit_price=limit_price)

    def order_target_value(self, asset: str, target_value: float, limit_price: float | None = None) -> str | None:
        """
        Order to reach a target position value.

        Args:
            asset: Asset name
            target_value: Desired position value in currency
            limit_price: Optional limit price

        Returns:
            order_id if order was placed, None otherwise
        """
        price = limit_price if limit_price is not None else self._current_prices.get(asset)
        if price is None or price <= 0:
            return None

        target_quantity = target_value / price
        return self.order_target(asset, target_quantity, limit_price)

    def _estimate_fee_adjusted_quantity(self, quantity_delta: float, price: float) -> float:
        """Adjust a buy quantity downward to account for slippage and commission.

        When buying, the execution engine charges slippage (higher price) and
        commission on top of the gross cost. This method reduces the quantity
        so that the total cost (including fees) equals the original gross value.

        For sells and zero-fee scenarios the quantity is returned unchanged.

        Args:
            quantity_delta: Signed order quantity (positive = buy).
            price: Expected execution price (before slippage).

        Returns:
            Adjusted quantity delta.
        """
        if quantity_delta <= 0 or price <= 0:
            return quantity_delta

        # Effective price after slippage
        effective_price = self.slippage_model.calculate(price, quantity_delta, True)

        # Estimate per-unit commission by probing the commission model.
        # commission(q, p) is often proportional to q*p, but may have fixed
        # components. We compute the rate as commission(1, effective_price) / effective_price.
        probe_commission = self.commission_model.calculate(1.0, effective_price)
        commission_rate = probe_commission / effective_price if effective_price > 0 else 0.0

        # adjusted_qty * effective_price * (1 + commission_rate) = original_qty * price
        # => adjusted_qty = original_qty * price / (effective_price * (1 + commission_rate))
        cost_multiplier = effective_price * (1 + commission_rate)
        if cost_multiplier <= 0:
            return quantity_delta

        adjusted = quantity_delta * price / cost_multiplier

        # Tiny safety margin to avoid floating-point rounding making the order
        # exceed available cash by a fraction of a cent. Only applied when
        # there are actual costs to account for.
        if cost_multiplier > price:
            adjusted *= 1 - 1e-9

        return adjusted

    def order_target_percent(self, asset: str, target_percent: float, limit_price: float | None = None) -> str | None:
        """Order to reach a target percentage of portfolio value.

        The target percentage is inclusive of fees and slippage. For example,
        ordering 100% will allocate all available equity to the position,
        automatically reserving enough cash to cover commission and slippage
        so the order is not rejected.

        Args:
            asset: Asset name.
            target_percent: Desired position as fraction of portfolio (e.g., 0.5 = 50%).
            limit_price: Optional limit price.

        Returns:
            order_id if order was placed, None otherwise.
        """
        price = limit_price if limit_price is not None else self._current_prices.get(asset)
        if price is None or price <= 0:
            return None

        portfolio_value = self.get_value()
        current_position = self.get_position(asset)
        current_value = current_position * price

        # Calculate target value and the quantity delta needed
        target_value = portfolio_value * target_percent
        value_delta = target_value - current_value

        if abs(value_delta) < 1e-6:  # Already at target
            return None

        quantity_delta = value_delta / price

        # Adjust buy orders downward so total cost (including fees) fits
        adjusted_delta = self._estimate_fee_adjusted_quantity(quantity_delta, price)

        target_quantity = current_position + adjusted_delta
        return self.order_target(asset, target_quantity, limit_price)

    def rebalance(self, weights: dict[str, float]) -> list[str | None]:
        """Atomically rebalance the portfolio to target weights.

        Computes all target quantities from a single ``get_value()`` snapshot,
        then executes orders: closes positions not in the target, sells before
        buys, and fee-adjusts buy quantities so total cost fits available cash.

        Args:
            weights: Mapping of symbol to target weight (fraction of portfolio value).
                     Symbols not in *weights* are closed.

        Returns:
            List of order IDs (one per order placed, may contain None for skipped orders).
        """
        portfolio_value = self.get_value()
        if portfolio_value <= 0:
            return []

        # Compute target quantities for all symbols
        targets: dict[str, float] = {}
        for sym, w in weights.items():
            price = self._current_prices.get(sym)
            if price is None or price <= 0:
                continue
            targets[sym] = portfolio_value * w / price

        # Close positions not in target weights
        sells: list[tuple[str, float]] = []
        buys: list[tuple[str, float]] = []

        # First: handle existing positions not in target (close them)
        for sym in list(self.positions.keys()):
            target_qty = targets.get(sym, 0.0)
            current_qty = self.positions.get(sym, 0.0)
            delta = target_qty - current_qty
            if abs(delta) < 1e-10:
                continue
            if delta < 0:
                sells.append((sym, delta))
            else:
                buys.append((sym, delta))

        # Then: handle new positions (symbols in target but not in positions)
        for sym, target_qty in targets.items():
            if sym in self.positions:
                continue
            delta = target_qty
            if abs(delta) < 1e-10:
                continue
            if delta < 0:
                sells.append((sym, delta))
            else:
                buys.append((sym, delta))

        # Execute sells first (frees cash), then buys
        order_ids: list[str | None] = []
        for sym, delta in sells:
            order_ids.append(self.order(sym, delta))

        # Fee-adjust buy quantities
        for sym, delta in buys:
            price = self._current_prices.get(sym, 0.0)
            adjusted = self._estimate_fee_adjusted_quantity(delta, price)
            order_ids.append(self.order(sym, adjusted))

        return order_ids

    def close_position(self, asset: str, limit_price: float | None = None) -> str | None:
        """
        Close entire position in an asset.

        Args:
            asset: Asset name
            limit_price: Optional limit price

        Returns:
            order_id if order was placed, None otherwise
        """
        return self.order_target(asset, 0, limit_price)

    def close_all_positions(self) -> None:
        """Close all open positions."""
        assets = list(self.positions.keys())
        for asset in assets:
            self.close_position(asset)

    def order_with_sizer(
        self,
        asset: str,
        sizer: Any,
        direction: float,
        price: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        order_type: Any = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Place an order with quantity determined by a Sizer.

        Args:
            asset: Asset symbol.
            sizer: A Sizer instance that computes the unsigned quantity.
            direction: Positive for buy, negative for sell. Only the sign matters.
            price: Price to pass to the sizer for calculation. If None, uses
                the current asset price from the last known bar.
            limit_price: Optional limit price for the order.
            stop_price: Optional stop price for the order.
            order_type: Optional OrderType override.
            tags: Optional order tags.
            **kwargs: Extra keyword arguments forwarded to sizer.size().

        Returns:
            Order ID string if placed, None if rejected or size is zero.
        """
        from polarbt.sizers import Sizer as SizerClass

        if not isinstance(sizer, SizerClass):
            raise TypeError(f"sizer must be a Sizer instance, got {type(sizer).__name__}")

        if price is None:
            price = self._current_prices.get(asset, 0.0)
        if price <= 0:
            return None

        sign = 1.0 if direction > 0 else -1.0 if direction < 0 else 0.0
        if sign == 0.0:
            return None

        quantity = sizer.size(self, asset, price, **kwargs)
        if quantity <= 0:
            return None

        return self.order(
            asset,
            quantity * sign,
            limit_price=limit_price,
            stop_price=stop_price,
            order_type=order_type,
            tags=tags,
        )

    def order_day(
        self,
        asset: str,
        quantity: float,
        limit_price: float | None = None,
        bars_valid: int | None = None,
    ) -> str | None:
        """
        Place a Day order that expires at end of trading day.

        The expiry is calculated automatically using one of these methods (in priority order):
        1. Timestamp-based: If timestamps are available, order expires when date changes
        2. bars_valid parameter: If specified, order expires after that many bars
        3. bars_per_day config: If Portfolio was initialized with bars_per_day, use that
        4. Default: Falls back to 1 bar if nothing else specified

        Args:
            asset: Asset name
            quantity: Number of units to buy (positive) or sell (negative)
            limit_price: Optional limit price
            bars_valid: Optional number of bars order remains valid. If None, will auto-calculate
                       based on timestamps or bars_per_day configuration.

        Returns:
            order_id if order was placed, None if rejected

        Example:
            # Auto-detect day boundaries from timestamps
            portfolio.order_day("BTC", 0.1, limit_price=50000)

            # Or specify bars explicitly
            portfolio.order_day("BTC", 0.1, limit_price=50000, bars_valid=390)  # For 1-min bars
        """
        order_id = self.order(asset, quantity, limit_price=limit_price)
        if not order_id:
            return None

        order = self.orders[order_id]

        # Strategy 1: Use explicit bars_valid parameter (highest priority)
        if bars_valid is not None:
            order.valid_until = self._current_bar + bars_valid
        # Strategy 2: Try timestamp-based day detection
        elif (current_date := _extract_date(self._current_timestamp)) is not None:
            order.expiry_date = current_date
            # Also set valid_until as a backup (will be checked in expiry logic)
            # Set to a large number so it doesn't expire before date check
            order.valid_until = self._current_bar + 999999
        # Strategy 3: Use bars_per_day configuration (smart end-of-day calculation)
        elif self.bars_per_day is not None:
            # Calculate which bar within the current trading day we're on
            # Example: If bars_per_day=390 and current_bar=450, we're on bar 60 of day 1
            bpd = self.bars_per_day
            current_bar_in_day = self._current_bar % bpd
            # Calculate bars remaining until end of current day
            # Example: 390 - 60 - 1 = 329 bars until end of day
            bars_until_eod = bpd - current_bar_in_day - 1
            # Order expires at end of current trading day
            order.valid_until = int(self._current_bar + bars_until_eod)
        # Strategy 4: Default to 1 bar
        else:
            order.valid_until = self._current_bar + 1

        return order_id

    def order_gtc(
        self,
        asset: str,
        quantity: float,
        limit_price: float | None = None,
    ) -> str | None:
        """
        Place a Good-Till-Cancelled (GTC) order that remains active until filled or manually cancelled.

        Args:
            asset: Asset name
            quantity: Number of units to buy (positive) or sell (negative)
            limit_price: Optional limit price

        Returns:
            order_id if order was placed, None if rejected

        Example:
            # Order that stays active until filled
            portfolio.order_gtc("BTC", 0.1, limit_price=50000)

        Note:
            GTC orders have valid_until=None, which means they never expire automatically.
            Regular order() calls default to GTC behavior.
        """
        # Regular order() already defaults to GTC (valid_until=None)
        return self.order(asset, quantity, limit_price=limit_price)

    def get_orders(self, status: OrderStatus | None = None, asset: str | None = None) -> list[Order]:
        """
        Get orders, optionally filtered by status and/or asset.

        Args:
            status: Filter by order status (None = all orders)
            asset: Filter by asset (None = all assets)

        Returns:
            List of Order objects
        """
        orders = list(self.orders.values())

        if status is not None:
            orders = [o for o in orders if o.status == status]

        if asset is not None:
            orders = [o for o in orders if o.asset == asset]

        return orders

    def get_order(self, order_id: str) -> Order | None:
        """
        Get a specific order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order object or None if not found
        """
        return self.orders.get(order_id)

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled, False if order not found or can't be cancelled
        """
        order = self.orders.get(order_id)
        if order is None:
            return False

        if not order.can_be_cancelled():
            return False

        order.mark_cancelled()
        self._active_order_ids.discard(order_id)
        return True

    def set_stop_loss(self, asset: str, stop_price: float | None = None, stop_pct: float | None = None) -> str | None:
        """
        Set a stop-loss order for a position.

        Args:
            asset: Asset symbol
            stop_price: Absolute stop price
            stop_pct: Stop as percentage below entry (e.g., -0.05 for 5% stop)

        Returns:
            order_id if stop-loss was set, None otherwise
        """
        position = self.get_position(asset)
        if position == 0:
            return None  # No position to protect

        # Calculate stop price
        if stop_price is None and stop_pct is not None:
            current_price = self._current_prices.get(asset)
            if current_price is None:
                return None
            stop_price = current_price * (1 + stop_pct)

        if stop_price is None:
            return None

        # Store stop-loss info
        order_id = self._generate_order_id()
        self._stop_losses[asset] = {"stop_price": stop_price, "order_id": order_id}

        return order_id

    def remove_stop_loss(self, asset: str) -> bool:
        """
        Remove stop-loss for an asset.

        Args:
            asset: Asset symbol

        Returns:
            True if stop-loss was removed, False if none existed
        """
        if asset in self._stop_losses:
            del self._stop_losses[asset]
            return True
        return False

    def get_stop_loss(self, asset: str) -> float | None:
        """
        Get stop-loss price for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Stop-loss price or None if not set
        """
        if asset in self._stop_losses:
            return float(self._stop_losses[asset]["stop_price"])
        return None

    def set_take_profit(
        self, asset: str, target_price: float | None = None, target_pct: float | None = None
    ) -> str | None:
        """
        Set a take-profit order for a position.

        Args:
            asset: Asset symbol
            target_price: Absolute target price
            target_pct: Target as percentage above entry (e.g., 0.10 for 10% profit)

        Returns:
            order_id if take-profit was set, None otherwise
        """
        position = self.get_position(asset)
        if position == 0:
            return None  # No position to protect

        # Calculate target price
        if target_price is None and target_pct is not None:
            current_price = self._current_prices.get(asset)
            if current_price is None:
                return None
            target_price = current_price * (1 + target_pct)

        if target_price is None:
            return None

        # Store take-profit info
        order_id = self._generate_order_id()
        self._take_profits[asset] = {"target_price": target_price, "order_id": order_id}

        return order_id

    def remove_take_profit(self, asset: str) -> bool:
        """
        Remove take-profit for an asset.

        Args:
            asset: Asset symbol

        Returns:
            True if take-profit was removed, False if none existed
        """
        if asset in self._take_profits:
            del self._take_profits[asset]
            return True
        return False

    def get_take_profit(self, asset: str) -> float | None:
        """
        Get take-profit price for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Take-profit price or None if not set
        """
        if asset in self._take_profits:
            return float(self._take_profits[asset]["target_price"])
        return None

    def set_trailing_stop(
        self,
        asset: str,
        trail_pct: float | None = None,
        trail_amount: float | None = None,
    ) -> str | None:
        """
        Set a trailing stop-loss order for a position.

        The stop price will trail the market price by a fixed percentage or amount,
        moving up (for long) or down (for short) as the price moves favorably.

        Args:
            asset: Asset symbol
            trail_pct: Trail by percentage (e.g., 0.05 for 5% trailing stop)
            trail_amount: Trail by absolute amount (e.g., 1000 for $1000 trailing stop)

        Returns:
            order_id if trailing stop was set, None otherwise
        """
        position = self.get_position(asset)
        if position == 0:
            return None  # No position to protect

        if trail_pct is None and trail_amount is None:
            return None  # Need either percentage or amount

        # Get current price
        current_price = self._current_prices.get(asset)
        if current_price is None:
            return None

        # Initialize highest/lowest price for tracking
        if position > 0:
            # Long position - track highest price
            highest_price = current_price
            stop_price = current_price * (1 - trail_pct) if trail_pct is not None else current_price - trail_amount  # type: ignore
        else:
            # Short position - track lowest price
            highest_price = current_price  # We'll use this field for lowest in shorts
            stop_price = current_price * (1 + trail_pct) if trail_pct is not None else current_price + trail_amount  # type: ignore

        # Store trailing stop info
        order_id = self._generate_order_id()
        self._trailing_stops[asset] = {
            "trail_pct": trail_pct,
            "trail_amount": trail_amount,
            "highest_price": highest_price,
            "stop_price": stop_price,
            "order_id": order_id,
        }

        return order_id

    def remove_trailing_stop(self, asset: str) -> bool:
        """
        Remove trailing stop for an asset.

        Args:
            asset: Asset symbol

        Returns:
            True if trailing stop was removed, False if none existed
        """
        if asset in self._trailing_stops:
            del self._trailing_stops[asset]
            return True
        return False

    def get_trailing_stop(self, asset: str) -> float | None:
        """
        Get current trailing stop price for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Trailing stop price or None if not set
        """
        if asset in self._trailing_stops:
            return float(self._trailing_stops[asset]["stop_price"])
        return None

    def _update_and_check_trailing_stops(self) -> None:
        """
        Update and check trailing stops in one pass to handle intra-bar price action correctly.

        For each asset with a trailing stop:
        1. First check if the OLD stop price was hit by the bar's low (long) or high (short)
        2. If not hit, update the stop price based on new highs (long) or lows (short)

        This avoids the bug where we update based on the bar high, then trigger on the bar low.
        """
        for asset in list(self._trailing_stops.keys()):
            trail_info = self._trailing_stops[asset]
            old_stop_price = trail_info["stop_price"]

            position_size = self.get_position(asset)

            # If position is already closed, remove trailing stop
            if position_size == 0:
                del self._trailing_stops[asset]
                continue

            # Get OHLC data for this asset
            current_price = self._current_prices.get(asset)
            if current_price is None:
                continue

            ohlc = self._current_ohlc.get(asset, {})
            low = ohlc.get("low", current_price)
            high = ohlc.get("high", current_price)

            trail_pct = trail_info["trail_pct"]
            trail_amount = trail_info["trail_amount"]

            # STEP 1: Check if OLD stop price was hit
            stop_triggered = False
            if position_size > 0 and low <= old_stop_price:
                # Long position stop hit
                stop_triggered = True
            elif position_size < 0 and high >= old_stop_price:
                # Short position stop hit
                stop_triggered = True

            if stop_triggered:
                # Stop was hit - close at the trailing stop price
                self._close_at_price(asset, old_stop_price)
                if asset in self._trailing_stops:
                    del self._trailing_stops[asset]
            else:
                # STEP 2: Stop not hit - update if price made new high/low
                if position_size > 0:
                    # Long position - update if HIGH makes new high
                    if high > trail_info["highest_price"]:
                        trail_info["highest_price"] = high
                        # Recalculate stop price based on the new high
                        if trail_pct is not None:
                            trail_info["stop_price"] = high * (1 - trail_pct)
                        else:
                            trail_info["stop_price"] = high - trail_amount
                elif position_size < 0 and low < trail_info["highest_price"]:
                    # Short position - update if LOW makes new low
                    # Note: highest_price stores the lowest price for shorts
                    trail_info["highest_price"] = low
                    # Recalculate stop price based on the new low
                    if trail_pct is not None:
                        trail_info["stop_price"] = low * (1 + trail_pct)
                    else:
                        trail_info["stop_price"] = low + trail_amount

    def order_bracket(
        self,
        asset: str,
        quantity: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        limit_price: float | None = None,
    ) -> dict[str, str | None]:
        """
        Place a bracket order: entry order with attached stop-loss and take-profit.

        When the entry order fills, SL and TP are set automatically. When either SL or TP
        triggers, the position is closed and both are removed (OCO = One-Cancels-Other).

        Args:
            asset: Asset symbol
            quantity: Size to trade (positive=buy, negative=sell)
            stop_loss: Absolute stop-loss price
            take_profit: Absolute take-profit price
            stop_loss_pct: Stop-loss as percentage below/above entry (e.g., 0.05 for 5%)
            take_profit_pct: Take-profit as percentage above/below entry (e.g., 0.10 for 10%)
            limit_price: Optional limit price for entry order

        Returns:
            Dictionary with "entry", "stop_loss", and "take_profit" status

        Example:
            # Long position with 5% stop-loss and 10% take-profit
            result = portfolio.order_bracket(
                "BTC", 0.1, stop_loss_pct=0.05, take_profit_pct=0.10
            )

        Note:
            OCO behavior is automatic - when position closes (from either SL or TP),
            both stops are removed. See _update_trade_tracker() for cleanup logic.
        """
        # Place entry order
        entry_order_id = self.order(asset, quantity, limit_price=limit_price)
        if entry_order_id is None:
            return {"entry": None, "stop_loss": None, "take_profit": None}

        entry_order = self.orders[entry_order_id]

        # Get entry price (use fill price if filled, otherwise estimate)
        if entry_order.is_filled():
            entry_price = entry_order.filled_price
        elif limit_price is not None:
            entry_price = limit_price
        else:
            entry_price = self._current_prices.get(asset)

        if entry_price is None:
            return {"entry": entry_order_id, "stop_loss": None, "take_profit": None}

        # Calculate absolute prices from percentages
        if stop_loss is None and stop_loss_pct is not None:
            stop_loss = entry_price * (1 - stop_loss_pct) if quantity > 0 else entry_price * (1 + stop_loss_pct)

        if take_profit is None and take_profit_pct is not None:
            take_profit = entry_price * (1 + take_profit_pct) if quantity > 0 else entry_price * (1 - take_profit_pct)

        # If entry order filled immediately (order_delay=0), set stops on position
        if entry_order.is_filled():
            sl_id = None
            tp_id = None

            if stop_loss is not None:
                sl_id = self.set_stop_loss(asset, stop_price=stop_loss)

            if take_profit is not None:
                tp_id = self.set_take_profit(asset, target_price=take_profit)

            return {"entry": entry_order_id, "stop_loss": sl_id, "take_profit": tp_id}

        # Entry not yet filled — store bracket metadata on the order
        # SL/TP will be applied automatically when the order fills (see _try_execute_order)
        entry_order.bracket_stop_loss = stop_loss
        entry_order.bracket_take_profit = take_profit
        return {"entry": entry_order_id, "stop_loss": None, "take_profit": None}

    def get_trades(self) -> pl.DataFrame:
        """
        Get all completed trades as a DataFrame.

        Returns:
            Polars DataFrame with trade history
        """
        return self.trade_tracker.get_trades_df()

    def get_trade_stats(self) -> TradeStats:
        """Get aggregate trade statistics.

        Returns:
            TradeStats with win_rate, avg_win, avg_loss, profit_factor, etc.
        """
        return self.trade_tracker.get_trade_stats()

    def record_equity(self, timestamp: Any) -> None:
        """
        Record current portfolio value for metrics calculation.

        Args:
            timestamp: Current timestamp
        """
        self.equity_curve.append(self.get_value())
        self.timestamps.append(timestamp)


class Param:
    """Descriptor that reads/writes strategy parameters through ``self.params``.

    Use the :func:`param` factory to create instances on a Strategy subclass::

        class MyStrategy(Strategy):
            fast = param(10)
            slow = param(30)

    When accessed on an instance, the descriptor returns
    ``self.params.get(name, default)``.  When set, it writes into
    ``self.params[name]``.
    """

    def __init__(self, default: Any = None) -> None:
        self.default = default
        self.name: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return obj.params.get(self.name, self.default)

    def __set__(self, obj: Any, value: Any) -> None:
        obj.params[self.name] = value


def param(default: Any = None) -> Any:
    """Declare a strategy parameter with an optional default value.

    Usage::

        class MyStrategy(Strategy):
            fast_period = param(10)
            slow_period = param(30)

            def preprocess(self, df):
                return df.with_columns(
                    ind.sma("close", self.fast_period).alias("sma_fast"),
                    ind.sma("close", self.slow_period).alias("sma_slow"),
                )

    Parameters declared with ``param()`` are automatically populated from
    keyword arguments passed to ``Strategy.__init__()`` (or via ``backtest(params=...)``).
    No ``__init__`` override is needed.

    Args:
        default: Default value when the parameter is not provided.
    """
    return Param(default)


class Strategy(ABC):
    """
    Base class for trading strategies.

    Subclasses should implement:
    - preprocess(): Vectorized feature engineering using Polars
    - next(): Event-driven logic called on each bar

    Strategy parameters can be declared as class attributes using :func:`param`::

        class MyStrategy(Strategy):
            fast = param(10)
            slow = param(30)

    These are automatically populated from keyword arguments and accessible
    as ``self.fast`` / ``self.slow``.  The traditional ``self.params.get()``
    pattern is also supported.
    """

    def __init__(self, **params: Any) -> None:
        """
        Initialize strategy with parameters.

        Args:
            **params: Strategy parameters (e.g., sma_period=20)
        """
        self.params = params

    @abstractmethod
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Preprocess data using vectorized Polars operations.

        This method is called once before the backtest loop starts.
        Use it to calculate indicators, features, or ML predictions.

        Args:
            df: Input DataFrame with OHLCV data

        Returns:
            DataFrame with added indicator columns

        Example:
            def preprocess(self, df):
                from polarbt import indicators as ind
                return df.with_columns([
                    ind.sma("close", 20).alias("sma_20"),
                    ind.rsi("close", 14).alias("rsi_14")
                ])
        """
        pass

    @abstractmethod
    def next(self, ctx: BacktestContext) -> None:
        """
        Execute strategy logic for the current bar.

        This method is called on every bar during the backtest.
        Use ctx.portfolio to place orders.

        Args:
            ctx: BacktestContext containing current bar data and portfolio

        Example:
            def next(self, ctx):
                if ctx.row["rsi_14"] < 30:
                    ctx.portfolio.order_target_percent("BTC", 0.5)
                elif ctx.row["rsi_14"] > 70:
                    ctx.portfolio.close_position("BTC")
        """
        pass

    def on_start(self, portfolio: Portfolio) -> None:  # noqa: B027
        """
        Called once before the backtest starts.

        Override this to perform any initialization logic.

        Args:
            portfolio: The Portfolio instance
        """
        pass

    def on_finish(self, portfolio: Portfolio) -> None:  # noqa: B027
        """
        Called once after the backtest completes.

        Override this to perform cleanup or final analysis.

        Args:
            portfolio: The Portfolio instance
        """
        pass


class WeightStrategy(Strategy):
    """Strategy that expresses positions as target weights per symbol.

    Subclasses implement ``get_weights()`` instead of ``next()``.
    On each bar the returned weights are passed to ``Portfolio.rebalance()``,
    which uses the unified order execution path (commissions, slippage,
    stop-loss/take-profit, leverage all apply).

    Example::

        class EqualWeight(WeightStrategy):
            def preprocess(self, df):
                return df

            def get_weights(self, ctx):
                n = len(ctx.symbols)
                return {sym: 1.0 / n for sym in ctx.symbols}
    """

    @abstractmethod
    def get_weights(self, ctx: BacktestContext) -> dict[str, float]:
        """Return target portfolio weights for each symbol.

        Args:
            ctx: Current bar context.

        Returns:
            Mapping of symbol to target weight (fraction of portfolio value).
        """
        ...

    def next(self, ctx: BacktestContext) -> None:
        """Execute rebalance based on ``get_weights()``."""
        weights = self.get_weights(ctx)
        ctx.portfolio.rebalance(weights)


class Engine:
    """
    Backtesting engine that executes the strategy simulation.

    The engine:
    1. Preprocesses data using the strategy
    2. Iterates through each bar
    3. Updates portfolio prices
    4. Calls strategy.next()
    5. Records metrics
    """

    def __init__(
        self,
        strategy: Strategy,
        data: pl.DataFrame | dict[str, pl.DataFrame],
        initial_cash: float = 100_000.0,
        commission: float | tuple[float, float] | CommissionModel = 0.0,
        slippage: float | SlippageModel = 0.0,
        price_columns: dict[str, str] | None = None,
        warmup: int | str = "auto",
        order_delay: int = 0,
        borrow_rate: float = 0.0,
        bars_per_day: float | None = None,
        max_position_size: float | None = None,
        max_total_exposure: float | None = None,
        max_drawdown_stop: float | None = None,
        daily_loss_limit: float | None = None,
        leverage: float = 1.0,
        maintenance_margin: float | None = None,
        fractional_shares: bool = True,
        factor_column: str | None = None,
        universe_provider: UniverseProvider | None = None,
        exchange_rate: pl.DataFrame | None = None,
    ):
        """
        Initialize the backtesting engine.

        Args:
            strategy: Strategy instance to backtest
            data: Polars DataFrame with price data OR dict mapping asset names to DataFrames
            initial_cash: Starting cash balance
            commission: Commission specification. Accepts a percentage float, a (fixed, percent) tuple,
                       or a CommissionModel instance
            slippage: Slippage rate as fraction or a SlippageModel instance
            price_columns: Dict mapping asset names to price columns
                          (default: auto-detected for dict input, {"asset": "close"} for single DataFrame)
            warmup: Number of bars to skip before executing strategy, or "auto" to automatically
                   detect when all indicators are ready (default "auto")
            order_delay: Number of bars to delay order execution (default 0, max realism is 1)
            borrow_rate: Annual borrow rate for short positions (e.g., 0.02 = 2% per year)
            bars_per_day: Number of bars in a trading day (used for day order expiry and borrow cost calculation)
            max_position_size: Maximum single position size as fraction of portfolio value (e.g., 0.5 = 50%)
            max_total_exposure: Maximum total exposure as fraction of portfolio value (e.g., 1.5 = 150%)
            max_drawdown_stop: Maximum drawdown before halting trading (e.g., 0.2 = 20%)
            daily_loss_limit: Maximum daily loss before halting trading for the day (e.g., 0.05 = 5%)
            leverage: Maximum leverage multiplier (e.g., 2.0 = 2x leverage). Default 1.0.
            maintenance_margin: Minimum margin ratio before margin call (e.g., 0.25 = 25%). Default None.
            fractional_shares: Whether to allow fractional share quantities (default True).
            factor_column: Optional column name for price adjustment factor. When set,
                          commissions are calculated on raw prices (adjusted_price / factor).
            universe_provider: Optional provider that filters tradeable symbols each bar.
                              When set, ``ctx.symbols`` contains only the filtered subset;
                              ``ctx.available_symbols`` contains all symbols with data.
            exchange_rate: Optional DataFrame with ``(timestamp, rate)`` columns for
                          quote-to-USD conversion. Rate is forward-filled to bar timestamps.
                          When provided, BacktestMetrics includes USD-denominated metrics.
        """
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.borrow_rate = borrow_rate
        self.bars_per_day = bars_per_day
        self.max_position_size = max_position_size
        self.max_total_exposure = max_total_exposure
        self.max_drawdown_stop = max_drawdown_stop
        self.daily_loss_limit = daily_loss_limit
        self.leverage = leverage
        self.maintenance_margin = maintenance_margin
        self.fractional_shares = fractional_shares
        self.factor_column = factor_column
        self.universe_provider = universe_provider
        self.exchange_rate = exchange_rate

        # Validate warmup parameter
        if isinstance(warmup, str):
            if warmup != "auto":
                raise ValueError(f"warmup must be an integer or 'auto', got '{warmup}'")
        elif not isinstance(warmup, int):
            raise ValueError(f"warmup must be an integer or 'auto', got {type(warmup)}")

        self.warmup = warmup
        self.order_delay = order_delay

        # --- Normalize input to long format ---
        # _long_data: long-format DataFrame with 'symbol' column (internal canonical form)
        # self.data / self.price_columns: preserved for backward-compat _calculate_results()

        if isinstance(data, dict):
            # Form B: dict[str, pl.DataFrame] -> tag each DF and concat vertically
            frames: list[pl.DataFrame] = []
            for asset_name, asset_df in data.items():
                sdf = standardize_dataframe(asset_df)
                sdf = sdf.with_columns(pl.lit(asset_name).alias("symbol"))
                frames.append(sdf)
            self._long_data: pl.DataFrame = pl.concat(frames, how="diagonal_relaxed")
            if "timestamp" in self._long_data.columns:
                self._long_data = self._long_data.sort(["timestamp", "symbol"])

            # Legacy wide-format for _calculate_results buy-hold
            self.data, auto_price_columns = merge_asset_dataframes(data)
            self.price_columns = price_columns if price_columns is not None else auto_price_columns
        else:
            sdf = standardize_dataframe(data)

            if "symbol" in sdf.columns:
                # Form C: already long format
                self._long_data = sdf
                if "timestamp" in self._long_data.columns:
                    self._long_data = self._long_data.sort(["timestamp", "symbol"])
                # Legacy: store as-is, detect first symbol's close for buy-hold
                self.data = sdf
                self.price_columns = price_columns if price_columns is not None else {"_first_": "close"}
            else:
                # Form A: single-asset DataFrame -> add symbol=DEFAULT_ASSET_NAME
                sdf = sdf.with_columns(pl.lit(DEFAULT_ASSET_NAME).alias("symbol"))
                self._long_data = sdf

                self.data = sdf
                if price_columns is None:
                    if "close" in sdf.columns:
                        self.price_columns = {DEFAULT_ASSET_NAME: "close"}
                    else:
                        numeric_cols = [
                            c for c in sdf.columns if sdf[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
                        ]
                        if numeric_cols:
                            self.price_columns = {DEFAULT_ASSET_NAME: numeric_cols[0]}
                        else:
                            raise ValueError("No price columns found in data")
                else:
                    self.price_columns = price_columns

        self.portfolio: Portfolio | None = None
        self.results: BacktestMetrics | None = None
        self.processed_data: pl.DataFrame | None = None

    def _calculate_auto_warmup(self, df: pl.DataFrame) -> int:
        """Calculate automatic warmup period by finding the first timestamp where all columns are non-null.

        For long-format data, this checks per-symbol rows and finds the first
        timestamp index where every symbol has all columns non-null.

        Args:
            df: Preprocessed long-format DataFrame with indicators.

        Returns:
            Integer warmup period (number of timestamp groups to skip).
        """
        skip_cols = {"timestamp", "date", "datetime", "time", "dt", "_index", "symbol"}
        cols_to_check = [col for col in df.columns if col not in skip_cols]

        if not cols_to_check:
            return 0

        # Add per-row validity flag
        df_with_valid = df.with_columns(
            pl.all_horizontal([pl.col(col).is_not_null() for col in cols_to_check]).alias("_all_valid")
        )

        # Detect timestamp column
        ts_col = None
        for candidate in ("timestamp", "date", "time", "_index"):
            if candidate in df.columns:
                ts_col = candidate
                break

        if ts_col is None or "symbol" not in df.columns:
            # Fallback: flat row-level check (single symbol or no grouping)
            all_valid = df_with_valid["_all_valid"]
            if not all_valid.any():
                n_unique = df[ts_col].n_unique() if ts_col else len(df)
                return max(0, n_unique - 1)
            return int(all_valid.arg_max())  # type: ignore[arg-type]

        # For multi-symbol: all symbols must be valid on a given timestamp
        ts_valid = df_with_valid.group_by(ts_col, maintain_order=True).agg(pl.col("_all_valid").all().alias("ts_valid"))
        valid_series = ts_valid["ts_valid"]
        if not valid_series.any():
            return max(0, len(ts_valid) - 1)
        return int(valid_series.arg_max())  # type: ignore[arg-type]

    def _clear_portfolio(self) -> None:
        """Eagerly release large lists inside the current portfolio.

        Clears equity curves, timestamps, orders, trades and positions so the
        memory is freed immediately rather than waiting for garbage collection.
        """
        if self.portfolio is not None:
            self.portfolio.equity_curve.clear()
            self.portfolio.timestamps.clear()
            self.portfolio.orders.clear()
            self.portfolio.trade_tracker.trades.clear()
            self.portfolio.trade_tracker.open_positions.clear()
            self.portfolio.positions.clear()

    def cleanup(self) -> None:
        """Release references to large internal objects for memory management.

        Call this after extracting results from a completed backtest to free
        memory occupied by the processed DataFrame, portfolio state, and
        intermediate data. Useful when running multiple sequential backtests
        in the same process to prevent memory exhaustion and segfaults.
        """
        self._clear_portfolio()
        self.portfolio = None
        self.processed_data = None
        self.results = None
        gc.collect()

    def __del__(self) -> None:
        """Release internal objects when the engine is garbage collected.

        Acts as a safety net for callers that do not invoke :meth:`cleanup`
        explicitly (e.g. when creating a new ``Engine`` per backtest inside a
        loop).  Does **not** call ``gc.collect()`` to avoid re-entrancy.
        """
        try:
            self._clear_portfolio()
            self.portfolio = None
            self.processed_data = None
            self.results = None
        except Exception:
            pass

    def run(self) -> BacktestMetrics:
        """Run the backtest simulation.

        Returns:
            BacktestMetrics with all performance metrics and trade data.
        """
        # Eagerly release large objects from any previous run so memory is
        # reclaimed before the new portfolio and processed data are allocated.
        self._clear_portfolio()
        self.portfolio = None
        self.processed_data = None
        self.results = None

        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_cash=self.initial_cash,
            commission=self.commission,
            slippage=self.slippage,
            order_delay=self.order_delay,
            borrow_rate=self.borrow_rate,
            bars_per_day=self.bars_per_day,
            max_position_size=self.max_position_size,
            max_total_exposure=self.max_total_exposure,
            max_drawdown_stop=self.max_drawdown_stop,
            daily_loss_limit=self.daily_loss_limit,
            leverage=self.leverage,
            maintenance_margin=self.maintenance_margin,
            fractional_shares=self.fractional_shares,
            factor_column=self.factor_column,
        )

        # Preprocess data using strategy (long-format)
        processed_data = self.strategy.preprocess(self._long_data)
        self.processed_data = processed_data

        # Ensure we have a timestamp column
        timestamp_col = None
        if "timestamp" in processed_data.columns:
            timestamp_col = "timestamp"
        elif "date" in processed_data.columns:
            timestamp_col = "date"
        elif "time" in processed_data.columns:
            timestamp_col = "time"
        else:
            # Use row index as timestamp — assign per-symbol group rank
            processed_data = processed_data.with_row_index("_index")
            timestamp_col = "_index"

        # Calculate warmup period if set to "auto"
        warmup_periods: int
        warmup_periods = self._calculate_auto_warmup(processed_data) if self.warmup == "auto" else self.warmup  # type: ignore

        # Call strategy initialization
        self.strategy.on_start(self.portfolio)

        # --- Main event loop: iterate by timestamp over long-format data ---
        # Group by timestamp to get all symbols' data per bar
        ts_col_series = processed_data[timestamp_col]
        unique_timestamps = ts_col_series.unique(maintain_order=True).sort()

        # Pre-partition data by timestamp for efficient lookup
        grouped = processed_data.partition_by(timestamp_col, as_dict=True, maintain_order=True)

        # Token lifecycle tracking
        first_seen_bar: dict[str, int] = {}
        bar_count: dict[str, int] = {}

        for idx, ts_value in enumerate(unique_timestamps):
            group_key = ts_value
            group_df = grouped.get(group_key)
            if group_df is None:
                # partition_by with single column returns scalar keys
                # Try tuple key as fallback
                group_df = grouped.get((group_key,))
            if group_df is None:
                continue

            current_prices: dict[str, float] = {}
            ohlc_data: dict[str, dict[str, float]] = {}
            bar_data: dict[str, dict[str, Any]] = {}

            for row_dict in group_df.iter_rows(named=True):
                sym = row_dict.get("symbol", DEFAULT_ASSET_NAME)
                close_val = row_dict.get("close")
                close_price = float(close_val) if close_val is not None else 0.0
                current_prices[sym] = close_price

                ohlc_data[sym] = {
                    "open": float(row_dict["open"]) if row_dict.get("open") is not None else close_price,
                    "high": float(row_dict["high"]) if row_dict.get("high") is not None else close_price,
                    "low": float(row_dict["low"]) if row_dict.get("low") is not None else close_price,
                    "close": close_price,
                }
                bar_data[sym] = row_dict

            current_timestamp = ts_value

            # Update token lifecycle tracking
            for sym in bar_data:
                if sym not in first_seen_bar:
                    first_seen_bar[sym] = idx
                bar_count[sym] = bar_count.get(sym, 0) + 1

            # Update factor data for commission calculation on raw prices
            if self.factor_column is not None:
                for sym, row_dict in bar_data.items():
                    factor_val = row_dict.get(self.factor_column)
                    if factor_val is not None:
                        self.portfolio._factors[sym] = float(factor_val)

            # Update portfolio with ALL current prices (including filtered-out symbols)
            self.portfolio._current_bar_data = bar_data
            self.portfolio.update_prices(current_prices, idx, ohlc_data, current_timestamp)

            # Determine tradeable universe
            available_symbols = list(bar_data.keys())
            if self.universe_provider is not None:
                universe_ctx = UniverseContext(
                    timestamp=current_timestamp,
                    bar_index=idx,
                    available_symbols=available_symbols,
                    bar_data=bar_data,
                    first_seen_bar=first_seen_bar,
                    bar_count=bar_count,
                )
                symbols_list = self.universe_provider.get_universe(universe_ctx)
            else:
                symbols_list = available_symbols

            # Create context for strategy
            row_accessor = _RowAccessor(bar_data, symbols_list)
            ctx = BacktestContext(
                timestamp=current_timestamp,
                bar_index=idx,
                portfolio=self.portfolio,
                symbols=symbols_list,
                data=bar_data,
                row=row_accessor,
                first_seen_bar=first_seen_bar,
                bar_count=bar_count,
                available_symbols=available_symbols,
            )

            # Call strategy logic (skip warmup period)
            if idx >= warmup_periods:
                self.strategy.next(ctx)

                # Record equity only after warmup to avoid diluting metrics
                self.portfolio.record_equity(ctx.timestamp)

        # Call strategy finalization
        self.strategy.on_finish(self.portfolio)

        # Calculate and return results
        self.results = self._calculate_results()
        return self.results

    def _compute_usd_metrics(self, equity_df: pl.DataFrame, metrics: dict[str, Any]) -> None:
        """Compute USD-denominated metrics by converting the equity curve.

        Joins exchange rate data to the equity curve via forward-fill asof join,
        then computes key metrics on the USD-converted equity.

        Args:
            equity_df: DataFrame with ``timestamp`` and ``equity`` columns.
            metrics: Metrics dict to update with USD fields.
        """
        from polarbt.metrics import calculate_metrics

        assert self.exchange_rate is not None
        rate_df = self.exchange_rate.sort("timestamp").select(
            pl.col("timestamp").alias("_rate_ts"),
            pl.col("rate"),
        )

        usd_df = equity_df.sort("timestamp").join_asof(
            rate_df,
            left_on="timestamp",
            right_on="_rate_ts",
            strategy="backward",
        )

        if usd_df["rate"].null_count() == usd_df.height:
            return

        usd_df = usd_df.with_columns(
            (pl.col("equity") * pl.col("rate").forward_fill()).alias("equity_usd"),
        ).drop_nulls("equity_usd")

        if usd_df.height == 0:
            return

        initial_equity_usd = float(usd_df["equity_usd"][0])
        usd_metrics = calculate_metrics(
            usd_df.select(pl.col("timestamp"), pl.col("equity_usd").alias("equity")),
            initial_equity_usd,
        )

        metrics["final_equity_usd"] = float(usd_df["equity_usd"][-1])
        metrics["total_return_usd"] = usd_metrics.get("total_return", 0.0)
        metrics["sharpe_ratio_usd"] = usd_metrics.get("sharpe_ratio", 0.0)
        metrics["max_drawdown_usd"] = usd_metrics.get("max_drawdown", 0.0)

    def _calculate_results(self) -> BacktestMetrics:
        """Calculate backtest metrics.

        Returns:
            BacktestMetrics with all performance data.
        """
        from polarbt.metrics import calculate_metrics

        if not self.portfolio:
            return BacktestMetrics()

        # Create equity curve DataFrame
        equity_df = pl.DataFrame(
            {
                "timestamp": self.portfolio.timestamps,
                "equity": self.portfolio.equity_curve,
            },
            strict=False,  # Allow mixed types
        )

        # Calculate base metrics (still a dict from calculate_metrics)
        metrics = calculate_metrics(equity_df, self.initial_cash)

        # Portfolio info
        metrics["final_equity"] = self.portfolio.get_value()
        metrics["equity_peak"] = float(equity_df["equity"].max()) if len(equity_df) > 0 else self.initial_cash  # type: ignore[arg-type]
        metrics["final_positions"] = dict(self.portfolio.positions)
        metrics["final_cash"] = self.portfolio.cash

        # Trade information
        trades_df = self.portfolio.get_trades()
        trade_stats = self.portfolio.get_trade_stats()
        metrics["trades"] = trades_df
        metrics["win_rate"] = trade_stats.win_rate

        # Return (Ann.) — same as CAGR, kept as explicit alias
        metrics["return_annualized"] = metrics.get("cagr", 0.0)

        # Buy & Hold return: compare first vs last close price of first symbol
        metrics["buy_hold_return"] = 0.0
        if "symbol" in self._long_data.columns and "close" in self._long_data.columns:
            first_sym = self._long_data["symbol"][0]
            sym_prices = self._long_data.filter(pl.col("symbol") == first_sym)["close"].drop_nulls()
            if len(sym_prices) >= 2:
                first_price = float(sym_prices[0])
                last_price = float(sym_prices[-1])
                if first_price != 0:
                    metrics["buy_hold_return"] = (last_price - first_price) / first_price
        elif "close" in self._long_data.columns:
            prices = self._long_data["close"].drop_nulls()
            if len(prices) >= 2:
                first_price = float(prices[0])
                last_price = float(prices[-1])
                if first_price != 0:
                    metrics["buy_hold_return"] = (last_price - first_price) / first_price

        # Trade-level detailed metrics from the trades DataFrame
        if len(trades_df) > 0:
            pct_col = trades_df["return_pct"]
            bars_col = trades_df["bars_held"]

            metrics["best_trade_pct"] = float(pct_col.max())  # type: ignore[arg-type]
            metrics["worst_trade_pct"] = float(pct_col.min())  # type: ignore[arg-type]
            metrics["avg_trade_pct"] = float(pct_col.mean())  # type: ignore[arg-type]
            metrics["max_trade_duration"] = float(bars_col.max())  # type: ignore[arg-type]
            metrics["avg_trade_duration"] = float(bars_col.mean())  # type: ignore[arg-type]

            from polarbt.metrics import trade_level_metrics

            _tlm = trade_level_metrics(self.portfolio.trade_tracker.trades)
            metrics["expectancy"] = _tlm["expectancy"]
            metrics["sqn"] = _tlm["sqn"]
            metrics["kelly_criterion"] = _tlm["kelly_criterion"]
        else:
            metrics["best_trade_pct"] = 0.0
            metrics["worst_trade_pct"] = 0.0
            metrics["avg_trade_pct"] = 0.0
            metrics["max_trade_duration"] = 0.0
            metrics["avg_trade_duration"] = 0.0
            metrics["expectancy"] = 0.0
            metrics["sqn"] = 0.0
            metrics["kelly_criterion"] = 0.0

        # Liquidity metrics (Feature 4) — only if relevant columns exist
        from polarbt.metrics import liquidity_metrics

        liq = liquidity_metrics(trades_df, self.data)
        metrics.update({k: v for k, v in liq.items() if v is not None})

        # USD-denominated metrics (Phase 5) — when exchange_rate is provided
        if self.exchange_rate is not None and len(equity_df) > 0:
            self._compute_usd_metrics(equity_df, metrics)

        return _backtest_metrics_from_dict(metrics, trade_stats)
