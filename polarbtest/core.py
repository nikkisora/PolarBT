"""
Core backtesting engine and components.

This module provides the fundamental building blocks for backtesting:
- Portfolio: Manages positions and cash
- Strategy: Base class for defining trading strategies
- Engine: Executes the backtest simulation
- BacktestContext: Data container passed to strategy.next()
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import polars as pl

from polarbtest.orders import Order, OrderStatus, OrderType
from polarbtest.trades import TradeTracker


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


def standardize_dataframe(
    df: pl.DataFrame,
    timestamp_col: str | None = None,
    auto_detect: bool = True,
) -> pl.DataFrame:
    """
    Standardize a DataFrame by renaming common timestamp column names to 'timestamp'.

    Args:
        df: Input DataFrame
        timestamp_col: Specific timestamp column to rename (if None, auto-detect)
        auto_detect: Auto-detect common timestamp column names (default True)

    Returns:
        DataFrame with standardized column names

    Example:
        # Auto-detect and rename
        df = standardize_dataframe(df)

        # Specify column explicitly
        df = standardize_dataframe(df, timestamp_col="datetime")
    """
    df = df.clone()

    # If timestamp column already exists, we're done
    if "timestamp" in df.columns:
        return df

    # If specific column provided, rename it
    if timestamp_col and timestamp_col in df.columns:
        return df.rename({timestamp_col: "timestamp"})

    # Auto-detect common timestamp column names
    if auto_detect:
        common_names = ["date", "datetime", "time", "dt", "Date", "DateTime", "Time"]
        for name in common_names:
            if name in df.columns:
                return df.rename({name: "timestamp"})

    # No timestamp column found or needed
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


@dataclass
class BacktestContext:
    """
    Context object passed to Strategy.next() on each bar.

    Attributes:
        timestamp: Current timestamp
        row: Dictionary containing current bar data (prices, indicators, etc.)
        portfolio: Reference to the Portfolio instance
        bar_index: Current bar index in the dataset
    """

    timestamp: Any
    row: dict[str, Any]
    portfolio: "Portfolio"
    bar_index: int


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
        commission: float | tuple[float, float] = 0.0,
        slippage: float = 0.0,
        order_delay: int = 0,
        bars_per_day: int | None = None,
        borrow_rate: float = 0.0,
    ):
        """
        Initialize a new portfolio.

        Args:
            initial_cash: Starting cash balance
            commission: Commission as a percentage (e.g., 0.001 = 0.1%) or tuple of (fixed_commission, percent_commission)
                       For example: 0.001 means 0.1% per trade, (5.0, 0.001) means $5 + 0.1% per trade
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
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash

        # Parse commission format
        if isinstance(commission, tuple):
            self.commission_fixed = commission[0]
            self.commission_percent = commission[1]
        else:
            self.commission_fixed = 0.0
            self.commission_percent = commission

        self.slippage = slippage
        self.order_delay = order_delay
        self.bars_per_day = bars_per_day
        self.borrow_rate = borrow_rate

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

        # Order management
        self.orders: dict[str, Order] = {}
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

        # Update MAE/MFE for open positions
        for asset in self.trade_tracker.open_positions:
            if asset in prices:
                self.trade_tracker.update_mae_mfe(asset, prices[asset])

        # Check and execute stop-loss orders first
        self._check_stop_losses()

        # Check and execute take-profit orders
        self._check_take_profits()

        # Update and check trailing stops (combined to avoid intra-bar issues)
        self._update_and_check_trailing_stops()

        # Check and expire orders that have exceeded their valid_until time
        self._check_order_expiry()

        # Execute pending orders
        self._execute_pending_orders()

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

    def get_value(self) -> float:
        """
        Calculate total portfolio value (cash + positions).

        Returns:
            Total portfolio value
        """
        positions_value = sum(qty * self._current_prices.get(asset, 0.0) for asset, qty in self.positions.items())
        return self.cash + positions_value

    def get_position(self, asset: str) -> float:
        """
        Get current position size for an asset.

        Args:
            asset: Asset name

        Returns:
            Position quantity (can be fractional)
        """
        return self.positions.get(asset, 0.0)

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
                # Long position stop hit - close position
                self.close_position(asset)
                # Remove stop-loss if it still exists (may have been removed by close_position cleanup)
                if asset in self._stop_losses:
                    del self._stop_losses[asset]
            elif position_size < 0 and high >= stop_price:
                # Short position stop hit - close position
                self.close_position(asset)
                # Remove stop-loss if it still exists
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
                # Long position take-profit hit - close position
                self.close_position(asset)
                # Remove take-profit if it still exists (may have been removed by close_position cleanup)
                if asset in self._take_profits:
                    del self._take_profits[asset]
            elif position_size < 0 and low <= target_price:
                # Short position take-profit hit - close position
                self.close_position(asset)
                # Remove take-profit if it still exists
                if asset in self._take_profits:
                    del self._take_profits[asset]

    def _execute_pending_orders(self) -> None:
        """Execute orders that are due."""
        # Get all pending orders that should execute this bar
        orders_to_execute = [
            order
            for order in self.orders.values()
            if order.is_active() and order.created_bar + self.order_delay <= self._current_bar
        ]

        for order in orders_to_execute:
            self._try_execute_order(order)

    def _check_order_expiry(self) -> None:
        """Check and expire orders that have passed their valid_until time or expiry_date."""
        current_date = _extract_date(self._current_timestamp)

        for order in self.orders.values():
            if not order.is_active():
                continue

            # Check date-based expiry first (if order has expiry_date set)
            if order.expiry_date is not None and current_date is not None:
                if current_date > order.expiry_date:
                    order.mark_expired()
                    continue

            # Check bar-based expiry
            if order.valid_until is not None and self._current_bar > order.valid_until:
                # Skip if this order uses date-based expiry (valid_until was set to large number)
                if order.expiry_date is None or order.valid_until < 999999:
                    order.mark_expired()

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
        if order.order_type == OrderType.STOP and order.stop_price is not None:
            price: float = order.stop_price
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
        execution_price = price * (1 + self.slippage) if order.is_buy() else price * (1 - self.slippage)

        # Calculate costs
        gross_cost = abs(order.size) * execution_price
        commission_cost = self.commission_fixed + (gross_cost * self.commission_percent)

        # Track old position for trade tracking
        old_position = self.get_position(order.asset)

        # Determine if this is opening/increasing a short position
        current_position = self.positions.get(order.asset, 0.0)

        if order.is_buy():
            if current_position < 0:
                # Covering a short position (partially or fully)
                cover_size = min(abs(order.size), abs(current_position))
                new_long_size = abs(order.size) - cover_size

                # Cost to cover: buy back shares at execution_price
                cover_cost = cover_size * execution_price + commission_cost
                # Cost to open new long (if any)
                new_long_cost = new_long_size * execution_price
                total_cost = cover_cost + new_long_cost

                if total_cost > self.cash:
                    order.mark_rejected()
                    return False
                self.cash -= total_cost
            else:
                # Opening or increasing a long position
                total_cost = gross_cost + commission_cost
                if total_cost > self.cash:
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
                # Proceeds from closing long
                long_close_proceeds = current_position * execution_price - commission_cost
                # Short sale proceeds for the remainder
                short_size = sell_size - current_position
                short_proceeds = short_size * execution_price
                self.cash += long_close_proceeds + short_proceeds
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
        if quantity == 0:
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

    def order_target_percent(self, asset: str, target_percent: float, limit_price: float | None = None) -> str | None:
        """
        Order to reach a target percentage of portfolio value.

        Args:
            asset: Asset name
            target_percent: Desired position as fraction of portfolio (e.g., 0.5 = 50%)
            limit_price: Optional limit price

        Returns:
            order_id if order was placed, None otherwise
        """
        price = limit_price if limit_price is not None else self._current_prices.get(asset)
        if price is None or price <= 0:
            return None

        portfolio_value = self.get_value()
        current_position = self.get_position(asset)
        current_value = current_position * price

        # Calculate target value
        target_value = portfolio_value * target_percent

        # Calculate the difference we need to trade
        value_delta = target_value - current_value

        if abs(value_delta) < 1e-6:  # Already at target
            return None

        # Account for fees when calculating target quantity
        # When buying: we pay slippage + commission on the gross cost
        # When selling: we receive slippage - commission on the gross proceeds
        if value_delta > 0:  # Buying
            # For buying: total_cost = quantity * execution_price * (1 + percent_commission) + fixed_commission
            # We need: total_cost = value_delta
            # Solving: quantity = (value_delta - fixed_commission) / (execution_price * (1 + percent_commission))
            execution_price = price * (1 + self.slippage)
            cost_multiplier = 1 + self.commission_percent
            quantity_to_buy = (value_delta - self.commission_fixed) / (execution_price * cost_multiplier)
            target_quantity = current_position + quantity_to_buy
        else:  # Selling
            # For selling: net_proceeds = quantity * execution_price * (1 - percent_commission) - fixed_commission
            # We need: net_proceeds = abs(value_delta)
            # Solving: quantity = (abs(value_delta) + fixed_commission) / (execution_price * (1 - percent_commission))
            execution_price = price * (1 - self.slippage)
            proceeds_multiplier = 1 - self.commission_percent
            quantity_to_sell = (abs(value_delta) + self.commission_fixed) / (execution_price * proceeds_multiplier)
            target_quantity = current_position - quantity_to_sell

        return self.order_target(asset, target_quantity, limit_price)

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
            current_bar_in_day = self._current_bar % self.bars_per_day
            # Calculate bars remaining until end of current day
            # Example: 390 - 60 - 1 = 329 bars until end of day
            bars_until_eod = self.bars_per_day - current_bar_in_day - 1
            # Order expires at end of current trading day
            order.valid_until = self._current_bar + bars_until_eod
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
            if trail_pct is not None:
                stop_price = current_price * (1 - trail_pct)
            else:
                stop_price = current_price - trail_amount  # type: ignore
        else:
            # Short position - track lowest price
            highest_price = current_price  # We'll use this field for lowest in shorts
            if trail_pct is not None:
                stop_price = current_price * (1 + trail_pct)
            else:
                stop_price = current_price + trail_amount  # type: ignore

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
                # Stop was hit - close position and remove trailing stop
                self.close_position(asset)
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
                elif position_size < 0:
                    # Short position - update if LOW makes new low
                    # Note: highest_price stores the lowest price for shorts
                    if low < trail_info["highest_price"]:
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
            if quantity > 0:  # Long position
                stop_loss = entry_price * (1 - stop_loss_pct)
            else:  # Short position
                stop_loss = entry_price * (1 + stop_loss_pct)

        if take_profit is None and take_profit_pct is not None:
            if quantity > 0:  # Long position
                take_profit = entry_price * (1 + take_profit_pct)
            else:  # Short position
                take_profit = entry_price * (1 - take_profit_pct)

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

    def get_trade_stats(self) -> dict[str, float]:
        """
        Get aggregate trade statistics.

        Returns:
            Dictionary with win_rate, avg_win, avg_loss, profit_factor, etc.
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


class Strategy(ABC):
    """
    Base class for trading strategies.

    Subclasses should implement:
    - preprocess(): Vectorized feature engineering using Polars
    - next(): Event-driven logic called on each bar
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
                from polarbtest import indicators as ind
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
        commission: float | tuple[float, float] = 0.0,
        slippage: float = 0.0,
        price_columns: dict[str, str] | None = None,
        warmup: int | str = "auto",
        order_delay: int = 0,
        borrow_rate: float = 0.0,
        bars_per_day: int | None = None,
    ):
        """
        Initialize the backtesting engine.

        Args:
            strategy: Strategy instance to backtest
            data: Polars DataFrame with price data OR dict mapping asset names to DataFrames
            initial_cash: Starting cash balance
            commission: Commission as a percentage (e.g., 0.001 = 0.1%) or tuple of (fixed_commission, percent_commission)
                       For example: 0.001 means 0.1% per trade, (5.0, 0.001) means $5 + 0.1% per trade
            slippage: Slippage rate as fraction
            price_columns: Dict mapping asset names to price columns
                          (default: auto-detected for dict input, {"asset": "close"} for single DataFrame)
            warmup: Number of bars to skip before executing strategy, or "auto" to automatically
                   detect when all indicators are ready (default "auto")
            order_delay: Number of bars to delay order execution (default 0, max realism is 1)
            borrow_rate: Annual borrow rate for short positions (e.g., 0.02 = 2% per year)
            bars_per_day: Number of bars in a trading day (used for day order expiry and borrow cost calculation)
        """
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.borrow_rate = borrow_rate
        self.bars_per_day = bars_per_day

        # Validate warmup parameter
        if isinstance(warmup, str):
            if warmup != "auto":
                raise ValueError(f"warmup must be an integer or 'auto', got '{warmup}'")
        elif not isinstance(warmup, int):
            raise ValueError(f"warmup must be an integer or 'auto', got {type(warmup)}")

        self.warmup = warmup
        self.order_delay = order_delay

        # Handle dict of dataframes or single dataframe
        if isinstance(data, dict):
            # Merge multiple asset dataframes
            self.data, auto_price_columns = merge_asset_dataframes(data)
            # Use auto-detected price columns if not specified
            if price_columns is None:
                self.price_columns = auto_price_columns
            else:
                self.price_columns = price_columns
        else:
            # Single dataframe - standardize it
            self.data = standardize_dataframe(data)

            # If no price columns specified, assume single asset with "close" column
            if price_columns is None:
                # Try to detect available price column
                if "close" in self.data.columns:
                    self.price_columns = {"asset": "close"}
                else:
                    # Find first numeric column
                    numeric_cols = [
                        c
                        for c in self.data.columns
                        if self.data[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
                    ]
                    if numeric_cols:
                        self.price_columns = {"asset": numeric_cols[0]}
                    else:
                        raise ValueError("No price columns found in data")
            else:
                self.price_columns = price_columns

        self.portfolio: Portfolio | None = None
        self.results: dict[str, Any] | None = None

    def _calculate_auto_warmup(self, df: pl.DataFrame) -> int:
        """
        Calculate automatic warmup period by finding the first row where all columns are non-null.

        This method finds the first bar where all indicators and data are ready.
        Excludes timestamp columns from the check.

        Args:
            df: Preprocessed DataFrame with indicators

        Returns:
            Integer warmup period (number of bars to skip before executing strategy)

        Example:
            If indicators need 20 bars to warm up, this will return 20.
        """
        # Get columns to check (exclude timestamp-related columns)
        timestamp_cols = {"timestamp", "date", "datetime", "time", "dt", "_index"}
        cols_to_check = [col for col in df.columns if col not in timestamp_cols]

        if not cols_to_check:
            # No columns to check, no warmup needed
            return 0

        # Find the first row where all columns are non-null
        # Create a boolean column that is True when all cols_to_check are non-null
        all_non_null = df.select(
            pl.all_horizontal([pl.col(col).is_not_null() for col in cols_to_check]).alias("all_valid")
        )

        # Find the index of the first True value
        for idx, row in enumerate(all_non_null.iter_rows()):
            if row[0]:  # First (and only) column is "all_valid"
                return idx

        # If no row has all non-null values, return length - 1 (skip all but last)
        return max(0, len(df) - 1)

    def run(self) -> dict[str, Any]:
        """
        Run the backtest simulation.

        Returns:
            Dictionary containing backtest results and metrics
        """
        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_cash=self.initial_cash,
            commission=self.commission,
            slippage=self.slippage,
            order_delay=self.order_delay,
            borrow_rate=self.borrow_rate,
            bars_per_day=self.bars_per_day,
        )

        # Preprocess data using strategy
        processed_data = self.strategy.preprocess(self.data)

        # Calculate warmup period if set to "auto"
        warmup_periods: int
        warmup_periods = self._calculate_auto_warmup(processed_data) if self.warmup == "auto" else self.warmup  # type: ignore

        # Ensure we have a timestamp column
        timestamp_col = None
        if "timestamp" in processed_data.columns:
            timestamp_col = "timestamp"
        elif "date" in processed_data.columns:
            timestamp_col = "date"
        elif "time" in processed_data.columns:
            timestamp_col = "time"
        else:
            # Use index as timestamp
            processed_data = processed_data.with_row_count("_index")
            timestamp_col = "_index"

        # Call strategy initialization
        self.strategy.on_start(self.portfolio)

        # Main event loop - iterate through bars
        for idx, row_dict in enumerate(processed_data.iter_rows(named=True)):
            # Extract current prices for all assets
            current_prices: dict[str, float] = {
                asset: float(row_dict.get(price_col, 0.0)) if row_dict.get(price_col) is not None else 0.0
                for asset, price_col in self.price_columns.items()
            }

            # Extract OHLC data for all assets
            ohlc_data: dict[str, dict[str, float]] = {}
            for asset in self.price_columns:
                # Try to find OHLC columns for this asset
                # Support both prefixed (BTC_open) and unprefixed (open) formats
                if asset == "asset":
                    # Single asset case - use unprefixed columns
                    ohlc_data[asset] = {
                        "open": float(row_dict.get("open", current_prices[asset]))
                        if row_dict.get("open") is not None
                        else current_prices[asset],
                        "high": float(row_dict.get("high", current_prices[asset]))
                        if row_dict.get("high") is not None
                        else current_prices[asset],
                        "low": float(row_dict.get("low", current_prices[asset]))
                        if row_dict.get("low") is not None
                        else current_prices[asset],
                        "close": current_prices[asset],
                    }
                else:
                    # Multi-asset case - try prefixed columns
                    ohlc_data[asset] = {
                        "open": float(row_dict.get(f"{asset}_open", current_prices[asset]))
                        if row_dict.get(f"{asset}_open") is not None
                        else current_prices[asset],
                        "high": float(row_dict.get(f"{asset}_high", current_prices[asset]))
                        if row_dict.get(f"{asset}_high") is not None
                        else current_prices[asset],
                        "low": float(row_dict.get(f"{asset}_low", current_prices[asset]))
                        if row_dict.get(f"{asset}_low") is not None
                        else current_prices[asset],
                        "close": current_prices[asset],
                    }

            # Get timestamp
            current_timestamp = row_dict.get(timestamp_col)

            # Update portfolio with current prices and OHLC data
            self.portfolio.update_prices(current_prices, idx, ohlc_data, current_timestamp)

            # Create context for strategy
            ctx = BacktestContext(
                timestamp=current_timestamp,
                row=row_dict,
                portfolio=self.portfolio,
                bar_index=idx,
            )

            # Call strategy logic (skip warmup period)
            if idx >= warmup_periods:
                self.strategy.next(ctx)

            # Record equity for metrics
            self.portfolio.record_equity(ctx.timestamp)

        # Call strategy finalization
        self.strategy.on_finish(self.portfolio)

        # Calculate and return results
        self.results = self._calculate_results()
        return self.results

    def _calculate_results(self) -> dict[str, Any]:
        """
        Calculate backtest metrics.

        Returns:
            Dictionary with performance metrics
        """
        from polarbtest.metrics import calculate_metrics

        if not self.portfolio:
            return {}

        # Create equity curve DataFrame
        equity_df = pl.DataFrame(
            {
                "timestamp": self.portfolio.timestamps,
                "equity": self.portfolio.equity_curve,
            },
            strict=False,  # Allow mixed types
        )

        # Calculate metrics
        metrics = calculate_metrics(equity_df, self.initial_cash)

        # Add portfolio info
        metrics["final_equity"] = self.portfolio.get_value()
        metrics["final_positions"] = dict(self.portfolio.positions)
        metrics["final_cash"] = self.portfolio.cash

        # Add trade information
        metrics["trades"] = self.portfolio.get_trades()
        metrics["trade_stats"] = self.portfolio.get_trade_stats()

        return metrics
