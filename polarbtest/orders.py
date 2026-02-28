"""
Order management system for backtesting.

This module provides order types, statuses, and the Order dataclass for
tracking order lifecycle in backtesting simulations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OrderType(Enum):
    """Order execution types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order lifecycle states."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """
    Represents a trading order with complete lifecycle tracking.

    Attributes:
        order_id: Unique identifier
        asset: Asset symbol/name
        size: Quantity to trade (positive=buy, negative=sell)
        order_type: Type of order (MARKET, LIMIT, STOP, STOP_LIMIT)
        status: Current order status

        limit_price: Limit price for LIMIT orders
        stop_price: Stop trigger price for STOP orders

        created_bar: Bar index when order was created
        created_timestamp: Timestamp when order was created
        filled_bar: Bar index when order was filled
        filled_timestamp: Timestamp when order was filled
        filled_price: Actual execution price
        filled_size: Actual executed quantity

        valid_until: Bar index when order expires (None = GTC)
        expiry_date: Date when order expires (for day orders with timestamp-based expiry)
        parent_order: Reference to parent order (for OCO, bracket)
        child_orders: List of child order IDs (SL/TP attached to this)

        commission_paid: Commission paid on this order
        slippage_cost: Slippage cost (informational)

        tags: User-defined tags for categorization
        notes: Optional notes

    Example:
        # Market order
        order = Order(
            order_id="1",
            asset="BTC",
            size=0.5,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING
        )

        # Limit order
        order = Order(
            order_id="2",
            asset="BTC",
            size=1.0,
            order_type=OrderType.LIMIT,
            status=OrderStatus.PENDING,
            limit_price=50000
        )
    """

    order_id: str
    asset: str
    size: float
    order_type: OrderType
    status: OrderStatus

    limit_price: float | None = None
    stop_price: float | None = None

    created_bar: int = 0
    created_timestamp: Any = None
    filled_bar: int | None = None
    filled_timestamp: Any = None
    filled_price: float | None = None
    filled_size: float = 0.0

    valid_until: int | None = None
    expiry_date: Any = None  # For day-based expiry (date when order expires)
    parent_order: str | None = None
    child_orders: list[str] = field(default_factory=list)

    commission_paid: float = 0.0
    slippage_cost: float = 0.0

    # STOP_LIMIT: tracks whether the stop has been triggered (converts to limit)
    triggered: bool = False

    # Bracket order metadata: stored on entry order, applied when it fills
    bracket_stop_loss: float | None = None
    bracket_take_profit: float | None = None

    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    def is_active(self) -> bool:
        """Check if order is still active (pending or partial)."""
        return self.status in (OrderStatus.PENDING, OrderStatus.PARTIAL)

    def can_be_cancelled(self) -> bool:
        """Check if order can be cancelled."""
        return self.is_active()

    def is_buy(self) -> bool:
        """Check if this is a buy order."""
        return self.size > 0

    def is_sell(self) -> bool:
        """Check if this is a sell order."""
        return self.size < 0

    def mark_filled(self, bar: int, timestamp: Any, price: float, commission: float, slippage: float) -> None:
        """
        Mark order as filled.

        Args:
            bar: Bar index when filled
            timestamp: Timestamp when filled
            price: Execution price
            commission: Commission paid
            slippage: Slippage cost
        """
        self.status = OrderStatus.FILLED
        self.filled_bar = bar
        self.filled_timestamp = timestamp
        self.filled_price = price
        self.filled_size = self.size
        self.commission_paid = commission
        self.slippage_cost = slippage

    def mark_cancelled(self) -> None:
        """Mark order as cancelled."""
        self.status = OrderStatus.CANCELLED

    def mark_rejected(self) -> None:
        """Mark order as rejected."""
        self.status = OrderStatus.REJECTED

    def mark_expired(self) -> None:
        """Mark order as expired."""
        self.status = OrderStatus.EXPIRED
