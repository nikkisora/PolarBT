"""Tests for order management system."""

from polarbt.orders import Order, OrderStatus, OrderType


class TestOrderType:
    """Test OrderType enum."""

    def test_order_types_exist(self):
        """Test all order types are defined."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"


class TestOrderStatus:
    """Test OrderStatus enum."""

    def test_order_statuses_exist(self):
        """Test all order statuses are defined."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.PARTIAL.value == "partial"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired"


class TestOrder:
    """Test Order dataclass."""

    def test_order_creation(self):
        """Test creating an order."""
        order = Order(order_id="1", asset="BTC", size=1.0, order_type=OrderType.MARKET, status=OrderStatus.PENDING)

        assert order.order_id == "1"
        assert order.asset == "BTC"
        assert order.size == 1.0
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING

    def test_limit_order_creation(self):
        """Test creating a limit order."""
        order = Order(
            order_id="2",
            asset="ETH",
            size=10.0,
            order_type=OrderType.LIMIT,
            status=OrderStatus.PENDING,
            limit_price=2000.0,
        )

        assert order.limit_price == 2000.0
        assert order.order_type == OrderType.LIMIT

    def test_is_filled(self):
        """Test is_filled method."""
        order = Order(order_id="1", asset="BTC", size=1.0, order_type=OrderType.MARKET, status=OrderStatus.PENDING)

        assert not order.is_filled()

        order.status = OrderStatus.FILLED
        assert order.is_filled()

    def test_is_active(self):
        """Test is_active method."""
        order = Order(order_id="1", asset="BTC", size=1.0, order_type=OrderType.MARKET, status=OrderStatus.PENDING)

        assert order.is_active()

        order.status = OrderStatus.FILLED
        assert not order.is_active()

        order.status = OrderStatus.CANCELLED
        assert not order.is_active()

        order.status = OrderStatus.PARTIAL
        assert order.is_active()

    def test_can_be_cancelled(self):
        """Test can_be_cancelled method."""
        order = Order(order_id="1", asset="BTC", size=1.0, order_type=OrderType.MARKET, status=OrderStatus.PENDING)

        assert order.can_be_cancelled()

        order.status = OrderStatus.FILLED
        assert not order.can_be_cancelled()

    def test_is_buy(self):
        """Test is_buy method."""
        buy_order = Order(order_id="1", asset="BTC", size=1.0, order_type=OrderType.MARKET, status=OrderStatus.PENDING)

        sell_order = Order(
            order_id="2", asset="BTC", size=-1.0, order_type=OrderType.MARKET, status=OrderStatus.PENDING
        )

        assert buy_order.is_buy()
        assert not sell_order.is_buy()

    def test_is_sell(self):
        """Test is_sell method."""
        buy_order = Order(order_id="1", asset="BTC", size=1.0, order_type=OrderType.MARKET, status=OrderStatus.PENDING)

        sell_order = Order(
            order_id="2", asset="BTC", size=-1.0, order_type=OrderType.MARKET, status=OrderStatus.PENDING
        )

        assert not buy_order.is_sell()
        assert sell_order.is_sell()

    def test_mark_filled(self):
        """Test mark_filled method."""
        order = Order(order_id="1", asset="BTC", size=1.0, order_type=OrderType.MARKET, status=OrderStatus.PENDING)

        order.mark_filled(bar=10, timestamp=1000, price=50000.0, commission=50.0, slippage=25.0)

        assert order.is_filled()
        assert order.filled_bar == 10
        assert order.filled_timestamp == 1000
        assert order.filled_price == 50000.0
        assert order.filled_size == 1.0
        assert order.commission_paid == 50.0
        assert order.slippage_cost == 25.0

    def test_mark_cancelled(self):
        """Test mark_cancelled method."""
        order = Order(order_id="1", asset="BTC", size=1.0, order_type=OrderType.MARKET, status=OrderStatus.PENDING)

        order.mark_cancelled()
        assert order.status == OrderStatus.CANCELLED

    def test_mark_rejected(self):
        """Test mark_rejected method."""
        order = Order(order_id="1", asset="BTC", size=1.0, order_type=OrderType.MARKET, status=OrderStatus.PENDING)

        order.mark_rejected()
        assert order.status == OrderStatus.REJECTED

    def test_mark_expired(self):
        """Test mark_expired method."""
        order = Order(order_id="1", asset="BTC", size=1.0, order_type=OrderType.MARKET, status=OrderStatus.PENDING)

        order.mark_expired()
        assert order.status == OrderStatus.EXPIRED

    def test_order_with_tags(self):
        """Test order with tags."""
        order = Order(
            order_id="1",
            asset="BTC",
            size=1.0,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            tags=["momentum", "long"],
        )

        assert "momentum" in order.tags
        assert "long" in order.tags

    def test_order_with_parent_child(self):
        """Test order with parent/child relationships."""
        parent_order = Order(
            order_id="1", asset="BTC", size=1.0, order_type=OrderType.MARKET, status=OrderStatus.FILLED
        )

        sl_order = Order(
            order_id="2",
            asset="BTC",
            size=-1.0,
            order_type=OrderType.STOP,
            status=OrderStatus.PENDING,
            parent_order="1",
        )

        parent_order.child_orders.append("2")

        assert sl_order.parent_order == "1"
        assert "2" in parent_order.child_orders
