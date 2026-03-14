"""Slippage models for realistic execution price simulation.

Provides a protocol and built-in implementations for computing slippage-adjusted
execution prices. Models range from simple flat-percentage to AMM-aware
constant-product slippage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SlippageModel(ABC):
    """Base class for slippage models.

    Subclasses implement ``calculate()`` to return the slippage-adjusted
    execution price given a base price, order size, and optional context.
    """

    @abstractmethod
    def calculate(self, price: float, size: float, is_buy: bool, context: dict[str, Any] | None = None) -> float:
        """Return the slippage-adjusted execution price.

        Args:
            price: Base execution price (e.g. close, limit price, or stop trigger).
            size: Absolute order quantity.
            is_buy: True for buy orders, False for sell orders.
            context: Optional per-symbol bar data dict. Used by models that
                need market microstructure data (e.g. pool reserve).

        Returns:
            Adjusted execution price after slippage.
        """
        ...

    def estimate_rate(self, price: float, size: float, is_buy: bool, context: dict[str, Any] | None = None) -> float:
        """Return the effective slippage rate as a fraction.

        Used by fee-adjustment calculations that need to estimate total cost
        before executing. Default implementation derives rate from ``calculate()``.

        Args:
            price: Base price.
            size: Absolute order quantity.
            is_buy: True for buy, False for sell.
            context: Optional bar data context.

        Returns:
            Slippage rate as a non-negative fraction (e.g. 0.01 = 1%).
        """
        if price <= 0:
            return 0.0
        adjusted = self.calculate(price, size, is_buy, context)
        return abs(adjusted - price) / price


class FlatSlippage(SlippageModel):
    """Fixed percentage slippage applied uniformly to all orders.

    Buys execute at ``price * (1 + pct)``, sells at ``price * (1 - pct)``.
    This matches the Engine's original ``slippage: float`` behavior.

    Args:
        pct: Slippage rate as fraction (e.g. 0.001 = 0.1%).
    """

    def __init__(self, pct: float = 0.0) -> None:
        if pct < 0:
            raise ValueError(f"pct must be non-negative, got {pct}")
        self.pct = pct

    def calculate(self, price: float, size: float, is_buy: bool, context: dict[str, Any] | None = None) -> float:
        """Apply flat percentage slippage."""
        return price * (1 + self.pct) if is_buy else price * (1 - self.pct)

    def estimate_rate(self, price: float, size: float, is_buy: bool, context: dict[str, Any] | None = None) -> float:
        """Return the fixed slippage rate."""
        return self.pct


class AMMSlippage(SlippageModel):
    """Constant-product AMM slippage model.

    For a constant-product AMM (x * y = k), the price impact of a trade is::

        slippage_rate = trade_value / (pool_reserve + trade_value)

    where ``trade_value = size * price`` (the quote-currency value of the trade).

    Requires ``pool_reserve`` (or a custom key) in the ``context`` dict.
    Falls back to zero slippage when reserve data is unavailable.

    Args:
        reserve_key: Key in the context dict for pool reserve value.
            Defaults to ``"pool_reserve_last"``.
        min_slippage: Minimum slippage rate floor (e.g. for network fees).
            Defaults to 0.0.
    """

    def __init__(self, reserve_key: str = "pool_reserve_last", min_slippage: float = 0.0) -> None:
        self.reserve_key = reserve_key
        self.min_slippage = min_slippage

    def calculate(self, price: float, size: float, is_buy: bool, context: dict[str, Any] | None = None) -> float:
        """Apply constant-product AMM slippage."""
        rate = self._compute_rate(price, size, context)
        return price * (1 + rate) if is_buy else price * (1 - rate)

    def estimate_rate(self, price: float, size: float, is_buy: bool, context: dict[str, Any] | None = None) -> float:
        """Return the AMM slippage rate."""
        return self._compute_rate(price, size, context)

    def _compute_rate(self, price: float, size: float, context: dict[str, Any] | None) -> float:
        """Compute slippage rate from pool reserve."""
        if context is None:
            return self.min_slippage

        reserve = context.get(self.reserve_key)
        if reserve is None or float(reserve) <= 0:
            return self.min_slippage

        reserve_f = float(reserve)
        trade_value = abs(size) * price
        rate = trade_value / (reserve_f + trade_value)
        return max(rate, self.min_slippage)


def make_slippage_model(slippage: float | SlippageModel) -> SlippageModel:
    """Normalize a slippage specification into a SlippageModel.

    Accepts either a float (converted to ``FlatSlippage``) or an existing
    ``SlippageModel`` instance (returned as-is).

    Args:
        slippage: Float rate or SlippageModel instance.

    Returns:
        A SlippageModel instance.
    """
    if isinstance(slippage, SlippageModel):
        return slippage
    return FlatSlippage(pct=slippage)
