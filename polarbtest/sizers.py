"""Position sizing strategies for order quantity calculation.

Provides a base class and implementations for computing trade sizes
based on portfolio state, price, and risk parameters.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polarbtest.core import Portfolio


class Sizer(ABC):
    """Base class for position sizers.

    Subclasses implement `size()` to return the unsigned quantity
    to trade. The caller is responsible for applying direction
    (positive for buy, negative for sell).
    """

    @abstractmethod
    def size(self, portfolio: Portfolio, asset: str, price: float, **kwargs: Any) -> float:
        """Calculate unsigned position size.

        Args:
            portfolio: Current portfolio state.
            asset: Asset symbol being traded.
            price: Current or expected execution price.
            **kwargs: Sizer-specific parameters (e.g., stop_distance).

        Returns:
            Unsigned quantity to trade. Returns 0.0 if sizing is not possible.
        """
        ...


class FixedSizer(Sizer):
    """Always returns a fixed number of units.

    Args:
        quantity: Fixed number of units to trade.
    """

    def __init__(self, quantity: float) -> None:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        self.quantity = quantity

    def size(self, portfolio: Portfolio, asset: str, price: float, **kwargs: Any) -> float:
        return self.quantity


class PercentSizer(Sizer):
    """Size as a percentage of portfolio value.

    Args:
        percent: Fraction of portfolio value to allocate (e.g., 0.1 = 10%).
    """

    def __init__(self, percent: float) -> None:
        if not 0 < percent <= 1.0:
            raise ValueError("percent must be in (0, 1.0]")
        self.percent = percent

    def size(self, portfolio: Portfolio, asset: str, price: float, **kwargs: Any) -> float:
        if price <= 0:
            return 0.0
        value = portfolio.get_value() * self.percent
        return math.floor(value / price * 1e8) / 1e8


class FixedRiskSizer(Sizer):
    """Risk a fixed percentage of portfolio per trade.

    Calculates quantity so that if price moves by `stop_distance`,
    the loss equals `risk_percent` of portfolio value.

    Args:
        risk_percent: Fraction of portfolio to risk (e.g., 0.02 = 2%).

    Keyword Args (passed to size()):
        stop_distance: Absolute price distance to stop-loss.
    """

    def __init__(self, risk_percent: float) -> None:
        if not 0 < risk_percent <= 1.0:
            raise ValueError("risk_percent must be in (0, 1.0]")
        self.risk_percent = risk_percent

    def size(self, portfolio: Portfolio, asset: str, price: float, **kwargs: Any) -> float:
        stop_distance = kwargs.get("stop_distance")
        if stop_distance is None:
            raise ValueError("FixedRiskSizer requires 'stop_distance' keyword argument")
        stop_distance = float(stop_distance)
        if stop_distance <= 0:
            return 0.0
        risk_amount = portfolio.get_value() * self.risk_percent
        return math.floor(risk_amount / stop_distance * 1e8) / 1e8


class KellySizer(Sizer):
    """Size based on the Kelly criterion.

    Kelly fraction = win_rate - (1 - win_rate) / payoff_ratio,
    where payoff_ratio = avg_win / avg_loss. The fraction is clamped
    to [0, max_fraction] and applied as a percentage of portfolio value.

    Args:
        win_rate: Historical win rate (0 to 1).
        avg_win: Average winning trade return (absolute value).
        avg_loss: Average losing trade return (absolute value).
        max_fraction: Maximum Kelly fraction to use (default 0.25).
            Full Kelly is aggressive; half-Kelly (0.5x) is common in practice.
    """

    def __init__(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_fraction: float = 0.25,
    ) -> None:
        if not 0 < win_rate < 1:
            raise ValueError("win_rate must be in (0, 1)")
        if avg_win <= 0 or avg_loss <= 0:
            raise ValueError("avg_win and avg_loss must be positive")
        if max_fraction <= 0:
            raise ValueError("max_fraction must be positive")
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.max_fraction = max_fraction

    @property
    def kelly_fraction(self) -> float:
        """Raw Kelly fraction (can be negative if edge is negative)."""
        payoff_ratio = self.avg_win / self.avg_loss
        return self.win_rate - (1 - self.win_rate) / payoff_ratio

    def size(self, portfolio: Portfolio, asset: str, price: float, **kwargs: Any) -> float:
        if price <= 0:
            return 0.0
        fraction = max(0.0, min(self.kelly_fraction, self.max_fraction))
        if fraction == 0.0:
            return 0.0
        value = portfolio.get_value() * fraction
        return math.floor(value / price * 1e8) / 1e8


class VolatilitySizer(Sizer):
    """ATR-based sizing for constant risk per trade.

    Calculates quantity so that one ATR move equals `target_risk_percent`
    of portfolio value. This normalizes position sizes across assets
    with different volatility levels.

    Args:
        target_risk_percent: Fraction of portfolio value that one ATR move represents
            (e.g., 0.02 = 2%).

    Keyword Args (passed to size()):
        atr: Current ATR value for the asset.
    """

    def __init__(self, target_risk_percent: float) -> None:
        if not 0 < target_risk_percent <= 1.0:
            raise ValueError("target_risk_percent must be in (0, 1.0]")
        self.target_risk_percent = target_risk_percent

    def size(self, portfolio: Portfolio, asset: str, price: float, **kwargs: Any) -> float:
        atr_value = kwargs.get("atr")
        if atr_value is None:
            raise ValueError("VolatilitySizer requires 'atr' keyword argument")
        atr_value = float(atr_value)
        if atr_value <= 0 or price <= 0:
            return 0.0
        risk_amount = portfolio.get_value() * self.target_risk_percent
        return math.floor(risk_amount / atr_value * 1e8) / 1e8


class MaxPositionSizer(Sizer):
    """Wrapper that caps the output of another sizer.

    Applies limits after the inner sizer calculates its quantity.
    Multiple limits can be combined — the most restrictive wins.

    Args:
        sizer: Inner sizer to wrap.
        max_quantity: Maximum number of units (optional).
        max_percent: Maximum percentage of portfolio value (optional).
    """

    def __init__(
        self,
        sizer: Sizer,
        max_quantity: float | None = None,
        max_percent: float | None = None,
    ) -> None:
        if max_quantity is not None and max_quantity <= 0:
            raise ValueError("max_quantity must be positive")
        if max_percent is not None and not 0 < max_percent <= 1.0:
            raise ValueError("max_percent must be in (0, 1.0]")
        if max_quantity is None and max_percent is None:
            raise ValueError("At least one of max_quantity or max_percent must be provided")
        self.sizer = sizer
        self.max_quantity = max_quantity
        self.max_percent = max_percent

    def size(self, portfolio: Portfolio, asset: str, price: float, **kwargs: Any) -> float:
        qty = self.sizer.size(portfolio, asset, price, **kwargs)
        if self.max_quantity is not None:
            qty = min(qty, self.max_quantity)
        if self.max_percent is not None and price > 0:
            max_value = portfolio.get_value() * self.max_percent
            max_qty = math.floor(max_value / price * 1e8) / 1e8
            qty = min(qty, max_qty)
        return qty
