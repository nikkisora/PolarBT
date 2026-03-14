"""Commission models for flexible fee calculation.

Provides a base class and implementations for computing trade commissions
based on order size, price, and cumulative volume.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable


class CommissionModel(ABC):
    """Base class for commission models.

    Subclasses implement `calculate()` to return the total commission
    cost for a single order execution.
    """

    @abstractmethod
    def calculate(self, size: float, price: float, is_reversal: bool = False) -> float:
        """Calculate commission for an order execution.

        Args:
            size: Unsigned quantity being traded.
            price: Execution price per unit.
            is_reversal: Whether this order crosses zero (long→short or short→long),
                        which may incur double fixed fees.

        Returns:
            Total commission cost (always non-negative).
        """
        ...


class PercentCommission(CommissionModel):
    """Percentage-only commission (the current default behavior).

    Args:
        rate: Commission rate as a fraction (e.g., 0.001 = 0.1%).
    """

    def __init__(self, rate: float) -> None:
        if rate < 0:
            raise ValueError("rate must be non-negative")
        self.rate = rate

    def calculate(self, size: float, price: float, is_reversal: bool = False) -> float:
        return size * price * self.rate


class FixedPlusPercentCommission(CommissionModel):
    """Fixed dollar amount plus percentage commission.

    This matches the existing tuple-based commission behavior:
    fixed fee per trade + percentage of gross value. Reversals
    charge the fixed fee twice.

    Args:
        fixed: Fixed dollar amount per trade.
        percent: Percentage rate as a fraction (e.g., 0.001 = 0.1%).
    """

    def __init__(self, fixed: float, percent: float) -> None:
        if fixed < 0:
            raise ValueError("fixed must be non-negative")
        if percent < 0:
            raise ValueError("percent must be non-negative")
        self.fixed = fixed
        self.percent = percent

    def calculate(self, size: float, price: float, is_reversal: bool = False) -> float:
        num_fixed = 2 if is_reversal else 1
        return self.fixed * num_fixed + (size * price * self.percent)


class MakerTakerCommission(CommissionModel):
    """Different rates for maker (limit) and taker (market) orders.

    Since order type is not passed to `calculate()`, this model uses
    a default side that can be overridden per-instance. Limit orders
    are typically "maker" and market/stop orders are "taker".

    In practice, the Portfolio always calls `calculate()` at execution time.
    For simplicity, the `is_maker` flag is set at construction and applies
    to all orders using this model. Create separate instances for different
    order routing if needed.

    Args:
        maker_rate: Commission rate for maker (limit) orders.
        taker_rate: Commission rate for taker (market/stop) orders.
        is_maker: Whether this instance uses maker rate (default False = taker).
        fixed: Optional fixed fee per trade.
    """

    def __init__(
        self,
        maker_rate: float,
        taker_rate: float,
        is_maker: bool = False,
        fixed: float = 0.0,
    ) -> None:
        if maker_rate < 0:
            raise ValueError("maker_rate must be non-negative")
        if taker_rate < 0:
            raise ValueError("taker_rate must be non-negative")
        if fixed < 0:
            raise ValueError("fixed must be non-negative")
        self.maker_rate = maker_rate
        self.taker_rate = taker_rate
        self.is_maker = is_maker
        self.fixed = fixed

    def calculate(self, size: float, price: float, is_reversal: bool = False) -> float:
        rate = self.maker_rate if self.is_maker else self.taker_rate
        num_fixed = 2 if is_reversal else 1
        return self.fixed * num_fixed + (size * price * rate)


class TieredCommission(CommissionModel):
    """Volume-based tiered commission rates.

    Tiers are defined as a list of (volume_threshold, rate) tuples sorted
    by ascending threshold. The rate for the highest threshold that the
    cumulative 30-day volume exceeds is used.

    The model tracks cumulative traded volume internally. Call `reset_volume()`
    to reset (e.g., at the start of a new period).

    Args:
        tiers: List of (volume_threshold, rate) tuples. Must be sorted by
              ascending threshold. The first tier's threshold should be 0.
        fixed: Optional fixed fee per trade.

    Example:
        >>> model = TieredCommission(tiers=[(0, 0.001), (100_000, 0.0008), (1_000_000, 0.0005)])
    """

    def __init__(self, tiers: list[tuple[float, float]], fixed: float = 0.0) -> None:
        if not tiers:
            raise ValueError("tiers must not be empty")
        if fixed < 0:
            raise ValueError("fixed must be non-negative")
        self.tiers = sorted(tiers, key=lambda t: t[0])
        self.fixed = fixed
        self.cumulative_volume = 0.0

    def _get_rate(self) -> float:
        """Get the commission rate for the current cumulative volume."""
        rate = self.tiers[0][1]
        for threshold, tier_rate in self.tiers:
            if self.cumulative_volume >= threshold:
                rate = tier_rate
            else:
                break
        return rate

    def calculate(self, size: float, price: float, is_reversal: bool = False) -> float:
        gross = size * price
        rate = self._get_rate()
        self.cumulative_volume += gross
        num_fixed = 2 if is_reversal else 1
        return self.fixed * num_fixed + (gross * rate)

    def reset_volume(self) -> None:
        """Reset cumulative traded volume to zero."""
        self.cumulative_volume = 0.0


class CustomCommission(CommissionModel):
    """User-provided callable for commission calculation.

    Args:
        func: Callable with signature (size, price, is_reversal) -> float.

    Example:
        >>> model = CustomCommission(lambda size, price, is_reversal: max(1.0, size * price * 0.001))
    """

    def __init__(self, func: Callable[[float, float, bool], float]) -> None:
        self.func = func

    def calculate(self, size: float, price: float, is_reversal: bool = False) -> float:
        return self.func(size, price, is_reversal)


def make_commission_model(commission: float | tuple[float, float] | CommissionModel) -> CommissionModel:
    """Convert legacy commission formats to a CommissionModel.

    Args:
        commission: Commission specification. Accepts:
            - float: percentage-only (e.g., 0.001 = 0.1%)
            - tuple[float, float]: (fixed, percent) (e.g., (5.0, 0.001))
            - CommissionModel: returned as-is

    Returns:
        A CommissionModel instance.
    """
    if isinstance(commission, CommissionModel):
        return commission
    if isinstance(commission, tuple):
        return FixedPlusPercentCommission(fixed=commission[0], percent=commission[1])
    return PercentCommission(rate=commission)


# ---------------------------------------------------------------------------
# Fee presets
# ---------------------------------------------------------------------------

#: Solana Pump.fun fee preset: ~0.000005 SOL base fee + 1% platform fee.
SOLANA_PUMPFUN = FixedPlusPercentCommission(fixed=0.000005, percent=0.01)
