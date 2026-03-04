"""
Trade tracking system for backtesting.

This module provides trade tracking functionality to record complete trades
(entry to exit) and calculate trade-level performance metrics.
"""

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import polars as pl

from polarbt.results import TradeStats


@dataclass
class Trade:
    """
    Represents a complete trade (entry to exit).

    A trade is created when a position is fully or partially closed.
    Tracks entry/exit details and calculates P&L automatically.

    Attributes:
        trade_id: Unique identifier
        asset: Asset symbol
        direction: "long" or "short"

        entry_bar: Bar index of entry
        entry_timestamp: Timestamp of entry
        entry_price: Average entry price
        entry_size: Position size at entry (always positive)
        entry_value: Total entry value (price * size)
        entry_commission: Commission paid on entry

        exit_bar: Bar index of exit
        exit_timestamp: Timestamp of exit
        exit_price: Average exit price
        exit_size: Size closed (should match entry_size)
        exit_value: Total exit value
        exit_commission: Commission paid on exit

        pnl: Profit/loss (exit_value - entry_value - commissions)
        pnl_pct: Profit/loss as fraction (e.g., 0.04 = 4%)
        return_pct: Return as fraction (pnl / entry_value)
        bars_held: Number of bars position was held

        mae: Maximum Adverse Excursion (worst drawdown during trade)
        mfe: Maximum Favorable Excursion (best profit during trade)

        tags: User tags for categorization

    Example:
        trade = Trade(
            trade_id="1",
            asset="BTC",
            direction="long",
            entry_bar=10,
            entry_timestamp=1000,
            entry_price=50000,
            entry_size=1.0,
            entry_value=50000,
            exit_bar=20,
            exit_timestamp=2000,
            exit_price=52000,
            exit_size=1.0,
            exit_value=52000
        )
        # trade.pnl = 2000
        # trade.pnl_pct = 0.04
    """

    trade_id: str
    asset: str
    direction: str

    entry_bar: int
    entry_timestamp: Any
    entry_price: float
    entry_size: float
    entry_value: float
    entry_commission: float = 0.0

    exit_bar: int = 0
    exit_timestamp: Any = None
    exit_price: float = 0.0
    exit_size: float = 0.0
    exit_value: float = 0.0
    exit_commission: float = 0.0

    pnl: float = 0.0
    pnl_pct: float = 0.0
    return_pct: float = 0.0
    bars_held: int = 0

    mae: float | None = None
    mfe: float | None = None

    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Calculate derived metrics after initialization."""
        self._calculate_metrics()

    def _calculate_metrics(self) -> None:
        """Calculate P&L and performance metrics."""
        if self.direction == "long":
            gross_pnl = self.exit_value - self.entry_value
        else:
            gross_pnl = self.entry_value - self.exit_value

        self.pnl = gross_pnl - self.entry_commission - self.exit_commission

        if self.entry_value > 0:
            self.pnl_pct = self.pnl / self.entry_value
            self.return_pct = self.pnl_pct
        else:
            self.pnl_pct = 0.0
            self.return_pct = 0.0

        self.bars_held = self.exit_bar - self.entry_bar

    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0

    def is_loser(self) -> bool:
        """Check if trade was unprofitable."""
        return self.pnl < 0

    def is_long(self) -> bool:
        """Check if trade was long."""
        return self.direction == "long"

    def is_short(self) -> bool:
        """Check if trade was short."""
        return self.direction == "short"


class TradeTracker:
    """
    Tracks open positions and generates Trade objects when positions close.

    Uses simple full position tracking: each position open/close creates
    a complete trade. Partial closes create multiple trades.

    Example:
        tracker = TradeTracker()

        # Open position
        tracker.on_position_opened("BTC", 1.0, 50000, 10, 1000, 50)

        # Close position (creates trade)
        trade = tracker.on_position_closed("BTC", 1.0, 52000, 20, 2000, 52)
        print(f"P&L: {trade.pnl}")
    """

    def __init__(self) -> None:
        self.trades: list[Trade] = []
        self.open_positions: dict[str, dict[str, Any]] = {}

    def on_position_opened(
        self, asset: str, size: float, price: float, bar: int, timestamp: Any, commission: float
    ) -> None:
        """
        Record position opening.

        Args:
            asset: Asset symbol
            size: Position size (positive for long, negative for short)
            price: Entry price
            bar: Bar index
            timestamp: Timestamp
            commission: Commission paid
        """
        abs_size = abs(size)
        direction = "long" if size > 0 else "short"

        self.open_positions[asset] = {
            "entry_bar": bar,
            "entry_timestamp": timestamp,
            "entry_price": price,
            "entry_size": abs_size,
            "entry_value": abs_size * price,
            "entry_commission": commission,
            "direction": direction,
            "mae": 0.0,  # Maximum Adverse Excursion
            "mfe": 0.0,  # Maximum Favorable Excursion
        }

    def update_mae_mfe(self, asset: str, current_price: float) -> None:
        """
        Update Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE)
        for an open position.

        Args:
            asset: Asset symbol
            current_price: Current market price

        Note:
            Should be called on each bar while position is open.
        """
        if asset not in self.open_positions:
            return

        position = self.open_positions[asset]
        entry_price = position["entry_price"]
        direction = position["direction"]

        # Calculate unrealized P&L
        unrealized_pnl = current_price - entry_price if direction == "long" else entry_price - current_price

        # Update MAE (most negative P&L seen)
        if unrealized_pnl < position["mae"]:
            position["mae"] = unrealized_pnl

        # Update MFE (most positive P&L seen)
        if unrealized_pnl > position["mfe"]:
            position["mfe"] = unrealized_pnl

    def on_position_increased(self, asset: str, added_size: float, price: float, commission: float) -> None:
        """Update average entry price when position size increases.

        Computes a volume-weighted average entry price and adds the
        additional commission to the running entry commission total.

        Args:
            asset: Asset symbol.
            added_size: Additional size added (positive value).
            price: Execution price of the new portion.
            commission: Commission paid for this addition.
        """
        if asset not in self.open_positions:
            return

        pos = self.open_positions[asset]
        old_size = pos["entry_size"]
        old_value = pos["entry_value"]
        added_value = added_size * price
        new_size = old_size + added_size
        new_value = old_value + added_value

        pos["entry_size"] = new_size
        pos["entry_value"] = new_value
        pos["entry_price"] = new_value / new_size if new_size > 0 else price
        pos["entry_commission"] += commission

    def on_position_closed(
        self, asset: str, size_closed: float, price: float, bar: int, timestamp: Any, commission: float
    ) -> Trade | None:
        """
        Record position closing and create Trade object.

        Args:
            asset: Asset symbol
            size_closed: Size being closed (positive value)
            price: Exit price
            bar: Bar index
            timestamp: Timestamp
            commission: Commission paid

        Returns:
            Trade object if position found, None otherwise
        """
        if asset not in self.open_positions:
            return None

        entry_info = self.open_positions[asset]
        abs_size_closed = abs(size_closed)

        if abs_size_closed >= entry_info["entry_size"]:
            trade = Trade(
                trade_id=str(uuid4()),
                asset=asset,
                direction=entry_info["direction"],
                entry_bar=entry_info["entry_bar"],
                entry_timestamp=entry_info["entry_timestamp"],
                entry_price=entry_info["entry_price"],
                entry_size=entry_info["entry_size"],
                entry_value=entry_info["entry_value"],
                entry_commission=entry_info["entry_commission"],
                exit_bar=bar,
                exit_timestamp=timestamp,
                exit_price=price,
                exit_size=entry_info["entry_size"],
                exit_value=entry_info["entry_size"] * price,
                exit_commission=commission,
                mae=entry_info.get("mae"),
                mfe=entry_info.get("mfe"),
            )

            self.trades.append(trade)
            del self.open_positions[asset]
            return trade
        else:
            partial_value = abs_size_closed * entry_info["entry_price"]
            partial_commission = entry_info["entry_commission"] * (abs_size_closed / entry_info["entry_size"])

            trade = Trade(
                trade_id=str(uuid4()),
                asset=asset,
                direction=entry_info["direction"],
                entry_bar=entry_info["entry_bar"],
                entry_timestamp=entry_info["entry_timestamp"],
                entry_price=entry_info["entry_price"],
                entry_size=abs_size_closed,
                entry_value=partial_value,
                entry_commission=partial_commission,
                exit_bar=bar,
                exit_timestamp=timestamp,
                exit_price=price,
                exit_size=abs_size_closed,
                exit_value=abs_size_closed * price,
                exit_commission=commission,
                mae=entry_info.get("mae"),
                mfe=entry_info.get("mfe"),
            )

            self.trades.append(trade)

            entry_info["entry_size"] -= abs_size_closed
            entry_info["entry_value"] -= partial_value
            entry_info["entry_commission"] -= partial_commission

            return trade

    def on_position_reversed(
        self,
        asset: str,
        old_size: float,
        new_size: float,
        price: float,
        bar: int,
        timestamp: Any,
        commission: float,
    ) -> Trade | None:
        """
        Handle position reversal (e.g., long -> short or short -> long).

        Args:
            asset: Asset symbol
            old_size: Old position size
            new_size: New position size
            price: Price at reversal
            bar: Bar index
            timestamp: Timestamp
            commission: Commission paid

        Returns:
            Trade object for closed position
        """
        if asset in self.open_positions:
            trade = self.on_position_closed(asset, abs(old_size), price, bar, timestamp, commission)
            self.on_position_opened(asset, new_size, price, bar, timestamp, 0.0)
            return trade
        return None

    def get_trades(self) -> list[Trade]:
        """
        Get all completed trades.

        Returns:
            List of Trade objects
        """
        return self.trades

    def get_trades_df(self) -> pl.DataFrame:
        """
        Export all trades as a Polars DataFrame.

        Returns:
            DataFrame with trade details and metrics

        Example:
            >>> tracker = TradeTracker()
            >>> # ... open/close positions ...
            >>> df = tracker.get_trades_df()
            >>> print(df.select(["asset", "pnl", "pnl_pct", "bars_held"]))
        """
        if not self.trades:
            return pl.DataFrame(
                schema={
                    "trade_id": pl.Utf8,
                    "asset": pl.Utf8,
                    "direction": pl.Utf8,
                    "entry_bar": pl.Int64,
                    "entry_price": pl.Float64,
                    "entry_size": pl.Float64,
                    "entry_value": pl.Float64,
                    "exit_bar": pl.Int64,
                    "exit_price": pl.Float64,
                    "exit_value": pl.Float64,
                    "pnl": pl.Float64,
                    "pnl_pct": pl.Float64,
                    "bars_held": pl.Int64,
                    "mae": pl.Float64,
                    "mfe": pl.Float64,
                }
            )

        trade_dicts = []
        for trade in self.trades:
            trade_dicts.append(
                {
                    "trade_id": trade.trade_id,
                    "asset": trade.asset,
                    "direction": trade.direction,
                    "entry_bar": trade.entry_bar,
                    "entry_timestamp": trade.entry_timestamp,
                    "entry_price": trade.entry_price,
                    "entry_size": trade.entry_size,
                    "entry_value": trade.entry_value,
                    "entry_commission": trade.entry_commission,
                    "exit_bar": trade.exit_bar,
                    "exit_timestamp": trade.exit_timestamp,
                    "exit_price": trade.exit_price,
                    "exit_size": trade.exit_size,
                    "exit_value": trade.exit_value,
                    "exit_commission": trade.exit_commission,
                    "pnl": trade.pnl,
                    "pnl_pct": trade.pnl_pct,
                    "return_pct": trade.return_pct,
                    "bars_held": trade.bars_held,
                    "mae": trade.mae,
                    "mfe": trade.mfe,
                }
            )

        return pl.DataFrame(trade_dicts)

    def get_trade_stats(self) -> TradeStats:
        """Calculate aggregate trade statistics.

        Returns:
            TradeStats with win_rate, avg_win, avg_loss, profit_factor, etc.
        """
        if not self.trades:
            return TradeStats()

        winners = [t for t in self.trades if t.is_winner()]
        losers = [t for t in self.trades if t.is_loser()]

        total_wins = sum(t.pnl for t in winners)
        total_losses = abs(sum(t.pnl for t in losers))

        return TradeStats(
            total_trades=len(self.trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=len(winners) / len(self.trades) if self.trades else 0.0,
            avg_win=total_wins / len(winners) if winners else 0.0,
            avg_loss=total_losses / len(losers) if losers else 0.0,
            avg_pnl=sum(t.pnl for t in self.trades) / len(self.trades),
            profit_factor=total_wins / total_losses if total_losses > 0 else float("inf"),
            total_pnl=sum(t.pnl for t in self.trades),
        )
