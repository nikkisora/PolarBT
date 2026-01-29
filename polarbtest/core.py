"""
Core backtesting engine and components.

This module provides the fundamental building blocks for backtesting:
- Portfolio: Manages positions and cash
- Strategy: Base class for defining trading strategies
- Engine: Executes the backtest simulation
- BacktestContext: Data container passed to strategy.next()
"""

import polars as pl
from typing import Dict, Optional, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict


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
    row: Dict[str, Any]
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
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,  # 0.05% slippage
    ):
        """
        Initialize a new portfolio.

        Args:
            initial_cash: Starting cash balance
            commission: Commission rate as a fraction (e.g., 0.001 = 0.1%)
            slippage: Slippage rate as a fraction (e.g., 0.0005 = 0.05%)
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission = commission
        self.slippage = slippage

        # Asset positions: {asset_name: quantity}
        self.positions: Dict[str, float] = defaultdict(float)

        # History tracking for metrics calculation
        self.equity_curve: List[float] = []
        self.timestamps: List[Any] = []

        # Current prices for portfolio valuation
        self._current_prices: Dict[str, float] = {}

    def update_prices(self, prices: Dict[str, float]):
        """
        Update current market prices for all assets.

        Args:
            prices: Dictionary mapping asset names to current prices
        """
        self._current_prices = prices

    def get_value(self) -> float:
        """
        Calculate total portfolio value (cash + positions).

        Returns:
            Total portfolio value
        """
        positions_value = sum(
            qty * self._current_prices.get(asset, 0.0)
            for asset, qty in self.positions.items()
        )
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

    def order(
        self, asset: str, quantity: float, limit_price: Optional[float] = None
    ) -> bool:
        """
        Place an order for an asset.

        Args:
            asset: Asset name
            quantity: Number of units to buy (positive) or sell (negative)
            limit_price: Optional limit price (uses current market price if None)

        Returns:
            True if order was executed, False otherwise
        """
        if quantity == 0:
            return False

        # Use limit price or current market price
        price = (
            limit_price if limit_price is not None else self._current_prices.get(asset)
        )

        if price is None or price <= 0:
            return False

        # Apply slippage (buy at higher price, sell at lower price)
        if quantity > 0:  # Buy
            execution_price = price * (1 + self.slippage)
        else:  # Sell
            execution_price = price * (1 - self.slippage)

        # Calculate total cost including commission
        gross_cost = abs(quantity) * execution_price
        commission_cost = gross_cost * self.commission
        total_cost = gross_cost + commission_cost

        # Check if we have enough cash (for buys) or shares (for sells)
        if quantity > 0:  # Buy
            if total_cost > self.cash:
                return False  # Not enough cash
            self.cash -= total_cost
            self.positions[asset] += quantity
        else:  # Sell
            if abs(quantity) > self.positions.get(asset, 0):
                return False  # Not enough shares
            self.cash += gross_cost - commission_cost
            self.positions[asset] += quantity  # quantity is negative

            # Clean up zero positions
            if abs(self.positions[asset]) < 1e-10:
                del self.positions[asset]

        return True

    def order_target(
        self, asset: str, target_quantity: float, limit_price: Optional[float] = None
    ) -> bool:
        """
        Order to reach a target position size.

        Args:
            asset: Asset name
            target_quantity: Desired final position quantity
            limit_price: Optional limit price

        Returns:
            True if order was executed, False otherwise
        """
        current_position = self.get_position(asset)
        delta = target_quantity - current_position
        return self.order(asset, delta, limit_price)

    def order_target_value(
        self, asset: str, target_value: float, limit_price: Optional[float] = None
    ) -> bool:
        """
        Order to reach a target position value.

        Args:
            asset: Asset name
            target_value: Desired position value in currency
            limit_price: Optional limit price

        Returns:
            True if order was executed, False otherwise
        """
        price = (
            limit_price if limit_price is not None else self._current_prices.get(asset)
        )
        if price is None or price <= 0:
            return False

        target_quantity = target_value / price
        return self.order_target(asset, target_quantity, limit_price)

    def order_target_percent(
        self, asset: str, target_percent: float, limit_price: Optional[float] = None
    ) -> bool:
        """
        Order to reach a target percentage of portfolio value.

        Args:
            asset: Asset name
            target_percent: Desired position as fraction of portfolio (e.g., 0.5 = 50%)
            limit_price: Optional limit price

        Returns:
            True if order was executed, False otherwise
        """
        portfolio_value = self.get_value()
        target_value = portfolio_value * target_percent
        return self.order_target_value(asset, target_value, limit_price)

    def close_position(self, asset: str, limit_price: Optional[float] = None) -> bool:
        """
        Close entire position in an asset.

        Args:
            asset: Asset name
            limit_price: Optional limit price

        Returns:
            True if position was closed, False otherwise
        """
        return self.order_target(asset, 0, limit_price)

    def close_all_positions(self) -> None:
        """Close all open positions."""
        assets = list(self.positions.keys())
        for asset in assets:
            self.close_position(asset)

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

    def __init__(self, **params):
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

    def on_start(self, portfolio: Portfolio) -> None:
        """
        Called once before the backtest starts.

        Override this to perform any initialization logic.

        Args:
            portfolio: The Portfolio instance
        """
        pass

    def on_finish(self, portfolio: Portfolio) -> None:
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
        data: pl.DataFrame,
        initial_cash: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        price_columns: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the backtesting engine.

        Args:
            strategy: Strategy instance to backtest
            data: Polars DataFrame with price data
            initial_cash: Starting cash balance
            commission: Commission rate as fraction
            slippage: Slippage rate as fraction
            price_columns: Dict mapping asset names to price columns
                          (default: {"close": "close"} for single asset)
        """
        self.strategy = strategy
        self.data = data
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage

        # If no price columns specified, assume single asset with "close" column
        if price_columns is None:
            # Try to detect available price column
            if "close" in data.columns:
                self.price_columns = {"asset": "close"}
            else:
                # Find first numeric column
                numeric_cols = [
                    c
                    for c in data.columns
                    if data[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
                ]
                if numeric_cols:
                    self.price_columns = {"asset": numeric_cols[0]}
                else:
                    raise ValueError("No price columns found in data")
        else:
            self.price_columns = price_columns

        self.portfolio: Optional[Portfolio] = None
        self.results: Optional[Dict[str, Any]] = None

    def run(self) -> Dict[str, Any]:
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
        )

        # Preprocess data using strategy
        processed_data = self.strategy.preprocess(self.data)

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
            current_prices = {
                asset: row_dict.get(price_col)
                for asset, price_col in self.price_columns.items()
            }

            # Update portfolio with current prices
            self.portfolio.update_prices(current_prices)

            # Create context for strategy
            ctx = BacktestContext(
                timestamp=row_dict.get(timestamp_col),
                row=row_dict,
                portfolio=self.portfolio,
                bar_index=idx,
            )

            # Call strategy logic
            self.strategy.next(ctx)

            # Record equity for metrics
            self.portfolio.record_equity(ctx.timestamp)

        # Call strategy finalization
        self.strategy.on_finish(self.portfolio)

        # Calculate and return results
        self.results = self._calculate_results()
        return self.results

    def _calculate_results(self) -> Dict[str, Any]:
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

        return metrics
