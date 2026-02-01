"""
Core backtesting engine and components.

This module provides the fundamental building blocks for backtesting:
- Portfolio: Manages positions and cash
- Strategy: Base class for defining trading strategies
- Engine: Executes the backtest simulation
- BacktestContext: Data container passed to strategy.next()
"""

import polars as pl
from typing import Dict, Optional, Any, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict


def standardize_dataframe(
    df: pl.DataFrame,
    timestamp_col: Optional[str] = None,
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
    data_dict: Dict[str, pl.DataFrame],
    price_column: str = "close",
) -> tuple[pl.DataFrame, Dict[str, str]]:
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
        standardized[first_asset].select(["timestamp"])
        if "timestamp" in standardized[first_asset].columns
        else None
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
            df_subset = df.select(["timestamp", price_column]).rename(
                {price_column: new_col_name}
            )

            if merged is None:
                merged = df_subset
            else:
                # Join on timestamp using coalesce to avoid duplicate timestamp columns
                merged = merged.join(
                    df_subset, on="timestamp", how="outer", coalesce=True
                )
        else:
            # No timestamp - add index and merge
            df_subset = df.select([price_column]).rename({price_column: new_col_name})
            if merged is None:
                merged = df_subset
            else:
                merged = pl.concat([merged, df_subset], how="horizontal")

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
        commission: Union[
            float, tuple[float, float]
        ] = 0.0,
        slippage: float = 0.0,
        order_delay: int = 0,
    ):
        """
        Initialize a new portfolio.

        Args:
            initial_cash: Starting cash balance
            commission: Commission as a percentage (e.g., 0.001 = 0.1%) or tuple of (fixed_commission, percent_commission)
                       For example: 0.001 means 0.1% per trade, (5.0, 0.001) means $5 + 0.1% per trade
            slippage: Slippage rate as a fraction (e.g., 0.0005 = 0.05%)
            order_delay: Number of bars to delay order execution (0 = immediate, 1 = next bar)
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

        # Asset positions: {asset_name: quantity}
        self.positions: Dict[str, float] = defaultdict(float)

        # History tracking for metrics calculation
        self.equity_curve: List[float] = []
        self.timestamps: List[Any] = []

        # Current prices for portfolio valuation
        self._current_prices: Dict[str, float] = {}

        # Pending orders queue: List of (bar_to_execute, asset, quantity, limit_price)
        self._pending_orders: List[tuple[int, str, float, Optional[float]]] = []
        self._current_bar: int = 0

    def update_prices(self, prices: Dict[str, float], bar_index: int = 0):
        """
        Update current market prices for all assets and execute pending orders.

        Args:
            prices: Dictionary mapping asset names to current prices
            bar_index: Current bar index (used for order delay)
        """
        self._current_prices = prices
        self._current_bar = bar_index
        self._execute_pending_orders()

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

    def _execute_pending_orders(self):
        """Execute orders that are due based on order_delay."""
        if not self._pending_orders:
            return

        # Filter orders that should execute this bar
        orders_to_execute = [
            (asset, qty, price)
            for (bar, asset, qty, price) in self._pending_orders
            if bar <= self._current_bar
        ]

        # Remove executed orders from pending
        self._pending_orders = [
            order for order in self._pending_orders if order[0] > self._current_bar
        ]

        # Execute the orders
        for asset, quantity, limit_price in orders_to_execute:
            self._execute_order_immediate(asset, quantity, limit_price)

    def _execute_order_immediate(
        self, asset: str, quantity: float, limit_price: Optional[float] = None
    ) -> bool:
        """
        Execute an order immediately at current prices.

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
        commission_cost = self.commission_fixed + (gross_cost * self.commission_percent)
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
            True if order was placed/queued, False otherwise
        """
        if quantity == 0:
            return False

        # If order_delay is 0, execute immediately
        if self.order_delay == 0:
            return self._execute_order_immediate(asset, quantity, limit_price)

        # Otherwise, queue the order for future execution
        execute_at_bar = self._current_bar + self.order_delay
        self._pending_orders.append((execute_at_bar, asset, quantity, limit_price))
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
        price = (
            limit_price if limit_price is not None else self._current_prices.get(asset)
        )
        if price is None or price <= 0:
            return False

        portfolio_value = self.get_value()
        current_position = self.get_position(asset)
        current_value = current_position * price

        # Calculate target value
        target_value = portfolio_value * target_percent

        # Calculate the difference we need to trade
        value_delta = target_value - current_value

        if abs(value_delta) < 1e-6:  # Already at target
            return False

        # Account for fees when calculating target quantity
        # When buying: we pay slippage + commission on the gross cost
        # When selling: we receive slippage - commission on the gross proceeds
        if value_delta > 0:  # Buying
            # For buying: total_cost = quantity * execution_price * (1 + percent_commission) + fixed_commission
            # We need: total_cost = value_delta
            # Solving: quantity = (value_delta - fixed_commission) / (execution_price * (1 + percent_commission))
            execution_price = price * (1 + self.slippage)
            cost_multiplier = 1 + self.commission_percent
            quantity_to_buy = (value_delta - self.commission_fixed) / (
                execution_price * cost_multiplier
            )
            target_quantity = current_position + quantity_to_buy
        else:  # Selling
            # For selling: net_proceeds = quantity * execution_price * (1 - percent_commission) - fixed_commission
            # We need: net_proceeds = abs(value_delta)
            # Solving: quantity = (abs(value_delta) + fixed_commission) / (execution_price * (1 - percent_commission))
            execution_price = price * (1 - self.slippage)
            proceeds_multiplier = 1 - self.commission_percent
            quantity_to_sell = (abs(value_delta) + self.commission_fixed) / (
                execution_price * proceeds_multiplier
            )
            target_quantity = current_position - quantity_to_sell

        return self.order_target(asset, target_quantity, limit_price)

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
        data: Union[pl.DataFrame, Dict[str, pl.DataFrame]],
        initial_cash: float = 100_000.0,
        commission: Union[float, tuple[float, float]] = 0.0,
        slippage: float = 0.0,
        price_columns: Optional[Dict[str, str]] = None,
        warmup: Union[int, str] = "auto",
        order_delay: int = 0,
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
        """
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage

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
                        if self.data[c].dtype
                        in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
                    ]
                    if numeric_cols:
                        self.price_columns = {"asset": numeric_cols[0]}
                    else:
                        raise ValueError("No price columns found in data")
            else:
                self.price_columns = price_columns

        self.portfolio: Optional[Portfolio] = None
        self.results: Optional[Dict[str, Any]] = None

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
            pl.all_horizontal(
                [pl.col(col).is_not_null() for col in cols_to_check]
            ).alias("all_valid")
        )

        # Find the index of the first True value
        for idx, row in enumerate(all_non_null.iter_rows()):
            if row[0]:  # First (and only) column is "all_valid"
                return idx

        # If no row has all non-null values, return length - 1 (skip all but last)
        return max(0, len(df) - 1)

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
            order_delay=self.order_delay,
        )

        # Preprocess data using strategy
        processed_data = self.strategy.preprocess(self.data)

        # Calculate warmup period if set to "auto"
        warmup_periods: int
        if self.warmup == "auto":
            warmup_periods = self._calculate_auto_warmup(processed_data)
        else:
            warmup_periods = self.warmup  # type: ignore

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
            self.portfolio.update_prices(current_prices, idx)

            # Create context for strategy
            ctx = BacktestContext(
                timestamp=row_dict.get(timestamp_col),
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
