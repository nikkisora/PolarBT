# Trade-Level Data Support — Implementation Plan

Extend PolarBT to support backtesting strategies on raw trade/tick data from DEX/AMM sources (e.g. Pump.fun memecoins on Solana). This requires first unifying multi-asset handling into a single long-format interface, then building the trade data pipeline on top of it.

## Trade Data Standard

### Required columns

| Column | Dtype | Description |
|---|---|---|
| `timestamp` | `Datetime(us)` | Trade execution time (UTC) |
| `symbol` | `String` | Token/asset identifier |
| `price` | `Float64` | Execution price in quote currency |
| `amount` | `Float64` | Trade size in quote currency |
| `side` | `String` | `"buy"` or `"sell"` |

### Optional columns (recognized by the pipeline)

| Column | Dtype | Description |
|---|---|---|
| `base_amount` | `Float64` | Quantity of base token traded |
| `pool_reserve` | `Float64` | Quote currency in liquidity pool |
| `trader` | `String` | Wallet address |
| `tx_id` | `String` | Transaction identifier |

Any additional columns are passed through to the aggregation output (last value per bucket by default, or custom aggregation if specified). Users can pre-label trades with `is_dev`, `wallet_type`, etc. and have those survive aggregation.

### Design decisions

- **`amount` is quote-denominated** (e.g. SOL, not token quantity). Base token supply varies wildly across memecoins; quote-denominated volume is comparable across assets and maps directly to portfolio cash flows.
- **`side` as String** (`"buy"` / `"sell"`), not a boolean. This is the dominant convention across FIX protocol, Binance, Coinbase, Kraken, and most exchange APIs. More extensible if future types are needed.
- **`symbol` not `mint`**. Consistent with `weight_backtest.py` and generic across chains.
- **Units are the user's responsibility**. The library doesn't care if `price` is in SOL, ETH, or USD — just that it's consistent. Lamport-to-SOL, wei-to-ETH etc. is normalization that happens before data enters the pipeline.

---

## Phase 0: Unified Multi-Asset Engine

Rework the Engine to use long-format data internally for all cases (single-asset, multi-asset, dynamic universe). This replaces the current wide-format approach and eliminates `weight_backtest.py` as a separate code path.

### Problems with current approach

1. **`merge_asset_dataframes()` drops OHLC data.** Only the `close` column is merged per asset. The Engine tries to read `{asset}_open`, `{asset}_high`, `{asset}_low` via naming convention, but they don't exist after merging — so stop-losses, take-profits, and limit orders silently lose intra-bar fidelity in multi-asset mode.
2. **`"asset"` sentinel pattern.** The main loop branches on `if asset == "asset"` to distinguish single vs multi-asset for OHLC lookup, factor lookup, and column naming. Every new feature must handle both paths.
3. **`weight_backtest.py` duplicates core logic.** Stop-loss/take-profit with OHLC priority, position tracking, commission calculation, and trade recording are all reimplemented outside of Portfolio. It uses a flat `fee_ratio` instead of `CommissionModel`, has no order delay or leverage support, and returns a different result type.
4. **Strategy API leaks internal format.** In `preprocess()`, users must write `ind.sma("BTC_close", 20)` instead of `ind.sma("close", 20)` for asset "BTC". Column naming conventions are part of the interface.
5. **Wide format doesn't scale.** With 200K+ tokens, creating one column per asset per OHLCV field is impractical.

### 0.1 Long-format data model

The Engine accepts data in one of three forms, all normalized internally to long format:

**Form A — Single DataFrame without `symbol` column (single-asset):**
```python
# Existing API, unchanged
engine = Engine(strategy, df)  # df has timestamp, open, high, low, close, volume
```
Internally assigns `symbol = "asset"` to every row. Single-asset strategies work exactly as before with no code changes.

**Form B — `dict[str, pl.DataFrame]` (named multi-asset):**
```python
# Existing API, unchanged
engine = Engine(strategy, {"BTC": btc_df, "ETH": eth_df})
```
Each DataFrame is tagged with its symbol name and concatenated vertically. All OHLCV columns are preserved (fixing the current bug).

**Form C — Single DataFrame with `symbol` column (long-format, new):**
```python
# New API for large/dynamic universes
engine = Engine(strategy, long_df)  # long_df has timestamp, symbol, open, high, low, close, volume
```
Used directly. This is the native format for trade data aggregation output and the format `weight_backtest.py` already expects.

Auto-detection: if the input DataFrame has a `symbol` column, treat as Form C. Otherwise Form A. `dict` input is always Form B.

### 0.2 Revised `BacktestContext`

```python
@dataclass
class BacktestContext:
    timestamp: Any
    bar_index: int
    portfolio: Portfolio
    symbols: list[str]                    # symbols with data on this bar
    data: dict[str, dict[str, Any]]       # {symbol: {col: value, ...}}

    def row(self, symbol: str | None = None) -> dict[str, Any]:
        """Get bar data for a symbol. Defaults to the only symbol in single-asset mode."""
        ...
```

- `ctx.data["BTC"]["close"]` — explicit symbol access
- `ctx.row()` — convenience for single-asset (returns the only entry), raises if ambiguous
- `ctx.row("BTC")` — explicit single-symbol access
- `ctx.symbols` — iterate over available symbols this bar

For backward compatibility, `ctx.row` can also work as a property (not just method) in single-asset mode, returning the flat dict directly. This preserves `ctx.row["close"]` syntax for existing strategies.

### 0.3 Revised `Strategy.preprocess()`

`preprocess()` receives the full long-format DataFrame. The engine provides a helper for per-symbol indicator application:

```python
class Strategy(ABC):
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        # Per-symbol indicators — standard column names, partitioned automatically
        df = df.with_columns(
            ind.sma("close", 20).over("symbol").alias("sma_20"),
            ind.rsi("close", 14).over("symbol").alias("rsi"),
        )
        # Cross-sectional indicators — operate across all symbols per timestamp
        df = df.with_columns(
            pl.col("volume").rank().over("timestamp").alias("volume_rank"),
        )
        return df
```

Key improvement: indicators use standard column names (`"close"`, not `"BTC_close"`) with Polars' `.over("symbol")` for partitioning. No naming convention leakage.

### 0.4 Revised main loop

```python
# Pseudocode for the new Engine.run() main loop
for idx, (timestamp, group) in enumerate(iter_by_timestamp(processed_data)):
    # group is the subset of rows for this timestamp
    current_prices = {}
    ohlc_data = {}
    bar_data = {}

    for row in group.iter_rows(named=True):
        sym = row["symbol"]
        current_prices[sym] = row["close"]
        ohlc_data[sym] = {"open": row["open"], "high": row["high"], "low": row["low"], "close": row["close"]}
        bar_data[sym] = row

    portfolio.update_prices(current_prices, idx, ohlc_data, timestamp)

    ctx = BacktestContext(
        timestamp=timestamp,
        bar_index=idx,
        portfolio=portfolio,
        symbols=list(bar_data.keys()),
        data=bar_data,
    )

    if idx >= warmup_periods:
        strategy.next(ctx)
        portfolio.record_equity(ctx.timestamp)
```

For small universes (< ~50 symbols), the per-timestamp grouping can be optimized by pre-pivoting to wide format for the hot loop. This is an internal optimization — the strategy API is the same regardless.

### 0.5 Atomic rebalance on Portfolio

Add `Portfolio.rebalance(weights: dict[str, float])` — computes all target quantities from a single `get_value()` snapshot, then executes all orders. This avoids the sequential drift problem where calling `order_target_percent` in a loop causes each fill to change cash/equity before the next order is sized.

Without this, a loop like:
```python
ctx.portfolio.order_target_percent("BTC", 0.5)  # fills, changes cash
ctx.portfolio.order_target_percent("ETH", 0.5)  # sees different portfolio_value
```
produces allocations that don't sum to the intended weights. The error compounds with more assets and higher fee rates.

`rebalance()` internally:
1. Snapshots `portfolio_value = self.get_value()`
2. Computes `target_qty = portfolio_value * weight / price` for all symbols
3. Closes positions not in the target weights
4. Executes sells before buys (frees cash first)
5. Fee-adjusts buy quantities so the total cost fits available cash
6. All orders go through the normal `order_target()` path (commissions, slippage, trade tracking, risk limits all apply)

### 0.6 Retire `weight_backtest.py`

Replace with a `WeightStrategy` base class that uses the unified engine:

```python
class WeightStrategy(Strategy):
    """Strategy that expresses positions as target weights per symbol."""

    @abstractmethod
    def get_weights(self, ctx: BacktestContext) -> dict[str, float]:
        """Return target portfolio weights for each symbol."""
        ...

    def next(self, ctx: BacktestContext) -> None:
        weights = self.get_weights(ctx)
        ctx.portfolio.rebalance(weights)
```

This reuses Portfolio's order execution, commission models, stop-loss/take-profit, and leverage — eliminating all duplication. The existing `backtest_weights()` convenience function is preserved as a wrapper that constructs a `WeightStrategy` from a DataFrame with a `weight` column.

### 0.7 Backward compatibility

Existing single-asset and multi-asset strategies continue to work:
- Single-asset: `ctx.row["close"]` still works (property returns the single symbol's data)
- Multi-asset with `dict` input: accepted and converted to long format internally
- `price_columns` parameter: still accepted for explicit column-to-symbol mapping on wide-format input
- `backtest_weights()`: still available as a convenience function
- `merge_asset_dataframes()`: deprecated, kept for one version

---

## Phase 1: Trade Data Ingestion and Aggregation

**New module**: `data/trades.py`

### 1.1 Validation

`validate_trades(df) -> ValidationResult` — analogous to existing `validate()` in `data/validation.py`:

- Required columns present with correct dtypes
- `side` values are only `"buy"` or `"sell"`
- `price` and `amount` are positive (no nulls)
- `timestamp` sorted ascending per symbol
- Warn on duplicate `tx_id` values (if column present)

### 1.2 Aggregation pipeline

`aggregate_trades(df, interval, ...) -> pl.DataFrame`

Converts trade-level data to OHLCV bars per symbol using `group_by_dynamic`.

**Parameters**:
- `df: pl.DataFrame` — validated trade data
- `interval: str` — bar duration (`"1m"`, `"5m"`, `"1h"`, `"1d"`, etc.)
- `exchange_rate: pl.DataFrame | None` — optional `(timestamp, rate)` DataFrame for quote-to-USD conversion. When provided, adds USD-denominated volume columns alongside quote-denominated ones.
- `min_trades: int = 1` — minimum trades per bar to emit (filters dust)
- `extra_aggs: dict[str, pl.Expr] | None` — custom aggregations for pass-through columns

**Output columns** (per symbol per bar):

| Column | Source |
|---|---|
| `timestamp` | Bar open time |
| `symbol` | Group key |
| `open` | First `price` in bucket |
| `high` | Max `price` |
| `low` | Min `price` |
| `close` | Last `price` |
| `volume` | Sum of `amount` (quote-denominated) |
| `trades` | Row count |
| `buy_volume` | Sum of `amount` where `side == "buy"` |
| `sell_volume` | Sum of `amount` where `side == "sell"` |
| `vwap` | Volume-weighted average price |

When `trader` column is present:
| `unique_traders` | `n_unique(trader)` |

When `pool_reserve` column is present:
| `pool_reserve_last` | Last `pool_reserve` value |

When `exchange_rate` is provided:
| `volume_usd` | `volume * interpolated_rate` |
| `buy_volume_usd` | `buy_volume * interpolated_rate` |
| `sell_volume_usd` | `sell_volume * interpolated_rate` |

### 1.3 Trade-count bars (alternative aggregation)

`aggregate_trades_by_count(df, n_trades, ...) -> pl.DataFrame`

Groups every N trades into one bar instead of using fixed time intervals. Useful for tokens with highly irregular activity.

---

## Phase 2: Dynamic Universe Support

Builds on the long-format engine from Phase 0 to handle assets that appear and disappear over time.

### 2.1 Universe provider

`UniverseProvider` protocol that the engine calls each bar to determine tradeable symbols:

```python
class UniverseProvider(Protocol):
    def get_universe(self, timestamp: Any, available_symbols: list[str]) -> list[str]:
        """Return the subset of symbols the strategy is allowed to trade this bar."""
        ...
```

Built-in implementations:
- `AllSymbols()` — pass through everything available
- `VolumeFilter(min_volume, lookback)` — only symbols exceeding volume threshold
- `AgeFilter(min_bars)` — only symbols that have existed for N bars
- `TopN(n, sort_by)` — top N symbols by a metric
- `CompositeFilter(*filters)` — chain multiple filters

### 2.2 Token lifecycle tracking

The engine tracks per-symbol metadata:
- `first_seen_bar: dict[str, int]` — when each symbol first appeared
- `bar_count: dict[str, int]` — how many bars each symbol has been active
- Accessible from `BacktestContext` so strategies can filter by token age

---

## Phase 3: DeFi / Memecoin Indicators

**New module**: `indicators_defi.py`

Polars expressions designed for the aggregated bar data with the extra columns from Phase 1.

### Token activity indicators
- `token_age(symbol_col)` — bars since first appearance
- `buy_sell_ratio(buy_vol_col, sell_vol_col)` — buy_volume / total_volume per bar
- `net_flow(buy_vol_col, sell_vol_col)` — buy_volume - sell_volume
- `trade_intensity(trades_col, window)` — rolling trade count acceleration
- `unique_trader_growth(traders_col, window)` — rate of new unique traders

### Liquidity indicators
- `pool_depth(reserve_col)` — raw pool reserve as liquidity proxy
- `price_impact_estimate(amount, reserve_col)` — estimated slippage for a given trade size based on constant-product AMM formula
- `liquidity_ratio(volume_col, reserve_col)` — volume relative to pool depth

### Momentum indicators (memecoin-tuned)
- `launch_velocity(price_col, window)` — price change rate in first N bars after token appears
- `pump_detector(price_col, volume_col, thresholds)` — flags bars with simultaneous price and volume spikes
- `rug_pull_detector(price_col, volume_col, sell_vol_col)` — flags sharp price drops with high sell volume

---

## Phase 4: AMM-Aware Execution Model

### 4.1 Slippage model protocol

```python
class SlippageModel(Protocol):
    def calculate(self, price: float, size: float, context: dict[str, Any]) -> float:
        """Return the slippage-adjusted execution price."""
        ...
```

Built-in implementations:
- `FlatSlippage(pct)` — existing behavior, extracted into protocol
- `AMMSlippage()` — constant-product formula: `slippage = trade_size / (pool_reserve + trade_size)`. Requires `pool_reserve` in bar data, passed via `context`.

### 4.2 Fee model preset

Solana DEX fee preset using existing commission infrastructure:

```python
SOLANA_PUMPFUN = FixedPlusPercentCommission(fixed=0.000005, percent=0.01)
```

### 4.3 Execution realism (stretch)

- Configurable transaction failure rate (Solana txs can fail)
- Priority fee modeling (higher fee = earlier execution in block)
- Minimum trade size enforcement (dust threshold)

---

## Phase 5: Exchange Rate Support

Allow strategies to reason about USD-denominated values when the underlying data is in a non-USD quote currency (e.g. SOL).

- `Engine` accepts an optional `exchange_rate: pl.DataFrame` with `(timestamp, rate)` columns
- Rate is interpolated to bar timestamps (forward-fill)
- `BacktestMetrics` can report in both quote and USD terms
- Portfolio equity curve optionally converted to USD for metrics like Sharpe

---

## Implementation Order

| Priority | Phase | Depends on | Estimated scope |
|---|---|---|---|
| P0 | 0.1-0.4 Long-format engine + revised context | — | Large |
| P0 | 0.5 Atomic rebalance on Portfolio | 0.1-0.4 | Medium |
| P0 | 0.6 Retire weight_backtest.py | 0.5 | Medium |
| P0 | 0.7 Backward compatibility verification | 0.6 | Medium (mostly tests) |
| P0 | 1.1 Trade validation | — | Small |
| P0 | 1.2 Time-based aggregation | 1.1 | Medium |
| P1 | 2.1 Universe provider | 0.1-0.4 | Medium |
| P1 | 2.2 Token lifecycle tracking | 0.1-0.4 | Small |
| P1 | 3 DeFi indicators | 1.2 | Medium |
| P1 | 1.3 Trade-count bars | 1.2 | Small |
| P2 | 4.1-4.2 AMM slippage + fee presets | 0.1-0.4 | Medium |
| P2 | 5 Exchange rate support | 1.2 | Medium |
| P3 | 4.3 Execution realism | 4.1 | Small |
