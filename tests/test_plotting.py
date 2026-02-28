"""Tests for the plotting module."""

import os
import tempfile
from datetime import date, datetime, timedelta

import polars as pl
import pytest

from polarbtest import Engine, Strategy, indicators
from polarbtest.plotting import plot_backtest, plot_returns_distribution
from polarbtest.plotting.charts import _format_duration

CLOSE_PRICES = [100, 102, 104, 101, 99, 97, 100, 103, 105, 102, 98, 96, 99, 103, 106, 104, 100, 97, 101, 105]


class SMAStrategy(Strategy):
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([indicators.sma("close", 3).alias("sma_3")])

    def next(self, ctx):
        if ctx.row["close"] > ctx.row["sma_3"]:
            ctx.portfolio.order_target_percent("asset", 1.0)
        else:
            ctx.portfolio.close_position("asset")


class PassthroughStrategy(Strategy):
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def next(self, ctx):
        pass


def _make_ohlcv_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": list(range(20)),
            "open": [c - 0.5 for c in CLOSE_PRICES],
            "high": [c + 1 for c in CLOSE_PRICES],
            "low": [c - 1 for c in CLOSE_PRICES],
            "close": CLOSE_PRICES,
            "volume": [1000 + i * 100 for i in range(20)],
        }
    )


@pytest.fixture
def sample_ohlcv_data():
    return _make_ohlcv_data()


@pytest.fixture
def engine_with_trades(sample_ohlcv_data):
    engine = Engine(strategy=SMAStrategy(), data=sample_ohlcv_data, initial_cash=100_000)
    engine.run()
    return engine


@pytest.fixture
def engine_no_trades(sample_ohlcv_data):
    engine = Engine(strategy=PassthroughStrategy(), data=sample_ohlcv_data, initial_cash=100_000)
    engine.run()
    return engine


class TestPlotBacktest:
    def test_basic_plot(self, engine_with_trades):
        fig = plot_backtest(engine_with_trades)
        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_with_ohlc_bars(self, engine_with_trades):
        """OHLC data should produce a Candlestick trace."""
        fig = plot_backtest(engine_with_trades)
        names = [t.name for t in fig.data]
        assert "OHLC" in names

    def test_plot_without_ohlc(self):
        """Close-only data should produce line chart, no OHLC bars."""
        df = pl.DataFrame({"close": [100 + i for i in range(20)]})
        engine = Engine(strategy=PassthroughStrategy(), data=df, initial_cash=100_000)
        engine.run()
        fig = plot_backtest(engine)
        names = [t.name for t in fig.data]
        assert "Price" in names
        assert "OHLC" not in names

    def test_plot_with_volume(self, engine_with_trades):
        fig = plot_backtest(engine_with_trades, show_volume=True)
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Bar" in trace_types

    def test_plot_without_volume(self, engine_with_trades):
        fig = plot_backtest(engine_with_trades, show_volume=False)
        names = [t.name for t in fig.data if t.name]
        assert "Volume" not in names

    def test_plot_with_indicators(self, engine_with_trades):
        fig = plot_backtest(engine_with_trades, indicators=["sma_3"])
        names = [t.name for t in fig.data]
        assert "sma_3" in names

    def test_plot_with_bands(self):
        df = pl.DataFrame(
            {
                "timestamp": list(range(20)),
                "close": [100 + i for i in range(20)],
                "upper": [105 + i for i in range(20)],
                "lower": [95 + i for i in range(20)],
            }
        )
        engine = Engine(strategy=PassthroughStrategy(), data=df, initial_cash=100_000)
        engine.run()
        fig = plot_backtest(engine, bands=[("upper", "lower")])
        names = [t.name for t in fig.data]
        assert "upper" in names
        assert "lower" in names

    def test_trade_markers(self, engine_with_trades):
        fig = plot_backtest(engine_with_trades, show_trades=True)
        names = [t.name for t in fig.data]
        # Oscillating prices should generate at least one closed trade
        assert "Entry" in names
        assert "Exit" in names

    def test_no_trade_markers(self, engine_no_trades):
        fig = plot_backtest(engine_no_trades, show_trades=True)
        names = [t.name for t in fig.data]
        assert "Entry" not in names

    def test_save_html(self, engine_with_trades):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            plot_backtest(engine_with_trades, save_html=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_custom_title_and_height(self, engine_with_trades):
        fig = plot_backtest(engine_with_trades, title="My Test", height=600)
        assert fig.layout.title.text == "My Test"
        assert fig.layout.height == 600

    def test_engine_not_run_raises(self, sample_ohlcv_data):
        engine = Engine(strategy=SMAStrategy(), data=sample_ohlcv_data)
        with pytest.raises(ValueError, match="Engine must be run"):
            plot_backtest(engine)

    def test_equity_subplot_with_stats(self, engine_with_trades):
        fig = plot_backtest(engine_with_trades)
        names = [t.name for t in fig.data]
        assert "Equity" in names
        # Stats markers should be present
        assert any("Peak" in n for n in names if n)
        assert any("Final" in n for n in names if n)
        assert any("Max Drawdown" in n for n in names if n)

    def test_pnl_subplot(self, engine_with_trades):
        fig = plot_backtest(engine_with_trades)
        names = [t.name for t in fig.data]
        assert "Trades P/L" in names

    def test_trade_count_in_title(self, engine_with_trades):
        fig = plot_backtest(engine_with_trades)
        # Subplot title should contain "Trades (N)"
        annotations = [a.text for a in fig.layout.annotations if hasattr(a, "text")]
        assert any("Trades (" in a for a in annotations)

    def test_colored_candles(self):
        """Candlestick trace should have green increasing and red decreasing colors."""
        # Explicit data with both up and down candles
        df = pl.DataFrame(
            {
                "timestamp": list(range(4)),
                "open": [100.0, 105.0, 100.0, 108.0],  # bar 0,2: up; bar 1,3: down
                "high": [110.0, 110.0, 110.0, 110.0],
                "low": [90.0, 90.0, 90.0, 90.0],
                "close": [105.0, 100.0, 108.0, 102.0],
            }
        )
        engine = Engine(strategy=PassthroughStrategy(), data=df, initial_cash=100_000)
        engine.run()
        fig = plot_backtest(engine)
        ohlc_traces = [t for t in fig.data if t.name == "OHLC"]
        assert len(ohlc_traces) == 1
        candle = ohlc_traces[0]
        assert candle.increasing.fillcolor == "#26a69a"  # green
        assert candle.decreasing.fillcolor == "#ef5350"  # red

    def test_trade_arrows(self, engine_with_trades):
        """Trade arrows should create annotations with arrowhead."""
        fig = plot_backtest(engine_with_trades, show_trades=True)
        arrow_annotations = [a for a in fig.layout.annotations if getattr(a, "arrowhead", None)]
        assert len(arrow_annotations) > 0


class TestPlotReturnsDistribution:
    def test_basic_histogram(self, engine_with_trades):
        fig = plot_returns_distribution(engine_with_trades)
        assert fig is not None
        assert len(fig.data) > 0
        assert type(fig.data[0]).__name__ == "Histogram"

    def test_save_html(self, engine_with_trades):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            plot_returns_distribution(engine_with_trades, save_html=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_engine_not_run_raises(self, sample_ohlcv_data):
        engine = Engine(strategy=SMAStrategy(), data=sample_ohlcv_data)
        with pytest.raises(ValueError, match="Engine must be run"):
            plot_returns_distribution(engine)

    def test_custom_bins(self, engine_with_trades):
        fig = plot_returns_distribution(engine_with_trades, bins=10)
        assert fig.data[0].nbinsx == 10


class TestFormatDuration:
    def test_integer_timestamps_show_bars(self):
        timestamps = list(range(100))
        assert _format_duration(timestamps, 10, 50, 40) == "40 bars"

    def test_date_timestamps_show_days(self):
        base = date(2024, 1, 1)
        timestamps = [base + timedelta(days=i) for i in range(100)]
        assert _format_duration(timestamps, 0, 42, 42) == "42 days"

    def test_date_timestamps_show_months(self):
        base = date(2024, 1, 1)
        timestamps = [base + timedelta(days=i) for i in range(200)]
        result = _format_duration(timestamps, 0, 90, 90)
        assert "months" in result

    def test_datetime_timestamps_show_days(self):
        base = datetime(2024, 1, 1, 9, 30)
        timestamps = [base + timedelta(days=i) for i in range(100)]
        assert _format_duration(timestamps, 0, 30, 30) == "30 days"

    def test_datetime_short_duration_shows_hours(self):
        base = datetime(2024, 1, 1, 9, 0)
        timestamps = [base + timedelta(hours=i) for i in range(50)]
        result = _format_duration(timestamps, 0, 5, 5)
        assert "hours" in result

    def test_dd_duration_with_datetime_in_plot(self):
        """Full integration: datetime timestamps should show days in DD duration legend."""
        base = datetime(2024, 1, 1)
        n = 20
        timestamps = [base + timedelta(days=i) for i in range(n)]
        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100 - 0.5] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "close": CLOSE_PRICES,
                "volume": [1000] * n,
            }
        )
        engine = Engine(strategy=SMAStrategy(), data=df, initial_cash=100_000)
        engine.run()
        fig = plot_backtest(engine)
        names = [t.name for t in fig.data if t.name and "Max Dd Dur" in t.name]
        assert len(names) >= 1
        # Should show days, not bars
        assert "days" in names[0]
        assert "bars" not in names[0]
