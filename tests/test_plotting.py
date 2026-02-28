"""Tests for the plotting module."""

import os
import tempfile

import polars as pl
import pytest

from polarbtest import Engine, Strategy, indicators
from polarbtest.plotting import plot_backtest, plot_returns_distribution


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


def _make_ohlcv_data():
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

    def test_plot_with_candlestick(self, engine_with_trades):
        fig = plot_backtest(engine_with_trades)
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Candlestick" in trace_types

    def test_plot_without_ohlc(self):
        df = pl.DataFrame({"close": [100 + i for i in range(20)]})
        engine = Engine(strategy=PassthroughStrategy(), data=df, initial_cash=100_000)
        engine.run()
        fig = plot_backtest(engine)
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Scatter" in trace_types
        assert "Candlestick" not in trace_types

    def test_plot_with_volume(self, engine_with_trades):
        fig = plot_backtest(engine_with_trades, show_volume=True)
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Bar" in trace_types

    def test_plot_without_volume(self, engine_with_trades):
        fig = plot_backtest(engine_with_trades, show_volume=False)
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Bar" not in trace_types

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

    def test_drawdown_subplot(self, engine_with_trades):
        fig = plot_backtest(engine_with_trades)
        names = [t.name for t in fig.data]
        assert "Drawdown" in names

    def test_equity_subplot(self, engine_with_trades):
        fig = plot_backtest(engine_with_trades)
        names = [t.name for t in fig.data]
        assert "Equity" in names


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
