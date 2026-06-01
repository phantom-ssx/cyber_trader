"""Tests for the multi-factor engine and individual factors."""

from __future__ import annotations

import pytest
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import AggregationSource, BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Price, Quantity

from cyber_trader.indicators.factor_engine import (
    BollingerFactor,
    EMACrossoverFactor,
    FactorEngine,
    MACDFactor,
    MomentumFactor,
    RSIFactor,
    VolumeMomentumFactor,
)


def _make_bar(close: float, high: float | None = None, low: float | None = None, volume: float = 1000.0, ts: int = 0) -> Bar:
    instrument_id = InstrumentId(Symbol("ETH-USDT-SWAP"), Venue("OKX"))
    bar_type = BarType(
        instrument_id,
        BarSpecification(1, BarAggregation.HOUR, PriceType.LAST),
        AggregationSource.EXTERNAL,
    )
    h = high or close * 1.001
    l = low or close * 0.999
    return Bar(
        bar_type=bar_type,
        open=Price(close, 2),
        high=Price(h, 2),
        low=Price(l, 2),
        close=Price(close, 2),
        volume=Quantity(volume, 2),
        ts_event=ts,
        ts_init=ts,
    )


def _feed(factor, prices: list[float]) -> None:
    for i, p in enumerate(prices):
        factor.update(_make_bar(p, ts=i * 3_600_000_000_000))


class TestEMACrossoverFactor:
    def test_not_initialized_before_warmup(self):
        f = EMACrossoverFactor(fast=3, slow=5)
        assert not f.is_initialized

    def test_initialized_after_warmup(self):
        f = EMACrossoverFactor(fast=3, slow=5)
        _feed(f, [100.0] * 10)
        assert f.is_initialized

    def test_score_range(self):
        f = EMACrossoverFactor(fast=3, slow=10)
        prices = [100 + i * 0.5 for i in range(30)]  # uptrend
        _feed(f, prices)
        score = f.score()
        assert -1.0 <= score <= 1.0

    def test_uptrend_positive_score(self):
        f = EMACrossoverFactor(fast=3, slow=10)
        _feed(f, [100 + i for i in range(30)])  # strong uptrend
        assert f.score() > 0

    def test_downtrend_negative_score(self):
        f = EMACrossoverFactor(fast=3, slow=10)
        _feed(f, [200 - i for i in range(30)])  # strong downtrend
        assert f.score() < 0


class TestRSIFactor:
    def test_oversold_positive_score(self):
        f = RSIFactor(period=5)
        _feed(f, [100 - i * 2 for i in range(20)])  # steep decline → oversold
        if f.is_initialized:
            assert f.score() > 0  # oversold → long bias

    def test_score_bounded(self):
        f = RSIFactor(period=5)
        _feed(f, list(range(100, 200, 2)))
        if f.is_initialized:
            assert -1.0 <= f.score() <= 1.0


class TestMomentumFactor:
    def test_not_initialized_before_period(self):
        f = MomentumFactor(period=10)
        _feed(f, [100.0] * 5)
        assert not f.is_initialized

    def test_initialized_after_period(self):
        f = MomentumFactor(period=10)
        _feed(f, [100.0] * 12)
        assert f.is_initialized

    def test_positive_momentum(self):
        f = MomentumFactor(period=5)
        _feed(f, [100, 102, 104, 106, 108, 110, 112])
        assert f.score() > 0

    def test_negative_momentum(self):
        f = MomentumFactor(period=5)
        _feed(f, [110, 108, 106, 104, 102, 100, 98])
        assert f.score() < 0


class TestFactorEngine:
    def _make_engine(self) -> FactorEngine:
        return FactorEngine(
            factors=[
                EMACrossoverFactor(fast=3, slow=5, weight=0.6),
                MomentumFactor(period=5, weight=0.4),
            ],
            long_threshold=0.3,
            short_threshold=-0.3,
        )

    def test_composite_score_bounded(self):
        engine = self._make_engine()
        bars = [_make_bar(100 + i) for i in range(20)]
        for b in bars:
            engine.update(b)
        if engine.is_initialized:
            score = engine.composite_score()
            assert -1.0 <= score <= 1.0

    def test_long_signal_on_uptrend(self):
        engine = self._make_engine()
        bars = [_make_bar(100 + i * 2) for i in range(30)]
        for b in bars:
            engine.update(b)
        if engine.is_initialized:
            # Strong uptrend should trigger long
            assert engine.is_long() or engine.composite_score() > 0

    def test_factor_scores_dict(self):
        engine = self._make_engine()
        bars = [_make_bar(100.0) for _ in range(20)]
        for b in bars:
            engine.update(b)
        scores = engine.factor_scores()
        assert len(scores) == 2
        for v in scores.values():
            assert -1.0 <= v <= 1.0


class TestFeishuNotifier:
    """Smoke-test the notifier (no actual HTTP call)."""

    def test_notifier_skips_without_url(self):
        from cyber_trader.notifications.feishu import FeishuNotifier, TradeSignal
        notifier = FeishuNotifier(webhook_url="")
        sig = TradeSignal(
            symbol="ETH-USDT-SWAP",
            direction="LONG",
            strategy="Test",
            price=3000.0,
            composite_score=0.5,
            factors={"EMA": 0.6, "MACD": 0.4},
            timeframe="1H",
        )
        result = notifier.send_signal(sig)
        assert result is False
