"""Multi-factor signal engine.

Each Factor produces a normalised score in [-1, +1]:
  +1 = strong long signal
  -1 = strong short signal
   0 = neutral

The FactorEngine combines factors by weight to produce a composite score
and decides whether the score crosses the entry/exit thresholds.

Import paths verified against nautilus_trader 1.221+:
  averages  → ExponentialMovingAverage
  trend     → MovingAverageConvergenceDivergence  (fast_period, slow_period only)
  momentum  → RelativeStrengthIndex
  volatility→ BollingerBands (update_raw(high, low, close)), AverageTrueRange
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
from nautilus_trader.model.data import Bar
from nautilus_trader.indicators.averages import ExponentialMovingAverage
from nautilus_trader.indicators.trend import MovingAverageConvergenceDivergence
from nautilus_trader.indicators.momentum import RelativeStrengthIndex
from nautilus_trader.indicators.volatility import BollingerBands, AverageTrueRange


@dataclass(frozen=True)
class FactorSignal:
    name: str
    score: float          # [-1, +1]
    weight: float
    weighted_score: float
    is_initialized: bool


class Factor(ABC):
    """Base class for a single quantitative factor."""

    def __init__(self, name: str, weight: float = 1.0) -> None:
        self.name = name
        self.weight = weight

    @property
    @abstractmethod
    def is_initialized(self) -> bool: ...

    @abstractmethod
    def update(self, bar: Bar) -> None: ...

    @abstractmethod
    def score(self) -> float:
        """Return normalised score in [-1, +1]."""
        ...

    def weighted_score(self) -> float:
        if not self.is_initialized:
            return 0.0
        return self.weight * self.score()

    def to_signal(self) -> FactorSignal:
        s = self.score() if self.is_initialized else 0.0
        return FactorSignal(
            name=self.name,
            score=s,
            weight=self.weight,
            weighted_score=self.weight * s,
            is_initialized=self.is_initialized,
        )


# ── Concrete factors ──────────────────────────────────────────────────────────

class EMACrossoverFactor(Factor):
    """EMA fast/slow crossover → normalised by midpoint distance."""

    def __init__(self, fast: int = 9, slow: int = 21, weight: float = 1.0) -> None:
        super().__init__(f"EMA_CROSS({fast},{slow})", weight)
        self._fast = ExponentialMovingAverage(fast)
        self._slow = ExponentialMovingAverage(slow)

    @property
    def is_initialized(self) -> bool:
        return self._fast.initialized and self._slow.initialized

    def update(self, bar: Bar) -> None:
        price = bar.close.as_double()
        self._fast.update_raw(price)
        self._slow.update_raw(price)

    def score(self) -> float:
        if not self.is_initialized:
            return 0.0
        diff = self._fast.value - self._slow.value
        mid = (self._fast.value + self._slow.value) / 2
        if mid == 0:
            return 0.0
        return max(-1.0, min(1.0, diff / mid * 20))


class MACDFactor(Factor):
    """MACD line (fast EMA - slow EMA) normalised by ATR.

    Note: nautilus_trader's MACD does not include a signal line or histogram;
    it exposes only the raw divergence value between the two MAs.
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        atr_period: int = 14,
        weight: float = 1.0,
    ) -> None:
        super().__init__(f"MACD({fast},{slow})", weight)
        self._macd = MovingAverageConvergenceDivergence(fast, slow)
        self._atr = AverageTrueRange(atr_period)

    @property
    def is_initialized(self) -> bool:
        return self._macd.initialized and self._atr.initialized

    def update(self, bar: Bar) -> None:
        close = bar.close.as_double()
        self._macd.update_raw(close)
        self._atr.update_raw(
            bar.high.as_double(),
            bar.low.as_double(),
            close,
        )

    def score(self) -> float:
        if not self.is_initialized:
            return 0.0
        atr = self._atr.value
        if atr == 0:
            return 0.0
        return max(-1.0, min(1.0, self._macd.value / atr))


class RSIFactor(Factor):
    """RSI mapped to [-1, +1]: oversold → +1 (long bias), overbought → -1."""

    def __init__(self, period: int = 14, weight: float = 1.0) -> None:
        super().__init__(f"RSI({period})", weight)
        self._rsi = RelativeStrengthIndex(period)

    @property
    def is_initialized(self) -> bool:
        return self._rsi.initialized

    def update(self, bar: Bar) -> None:
        self._rsi.update_raw(bar.close.as_double())

    def score(self) -> float:
        if not self.is_initialized:
            return 0.0
        rsi = self._rsi.value
        # RSI 50 = neutral; 30 = oversold (+1); 70 = overbought (-1)
        return max(-1.0, min(1.0, (50 - rsi) / 50))


class BollingerFactor(Factor):
    """Bollinger Band position: below lower → long, above upper → short."""

    def __init__(self, period: int = 20, k: float = 2.0, weight: float = 1.0) -> None:
        super().__init__(f"BB({period},{k})", weight)
        self._bb = BollingerBands(period, k)
        self._last_close: float = 0.0

    @property
    def is_initialized(self) -> bool:
        return self._bb.initialized

    def update(self, bar: Bar) -> None:
        self._last_close = bar.close.as_double()
        # BollingerBands.update_raw takes (high, low, close)
        self._bb.update_raw(
            bar.high.as_double(),
            bar.low.as_double(),
            self._last_close,
        )

    def score(self) -> float:
        if not self.is_initialized or self._last_close == 0.0:
            return 0.0
        upper = self._bb.upper
        lower = self._bb.lower
        mid = self._bb.middle
        band_half = (upper - lower) / 2
        if band_half == 0:
            return 0.0
        # Positive score when price is below mid (mean-reversion long signal)
        return max(-1.0, min(1.0, (mid - self._last_close) / band_half))


class MomentumFactor(Factor):
    """Rate-of-change momentum over N bars."""

    def __init__(self, period: int = 10, weight: float = 1.0) -> None:
        super().__init__(f"MOM({period})", weight)
        self._period = period
        self._prices: list[float] = []

    @property
    def is_initialized(self) -> bool:
        return len(self._prices) >= self._period + 1

    def update(self, bar: Bar) -> None:
        self._prices.append(bar.close.as_double())
        if len(self._prices) > self._period + 1:
            self._prices.pop(0)

    def score(self) -> float:
        if not self.is_initialized:
            return 0.0
        old = self._prices[0]
        new = self._prices[-1]
        if old == 0:
            return 0.0
        roc = (new - old) / old
        # Normalise: assume ±5% move over period is a strong signal
        return max(-1.0, min(1.0, roc / 0.05))


class VolumeMomentumFactor(Factor):
    """Volume spike as confirmation of price momentum."""

    def __init__(self, period: int = 20, weight: float = 0.5) -> None:
        super().__init__(f"VOLMOM({period})", weight)
        self._period = period
        self._volumes: list[float] = []
        self._closes: list[float] = []

    @property
    def is_initialized(self) -> bool:
        return len(self._volumes) >= self._period

    def update(self, bar: Bar) -> None:
        self._volumes.append(bar.volume.as_double())
        self._closes.append(bar.close.as_double())
        if len(self._volumes) > self._period:
            self._volumes.pop(0)
            self._closes.pop(0)

    def score(self) -> float:
        if not self.is_initialized:
            return 0.0
        avg_vol = sum(self._volumes[:-1]) / (len(self._volumes) - 1)
        if avg_vol == 0:
            return 0.0
        vol_ratio = self._volumes[-1] / avg_vol
        price_direction = 1.0 if self._closes[-1] > self._closes[-2] else -1.0
        # Volume spike (>1x above average) weighted by price direction
        spike_score = min(1.0, (vol_ratio - 1.0) / 1.0)
        return max(-1.0, min(1.0, price_direction * spike_score))


# ── Factor Engine ─────────────────────────────────────────────────────────────

class FactorEngine:
    """Combines multiple factors into a single composite signal.

    An optional trend-regime filter (a long-period EMA) gates entries so that
    longs are only taken while price trades above the regime EMA and shorts only
    below it. This removes most counter-trend whipsaw — the single biggest source
    of losing trades for crossover/oscillator factors.
    """

    def __init__(
        self,
        factors: list[Factor],
        long_threshold: float = 0.3,
        short_threshold: float = -0.3,
        trend_ema_period: int = 0,
    ) -> None:
        self.factors = factors
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self._trend_ema = (
            ExponentialMovingAverage(trend_ema_period) if trend_ema_period > 0 else None
        )
        self._last_close: float = 0.0

    @property
    def is_initialized(self) -> bool:
        factors_ready = all(f.is_initialized for f in self.factors)
        regime_ready = self._trend_ema is None or self._trend_ema.initialized
        return factors_ready and regime_ready

    def update(self, bar: Bar) -> None:
        self._last_close = bar.close.as_double()
        if self._trend_ema is not None:
            self._trend_ema.update_raw(self._last_close)
        for f in self.factors:
            f.update(bar)

    def _regime_allows_long(self) -> bool:
        if self._trend_ema is None:
            return True
        return self._last_close >= self._trend_ema.value

    def _regime_allows_short(self) -> bool:
        if self._trend_ema is None:
            return True
        return self._last_close <= self._trend_ema.value

    def composite_score(self) -> float:
        total_weight = sum(abs(f.weight) for f in self.factors if f.is_initialized)
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(f.weighted_score() for f in self.factors if f.is_initialized)
        return weighted_sum / total_weight

    def factor_scores(self) -> dict[str, float]:
        return {f.name: (f.score() if f.is_initialized else 0.0) for f in self.factors}

    def signal_details(self) -> list[FactorSignal]:
        return [f.to_signal() for f in self.factors]

    def is_long(self) -> bool:
        return (
            self.is_initialized
            and self.composite_score() >= self.long_threshold
            and self._regime_allows_long()
        )

    def is_short(self) -> bool:
        return (
            self.is_initialized
            and self.composite_score() <= self.short_threshold
            and self._regime_allows_short()
        )

    def is_neutral(self) -> bool:
        return not self.is_long() and not self.is_short()


# ── Auxiliary data-backed factors ─────────────────────────────────────────────

class FundingRateFactor(Factor):
    """Perpetual swap funding rate as a contrarian sentiment signal.

    Logic (contrarian):
      High positive funding → longs crowded → bearish  (score → -1)
      High negative funding → shorts crowded → bullish (score → +1)
      Neutral (~0)         → score = 0

    score = -clamp(rate / threshold, -1, +1)
    Default threshold 0.0003 means a funding rate of 0.03%/8h maps to ±1.
    """

    def __init__(
        self,
        data: pd.Series | None,
        threshold: float = 0.0003,
        weight: float = 1.0,
    ) -> None:
        super().__init__("FUNDING_RATE", weight)
        self._threshold = threshold
        self._current_rate: float = 0.0
        self._score_series: pd.Series | None = None
        self._ready: bool = False

        if data is not None and not data.empty:
            # Precompute score series for fast asof lookup during backtesting
            clipped = (-data / threshold).clip(-1.0, 1.0)
            self._score_series = clipped
            self._ready = True

    @property
    def is_initialized(self) -> bool:
        return self._ready

    def update(self, bar: Bar) -> None:
        if self._score_series is None:
            return
        ts = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")
        val = self._score_series.asof(ts)
        if pd.notna(val):
            self._current_rate = float(val)

    def score(self) -> float:
        return self._current_rate

    def set_rate(self, rate: float) -> None:
        """Live trading: inject current funding rate directly."""
        if self._threshold != 0:
            self._current_rate = max(-1.0, min(1.0, -rate / self._threshold))
        self._ready = True


class LongShortRatioFactor(Factor):
    """Account-level long/short ratio as a contrarian sentiment signal.

    Uses a rolling z-score to normalise across recent history, then negates
    (contrarian): z-score above 0 (many longs) → bearish, below 0 → bullish.

    score = -clamp(z_score / 2, -1, +1)   (z=2 → max score)
    """

    def __init__(
        self,
        data: pd.Series | None,
        zscore_window: int = 48,   # 48 × 1H ≈ 2 days of look-back
        weight: float = 1.0,
    ) -> None:
        super().__init__("LS_RATIO", weight)
        self._current_score: float = 0.0
        self._score_series: pd.Series | None = None
        self._ready: bool = False

        if data is not None and not data.empty:
            min_periods = max(1, zscore_window // 4)
            roll_mean = data.rolling(zscore_window, min_periods=min_periods).mean()
            roll_std = (
                data.rolling(zscore_window, min_periods=min_periods)
                .std()
                .replace(0.0, float("nan"))
                .fillna(1e-9)
            )
            z = (data - roll_mean) / roll_std
            # Contrarian: negate z, scale so z=±2 → score=±1
            self._score_series = (-z / 2.0).clip(-1.0, 1.0)
            self._ready = True

    @property
    def is_initialized(self) -> bool:
        return self._ready

    def update(self, bar: Bar) -> None:
        if self._score_series is None:
            return
        ts = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")
        val = self._score_series.asof(ts)
        if pd.notna(val):
            self._current_score = float(val)

    def score(self) -> float:
        return self._current_score

    def set_ratio(self, ratio: float, rolling_mean: float, rolling_std: float) -> None:
        """Live trading: inject the current ratio with its rolling statistics."""
        if rolling_std > 0:
            z = (ratio - rolling_mean) / rolling_std
            self._current_score = max(-1.0, min(1.0, -z / 2.0))
        self._ready = True
