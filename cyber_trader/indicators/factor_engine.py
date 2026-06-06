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
from nautilus_trader.indicators.momentum import RelativeStrengthIndex, Stochastics
from nautilus_trader.indicators.volatility import BollingerBands, AverageTrueRange, KeltnerChannel


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


class KeltnerBreakoutFactor(Factor):
    """Keltner Channel position: trend-following signal based on price vs. channel.

    Price above middle → bullish (+score), below → bearish (-score).
    At/above upper band → score ≈ +1 (strong breakout long).
    At/below lower band → score ≈ -1 (strong breakout short).
    """

    def __init__(self, period: int = 20, k: float = 2.0, weight: float = 1.0) -> None:
        super().__init__(f"KC_BREAK({period},{k})", weight)
        self._kc = KeltnerChannel(period, k)
        self._last_close: float = 0.0

    @property
    def is_initialized(self) -> bool:
        return self._kc.initialized

    def update(self, bar: Bar) -> None:
        self._last_close = bar.close.as_double()
        self._kc.update_raw(bar.high.as_double(), bar.low.as_double(), self._last_close)

    def score(self) -> float:
        if not self.is_initialized or self._last_close == 0.0:
            return 0.0
        upper = self._kc.upper
        middle = self._kc.middle
        lower = self._kc.lower
        if self._last_close >= middle:
            half = upper - middle
            if half == 0:
                return 0.0
            return max(-1.0, min(1.0, (self._last_close - middle) / half))
        else:
            half = middle - lower
            if half == 0:
                return 0.0
            return max(-1.0, min(1.0, (self._last_close - middle) / half))


class StochasticFactor(Factor):
    """Stochastic %K mapped to [-1, +1] as a trend-following momentum signal.

    %K > 50 → bullish momentum (positive score).
    %K < 50 → bearish momentum (negative score).
    """

    def __init__(self, period_k: int = 14, period_d: int = 3, weight: float = 1.0) -> None:
        super().__init__(f"STOCH({period_k},{period_d})", weight)
        self._stoch = Stochastics(period_k, period_d)

    @property
    def is_initialized(self) -> bool:
        return self._stoch.initialized

    def update(self, bar: Bar) -> None:
        self._stoch.update_raw(bar.high.as_double(), bar.low.as_double(), bar.close.as_double())

    def score(self) -> float:
        if not self.is_initialized:
            return 0.0
        return max(-1.0, min(1.0, (self._stoch.value_k - 50.0) / 50.0))


# ── ADX regime filter ─────────────────────────────────────────────────────────

class _ADXFilter:
    """Wilder's Average Directional Index.

    ADX > threshold → trending market (entries allowed).
    ADX ≤ threshold → ranging/choppy market (entries blocked).
    Threshold of 20 is the standard "no trend" cutoff.
    """

    def __init__(self, period: int = 14) -> None:
        self._period = period
        self._prev_high: float = 0.0
        self._prev_low: float = 0.0
        self._prev_close: float = 0.0

        # Wilder-smoothed accumulators (initialised after first full period)
        self._smooth_tr: float = 0.0
        self._smooth_dm_plus: float = 0.0
        self._smooth_dm_minus: float = 0.0

        # First-period sums (before Wilder smoothing kicks in)
        self._tr_sum: float = 0.0
        self._dm_plus_sum: float = 0.0
        self._dm_minus_sum: float = 0.0

        # ADX itself needs another full period of DX values
        self._dx_sum: float = 0.0
        self._dx_count: int = 0
        self._adx: float = 0.0

        self._count: int = 0
        self._initialized: bool = False

    @property
    def value(self) -> float:
        return self._adx

    @property
    def initialized(self) -> bool:
        return self._initialized

    def update(self, high: float, low: float, close: float) -> None:
        if self._count == 0:
            self._prev_high = high
            self._prev_low = low
            self._prev_close = close
            self._count += 1
            return

        tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        up_move = high - self._prev_high
        down_move = self._prev_low - low
        dm_plus = up_move if (up_move > down_move and up_move > 0) else 0.0
        dm_minus = down_move if (down_move > up_move and down_move > 0) else 0.0

        self._count += 1

        if self._count <= self._period:
            # Accumulate raw sums for the seed period
            self._tr_sum += tr
            self._dm_plus_sum += dm_plus
            self._dm_minus_sum += dm_minus
            if self._count == self._period:
                self._smooth_tr = self._tr_sum
                self._smooth_dm_plus = self._dm_plus_sum
                self._smooth_dm_minus = self._dm_minus_sum
        else:
            # Wilder smoothing: subtract 1/period of old value, add new value
            self._smooth_tr = self._smooth_tr - self._smooth_tr / self._period + tr
            self._smooth_dm_plus = self._smooth_dm_plus - self._smooth_dm_plus / self._period + dm_plus
            self._smooth_dm_minus = self._smooth_dm_minus - self._smooth_dm_minus / self._period + dm_minus

            if self._smooth_tr == 0:
                dx = 0.0
            else:
                di_plus = 100.0 * self._smooth_dm_plus / self._smooth_tr
                di_minus = 100.0 * self._smooth_dm_minus / self._smooth_tr
                di_sum = di_plus + di_minus
                dx = 100.0 * abs(di_plus - di_minus) / di_sum if di_sum != 0 else 0.0

            if not self._initialized:
                # Seed ADX with the average of the first `period` DX values
                self._dx_sum += dx
                self._dx_count += 1
                if self._dx_count >= self._period:
                    self._adx = self._dx_sum / self._dx_count
                    self._initialized = True
            else:
                self._adx = (self._adx * (self._period - 1) + dx) / self._period

        self._prev_high = high
        self._prev_low = low
        self._prev_close = close


# ── Factor Engine ─────────────────────────────────────────────────────────────

class FactorEngine:
    """Combines multiple factors into a single composite signal.

    Two optional regime gates control when entries are allowed:
    - trend_ema_period: price must be above EMA for longs / below for shorts.
    - adx_period / adx_threshold: ADX must exceed threshold (market is trending);
      when ADX ≤ threshold the market is ranging and all entries are blocked.
    """

    def __init__(
        self,
        factors: list[Factor],
        long_threshold: float = 0.3,
        short_threshold: float = -0.3,
        trend_ema_period: int = 0,
        adx_period: int = 0,
        adx_threshold: float = 20.0,
        adx_below: bool = False,
    ) -> None:
        self.factors = factors
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self._trend_ema = (
            ExponentialMovingAverage(trend_ema_period) if trend_ema_period > 0 else None
        )
        self._adx_filter = _ADXFilter(adx_period) if adx_period > 0 else None
        self._adx_threshold = adx_threshold
        # adx_below=True  → entries only when ADX ≤ threshold (ranging/mean-reversion mode)
        # adx_below=False → entries only when ADX > threshold (trending mode, default)
        self._adx_below = adx_below
        self._last_close: float = 0.0

    @property
    def is_initialized(self) -> bool:
        factors_ready = all(f.is_initialized for f in self.factors)
        regime_ready = self._trend_ema is None or self._trend_ema.initialized
        adx_ready = self._adx_filter is None or self._adx_filter.initialized
        return factors_ready and regime_ready and adx_ready

    def update(self, bar: Bar) -> None:
        self._last_close = bar.close.as_double()
        if self._trend_ema is not None:
            self._trend_ema.update_raw(self._last_close)
        if self._adx_filter is not None:
            self._adx_filter.update(
                bar.high.as_double(),
                bar.low.as_double(),
                self._last_close,
            )
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

    def _adx_allows_entry(self) -> bool:
        if self._adx_filter is None or not self._adx_filter.initialized:
            return True
        adx = self._adx_filter.value
        return adx <= self._adx_threshold if self._adx_below else adx > self._adx_threshold

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
            and self._adx_allows_entry()
        )

    def is_short(self) -> bool:
        return (
            self.is_initialized
            and self.composite_score() <= self.short_threshold
            and self._regime_allows_short()
            and self._adx_allows_entry()
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
