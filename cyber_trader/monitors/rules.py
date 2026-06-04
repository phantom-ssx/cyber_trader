"""Alert rule definitions for the volatility monitor.

To add a new rule:
  1. Subclass AlertRule and implement window_size + evaluate.
  2. Register it with @register_rule("your_type_name").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class AlertResult:
    label: str        # rule name, e.g. "单K线异动"
    pct: float        # magnitude of the move (always positive)
    from_price: float # price at the start of the window
    to_price: float   # price at the end of the window (current bar close)
    window_bars: int  # number of bars the window spans
    rising: bool      # True = net upward move


# ── Base class ────────────────────────────────────────────────────────────────

class AlertRule(ABC):
    """
    Stateless rule evaluated against a sliding window of closing prices.

    Each rule declares how many bars of history it needs (window_size).
    The monitor maintains a per-instrument deque sized to max(window_size)
    across all active rules.
    """

    @property
    @abstractmethod
    def window_size(self) -> int:
        """Minimum number of closes required to evaluate this rule."""

    @abstractmethod
    def evaluate(self, closes: deque[float]) -> AlertResult | None:
        """Return an AlertResult if the rule fires, else None."""

    def describe(self) -> str:
        return repr(self)


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[AlertRule]] = {}


def register_rule(type_name: str):
    def decorator(cls: type[AlertRule]) -> type[AlertRule]:
        _REGISTRY[type_name] = cls
        return cls
    return decorator


def build_rule(spec: dict[str, Any]) -> AlertRule:
    spec = dict(spec)
    rule_type = spec.pop("type")
    if rule_type not in _REGISTRY:
        raise ValueError(
            f"Unknown rule type '{rule_type}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[rule_type](**spec)


# ── Built-in rules ────────────────────────────────────────────────────────────

@register_rule("single_bar")
class SingleBarRule(AlertRule):
    """Alert when a single bar's abs % change >= threshold_pct."""

    def __init__(self, threshold_pct: float = 0.5) -> None:
        self.threshold_pct = threshold_pct

    @property
    def window_size(self) -> int:
        return 2

    def evaluate(self, closes: deque[float]) -> AlertResult | None:
        if len(closes) < 2:
            return None
        cl = list(closes)
        prev, curr = cl[-2], cl[-1]
        pct = abs(curr - prev) / prev * 100
        if pct >= self.threshold_pct:
            return AlertResult(
                label="单K线异动",
                pct=pct,
                from_price=prev,
                to_price=curr,
                window_bars=1,
                rising=curr > prev,
            )
        return None

    def describe(self) -> str:
        return f"SingleBar(≥{self.threshold_pct}%)"


@register_rule("cumulative")
class CumulativeRule(AlertRule):
    """Alert when the sum of abs % changes over `bars` bars >= threshold_pct."""

    def __init__(self, threshold_pct: float = 1.0, bars: int = 5) -> None:
        self.threshold_pct = threshold_pct
        self.bars = bars

    @property
    def window_size(self) -> int:
        return self.bars + 1

    def evaluate(self, closes: deque[float]) -> AlertResult | None:
        if len(closes) < self.bars + 1:
            return None
        cl = list(closes)[-(self.bars + 1):]
        cumulative = sum(
            abs(cl[i] - cl[i - 1]) / cl[i - 1] * 100
            for i in range(1, len(cl))
        )
        if cumulative >= self.threshold_pct:
            return AlertResult(
                label=f"累计{self.bars}分钟异动",
                pct=cumulative,
                from_price=cl[0],
                to_price=cl[-1],
                window_bars=self.bars,
                rising=cl[-1] > cl[0],
            )
        return None

    def describe(self) -> str:
        return f"Cumulative({self.bars}bars ≥{self.threshold_pct}%)"


@register_rule("directional")
class DirectionalRule(AlertRule):
    """
    Alert when price moves >= threshold_pct in the same direction over
    `bars` consecutive bars (all up or all down).
    """

    def __init__(self, threshold_pct: float = 0.8, bars: int = 3) -> None:
        self.threshold_pct = threshold_pct
        self.bars = bars

    @property
    def window_size(self) -> int:
        return self.bars + 1

    def evaluate(self, closes: deque[float]) -> AlertResult | None:
        if len(closes) < self.bars + 1:
            return None
        cl = list(closes)[-(self.bars + 1):]
        moves = [cl[i] - cl[i - 1] for i in range(1, len(cl))]
        if all(m > 0 for m in moves):
            pct = (cl[-1] - cl[0]) / cl[0] * 100
            if pct >= self.threshold_pct:
                return AlertResult(
                    label=f"持续上涨{self.bars}分钟",
                    pct=pct,
                    from_price=cl[0],
                    to_price=cl[-1],
                    window_bars=self.bars,
                    rising=True,
                )
        elif all(m < 0 for m in moves):
            pct = (cl[0] - cl[-1]) / cl[0] * 100
            if pct >= self.threshold_pct:
                return AlertResult(
                    label=f"持续下跌{self.bars}分钟",
                    pct=pct,
                    from_price=cl[0],
                    to_price=cl[-1],
                    window_bars=self.bars,
                    rising=False,
                )
        return None

    def describe(self) -> str:
        return f"Directional({self.bars}bars ≥{self.threshold_pct}%)"


@register_rule("round_level")
class RoundLevelRule(AlertRule):
    """
    Alert when price crosses a round-number boundary.
    interval=100 → alerts at 2700, 2800, 2900, …
    interval=1000 → alerts at 90000, 91000, …
    """

    def __init__(self, interval: float = 100.0) -> None:
        self.interval = interval

    @property
    def window_size(self) -> int:
        return 2

    def evaluate(self, closes: deque[float]) -> AlertResult | None:
        if len(closes) < 2:
            return None
        cl = list(closes)
        prev, curr = cl[-2], cl[-1]
        prev_level = int(prev / self.interval)
        curr_level = int(curr / self.interval)
        if curr_level != prev_level:
            crossed = curr_level * self.interval
            pct = abs(curr - prev) / prev * 100
            return AlertResult(
                label=f"突破整数关口 {crossed:,.0f}",
                pct=pct,
                from_price=prev,
                to_price=curr,
                window_bars=1,
                rising=curr > prev,
            )
        return None

    def describe(self) -> str:
        return f"RoundLevel(interval={self.interval:g})"
