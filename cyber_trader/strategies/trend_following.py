"""Trend-following strategy: EMA crossover + MACD confirmation."""

from __future__ import annotations

from nautilus_trader.config import StrategyConfig

from cyber_trader.indicators.factor_engine import (
    EMACrossoverFactor,
    FactorEngine,
    MACDFactor,
)
from .base import BaseStrategy, BaseStrategyConfig


class TrendFollowingConfig(BaseStrategyConfig, frozen=True):
    # EMA parameters
    ema_fast: int = 9
    ema_slow: int = 21

    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26

    # Factor weights (EMA has more weight in trend-following)
    ema_weight: float = 0.6
    macd_weight: float = 0.4

    # Override thresholds for trend-following (requires stronger signal)
    long_threshold: float = 0.25
    short_threshold: float = -0.25


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend-following strategy combining EMA crossover and MACD.

    Entry:
      LONG  when composite_score >= long_threshold
      SHORT when composite_score <= short_threshold

    Exit:
      Neutral zone (composite score between thresholds)
    """

    def __init__(self, config: TrendFollowingConfig) -> None:
        super().__init__(config)
        self._tf_config = config

    def build_factor_engine(self) -> FactorEngine:
        cfg = self._tf_config
        return FactorEngine(
            factors=[
                EMACrossoverFactor(
                    fast=cfg.ema_fast,
                    slow=cfg.ema_slow,
                    weight=cfg.ema_weight,
                ),
                MACDFactor(
                    fast=cfg.macd_fast,
                    slow=cfg.macd_slow,
                    weight=cfg.macd_weight,
                ),
            ],
            long_threshold=cfg.long_threshold,
            short_threshold=cfg.short_threshold,
        )
