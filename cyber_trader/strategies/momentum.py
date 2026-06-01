"""Momentum strategy: price ROC + volume momentum."""

from __future__ import annotations

from cyber_trader.indicators.factor_engine import (
    FactorEngine,
    MomentumFactor,
    VolumeMomentumFactor,
)
from .base import BaseStrategy, BaseStrategyConfig


class MomentumConfig(BaseStrategyConfig, frozen=True):
    # Price momentum period
    mom_period: int = 10

    # Volume momentum period
    vol_period: int = 20

    # Factor weights
    mom_weight: float = 0.7
    vol_weight: float = 0.3

    # Momentum needs a clear signal to avoid chasing noise
    long_threshold: float = 0.30
    short_threshold: float = -0.30


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy using price rate-of-change and volume confirmation.

    Entry:
      LONG  when recent N-bar return is strongly positive AND volume confirms
      SHORT when recent N-bar return is strongly negative AND volume confirms

    Exit:
      Momentum fades to neutral zone
    """

    def __init__(self, config: MomentumConfig) -> None:
        super().__init__(config)
        self._mom_config = config

    def build_factor_engine(self) -> FactorEngine:
        cfg = self._mom_config
        return FactorEngine(
            factors=[
                MomentumFactor(
                    period=cfg.mom_period,
                    weight=cfg.mom_weight,
                ),
                VolumeMomentumFactor(
                    period=cfg.vol_period,
                    weight=cfg.vol_weight,
                ),
            ],
            long_threshold=cfg.long_threshold,
            short_threshold=cfg.short_threshold,
        )
