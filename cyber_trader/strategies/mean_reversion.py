"""Mean-reversion strategy: Bollinger Bands + RSI."""

from __future__ import annotations

from cyber_trader.indicators.factor_engine import (
    BollingerFactor,
    FactorEngine,
    RSIFactor,
)
from .base import BaseStrategy, BaseStrategyConfig


class MeanReversionConfig(BaseStrategyConfig, frozen=True):
    # Bollinger parameters
    bb_period: int = 20
    bb_k: float = 2.0

    # RSI parameters
    rsi_period: int = 14

    # Factor weights (BB slightly higher)
    bb_weight: float = 0.55
    rsi_weight: float = 0.45

    # Mean-reversion uses tighter thresholds (signals are less frequent but higher conviction)
    long_threshold: float = 0.35
    short_threshold: float = -0.35


class MeanReversionStrategy(BaseStrategy):
    """
    Mean-reversion strategy using Bollinger Bands and RSI.

    Entry:
      LONG  when price is below lower BB AND RSI is oversold  (composite >= threshold)
      SHORT when price is above upper BB AND RSI is overbought (composite <= -threshold)

    Exit:
      Price returns to middle band (neutral zone)
    """

    def __init__(self, config: MeanReversionConfig) -> None:
        super().__init__(config)
        self._mr_config = config

    def build_factor_engine(self) -> FactorEngine:
        cfg = self._mr_config
        return FactorEngine(
            factors=[
                BollingerFactor(
                    period=cfg.bb_period,
                    k=cfg.bb_k,
                    weight=cfg.bb_weight,
                ),
                RSIFactor(
                    period=cfg.rsi_period,
                    weight=cfg.rsi_weight,
                ),
            ],
            long_threshold=cfg.long_threshold,
            short_threshold=cfg.short_threshold,
        )
