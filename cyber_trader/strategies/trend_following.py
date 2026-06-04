"""Trend-following strategy: EMA crossover + MACD confirmation."""

from __future__ import annotations

from nautilus_trader.config import StrategyConfig

from cyber_trader.indicators.factor_engine import (
    EMACrossoverFactor,
    FactorEngine,
    FundingRateFactor,
    LongShortRatioFactor,
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

    # Trend-regime filter EMA (only trade with the dominant trend)
    trend_ema_period: int = 200

    # ADX regime gate: block all entries when ADX ≤ adx_threshold (choppy market)
    # Set adx_period=0 to disable. Typical values: period=14, threshold=20.
    adx_period: int = 14
    adx_threshold: float = 20.0

    # Deploy more capital per signal than the ultra-conservative 10% default,
    # while staying well within margin so the drawdown breaker isn't tripped.
    max_position_pct: float = 0.30

    # Auxiliary sentiment factors (set weight > 0 to enable; requires aux data download)
    funding_rate_weight: float = 0.0       # e.g. 0.2 to use 20% weight
    funding_rate_threshold: float = 0.0003 # 0.03%/8h = max signal strength
    long_short_ratio_weight: float = 0.0   # e.g. 0.2 to use 20% weight
    long_short_ratio_window: int = 48      # rolling z-score window (hours)


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend-following strategy combining EMA crossover and MACD.

    Entry:
      LONG  when composite_score >= long_threshold
      SHORT when composite_score <= short_threshold

    Exit:
      Neutral zone (composite score between thresholds)

    Optional auxiliary factors (set weight > 0 in config and download aux data first):
      - FundingRateFactor:    contrarian signal from 8h perpetual funding rate
      - LongShortRatioFactor: contrarian signal from account-level L/S ratio
    """

    def __init__(self, config: TrendFollowingConfig) -> None:
        super().__init__(config)
        self._tf_config = config

    def build_factor_engine(self) -> FactorEngine:
        cfg = self._tf_config
        factors = [
            EMACrossoverFactor(fast=cfg.ema_fast, slow=cfg.ema_slow, weight=cfg.ema_weight),
            MACDFactor(fast=cfg.macd_fast, slow=cfg.macd_slow, weight=cfg.macd_weight),
        ]

        # Derive base symbol (e.g. "ETH-USDT-SWAP" → "ETH-USDT")
        raw_symbol = str(self._instrument_id.symbol)
        base_symbol = "-".join(raw_symbol.replace("-SWAP", "").split("-")[:2])

        if cfg.funding_rate_weight > 0:
            from cyber_trader.data.aux_downloader import load_series
            data = load_series(base_symbol, "funding_rate")
            factors.append(
                FundingRateFactor(
                    data=data,
                    threshold=cfg.funding_rate_threshold,
                    weight=cfg.funding_rate_weight,
                )
            )

        if cfg.long_short_ratio_weight > 0:
            from cyber_trader.data.aux_downloader import load_series
            data = load_series(base_symbol, "long_short_ratio")
            factors.append(
                LongShortRatioFactor(
                    data=data,
                    zscore_window=cfg.long_short_ratio_window,
                    weight=cfg.long_short_ratio_weight,
                )
            )

        return FactorEngine(
            factors=factors,
            long_threshold=cfg.long_threshold,
            short_threshold=cfg.short_threshold,
            trend_ema_period=cfg.trend_ema_period,
            adx_period=cfg.adx_period,
            adx_threshold=cfg.adx_threshold,
        )
