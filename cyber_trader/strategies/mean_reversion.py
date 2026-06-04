"""Mean-reversion strategy: Bollinger Bands + RSI, optimised for ranging markets."""

from __future__ import annotations

from loguru import logger
from nautilus_trader.model.data import Bar
from nautilus_trader.model.objects import Currency

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

    # Factor weights
    bb_weight: float = 0.6
    rsi_weight: float = 0.4

    # Require strong signal (only at clear BB extremes with RSI confirmation)
    long_threshold: float = 0.40
    short_threshold: float = -0.40

    # ADX gate: block entries when ADX > adx_threshold (market is trending, not ranging)
    # Set adx_period=0 to disable.
    adx_period: int = 14
    adx_threshold: float = 25.0

    # Exit when score drops below this fraction of the entry threshold.
    # 0.0 = hold until fully reverted to mid-band; 1.0 = same as entry threshold.
    # Mean reversion benefits from holding through partial reversion, so default=0.1.
    exit_threshold_ratio: float = 0.1


class MeanReversionStrategy(BaseStrategy):
    """
    Mean-reversion strategy for ranging / choppy markets.

    Entry:
      LONG  when price is near/below lower BB AND RSI oversold  (composite >= threshold)
      SHORT when price is near/above upper BB AND RSI overbought (composite <= -threshold)

    Exit (whichever comes first):
      1. Hard take-profit: price hits take_profit_pct from entry
      2. Hard stop-loss:   price hits stop_loss_pct from entry (cap the "knife-catch" loss)
      3. Signal exit:      composite score drops below exit_threshold_ratio * threshold
                           (price has reverted most of the way to mid-band)

    ADX gate:
      Entries blocked when ADX > adx_threshold (trending market).
      Switch to TrendFollowingStrategy when ADX persistently exceeds threshold.
    """

    def __init__(self, config: MeanReversionConfig) -> None:
        super().__init__(config)
        self._mr_config = config
        self._entry_price: float = 0.0
        self._entry_is_long: bool = True

    def build_factor_engine(self) -> FactorEngine:
        cfg = self._mr_config
        return FactorEngine(
            factors=[
                BollingerFactor(period=cfg.bb_period, k=cfg.bb_k, weight=cfg.bb_weight),
                RSIFactor(period=cfg.rsi_period, weight=cfg.rsi_weight),
            ],
            long_threshold=cfg.long_threshold,
            short_threshold=cfg.short_threshold,
            adx_period=cfg.adx_period,
            adx_threshold=cfg.adx_threshold,
            adx_below=True,  # ranging mode: only trade when ADX ≤ threshold
        )

    # ── Override signal processing to add hard SL/TP ─────────────────────────

    def _enter_long(self, bar: Bar, score: float, equity: float) -> None:
        self._entry_price = bar.close.as_double()
        self._entry_is_long = True
        super()._enter_long(bar, score, equity)

    def _enter_short(self, bar: Bar, score: float, equity: float) -> None:
        self._entry_price = bar.close.as_double()
        self._entry_is_long = False
        super()._enter_short(bar, score, equity)

    def _process_signal(self, bar: Bar) -> None:
        if self._factor_engine is None:
            return

        score = self._factor_engine.composite_score()
        net_qty = self._net_position()
        price = bar.close.as_double()

        account = self.cache.account_for_venue(self._instrument_id.venue)
        currency = Currency.from_str(self.cfg.currency)
        equity = float(account.balance_total(currency).as_double()) if account else 1_000_000.0

        # ── Hard SL/TP check on open position ────────────────────────────────
        if net_qty != 0 and self._entry_price > 0:
            sl_pct = self.cfg.stop_loss_pct
            tp_pct = self.cfg.take_profit_pct
            if self._entry_is_long:
                sl_price = self._entry_price * (1 - sl_pct)
                tp_price = self._entry_price * (1 + tp_pct)
                if price <= sl_price:
                    logger.info(f"[{self.id}] STOP-LOSS hit @ {price:.2f} (entry={self._entry_price:.2f})")
                    self._exit_position(bar, score)
                    return
                if price >= tp_price:
                    logger.info(f"[{self.id}] TAKE-PROFIT hit @ {price:.2f} (entry={self._entry_price:.2f})")
                    self._exit_position(bar, score)
                    return
            else:
                sl_price = self._entry_price * (1 + sl_pct)
                tp_price = self._entry_price * (1 - tp_pct)
                if price >= sl_price:
                    logger.info(f"[{self.id}] STOP-LOSS hit @ {price:.2f} (entry={self._entry_price:.2f})")
                    self._exit_position(bar, score)
                    return
                if price <= tp_price:
                    logger.info(f"[{self.id}] TAKE-PROFIT hit @ {price:.2f} (entry={self._entry_price:.2f})")
                    self._exit_position(bar, score)
                    return

        # ── Signal-based entry ────────────────────────────────────────────────
        allowed, reason = self._risk.can_trade(equity)

        if self._factor_engine.is_long() and net_qty <= 0:
            if not allowed:
                logger.debug(f"[{self.id}] Long signal blocked: {reason}")
                return
            self._enter_long(bar, score, equity)

        elif self._factor_engine.is_short() and net_qty >= 0:
            if not allowed:
                logger.debug(f"[{self.id}] Short signal blocked: {reason}")
                return
            self._enter_short(bar, score, equity)

        # ── Signal-based exit (price reverted to near mid-band) ───────────────
        elif net_qty != 0:
            cfg = self._mr_config
            exit_long_thr = cfg.long_threshold * cfg.exit_threshold_ratio
            exit_short_thr = cfg.short_threshold * cfg.exit_threshold_ratio
            if (net_qty > 0 and score < exit_long_thr) or (net_qty < 0 and score > exit_short_thr):
                self._exit_position(bar, score)
