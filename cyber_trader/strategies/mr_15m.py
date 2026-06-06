"""
15-minute Mean-Reversion strategy.

Design rationale
────────────────
Mean reversion works best in ranging/choppy markets. Three factors combined:

  1. BollingerFactor  (weight 0.45)
     Price below lower band → oversold → long (+1).
     Price above upper band → overbought → short (−1).

  2. RSIFactor  (weight 0.35)
     Period 14. Oversold → long, overbought → short.
     NOTE: period 9 was tried but saturates in sustained trends (stuck at +0.99
     for hundreds of bars), removing all discriminatory power. Period 14 is slower
     but retains signal quality when price is in a slow grind.

  3. StochasticMRFactor  (weight 0.20)
     Inverted Stochastic: %K < 20 → oversold long, %K > 80 → overbought short.
     Uses high/low context (unlike RSI which is close-only).

Regime gates (two layers)
─────────────────────────
  1. ADX < adx_threshold (adx_below=True):
     Block entries when ADX is high — trending market, mean reversion unsafe.

  2. Multi-Timeframe (MTF) EMA direction gate:
     Subscribe to a higher-TF bar (default 1H EMA 50).
     LONG entries: only when HTF close ≥ HTF EMA  (macro uptrend or neutral).
     SHORT entries: only when HTF close ≤ HTF EMA  (macro downtrend or neutral).
     This is the most critical guard — without it the strategy catches falling
     knives in sustained bear markets (empirically validated in backtest).

Exit logic
──────────
  1. ATR-based hard stop-loss / take-profit (adapts to volatility).
  2. Score-based mid-band exit: composite score crosses back through
     exit_threshold_ratio × entry_threshold (price reverted to mean).
  3. Max hold-time guard: force exit after max_hold_bars (default 32 = 8 hours).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from loguru import logger
from nautilus_trader.indicators.averages import ExponentialMovingAverage
from nautilus_trader.indicators.volatility import AverageTrueRange
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.objects import Currency

from cyber_trader.indicators.factor_engine import (
    BollingerFactor,
    FactorEngine,
    RSIFactor,
    StochasticMRFactor,
)
from .base import BaseStrategy, BaseStrategyConfig


@dataclass
class _TradeRecord:
    direction: str
    entry_time: str
    entry_price: float
    sl_price: float
    tp_price: float
    composite_score: float = 0.0
    factor_scores: dict = field(default_factory=dict)
    htf_ema: float = 0.0
    htf_close: float = 0.0
    atr: float = 0.0
    exit_time: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""   # SL / TP / SIGNAL / MAX_HOLD
    pnl_pct: float = 0.0


class MR15mConfig(BaseStrategyConfig, frozen=True):
    # ── Bollinger Bands ─────────────────────────────────────────────────────
    bb_period: int = 20
    bb_k: float = 2.0
    bb_weight: float = 0.45

    # ── RSI (period 14 avoids saturation in slow trends) ────────────────────
    rsi_period: int = 14
    rsi_weight: float = 0.35

    # ── Stochastic mean-reversion (contrarian) ──────────────────────────────
    stoch_period_k: int = 14
    stoch_period_d: int = 3
    stoch_slowing: int = 3
    stoch_weight: float = 0.20

    # ── Entry / exit thresholds ─────────────────────────────────────────────
    # 0.50 requires stronger multi-factor agreement than the original 0.40,
    # reducing trade count and filtering weaker signals.
    long_threshold: float = 0.50
    short_threshold: float = -0.50

    exit_threshold_ratio: float = 0.15

    # ── ADX regime gate (ranging market only) ───────────────────────────────
    adx_period: int = 14
    adx_threshold: float = 25.0

    # ── Multi-Timeframe direction gate ──────────────────────────────────────
    # Full bar type string, e.g. "ETH-USDT-SWAP.OKX-1-HOUR-LAST-EXTERNAL"
    # Leave empty to disable (not recommended — strong bear/bull markets will
    # cause the strategy to catch falling knives or short exhausted rallies).
    higher_tf_bar_type: str = ""
    higher_tf_ema_period: int = 50   # 50 × 1H ≈ 2 days macro direction

    # ── ATR-based SL/TP ─────────────────────────────────────────────────────
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_rr_ratio: float = 2.0      # 2:1 R:R suits mean reversion (smaller moves)

    stop_loss_pct: float = 0.015
    take_profit_pct: float = 0.030

    # ── Risk ────────────────────────────────────────────────────────────────
    max_position_pct: float = 0.25

    # ── Max hold-time guard ──────────────────────────────────────────────────
    max_hold_bars: int = 32        # 32 × 15m = 8 hours


class MR15mStrategy(BaseStrategy):
    """
    15-minute mean-reversion strategy with MTF trend gate.

    Combines BollingerFactor, RSI(14), and StochasticMR with:
    - ADX ranging-market gate
    - Higher-TF EMA direction filter (prevents catching falling knives)
    - ATR-based SL/TP + mid-band score exit + max hold guard
    """

    def __init__(self, config: MR15mConfig) -> None:
        super().__init__(config)
        self._mr_cfg = config
        self._atr = AverageTrueRange(config.atr_period)

        # Multi-timeframe setup
        if config.higher_tf_bar_type:
            self._htf_bar_type: BarType | None = BarType.from_str(config.higher_tf_bar_type)
            self._htf_ema: ExponentialMovingAverage | None = ExponentialMovingAverage(
                config.higher_tf_ema_period
            )
        else:
            self._htf_bar_type = None
            self._htf_ema = None
        self._htf_ema_val: float = 0.0
        self._htf_last_close: float = 0.0

        self._entry_price: float = 0.0
        self._entry_is_long: bool = True
        self._sl_price: float = 0.0
        self._tp_price: float = 0.0
        self._bars_in_trade: int = 0

        self._trades: list[_TradeRecord] = []
        self._open_trade: Optional[_TradeRecord] = None

        # Diagnostics
        self._diag_bars: int = 0
        self._diag_score_long: int = 0
        self._diag_score_short: int = 0
        self._diag_adx_blocked: int = 0
        self._diag_htf_blocked: int = 0
        self._diag_max_hold_exits: int = 0
        self._diag_entered: int = 0

    # ── Life cycle ────────────────────────────────────────────────────────────

    def on_start(self) -> None:
        super().on_start()
        if self._htf_bar_type is not None:
            self.subscribe_bars(self._htf_bar_type)
            logger.info(
                f"[{self.id}] MTF filter enabled — subscribed to {self._htf_bar_type} "
                f"(EMA {self._mr_cfg.higher_tf_ema_period})"
            )

    def on_stop(self) -> None:
        if self._htf_bar_type is not None:
            self.unsubscribe_bars(self._htf_bar_type)
        logger.info(
            f"[{self.id}] ── Signal funnel ──────────────────────────\n"
            f"  Bars processed      : {self._diag_bars}\n"
            f"  Score ≥ threshold   : long={self._diag_score_long}  short={self._diag_score_short}\n"
            f"  Blocked by ADX      : {self._diag_adx_blocked}\n"
            f"  Blocked by HTF EMA  : {self._diag_htf_blocked}\n"
            f"  Max-hold exits      : {self._diag_max_hold_exits}\n"
            f"  Actually entered    : {self._diag_entered}\n"
            f"  Pass rate           : "
            f"{self._diag_entered / max(1, self._diag_score_long + self._diag_score_short):.1%}"
        )
        self._print_trade_log()
        super().on_stop()

    def _print_trade_log(self) -> None:
        if not self._trades:
            logger.info(f"[{self.id}] No completed trades.")
            return

        factor_names: list[str] = []
        orig_names: list[str] = []
        for t in self._trades:
            if t.factor_scores:
                orig_names = list(t.factor_scores.keys())
                factor_names = [n.split("(")[0] for n in orig_names]
                break

        W = 66
        sep = "─" * W
        sep2 = "·" * W
        lines: list[str] = [sep]
        total_pnl = 0.0
        wins = 0

        for i, t in enumerate(self._trades, 1):
            pnl_sign = "+" if t.pnl_pct >= 0 else ""
            result_sym = {"TP": "✓TP", "SL": "✗SL", "SIGNAL": "~MID", "MAX_HOLD": "⏱MAX"}.get(
                t.exit_reason, t.exit_reason
            )
            htf_diff = t.htf_close - t.htf_ema
            htf_str = f"  HTF {htf_diff:+.0f}" if t.htf_ema > 0 else ""
            lines.append(
                f"#{i:<2} {t.direction}  "
                f"{t.entry_time} → {t.exit_time}  "
                f"{t.entry_price:.2f} → {t.exit_price:.2f}  "
                f"{result_sym}  {pnl_sign}{t.pnl_pct:.2f}%"
            )
            f_parts = "  ".join(
                f"{fn}:{t.factor_scores[on]:+.2f}"
                for fn, on in zip(factor_names, orig_names)
            )
            lines.append(
                f"    SL={t.sl_price:.2f}  TP={t.tp_price:.2f}  "
                f"Score={t.composite_score:+.3f}  [{f_parts}]{htf_str}"
            )
            lines.append(sep2)
            total_pnl += t.pnl_pct
            if t.pnl_pct >= 0:
                wins += 1

        lines[-1] = sep
        lines.append(
            f"Trades={len(self._trades)}  "
            f"Wins={wins}  Losses={len(self._trades) - wins}  "
            f"WinRate={wins / len(self._trades):.1%}  "
            f"TotalPnL={total_pnl:+.2f}%"
        )
        logger.info(
            f"[{self.id}] ── Trade Log ──────────────────────────\n"
            + "\n".join(lines)
        )

    # ── Factor engine ─────────────────────────────────────────────────────────

    def build_factor_engine(self) -> FactorEngine:
        cfg = self._mr_cfg
        return FactorEngine(
            factors=[
                BollingerFactor(period=cfg.bb_period, k=cfg.bb_k, weight=cfg.bb_weight),
                RSIFactor(period=cfg.rsi_period, weight=cfg.rsi_weight),
                StochasticMRFactor(
                    period_k=cfg.stoch_period_k,
                    period_d=cfg.stoch_period_d,
                    slowing=cfg.stoch_slowing,
                    weight=cfg.stoch_weight,
                ),
            ],
            long_threshold=cfg.long_threshold,
            short_threshold=cfg.short_threshold,
            adx_period=cfg.adx_period,
            adx_threshold=cfg.adx_threshold,
            adx_below=True,
        )

    # ── Bar routing ───────────────────────────────────────────────────────────

    def on_bar(self, bar: Bar) -> None:
        if self._htf_bar_type is not None and bar.bar_type == self._htf_bar_type:
            self._handle_htf_bar(bar)
            return
        self._atr.update_raw(bar.high.as_double(), bar.low.as_double(), bar.close.as_double())
        if self._net_position() != 0:
            self._bars_in_trade += 1
        super().on_bar(bar)

    def _handle_htf_bar(self, bar: Bar) -> None:
        close = bar.close.as_double()
        if self._htf_ema is not None:
            self._htf_ema.update_raw(close)
            if self._htf_ema.initialized:
                self._htf_ema_val = self._htf_ema.value
        self._htf_last_close = close

    # ── MTF direction gates ───────────────────────────────────────────────────

    def _htf_allows_long(self) -> bool:
        if self._htf_ema is None:
            return True
        if not self._htf_ema.initialized:
            return False
        return self._htf_last_close >= self._htf_ema_val

    def _htf_allows_short(self) -> bool:
        if self._htf_ema is None:
            return True
        if not self._htf_ema.initialized:
            return False
        return self._htf_last_close <= self._htf_ema_val

    # ── ATR-based SL/TP ───────────────────────────────────────────────────────

    def _compute_sl_tp(self, entry: float, is_long: bool) -> tuple[float, float]:
        cfg = self._mr_cfg
        if self._atr.initialized and self._atr.value > 0:
            atr = min(self._atr.value, entry * 0.03)
            sl_dist = atr * cfg.atr_sl_mult
            tp_dist = sl_dist * cfg.atr_rr_ratio
        else:
            sl_dist = entry * cfg.stop_loss_pct
            tp_dist = entry * cfg.take_profit_pct
        if is_long:
            return entry - sl_dist, entry + tp_dist
        else:
            return entry + sl_dist, entry - tp_dist

    # ── Trade recording ───────────────────────────────────────────────────────

    @staticmethod
    def _fmt_ts(ts_ns: int) -> str:
        return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    def _record_exit(self, bar: Bar, price: float, reason: str) -> None:
        if self._open_trade is None:
            return
        t = self._open_trade
        t.exit_time = self._fmt_ts(bar.ts_event)
        t.exit_price = price
        t.exit_reason = reason
        if t.entry_price > 0:
            if t.direction == "LONG":
                t.pnl_pct = (price - t.entry_price) / t.entry_price * 100
            else:
                t.pnl_pct = (t.entry_price - price) / t.entry_price * 100
        self._trades.append(t)
        self._open_trade = None

    def _enter_long(self, bar: Bar, score: float, equity: float) -> None:
        self._entry_price = bar.close.as_double()
        self._entry_is_long = True
        self._bars_in_trade = 0
        self._sl_price, self._tp_price = self._compute_sl_tp(self._entry_price, True)
        self._open_trade = _TradeRecord(
            direction="LONG",
            entry_time=self._fmt_ts(bar.ts_event),
            entry_price=self._entry_price,
            sl_price=self._sl_price,
            tp_price=self._tp_price,
            composite_score=score,
            factor_scores=self._factor_engine.factor_scores() if self._factor_engine else {},
            htf_ema=self._htf_ema_val,
            htf_close=self._htf_last_close,
            atr=self._atr.value if self._atr.initialized else 0.0,
        )
        super()._enter_long(bar, score, equity)

    def _enter_short(self, bar: Bar, score: float, equity: float) -> None:
        self._entry_price = bar.close.as_double()
        self._entry_is_long = False
        self._bars_in_trade = 0
        self._sl_price, self._tp_price = self._compute_sl_tp(self._entry_price, False)
        self._open_trade = _TradeRecord(
            direction="SHORT",
            entry_time=self._fmt_ts(bar.ts_event),
            entry_price=self._entry_price,
            sl_price=self._sl_price,
            tp_price=self._tp_price,
            composite_score=score,
            factor_scores=self._factor_engine.factor_scores() if self._factor_engine else {},
            htf_ema=self._htf_ema_val,
            htf_close=self._htf_last_close,
            atr=self._atr.value if self._atr.initialized else 0.0,
        )
        super()._enter_short(bar, score, equity)

    # ── Signal processing ─────────────────────────────────────────────────────

    def _process_signal(self, bar: Bar) -> None:
        if self._factor_engine is None:
            return

        self._diag_bars += 1
        score = self._factor_engine.composite_score()
        net_qty = self._net_position()
        price = bar.close.as_double()

        account = self.cache.account_for_venue(self._instrument_id.venue)
        currency = Currency.from_str(self.cfg.currency)
        equity = float(account.balance_total(currency).as_double()) if account else 1_000_000.0

        cfg = self._mr_cfg

        # ── Hard SL/TP on open position ────────────────────────────────────
        if net_qty != 0 and self._entry_price > 0:
            if self._entry_is_long:
                if price <= self._sl_price:
                    logger.info(f"[{self.id}] SL @ {price:.4f} (entry={self._entry_price:.4f})")
                    self._record_exit(bar, price, "SL")
                    self._bars_in_trade = 0
                    self._exit_position(bar, score)
                    return
                if price >= self._tp_price:
                    logger.info(f"[{self.id}] TP @ {price:.4f} (entry={self._entry_price:.4f})")
                    self._record_exit(bar, price, "TP")
                    self._bars_in_trade = 0
                    self._exit_position(bar, score)
                    return
            else:
                if price >= self._sl_price:
                    logger.info(f"[{self.id}] SL @ {price:.4f} (entry={self._entry_price:.4f})")
                    self._record_exit(bar, price, "SL")
                    self._bars_in_trade = 0
                    self._exit_position(bar, score)
                    return
                if price <= self._tp_price:
                    logger.info(f"[{self.id}] TP @ {price:.4f} (entry={self._entry_price:.4f})")
                    self._record_exit(bar, price, "TP")
                    self._bars_in_trade = 0
                    self._exit_position(bar, score)
                    return

        # ── Max hold-time guard ────────────────────────────────────────────
        if net_qty != 0 and cfg.max_hold_bars > 0 and self._bars_in_trade >= cfg.max_hold_bars:
            logger.info(
                f"[{self.id}] MAX_HOLD exit after {self._bars_in_trade} bars @ {price:.4f}"
            )
            self._record_exit(bar, price, "MAX_HOLD")
            self._diag_max_hold_exits += 1
            self._bars_in_trade = 0
            self._exit_position(bar, score)
            net_qty = self._net_position()

        # ── Mid-band signal exit (price reverted to mean) ──────────────────
        if net_qty != 0:
            exit_long_thr = cfg.long_threshold * cfg.exit_threshold_ratio
            exit_short_thr = cfg.short_threshold * cfg.exit_threshold_ratio
            if (net_qty > 0 and score < exit_long_thr) or (net_qty < 0 and score > exit_short_thr):
                self._record_exit(bar, price, "SIGNAL")
                self._bars_in_trade = 0
                self._exit_position(bar, score)
                net_qty = self._net_position()

        # ── ADX diagnostic ─────────────────────────────────────────────────
        raw_long = score >= cfg.long_threshold
        raw_short = score <= cfg.short_threshold
        adx_ok = self._factor_engine._adx_allows_entry()  # noqa: SLF001
        if (raw_long or raw_short) and not adx_ok:
            self._diag_adx_blocked += 1

        # ── Entry ──────────────────────────────────────────────────────────
        allowed, reason = self._risk.can_trade(equity)

        if self._factor_engine.is_long() and net_qty <= 0:
            self._diag_score_long += 1
            if not self._htf_allows_long():
                logger.debug(f"[{self.id}] Long blocked: HTF bearish")
                self._diag_htf_blocked += 1
                return
            if not allowed:
                logger.debug(f"[{self.id}] Long blocked: {reason}")
                return
            self._diag_entered += 1
            self._enter_long(bar, score, equity)

        elif self._factor_engine.is_short() and net_qty >= 0:
            self._diag_score_short += 1
            if not self._htf_allows_short():
                logger.debug(f"[{self.id}] Short blocked: HTF bullish")
                self._diag_htf_blocked += 1
                return
            if not allowed:
                logger.debug(f"[{self.id}] Short blocked: {reason}")
                return
            self._diag_entered += 1
            self._enter_short(bar, score, equity)
