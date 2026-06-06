"""
15-minute trend-scalp strategy with three key upgrades over a single-timeframe baseline:

  1. Multi-Timeframe (MTF) Direction Gate
     Subscribe to a higher-timeframe bar (default 1H). Only take LONG signals when
     price is above the 1H EMA(50); only take SHORT signals when below. This removes
     counter-trend entries caused by 15m noise that is invisible on higher charts.
     Look-ahead bias is avoided because the 1H EMA is only updated when a 1H bar
     *closes*, and 15m signals use the last confirmed 1H EMA value.

  2. ATR-Based Dynamic SL/TP
     Stop-loss = entry ± ATR × atr_sl_mult. Take-profit = SL distance × atr_rr_ratio.
     Adapts to current volatility: tight in quiet markets, wider when price is moving.

  3. Volume Confirmation Gate
     Entry is only allowed when the current 15m bar's volume exceeds the rolling mean
     of the previous `vol_confirm_bars` bars times `vol_confirm_factor`. Filters
     "false breakouts" on low-liquidity candles.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from loguru import logger
from nautilus_trader.indicators.averages import ExponentialMovingAverage
from nautilus_trader.indicators.volatility import AverageTrueRange
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.objects import Currency

@dataclass
class _TradeRecord:
    direction: str           # LONG / SHORT
    entry_time: str
    entry_price: float
    sl_price: float
    tp_price: float
    composite_score: float = 0.0
    factor_scores: dict = field(default_factory=dict)   # {factor_name: score}
    htf_ema: float = 0.0
    htf_close: float = 0.0
    atr: float = 0.0
    exit_time: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""    # SL / TP / SIGNAL
    pnl_pct: float = 0.0


from cyber_trader.indicators.factor_engine import (
    EMACrossoverFactor,
    FactorEngine,
    KeltnerBreakoutFactor,
    StochasticFactor,
)
from .base import BaseStrategy, BaseStrategyConfig


class Scalp15mConfig(BaseStrategyConfig, frozen=True):
    # ── 15m signal factors ──────────────────────────────────────────────────
    kc_period: int = 20
    kc_k: float = 2.0
    ema_fast: int = 8
    ema_slow: int = 21
    stoch_period_k: int = 14
    stoch_period_d: int = 3
    stoch_slowing: int = 1

    kc_weight: float = 0.5
    ema_weight: float = 0.3
    stoch_weight: float = 0.2

    long_threshold: float = 0.35
    short_threshold: float = -0.35

    # Exit when an opposite-direction signal fires (score crosses this)
    signal_exit_long: float = -0.35
    signal_exit_short: float = 0.35

    # ── 15m regime gate (secondary, fallback when MTF disabled) ─────────────
    # Set trend_ema_period=0 when higher_tf_bar_type is provided.
    trend_ema_period: int = 100

    # ADX gate: only enter when market IS trending (ADX > threshold)
    adx_period: int = 14
    adx_threshold: float = 25.0

    # ── Multi-Timeframe direction gate ──────────────────────────────────────
    # Full bar type string for the higher timeframe, e.g.:
    #   "ETH-USDT-SWAP.OKX-1-HOUR-LAST-EXTERNAL"
    # Leave empty to disable MTF (falls back to trend_ema_period above).
    higher_tf_bar_type: str = ""
    higher_tf_ema_period: int = 50   # 50 × 1H = 50 hours ≈ 2 days

    # ── Volume confirmation ──────────────────────────────────────────────────
    # Only enter when current volume > mean(last N bars) × factor.
    # vol_confirm_factor_long overrides for LONG entries (reversals start quietly).
    # Set either factor to 0.0 to disable that direction's check.
    vol_confirm_bars: int = 5
    vol_confirm_factor: float = 1.2        # for SHORT entries
    vol_confirm_factor_long: float = -1.0  # -1 = use vol_confirm_factor for both

    # ── ATR-based SL/TP ─────────────────────────────────────────────────────
    atr_period: int = 14
    atr_sl_mult: float = 1.5   # SL = entry ± ATR × atr_sl_mult
    atr_rr_ratio: float = 3.0  # TP = SL distance × atr_rr_ratio

    # Fixed-% fallback when ATR not yet initialized
    stop_loss_pct: float = 0.015
    take_profit_pct: float = 0.060

    # ── Position sizing ──────────────────────────────────────────────────────
    max_position_pct: float = 0.25

    # ── Hold time guard ──────────────────────────────────────────────────────
    # Minimum bars to hold a position before allowing a SIGNAL-based exit.
    # SL/TP exits are never blocked. Prevents paying double commission on
    # entries that reverse immediately on noise.
    min_hold_bars: int = 0

    # ── KC position filter for shorts ────────────────────────────────────────
    # Only allow SHORT entry when KC score >= this value.
    # KC=-1 = price at lower Keltner band (bounce zone, wrong place to short).
    # -0.30 requires price near or above the midline before shorting.
    # Default -1.0 = disabled.
    kc_short_min: float = -1.0


class Scalp15mStrategy(BaseStrategy):
    """
    15-minute trend-scalp strategy with MTF filtering, ATR-based SL/TP,
    and volume confirmation.
    """

    def __init__(self, config: Scalp15mConfig) -> None:
        super().__init__(config)
        self._s15_cfg = config

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

        # ATR indicator (for dynamic SL/TP)
        self._atr = AverageTrueRange(config.atr_period)

        # Volume confirmation buffer
        self._vol_buf: deque[float] = deque(maxlen=config.vol_confirm_bars)

        # Per-trade tracking
        self._entry_price: float = 0.0
        self._entry_is_long: bool = True
        self._sl_price: float = 0.0
        self._tp_price: float = 0.0

        # Trade log
        self._trades: list[_TradeRecord] = []
        self._open_trade: Optional[_TradeRecord] = None

        # Hold time guard
        self._bars_in_trade: int = 0

        # Diagnostic counters
        self._diag_bars: int = 0
        self._diag_score_long: int = 0
        self._diag_score_short: int = 0
        self._diag_adx_blocked: int = 0
        self._diag_htf_blocked: int = 0
        self._diag_vol_blocked: int = 0
        self._diag_kc_blocked: int = 0
        self._diag_entered: int = 0

    # ── Life cycle ────────────────────────────────────────────────────────────

    def on_start(self) -> None:
        super().on_start()
        if self._htf_bar_type is not None:
            self.subscribe_bars(self._htf_bar_type)
            logger.info(
                f"[{self.id}] MTF filter enabled — subscribed to {self._htf_bar_type} "
                f"(EMA {self._s15_cfg.higher_tf_ema_period})"
            )

    def on_stop(self) -> None:
        if self._htf_bar_type is not None:
            self.unsubscribe_bars(self._htf_bar_type)
        logger.info(
            f"[{self.id}] ── Signal funnel ──────────────────────────\n"
            f"  Bars processed      : {self._diag_bars}\n"
            f"  Score ≥ threshold   : long={self._diag_score_long}  short={self._diag_score_short}  "
            f"total={self._diag_score_long + self._diag_score_short}\n"
            f"  Blocked by ADX      : {self._diag_adx_blocked}\n"
            f"  Blocked by HTF EMA  : {self._diag_htf_blocked}\n"
            f"  Blocked by Volume   : {self._diag_vol_blocked}\n"
            f"  Blocked by KC pos   : {self._diag_kc_blocked}\n"
            f"  Actually entered    : {self._diag_entered}\n"
            f"  Pass rate           : "
            f"{self._diag_entered / max(1, self._diag_score_long + self._diag_score_short):.1%}"
        )
        self._print_trade_log()
        super().on_stop()

    def _print_trade_log(self) -> None:
        if not self._trades:
            logger.info(f"[{self.id}] No completed trades to display.")
            return

        # Collect factor names
        factor_names: list[str] = []
        orig_names:   list[str] = []
        for t in self._trades:
            if t.factor_scores:
                orig_names   = list(t.factor_scores.keys())
                factor_names = [n.split("(")[0] for n in orig_names]
                break

        W = 62  # width of the top line (trade summary)
        sep  = "─" * W
        sep2 = "·" * W

        lines: list[str] = [sep]
        total_pnl = 0.0
        wins = 0

        for i, t in enumerate(self._trades, 1):
            pnl_sign   = "+" if t.pnl_pct >= 0 else ""
            result_sym = {"TP": "✓TP", "SL": "✗SL", "SIGNAL": "~SIG"}.get(t.exit_reason, t.exit_reason)
            htf_diff   = t.htf_close - t.htf_ema
            htf_str    = f"HTF {htf_diff:+.0f}" if t.htf_ema > 0 else ""

            # Line 1: trade basics
            lines.append(
                f"#{i:<2} {t.direction}  "
                f"{t.entry_time} → {t.exit_time}  "
                f"{t.entry_price:.2f} → {t.exit_price:.2f}  "
                f"{result_sym}  {pnl_sign}{t.pnl_pct:.2f}%"
            )
            # Line 2: SL/TP + factor scores
            f_parts = "  ".join(
                f"{fn}:{t.factor_scores[on]:+.2f}"
                for fn, on in zip(factor_names, orig_names)
            )
            lines.append(
                f"    SL={t.sl_price:.2f}  TP={t.tp_price:.2f}  "
                f"Score={t.composite_score:+.3f}  [{f_parts}]  {htf_str}"
            )
            lines.append(sep2)

            total_pnl += t.pnl_pct
            if t.pnl_pct >= 0:
                wins += 1

        lines[-1] = sep  # replace last dotted line with solid
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

    # ── Bar routing ───────────────────────────────────────────────────────────

    def on_bar(self, bar: Bar) -> None:
        if self._htf_bar_type is not None and bar.bar_type == self._htf_bar_type:
            self._handle_htf_bar(bar)
            return
        # Primary 15m bar: update ATR and vol buffer before factor engine
        self._atr.update_raw(bar.high.as_double(), bar.low.as_double(), bar.close.as_double())
        self._vol_buf.append(bar.volume.as_double())
        if self._net_position() != 0:
            self._bars_in_trade += 1
        super().on_bar(bar)

    def _handle_htf_bar(self, bar: Bar) -> None:
        """Update higher-TF EMA when the bar CLOSES (no look-ahead)."""
        close = bar.close.as_double()
        if self._htf_ema is not None:
            self._htf_ema.update_raw(close)
            if self._htf_ema.initialized:
                self._htf_ema_val = self._htf_ema.value
        self._htf_last_close = close

    # ── Factor engine ─────────────────────────────────────────────────────────

    def build_factor_engine(self) -> FactorEngine:
        cfg = self._s15_cfg
        return FactorEngine(
            factors=[
                KeltnerBreakoutFactor(period=cfg.kc_period, k=cfg.kc_k, weight=cfg.kc_weight),
                EMACrossoverFactor(fast=cfg.ema_fast, slow=cfg.ema_slow, weight=cfg.ema_weight),
                StochasticFactor(
                    period_k=cfg.stoch_period_k,
                    period_d=cfg.stoch_period_d,
                    slowing=cfg.stoch_slowing,
                    weight=cfg.stoch_weight,
                ),
            ],
            long_threshold=cfg.long_threshold,
            short_threshold=cfg.short_threshold,
            # When MTF is active, disable the built-in 15m trend EMA (set period=0)
            # to avoid double-counting the trend filter.
            trend_ema_period=0 if cfg.higher_tf_bar_type else cfg.trend_ema_period,
            adx_period=cfg.adx_period,
            adx_threshold=cfg.adx_threshold,
        )

    # ── MTF direction gates ───────────────────────────────────────────────────

    def _htf_allows_long(self) -> bool:
        if self._htf_ema is None:
            return True  # MTF disabled
        if not self._htf_ema.initialized:
            return False  # block until HTF EMA warms up
        return self._htf_last_close >= self._htf_ema_val

    def _htf_allows_short(self) -> bool:
        if self._htf_ema is None:
            return True
        if not self._htf_ema.initialized:
            return False
        return self._htf_last_close <= self._htf_ema_val

    # ── Volume confirmation ───────────────────────────────────────────────────

    def _volume_confirms(self, bar: Bar, is_long: bool = False) -> bool:
        cfg = self._s15_cfg
        factor = (
            cfg.vol_confirm_factor_long
            if (is_long and cfg.vol_confirm_factor_long >= 0)
            else cfg.vol_confirm_factor
        )
        if factor <= 0 or len(self._vol_buf) < cfg.vol_confirm_bars:
            return True
        avg_vol = sum(self._vol_buf) / len(self._vol_buf)
        return bar.volume.as_double() >= avg_vol * factor

    # ── ATR-based SL/TP ───────────────────────────────────────────────────────

    def _compute_sl_tp(self, entry: float, is_long: bool) -> tuple[float, float]:
        cfg = self._s15_cfg
        if self._atr.initialized and self._atr.value > 0:
            # Cap ATR at 3% of current price to guard against corrupted data points
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

    def _make_trade_record(self, direction: str, bar: Bar, score: float) -> _TradeRecord:
        factor_scores = self._factor_engine.factor_scores() if self._factor_engine else {}
        return _TradeRecord(
            direction=direction,
            entry_time=self._fmt_ts(bar.ts_event),
            entry_price=self._entry_price,
            sl_price=self._sl_price,
            tp_price=self._tp_price,
            composite_score=score,
            factor_scores=factor_scores,
            htf_ema=self._htf_ema_val,
            htf_close=self._htf_last_close,
            atr=self._atr.value if self._atr.initialized else 0.0,
        )

    def _enter_long(self, bar: Bar, score: float, equity: float) -> None:
        self._entry_price = bar.close.as_double()
        self._entry_is_long = True
        self._bars_in_trade = 0
        self._sl_price, self._tp_price = self._compute_sl_tp(self._entry_price, True)
        self._open_trade = self._make_trade_record("LONG", bar, score)
        logger.debug(
            f"[{self.id}] LONG SL={self._sl_price:.4f} TP={self._tp_price:.4f} "
            f"ATR={self._atr.value:.4f}"
        )
        super()._enter_long(bar, score, equity)

    def _enter_short(self, bar: Bar, score: float, equity: float) -> None:
        self._entry_price = bar.close.as_double()
        self._entry_is_long = False
        self._bars_in_trade = 0
        self._sl_price, self._tp_price = self._compute_sl_tp(self._entry_price, False)
        self._open_trade = self._make_trade_record("SHORT", bar, score)
        logger.debug(
            f"[{self.id}] SHORT SL={self._sl_price:.4f} TP={self._tp_price:.4f} "
            f"ATR={self._atr.value:.4f}"
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

        # ── ATR SL/TP on open position ────────────────────────────────────────
        if net_qty != 0 and self._entry_price > 0:
            if self._entry_is_long:
                if price <= self._sl_price:
                    logger.info(f"[{self.id}] SL @ {price:.4f} (entry={self._entry_price:.4f})")
                    self._bars_in_trade = 0
                    self._record_exit(bar, price, "SL")
                    self._exit_position(bar, score)
                    return
                if price >= self._tp_price:
                    logger.info(f"[{self.id}] TP @ {price:.4f} (entry={self._entry_price:.4f})")
                    self._bars_in_trade = 0
                    self._record_exit(bar, price, "TP")
                    self._exit_position(bar, score)
                    return
            else:
                if price >= self._sl_price:
                    logger.info(f"[{self.id}] SL @ {price:.4f} (entry={self._entry_price:.4f})")
                    self._bars_in_trade = 0
                    self._record_exit(bar, price, "SL")
                    self._exit_position(bar, score)
                    return
                if price <= self._tp_price:
                    logger.info(f"[{self.id}] TP @ {price:.4f} (entry={self._entry_price:.4f})")
                    self._bars_in_trade = 0
                    self._record_exit(bar, price, "TP")
                    self._exit_position(bar, score)
                    return

        # ── Signal-based exit ─────────────────────────────────────────────────
        if net_qty != 0:
            cfg = self._s15_cfg
            signal_exit_triggered = (net_qty > 0 and score < cfg.signal_exit_long) or (
                net_qty < 0 and score > cfg.signal_exit_short
            )
            if signal_exit_triggered:
                if self._bars_in_trade < cfg.min_hold_bars:
                    logger.debug(
                        f"[{self.id}] Signal exit suppressed: only {self._bars_in_trade} bars "
                        f"held (min={cfg.min_hold_bars})"
                    )
                else:
                    self._bars_in_trade = 0
                    self._record_exit(bar, price, "SIGNAL")
                    self._exit_position(bar, score)
                    net_qty = self._net_position()  # refresh; fall through to check reverse entry

        # ── Diagnostic: count raw score crossings before any gate ─────────────
        cfg = self._s15_cfg
        raw_long  = score >= cfg.long_threshold
        raw_short = score <= cfg.short_threshold
        # ADX-blocked = raw signal exists but FactorEngine rejects due to ADX
        adx_ok = self._factor_engine._adx_allows_entry()  # noqa: SLF001
        if raw_long and not adx_ok:
            self._diag_adx_blocked += 1
        if raw_short and not adx_ok:
            self._diag_adx_blocked += 1

        # ── Entry gates ───────────────────────────────────────────────────────
        allowed, reason = self._risk.can_trade(equity)

        if self._factor_engine.is_long() and net_qty <= 0:
            self._diag_score_long += 1
            if not self._htf_allows_long():
                logger.debug(f"[{self.id}] Long blocked: HTF bearish")
                self._diag_htf_blocked += 1
                return
            if not self._volume_confirms(bar, is_long=True):
                logger.debug(f"[{self.id}] Long blocked: low volume")
                self._diag_vol_blocked += 1
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
            if not self._volume_confirms(bar):
                logger.debug(f"[{self.id}] Short blocked: low volume")
                self._diag_vol_blocked += 1
                return
            if cfg.kc_short_min > -1.0:
                kc_score = self._factor_engine.factor_scores().get(
                    f"KC_BREAK({cfg.kc_period},{cfg.kc_k})", -1.0
                )
                if kc_score < cfg.kc_short_min:
                    logger.debug(
                        f"[{self.id}] Short blocked: KC={kc_score:.3f} < min {cfg.kc_short_min}"
                    )
                    self._diag_kc_blocked += 1
                    return
            if not allowed:
                logger.debug(f"[{self.id}] Short blocked: {reason}")
                return
            self._diag_entered += 1
            self._enter_short(bar, score, equity)
