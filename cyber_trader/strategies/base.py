"""Base strategy with multi-factor engine, risk management and Feishu notifications."""

from __future__ import annotations

from abc import abstractmethod

from loguru import logger
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Currency, Quantity
from nautilus_trader.trading.strategy import Strategy

from cyber_trader.config import get_settings
from cyber_trader.indicators.factor_engine import FactorEngine
from cyber_trader.notifications.feishu import FeishuNotifier, TradeSignal
from cyber_trader.risk.manager import RiskConfig, RiskManager


class BaseStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: str
    currency: str = "USDT"

    # Risk
    risk_per_trade: float = 0.01
    max_position_pct: float = 0.10
    leverage: float = 1.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_drawdown_pct: float = 0.15
    max_daily_loss_pct: float = 0.05
    drawdown_cooldown_bars: int = 30

    # Factor thresholds
    long_threshold: float = 0.3
    short_threshold: float = -0.3

    # Trend-regime filter: only take longs above / shorts below this EMA.
    # 0 disables the filter.
    trend_ema_period: int = 0

    # Enable Feishu notifications (only in paper/live mode)
    enable_notifications: bool = True


class BaseStrategy(Strategy):
    """
    Abstract base for all CyberTrader strategies.

    Subclasses must implement:
      - build_factor_engine() → FactorEngine
    """

    def __init__(self, config: BaseStrategyConfig) -> None:
        super().__init__(config)
        self.cfg = config
        self._instrument_id = InstrumentId.from_str(config.instrument_id)
        self._bar_type = BarType.from_str(config.bar_type)

        risk_cfg = RiskConfig(
            risk_per_trade=config.risk_per_trade,
            max_position_pct=config.max_position_pct,
            leverage=config.leverage,
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            max_drawdown_pct=config.max_drawdown_pct,
            max_daily_loss_pct=config.max_daily_loss_pct,
            drawdown_cooldown_bars=config.drawdown_cooldown_bars,
        )
        self._risk = RiskManager(risk_cfg)

        settings = get_settings()
        self._notifier: FeishuNotifier | None = (
            FeishuNotifier(settings.feishu_webhook_url, settings.feishu_secret)
            if config.enable_notifications and settings.feishu_webhook_url
            else None
        )

        self._factor_engine: FactorEngine | None = None
        self._instrument: Instrument | None = None
        self._bar_count: int = 0

    # ── Life cycle ────────────────────────────────────────────────────────────

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self._instrument_id)
        if self._instrument is None:
            self.log.error(f"Instrument {self._instrument_id} not found in cache")
            self.stop()
            return

        self._factor_engine = self.build_factor_engine()
        self._factor_engine.long_threshold = self.cfg.long_threshold
        self._factor_engine.short_threshold = self.cfg.short_threshold

        self.subscribe_bars(self._bar_type)
        logger.info(f"[{self.id}] started on {self._instrument_id} bar_type={self._bar_type}")

        if self._notifier:
            self._notifier.send_text(
                f"🚀 策略启动: {self.__class__.__name__}\n"
                f"交易对: {self._instrument_id}\n"
                f"周期: {self._bar_type}"
            )

    def on_stop(self) -> None:
        self.unsubscribe_bars(self._bar_type)
        self.close_all_positions()

    def on_reset(self) -> None:
        self._bar_count = 0

    # ── Bar handler ───────────────────────────────────────────────────────────

    def on_bar(self, bar: Bar) -> None:
        if self._factor_engine is None:
            return

        self._bar_count += 1
        self._factor_engine.update(bar)

        if not self._factor_engine.is_initialized:
            return

        self._update_risk_equity(bar)
        self._process_signal(bar)

    def _update_risk_equity(self, bar: Bar) -> None:
        account = self.cache.account_for_venue(self._instrument_id.venue)
        if account:
            currency = Currency.from_str(self.cfg.currency)
            equity = float(account.balance_total(currency).as_double())
            self._risk.init_equity(equity)
            self._risk.update_equity(equity, bar.ts_event)

    def _process_signal(self, bar: Bar) -> None:
        score = self._factor_engine.composite_score()
        net_qty = self._net_position()

        account = self.cache.account_for_venue(self._instrument_id.venue)
        currency = Currency.from_str(self.cfg.currency)
        equity = float(account.balance_total(currency).as_double()) if account else 1_000_000.0

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

        elif self._factor_engine.is_neutral() and net_qty != 0:
            self._exit_position(bar, score)

    def _net_position(self) -> float:
        positions = self.cache.positions_open(instrument_id=self._instrument_id)
        net = 0.0
        for p in positions:
            sq = p.signed_qty
            net += float(sq) if not isinstance(sq, float) else sq
        return net

    def _enter_long(self, bar: Bar, score: float, equity: float) -> None:
        self.close_all_positions()
        price = bar.close.as_double()
        size = self._risk.position_size(
            equity, price,
            self._instrument.size_increment.as_double() if self._instrument else 0.0001,
        )
        if size <= 0:
            return
        qty = Quantity(size, self._instrument.size_precision if self._instrument else 4)
        order = self.order_factory.market(
            instrument_id=self._instrument_id,
            order_side=OrderSide.BUY,
            quantity=qty,
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
        logger.info(f"[{self.id}] LONG {size} {self._instrument_id} @ ~{price:.4f} score={score:+.3f}")
        self._notify_signal("LONG", price, score)

    def _enter_short(self, bar: Bar, score: float, equity: float) -> None:
        self.close_all_positions()
        price = bar.close.as_double()
        size = self._risk.position_size(
            equity, price,
            self._instrument.size_increment.as_double() if self._instrument else 0.0001,
        )
        if size <= 0:
            return
        qty = Quantity(size, self._instrument.size_precision if self._instrument else 4)
        order = self.order_factory.market(
            instrument_id=self._instrument_id,
            order_side=OrderSide.SELL,
            quantity=qty,
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
        logger.info(f"[{self.id}] SHORT {size} {self._instrument_id} @ ~{price:.4f} score={score:+.3f}")
        self._notify_signal("SHORT", price, score)

    def _exit_position(self, bar: Bar, score: float) -> None:
        self.close_all_positions()
        logger.info(f"[{self.id}] CLOSE {self._instrument_id} score={score:+.3f}")
        self._notify_signal("CLOSE", bar.close.as_double(), score)

    def _notify_signal(self, direction: str, price: float, score: float) -> None:
        if not self._notifier or self._factor_engine is None:
            return
        bar_type_str = str(self._bar_type)
        timeframe = bar_type_str.split("-")[2] if "-" in bar_type_str else bar_type_str
        signal = TradeSignal(
            symbol=str(self._instrument_id.symbol),
            direction=direction,  # type: ignore[arg-type]
            strategy=self.__class__.__name__,
            price=price,
            composite_score=score,
            factors=self._factor_engine.factor_scores(),
            timeframe=timeframe,
        )
        self._notifier.send_signal(signal)

    def close_all_positions(self) -> None:
        for position in self.cache.positions_open(instrument_id=self._instrument_id):
            if position.is_open:
                close_order = self.order_factory.market(
                    instrument_id=self._instrument_id,
                    order_side=OrderSide.SELL if position.is_long else OrderSide.BUY,
                    quantity=position.quantity,
                    time_in_force=TimeInForce.GTC,
                )
                self.submit_order(close_order)

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def build_factor_engine(self) -> FactorEngine:
        """Construct and return the FactorEngine for this strategy."""
        ...
