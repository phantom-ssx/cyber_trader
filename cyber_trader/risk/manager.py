"""Risk management: position sizing, drawdown limits, stop-loss / take-profit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nautilus_trader.model.objects import Money, Price, Quantity


@dataclass
class RiskConfig:
    # Position sizing
    risk_per_trade: float = 0.01          # fraction of equity risked per trade (1%)
    max_position_pct: float = 0.10        # max single position as % of equity (10%)
    leverage: float = 1.0                 # leverage multiplier

    # Stop / take profit (as % of entry price)
    stop_loss_pct: float = 0.02           # 2% stop loss
    take_profit_pct: float = 0.04         # 4% take profit (2:1 RR)
    trailing_stop_pct: float = 0.0        # 0 = disabled

    # Daily / drawdown limits
    max_daily_loss_pct: float = 0.05      # 5% daily loss limit
    max_drawdown_pct: float = 0.15        # 15% max drawdown → circuit breaker

    # When the drawdown breaker trips, pause for this many bars then reset the
    # equity peak and resume (instead of halting the strategy forever).
    drawdown_cooldown_bars: int = 30

    # Cooldown after loss
    loss_cooldown_bars: int = 0           # bars to skip after a losing trade (0=disabled)


class RiskManager:
    """Stateful risk checks used by strategies."""

    def __init__(self, config: RiskConfig) -> None:
        self.cfg = config
        self._peak_equity: float = 0.0
        self._daily_start_equity: float = 0.0
        self._losses_today: float = 0.0
        self._cooldown_remaining: int = 0
        self._dd_cooldown_remaining: int = 0
        self._current_day: int | None = None
        self._wins: int = 0
        self._losses: int = 0
        self._total_pnl: float = 0.0

    def init_equity(self, equity: float) -> None:
        if self._peak_equity == 0.0:
            self._peak_equity = equity
            self._daily_start_equity = equity

    def update_equity(self, equity: float, ts_event: int | None = None) -> None:
        self._peak_equity = max(self._peak_equity, equity)
        # Roll the daily-loss reference at each UTC day boundary so the daily
        # loss gate is genuinely *daily* (otherwise it measures loss since
        # inception and permanently freezes the strategy after one bad stretch).
        if ts_event is not None:
            day = ts_event // 86_400_000_000_000  # ns → day index
            if self._current_day is None:
                self._current_day = day
            elif day != self._current_day:
                self._current_day = day
                self._daily_start_equity = equity

    # ── Position sizing ───────────────────────────────────────────────────────

    def position_size(
        self,
        equity: float,
        entry_price: float,
        instrument_lot: float = 1.0,
    ) -> float:
        """Returns number of contracts/units to trade."""
        if entry_price <= 0 or equity <= 0:
            return 0.0
        risk_amount = equity * self.cfg.risk_per_trade
        stop_distance = entry_price * self.cfg.stop_loss_pct
        if stop_distance == 0:
            return 0.0
        size = (risk_amount / stop_distance) * self.cfg.leverage
        max_size = (equity * self.cfg.max_position_pct * self.cfg.leverage) / entry_price
        size = min(size, max_size)
        # Round down to lot size
        if instrument_lot > 0:
            size = (size // instrument_lot) * instrument_lot
        return max(size, 0.0)

    def stop_loss_price(self, entry: float, is_long: bool) -> float:
        if is_long:
            return entry * (1 - self.cfg.stop_loss_pct)
        return entry * (1 + self.cfg.stop_loss_pct)

    def take_profit_price(self, entry: float, is_long: bool) -> float:
        if is_long:
            return entry * (1 + self.cfg.take_profit_pct)
        return entry * (1 - self.cfg.take_profit_pct)

    # ── Gate checks ───────────────────────────────────────────────────────────

    def can_trade(self, equity: float) -> tuple[bool, str]:
        """Returns (allowed, reason_if_blocked)."""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return False, f"cooldown ({self._cooldown_remaining} bars left)"

        # Drawdown circuit breaker: pause for a cooldown then reset the peak so
        # the strategy can recover, rather than freezing for the whole run.
        if self._dd_cooldown_remaining > 0:
            self._dd_cooldown_remaining -= 1
            if self._dd_cooldown_remaining == 0:
                self._peak_equity = equity  # reset reference after the pause
            return False, f"drawdown cooldown ({self._dd_cooldown_remaining} bars left)"

        drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        if drawdown >= self.cfg.max_drawdown_pct:
            self._dd_cooldown_remaining = self.cfg.drawdown_cooldown_bars
            if self._dd_cooldown_remaining == 0:
                self._peak_equity = equity
            return False, f"max drawdown reached ({drawdown:.1%}) → pausing"

        daily_loss = (
            (self._daily_start_equity - equity) / self._daily_start_equity
            if self._daily_start_equity > 0 else 0.0
        )
        if daily_loss >= self.cfg.max_daily_loss_pct:
            return False, f"daily loss limit reached ({daily_loss:.1%})"

        return True, ""

    # ── Trade outcome recording ───────────────────────────────────────────────

    def record_trade(self, pnl: float) -> None:
        self._total_pnl += pnl
        if pnl >= 0:
            self._wins += 1
        else:
            self._losses += 1
            if self.cfg.loss_cooldown_bars > 0:
                self._cooldown_remaining = self.cfg.loss_cooldown_bars

    def reset_daily(self, equity: float) -> None:
        self._daily_start_equity = equity
        self._losses_today = 0.0

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def win_rate(self) -> float:
        total = self._wins + self._losses
        return self._wins / total if total > 0 else 0.0

    @property
    def total_trades(self) -> int:
        return self._wins + self._losses

    @property
    def total_pnl(self) -> float:
        return self._total_pnl

    def drawdown(self, equity: float) -> float:
        if self._peak_equity == 0:
            return 0.0
        return (self._peak_equity - equity) / self._peak_equity
