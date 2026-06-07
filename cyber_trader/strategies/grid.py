"""Grid trading strategy with three directional modes.

Modes
-----
neutral : BUY orders below + SELL orders above center price.
          Profits from bi-directional oscillation. Rebalances on any drift.
bull    : Only BUY orders below center price initially.
          As buys fill, SELL orders are placed one level above.
          Holds a net long bias; only rebalances when price drops through floor.
bear    : Only SELL orders above center price initially.
          As sells fill, BUY orders are placed one level below.
          Holds a net short/cash bias; only rebalances when price rises through ceiling.
"""

from __future__ import annotations

from typing import Literal, Optional

from loguru import logger
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy

GridMode = Literal["neutral", "bull", "bear"]


class GridStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: str
    currency: str = "USDT"

    # "neutral" | "bull" | "bear"
    mode: str = "neutral"

    # Spacing between grid levels as a fraction of price (e.g. 0.01 = 1%)
    grid_spacing_pct: float = 0.01
    # Number of levels on the active side(s) of center
    num_levels: int = 5
    # USDT notional value per grid level
    order_size_usdt: float = 200.0

    # Fraction of drift from center that triggers a grid rebuild.
    # bull:    only triggers when price falls below  center * (1 - threshold)
    # bear:    only triggers when price rises above   center * (1 + threshold)
    # neutral: triggers in either direction
    rebalance_threshold_pct: float = 0.08

    enable_notifications: bool = False


class GridStrategy(Strategy):
    """
    Grid trading strategy supporting three directional modes.

    neutral — BUY below + SELL above current price; pure oscillation profit.
    bull    — Only BUY orders initially; accumulates longs on dips, sells on
              rallies. Profits from uptrend + oscillation.
    bear    — Only SELL orders initially; accumulates quote on rallies, buys
              on dips. Profits from downtrend + oscillation.

    Fill mechanics (all modes):
      BUY filled at level L  → place SELL at L × (1 + spacing)
      SELL filled at level L → place BUY  at L × (1 − spacing)
    """

    def __init__(self, config: GridStrategyConfig) -> None:
        super().__init__(config)
        self.cfg = config
        self._instrument_id = InstrumentId.from_str(config.instrument_id)
        self._bar_type = BarType.from_str(config.bar_type)
        self._instrument: Optional[Instrument] = None

        self._center_price: float = 0.0
        # level-key → price for orders currently live in the market
        self._active_buy_levels: dict[str, float] = {}
        self._active_sell_levels: dict[str, float] = {}
        # ClientOrderId.value → original price level (for fill lookup)
        self._order_level: dict[str, float] = {}

        self._total_fills: int = 0
        self._grid_profit_usdt: float = 0.0  # cumulative profit from completed cycles

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self._instrument_id)
        if self._instrument is None:
            self.log.error(f"Instrument {self._instrument_id} not found in cache")
            self.stop()
            return
        self.subscribe_bars(self._bar_type)
        logger.info(
            f"[Grid/{self.cfg.mode}] started — {self._instrument_id} "
            f"spacing={self.cfg.grid_spacing_pct:.1%} "
            f"levels={self.cfg.num_levels} "
            f"size={self.cfg.order_size_usdt} USDT/level"
        )

    def on_stop(self) -> None:
        self.cancel_all_orders(self._instrument_id)
        self.close_all_positions(self._instrument_id)
        self.unsubscribe_bars(self._bar_type)
        logger.info(
            f"[Grid/{self.cfg.mode}] stopped — "
            f"fills={self._total_fills} "
            f"grid_profit≈{self._grid_profit_usdt:+.2f} USDT"
        )

    def on_reset(self) -> None:
        self._center_price = 0.0
        self._active_buy_levels.clear()
        self._active_sell_levels.clear()
        self._order_level.clear()
        self._total_fills = 0
        self._grid_profit_usdt = 0.0

    # ── Bar handler ───────────────────────────────────────────────────────────

    def on_bar(self, bar: Bar) -> None:
        price = bar.close.as_double()

        if self._center_price == 0.0:
            self._center_price = price
            self._build_grid(price)
            return

        if self._should_rebalance(price):
            logger.info(
                f"[Grid/{self.cfg.mode}] Rebalancing — "
                f"center={self._center_price:.4f} → {price:.4f}"
            )
            self._center_price = price
            self.cancel_all_orders(self._instrument_id)
            self._active_buy_levels.clear()
            self._active_sell_levels.clear()
            self._order_level.clear()
            self._build_grid(price)

    def _should_rebalance(self, price: float) -> bool:
        thr = self.cfg.rebalance_threshold_pct
        mode = self.cfg.mode
        if mode == "bull":
            # Only rebalance when price falls well below center (grid floor breached)
            return price < self._center_price * (1 - thr)
        if mode == "bear":
            # Only rebalance when price rises well above center (grid ceiling breached)
            return price > self._center_price * (1 + thr)
        # neutral: rebalance in either direction
        return abs(price - self._center_price) / self._center_price > thr

    # ── Order fill handler ────────────────────────────────────────────────────

    def on_order_filled(self, event: OrderFilled) -> None:
        cid = event.client_order_id.value
        level = self._order_level.pop(cid, None)
        if level is None:
            return  # position-close market order, not a grid level

        self._total_fills += 1
        fill_px = event.last_px.as_double()
        fill_qty = event.last_qty.as_double()
        key = _level_key(level)
        spacing = self.cfg.grid_spacing_pct

        if event.order_side == OrderSide.BUY:
            self._active_buy_levels.pop(key, None)
            sell_level = level * (1 + spacing)
            self._place_limit(OrderSide.SELL, sell_level)
            # Each completed BUY→SELL cycle earns ≈ spacing × notional
            self._grid_profit_usdt += fill_qty * fill_px * spacing
            logger.debug(
                f"[Grid/{self.cfg.mode}] BUY filled @ {fill_px:.4f} "
                f"→ SELL queued @ {sell_level:.4f}"
            )

        elif event.order_side == OrderSide.SELL:
            self._active_sell_levels.pop(key, None)
            buy_level = level * (1 - spacing)
            self._place_limit(OrderSide.BUY, buy_level)
            logger.debug(
                f"[Grid/{self.cfg.mode}] SELL filled @ {fill_px:.4f} "
                f"→ BUY queued @ {buy_level:.4f}"
            )

    # ── Grid construction ─────────────────────────────────────────────────────

    def _build_grid(self, center: float) -> None:
        """Place initial limit orders according to the selected mode."""
        spacing = self.cfg.grid_spacing_pct
        mode = self.cfg.mode
        count = 0

        for i in range(1, self.cfg.num_levels + 1):
            if mode in ("neutral", "bull"):
                self._place_limit(OrderSide.BUY, center * (1 - spacing * i))
                count += 1
            if mode in ("neutral", "bear"):
                self._place_limit(OrderSide.SELL, center * (1 + spacing * i))
                count += 1

        logger.info(
            f"[Grid/{mode}] Grid built — center={center:.4f}, {count} orders"
        )

    def _place_limit(self, side: OrderSide, price_level: float) -> None:
        if self._instrument is None or price_level <= 0:
            return

        key = _level_key(price_level)
        if side == OrderSide.BUY and key in self._active_buy_levels:
            return
        if side == OrderSide.SELL and key in self._active_sell_levels:
            return

        tick = self._instrument.price_increment.as_double()
        snapped_px = round(price_level / tick) * tick
        if snapped_px <= 0:
            return

        size_inc = self._instrument.size_increment.as_double()
        qty_raw = self.cfg.order_size_usdt / snapped_px
        qty = max(size_inc, round(qty_raw / size_inc) * size_inc)

        order = self.order_factory.limit(
            instrument_id=self._instrument_id,
            order_side=side,
            quantity=Quantity(qty, self._instrument.size_precision),
            price=Price(snapped_px, self._instrument.price_precision),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

        cid = order.client_order_id.value
        self._order_level[cid] = price_level
        if side == OrderSide.BUY:
            self._active_buy_levels[key] = price_level
        else:
            self._active_sell_levels[key] = price_level


def _level_key(price: float) -> str:
    return f"{price:.8f}"
