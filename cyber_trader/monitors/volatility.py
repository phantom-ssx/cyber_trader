"""Volatility anomaly monitor — alerts when price moves too fast."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Any

from loguru import logger
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.trading.strategy import Strategy

from cyber_trader.config import get_settings
from cyber_trader.monitors.rules import AlertRule, build_rule
from cyber_trader.notifications.bark import BarkNotifier

_DEFAULT_RULES: str = json.dumps([
    {"type": "single_bar", "threshold_pct": 0.5},
    {"type": "cumulative", "threshold_pct": 1.0, "bars": 5},
])


class VolatilityMonitorConfig(StrategyConfig, frozen=True):
    """Configuration for the volatility anomaly monitor."""

    # Instruments to watch, e.g. ("ETH-USDT-SWAP.OKX", "BTC-USDT-SWAP.OKX")
    instrument_ids: tuple[str, ...]

    # Bar specification appended after the instrument id
    bar_spec: str = "1-MINUTE-LAST-EXTERNAL"

    # After ANY rule fires, suppress further alerts for this instrument for N bars
    cooldown_bars: int = 10

    enable_notifications: bool = True

    # JSON list of rule spec dicts applied to all instruments by default.
    # Available types: single_bar, cumulative, directional, round_level
    # (see cyber_trader/monitors/rules.py)
    rules_json: str = _DEFAULT_RULES

    # Optional per-instrument rule overrides.  JSON object keyed by instrument_id,
    # value is the same list format as rules_json.  Instruments listed here use
    # their own rule set instead of rules_json.
    # e.g. {"ETH-USDT-SWAP.OKX": [...], "BTC-USDT-SWAP.OKX": [...]}
    per_instrument_rules_json: str = "{}"


class VolatilityMonitor(Strategy):
    """
    Read-only strategy that fires Bark alerts when price velocity
    exceeds any of the configured rules.  Never places orders.
    """

    def __init__(self, config: VolatilityMonitorConfig) -> None:
        super().__init__(config)
        self.cfg = config

        settings = get_settings()
        self._notifier: BarkNotifier | None = (
            BarkNotifier(settings.bark_key, settings.bark_server)
            if config.enable_notifications and settings.bark_key
            else None
        )

        self._default_rules: list[AlertRule] = self.build_rules()
        self._per_instrument_rules: dict[str, list[AlertRule]] = self.build_per_instrument_rules()

        # Size the deque to the largest window needed across all rule sets
        all_rules = self._default_rules + [
            r for rules in self._per_instrument_rules.values() for r in rules
        ]
        max_window = max((r.window_size for r in all_rules), default=2)
        self._closes: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max_window)
        )
        self._cooldown: dict[str, int] = defaultdict(int)

    def build_rules(self) -> list[AlertRule]:
        """
        Build the default rule list applied to all instruments.
        Override in a subclass to add rules programmatically.
        """
        specs: list[dict[str, Any]] = json.loads(self.cfg.rules_json)
        return [build_rule(s) for s in specs]

    def build_per_instrument_rules(self) -> dict[str, list[AlertRule]]:
        """
        Build per-instrument rule overrides from config.
        Override in a subclass for programmatic customisation.
        """
        mapping: dict[str, list[dict[str, Any]]] = json.loads(
            self.cfg.per_instrument_rules_json
        )
        return {inst: [build_rule(s) for s in specs] for inst, specs in mapping.items()}

    # ── Life cycle ────────────────────────────────────────────────────────────

    def on_start(self) -> None:
        for inst_str in self.cfg.instrument_ids:
            bar_type = BarType.from_str(f"{inst_str}-{self.cfg.bar_spec}")
            self.subscribe_bars(bar_type)

        logger.info(
            f"[VolatilityMonitor] watching {len(self.cfg.instrument_ids)} instrument(s): "
            + ", ".join(self.cfg.instrument_ids)
        )
        default_desc = "  |  ".join(r.describe() for r in self._default_rules)
        logger.info(f"[VolatilityMonitor] default rules: {default_desc}")
        for inst, rules in self._per_instrument_rules.items():
            logger.info(
                f"[VolatilityMonitor] {inst} rules: "
                + "  |  ".join(r.describe() for r in rules)
            )
        logger.info(f"[VolatilityMonitor] cooldown={self.cfg.cooldown_bars}bars")

        if self._notifier:
            from datetime import datetime
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " UTC+8"
            price_lines = "\n".join(
                f"{inst_id.split('.')[0]}: {self._fetch_last_price(inst_id)}"
                for inst_id in self.cfg.instrument_ids
            )
            self._notifier.send_text(
                f"⚡ 波动监控已启动\n"
                f"时间: {now}\n"
                f"当前价格:\n{price_lines}\n"
                f"规则: {default_desc}"
            )

    def on_stop(self) -> None:
        for inst_str in self.cfg.instrument_ids:
            bar_type = BarType.from_str(f"{inst_str}-{self.cfg.bar_spec}")
            self.unsubscribe_bars(bar_type)

    # ── Bar handler ───────────────────────────────────────────────────────────

    def on_bar(self, bar: Bar) -> None:
        key = str(bar.bar_type.instrument_id)
        closes = self._closes[key]
        closes.append(bar.close.as_double())

        if self._cooldown[key] > 0:
            self._cooldown[key] -= 1
            return

        rules = self._per_instrument_rules.get(key, self._default_rules)
        for rule in rules:
            result = rule.evaluate(closes)
            if result is not None:
                self._alert(key, result)
                self._cooldown[key] = self.cfg.cooldown_bars
                break  # one alert per bar per instrument

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_last_price(self, instrument_id: str) -> str:
        """Fetch current last price from OKX public ticker API."""
        import httpx
        # Convert nautilus format "ETH-USDT-SWAP.OKX" → OKX instId "ETH-USDT-SWAP"
        inst_id = instrument_id.split(".")[0]
        try:
            resp = httpx.get(
                "https://www.okx.com/api/v5/market/ticker",
                params={"instId": inst_id},
                timeout=5,
            )
            data = resp.json()
            last = float(data["data"][0]["last"])
            return f"{last:,.2f}"
        except Exception as e:
            logger.warning(f"[VolatilityMonitor] 获取 {inst_id} 价格失败: {e}")
            return "N/A"

    def _alert(self, instrument_key: str, result: "AlertResult") -> None:
        from datetime import datetime

        symbol = instrument_key.split(".")[0]
        arrow = "🔺" if result.rising else "🔻"
        direction = "上涨" if result.rising else "下跌"
        signed_pct = f"+{result.pct:.2f}%" if result.rising else f"-{result.pct:.2f}%"
        _now = datetime.now().astimezone()
        _offset = _now.strftime("%z")  # e.g. "+0800"
        _tz = f"UTC{_offset[:3]}:{_offset[3:]}"  # e.g. "UTC+08:00"
        now = _now.strftime("%H:%M:%S") + f" ({_tz})"

        title = f"{arrow} {symbol}  {signed_pct}"
        body = (
            f"{result.label}\n"
            f"{result.window_bars}分钟内{direction}  {result.from_price:,.2f} → {result.to_price:,.2f}\n"
            f"波动幅度: {signed_pct}  时间: {now}"
        )

        logger.warning(
            f"[VolatilityMonitor] {symbol} | {result.label} | "
            f"{result.window_bars}min {direction} "
            f"{result.from_price:,.2f}→{result.to_price:,.2f} ({signed_pct}) @ {now}"
        )

        if self._notifier:
            self._notifier.send_volatility_alert(
                symbol=symbol,
                title=title,
                body=body,
            )
