"""Bark push notification client (iOS)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import httpx
from loguru import logger


@dataclass
class TradeSignal:
    symbol: str
    direction: Literal["LONG", "SHORT", "CLOSE"]
    strategy: str
    price: float
    composite_score: float
    factors: dict[str, float]
    timeframe: str
    extra: dict | None = None


@dataclass
class TradeMetrics:
    symbol: str
    strategy: str
    realized_pnl: float
    unrealized_pnl: float
    win_rate: float
    sharpe: float
    max_drawdown: float
    total_trades: int


class BarkNotifier:
    """Send push notifications to an iOS device via Bark."""

    def __init__(self, device_key: str, server: str = "https://api.day.app") -> None:
        self._key = device_key
        self._server = server.rstrip("/")
        self._client = httpx.Client(timeout=10)

    def _post(self, title: str, body: str, group: str = "CyberTrader", level: str = "active") -> bool:
        if not self._key:
            logger.debug("Bark device key not configured, skipping notification")
            return False
        try:
            resp = self._client.post(
                f"{self._server}/push",
                json={
                    "title": title,
                    "body": body,
                    "device_key": self._key,
                    "group": group,
                    "level": level,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code", 200) != 200:
                logger.warning(f"Bark API error: {data}")
                return False
            return True
        except Exception as e:
            logger.error(f"Bark notification failed: {e}")
            return False

    def send_signal(self, signal: TradeSignal) -> bool:
        direction_emoji = {"LONG": "📈", "SHORT": "📉", "CLOSE": "🔄"}[signal.direction]
        title = f"{direction_emoji} {signal.direction} {signal.symbol}"

        factor_lines = "\n".join(
            f"  {k}: {v:+.3f}" for k, v in signal.factors.items()
        )
        body = (
            f"价格: {signal.price:,.4f}  得分: {signal.composite_score:+.3f}\n"
            f"策略: {signal.strategy}  周期: {signal.timeframe}\n"
            f"{factor_lines}"
        )
        if signal.extra:
            body += "\n" + "\n".join(f"{k}: {v}" for k, v in signal.extra.items())

        return self._post(title, body)

    def send_metrics(self, metrics: TradeMetrics) -> bool:
        pnl_emoji = "💰" if metrics.realized_pnl >= 0 else "🔴"
        title = f"{pnl_emoji} 绩效报告 {metrics.symbol}"
        body = (
            f"已实现盈亏: {metrics.realized_pnl:+.2f} USDT\n"
            f"未实现盈亏: {metrics.unrealized_pnl:+.2f} USDT\n"
            f"胜率: {metrics.win_rate:.1%}  Sharpe: {metrics.sharpe:.2f}\n"
            f"最大回撤: {metrics.max_drawdown:.1%}  总交易: {metrics.total_trades}"
        )
        return self._post(title, body)

    def send_text(self, text: str) -> bool:
        lines = text.strip().splitlines()
        title = lines[0] if lines else "CyberTrader"
        body = "\n".join(lines[1:]) if len(lines) > 1 else ""
        return self._post(title, body)

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
