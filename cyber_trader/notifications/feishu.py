"""Feishu (Lark) robot webhook notifications."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
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


class FeishuNotifier:
    """Send rich-card messages to a Feishu group robot."""

    def __init__(self, webhook_url: str, secret: str = "") -> None:
        self._webhook_url = webhook_url
        self._secret = secret
        self._client = httpx.Client(timeout=10)

    def _sign(self, timestamp: int) -> str:
        if not self._secret:
            return ""
        msg = f"{timestamp}\n{self._secret}"
        mac = hmac.new(
            self._secret.encode("utf-8"),
            msg.encode("utf-8"),
            digestmod=hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode("utf-8")

    def _post(self, payload: dict) -> bool:
        if not self._webhook_url:
            logger.debug("Feishu webhook not configured, skipping notification")
            return False
        try:
            ts = int(time.time())
            payload["timestamp"] = str(ts)
            sign = self._sign(ts)
            if sign:
                payload["sign"] = sign

            resp = self._client.post(self._webhook_url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code", 0) != 0:
                logger.warning(f"Feishu API error: {data}")
                return False
            return True
        except Exception as e:
            logger.error(f"Feishu notification failed: {e}")
            return False

    def send_signal(self, signal: TradeSignal) -> bool:
        direction_emoji = {"LONG": "📈", "SHORT": "📉", "CLOSE": "🔄"}[signal.direction]
        color = {"LONG": "green", "SHORT": "red", "CLOSE": "orange"}[signal.direction]

        factor_lines = "\n".join(
            f"  • {k}: **{v:+.3f}**" for k, v in signal.factors.items()
        )

        content = (
            f"{direction_emoji} **{signal.direction}** {signal.symbol} @ {signal.price:,.4f}\n\n"
            f"**策略**: {signal.strategy}　**周期**: {signal.timeframe}\n"
            f"**综合得分**: {signal.composite_score:+.3f}\n\n"
            f"**因子明细**:\n{factor_lines}"
        )
        if signal.extra:
            extra_lines = "\n".join(f"  {k}: {v}" for k, v in signal.extra.items())
            content += f"\n\n**附加信息**:\n{extra_lines}"

        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"content": "量化交易信号", "tag": "plain_text"},
                    "template": color,
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {"content": content, "tag": "lark_md"},
                    },
                    {
                        "tag": "note",
                        "elements": [
                            {
                                "tag": "plain_text",
                                "content": f"CyberTrader · {time.strftime('%Y-%m-%d %H:%M:%S')}",
                            }
                        ],
                    },
                ],
            },
        }
        return self._post(payload)

    def send_metrics(self, metrics: TradeMetrics) -> bool:
        pnl_emoji = "💰" if metrics.realized_pnl >= 0 else "🔴"
        content = (
            f"**策略**: {metrics.strategy}　**交易对**: {metrics.symbol}\n\n"
            f"{pnl_emoji} **已实现盈亏**: {metrics.realized_pnl:+.2f} USDT\n"
            f"📊 **未实现盈亏**: {metrics.unrealized_pnl:+.2f} USDT\n\n"
            f"| 指标 | 值 |\n|---|---|\n"
            f"| 胜率 | {metrics.win_rate:.1%} |\n"
            f"| Sharpe | {metrics.sharpe:.2f} |\n"
            f"| 最大回撤 | {metrics.max_drawdown:.1%} |\n"
            f"| 总交易次数 | {metrics.total_trades} |"
        )
        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"content": "交易绩效报告", "tag": "plain_text"},
                    "template": "blue",
                },
                "elements": [
                    {"tag": "div", "text": {"content": content, "tag": "lark_md"}},
                    {
                        "tag": "note",
                        "elements": [
                            {
                                "tag": "plain_text",
                                "content": f"CyberTrader · {time.strftime('%Y-%m-%d %H:%M:%S')}",
                            }
                        ],
                    },
                ],
            },
        }
        return self._post(payload)

    def send_text(self, text: str) -> bool:
        payload = {"msg_type": "text", "content": {"text": text}}
        return self._post(payload)

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
