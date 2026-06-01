"""Live trading engine — connects to OKX mainnet via TradingNode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger
from nautilus_trader.adapters.okx.config import OKXDataClientConfig, OKXExecClientConfig
from nautilus_trader.adapters.okx.factories import (
    OKXLiveDataClientFactory,
    OKXLiveExecClientFactory,
)
from nautilus_trader.config import (
    ImportableStrategyConfig,
    LiveDataEngineConfig,
    LiveExecEngineConfig,
    LiveRiskEngineConfig,
    LoggingConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.node import TradingNode

from cyber_trader.config import get_settings


@dataclass
class LiveConfig:
    """Configuration for live trading."""

    strategy_path: str
    config_path: str
    strategy_config: dict[str, Any]

    instrument_ids: list[str]
    bar_types: list[str]

    log_level: str = "INFO"

    # Safety guard: must be explicitly set to True to allow live orders
    confirmed: bool = False


class LiveRunner:
    """
    Live trading runner connecting to OKX mainnet.

    CAUTION: This places real orders with real money.
    Set confirmed=True in LiveConfig to acknowledge this.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def build_node(self, cfg: LiveConfig) -> TradingNode:
        settings = self._settings

        strat_kwargs = dict(cfg.strategy_config)
        strat_kwargs.setdefault("enable_notifications", True)

        data_client_config = OKXDataClientConfig(
            api_key=settings.okx_api_key,
            api_secret=settings.okx_api_secret,
            passphrase=settings.okx_passphrase,
            is_demo=False,
        )

        exec_client_config = OKXExecClientConfig(
            api_key=settings.okx_api_key,
            api_secret=settings.okx_api_secret,
            passphrase=settings.okx_passphrase,
            is_demo=False,
        )

        node_config = TradingNodeConfig(
            trader_id=settings.trader_id,
            logging=LoggingConfig(log_level=cfg.log_level),
            data_engine=LiveDataEngineConfig(),
            risk_engine=LiveRiskEngineConfig(bypass=False),
            exec_engine=LiveExecEngineConfig(),
            data_clients={"OKX": data_client_config},
            exec_clients={"OKX": exec_client_config},
            strategies=[
                ImportableStrategyConfig(
                    strategy_path=cfg.strategy_path,
                    config_path=cfg.config_path,
                    config=strat_kwargs,
                )
            ],
            timeout_connection=30.0,
            timeout_reconciliation=10.0,
            timeout_portfolio=10.0,
            timeout_disconnection=10.0,
            timeout_post_stop=5.0,
        )

        node = TradingNode(config=node_config)
        node.add_data_client_factory("OKX", OKXLiveDataClientFactory)
        node.add_exec_client_factory("OKX", OKXLiveExecClientFactory)
        node.build()
        return node

    def run(self, cfg: LiveConfig) -> None:
        if not cfg.confirmed:
            raise RuntimeError(
                "Live trading requires confirmed=True in LiveConfig. "
                "This places REAL orders with REAL money."
            )

        logger.warning("⚠️  Starting LIVE trading on OKX MAINNET — real orders will be placed!")
        node = self.build_node(cfg)
        try:
            node.run()
        except KeyboardInterrupt:
            logger.info("Stopping live trading (KeyboardInterrupt)")
        finally:
            node.stop()
            node.dispose()
