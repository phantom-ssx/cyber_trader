"""Paper trading engine — connects to OKX demo environment via TradingNode."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from loguru import logger
from nautilus_trader.adapters.okx.config import OKXDataClientConfig, OKXExecClientConfig
from nautilus_trader.adapters.okx.factories import (
    OKXLiveDataClientFactory,
    OKXLiveExecClientFactory,
)
from nautilus_trader.common.config import InstrumentProviderConfig
from nautilus_trader.config import (
    ImportableStrategyConfig,
    LiveDataEngineConfig,
    LiveExecEngineConfig,
    LiveRiskEngineConfig,
    LoggingConfig,
    TradingNodeConfig,
)
from nautilus_trader.core.nautilus_pyo3.okx import OKXEnvironment, OKXInstrumentType
from nautilus_trader.live.node import TradingNode

from cyber_trader.config import get_settings


@dataclass
class PaperConfig:
    """Configuration for paper trading."""

    strategy_path: str
    config_path: str
    strategy_config: dict[str, Any]

    instrument_ids: list[str]     # instruments to subscribe
    bar_types: list[str]          # bar types to subscribe

    is_demo: bool = True
    log_level: str = "INFO"


class PaperRunner:
    """
    Paper trading runner using OKX demo account (is_demo=True).

    OKX demo trading provides real market data with simulated order execution.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def build_node(self, cfg: PaperConfig) -> TradingNode:
        settings = self._settings
        okx_env = OKXEnvironment.DEMO if cfg.is_demo else OKXEnvironment.LIVE

        strat_kwargs = dict(cfg.strategy_config)
        strat_kwargs.setdefault("enable_notifications", True)

        instrument_ids = cfg.instrument_ids
        instrument_provider = InstrumentProviderConfig(
            load_ids=frozenset(instrument_ids),
        )

        data_client_config = OKXDataClientConfig(
            api_key=settings.okx_api_key,
            api_secret=settings.okx_api_secret,
            api_passphrase=settings.okx_passphrase,
            instrument_types=(OKXInstrumentType.SWAP,),
            instrument_provider=instrument_provider,
            environment=okx_env,
        )

        exec_client_config = OKXExecClientConfig(
            api_key=settings.okx_api_key,
            api_secret=settings.okx_api_secret,
            api_passphrase=settings.okx_passphrase,
            instrument_types=(OKXInstrumentType.SWAP,),
            instrument_provider=instrument_provider,
            environment=okx_env,
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

    def _export_env(self, cfg: PaperConfig) -> None:
        import os
        s = self._settings
        os.environ.setdefault("OKX_API_KEY", s.okx_api_key)
        os.environ.setdefault("OKX_API_SECRET", s.okx_api_secret)
        os.environ.setdefault("OKX_API_PASSPHRASE", s.okx_passphrase)
        os.environ["OKX_IS_DEMO"] = "true" if cfg.is_demo else "false"

    def run(self, cfg: PaperConfig) -> None:
        env_label = "OKX demo" if cfg.is_demo else "OKX MAINNET"
        logger.info(f"Starting paper trading ({env_label})")
        self._export_env(cfg)
        node = self.build_node(cfg)
        try:
            node.run()
        except KeyboardInterrupt:
            logger.info("Stopping paper trading (KeyboardInterrupt)")
        finally:
            node.stop()
            node.dispose()
