"""Backtest engine wrapper around nautilus_trader BacktestNode."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Type

import pandas as pd
from loguru import logger
from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.config import (
    BacktestDataConfig,
    BacktestEngineConfig,
    BacktestRunConfig,
    BacktestVenueConfig,
    ImportableStrategyConfig,
    LoggingConfig,
    RiskEngineConfig,
)
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue

from cyber_trader.config import get_settings
from cyber_trader.data.catalog import get_catalog
from cyber_trader.data.okx_downloader import SUPPORTED_TIMEFRAMES, timeframe_to_bar_type


@dataclass
class BacktestConfig:
    """User-facing config for a single backtest run."""

    strategy_path: str                      # e.g. "cyber_trader.strategies:TrendFollowingStrategy"
    config_path: str                        # e.g. "cyber_trader.strategies:TrendFollowingConfig"
    strategy_config: dict[str, Any]         # kwargs for the strategy config

    instrument_id: str                      # e.g. "ETH-USDT-SWAP.OKX"
    timeframe: str                          # e.g. "1m", "15m", "1h", "4h", "1d"

    start: str                              # ISO datetime e.g. "2024-01-01"
    end: str                                # ISO datetime e.g. "2024-12-31"

    venue: str = "OKX"
    starting_balance: float = 10_000.0
    currency: str = "USDT"

    # Venue simulation settings
    fill_model_slippage_factor: float = 0.0001
    latency_model_base_secs: float = 0.001

    log_level: str = "WARNING"

    # Derived — set automatically in __post_init__
    bar_type: str = field(init=False)

    def __post_init__(self) -> None:
        if self.timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Unsupported timeframe '{self.timeframe}'. "
                f"Choose from: {SUPPORTED_TIMEFRAMES}"
            )
        self.bar_type = timeframe_to_bar_type(self.instrument_id, self.timeframe)


class BacktestRunner:
    """
    Runs a backtest and returns performance statistics.

    Usage:
        runner = BacktestRunner()
        result = runner.run(config)
        print(result.stats)
    """

    def __init__(self) -> None:
        self._catalog = get_catalog()
        self._settings = get_settings()

    def run(self, cfg: BacktestConfig) -> "BacktestResult":
        logger.info(f"Starting backtest: {cfg.strategy_path} on {cfg.instrument_id}")

        # Merge strategy_config with required fields
        strat_kwargs = dict(cfg.strategy_config)
        strat_kwargs.setdefault("instrument_id", cfg.instrument_id)
        strat_kwargs.setdefault("bar_type", cfg.bar_type)
        strat_kwargs["enable_notifications"] = False  # never notify during backtests

        run_config = BacktestRunConfig(
            engine=BacktestEngineConfig(
                logging=LoggingConfig(log_level=cfg.log_level),
                risk_engine=RiskEngineConfig(bypass=False),
                strategies=[
                    ImportableStrategyConfig(
                        strategy_path=cfg.strategy_path,
                        config_path=cfg.config_path,
                        config=strat_kwargs,
                    )
                ],
            ),
            venues=[
                BacktestVenueConfig(
                    name=cfg.venue,
                    oms_type=OmsType.NETTING,
                    account_type=AccountType.MARGIN,
                    starting_balances=[f"{cfg.starting_balance} {cfg.currency}"],
                )
            ],
            data=[
                BacktestDataConfig(
                    catalog_path=str(self._settings.data_catalog_path),
                    data_cls=Bar,
                    instrument_id=cfg.instrument_id,
                    bar_types=[cfg.bar_type],
                    start_time=cfg.start,
                    end_time=cfg.end,
                )
            ],
        )

        node = BacktestNode(configs=[run_config])
        results = node.run()

        logger.info("Backtest complete")
        return BacktestResult(run_results=results, config=cfg)


@dataclass
class BacktestResult:
    run_results: list
    config: BacktestConfig

    def stats(self) -> dict[str, Any]:
        """Extract key performance statistics."""
        if not self.run_results:
            return {}
        r = self.run_results[0]
        try:
            pnls = r.stats_pnls.get("USDT", {})
            rets = r.stats_returns
            return {
                "total_return_pct": pnls.get("PnL% (total)", 0),
                "total_pnl":        pnls.get("PnL (total)", 0),
                "sharpe_ratio":     rets.get("Sharpe Ratio (252 days)", 0),
                "sortino_ratio":    rets.get("Sortino Ratio (252 days)", 0),
                "profit_factor":    rets.get("Profit Factor", 0),
                "win_rate":         pnls.get("Win Rate", 0),
                "avg_winner":       pnls.get("Avg Winner", 0),
                "avg_loser":        pnls.get("Avg Loser", 0),
                "expectancy":       pnls.get("Expectancy", 0),
                "total_orders":     r.total_orders,
            }
        except Exception as e:
            logger.warning(f"Could not extract stats: {e}")
            return {}

    def print_summary(self) -> None:
        from rich.console import Console
        from rich.table import Table

        stats = self.stats()
        console = Console()
        table = Table(title=f"Backtest Results — {self.config.instrument_id}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for k, v in stats.items():
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        console.print(table)
