"""Run the volatility anomaly monitor.

Usage:
  python scripts/run_monitor.py --config config/monitor.yaml

The monitor connects to OKX (demo or live) via the same data feed used for
paper trading but never places any orders.  Bark push notifications are sent
when price velocity exceeds the configured thresholds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from cyber_trader.engines.paper import PaperConfig, PaperRunner


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@click.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Path to monitor YAML config",
)
@click.option(
    "--live",
    is_flag=True,
    default=False,
    help="Connect to OKX mainnet instead of demo (data-only, no orders)",
)
def main(config: str, live: bool) -> None:
    """Start the volatility anomaly monitor."""
    raw = _load_config(config)
    mon = raw["monitor"]
    strat = raw["strategy"]

    if live:
        logger.info("Connecting to OKX MAINNET for market data (no orders will be placed)")
    else:
        logger.info("Connecting to OKX DEMO for market data")

    cfg = PaperConfig(
        strategy_path=strat["path"],
        config_path=strat["config_path"],
        strategy_config=strat.get("params", {}),
        instrument_ids=mon.get("instrument_ids", []),
        bar_types=mon.get("bar_types", []),
        is_demo=not live,
        log_level=mon.get("log_level", "INFO"),
    )

    runner = PaperRunner()
    runner.run(cfg)


if __name__ == "__main__":
    main()
