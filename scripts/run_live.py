"""Run live trading from a YAML config file.

Usage:
  python scripts/run_live.py --config config/live_trading.yaml --confirm

The --confirm flag is required to prevent accidental live order placement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from cyber_trader.engines.live import LiveConfig, LiveRunner


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to live trading YAML")
@click.option(
    "--confirm",
    is_flag=True,
    default=False,
    help="⚠️  Required: confirms you want to place REAL orders with REAL money",
)
def main(config: str, confirm: bool) -> None:
    """Start live trading on OKX mainnet. Requires --confirm flag."""
    if not confirm:
        click.echo(
            "⚠️  WARNING: Live trading places real orders with real money.\n"
            "Pass --confirm to proceed.\n"
            "Tip: use run_paper.py first to validate your strategy.",
            err=True,
        )
        sys.exit(1)

    raw = _load_config(config)
    live_cfg = raw["live"]
    strat = raw["strategy"]

    cfg = LiveConfig(
        strategy_path=strat["path"],
        config_path=strat["config_path"],
        strategy_config=strat.get("params", {}),
        instrument_ids=live_cfg.get("instrument_ids", []),
        bar_types=live_cfg.get("bar_types", []),
        log_level=live_cfg.get("log_level", "INFO"),
        confirmed=True,
    )

    runner = LiveRunner()
    runner.run(cfg)


if __name__ == "__main__":
    main()
