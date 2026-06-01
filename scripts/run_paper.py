"""Run paper trading from a YAML config file.

Usage:
  python scripts/run_paper.py --config config/paper_trading.yaml
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
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to paper trading YAML")
def main(config: str) -> None:
    """Start paper trading using OKX demo environment."""
    raw = _load_config(config)
    paper = raw["paper"]
    strat = raw["strategy"]

    cfg = PaperConfig(
        strategy_path=strat["path"],
        config_path=strat["config_path"],
        strategy_config=strat.get("params", {}),
        instrument_ids=paper.get("instrument_ids", []),
        bar_types=paper.get("bar_types", []),
        log_level=paper.get("log_level", "INFO"),
    )

    runner = PaperRunner()
    runner.run(cfg)


if __name__ == "__main__":
    main()
