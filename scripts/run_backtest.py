"""Run a backtest from a YAML config file.

Usage:
  python scripts/run_backtest.py --config config/backtest.yaml
  python scripts/run_backtest.py --config config/backtest.yaml --timeframe 15m
  python scripts/run_backtest.py --config config/backtest.yaml --start 2024-06-01 --end 2024-12-31
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from cyber_trader.data.okx_downloader import SUPPORTED_TIMEFRAMES
from cyber_trader.engines.backtest import BacktestConfig, BacktestRunner


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to backtest YAML")
@click.option("--timeframe", default=None, type=click.Choice(SUPPORTED_TIMEFRAMES), help="Override timeframe")
@click.option("--start", default=None, help="Override start date")
@click.option("--end", default=None, help="Override end date")
@click.option("--balance", default=None, type=float, help="Override starting balance (USDT)")
@click.option("--log-level", default=None, type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def main(
    config: str,
    timeframe: str | None,
    start: str | None,
    end: str | None,
    balance: float | None,
    log_level: str | None,
) -> None:
    """Run a backtest using the specified YAML configuration."""
    raw = _load_config(config)
    bt = raw["backtest"]
    strat = raw["strategy"]

    cfg = BacktestConfig(
        strategy_path=strat["path"],
        config_path=strat["config_path"],
        strategy_config=strat.get("params", {}),
        instrument_id=bt["instrument_id"],
        timeframe=timeframe or bt["timeframe"],
        start=start or bt["start"],
        end=end or bt["end"],
        venue=bt.get("venue", "OKX"),
        starting_balance=balance or bt.get("starting_balance", 10_000.0),
        currency=bt.get("currency", "USDT"),
        fill_model_slippage_factor=bt.get("fill_model_slippage_factor", 0.0001),
        log_level=log_level or bt.get("log_level", "WARNING"),
        higher_timeframes=bt.get("higher_timeframes", []),
    )

    runner = BacktestRunner()
    result = runner.run(cfg)
    result.print_summary()


if __name__ == "__main__":
    main()
