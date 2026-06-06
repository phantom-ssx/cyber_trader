"""Incrementally collect OKX liquidation orders and save as aggregated Parquet.

OKX public API exposes ~24h of order-level liquidation history with no API key.
Run this script daily (or every 12h with --hours 13) via cron to accumulate data.

Usage:
  python scripts/download_liquidation.py --symbol ETH-USDT
  python scripts/download_liquidation.py --symbol ETH-USDT,BTC-USDT --interval 4h
  python scripts/download_liquidation.py --symbol ETH-USDT --hours 13 --interval 1h

Cron example (every 12 hours):
  0 */12 * * *  cd /path/to/cyber_trader && uv run python scripts/download_liquidation.py --symbol ETH-USDT,BTC-USDT --hours 13
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

from cyber_trader.config import get_settings
from cyber_trader.data.liquidation_downloader import (
    SUPPORTED_INTERVALS,
    OKXLiquidationDownloader,
)


@click.command()
@click.option("--symbol", required=True, help="Symbol(s) comma-separated, e.g. ETH-USDT,BTC-USDT")
@click.option(
    "--interval",
    default="1h",
    type=click.Choice(list(SUPPORTED_INTERVALS)),
    help="Aggregation bucket size (default: 1h)",
)
@click.option(
    "--hours",
    default=25,
    type=int,
    help="Lookback window in hours (default: 25, use >24 to avoid gaps with daily cron)",
)
def main(symbol: str, interval: str, hours: int) -> None:
    """Fetch OKX liquidation orders and append to local Parquet files."""
    settings = get_settings()
    symbols = [s.strip() for s in symbol.split(",")]

    downloader = OKXLiquidationDownloader()

    click.echo(f"\n── Fetching OKX liquidations [{interval}] last {hours}h ─────────────────────")

    results = asyncio.run(
        downloader.fetch_multi(symbols=symbols, hours=hours, interval=interval)
    )

    click.echo("\n── Summary ──────────────────────────────────────────────────────────────")
    for sym, count in results.items():
        status = f"{count} buckets saved" if count else "no data"
        click.echo(f"  {sym}: {status}")
    click.echo(f"\nData saved to: {settings.data_catalog_path.parent / 'aux'}")


if __name__ == "__main__":
    main()
