"""Download historical OHLCV data from OKX and write to local Parquet catalog.

Usage:
  python scripts/download_data.py --symbol ETH-USDT --timeframe 1h --start 2024-01-01
  python scripts/download_data.py --symbol ETH-USDT,BTC-USDT --timeframe 1h,4h --start 2024-01-01
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from cyber_trader.config import get_settings
from cyber_trader.data.okx_downloader import OKXDownloader


@click.command()
@click.option("--symbol", required=True, help="Symbol(s) comma-separated, e.g. ETH-USDT,BTC-USDT")
@click.option("--timeframe", default="1h", help="Timeframe(s) comma-separated, e.g. 1h,4h,1d")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD or ISO datetime)")
@click.option("--end", default=None, help="End date (default: now)")
@click.option("--market-type", default="swap", type=click.Choice(["swap", "spot"]), help="Market type")
@click.option("--batch-size", default=300, help="Bars per API request")
def main(
    symbol: str,
    timeframe: str,
    start: str,
    end: str | None,
    market_type: str,
    batch_size: int,
) -> None:
    """Download historical data from OKX to local Parquet catalog."""
    settings = get_settings()
    symbols = [s.strip() for s in symbol.split(",")]
    timeframes = [t.strip() for t in timeframe.split(",")]

    logger.info(f"Catalog path: {settings.data_catalog_path}")
    logger.info(f"Symbols: {symbols}  Timeframes: {timeframes}")

    downloader = OKXDownloader(
        api_key=settings.okx_api_key,
        api_secret=settings.okx_api_secret,
        passphrase=settings.okx_passphrase,
        is_demo=settings.okx_is_demo,
    )

    results = asyncio.run(
        downloader.download_multi(
            symbols=symbols,
            timeframes=timeframes,
            start=start,
            end=end,
            market_type=market_type,  # type: ignore[arg-type]
        )
    )

    click.echo("\n── Download summary ──────────────────────────────")
    for key, count in results.items():
        click.echo(f"  {key}: {count:,} bars")
    click.echo(f"\nData saved to: {settings.data_catalog_path}")


if __name__ == "__main__":
    main()
