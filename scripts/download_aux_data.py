"""Download auxiliary market data (funding rate, long/short ratio) from OKX.

Usage:
  python scripts/download_aux_data.py --symbol ETH-USDT --start 2025-01-01
  python scripts/download_aux_data.py --symbol ETH-USDT,BTC-USDT --type all --start 2025-01-01
  python scripts/download_aux_data.py --symbol ETH-USDT --type long_short_ratio --period 1H --start 2025-01-01
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from cyber_trader.config import get_settings
from cyber_trader.data.aux_downloader import OKXAuxDownloader, save_series


@click.command()
@click.option("--symbol", required=True, help="Symbol(s) comma-separated, e.g. ETH-USDT,BTC-USDT")
@click.option(
    "--type", "data_type",
    default="all",
    type=click.Choice(["funding_rate", "long_short_ratio", "all"]),
    help="Data type to download (default: all)",
)
@click.option("--start", required=True, help="Start date (YYYY-MM-DD or ISO datetime)")
@click.option("--end", default=None, help="End date (default: now)")
@click.option(
    "--period",
    default="1H",
    type=click.Choice(["5m", "1H", "4H", "1D"]),
    help="Granularity for long/short ratio (default: 1H)",
)
def main(
    symbol: str,
    data_type: str,
    start: str,
    end: str | None,
    period: str,
) -> None:
    """Download funding rate and/or long/short ratio history from OKX."""
    settings = get_settings()
    symbols = [s.strip() for s in symbol.split(",")]

    downloader = OKXAuxDownloader(
        api_key=settings.okx_api_key,
        api_secret=settings.okx_api_secret,
        passphrase=settings.okx_passphrase,
    )

    async def run() -> None:
        try:
            for sym in symbols:
                if data_type in ("funding_rate", "all"):
                    series = await downloader.fetch_funding_rate_history(sym, start, end)
                    if not series.empty:
                        path = save_series(sym, "funding_rate", series)
                        click.echo(f"  funding_rate  {sym}: {len(series):,} rows → {path}")
                    else:
                        click.echo(f"  funding_rate  {sym}: no data returned", err=True)

                if data_type in ("long_short_ratio", "all"):
                    series = await downloader.fetch_long_short_ratio(sym, period, start, end)
                    if not series.empty:
                        path = save_series(sym, "long_short_ratio", series)
                        click.echo(f"  long_short_ratio {sym}: {len(series):,} rows → {path}")
                    else:
                        click.echo(f"  long_short_ratio {sym}: no data returned", err=True)
        finally:
            await downloader.close()

    click.echo(f"\n── Downloading aux data ({data_type}) ─────────────────────────")
    asyncio.run(run())
    click.echo("Done.")


if __name__ == "__main__":
    main()
