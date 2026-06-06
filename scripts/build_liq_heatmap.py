"""Build an estimated liquidation heatmap from OKX public OI + OHLCV data.

No API key required. Uses OKX public endpoints only.

Usage:
  python scripts/build_liq_heatmap.py --symbol ETH-USDT
  python scripts/build_liq_heatmap.py --symbol ETH-USDT --hours 168
  python scripts/build_liq_heatmap.py --symbol BTC-USDT --hours 72 --top 15
  python scripts/build_liq_heatmap.py --symbol ETH-USDT --no-save
"""

from __future__ import annotations

import sys
from pathlib import Path

import ccxt
import click

sys.path.insert(0, str(Path(__file__).parent.parent))

from cyber_trader.data.liq_heatmap import (
    DEFAULT_LEVERAGE_DIST,
    build_and_save,
    build_heatmap,
    print_report,
)


@click.command()
@click.option("--symbol", default="ETH-USDT", show_default=True, help="Symbol, e.g. ETH-USDT")
@click.option("--hours",  default=72,  show_default=True, help="OI history lookback (max 720 = 30 days)")
@click.option("--top",    default=10,  show_default=True, help="Top N rows to show per side")
@click.option("--view",   default=20,  show_default=True, help="Price view range ±%% around current price")
@click.option("--bucket", default=5.0, show_default=True, help="Price bucket width in USD")
@click.option("--long-ratio", default=0.55, show_default=True, help="Assumed long fraction of new OI")
@click.option("--no-save", is_flag=True, default=False, help="Skip saving to parquet")
def main(
    symbol: str,
    hours: int,
    top: int,
    view: int,
    bucket: float,
    long_ratio: float,
    no_save: bool,
) -> None:
    """Build and display an estimated liquidation heatmap."""
    click.echo(f"\n── Building liquidation heatmap: {symbol} [{hours}h lookback] ─────────────")
    click.echo(f"   Leverage distribution: {DEFAULT_LEVERAGE_DIST}")
    click.echo(f"   Long/short ratio for new OI: {long_ratio:.0%} / {1-long_ratio:.0%}")

    if no_save:
        df = build_heatmap(symbol, hours=hours, long_ratio=long_ratio, bucket_size=bucket)
        click.echo("   (not saved)")
    else:
        df, path = build_and_save(symbol, hours=hours, long_ratio=long_ratio, bucket_size=bucket)
        click.echo(f"   Saved → {path}")

    # Fetch current price
    exchange = ccxt.okx({"options": {"defaultType": "swap"}})
    base, quote = symbol.split("-")
    ticker = exchange.fetch_ticker(f"{base}/{quote}:{quote}")
    current_price = float(ticker["last"])

    print_report(df, current_price, top_n=top, view_pct=view / 100)


if __name__ == "__main__":
    main()
