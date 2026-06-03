"""Resample existing 1-minute catalog bars into higher timeframes.

The OKX downloader stored 1-MINUTE bars in the ParquetDataCatalog. Trading a
trend/mean-reversion strategy on 1-minute data just chases microstructure noise
(very low win rate). This utility aggregates the stored 1m bars into higher
timeframe EXTERNAL bars (e.g. 1h, 4h, 1d) and writes them back to the catalog so
backtests can run on a cleaner signal without needing to re-download.

Usage:
  python scripts/resample_bars.py --instrument ETH-USDT-SWAP.OKX --timeframes 1h 4h 1d
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import AggregationSource, PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity

from cyber_trader.data.catalog import get_catalog
from cyber_trader.data.okx_downloader import _TIMEFRAME_MAP, SUPPORTED_TIMEFRAMES

# nautilus BarAggregation step → pandas resample rule
_PANDAS_RULE: dict[str, str] = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h", "1d": "1D",
}


def _bars_to_frame(bars: list[Bar]) -> pd.DataFrame:
    rows = [
        {
            "ts": pd.Timestamp(b.ts_event, unit="ns", tz="UTC"),
            "open": b.open.as_double(),
            "high": b.high.as_double(),
            "low": b.low.as_double(),
            "close": b.close.as_double(),
            "volume": b.volume.as_double(),
        }
        for b in bars
    ]
    df = pd.DataFrame(rows).set_index("ts").sort_index()
    return df


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = df.resample(rule, label="right", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return agg.dropna(subset=["open"])


@click.command()
@click.option("--instrument", default="ETH-USDT-SWAP.OKX", help="Instrument id")
@click.option(
    "--timeframes",
    multiple=True,
    default=("1h", "4h", "1d"),
    type=click.Choice([t for t in SUPPORTED_TIMEFRAMES if t != "1m"]),
    help="Target timeframes to build from 1m",
)
def main(instrument: str, timeframes: tuple[str, ...]) -> None:
    catalog = get_catalog()
    instrument_id = InstrumentId.from_str(instrument)

    src_bar_type = f"{instrument}-1-MINUTE-LAST-EXTERNAL"
    logger.info(f"Loading 1m bars: {src_bar_type}")
    bars = catalog.query(data_cls=Bar, bar_types=[src_bar_type])
    if not bars:
        raise SystemExit(f"No 1m bars found for {instrument}; download first.")
    logger.info(f"Loaded {len(bars)} 1m bars")

    df = _bars_to_frame(bars)

    # Reuse instrument precision from a sample bar.
    price_prec = bars[0].open.precision
    size_prec = bars[0].volume.precision

    for tf in timeframes:
        aggregation, step = _TIMEFRAME_MAP[tf]
        rule = _PANDAS_RULE[tf]
        agg = _resample(df, rule)

        bar_type = BarType(
            instrument_id,
            BarSpecification(step, aggregation, PriceType.LAST),
            AggregationSource.EXTERNAL,
        )

        out: list[Bar] = []
        for ts, row in agg.iterrows():
            ts_ns = int(ts.value)
            out.append(
                Bar(
                    bar_type=bar_type,
                    open=Price(row["open"], price_prec),
                    high=Price(row["high"], price_prec),
                    low=Price(row["low"], price_prec),
                    close=Price(row["close"], price_prec),
                    volume=Quantity(row["volume"], size_prec),
                    ts_event=ts_ns,
                    ts_init=ts_ns,
                )
            )
        catalog.write_data(out)
        logger.info(f"Wrote {len(out)} {tf} bars ({bar_type})")


if __name__ == "__main__":
    main()
