"""Plot K-line chart with EMA and MACD indicators.

Usage:
  python scripts/plot_kline.py
  python scripts/plot_kline.py --bars 300 --out charts/eth_kline.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
import mplfinance as mpf

sys.path.insert(0, str(Path(__file__).parent.parent))

from cyber_trader.data.catalog import get_catalog


def load_df(bar_type: str, n_bars: int) -> pd.DataFrame:
    catalog = get_catalog()
    bars = catalog.bars([bar_type])
    if not bars:
        raise RuntimeError(f"No data for bar type: {bar_type}")

    bars = bars[-n_bars:]
    df = pd.DataFrame(
        [
            {
                "Open":   b.open.as_double(),
                "High":   b.high.as_double(),
                "Low":    b.low.as_double(),
                "Close":  b.close.as_double(),
                "Volume": b.volume.as_double(),
            }
            for b in bars
        ],
        index=pd.DatetimeIndex(
            [pd.Timestamp(b.ts_event, unit="ns", tz="UTC") for b in bars]
        ),
    )
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["EMA9"]  = df["Close"].ewm(span=9,  adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]   = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"]   = df["MACD"] - df["Signal"]
    return df


@click.command()
@click.option("--bar-type", default="ETH-USDT-SWAP.OKX-1-MINUTE-LAST-EXTERNAL",
              help="Nautilus bar type string")
@click.option("--bars", default=200, show_default=True, help="Number of bars to plot")
@click.option("--out", default="eth_kline.png", show_default=True, help="Output image path")
def main(bar_type: str, bars: int, out: str) -> None:
    df = load_df(bar_type, bars)
    df = add_indicators(df)

    addplots = [
        mpf.make_addplot(df["EMA9"],   color="#f59e0b", width=1.2, label="EMA9"),
        mpf.make_addplot(df["EMA21"],  color="#06b6d4", width=1.2, label="EMA21"),
        mpf.make_addplot(df["Hist"],   type="bar", panel=2, color="gray", alpha=0.5, ylabel="MACD"),
        mpf.make_addplot(df["MACD"],   panel=2, color="#3b82f6", width=1.0, label="MACD"),
        mpf.make_addplot(df["Signal"], panel=2, color="#ef4444", width=1.0, label="Signal"),
    ]

    style = mpf.make_mpf_style(
        base_mpf_style="charles",
        rc={"font.size": 10},
        gridstyle="--",
        gridcolor="#334155",
    )

    Path(out).parent.mkdir(parents=True, exist_ok=True)

    mpf.plot(
        df,
        type="candle",
        style=style,
        title=f"\n{bar_type}  |  last {bars} bars",
        ylabel="Price (USDT)",
        volume=True,
        addplot=addplots,
        panel_ratios=(4, 1, 2),
        figsize=(16, 10),
        savefig=dict(fname=out, dpi=150, bbox_inches="tight"),
    )
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
