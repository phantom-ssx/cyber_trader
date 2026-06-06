"""Build an estimated liquidation heatmap from OKX public OI + price data.

Method:
  1. Fetch historical 1h OHLCV and OI (open interest) from OKX public API.
  2. For each hour where OI increased (new positions opened), distribute the
     new OI across leverage tiers to compute the liquidation price of each bucket.
  3. Aggregate by price level ($5 buckets) to get estimated long/short
     liquidation volume at each price.

Limitations:
  - This is a MODEL, not the actual exchange liquidation book.
  - Leverage distribution is assumed from market research defaults.
  - OI delta captures new positions but not exact entry prices.
  - OKX OI history goes back ~30 days (720 1h bars).

Storage: data/aux/{SYMBOL}_liq_heatmap.parquet
  Index: price (float, $5 buckets)
  Columns: long_liq_usd, short_liq_usd, total_liq_usd
"""

from __future__ import annotations

from pathlib import Path

import ccxt
import httpx
import pandas as pd

from cyber_trader.config import get_settings

# OKX maintenance margin rate for ETH/BTC USDT-margined perps (tier 1)
_MMR: dict[str, float] = {
    "ETH-USDT": 0.004,
    "BTC-USDT": 0.004,
}
_DEFAULT_MMR = 0.004

# Leverage distribution: {leverage: weight}
# Based on typical leveraged futures market composition
DEFAULT_LEVERAGE_DIST: dict[int, float] = {
    5:   0.10,
    10:  0.30,
    20:  0.35,
    50:  0.20,
    100: 0.05,
}

# Long/short split for new OI: 55% long, 45% short (slight long bias default)
DEFAULT_LONG_RATIO = 0.55

# Price bucket width in USD
BUCKET_SIZE = 5.0


# ── Storage helpers ───────────────────────────────────────────────────────────

def _aux_dir() -> Path:
    settings = get_settings()
    p = settings.data_catalog_path.parent / "aux"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _heatmap_path(symbol: str) -> Path:
    return _aux_dir() / f"{symbol}_liq_heatmap.parquet"


def load_liq_heatmap(symbol: str) -> pd.DataFrame | None:
    """Load a previously built heatmap. Returns None if not found."""
    path = _heatmap_path(symbol)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = df.index.astype(float)
    return df.sort_index()


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_ohlcv(symbol: str, hours: int) -> pd.DataFrame:
    """Fetch 1h OHLCV from OKX via ccxt. symbol e.g. 'ETH-USDT'."""
    import time
    base, quote = symbol.split("-")
    ccxt_symbol = f"{base}/{quote}:{quote}"

    exchange = ccxt.okx({"options": {"defaultType": "swap"}})
    since_ms = int((time.time() - (hours + 1) * 3600) * 1000)
    ohlcv = exchange.fetch_ohlcv(ccxt_symbol, timeframe="1h", since=since_ms, limit=hours + 5)

    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    df["mid"] = (df["high"] + df["low"]) / 2
    return df


def _fetch_oi_history(symbol: str, hours: int) -> pd.Series:
    """Fetch hourly OI (USD) history from OKX rubik stat API.

    Returns a Series indexed by UTC timestamp, named 'oi_usd'.
    """
    ccy = symbol.split("-")[0]  # "ETH-USDT" → "ETH"
    resp = httpx.get(
        "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume",
        params={"ccy": ccy, "period": "1H", "limit": str(min(hours + 5, 720))},
        timeout=20,
    )
    resp.raise_for_status()
    rows = resp.json()["data"]  # [[ts_ms, oi_usd, vol_usd], ...] newest first

    ts_list = [pd.Timestamp(int(r[0]), unit="ms", tz="UTC") for r in rows]
    oi_list = [float(r[1]) for r in rows]

    return pd.Series(oi_list, index=ts_list, name="oi_usd").sort_index()


# ── Heatmap builder ───────────────────────────────────────────────────────────

def build_heatmap(
    symbol: str,
    hours: int = 72,
    leverage_dist: dict[int, float] | None = None,
    long_ratio: float = DEFAULT_LONG_RATIO,
    bucket_size: float = BUCKET_SIZE,
) -> pd.DataFrame:
    """Build a liquidation heatmap for the given symbol.

    Args:
        symbol:        e.g. "ETH-USDT"
        hours:         Lookback window. OKX provides up to 720h (~30 days).
        leverage_dist: {leverage: weight} dict. Uses DEFAULT_LEVERAGE_DIST if None.
        long_ratio:    Fraction of new OI assumed to be long (default 0.55).
        bucket_size:   Price bucket width in USD (default $5).

    Returns:
        DataFrame indexed by price bucket with columns:
        long_liq_usd, short_liq_usd, total_liq_usd
    """
    lev_dist = leverage_dist or DEFAULT_LEVERAGE_DIST
    mmr = _MMR.get(symbol, _DEFAULT_MMR)

    price_df = _fetch_ohlcv(symbol, hours)
    oi_series = _fetch_oi_history(symbol, hours)

    merged = price_df[["mid", "close"]].join(oi_series, how="inner")
    merged["delta_oi"] = merged["oi_usd"].diff()
    merged = merged.dropna()

    long_liq: dict[float, float] = {}
    short_liq: dict[float, float] = {}

    for _, row in merged.iterrows():
        delta = float(row["delta_oi"])
        if delta <= 0:
            continue  # positions closed or liquidated, not new entries

        entry = float(row["mid"])
        long_oi  = delta * long_ratio
        short_oi = delta * (1.0 - long_ratio)

        for lev, weight in lev_dist.items():
            long_portion  = long_oi  * weight
            short_portion = short_oi * weight

            # Liquidation price formulas (OKX cross-margin, isolated approximation)
            long_liq_px  = round(entry * (1 - 1 / lev + mmr) / bucket_size) * bucket_size
            short_liq_px = round(entry * (1 + 1 / lev - mmr) / bucket_size) * bucket_size

            long_liq[long_liq_px]   = long_liq.get(long_liq_px, 0.0)   + long_portion
            short_liq[short_liq_px] = short_liq.get(short_liq_px, 0.0) + short_portion

    all_prices = sorted(set(long_liq) | set(short_liq))
    df = pd.DataFrame(
        {
            "long_liq_usd":  [long_liq.get(p, 0.0)  for p in all_prices],
            "short_liq_usd": [short_liq.get(p, 0.0) for p in all_prices],
        },
        index=pd.Index(all_prices, name="price"),
    )
    df["total_liq_usd"] = df["long_liq_usd"] + df["short_liq_usd"]
    return df.sort_index()


def build_and_save(
    symbol: str,
    hours: int = 72,
    leverage_dist: dict[int, float] | None = None,
    long_ratio: float = DEFAULT_LONG_RATIO,
    bucket_size: float = BUCKET_SIZE,
) -> tuple[pd.DataFrame, Path]:
    """Build heatmap and save to parquet. Returns (df, path)."""
    df = build_heatmap(symbol, hours, leverage_dist, long_ratio, bucket_size)
    path = _heatmap_path(symbol)
    df.to_parquet(path)
    return df, path


# ── Text report helper ────────────────────────────────────────────────────────

def print_report(
    df: pd.DataFrame,
    current_price: float,
    top_n: int = 10,
    view_pct: float = 0.20,
) -> None:
    """Print a text report of the heatmap around the current price."""
    lo = current_price * (1 - view_pct)
    hi = current_price * (1 + view_pct)
    nearby = df.loc[lo:hi].copy()

    bucket = 25.0
    nearby["bucket"] = (nearby.index // bucket) * bucket
    grouped = nearby.groupby("bucket")[["long_liq_usd", "short_liq_usd", "total_liq_usd"]].sum()

    below = grouped[grouped.index < current_price].sort_values("long_liq_usd", ascending=False)
    above = grouped[grouped.index > current_price].sort_values("short_liq_usd", ascending=False)

    total_long  = grouped["long_liq_usd"].sum()
    total_short = grouped["short_liq_usd"].sum()

    print(f"\n当前价格: ${current_price:,.2f}\n")

    print("=" * 60)
    print(f"▼ 下方多头清算区 TOP {top_n}（价格下跌触发）")
    print("=" * 60)
    for px, row in below.head(top_n).iterrows():
        pct  = row["long_liq_usd"] / total_long * 100 if total_long else 0
        dist = (current_price - px) / current_price * 100
        bar  = "█" * min(int(row["long_liq_usd"] / 3e6), 18)
        print(f"  ${px:>7,.0f}  -{dist:4.1f}%  ${row['long_liq_usd']/1e6:>6.1f}M  {pct:4.1f}%  {bar}")

    print()
    print("=" * 60)
    print(f"▲ 上方空头清算区 TOP {top_n}（价格上涨触发）")
    print("=" * 60)
    for px, row in above.head(top_n).iterrows():
        pct  = row["short_liq_usd"] / total_short * 100 if total_short else 0
        dist = (px - current_price) / current_price * 100
        bar  = "█" * min(int(row["short_liq_usd"] / 3e6), 18)
        print(f"  ${px:>7,.0f}  +{dist:4.1f}%  ${row['short_liq_usd']/1e6:>6.1f}M  {pct:4.1f}%  {bar}")

    print()
    print("=" * 60)
    print("📊 双向梯度图（$25 区间）")
    print("=" * 60)
    max_val = grouped["total_liq_usd"].max() or 1
    for px, row in grouped.sort_index(ascending=False).iterrows():
        liq_l = row["long_liq_usd"]
        liq_s = row["short_liq_usd"]
        bar_l = "◀" * min(int(liq_l / max_val * 12), 12)
        bar_s = "▶" * min(int(liq_s / max_val * 12), 12)
        dist  = (px - current_price) / current_price * 100
        marker = "  ◄ NOW" if abs(dist) < 1.8 else ""
        print(
            f"  ${px:>7,.0f} ({dist:+5.1f}%)  "
            f"{bar_l:>12}|{bar_s:<12}  "
            f"L${liq_l/1e6:4.1f}M S${liq_s/1e6:4.1f}M{marker}"
        )
