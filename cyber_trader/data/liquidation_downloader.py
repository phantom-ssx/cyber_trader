"""Collect OKX liquidation orders and aggregate into time-bucketed Parquet files.

Storage layout: data/aux/{SYMBOL}_liquidation_{interval}.parquet
  - SYMBOL: e.g. ETH-USDT, BTC-USDT
  - interval: 1h | 4h | 8h | 12h | 1d

Each file is a DataFrame with UTC-indexed columns:
  - long_liq_usd:  USD value of long positions liquidated
  - short_liq_usd: USD value of short positions liquidated
  - total_liq_usd: sum of both

OKX public endpoint returns ~24h of data per call. Run this script daily
(or every 12h with --hours 13) via cron to accumulate history over time.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger

from cyber_trader.config import get_settings

OKX_BASE = "https://www.okx.com"

SUPPORTED_INTERVALS = ("1h", "4h", "8h", "12h", "1d")
_INTERVAL_MINUTES: dict[str, int] = {
    "1h":  60,
    "4h":  240,
    "8h":  480,
    "12h": 720,
    "1d":  1440,
}


# ── Storage helpers ───────────────────────────────────────────────────────────

def _aux_dir() -> Path:
    settings = get_settings()
    p = settings.data_catalog_path.parent / "aux"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _liq_path(symbol: str, interval: str) -> Path:
    safe = symbol.replace("/", "_")
    return _aux_dir() / f"{safe}_liquidation_{interval}.parquet"


def load_liquidation(symbol: str, interval: str = "1h") -> pd.DataFrame | None:
    """Load stored liquidation DataFrame. Returns None if file doesn't exist."""
    path = _liq_path(symbol, interval)
    if not path.exists():
        logger.warning(f"Liquidation data not found: {path}  (run download_liquidation.py first)")
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def _save_liquidation(symbol: str, interval: str, df: pd.DataFrame) -> Path:
    """Persist aggregated liquidation data, merging with existing rows."""
    path = _liq_path(symbol, interval)
    if path.exists():
        existing = load_liquidation(symbol, interval)
        if existing is not None:
            df = pd.concat([existing, df]).groupby(level=0).last().sort_index()
    df.to_parquet(path)
    logger.info(f"Saved {len(df)} rows → {path}")
    return path


# ── Raw order fetch ───────────────────────────────────────────────────────────

async def _fetch_okx_liq_orders(
    client: httpx.AsyncClient,
    inst_family: str,   # e.g. "ETH-USDT"
    since_ms: int,
    end_ms: int,
) -> list[dict]:
    """Paginate OKX /api/v5/public/liquidation-orders backward from end_ms to since_ms.

    Returns flat list of raw detail dicts with keys: ts, posSide, sz, bkPx.
    """
    records: list[dict] = []
    cursor = end_ms  # 'after' param = return records older than this ts

    while cursor > since_ms:
        try:
            resp = await client.get(
                f"{OKX_BASE}/api/v5/public/liquidation-orders",
                params={
                    "instType": "SWAP",
                    "instFamily": inst_family,
                    "state": "filled",
                    "limit": "100",
                    "after": str(cursor),
                },
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.error(f"OKX liq fetch error at cursor={cursor}: {exc}")
            await asyncio.sleep(3)
            break

        if payload.get("code") != "0":
            logger.error(f"OKX API error: {payload.get('msg')}")
            break

        batches = payload.get("data", [])
        if not batches:
            break

        oldest_ts = cursor
        for batch in batches:
            for det in batch.get("details", []):
                ts_ms = int(det["ts"])
                if ts_ms >= since_ms:
                    records.append(det)
                if ts_ms < oldest_ts:
                    oldest_ts = ts_ms

        # Stop paginating if oldest record is before our window
        if oldest_ts <= since_ms:
            break

        cursor = oldest_ts - 1
        await asyncio.sleep(0.3)

    logger.debug(f"Fetched {len(records)} raw liq orders for {inst_family}")
    return records


# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate(records: list[dict], interval: str) -> pd.DataFrame:
    """Aggregate raw order list into OHLCV-style time buckets.

    USD value per order = sz (contracts) * bkPx (USDT price).
    For OKX USDT-margined perps: 1 contract = 1 base token (ETH, BTC, etc.).
    """
    if not records:
        return pd.DataFrame(columns=["long_liq_usd", "short_liq_usd", "total_liq_usd"])

    rows = []
    for det in records:
        ts = pd.Timestamp(int(det["ts"]), unit="ms", tz="UTC")
        sz = float(det.get("sz", 0))
        bk_px = float(det.get("bkPx", 0))
        usd_val = sz * bk_px
        side = det.get("posSide", "")  # "long" or "short"
        rows.append({"ts": ts, "side": side, "usd": usd_val})

    df_raw = pd.DataFrame(rows).set_index("ts")

    freq = f"{_INTERVAL_MINUTES[interval]}min"
    long_df = (
        df_raw[df_raw["side"] == "long"]["usd"]
        .resample(freq, label="left", closed="left")
        .sum()
        .rename("long_liq_usd")
    )
    short_df = (
        df_raw[df_raw["side"] == "short"]["usd"]
        .resample(freq, label="left", closed="left")
        .sum()
        .rename("short_liq_usd")
    )

    df = pd.concat([long_df, short_df], axis=1).fillna(0)
    df["total_liq_usd"] = df["long_liq_usd"] + df["short_liq_usd"]
    return df.sort_index()


# ── Downloader ────────────────────────────────────────────────────────────────

class OKXLiquidationDownloader:
    """Incrementally downloads OKX liquidation orders and saves as aggregated Parquet.

    OKX public API has ~24h of order-level history available.
    Run this script daily (or every 12h with overlap) via cron to build up history.
    """

    async def fetch_and_save(
        self,
        symbol: str,             # e.g. "ETH-USDT"
        hours: int = 25,         # lookback window (use >24 to avoid gaps)
        interval: str = "1h",    # aggregation bucket: 1h | 4h | 8h | 12h | 1d
    ) -> int:
        """Fetch recent liquidation orders and append to parquet. Returns new row count."""
        if interval not in SUPPORTED_INTERVALS:
            raise ValueError(f"Unsupported interval '{interval}'. Choose from: {SUPPORTED_INTERVALS}")

        now = datetime.now(tz=timezone.utc)
        since = now - timedelta(hours=hours)
        since_ms = int(since.timestamp() * 1000)
        end_ms = int(now.timestamp() * 1000)

        logger.info(f"Fetching OKX liquidations: {symbol} [{interval}] last {hours}h")

        async with httpx.AsyncClient(timeout=20) as client:
            records = await _fetch_okx_liq_orders(client, symbol, since_ms, end_ms)

        df = _aggregate(records, interval)
        if df.empty:
            logger.warning(f"No liquidation data for {symbol}")
            return 0

        _save_liquidation(symbol, interval, df)
        return len(df)

    async def fetch_multi(
        self,
        symbols: list[str],
        hours: int = 25,
        interval: str = "1h",
    ) -> dict[str, int]:
        """Fetch and save for multiple symbols. Returns {symbol: new_rows}."""
        results: dict[str, int] = {}
        for sym in symbols:
            count = await self.fetch_and_save(sym, hours, interval)
            results[sym] = count
        return results
