"""Download and store auxiliary market data (funding rate, long/short ratio) from OKX.

Storage layout: data/aux/{SYMBOL}_{type}.parquet
  - SYMBOL: e.g. ETH-USDT (without -SWAP suffix)
  - type:   funding_rate | long_short_ratio
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import ccxt.async_support as ccxt
import httpx
import pandas as pd
from loguru import logger

from cyber_trader.config import get_settings


# ── Storage helpers ───────────────────────────────────────────────────────────

def _aux_dir() -> Path:
    settings = get_settings()
    p = settings.data_catalog_path.parent / "aux"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _aux_path(symbol: str, data_type: str) -> Path:
    safe = symbol.replace("/", "_")
    return _aux_dir() / f"{safe}_{data_type}.parquet"


def load_series(symbol: str, data_type: str) -> pd.Series | None:
    """Load a stored auxiliary series. Returns None if file doesn't exist."""
    path = _aux_path(symbol, data_type)
    if not path.exists():
        logger.warning(f"Aux data not found: {path}  (run download_aux_data.py first)")
        return None
    df = pd.read_parquet(path)
    col = df.columns[0]
    s = df[col].copy()
    s.index = pd.to_datetime(df.index, utc=True)
    return s.sort_index()


def save_series(symbol: str, data_type: str, series: pd.Series) -> Path:
    """Persist a series to parquet, merging with existing data if present."""
    path = _aux_path(symbol, data_type)
    if path.exists():
        existing = load_series(symbol, data_type)
        if existing is not None:
            series = pd.concat([existing, series]).drop_duplicates().sort_index()
    series.to_frame().to_parquet(path)
    logger.info(f"Saved {len(series)} rows → {path}")
    return path


# ── Downloader ────────────────────────────────────────────────────────────────

class OKXAuxDownloader:
    """Downloads funding-rate history and long/short ratio from OKX."""

    OKX_BASE = "https://www.okx.com"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
    ) -> None:
        self._exchange = ccxt.okx({
            "apiKey": api_key,
            "secret": api_secret,
            "password": passphrase,
            "options": {"defaultType": "swap"},
        })

    # ── Funding Rate ──────────────────────────────────────────────────────────

    async def fetch_funding_rate_history(
        self,
        symbol: str,          # e.g. "ETH-USDT"
        start: str | datetime,
        end: str | datetime | None = None,
    ) -> pd.Series:
        """Fetch 8h funding rate history via ccxt.

        Returns a pd.Series indexed by UTC timestamps, named 'funding_rate'.
        """
        base, quote = symbol.split("-")
        ccxt_symbol = f"{base}/{quote}:{quote}"   # ETH/USDT:USDT

        if isinstance(start, str):
            start = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        if end is None:
            end = datetime.now(tz=timezone.utc)
        elif isinstance(end, str):
            end = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        records: list[tuple[int, float]] = []
        cursor = since_ms

        logger.info(f"Fetching funding rate: {symbol} {start.date()} → {end.date()}")

        while cursor < end_ms:
            try:
                batch = await self._exchange.fetch_funding_rate_history(
                    symbol=ccxt_symbol, since=cursor, limit=100
                )
            except Exception as exc:
                logger.error(f"Funding rate fetch error: {exc}")
                await asyncio.sleep(2)
                break

            if not batch:
                break

            for item in batch:
                ts_ms: int = item["timestamp"]
                if ts_ms <= end_ms:
                    records.append((ts_ms, float(item["fundingRate"])))

            new_cursor = batch[-1]["timestamp"] + 1
            if new_cursor <= cursor:
                break
            cursor = new_cursor
            await asyncio.sleep(0.3)

        if not records:
            logger.warning(f"No funding rate data returned for {symbol}")
            return pd.Series(dtype=float, name="funding_rate")

        ts_idx, vals = zip(*records)
        s = pd.Series(
            list(vals),
            index=pd.DatetimeIndex(
                [pd.Timestamp(t, unit="ms", tz="UTC") for t in ts_idx]
            ),
            name="funding_rate",
        )
        return s.sort_index().drop_duplicates()

    # ── Long/Short Ratio ──────────────────────────────────────────────────────

    async def fetch_long_short_ratio(
        self,
        symbol: str,            # e.g. "ETH-USDT"
        period: str = "1H",     # 5m | 1H | 4H | 1D
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> pd.Series:
        """Fetch account-level long/short ratio from OKX REST API.

        Endpoint: /api/v5/rubik/stat/contracts/long-short-account-ratio
        Uses ccy (base currency, e.g. "ETH") — not instId.

        Returns a pd.Series indexed by UTC timestamps, named 'long_short_ratio'.
        Ratio > 1 means more accounts are long; < 1 means more are short.
        """
        if isinstance(start, str):
            start = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        if isinstance(end, str):
            end = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

        end_dt = end or datetime.now(tz=timezone.utc)
        start_dt = start or datetime(2021, 1, 1, tzinfo=timezone.utc)

        end_ms = int(end_dt.timestamp() * 1000)
        start_ms = int(start_dt.timestamp() * 1000)

        # Extract base currency: "ETH-USDT" → "ETH"
        ccy = symbol.split("-")[0]

        url = f"{self.OKX_BASE}/api/v5/rubik/stat/contracts/long-short-account-ratio"
        records: list[tuple[int, float]] = []
        cursor_end = end_ms

        logger.info(f"Fetching L/S ratio: {symbol} ({ccy}) {start_dt.date()} → {end_dt.date()}")

        async with httpx.AsyncClient(timeout=15) as client:
            while True:
                params = {
                    "ccy": ccy,          # base currency, e.g. "ETH"
                    "period": period,
                    "end": str(cursor_end),
                    "limit": "100",
                }
                try:
                    resp = await client.get(url, params=params)
                    resp.raise_for_status()
                    payload = resp.json()
                except Exception as exc:
                    logger.error(f"L/S ratio fetch error: {exc}")
                    break

                if payload.get("code") != "0" or not payload.get("data"):
                    logger.warning(f"L/S ratio API response: {payload.get('msg', 'no data')}")
                    break

                rows = payload["data"]   # [[ts_ms_str, ratio_str], ...], newest first
                batch_records: list[tuple[int, float]] = []
                for row in rows:
                    ts_ms = int(row[0])
                    ratio = float(row[1])
                    if ts_ms >= start_ms:
                        batch_records.append((ts_ms, ratio))

                records.extend(batch_records)

                oldest_ts = int(rows[-1][0])
                if oldest_ts <= start_ms or len(rows) < 100:
                    break

                cursor_end = oldest_ts  # exclusive: next batch ends before this
                await asyncio.sleep(0.3)

        if not records:
            logger.warning(f"No L/S ratio data returned for {symbol}")
            return pd.Series(dtype=float, name="long_short_ratio")

        ts_idx, vals = zip(*records)
        s = pd.Series(
            list(vals),
            index=pd.DatetimeIndex(
                [pd.Timestamp(t, unit="ms", tz="UTC") for t in ts_idx]
            ),
            name="long_short_ratio",
        )
        return s.sort_index().drop_duplicates()

    async def close(self) -> None:
        await self._exchange.close()
