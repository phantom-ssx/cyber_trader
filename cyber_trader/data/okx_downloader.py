"""Download historical OHLCV data from OKX via ccxt and write to ParquetDataCatalog."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal

import ccxt.async_support as ccxt
import pandas as pd
from loguru import logger
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import AggregationSource, BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual, CurrencyPair
from nautilus_trader.model.objects import Currency, Price, Quantity

from .catalog import get_catalog

VENUE = Venue("OKX")

# OKX timeframe strings  →  (nautilus BarAggregation, step)
_TIMEFRAME_MAP: dict[str, tuple[BarAggregation, int]] = {
    "1m":  (BarAggregation.MINUTE, 1),
    "3m":  (BarAggregation.MINUTE, 3),
    "5m":  (BarAggregation.MINUTE, 5),
    "15m": (BarAggregation.MINUTE, 15),
    "30m": (BarAggregation.MINUTE, 30),
    "1h":  (BarAggregation.HOUR, 1),
    "2h":  (BarAggregation.HOUR, 2),
    "4h":  (BarAggregation.HOUR, 4),
    "6h":  (BarAggregation.HOUR, 6),
    "12h": (BarAggregation.HOUR, 12),
    "1d":  (BarAggregation.DAY, 1),
}

SUPPORTED_TIMEFRAMES = list(_TIMEFRAME_MAP.keys())

# Reverse lookup: (BarAggregation, step) → OKX timeframe string
_REVERSE_TF_MAP: dict[tuple[BarAggregation, int], str] = {v: k for k, v in _TIMEFRAME_MAP.items()}

# Duration of each timeframe in milliseconds
_TIMEFRAME_DURATION_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def timeframe_to_bar_type(instrument_id: str, timeframe: str) -> str:
    """Convert OKX timeframe string to a nautilus bar type string.

    e.g. timeframe_to_bar_type("ETH-USDT-SWAP.OKX", "1h")
         -> "ETH-USDT-SWAP.OKX-1-HOUR-LAST-EXTERNAL"
    """
    if timeframe not in _TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Choose from: {SUPPORTED_TIMEFRAMES}")
    aggregation, step = _TIMEFRAME_MAP[timeframe]
    return f"{instrument_id}-{step}-{aggregation.name}-LAST-EXTERNAL"

# Instrument metadata defaults for common pairs
_INSTRUMENT_META: dict[str, dict] = {
    "ETH": {"price_precision": 2, "size_precision": 4, "tick_size": "0.01", "lot_size": "0.0001"},
    "BTC": {"price_precision": 1, "size_precision": 5, "tick_size": "0.1",  "lot_size": "0.00001"},
    "SOL": {"price_precision": 3, "size_precision": 2, "tick_size": "0.001", "lot_size": "0.01"},
    "DEFAULT": {"price_precision": 4, "size_precision": 2, "tick_size": "0.0001", "lot_size": "0.01"},
}


def _make_currency(code: str) -> Currency:
    try:
        return Currency.from_str(code)
    except Exception:
        return Currency(code, precision=8, iso4217=0, name=code, currency_type=2)


def _make_instrument(
    symbol: str,
    market_type: Literal["swap", "spot"],
) -> CryptoPerpetual | CurrencyPair:
    """Build a nautilus instrument from a symbol like 'ETH-USDT'."""
    base_code, quote_code = symbol.split("-")
    meta = _INSTRUMENT_META.get(base_code, _INSTRUMENT_META["DEFAULT"])

    base = _make_currency(base_code)
    quote = _make_currency(quote_code)
    settle = _make_currency(quote_code)

    if market_type == "swap":
        native = f"{symbol}-SWAP"
        instrument_id = InstrumentId(Symbol(native), VENUE)
        return CryptoPerpetual(
            instrument_id=instrument_id,
            raw_symbol=Symbol(native),
            base_currency=base,
            quote_currency=quote,
            settlement_currency=settle,
            is_inverse=False,
            price_precision=meta["price_precision"],
            size_precision=meta["size_precision"],
            price_increment=Price.from_str(meta["tick_size"]),
            size_increment=Quantity.from_str(meta["lot_size"]),
            multiplier=Quantity.from_str("1"),
            lot_size=Quantity.from_str(meta["lot_size"]),
            max_quantity=Quantity.from_str("10000"),
            min_quantity=Quantity.from_str(meta["lot_size"]),
            max_notional=None,
            min_notional=None,
            max_price=None,
            min_price=None,
            margin_init=Decimal("0.05"),
            margin_maint=Decimal("0.025"),
            maker_fee=Decimal("0.0002"),
            taker_fee=Decimal("0.0005"),
            ts_event=0,
            ts_init=0,
        )
    else:
        instrument_id = InstrumentId(Symbol(symbol), VENUE)
        return CurrencyPair(
            instrument_id=instrument_id,
            raw_symbol=Symbol(symbol),
            base_currency=base,
            quote_currency=quote,
            price_precision=meta["price_precision"],
            size_precision=meta["size_precision"],
            price_increment=Price.from_str(meta["tick_size"]),
            size_increment=Quantity.from_str(meta["lot_size"]),
            lot_size=Quantity.from_str(meta["lot_size"]),
            max_quantity=Quantity.from_str("10000"),
            min_quantity=Quantity.from_str(meta["lot_size"]),
            max_notional=None,
            min_notional=None,
            max_price=None,
            min_price=None,
            margin_init=Decimal("0.05"),
            margin_maint=Decimal("0.025"),
            maker_fee=Decimal("0.0002"),
            taker_fee=Decimal("0.0005"),
            ts_event=0,
            ts_init=0,
        )


def _ohlcv_to_bars(
    ohlcv: list[list],
    bar_type: BarType,
    price_precision: int,
    size_precision: int,
) -> list[Bar]:
    bars: list[Bar] = []
    for row in ohlcv:
        ts_ms, open_, high, low, close, volume = row[:6]
        ts_ns = int(ts_ms) * 1_000_000
        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price(open_, price_precision),
                high=Price(high, price_precision),
                low=Price(low, price_precision),
                close=Price(close, price_precision),
                volume=Quantity(volume, size_precision),
                ts_event=ts_ns,
                ts_init=ts_ns,
            )
        )
    return bars


def fetch_recent_bars_sync(
    instrument_id_str: str,
    bar_type_str: str,
    count: int,
    api_key: str = "",
    api_secret: str = "",
    passphrase: str = "",
    is_demo: bool = False,
) -> list[Bar]:
    """Synchronously fetch the most recent `count` bars for indicator warm-up.

    Does NOT write to the catalog — intended for cold-start initialization only.
    Uses the ccxt synchronous API so it can be called from a non-async context.
    """
    import ccxt as _ccxt_sync  # sync version, separate from ccxt.async_support

    bar_type = BarType.from_str(bar_type_str)
    key = (bar_type.spec.aggregation, bar_type.spec.step)
    timeframe = _REVERSE_TF_MAP.get(key)
    if timeframe is None:
        raise ValueError(f"Cannot map bar type '{bar_type_str}' to a supported OKX timeframe")

    # Parse nautilus instrument_id → ccxt symbol + market type
    name = instrument_id_str.split(".")[0]  # strip venue suffix ".OKX"
    if name.endswith("-SWAP"):
        base_sym = name[:-5]  # e.g. "ETH-USDT"
        ccxt_symbol = base_sym.replace("-", "/") + ":USDT"
        market_type: Literal["swap", "spot"] = "swap"
    else:
        base_sym = name
        ccxt_symbol = name.replace("-", "/")
        market_type = "spot"

    options: dict = {"defaultType": market_type}
    if is_demo:
        options["sandboxMode"] = True

    exchange = _ccxt_sync.okx({
        "apiKey": api_key,
        "secret": api_secret,
        "password": passphrase,
        "options": options,
    })

    meta = _INSTRUMENT_META.get(base_sym.split("-")[0], _INSTRUMENT_META["DEFAULT"])
    bar_duration_ms = _TIMEFRAME_DURATION_MS[timeframe]

    now_ms = int(time.time() * 1000)
    since_ms = now_ms - (count + 5) * bar_duration_ms  # +5 buffer for incomplete candle
    end_ms = now_ms

    all_bars: list[Bar] = []
    cursor = since_ms

    while cursor < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(
                ccxt_symbol,
                timeframe=timeframe,
                since=cursor,
                limit=300,
            )
        except Exception as exc:
            logger.warning(f"Warmup fetch error at cursor={cursor}: {exc}")
            break

        if not ohlcv:
            break

        batch = _ohlcv_to_bars(ohlcv, bar_type, meta["price_precision"], meta["size_precision"])
        all_bars.extend(batch)
        cursor = ohlcv[-1][0] + 1

        if len(ohlcv) < 300:
            break

    try:
        exchange.close()
    except Exception:
        pass

    logger.info(f"Fetched {len(all_bars)} warmup bars for {instrument_id_str} [{timeframe}]")
    return all_bars[-count:] if len(all_bars) > count else all_bars


class OKXDownloader:
    """Download and persist historical bars from OKX."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        is_demo: bool = False,
    ) -> None:
        options: dict = {"defaultType": "swap"}
        if is_demo:
            options["sandboxMode"] = True

        self._exchange = ccxt.okx(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "password": passphrase,
                "options": options,
            }
        )
        self._catalog = get_catalog()

    async def download(
        self,
        symbol: str,
        timeframe: str,
        start: str | datetime,
        end: str | datetime | None = None,
        market_type: Literal["swap", "spot"] = "swap",
        batch_size: int = 300,
    ) -> int:
        """Download bars and write to catalog. Returns number of bars written."""
        if timeframe not in _TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Choose from {list(_TIMEFRAME_MAP)}")

        aggregation, step = _TIMEFRAME_MAP[timeframe]
        instrument = _make_instrument(symbol, market_type)
        meta = _INSTRUMENT_META.get(symbol.split("-")[0], _INSTRUMENT_META["DEFAULT"])

        bar_type = BarType(
            instrument.id,
            BarSpecification(step, aggregation, PriceType.LAST),
            AggregationSource.EXTERNAL,
        )

        # ccxt symbol format: ETH/USDT:USDT for perpetual swap
        ccxt_symbol = (
            f"{symbol.replace('-', '/')}" + (":USDT" if market_type == "swap" else "")
        )

        if isinstance(start, str):
            start = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        if end is None:
            end = datetime.now(tz=timezone.utc)
        elif isinstance(end, str):
            end = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        logger.info(f"Downloading {symbol} {timeframe} from {start.date()} to {end.date()}")

        self._catalog.write_data([instrument])

        all_bars: list[Bar] = []
        cursor = since_ms

        while cursor < end_ms:
            try:
                ohlcv = await self._exchange.fetch_ohlcv(
                    ccxt_symbol,
                    timeframe=timeframe,
                    since=cursor,
                    limit=batch_size,
                )
            except Exception as e:
                logger.error(f"Fetch error at {cursor}: {e}")
                await asyncio.sleep(2)
                continue

            if not ohlcv:
                break

            batch = _ohlcv_to_bars(
                ohlcv,
                bar_type,
                meta["price_precision"],
                meta["size_precision"],
            )
            all_bars.extend(batch)
            cursor = ohlcv[-1][0] + 1
            logger.debug(f"  fetched {len(batch)} bars, last ts={ohlcv[-1][0]}")
            await asyncio.sleep(0.2)

        if all_bars:
            self._catalog.write_data(all_bars)
            logger.info(f"Wrote {len(all_bars)} bars to catalog for {symbol} {timeframe}")

        await self._exchange.close()
        return len(all_bars)

    async def download_multi(
        self,
        symbols: list[str],
        timeframes: list[str],
        start: str | datetime,
        end: str | datetime | None = None,
        market_type: Literal["swap", "spot"] = "swap",
    ) -> dict[str, int]:
        """Download multiple symbols/timeframes. Returns {symbol_tf: count}."""
        results: dict[str, int] = {}
        for symbol in symbols:
            for tf in timeframes:
                count = await self.download(symbol, tf, start, end, market_type)
                results[f"{symbol}_{tf}"] = count
        return results
