"""ParquetDataCatalog wrapper with convenient helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from nautilus_trader.persistence.catalog import ParquetDataCatalog

from cyber_trader.config import get_settings


@lru_cache(maxsize=1)
def get_catalog() -> ParquetDataCatalog:
    settings = get_settings()
    settings.ensure_catalog_dir()
    return ParquetDataCatalog(str(settings.data_catalog_path))
