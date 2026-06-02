from nautilus_trader.persistence.catalog import ParquetDataCatalog
import pandas as pd

class TestData:
    def test_read_parquet(self):
        path = "data/catalog/data/bar/ETH-USDT-SWAP.OKX-1-MINUTE-LAST-EXTERNAL/2026-05-25T00-00-00-000000000Z_2026-05-30T04-59-00-000000000Z.parquet"
        df = pd.read_parquet(path)
        print(df.head())
        print(df.dtypes)



    def test_read_bar(self):
        catalog = ParquetDataCatalog("data/catalog")
        bars = catalog.bars(
            bar_types=["ETH-USDT-SWAP.OKX-1-MINUTE-LAST-EXTERNAL"],
        )
        # 转成 DataFrame
        df = pd.DataFrame([{
            "open": b.open.as_double(),
            "high": b.high.as_double(),
            "low": b.low.as_double(),
            "close": b.close.as_double(),
            "volume": b.volume.as_double(),
            "ts": pd.Timestamp(b.ts_event, unit="ns", tz="UTC"),
        } for b in bars])
        print()
        print(df.head())
        #或者更简洁地用内置转换：
        #df = pd.DataFrame([b.to_dict() for b in bars])
        #print(df.head())




