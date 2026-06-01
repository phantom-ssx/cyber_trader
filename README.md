# CyberTrader

基于 [nautilus_trader](https://github.com/nautechsystems/nautilus_trader) 的 BTC/ETH 合约量化交易系统。

## 功能

| 功能 | 说明 |
|------|------|
| 数据下载 | 通过 ccxt 从 OKX 下载历史 OHLCV，写入本地 Parquet catalog |
| 回测 | nautilus_trader BacktestNode，支持滑点/延迟模拟 |
| 模拟盘 | 连接 OKX demo 环境，真实行情 + 虚拟撮合 |
| 实盘 | 连接 OKX 主网，需 `--confirm` 二次确认 |
| 飞书通知 | 入场信号 + 绩效指标推送到飞书群机器人 |
| 多因子 | EMA 交叉 / MACD / RSI / 布林带 / 动量 / 量价动量 |
| 风控 | 仓位百分比、止损/止盈、最大回撤、日亏损限额 |

## 快速开始

### 1. 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. 配置

```bash
cp .env.example .env
# 填写 OKX API Key / Secret / Passphrase
# 填写飞书 Webhook URL（可选）
```

### 3. 下载历史数据

```bash
# 下载 ETH-USDT 永续合约 1h K线（2024全年）
python scripts/download_data.py \
  --symbol ETH-USDT \
  --timeframe 1h \
  --start 2024-01-01

# 同时下载多品种多周期
python scripts/download_data.py \
  --symbol ETH-USDT,BTC-USDT \
  --timeframe 1h,4h \
  --start 2024-01-01
```

### 4. 回测

```bash
# 使用默认配置（趋势跟踪策略）
python scripts/run_backtest.py --config config/backtest.yaml

# 覆盖时间范围
python scripts/run_backtest.py --config config/backtest.yaml \
  --start 2024-06-01 --end 2024-12-31

# 切换策略：编辑 config/backtest.yaml 中的 strategy 部分
```

### 5. 模拟盘交易

```bash
# 本地运行（需先配置 OKX Demo API Key）
python scripts/run_paper.py --config config/paper_trading.yaml

# Docker 运行
docker compose up paper-trader
```

### 6. 实盘交易

```bash
# ⚠️ 真实资金，请确认策略经过充分回测和模拟盘验证
python scripts/run_live.py --config config/live_trading.yaml --confirm

# Docker 运行
docker compose --profile live up live-trader
```

### 7. Jupyter 分析

```bash
jupyter notebook notebooks/backtest_analysis.ipynb
# 或 Docker：
docker compose --profile tools up jupyter
```

## 项目结构

```
cyber_trader/
├── cyber_trader/
│   ├── config/         # Pydantic settings（环境变量）
│   ├── data/           # OKX 数据下载 & ParquetDataCatalog
│   ├── indicators/     # 多因子引擎（6个内置因子）
│   ├── strategies/     # 三类策略 + 抽象基类
│   ├── risk/           # 风控管理（仓位/止损/回撤）
│   ├── notifications/  # 飞书 Webhook
│   └── engines/        # 回测 / 模拟盘 / 实盘 引擎
├── scripts/            # CLI 入口
├── config/             # YAML 配置文件
├── notebooks/          # Jupyter 分析
├── tests/              # 单元测试
├── Dockerfile
└── docker-compose.yml
```

## 新增策略

继承 `BaseStrategy`，只需实现 `build_factor_engine()`：

```python
from cyber_trader.strategies.base import BaseStrategy, BaseStrategyConfig
from cyber_trader.indicators.factor_engine import FactorEngine, EMACrossoverFactor

class MyConfig(BaseStrategyConfig, frozen=True):
    fast: int = 5
    slow: int = 20

class MyStrategy(BaseStrategy):
    def __init__(self, config: MyConfig):
        super().__init__(config)
        self._my_config = config

    def build_factor_engine(self) -> FactorEngine:
        return FactorEngine(
            factors=[EMACrossoverFactor(self._my_config.fast, self._my_config.slow)],
            long_threshold=0.3,
            short_threshold=-0.3,
        )
```

## 新增因子

继承 `Factor`，实现 `update()` 和 `score()` 即可自动集成到引擎：

```python
from cyber_trader.indicators.factor_engine import Factor
from nautilus_trader.model.data import Bar

class MyFactor(Factor):
    def __init__(self, period: int = 14, weight: float = 1.0):
        super().__init__(f"MY_FACTOR({period})", weight)
        self._period = period
        self._values: list[float] = []

    @property
    def is_initialized(self) -> bool:
        return len(self._values) >= self._period

    def update(self, bar: Bar) -> None:
        self._values.append(bar.close.as_double())
        if len(self._values) > self._period:
            self._values.pop(0)

    def score(self) -> float:
        if not self.is_initialized:
            return 0.0
        # 返回 [-1, +1] 的信号值
        return 0.0
```

## 新增交易对

在 `cyber_trader/data/okx_downloader.py` 的 `_INSTRUMENT_META` 字典中添加精度配置：

```python
_INSTRUMENT_META = {
    "ETH": {...},
    "BTC": {...},
    "DOGE": {"price_precision": 6, "size_precision": 0, "tick_size": "0.000001", "lot_size": "1"},
}
```

## 云服务器部署

```bash
# 1. git clone 到服务器
# 2. 配置 .env 文件
# 3. 启动模拟盘
docker compose up -d paper-trader

# 查看日志
docker compose logs -f paper-trader

# 启动实盘（需 --profile live）
docker compose --profile live up -d live-trader
```

## 运行测试

```bash
pytest tests/ -v
```
