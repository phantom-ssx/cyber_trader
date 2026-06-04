# CyberTrader

基于 [nautilus_trader](https://github.com/nautechsystems/nautilus_trader) 的 BTC/ETH 合约量化交易系统。

## 功能

| 功能 | 说明                                            |
|------|-----------------------------------------------|
| 数据下载 | 通过 ccxt 从 OKX 下载历史 OHLCV，写入本地 Parquet catalog |
| 回测 | nautilus_trader BacktestNode，支持滑点/延迟模拟        |
| 模拟盘 | 连接 OKX demo 环境，真实行情 + 虚拟撮合                    |
| 实盘 | 连接 OKX 主网，需 `--confirm` 二次确认                  |
| Bark 通知 | 入场信号 + 量化指标 + 价格异动推送到 iOS Bark App          |
| 价格监控 | 实时监控行情波动（单K线/累计/方向/整数关口），只读不下单           |
| 多因子 | EMA 交叉 / MACD / RSI / 布林带 / 动量 / 量价动量         |
| 风控 | 仓位百分比、止损/止盈、最大回撤、日亏损限额                        |
| K 线可视化 | mplfinance 绘制 K 线 + EMA + MACD，输出高清 PNG       |

## 快速开始

### 1. 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# K 线可视化（可选）
pip install mplfinance
```

### 2. 配置

```bash
cp .env.example .env
# 填写 OKX API Key / Secret / Passphrase
# 填写 Bark Device Key（可选，用于 iOS 推送通知）
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
    --start 2025-01-01 --end 2026-06-01 \
    --timeframe 1m \
    --balance 50000

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

### 7. K 线可视化

```bash
# 绘制最近 200 根 K 线（主图：K线+EMA，副图：成交量+MACD），输出 PNG
python scripts/plot_kline.py

# 自定义 K 线数量和输出路径
python scripts/plot_kline.py --bars 500 --out charts/eth.png

# 指定 bar type（默认 1 分钟）
python scripts/plot_kline.py --bar-type "ETH-USDT-SWAP.OKX-1-HOUR-LAST-EXTERNAL" --bars 200
```

### 8. 价格监控

实时监控行情，当价格异动时通过 Bark 推送到 iOS。监控器只订阅行情数据，**不下任何订单**。

```bash
# 连接 OKX Demo 环境（默认）
python scripts/run_monitor.py --config config/monitor.yaml

# 连接 OKX 主网实时行情
python scripts/run_monitor.py --config config/monitor.yaml --live
```

**监控规则类型（`config/monitor.yaml` 中 `rules_json` 配置）：**

| 类型 | 说明 | 关键参数 |
|------|------|----------|
| `single_bar` | 单根 K 线涨跌幅 ≥ 阈值 | `threshold_pct` |
| `cumulative` | N 根 K 线累计波动幅度之和 ≥ 阈值 | `threshold_pct`, `bars` |
| `directional` | 连续 N 根 K 线同向运动总幅度 ≥ 阈值 | `threshold_pct`, `bars` |
| `round_level` | 价格穿越整数关口（如 ETH 每 $100、BTC 每 $1000）| `interval` |

**示例：自定义规则**

```yaml
# config/monitor.yaml
strategy:
  params:
    cooldown_bars: 10          # 任意规则触发后，同品种冷却 N 根 K 线
    rules_json: >-
      [
        {"type": "single_bar",  "threshold_pct": 0.5},
        {"type": "cumulative",  "threshold_pct": 1.0, "bars": 5},
        {"type": "directional", "threshold_pct": 0.8, "bars": 3}
      ]
    per_instrument_rules_json: >-
      {
        "ETH-USDT-SWAP.OKX": [
          {"type": "round_level", "interval": 100}
        ],
        "BTC-USDT-SWAP.OKX": [
          {"type": "round_level", "interval": 1000}
        ]
      }
```

> `per_instrument_rules_json` 中列出的品种使用**独立规则集**（完全替换默认规则），未列出的品种使用 `rules_json` 默认规则。

**新增自定义规则**

继承 `AlertRule`，实现 `window_size` 和 `evaluate()`，用 `@register_rule` 注册：

```python
from collections import deque
from cyber_trader.monitors.rules import AlertRule, AlertResult, register_rule

@register_rule("my_rule")
class MyRule(AlertRule):
    def __init__(self, threshold_pct: float = 1.0) -> None:
        self.threshold_pct = threshold_pct

    @property
    def window_size(self) -> int:
        return 2

    def evaluate(self, closes: deque[float]) -> AlertResult | None:
        prev, curr = list(closes)[-2], list(closes)[-1]
        pct = abs(curr - prev) / prev * 100
        if pct >= self.threshold_pct:
            return AlertResult(
                label="自定义异动",
                pct=pct,
                from_price=prev,
                to_price=curr,
                window_bars=1,
                rising=curr > prev,
            )
        return None
```

### 9. Jupyter 分析

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
│   ├── notifications/  # Bark 推送通知
│   ├── monitors/       # 价格监控（VolatilityMonitor + 四类告警规则）
│   └── engines/        # 回测 / 模拟盘 / 实盘 引擎
├── scripts/            # CLI 入口（download / backtest / paper / live / plot_kline / run_monitor）
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

## 回测绩效指标说明

回测结束后输出的绩效指标含义如下：

| 指标 | 含义 |
|------|------|
| `total_return_pct` | 整个回测期间的总收益率（%），负值代表亏损 |
| `total_pnl` | 总盈亏金额（USDT），正为盈利，负为亏损 |
| `sharpe_ratio` | 夏普比率：风险调整后收益。>1 合格，>2 优秀，负值说明还不如持有无风险资产 |
| `sortino_ratio` | 索提诺比率：只惩罚下行波动的夏普变体，对亏损波动更敏感，通常比夏普比率更严苛 |
| `profit_factor` | 总盈利 / 总亏损。>1 才能长期盈利，=0.97 意味着每亏 1 元只赚回 0.97 元 |
| `win_rate` | 盈利交易占总交易次数的比例。需结合盈亏比一起看，胜率低但盈亏比高也可以盈利 |
| `avg_winner` | 盈利交易的平均每笔收益（USDT） |
| `avg_loser` | 亏损交易的平均每笔亏损（USDT，通常为负数） |
| `expectancy` | 每笔交易的期望收益 = `win_rate × avg_winner + (1 - win_rate) × avg_loser`，正值才代表策略长期可盈利 |
| `total_orders` | 回测期间总下单次数 |

**保本胜率公式**（盈亏比固定时的最低胜率要求）：

```
保本胜率 = |avg_loser| / (avg_winner + |avg_loser|)
```

例如 `avg_winner=18.56`，`avg_loser=-9.01`，则保本胜率 ≈ **32.6%**，实际胜率需高于此值策略才能盈利。

## 运行测试

```bash
pytest tests/ -v
```
