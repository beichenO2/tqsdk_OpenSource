# tqsdk — 使用指南

> 国内期货 + BTC 双市场量化交易平台

## 核心信息

| 维度 | 值 |
|---|---|
| 健康端点 | 端口 18900（/health） |
| 启动命令 | `cd data-collector && python collector.py` |
| 安装命令 | `cd data-collector && pip install -r requirements.txt` |
| 技术栈 | Python, TqSdk (期货), Binance API (加密) |

## 快速启动

```bash
cd ~/Polarisor/tqsdk
cd data-collector && pip install -r requirements.txt
cd data-collector && python collector.py
```

## 健康检查

```bash
curl -s http://127.0.0.1:18900/health
```

## 依赖服务

- 天勤 TqSdk (期货行情)
- Binance API (加密行情)
