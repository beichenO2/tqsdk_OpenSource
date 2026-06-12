# tqsdk — 部署指南

> 国内期货 + BTC 双市场量化交易平台

## 环境要求

- 技术栈：Python, TqSdk (期货), Binance API (加密)
- 安装：`cd data-collector && pip install -r requirements.txt`

## 安装步骤

```bash
cd ~/Polarisor/tqsdk
cd data-collector && pip install -r requirements.txt
```

## 启动方式

```bash
cd ~/Polarisor/tqsdk
cd data-collector && python collector.py
```

## 端口分配

| 端口 | 用途 |
|---|---|
| 18900 | 主服务 |

## 健康检查确认

```bash
curl -s http://127.0.0.1:18900/health
```

## 回滚方式

```bash
cd ~/Polarisor/tqsdk
git log --oneline -5
git checkout <previous-commit>
cd data-collector && pip install -r requirements.txt
cd data-collector && python collector.py
```
