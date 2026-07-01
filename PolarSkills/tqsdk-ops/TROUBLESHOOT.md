# tqsdk — 故障排查

> 国内期货 + BTC 双市场量化交易平台

## 健康检查

```bash
# 进程存活
pgrep -f "tqsdk" || echo "NOT RUNNING"

# HTTP 端点
curl -s http://127.0.0.1:18900/health
```

## 关键端口

| 端口 | 说明 |
|---|---|
| 18900 | tqsdk 主服务 |

## 常见故障

### 1. 行情断连

**修复**：`检查 TqSdk 连接状态和网络`

### 2. WEEX 签名失败

**修复**：`确认 PolarPrivate 已解锁，且 Vault 中 exchange.weex 绑定已配置`

### 3. 回测数据缺失

**修复**：`确认数据目录和日期范围`

## 依赖服务

- 天勤 TqSdk gateway（期货行情/下单）
- PolarPrivate B-class sign（WEEX 加密实盘）

## 紧急恢复

```bash
cd ~/Polarisor/tqsdk
cd data-collector && python collector.py
curl -s http://127.0.0.1:18900/health && echo 'OK' || echo 'BROKEN'
```
