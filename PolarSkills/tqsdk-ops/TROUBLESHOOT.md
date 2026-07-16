# tqsdk troubleshooting

## Read-only diagnosis

```bash
curl -fsS http://127.0.0.1:11050/api/health
curl -fsS http://127.0.0.1:11055/api/health
curl -fsS http://127.0.0.1:11055/api/services/tqsdk-data-collector
curl -fsS http://127.0.0.1:11055/api/services/tqsdk-gateway
curl -sS http://127.0.0.1:18900/health
curl -sS http://127.0.0.1:12890/health
```

Use PolarProcess's verified PID and PolarPort's service/project owner as the facts. Do not infer ownership from broad process-name matches.

## Collector waits for gateway

This is expected when `tqsdk-gateway` is intentionally stopped. The collector exposes health first, retries the gateway, reports the dependency failure, exits, and is recovered by PolarProcess according to its restart policy. Do not bypass the gateway and do not move credentials into the collector.

## Exact recovery

After confirming both authorities are healthy, restart only the affected service:

```bash
curl -fsS -X POST http://127.0.0.1:11055/api/services/tqsdk-data-collector/restart
```

If a trading operation explicitly requires the gateway, start only that service ID and then verify `connected` from its health response.

## Forbidden recovery shortcuts

Do not use background shell jobs, PID files, direct signals, launchd, raw Python/Uvicorn commands, or another process manager. Do not free a port owned by another service.
