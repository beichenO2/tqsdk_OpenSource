# tqsdk deployment

## Install transient dependencies

```bash
cd ~/Polarisor/tqsdk/data-collector
pip install -r requirements.txt
cd ../tqsdk-gateway
pip install -r requirements.txt
```

Dependency installation is transient and must not leave a listener.

## Register runtime contracts

```bash
cd ~/Polarisor/tqsdk
bash scripts/register-runtime.sh
curl -fsS http://127.0.0.1:11055/api/services/tqsdk-data-collector
curl -fsS http://127.0.0.1:11055/api/services/tqsdk-gateway
```

Registration reserves `tqsdk-data-collector/tqsdk:18900` and `tqsdk-gateway/tqsdk:12890`. It does not call a lifecycle endpoint. The gateway remains `auto_start=false`.

## Lifecycle actions

Only PolarProcess may act on a persistent service:

```bash
curl -fsS -X POST http://127.0.0.1:11055/api/services/tqsdk-data-collector/restart
```

Starting the credential gateway requires an explicit trading need:

```bash
curl -fsS -X POST http://127.0.0.1:11055/api/services/tqsdk-gateway/start
```

Never start the gateway merely to make collector health look green.

## Verify

```bash
curl -fsS http://127.0.0.1:11050/api/list
curl -fsS http://127.0.0.1:11055/api/services
curl -fsS http://127.0.0.1:18900/health
curl -fsS http://127.0.0.1:12890/health
```

Collector health may truthfully report `initializing` or `error` while the gateway is intentionally stopped.
