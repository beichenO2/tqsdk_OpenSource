---
name: tqsdk-ops
description: Use when inspecting, deploying, recovering, or changing the tqsdk data collector or TqSdk credential gateway, or when a tqsdk task may bind a stable port or leave a persistent process.
---

# tqsdk Operations

## Runtime contract (PolarManager)

**PolarManager** = PolarPort (`:11050`) + PolarProcess (`:11055`) + PolarBudget (`:11060`). Port/lifecycle for tqsdk services go through Port/Process; heavy collector backfills should consult Budget. The governed services are:

| Service ID | Preferred port | Auto-start | Health |
|---|---:|---:|---|
| `tqsdk-data-collector` | 18900 | true | `http://127.0.0.1:18900/health` |
| `tqsdk-gateway` | 12890 | false | `http://127.0.0.1:12890/health` |

The gateway owns plaintext TqSdk credentials. Keep it stopped unless an explicit trading operation requires it. Never obtain D-class secrets in the collector or trading platform.

## Required workflow

1. Read the `polar-runtime-governance` skill and run its read-only audit.
2. Inspect `http://127.0.0.1:11055/api/services` and `http://127.0.0.1:11050/api/list`.
3. Use only the exact PolarProcess action for the intended service.
4. Verify the service record, one PolarPort owner, one listener, and the declared health endpoint.

```bash
curl -fsS http://127.0.0.1:11055/api/services/tqsdk-data-collector
curl -fsS -X POST http://127.0.0.1:11055/api/services/tqsdk-data-collector/restart
curl -fsS http://127.0.0.1:18900/health
```

## Prohibited

- Do not invoke the foreground launchers manually.
- Do not use raw Python/Uvicorn commands for persistent services.
- Do not use `nohup`, background shell jobs, PID files, `pgrep`, direct signals, or launchd.
- Do not start the gateway as a side effect of collector diagnosis.
- Do not repair another service's port or process record.

See `DEPLOY.md` for registration and `TROUBLESHOOT.md` for read-only diagnosis.
