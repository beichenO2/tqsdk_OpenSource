# tqsdk 产物契约

## research_artifact

策略研究工作台的导出产物。

- **Schema**: `research_artifact.schema.json`
- **Example**: `examples/research_artifact.example.json`

### 消费方

| 消费方 | 消费格式 | 接口 |
|--------|----------|------|
| KnowLever | Markdown + metadata JSON | `GET /api/v1/research/runs/{id}/artifact/markdown` |
| AutoOffice | 结构化 JSON | `GET /api/v1/research/runs/{id}/artifact` |

### 变更历史

- 2026-05-01: 初始版本，含 run_id/variant/params/metrics/validation/risk/decision 字段
