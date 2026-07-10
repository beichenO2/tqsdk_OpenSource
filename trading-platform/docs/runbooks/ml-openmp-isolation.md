# ML OpenMP 隔离 Runbook

## 问题概述

### 崩溃现象

在 macOS 上，若同一 Python 进程同时加载 **pip 安装的 PyTorch** 与 **LightGBM**，进程可能在训练或推理阶段以 **SIGSEGV（段错误）** 异常退出，无 Python  traceback，仅留下 native crash 报告。

典型 crash 栈关键帧：

```
__kmp_suspend_initialize_thread
...
LGBM_DatasetCreateFromMat
```

前者来自 Intel OpenMP runtime（`libomp` / `libiomp5`），后者来自 LightGBM 原生扩展在创建 Dataset 时初始化线程池。

### 根因

| 组件 | OpenMP 来源 |
|------|-------------|
| pip wheel 版 `torch` | 自带 vendored `libomp`（wheel 内嵌） |
| pip wheel 版 `lightgbm` | 运行时通过 rpath 链接 Homebrew 或系统 `libomp` |

同一进程加载 **两份不同的 OpenMP runtime** 会导致内部状态冲突。社区共识（LightGBM FAQ、GitHub issue #6595、joblib/threadpoolctl 文档）：

- pip wheel 生态 **无法** 在同进程内可靠统一 OpenMP。
- 可行方案：**conda-forge 单一 OpenMP 栈**，或 **进程隔离**（训练与交易分进程）。

本项目决策：**交易进程零 LightGBM**；ML 训练走 **独立 worker 子进程**。

---

## 三条铁律

1. **交易进程禁止 import lightgbm**  
   适用于 API、gateway、strategy worker 及任何实盘/仿真热路径。这些进程可保留 `torch`、`xgboost`（生产 API 路径），但不得 `import lightgbm`。

2. **ML 训练必须独立子进程，且按框架懒加载**  
   LightGBM 仅在 ML worker 子进程中、且仅在需要该框架时才 import。不要在主进程或交易 worker 启动时预加载。

3. **同进程严禁 torch + lightgbm**  
   即使设置了线程数或环境变量 workaround，也不应在同一解释器内同时加载两者。这是已知 SIGSEGV 根因，不是性能调优问题。

---

## 依赖分层

### 主依赖（默认安装）

根 `pyproject.toml` / `requirements.txt` 保留：

- `torch>=2.5` — 策略、RL、多处引用
- `xgboost>=2.1` — API 生产推理路径

**不包含** `lightgbm`。

### 可选 `ml` extra

LightGBM 仅在可选依赖中声明：

```bash
# 可编辑安装（推荐）
pip install -e ".[ml]"

# 或仅补装 LightGBM
pip install -r requirements-ml.txt
```

`requirements-ml.txt` 头部有说明：仅供 **隔离的 ML 训练环境** 使用，不得装入交易/API 同一 venv 并在同进程与 torch 共存。

---

## conda-forge 方案（推荐用于 ML worker）

当 ML worker 需要 **PyTorch + LightGBM 同环境**（例如同一训练脚本混用）时，使用 conda-forge 提供的 **单一 OpenMP 栈**，而非 pip 混装。

### 何时使用 `tq-ml` 环境

- ML 训练 worker 子进程需要 LightGBM
- 同一 worker 内还需 PyTorch（或希望与 conda 版 xgboost/sklearn 对齐）
- 希望避免 pip wheel 双 libomp 问题

### 创建环境

```bash
./Start/setup-ml-env.sh
```

脚本行为：

- 检测 `mamba` 或 `conda`；均不可用则报错退出
- 若 `tq-ml` 已存在则提示并退出（幂等）
- 使用 `-c conda-forge --strict-channel-priority` 创建环境：`python=3.12`、`pytorch`、`lightgbm`、`xgboost`、`scikit-learn`、`pandas`、`pyarrow`

### 让训练 worker 使用该解释器

```bash
export ML_PYTHON_BIN=~/miniforge3/envs/tq-ml/bin/python
```

路径以本机 conda base 为准；`setup-ml-env.sh` 结束时会打印实际路径。

验证：

```bash
$ML_PYTHON_BIN -c "import lightgbm, torch; print('ok')"
```

---

## 应急手段与风险

以下手段 **不能** 替代进程隔离，仅作临时排查或降级；**不保证** 消除 SIGSEGV。

| 手段 | 说明 |
|------|------|
| `OMP_NUM_THREADS=1` | 限制 OpenMP 线程，有时减轻竞争，**不解决** 双 runtime 加载 |
| `n_jobs=1`（LightGBM / sklearn） | 减少并行线程池，同上 |
| `KMP_DUPLICATE_LIB_OK=TRUE` | Intel 文档与社区均标为 **unsafe**；可能掩盖崩溃而非修复。**本项目禁用**，不得作为默认配置 |

若在生产或 CI 中仍出现 native crash，优先检查是否有进程违反「三条铁律」，而非继续堆环境变量。

---

## 参考链接

- [LightGBM FAQ — segfault / OpenMP](https://lightgbm.readthedocs.io/en/latest/FAQ.html)
- [lightgbm-org/LightGBM#6595 — macOS pip torch + lightgbm SIGSEGV](https://github.com/microsoft/LightGBM/issues/6595)
- [joblib/threadpoolctl — multiple OpenMP runtimes](https://github.com/joblib/threadpoolctl/blob/master/docs/multiple_openmp.md)
