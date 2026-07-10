"""ML 模型训练管道 — 支持 CSV/Parquet 训练 XGBoost / LightGBM 价格方向预测模型。

OpenMP 隔离约束：本 worker 必须在独立子进程中运行，单进程只加载一种 OpenMP 运行时
（XGBoost 或 LightGBM）。禁止 import torch —— 与 LightGBM 同进程会触发 SIGSEGV。

用法:
    # 从 parquet 自动加载 (XGBoost)
    python -m apps.worker.train_ml --parquet --bars 5000 --output models/xgb_rb.json

    # LightGBM 框架
    python -m apps.worker.train_ml --parquet --framework lightgbm --output models/lgb_rb.txt

    # API 子进程模式（stdout 输出 JSON 结果）
    python -m apps.worker.train_ml --parquet --api-json --model-dir models --framework xgboost
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from polarisor_port_sdk import submit_task as _sdk_submit, complete_task as _sdk_complete
except ImportError:
    _sdk_submit = _sdk_complete = None

logger = logging.getLogger("train_ml")

FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "returns", "volatility", "volume_ratio",
]

_FRAMEWORK_ALIASES = {"xgb": "xgboost", "lgb": "lightgbm"}


def _ensure_packages_on_path() -> Path:
    repo = Path(__file__).resolve().parents[2]
    for sub in (repo / "packages" / "core", repo / "packages"):
        sub_str = str(sub)
        if sub_str not in sys.path:
            sys.path.insert(0, sub_str)
    return repo / "packages" / "ml"


def _load_ml_module(module_name: str, ml_dir: Path) -> Any:
    """Load ``ml.<module_name>`` without executing ``ml/__init__.py`` (avoids torch)."""
    import importlib.util

    full_name = f"ml.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    if "ml" not in sys.modules:
        pkg = importlib.util.module_from_spec(
            importlib.util.spec_from_loader(
                "ml",
                loader=None,
                is_package=True,
            )
        )
        pkg.__path__ = [str(ml_dir)]  # type: ignore[attr-defined]
        sys.modules["ml"] = pkg

    spec = importlib.util.spec_from_file_location(full_name, ml_dir / f"{module_name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {full_name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_framework(framework: str) -> tuple[type, Any, str]:
    """Lazy-import exactly one ML framework — never load torch in this process."""
    ml_dir = _ensure_packages_on_path()
    fw = _FRAMEWORK_ALIASES.get(framework, framework)
    base_mod = _load_ml_module("base", ml_dir)
    if fw == "lightgbm":
        lgb_mod = _load_ml_module("lightgbm_model", ml_dir)
        return lgb_mod.LightGBMModel, base_mod.MLFramework.LIGHTGBM, "lgb"
    if fw == "xgboost":
        xgb_mod = _load_ml_module("xgboost_model", ml_dir)
        return xgb_mod.XGBoostModel, base_mod.MLFramework.XGBOOST, "xgb"
    raise ValueError(f"Unsupported framework: {framework!r} (expected xgboost or lightgbm)")


def load_parquet_ohlcv(n_bars: int = 2000) -> dict[str, np.ndarray]:
    """从 parquet 缓存加载真实 OHLCV 数据。"""
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("需要 pandas: pip install pandas") from exc

    repo = Path(__file__).resolve().parents[2]
    search_dirs = [
        repo / "data" / "futures_cache",
        repo / "data" / "crypto_cache",
        repo / ".cache" / "bars",
    ]

    for d in search_dirs:
        if not d.exists():
            continue
        for fp in sorted(d.glob("**/*.parquet")):
            try:
                df = pd.read_parquet(fp)
                if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
                    continue
                if len(df) < n_bars:
                    continue
                df = df.tail(n_bars).reset_index(drop=True)
                logger.info("从 %s 加载 %d bars", fp.name, len(df))
                return {
                    col: df[col].to_numpy(dtype=np.float64)
                    for col in ["open", "high", "low", "close", "volume"]
                }
            except Exception:
                continue

    raise SystemExit(
        f"未找到 >= {n_bars} bars 的 parquet 文件。搜索目录: {[str(d) for d in search_dirs]}"
    )


def load_csv_ohlcv(csv_path: str) -> dict[str, np.ndarray]:
    """从 CSV 加载 OHLCV 数据。"""
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("需要 pandas: pip install pandas") from exc

    df = pd.read_csv(csv_path)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}. 需要: {required}")

    col_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=col_map)
    df = df.dropna(subset=list(required))

    return {col: df[col].to_numpy(dtype=np.float64) for col in ["open", "high", "low", "close", "volume"]}


def compute_features(
    ohlcv: dict[str, np.ndarray],
    volatility_window: int = 20,
    volume_ma_period: int = 20,
) -> np.ndarray:
    """从 OHLCV 计算与 MLFeatureStrategy 一致的 8 维特征矩阵。"""
    closes = ohlcv["close"]
    volumes = ohlcv["volume"]
    n = len(closes)

    returns = np.zeros(n)
    returns[1:] = np.diff(closes) / np.where(closes[:-1] != 0, closes[:-1], 1.0)

    volatility = np.zeros(n)
    for i in range(volatility_window + 1, n):
        window_rets = returns[i - volatility_window: i]
        volatility[i] = np.std(window_rets)

    volume_ratio = np.ones(n)
    for i in range(volume_ma_period, n):
        vma = np.mean(volumes[i - volume_ma_period: i])
        volume_ratio[i] = volumes[i] / vma if vma > 0 else 1.0

    return np.column_stack([
        ohlcv["open"],
        ohlcv["high"],
        ohlcv["low"],
        closes,
        volumes,
        returns,
        volatility,
        volume_ratio,
    ])


def compute_labels(closes: np.ndarray, lookahead: int = 1) -> np.ndarray:
    """标签: 下一根 K 线收盘价是否上涨 (1=涨, 0=跌)。"""
    labels = np.zeros(len(closes), dtype=np.int32)
    labels[:-lookahead] = (closes[lookahead:] > closes[:-lookahead]).astype(np.int32)
    return labels


def build_hyperparams(framework: str, args: argparse.Namespace) -> dict[str, Any]:
    if framework == "lightgbm":
        return {
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "learning_rate": args.lr,
            "subsample": args.subsample,
            "colsample_bytree": 0.8,
            "num_leaves": 31,
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
        }
    return {
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "learning_rate": args.lr,
        "subsample": args.subsample,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "objective": "binary:logistic",
    }


def train_model(
    framework: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hyperparams: dict[str, Any],
    model_id: str,
) -> tuple[Any, dict[str, Any]]:
    """训练指定框架模型并返回 (model, train_result_dict)。"""
    ml_dir = _ensure_packages_on_path()
    base_mod = _load_ml_module("base", ml_dir)
    model_cls, ml_framework, _prefix = _import_framework(framework)
    meta = base_mod.MLModelMeta(
        model_id=model_id,
        name=f"{framework.title()} Price Direction",
        framework=ml_framework,
        feature_columns=list(FEATURE_COLUMNS),
        target_column="direction",
        hyperparams=hyperparams,
    )

    model = model_cls(meta)
    result = asyncio.run(model.train(X_train, y_train, X_val, y_val))
    return model, result.model_dump()


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    return model.evaluate(X_test, y_test)


def save_report(
    output_dir: str,
    model_id: str,
    train_result: dict[str, Any],
    test_metrics: dict[str, float],
    hyperparams: dict[str, Any],
    data_info: dict[str, Any],
) -> str:
    report = {
        "model_id": model_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "hyperparams": hyperparams,
        "data_info": data_info,
        "train_result": train_result,
        "test_metrics": test_metrics,
        "feature_columns": FEATURE_COLUMNS,
    }
    report_path = os.path.join(output_dir, f"{model_id}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report_path


def _configure_logging(api_json: bool) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO if not api_json else logging.WARNING)


def _resolve_output_path(framework: str, args: argparse.Namespace, model_id: str) -> Path:
    prefix = _FRAMEWORK_ALIASES.get(args.framework, args.framework)
    if args.output:
        return Path(args.output)
    model_dir = Path(args.model_dir or "models")
    model_dir.mkdir(parents=True, exist_ok=True)
    ext = ".txt" if prefix == "lightgbm" else ".json"
    return model_dir / f"{model_id}{ext}"


def _build_api_response(
    *,
    model_id: str,
    model_path: str,
    report_path: str,
    train_result: dict[str, Any],
    test_metrics: dict[str, float],
    feature_importance: dict[str, float] | None,
    data_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "model_path": model_path,
        "report_path": report_path,
        "train_accuracy": train_result["train_score"],
        "val_accuracy": train_result.get("val_score"),
        "test_metrics": test_metrics,
        "feature_importance": feature_importance,
        "duration_seconds": train_result["duration_seconds"],
        "data_info": data_info,
    }


def main() -> None:
    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(
                task_type="ml-training",
                command="train_ml.py",
                requester="train-ml",
                estimated_duration_sec=3600,
            )
            _task_id = _tr.get("task_id")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="训练 ML 价格方向预测模型（OpenMP 隔离子进程）")
    data_group = parser.add_mutually_exclusive_group(required=False)
    data_group.add_argument("--parquet", action="store_true", help="从 parquet 缓存加载真实数据")
    data_group.add_argument("--csv", type=str, help="OHLCV CSV 文件路径")

    parser.add_argument("--framework", choices=["xgboost", "lightgbm", "xgb", "lgb"], default="xgboost")
    parser.add_argument("--api-json", action="store_true", help="API 模式：训练结果 JSON 写入 stdout")
    parser.add_argument("--model-dir", type=str, default=None, help="模型输出目录（API 模式）")
    parser.add_argument("--bars", type=int, default=5000, help="使用的 K 线数量")
    parser.add_argument("--output", type=str, default=None, help="模型保存路径（CLI 模式）")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--volatility-window", type=int, default=20)
    parser.add_argument("--volume-ma-period", type=int, default=20)

    args = parser.parse_args()
    _configure_logging(args.api_json)

    framework = _FRAMEWORK_ALIASES.get(args.framework, args.framework)

    if not args.parquet and not args.csv:
        parser.error("one of --parquet or --csv is required")

    if args.api_json and not args.model_dir:
        parser.error("--api-json requires --model-dir")

    logger.info("=" * 60)
    logger.info("ML 训练管道启动 (framework=%s)", framework)
    logger.info("=" * 60)

    if args.parquet:
        logger.info("从 parquet 加载 %d bars...", args.bars)
        ohlcv = load_parquet_ohlcv(n_bars=args.bars)
        data_source = f"parquet ({args.bars} bars)"
    else:
        logger.info("从 CSV 加载: %s", args.csv)
        ohlcv = load_csv_ohlcv(args.csv)
        data_source = f"csv ({args.csv}, {len(ohlcv['close'])} bars)"

    warmup = max(args.volatility_window, args.volume_ma_period) + 2
    logger.info("计算特征 (warmup=%d)...", warmup)
    X_all = compute_features(ohlcv, args.volatility_window, args.volume_ma_period)
    y_all = compute_labels(ohlcv["close"])

    X_all = X_all[warmup:-1]
    y_all = y_all[warmup:-1]
    n = len(X_all)
    logger.info("有效样本数: %d", n)

    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val, y_val = X_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
    X_test, y_test = X_all[n_train + n_val:], y_all[n_train + n_val:]
    logger.info("数据划分: train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test))

    hyperparams = build_hyperparams(framework, args)
    _, _, id_prefix = _import_framework(framework)
    model_id = f"{id_prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    logger.info("开始训练 %s (params=%s)...", framework, hyperparams)
    model, train_result = train_model(
        framework, X_train, y_train, X_val, y_val, hyperparams, model_id
    )
    logger.info(
        "训练完成: train_acc=%.4f, val_acc=%s, 耗时=%.2fs",
        train_result["train_score"],
        f'{train_result["val_score"]:.4f}' if train_result["val_score"] else "N/A",
        train_result["duration_seconds"],
    )

    logger.info("测试集评估...")
    test_metrics = evaluate_model(model, X_test, y_test)
    logger.info("测试结果: %s", {k: f"{v:.4f}" for k, v in test_metrics.items()})

    output_path = _resolve_output_path(framework, args, model_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = model.save(str(output_path))
    logger.info("模型已保存: %s", saved_path)

    fi = model.get_feature_importance()
    if fi and not args.api_json:
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        logger.info("特征重要性:")
        for feat, imp in sorted_fi:
            logger.info("  %-15s  %.4f", feat, imp)

    data_info = {
        "source": data_source,
        "total_samples": n,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
    }

    report_path = save_report(
        output_dir=str(output_path.parent),
        model_id=model.model_id,
        train_result=train_result,
        test_metrics=test_metrics,
        hyperparams=hyperparams,
        data_info=data_info,
    )
    logger.info("训练报告: %s", report_path)

    if args.api_json:
        payload = _build_api_response(
            model_id=model.model_id,
            model_path=saved_path,
            report_path=report_path,
            train_result=train_result,
            test_metrics=test_metrics,
            feature_importance=fi,
            data_info=data_info,
        )
        sys.stdout.write(json.dumps(payload))
        sys.stdout.flush()

    logger.info("=" * 60)
    logger.info("训练管道完成")
    logger.info("=" * 60)

    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    main()
