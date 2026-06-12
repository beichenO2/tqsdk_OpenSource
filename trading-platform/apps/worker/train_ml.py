"""ML 模型训练管道 — 支持 CSV/Parquet 训练 XGBoost 价格方向预测模型。

用法:
    # 从 parquet 自动加载
    python -m apps.worker.train_ml --parquet --bars 5000 --output models/xgb_rb.json

    # 从 CSV 训练 (CSV 需含 open, high, low, close, volume 列)
    python -m apps.worker.train_ml --csv data/rb2501_1min.csv --output models/xgb_rb.json

    # 自定义超参数
    python -m apps.worker.train_ml --parquet --max-depth 8 --n-estimators 200 --lr 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import os as _os
import sys as _sys

try:
    from polarisor_port_sdk import submit_task as _sdk_submit, complete_task as _sdk_complete
except ImportError:
    _sdk_submit = _sdk_complete = None

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_ml")


FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "returns", "volatility", "volume_ratio",
]


def load_parquet_ohlcv(n_bars: int = 2000) -> dict[str, np.ndarray]:
    """从 parquet 缓存加载真实 OHLCV 数据。"""
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("需要 pandas: pip install pandas") from exc

    from pathlib import Path
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


def compute_features(ohlcv: dict[str, np.ndarray], volatility_window: int = 20, volume_ma_period: int = 20) -> np.ndarray:
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

    features = np.column_stack([
        ohlcv["open"],
        ohlcv["high"],
        ohlcv["low"],
        closes,
        volumes,
        returns,
        volatility,
        volume_ratio,
    ])
    return features


def compute_labels(closes: np.ndarray, lookahead: int = 1) -> np.ndarray:
    """标签: 下一根 K 线收盘价是否上涨 (1=涨, 0=跌)。"""
    labels = np.zeros(len(closes), dtype=np.int32)
    labels[:-lookahead] = (closes[lookahead:] > closes[:-lookahead]).astype(np.int32)
    return labels


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hyperparams: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """训练 XGBoostModel 并返回 (model, train_result_dict)。"""
    from ml.base import MLFramework, MLModelMeta
    from ml.xgboost_model import XGBoostModel

    import asyncio

    meta = MLModelMeta(
        model_id=f"xgb_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        name="XGBoost Price Direction",
        framework=MLFramework.XGBOOST,
        feature_columns=list(FEATURE_COLUMNS),
        target_column="direction",
        hyperparams=hyperparams,
    )

    model = XGBoostModel(meta)
    result = asyncio.run(model.train(X_train, y_train, X_val, y_val))

    return model, result.model_dump()


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """在测试集上评估模型。"""
    return model.evaluate(X_test, y_test)


def save_report(
    output_dir: str,
    model_id: str,
    train_result: dict[str, Any],
    test_metrics: dict[str, float],
    hyperparams: dict[str, Any],
    data_info: dict[str, Any],
) -> str:
    """保存训练报告 JSON。"""
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


def main() -> None:

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="ml-training", command="train_ml.py", requester="train-ml", estimated_duration_sec=3600)
            _task_id = _tr.get("task_id")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="训练 XGBoost 价格方向预测模型")
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--parquet", action="store_true", help="从 parquet 缓存加载真实数据")
    data_group.add_argument("--csv", type=str, help="OHLCV CSV 文件路径")

    parser.add_argument("--bars", type=int, default=5000, help="使用的 K 线数量")
    parser.add_argument("--output", type=str, default="models/xgb_model.json", help="模型保存路径")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")

    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--volatility-window", type=int, default=20)
    parser.add_argument("--volume-ma-period", type=int, default=20)

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ML 训练管道启动")
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

    hyperparams = {
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "learning_rate": args.lr,
        "subsample": args.subsample,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "objective": "binary:logistic",
    }

    logger.info("开始训练 XGBoost (params=%s)...", hyperparams)
    model, train_result = train_model(X_train, y_train, X_val, y_val, hyperparams)
    logger.info(
        "训练完成: train_acc=%.4f, val_acc=%s, 耗时=%.2fs",
        train_result["train_score"],
        f'{train_result["val_score"]:.4f}' if train_result["val_score"] else "N/A",
        train_result["duration_seconds"],
    )

    logger.info("测试集评估...")
    test_metrics = evaluate_model(model, X_test, y_test)
    logger.info("测试结果: %s", {k: f"{v:.4f}" for k, v in test_metrics.items()})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = model.save(str(output_path))
    logger.info("模型已保存: %s", saved_path)

    fi = model.get_feature_importance()
    if fi:
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        logger.info("特征重要性:")
        for feat, imp in sorted_fi:
            logger.info("  %-15s  %.4f", feat, imp)

    report_path = save_report(
        output_dir=str(output_path.parent),
        model_id=model.model_id,
        train_result=train_result,
        test_metrics=test_metrics,
        hyperparams=hyperparams,
        data_info={
            "source": data_source,
            "total_samples": n,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "feature_columns": FEATURE_COLUMNS,
        },
    )
    logger.info("训练报告: %s", report_path)
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
