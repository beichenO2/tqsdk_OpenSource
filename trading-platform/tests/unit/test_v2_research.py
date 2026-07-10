"""v2.0 研究平台模块验证 — ML/DL/RL/实验管理。"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


# ─── Phase 15: ML Models ───


_REPO = Path(__file__).resolve().parents[2]
_SUBPROC_ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join(
        [
            str(_REPO),
            str(_REPO / "packages"),
            str(_REPO / "packages" / "core"),
        ]
    ),
}


def _run_isolated_script(body: str) -> subprocess.CompletedProcess[str]:
    """Run OpenMP-linked ML code (LightGBM/XGBoost) in a clean subprocess.

    torch's vendored libomp conflicts with theirs when loaded in the same
    process on macOS arm64, causing SIGSEGV.
    """
    return subprocess.run(
        [sys.executable, "-c", body],
        cwd=str(_REPO),
        env=_SUBPROC_ENV,
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestLightGBMModel:
    def _run_lgb_script(self, body: str) -> subprocess.CompletedProcess[str]:
        return _run_isolated_script(body)

    def test_train_predict_evaluate(self):
        script = """
import asyncio
import numpy as np
from ml.lightgbm_model import LightGBMModel
from ml.base import MLModelMeta, MLFramework

rng = np.random.default_rng(42)
X = rng.standard_normal((200, 10))
y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
X_tr, y_tr = X[:150], y[:150]
X_te, y_te = X[150:], y[150:]
meta = MLModelMeta(
    model_id="test_lgb",
    name="TestLGB",
    framework=MLFramework.LIGHTGBM,
    feature_columns=[f"f{i}" for i in range(10)],
)
model = LightGBMModel(meta)
result = asyncio.run(model.train(X_tr, y_tr, X_te, y_te))
assert result.train_score > 0.5
assert result.val_score is not None
pred = model.predict(X_te)
assert len(pred.predictions) == len(X_te)
assert pred.probabilities is not None
metrics = model.evaluate(X_te, y_te)
assert "accuracy" in metrics
assert metrics["accuracy"] > 0.5
print("LGB_TRAIN_OK")
"""
        result = self._run_lgb_script(script)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert "LGB_TRAIN_OK" in result.stdout

    def test_feature_importance(self):
        script = """
import asyncio
import numpy as np
from ml.lightgbm_model import LightGBMModel
from ml.base import MLModelMeta, MLFramework

rng = np.random.default_rng(42)
X = rng.standard_normal((200, 10))
y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
X_tr, y_tr = X[:150], y[:150]
meta = MLModelMeta(
    model_id="test_lgb_fi",
    name="TestLGB",
    framework=MLFramework.LIGHTGBM,
    feature_columns=[f"f{i}" for i in range(10)],
)
model = LightGBMModel(meta)
asyncio.run(model.train(X_tr, y_tr))
fi = model.get_feature_importance()
assert fi is not None
assert len(fi) == 10
print("LGB_FI_OK")
"""
        result = self._run_lgb_script(script)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert "LGB_FI_OK" in result.stdout


class TestXGBoostModel:
    def test_roundtrip(self, tmp_path):
        script = f"""
import asyncio
import numpy as np
from ml.xgboost_model import XGBoostModel
from ml.base import MLModelMeta, MLFramework

rng = np.random.default_rng(7)
X = rng.standard_normal((100, 5))
y = (X[:, 0] > 0).astype(np.int32)
meta = MLModelMeta(
    model_id="test_xgb",
    name="Test",
    framework=MLFramework.XGBOOST,
    feature_columns=[f"f{{i}}" for i in range(5)],
)
model = XGBoostModel(meta)
asyncio.run(model.train(X, y))

path = {str(tmp_path / "xgb.ubj")!r}
model.save(path)
model2 = XGBoostModel(meta)
model2.load(path)
pred = model2.predict(X[:3])
assert len(pred.predictions) == 3
print("XGB_ROUNDTRIP_OK")
"""
        result = _run_isolated_script(script)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert "XGB_ROUNDTRIP_OK" in result.stdout


# ─── Phase 16: DL Models ───
# NOTE: PyTorch LSTM/Transformer segfault under pytest + Python 3.13 + MPS threading.
# These models are verified via standalone scripts (see packages/ml/*_model.py).
# Run manually: PYTHONPATH=packages python -c "from ml.lstm_model import LSTMModel; ..."


class TestLSTMModel:
    def test_import_and_sequence_util(self):
        from ml.lstm_model import LSTMModel

        X = np.random.randn(50, 5).astype(np.float32)
        y = np.ones(50, dtype=np.int64)
        X_seq, y_seq = LSTMModel.make_sequences(X, y, seq_len=10)
        assert X_seq.shape == (40, 10, 5)
        assert y_seq.shape == (40,)


class TestTransformerModel:
    def test_import_and_sequence_util(self):
        from ml.transformer_model import TransformerModel

        X = np.random.randn(50, 8).astype(np.float32)
        y = np.ones(50, dtype=np.int64)
        X_seq, y_seq = TransformerModel.make_sequences(X, y, seq_len=10)
        assert X_seq.shape == (40, 10, 8)
        assert y_seq.shape == (40,)


# ─── Phase 16b: DL Strategy ───


class TestDLStrategy:
    def test_strategy_registration(self):
        from strategy.registry import StrategyRegistry, auto_register
        from strategy.futures.dl_strategy import DLTimeseriesStrategy

        if "dl_timeseries" not in StrategyRegistry.list_registered():
            StrategyRegistry.register("dl_timeseries", DLTimeseriesStrategy)

        assert "dl_timeseries" in StrategyRegistry.list_registered()

    @pytest.mark.asyncio
    async def test_signal_generation_without_model(self):
        from strategy.futures.dl_strategy import DLTimeseriesStrategy
        from strategy.base import StrategyConfig

        cfg = StrategyConfig(name="test_dl", symbols=["rb2501"], params={"sequence_length": 5})
        strat = DLTimeseriesStrategy(cfg)
        for i in range(10):
            bar = {"open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100.5 + i, "volume": 1000}
            signals = await strat.on_bar("rb2501", bar)
            assert signals == []


# ─── Phase 17: RL Environment ───


class TestFuturesTradingEnv:
    def test_env_reset_step(self):
        from rl.trading_env import FuturesTradingEnv

        rng = np.random.default_rng(42)
        bars = rng.standard_normal((200, 5)).cumsum(axis=0) + 5000
        bars = np.abs(bars)
        bars[:, 1] = np.maximum(bars[:, 0], bars[:, 1])
        bars[:, 2] = np.minimum(bars[:, 0], bars[:, 2])
        bars[:, 4] = np.abs(bars[:, 4]) * 1000

        env = FuturesTradingEnv(bars, {"window_size": 10})
        obs, info = env.reset()
        # window*OHLCV + position(3) + unrealized_pnl(1) + time(2) + TA features(10, on by default)
        assert obs.shape == env.observation_space.shape
        assert obs.shape[0] == 10 * 5 + 3 + 1 + 2 + 10
        assert "equity" in info

        obs2, reward, term, trunc, info2 = env.step(0)
        assert not term


# ─── Phase 18: Optuna ───


class TestOptunaSearch:
    def test_basic_search(self):
        from experiment.optuna_search import OptunaHyperSearch

        def objective(params):
            x = params["x"]
            return -(x - 3) ** 2

        searcher = OptunaHyperSearch(
            objective_fn=objective,
            param_space={"x": (0.0, 6.0)},
            direction="maximize",
        )
        result = searcher.run(n_trials=20)
        assert result.n_trials == 20
        assert abs(result.best_params["x"] - 3.0) < 1.5
        assert len(result.all_trials) == 20


# ─── Data Loader ───


class TestFuturesDataLoader:
    def test_import_and_init(self):
        from datahub.futures_loader import FuturesDataLoader

        loader = FuturesDataLoader()
        assert loader.data_root is not None

    def test_instrument_to_symbol(self):
        from datahub.futures_loader import instrument_to_symbol

        assert instrument_to_symbol("rb2509") == "RB"
        assert instrument_to_symbol("IF2501") == "IF"
        assert instrument_to_symbol("AP505") == "AP"
