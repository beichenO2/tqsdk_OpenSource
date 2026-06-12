"""Mamba-style 时序预测模型 — Selective State Space for Time Series.

Architecture: Simplified Mamba (Gu & Dao, 2024, ICML)
  - Selective State Space Model (S6) core: input-dependent state transitions
  - Linear-time sequence modeling (O(n) vs O(n^2) for Transformers)
  - Gated residual blocks with SiLU activation

Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  — Gu & Dao, ICML 2024

Implementation: Pure PyTorch (no mamba-ssm CUDA dependency).
Backward compatible: LSTMModel API unchanged, internal architecture upgraded.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import (
    BaseMLModel,
    MLFramework,
    MLModelMeta,
    MLModelStatus,
    PredictResult,
    TrainResult,
)

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as _e:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_ERR = _e
else:
    _TORCH_ERR = None

_DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "hidden_size": 128,
    "num_layers": 2,
    "state_size": 16,
    "dropout": 0.2,
    "sequence_length": 30,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 50,
    "early_stopping_patience": 8,
}


def _require_torch():
    if torch is None:
        raise ImportError(f"PyTorch required: {_TORCH_ERR!r}") from _TORCH_ERR


class _SelectiveSSM(nn.Module):
    """Simplified Selective State Space Model (S6) block.

    Core of Mamba (Gu & Dao 2024, ICML):
    - Input-dependent B, C, delta (selective scan)
    - Linear recurrence with discretized state transitions
    - O(L) sequential scan (no quadratic attention)
    """
    def __init__(self, d_model: int, state_size: int = 16):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size

        self.proj_delta = nn.Linear(d_model, d_model, bias=True)
        self.proj_B = nn.Linear(d_model, state_size, bias=False)
        self.proj_C = nn.Linear(d_model, state_size, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_size + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_batch, L, D = x.shape
        delta = nn.functional.softplus(self.proj_delta(x))  # (B, L, D)
        B_input = self.proj_B(x)  # (B, L, N)
        C_input = self.proj_C(x)  # (B, L, N)

        A = -torch.exp(self.A_log)  # (D, N)
        dA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, N)
        dB = delta.unsqueeze(-1) * B_input.unsqueeze(2)  # (B, L, D, N)

        h = torch.zeros(B_batch, D, self.state_size, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t, :].unsqueeze(-1)
            y_t = (h * C_input[:, t].unsqueeze(1)).sum(-1)  # (B, D)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, D)
        return y + x * self.D


class _MambaBlock(nn.Module):
    """Single Mamba block with gated residual."""
    def __init__(self, d_model: int, state_size: int = 16, dropout: float = 0.1, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner)
        self.ssm = _SelectiveSSM(d_inner, state_size)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_branch = self.conv1d(x_branch.transpose(1, 2)).transpose(1, 2)
        x_branch = nn.functional.silu(x_branch)
        x_branch = self.ssm(x_branch)

        z = nn.functional.silu(z)
        x_out = x_branch * z
        x_out = self.out_proj(x_out)
        return residual + self.dropout(x_out)


class _MambaNet(nn.Module):
    """Mamba-style sequence classifier: input_proj → N MambaBlocks → pool → classify."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 state_size: int = 16, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList([
            _MambaBlock(hidden_size, state_size, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, -1, :])
        return self.fc(x)


class LSTMModel(BaseMLModel):
    """Mamba-style classifier for time-series direction prediction.

    Architecture upgraded from LSTM to Mamba (Gu & Dao 2024, ICML).
    API remains backward compatible (class name kept for import compat).
    """

    def __init__(self, meta: MLModelMeta) -> None:
        if meta.framework != MLFramework.PYTORCH:
            meta = meta.model_copy(update={"framework": MLFramework.PYTORCH})
        merged = {**_DEFAULT_HYPERPARAMS, **meta.hyperparams}
        meta = meta.model_copy(update={"hyperparams": merged})
        super().__init__(meta)
        _require_torch()
        dev = merged.get("device")
        if dev:
            self._device = torch.device(dev)
        else:
            self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._net: _MambaNet | None = None
        self._seq_len: int = merged["sequence_length"]

    def _build_net(self, input_size: int) -> _MambaNet:
        hp = self.meta.hyperparams
        net = _MambaNet(
            input_size, hp["hidden_size"], hp["num_layers"],
            hp.get("state_size", 16), hp["dropout"],
        )
        return net.to(self._device)

    @staticmethod
    def make_sequences(X: np.ndarray, y: np.ndarray | None, seq_len: int):
        """Convert flat feature matrix to overlapping sequences for LSTM input."""
        xs, ys = [], []
        for i in range(seq_len, len(X)):
            xs.append(X[i - seq_len : i])
            if y is not None:
                ys.append(y[i])
        X_seq = np.array(xs, dtype=np.float32)
        y_seq = np.array(ys, dtype=np.int64) if y is not None else None
        return X_seq, y_seq

    async def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TrainResult:
        _require_torch()
        self.meta.status = MLModelStatus.TRAINING
        t0 = time.perf_counter()

        hp = self.meta.hyperparams
        X_tr_seq, y_tr_seq = self.make_sequences(X_train, y_train, self._seq_len)
        input_size = X_tr_seq.shape[2]
        self._net = self._build_net(input_size)

        train_ds = TensorDataset(
            torch.from_numpy(X_tr_seq).to(self._device),
            torch.from_numpy(y_tr_seq).to(self._device),
        )
        train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_v_seq, y_v_seq = self.make_sequences(X_val, y_val, self._seq_len)
            val_ds = TensorDataset(
                torch.from_numpy(X_v_seq).to(self._device),
                torch.from_numpy(y_v_seq).to(self._device),
            )
            val_loader = DataLoader(val_ds, batch_size=hp["batch_size"])

        optimizer = torch.optim.Adam(self._net.parameters(), lr=hp["learning_rate"])
        criterion = nn.CrossEntropyLoss()
        patience = hp["early_stopping_patience"]

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(hp["epochs"]):
            self._net.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = self._net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

            if val_loader is not None:
                self._net.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        val_loss += criterion(self._net(xb), yb).item() * xb.size(0)
                val_loss /= len(val_loader.dataset)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info("Early stop at epoch %d", epoch + 1)
                        break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        train_acc = self._accuracy(train_loader)
        val_acc = self._accuracy(val_loader) if val_loader else None

        self.meta.status = MLModelStatus.TRAINED
        self.meta.trained_at = datetime.now(UTC)
        self.meta.metrics = {
            "train_accuracy": train_acc,
            **({"val_accuracy": val_acc} if val_acc is not None else {}),
        }

        return TrainResult(
            train_score=train_acc,
            val_score=val_acc,
            metrics=dict(self.meta.metrics),
            duration_seconds=time.perf_counter() - t0,
            epochs=epoch + 1,
        )

    def _accuracy(self, loader) -> float:
        if loader is None or self._net is None:
            return 0.0
        self._net.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in loader:
                preds = self._net(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / max(total, 1)

    def predict(self, X: np.ndarray) -> PredictResult:
        _require_torch()
        if self._net is None:
            raise RuntimeError("Model not trained.")

        if X.ndim == 2:
            X_seq, _ = self.make_sequences(X, None, self._seq_len)
        else:
            X_seq = X.astype(np.float32)

        self._net.eval()
        with torch.no_grad():
            t = torch.from_numpy(X_seq).to(self._device)
            logits = self._net(t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

        return PredictResult(
            predictions=[float(p) for p in preds],
            probabilities=[row.tolist() for row in proba],
        )

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        X_seq, y_seq = self.make_sequences(X_test, y_test, self._seq_len)
        result = self.predict(X_seq)
        preds = np.array(result.predictions)
        from sklearn.metrics import accuracy_score, f1_score
        return {
            "accuracy": float(accuracy_score(y_seq, preds)),
            "f1": float(f1_score(y_seq, preds, zero_division=0)),
        }

    def save(self, path: str) -> str:
        if self._net is None:
            raise RuntimeError("No model to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self._net.state_dict(), "meta": self.meta.model_dump()}, path)
        self.meta.artifact_path = path
        return path

    def load(self, path: str) -> None:
        _require_torch()
        data = torch.load(path, map_location=self._device, weights_only=False)
        meta_dict = data.get("meta", {})
        input_size = len(meta_dict.get("feature_columns", [])) or 21
        self._net = self._build_net(input_size)
        self._net.load_state_dict(data["state_dict"])
        self.meta.status = MLModelStatus.TRAINED
        self.meta.artifact_path = path
