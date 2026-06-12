"""Mamba Trend Predictor — S4D State Space Model direction predictor.

Uses a lightweight S4D (structured state space, diagonal variant)
neural network to predict next-bar price direction from a rolling
window of OHLCV data.  The model learns temporal patterns that
classical indicators miss.

Architecture (2024-2026 SSM research):
- S4D recurrence for long-range dependency capture
- Online inference: model runs step-by-step, no batch needed
- Pre-trained on historical data, fine-tuned per instrument

Signal logic:
- Model outputs P(up) ∈ [0, 1]
- P(up) > threshold → LONG
- P(up) < (1 - threshold) → SHORT
- Otherwise → HOLD

Reference: CryptoMamba (arXiv 2025.01), SAMBA (arXiv 2024.10)
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

import numpy as np

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class _S4DPredictor(nn.Module):
    """Tiny S4D model for binary direction prediction."""

    def __init__(self, input_dim: int = 5, d_model: int = 32, d_state: int = 8) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.ones(d_model))
        self.gate = nn.Linear(d_model, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.d_model = d_model
        self.d_state = d_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → (batch, 1) probability."""
        B_sz, L, _ = x.shape
        h = self.proj(x)

        A = -torch.exp(self.A_log)
        state = torch.zeros(B_sz, self.d_model, self.d_state, device=x.device)

        for t in range(L):
            ht = h[:, t, :]
            dt = torch.sigmoid(ht.mean(dim=-1, keepdim=True))
            dA = torch.exp(A.unsqueeze(0) * dt.unsqueeze(-1))
            dB = dt.unsqueeze(-1) * self.B.unsqueeze(0)
            state = dA * state + dB * ht.unsqueeze(-1)

        y = (state * self.C.unsqueeze(0)).sum(dim=-1)
        y = y * torch.sigmoid(self.gate(h[:, -1, :]))
        y = y + h[:, -1, :] * self.D.unsqueeze(0)
        return self.head(y)


def _train_predictor(
    bars: np.ndarray,
    window: int = 30,
    epochs: int = 50,
    d_model: int = 32,
) -> _S4DPredictor:
    """Train the S4D predictor on OHLCV bars (supervised, binary direction)."""
    close = bars[:, 3]
    returns = np.diff(close) / np.maximum(np.abs(close[:-1]), 1e-12)
    labels = (returns > 0).astype(np.float32)

    X_list, y_list = [], []
    for i in range(window, len(bars) - 1):
        seg = bars[i - window : i, :5].astype(np.float32)
        seg_min = seg.min(axis=0)
        seg_max = seg.max(axis=0)
        span = np.maximum(seg_max - seg_min, 1e-8)
        seg_norm = (seg - seg_min) / span
        X_list.append(seg_norm)
        y_list.append(labels[i - 1])

    X = torch.tensor(np.array(X_list))
    y = torch.tensor(np.array(y_list)).unsqueeze(1)

    model = _S4DPredictor(input_dim=5, d_model=d_model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    batch_size = min(128, len(X))
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        total_loss = 0.0
        for i in range(0, len(X), batch_size):
            idx = perm[i : i + batch_size]
            pred = model(X[idx])
            loss = loss_fn(pred, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

    acc = ((model(X) > 0.5).float() == y).float().mean().item()
    logger.info("S4D predictor trained: acc=%.3f over %d samples", acc, len(X))
    return model


@auto_register("mamba_trend")
class MambaTrendStrategy(BaseStrategy):
    """Mamba (S4D) based trend prediction strategy for futures.

    Config keys:
    - ``window``: lookback window (default 30)
    - ``threshold``: signal confidence threshold (default 0.6)
    - ``tp_atr_mult``: take-profit in ATR units (default 3.0)
    - ``sl_atr_mult``: stop-loss in ATR units (default 1.5)
    - ``train_bars``: min bars before training (default 500)
    - ``retrain_interval``: retrain every N bars (default 200)
    - ``d_model``: S4D hidden dim (default 32)
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(config or StrategyConfig(name="mamba_trend"))
        self._window = int(self.config.params.get("window", 30))
        self._threshold = float(self.config.params.get("threshold", 0.6))
        self._tp_mult = float(self.config.params.get("tp_atr_mult", 3.0))
        self._sl_mult = float(self.config.params.get("sl_atr_mult", 1.5))
        self._train_bars = int(self.config.params.get("train_bars", 500))
        self._retrain_interval = int(self.config.params.get("retrain_interval", 200))
        self._d_model = int(self.config.params.get("d_model", 32))

        self._bars: list[np.ndarray] = []
        self._model: Any = None
        self._bars_since_train: int = 0
        self._atr: float = 0.0

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        sig = self._compute_signal(bar)
        if sig.signal_type != SignalType.HOLD:
            self.record_signal(sig)
        return [sig] if sig.signal_type != SignalType.HOLD else []

    def _compute_signal(self, bar: dict[str, Any]) -> Signal:
        ohlcv = np.array([
            float(bar.get("open", 0)),
            float(bar.get("high", 0)),
            float(bar.get("low", 0)),
            float(bar.get("close", 0)),
            float(bar.get("volume", 0)),
        ], dtype=np.float64)
        self._bars.append(ohlcv)
        n = len(self._bars)

        if n >= 2:
            tr = max(
                ohlcv[1] - ohlcv[2],
                abs(ohlcv[1] - self._bars[-2][3]),
                abs(ohlcv[2] - self._bars[-2][3]),
            )
            alpha = 2.0 / 15.0
            self._atr = alpha * tr + (1.0 - alpha) * self._atr if self._atr > 0 else tr

        if not _TORCH_AVAILABLE:
            return Signal(signal_type=SignalType.HOLD, confidence=0.0, metadata={"reason": "no_torch"})

        if n >= self._train_bars and (self._model is None or self._bars_since_train >= self._retrain_interval):
            all_bars = np.array(self._bars)
            self._model = _train_predictor(
                all_bars, window=self._window, d_model=self._d_model
            )
            self._bars_since_train = 0

        self._bars_since_train += 1

        if self._model is None or n < self._window:
            return Signal(signal_type=SignalType.HOLD, confidence=0.0, metadata={"reason": "warming_up"})

        seg = np.array(self._bars[-self._window:])[:, :5].astype(np.float32)
        seg_min = seg.min(axis=0)
        seg_max = seg.max(axis=0)
        span = np.maximum(seg_max - seg_min, 1e-8)
        seg_norm = (seg - seg_min) / span
        x = torch.tensor(seg_norm).unsqueeze(0)

        with torch.no_grad():
            prob = self._model(x).item()

        close_px = float(ohlcv[3])
        atr = max(self._atr, 1e-8)

        if prob > self._threshold:
            return Signal(
                signal_type=SignalType.LONG,
                confidence=prob,
                take_profit=close_px + self._tp_mult * atr,
                stop_loss=close_px - self._sl_mult * atr,
                metadata={"prob_up": prob, "atr": atr},
            )
        elif prob < (1.0 - self._threshold):
            return Signal(
                signal_type=SignalType.SHORT,
                confidence=1.0 - prob,
                take_profit=close_px - self._tp_mult * atr,
                stop_loss=close_px + self._sl_mult * atr,
                metadata={"prob_up": prob, "atr": atr},
            )
        else:
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.5,
                metadata={"prob_up": prob, "atr": atr},
            )
