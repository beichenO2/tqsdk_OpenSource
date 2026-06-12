"""Temporal Fusion Transformer (TFT) Trading Strategy.

Multi-horizon price direction forecasting with interpretable attention.
Uses Gated Residual Networks (GRN) for feature selection and multi-head
attention for temporal patterns. Outputs probabilistic quantile forecasts
for risk-aware position sizing.

Architecture:
  OHLCV + indicators → Variable Selection (GRN) → LSTM encoder →
  Multi-Head Attention (interpretable) → Quantile Output → Trading Signal

Usage:
    python scripts/research/frontier/tft_strategy.py [--symbol BTCUSDT] [--weeks 80]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys

import os as _os
import sys as _sys

try:
    from polarisor_port_sdk import submit_task as _sdk_submit, complete_task as _sdk_complete
except ImportError:
    _sdk_submit = _sdk_complete = None

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import add_technical_features, compute_metrics, load_crypto_bars

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    """GRN — core building block of TFT for non-linear feature processing."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1,
                 context_dim: int | None = None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.skip(x)
        h = F.elu(self.fc1(x))
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)
        h = self.dropout(F.elu(self.fc2(h)))
        h = self.glu(h)
        return self.layer_norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    """Learns which input features matter most — key for interpretability."""

    def __init__(self, n_vars: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        self.flattened_grn = GatedResidualNetwork(n_vars * hidden_dim, hidden_dim, n_vars, dropout)
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(n_vars)
        ])
        self.hidden_dim = hidden_dim

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # inputs: (batch, time, n_vars * hidden_dim) or (batch, n_vars * hidden_dim)
        has_time = inputs.dim() == 3
        if has_time:
            B, T, _ = inputs.shape
            flat = inputs.reshape(B * T, -1)
        else:
            flat = inputs

        weights = F.softmax(self.flattened_grn(flat), dim=-1)  # (B*T, n_vars)

        var_outputs = []
        for i, grn in enumerate(self.var_grns):
            var_input = flat[:, i * self.hidden_dim:(i + 1) * self.hidden_dim]
            var_outputs.append(grn(var_input))

        var_outputs = torch.stack(var_outputs, dim=1)  # (B*T, n_vars, hidden)
        selected = (weights.unsqueeze(-1) * var_outputs).sum(dim=1)  # (B*T, hidden)

        if has_time:
            selected = selected.reshape(B, T, -1)
            weights = weights.reshape(B, T, -1)

        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Attention with per-head interpretability — can visualize which timesteps matter."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, self.d_k)
        self.W_o = nn.Linear(self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = q.shape
        Q = self.W_q(q).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (B, heads, T, d_k)
        context = context.mean(dim=1)  # average over heads
        output = self.W_o(context)
        avg_attn = attn_weights.mean(dim=1)  # (B, T, T) for interpretability

        return output, avg_attn


class TFTModel(nn.Module):
    """Simplified Temporal Fusion Transformer for trading signal generation.

    Combines variable selection, LSTM temporal processing, and interpretable
    multi-head attention for multi-step direction prediction.
    """

    def __init__(self, n_features: int, hidden_dim: int = 64, n_heads: int = 4,
                 lstm_layers: int = 2, dropout: float = 0.15, n_quantiles: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features

        self.input_proj = nn.Linear(n_features, n_features * hidden_dim)
        self.vsn = VariableSelectionNetwork(n_features, hidden_dim, dropout)

        self.lstm_encoder = nn.LSTM(hidden_dim, hidden_dim, lstm_layers,
                                     batch_first=True, dropout=dropout if lstm_layers > 1 else 0)

        self.post_lstm_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.attention = InterpretableMultiHeadAttention(hidden_dim, n_heads, dropout)
        self.post_attn_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # down / neutral / up
        )

        self.quantile_head = nn.Linear(hidden_dim, n_quantiles)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, F_in = x.shape
        projected = self.input_proj(x)
        selected, var_weights = self.vsn(projected)

        lstm_out, _ = self.lstm_encoder(selected)
        lstm_enriched = self.post_lstm_grn(lstm_out)

        attn_out, attn_weights = self.attention(lstm_enriched, lstm_enriched, lstm_enriched)
        enriched = self.post_attn_grn(attn_out)
        enriched = self.layer_norm(enriched + lstm_enriched)

        final = enriched[:, -1, :]  # last timestep
        direction = self.direction_head(final)
        quantiles = self.quantile_head(final)

        return direction, quantiles, var_weights, attn_weights


def build_tft_dataset(bars: pd.DataFrame, seq_len: int = 30, horizon: int = 5,
                      threshold: float = 0.003):
    """Build sequences with multi-feature input and direction labels."""
    df = add_technical_features(bars)

    feature_cols = [
        "returns", "log_returns", "vol_ratio", "rsi", "bb_pctb",
        "macd_hist", "atr_14", "high_low_range", "volatility_20",
    ]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    features = df[feature_cols].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)

    labels = np.ones(len(closes), dtype=np.int64)
    returns_fwd = np.zeros(len(closes))
    for i in range(len(closes) - horizon):
        ret = (closes[i + horizon] - closes[i]) / (closes[i] + 1e-10)
        returns_fwd[i] = ret
        if ret > threshold:
            labels[i] = 2
        elif ret < -threshold:
            labels[i] = 0

    X_seqs, y_seqs, ret_seqs = [], [], []
    for i in range(seq_len, len(closes) - horizon):
        X_seqs.append(features[i - seq_len:i])
        y_seqs.append(labels[i])
        ret_seqs.append(returns_fwd[i])

    return np.array(X_seqs), np.array(y_seqs), np.array(ret_seqs), feature_cols


def train_tft(X: np.ndarray, y: np.ndarray, n_features: int,
              epochs: int = 40, lr: float = 5e-4, batch_size: int = 64):
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TFTModel(n_features=n_features, hidden_dim=48, n_heads=4, lstm_layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            direction, quantiles, _, _ = model(xb)
            loss = criterion(direction, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            xv = torch.FloatTensor(X_val).to(device)
            direction, _, var_weights, attn_weights = model(xv)
            preds = direction.argmax(dim=1).cpu().numpy()
            acc = (preds == y_val).mean()
            if acc > best_val_acc:
                best_val_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d/%d | val_acc=%.3f (best=%.3f)", epoch + 1, epochs, acc, best_val_acc)

    if best_state:
        model.load_state_dict(best_state)

    return model, mean, std, best_val_acc


def backtest_tft(
    bars: pd.DataFrame, model: TFTModel, mean: np.ndarray, std: np.ndarray,
    feature_cols: list[str], seq_len: int = 30, leverage: int = 8,
    initial_capital: float = 100.0, commission_pct: float = 0.0004,
    slippage_pct: float = 0.0003, confidence_threshold: float = 0.42,
) -> dict:
    df = add_technical_features(bars)
    features = df[feature_cols].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()

    capital = initial_capital
    holding = None
    trades = []
    equity_curve = [capital]
    cost = commission_pct + slippage_pct

    for i in range(seq_len, len(closes) - 1):
        seq = features[i - seq_len:i]
        seq_norm = (seq - mean) / std

        with torch.no_grad():
            inp = torch.FloatTensor(seq_norm).unsqueeze(0).to(device)
            direction, quantiles, var_weights, attn_weights = model(inp)
            probs = torch.softmax(direction, dim=1).cpu().numpy()[0]
            signal = int(direction.argmax(dim=1).item())
            confidence = float(max(probs))

        if confidence < confidence_threshold:
            signal = 1

        if holding is not None:
            bars_held = i - holding["entry_bar"]
            pnl_pct = (closes[i] / holding["entry_price"] - 1) * holding["direction"] * leverage

            should_close = (
                pnl_pct > 0.06
                or pnl_pct < -0.025
                or bars_held >= 48
                or (signal != 1 and (signal == 2) != (holding["direction"] == 1))
            )

            if should_close:
                realized = capital * abs(pnl_pct) * (1 if pnl_pct > 0 else -1) - capital * cost * 2
                capital += realized
                trades.append({
                    "entry_bar": holding["entry_bar"], "exit_bar": i,
                    "direction": holding["direction"], "pnl_pct": pnl_pct,
                    "realized": realized,
                })
                holding = None

        if holding is None and signal != 1 and confidence >= confidence_threshold:
            size_factor = min(1.0, (confidence - confidence_threshold) / (1.0 - confidence_threshold) + 0.5)
            direction = 1 if signal == 2 else -1
            holding = {"entry_bar": i, "entry_price": closes[i], "direction": direction,
                       "size": size_factor}
            capital -= capital * cost * size_factor

        equity_curve.append(capital)

    if holding is not None:
        i = len(closes) - 1
        pnl_pct = (closes[i] / holding["entry_price"] - 1) * holding["direction"] * leverage
        realized = capital * abs(pnl_pct) * (1 if pnl_pct > 0 else -1) - capital * cost * 2
        capital += realized
        trades.append({
            "entry_bar": holding["entry_bar"], "exit_bar": i,
            "direction": holding["direction"], "pnl_pct": pnl_pct,
            "realized": realized,
        })
        equity_curve.append(capital)

    return compute_metrics(trades, initial_capital, capital, equity_curve)


def main():

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="ml-training", command="tft_strategy.py", requester="tft-strategy", estimated_duration_sec=3600)
            _task_id = _tr.get("task_id")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="TFT Trading Strategy")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--weeks", type=int, default=80)
    parser.add_argument("--leverage", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=30)
    args = parser.parse_args()

    bars = load_crypto_bars(args.symbol, args.timeframe, args.weeks)
    if len(bars) < 200:
        logger.error("Insufficient data: %d bars", len(bars))
        return

    split_idx = int(len(bars) * 0.7)
    train_bars = bars.iloc[:split_idx]
    test_bars = bars.iloc[split_idx:]

    logger.info("Building TFT dataset (train=%d, test=%d bars)...", len(train_bars), len(test_bars))
    X, y, _, feature_cols = build_tft_dataset(train_bars, seq_len=args.seq_len)
    logger.info("Training: %d sequences, %d features", X.shape[0], X.shape[2])

    model, mean, std, val_acc = train_tft(X, y, n_features=len(feature_cols), epochs=args.epochs)
    logger.info("Training complete. Best val accuracy: %.3f", val_acc)

    logger.info("Backtesting on out-of-sample data...")
    results = backtest_tft(test_bars, model, mean, std, feature_cols,
                           seq_len=args.seq_len, leverage=args.leverage)

    logger.info("\n%s", "=" * 60)
    logger.info("TFT STRATEGY — %s (OOS)", args.symbol)
    logger.info("=" * 60)
    for k, v in results.items():
        logger.info("  %s: %s", k, v)

    out_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "frontier"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tft_results.json"
    with open(out_path, "w") as f:
        json.dump({"strategy": "TFT", "symbol": args.symbol, "leverage": args.leverage,
                    "val_accuracy": round(val_acc, 4), **results}, f, indent=2)
    logger.info("Results saved to %s", out_path)

    return results


    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    main()
