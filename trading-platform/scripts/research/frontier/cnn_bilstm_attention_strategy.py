"""CNN-BiLSTM-Attention Hybrid Trading Strategy.

Cross-domain architecture combining techniques from:
  - Computer vision (CNN): local pattern extraction on price/volume bars
  - NLP (BiLSTM + Attention): bidirectional sequential modeling with focus mechanism
  - Signal processing: multi-scale feature extraction via dilated convolutions

This is an ensemble of spatial + temporal + attention features that captures
both micro-structure patterns (candle patterns via CNN) and macro trends
(momentum regimes via BiLSTM).

Architecture:
  Input → [Multi-Scale CNN branch] + [BiLSTM branch] → Cross-Attention Fusion →
  Regime-Aware Classification Head → Trading Signal

Usage:
    python scripts/research/frontier/cnn_bilstm_attention_strategy.py [--symbol BTCUSDT]
"""

from __future__ import annotations

import argparse
import json
import logging
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


class MultiScaleCNN(nn.Module):
    """Multi-scale 1D convolutions with dilated kernels — inspired by WaveNet/TCN.

    Captures patterns at different time scales simultaneously:
    - kernel=3, dilation=1: short-term (3-bar patterns like hammers, engulfing)
    - kernel=3, dilation=2: medium-term (6-bar patterns)
    - kernel=3, dilation=4: longer-term (12-bar patterns, half-day on 1h)
    """

    def __init__(self, in_channels: int, hidden_channels: int = 32, dropout: float = 0.15):
        super().__init__()
        self.conv_d1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1, dilation=1)
        self.conv_d2 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=2, dilation=2)
        self.conv_d4 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=4, dilation=4)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)

        self.fusion = nn.Conv1d(hidden_channels * 3, hidden_channels, kernel_size=1)
        self.bn_fusion = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        h1 = F.gelu(self.bn1(self.conv_d1(x)))
        h2 = F.gelu(self.bn2(self.conv_d2(x)))
        h4 = F.gelu(self.bn4(self.conv_d4(x)))

        concat = torch.cat([h1, h2, h4], dim=1)  # (batch, 3*hidden, seq)
        fused = F.gelu(self.bn_fusion(self.fusion(concat)))
        return self.dropout(fused)  # (batch, hidden, seq)


class BiLSTMWithAttention(nn.Module):
    """Bidirectional LSTM + self-attention for temporal modeling."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True,
                               bidirectional=True, dropout=dropout if n_layers > 1 else 0)
        self.attn_query = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_key = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_value = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq, features)
        lstm_out, _ = self.bilstm(x)  # (batch, seq, hidden*2)

        Q = self.attn_query(lstm_out)
        K = self.attn_key(lstm_out)
        V = self.attn_value(lstm_out)

        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = self.attn_proj(context)
        output = self.layer_norm(lstm_out + context)

        return output, attn_weights


class CrossAttentionFusion(nn.Module):
    """Fuses CNN spatial features with BiLSTM temporal features via cross-attention.

    CNN branch queries BiLSTM branch (and vice versa), creating a richer
    representation than simple concatenation.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.cnn_to_lstm_attn = nn.MultiheadAttention(dim, num_heads=4, dropout=dropout, batch_first=True)
        self.lstm_to_cnn_attn = nn.MultiheadAttention(dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, cnn_feat: torch.Tensor, lstm_feat: torch.Tensor) -> torch.Tensor:
        cnn_enriched, _ = self.cnn_to_lstm_attn(cnn_feat, lstm_feat, lstm_feat)
        cnn_enriched = self.norm1(cnn_feat + cnn_enriched)

        lstm_enriched, _ = self.lstm_to_cnn_attn(lstm_feat, cnn_feat, cnn_feat)
        lstm_enriched = self.norm2(lstm_feat + lstm_enriched)

        combined = torch.cat([cnn_enriched, lstm_enriched], dim=-1)
        gate_weight = self.gate(combined)
        fused = self.proj(combined) * gate_weight

        return fused


class RegimeAwareHead(nn.Module):
    """Classification head that first detects market regime, then predicts direction
    conditioned on regime — avoids forcing one model to handle all regimes."""

    def __init__(self, input_dim: int, n_regimes: int = 3, dropout: float = 0.15):
        super().__init__()
        self.regime_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, n_regimes),
        )

        self.direction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, 3),  # down / neutral / up
            )
            for _ in range(n_regimes)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        regime_logits = self.regime_head(x)
        regime_probs = F.softmax(regime_logits, dim=-1)

        direction_logits = torch.stack([head(x) for head in self.direction_heads], dim=1)
        # Mixture of regime-conditioned direction predictions
        mixed_direction = (regime_probs.unsqueeze(-1) * direction_logits).sum(dim=1)

        return mixed_direction, regime_logits


class CNNBiLSTMAttention(nn.Module):
    """Full hybrid model combining all components."""

    def __init__(self, n_features: int, cnn_hidden: int = 32, lstm_hidden: int = 64,
                 n_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        fusion_dim = lstm_hidden * 2

        self.cnn = MultiScaleCNN(n_features, cnn_hidden, dropout)
        self.cnn_proj = nn.Linear(cnn_hidden, fusion_dim)

        self.bilstm = BiLSTMWithAttention(n_features, lstm_hidden, n_layers, dropout)

        self.cross_attn = CrossAttentionFusion(fusion_dim, dropout)

        self.head = RegimeAwareHead(fusion_dim, n_regimes=3, dropout=dropout)

        self.confidence_head = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # x: (batch, seq, features)
        cnn_input = x.transpose(1, 2)  # (batch, features, seq) for Conv1d
        cnn_out = self.cnn(cnn_input).transpose(1, 2)  # (batch, seq, cnn_hidden)
        cnn_projected = self.cnn_proj(cnn_out)  # (batch, seq, fusion_dim)

        lstm_out, attn_weights = self.bilstm(x)  # (batch, seq, lstm_hidden*2)

        fused = self.cross_attn(cnn_projected, lstm_out)  # (batch, seq, fusion_dim)
        final = fused[:, -1, :]  # last timestep

        direction, regime = self.head(final)
        confidence = self.confidence_head(final).squeeze(-1)

        return direction, regime, confidence, attn_weights


def build_dataset(bars: pd.DataFrame, seq_len: int = 30, horizon: int = 5,
                  threshold: float = 0.003):
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

    vol_20 = pd.Series(closes).pct_change().rolling(20).std().fillna(0).values
    vol_median = np.median(vol_20[vol_20 > 0]) if np.any(vol_20 > 0) else 0.01
    regime_labels = np.ones(len(closes), dtype=np.int64)
    for i in range(len(closes)):
        if vol_20[i] > vol_median * 1.5:
            regime_labels[i] = 2  # high vol
        elif vol_20[i] < vol_median * 0.5:
            regime_labels[i] = 0  # low vol

    direction_labels = np.ones(len(closes), dtype=np.int64)
    for i in range(len(closes) - horizon):
        ret = (closes[i + horizon] - closes[i]) / (closes[i] + 1e-10)
        if ret > threshold:
            direction_labels[i] = 2
        elif ret < -threshold:
            direction_labels[i] = 0

    X_seqs, y_dir, y_regime = [], [], []
    for i in range(seq_len, len(closes) - horizon):
        X_seqs.append(features[i - seq_len:i])
        y_dir.append(direction_labels[i])
        y_regime.append(regime_labels[i])

    return np.array(X_seqs), np.array(y_dir), np.array(y_regime), feature_cols


def train_model(X: np.ndarray, y_dir: np.ndarray, y_regime: np.ndarray,
                n_features: int, epochs: int = 50, lr: float = 3e-4, batch_size: int = 64):
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_dir_train, y_dir_val = y_dir[:split], y_dir[split:]
    y_reg_train, y_reg_val = y_regime[:split], y_regime[split:]

    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CNNBiLSTMAttention(n_features, cnn_hidden=32, lstm_hidden=48, dropout=0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr * 5, total_steps=epochs * max(1, len(X_train) // batch_size),
        pct_start=0.3,
    )

    dir_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    regime_criterion = nn.CrossEntropyLoss()
    confidence_criterion = nn.BCELoss()

    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_dir_train),
        torch.LongTensor(y_reg_train),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, ydb, yrb in train_dl:
            xb, ydb, yrb = xb.to(device), ydb.to(device), yrb.to(device)
            optimizer.zero_grad()

            direction, regime, confidence, _ = model(xb)

            loss_dir = dir_criterion(direction, ydb)
            loss_regime = regime_criterion(regime, yrb)

            correct = (direction.argmax(dim=1) == ydb).float()
            loss_conf = confidence_criterion(confidence, correct)

            loss = loss_dir + 0.3 * loss_regime + 0.2 * loss_conf
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            xv = torch.FloatTensor(X_val).to(device)
            direction, regime, confidence, _ = model(xv)
            preds = direction.argmax(dim=1).cpu().numpy()
            acc = (preds == y_dir_val).mean()
            regime_preds = regime.argmax(dim=1).cpu().numpy()
            regime_acc = (regime_preds == y_reg_val).mean()

            if acc > best_val_acc:
                best_val_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d/%d | dir_acc=%.3f (best=%.3f) regime_acc=%.3f",
                        epoch + 1, epochs, acc, best_val_acc, regime_acc)

    if best_state:
        model.load_state_dict(best_state)

    return model, mean, std, best_val_acc


def backtest_hybrid(
    bars: pd.DataFrame, model: CNNBiLSTMAttention, mean: np.ndarray, std: np.ndarray,
    feature_cols: list[str], seq_len: int = 30, leverage: int = 8,
    initial_capital: float = 100.0, commission_pct: float = 0.0004,
    slippage_pct: float = 0.0003, confidence_threshold: float = 0.40,
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
            direction, regime, confidence, attn = model(inp)
            probs = torch.softmax(direction, dim=1).cpu().numpy()[0]
            signal = int(direction.argmax(dim=1).item())
            conf = float(confidence.item())
            regime_pred = int(regime.argmax(dim=1).item())

        effective_conf = conf * max(probs)
        if effective_conf < confidence_threshold:
            signal = 1

        # Regime-adaptive risk management
        if regime_pred == 2:  # high vol
            tp_mult, sl_mult, max_bars = 0.08, 0.03, 36
        elif regime_pred == 0:  # low vol
            tp_mult, sl_mult, max_bars = 0.04, 0.015, 72
        else:  # normal
            tp_mult, sl_mult, max_bars = 0.06, 0.025, 48

        if holding is not None:
            bars_held = i - holding["entry_bar"]
            pnl_pct = (closes[i] / holding["entry_price"] - 1) * holding["direction"] * leverage

            should_close = (
                pnl_pct > tp_mult
                or pnl_pct < -sl_mult
                or bars_held >= max_bars
                or (signal != 1 and (signal == 2) != (holding["direction"] == 1) and effective_conf > 0.5)
            )

            if should_close:
                realized = capital * abs(pnl_pct) * (1 if pnl_pct > 0 else -1) - capital * cost * 2
                capital += realized
                trades.append({
                    "entry_bar": holding["entry_bar"], "exit_bar": i,
                    "direction": holding["direction"], "pnl_pct": pnl_pct,
                    "realized": realized, "regime": holding.get("regime", 1),
                })
                holding = None

        if holding is None and signal != 1 and effective_conf >= confidence_threshold:
            direction_val = 1 if signal == 2 else -1
            holding = {
                "entry_bar": i, "entry_price": closes[i],
                "direction": direction_val, "regime": regime_pred,
            }
            capital -= capital * cost

        equity_curve.append(capital)

    if holding is not None:
        i = len(closes) - 1
        pnl_pct = (closes[i] / holding["entry_price"] - 1) * holding["direction"] * leverage
        realized = capital * abs(pnl_pct) * (1 if pnl_pct > 0 else -1) - capital * cost * 2
        capital += realized
        trades.append({
            "entry_bar": holding["entry_bar"], "exit_bar": i,
            "direction": holding["direction"], "pnl_pct": pnl_pct,
            "realized": realized, "regime": holding.get("regime", 1),
        })
        equity_curve.append(capital)

    metrics = compute_metrics(trades, initial_capital, capital, equity_curve)

    # Regime breakdown
    regime_names = {0: "low_vol", 1: "normal", 2: "high_vol"}
    for r_id, r_name in regime_names.items():
        r_trades = [t for t in trades if t.get("regime") == r_id]
        if r_trades:
            r_wins = len([t for t in r_trades if t["pnl_pct"] > 0])
            metrics[f"regime_{r_name}_trades"] = len(r_trades)
            metrics[f"regime_{r_name}_winrate"] = round(r_wins / len(r_trades) * 100, 1)

    return metrics


def main():

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="ml-training", command="cnn_bilstm_attention_strategy.py", requester="cnn-bilstm-attention", estimated_duration_sec=3600)
            _task_id = _tr.get("task_id")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="CNN-BiLSTM-Attention Trading Strategy")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--weeks", type=int, default=80)
    parser.add_argument("--leverage", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=30)
    args = parser.parse_args()

    bars = load_crypto_bars(args.symbol, args.timeframe, args.weeks)
    if len(bars) < 200:
        logger.error("Insufficient data: %d bars", len(bars))
        return

    split_idx = int(len(bars) * 0.7)
    train_bars = bars.iloc[:split_idx]
    test_bars = bars.iloc[split_idx:]

    logger.info("Building dataset (train=%d, test=%d bars)...", len(train_bars), len(test_bars))
    X, y_dir, y_regime, feature_cols = build_dataset(train_bars, seq_len=args.seq_len)
    logger.info("Training: %d sequences, %d features", X.shape[0], X.shape[2])

    model, mean, std, val_acc = train_model(X, y_dir, y_regime, n_features=len(feature_cols),
                                             epochs=args.epochs)
    logger.info("Training complete. Best direction accuracy: %.3f", val_acc)

    logger.info("Backtesting on out-of-sample data...")
    results = backtest_hybrid(test_bars, model, mean, std, feature_cols,
                              seq_len=args.seq_len, leverage=args.leverage)

    logger.info("\n%s", "=" * 60)
    logger.info("CNN-BiLSTM-ATTENTION — %s (OOS)", args.symbol)
    logger.info("=" * 60)
    for k, v in results.items():
        logger.info("  %s: %s", k, v)

    out_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "frontier"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cnn_bilstm_attn_results.json"
    with open(out_path, "w") as f:
        json.dump({"strategy": "CNN_BiLSTM_Attention", "symbol": args.symbol,
                    "leverage": args.leverage, "val_accuracy": round(val_acc, 4), **results}, f, indent=2)
    logger.info("Results saved to %s", out_path)

    return results


    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    main()
