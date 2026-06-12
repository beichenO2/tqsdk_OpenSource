"""Temporal Fusion Transformer Direction Prediction Strategy.

Uses a simplified TFT architecture (multi-head attention + gating)
for multi-step directional forecasting. Generates trade signals
from attention-weighted predictions.

Usage:
    python scripts/research/tft_direction_strategy.py [--symbol BTCUSDT] [--weeks 80] [--leverage 8]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from datahub.crypto_loader import CryptoDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    logger.error("torch required: pip install torch")
    sys.exit(1)


class GatedResidualNetwork(nn.Module):
    """GRN: core building block of TFT for variable selection and gating."""

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        gate = torch.sigmoid(self.gate(h))
        return self.norm(x + gate * h)


class TemporalAttentionBlock(nn.Module):
    """Multi-head self-attention with interpretable weights."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out)), attn_weights


class SimpleTFT(nn.Module):
    """Simplified Temporal Fusion Transformer for direction prediction.

    Architecture: Input projection → GRN variable selection → LSTM encoder →
    Multi-head attention → GRN decoder → 3-class output (down/neutral/up).
    """

    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 lstm_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.var_select = GatedResidualNetwork(d_model, d_model * 2, dropout)
        self.lstm = nn.LSTM(d_model, d_model, lstm_layers, batch_first=True, dropout=0.0)
        self.attention = TemporalAttentionBlock(d_model, n_heads, dropout)
        self.decoder_grn = GatedResidualNetwork(d_model, d_model * 2, dropout)
        self.output = nn.Linear(d_model, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h = self.var_select(h)
        h, _ = self.lstm(h)
        h, attn_weights = self.attention(h)
        h = self.decoder_grn(h[:, -1, :])
        return self.output(h), attn_weights


def build_features(bars: pd.DataFrame) -> np.ndarray:
    """Build comprehensive feature set for TFT input."""
    closes = bars["close"].values.astype(np.float64)
    highs = bars["high"].values.astype(np.float64)
    lows = bars["low"].values.astype(np.float64)
    volumes = bars["volume"].values.astype(np.float64)
    tbv = bars.get("taker_buy_volume", bars["volume"] * 0.5).values.astype(np.float64)

    returns = np.diff(np.log(closes + 1e-10), prepend=np.log(closes[0] + 1e-10))

    vol_ratio = volumes / (pd.Series(volumes).rolling(20).mean().fillna(1).values + 1e-10)

    delta = pd.Series(closes).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = (100 - (100 / (1 + gain / (loss_s + 1e-10)))).fillna(50).values / 100.0

    tr = np.maximum(highs - lows, np.maximum(abs(highs - np.roll(closes, 1)), abs(lows - np.roll(closes, 1))))
    atr = pd.Series(tr).rolling(14).mean().fillna(0).values
    atr_norm = atr / (closes + 1e-10)

    ema9 = pd.Series(closes).ewm(span=9).mean().values
    ema21 = pd.Series(closes).ewm(span=21).mean().values
    ema50 = pd.Series(closes).ewm(span=50).mean().values
    ema_fast = (closes - ema9) / (closes + 1e-10)
    ema_med = (closes - ema21) / (closes + 1e-10)
    ema_slow = (closes - ema50) / (closes + 1e-10)

    macd = ema9 - ema21
    macd_signal = pd.Series(macd).ewm(span=9).mean().values
    macd_hist = macd - macd_signal

    bbm = pd.Series(closes).rolling(20).mean().fillna(closes[0]).values
    bbs = pd.Series(closes).rolling(20).std().fillna(1).values
    bb_pos = (closes - bbm) / (2 * bbs + 1e-10)

    tbr = tbv / (volumes + 1e-10)

    ret_5 = pd.Series(returns).rolling(5).sum().fillna(0).values
    ret_10 = pd.Series(returns).rolling(10).sum().fillna(0).values
    vol_5 = pd.Series(returns).rolling(5).std().fillna(0).values
    vol_20 = pd.Series(returns).rolling(20).std().fillna(0).values

    hour = np.zeros(len(closes))
    if "open_time" in bars.columns:
        try:
            hour = pd.to_datetime(bars["open_time"]).dt.hour.values / 24.0
        except Exception:
            pass

    features = np.column_stack([
        returns, vol_ratio, rsi, atr_norm,
        ema_fast, ema_med, ema_slow,
        macd_hist / (closes + 1e-10),
        bb_pos, tbr,
        ret_5, ret_10, vol_5, vol_20,
        hour,
    ])

    return features


def prepare_sequences(features: np.ndarray, closes: np.ndarray,
                      seq_len: int = 30, horizon: int = 5, threshold: float = 0.003):
    """Create labeled sequences for training."""
    mean = np.nanmean(features[:int(len(features)*0.7)], axis=0)
    std = np.nanstd(features[:int(len(features)*0.7)], axis=0) + 1e-8
    features_norm = np.clip((features - mean) / std, -5, 5)
    features_norm = np.nan_to_num(features_norm, nan=0.0)

    labels = np.ones(len(closes), dtype=np.int64)
    for i in range(len(closes) - horizon):
        future_ret = (closes[i + horizon] - closes[i]) / (closes[i] + 1e-10)
        if future_ret > threshold:
            labels[i] = 2
        elif future_ret < -threshold:
            labels[i] = 0

    X, y = [], []
    for i in range(seq_len, len(closes) - horizon):
        X.append(features_norm[i - seq_len : i])
        y.append(labels[i])

    return np.array(X), np.array(y), mean, std


def train_tft(X: np.ndarray, y: np.ndarray, epochs: int = 40, lr: float = 5e-4):
    """Train TFT model."""
    split = int(len(X) * 0.7)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = SimpleTFT(n_features=X.shape[2], d_model=64, n_heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    class_counts = np.bincount(y_train, minlength=3).astype(np.float32)
    weights = 1.0 / (class_counts + 1)
    weights = weights / weights.sum() * 3
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)

    best_acc = 0.0
    best_state = None
    patience = 8
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits, attn = model(torch.FloatTensor(X_val).to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            acc = (preds == y_val).mean()

            dir_correct = 0
            dir_total = 0
            for p, t in zip(preds, y_val):
                if p != 1 or t != 1:
                    dir_total += 1
                    if p == t:
                        dir_correct += 1
            dir_acc = dir_correct / max(dir_total, 1)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, loss={total_loss/len(train_dl):.4f}, "
                       f"val_acc={acc:.3f}, dir_acc={dir_acc:.3f}")

    if best_state:
        model.load_state_dict(best_state)

    return model, best_acc


def backtest_tft(
    bars: pd.DataFrame,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int = 30,
    leverage: int = 8,
    initial_capital: float = 100.0,
    commission_pct: float = 0.0004,
    slippage_pct: float = 0.0003,
) -> dict:
    """Run backtest using TFT predictions."""
    closes = bars["close"].values.astype(np.float64)
    features = build_features(bars)
    features_norm = np.clip((features - mean) / std, -5, 5)
    features_norm = np.nan_to_num(features_norm, nan=0.0)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()

    capital = initial_capital
    peak = capital
    max_dd = 0.0
    holding = None
    trades = []
    cost = commission_pct + slippage_pct

    for i in range(seq_len, len(closes) - 1):
        seq = features_norm[i - seq_len : i]

        with torch.no_grad():
            logits, _ = model(torch.FloatTensor(seq).unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            signal = logits.argmax(dim=1).item()

        confidence = max(probs)
        if confidence < 0.45:
            signal = 1

        if holding is not None:
            bars_held = i - holding["entry_bar"]
            pnl_pct = (closes[i] / holding["entry_price"] - 1) * holding["direction"] * leverage

            should_close = False
            if pnl_pct > 0.06:
                should_close = True
            elif pnl_pct < -0.025:
                should_close = True
            elif bars_held >= 48:
                should_close = True
            elif (closes[i] / holding["entry_price"] - 1) * holding["direction"] < -0.90 / leverage:
                should_close = True
            elif signal != 1 and (signal == 2) != (holding["direction"] == 1):
                should_close = True

            if should_close:
                realized = capital * abs(pnl_pct) * (1 if pnl_pct > 0 else -1) - capital * cost * 2
                capital += realized
                trades.append({"pnl_pct": pnl_pct, "direction": holding["direction"]})
                holding = None

        if holding is None and signal != 1 and confidence >= 0.50:
            direction = 1 if signal == 2 else -1
            holding = {"entry_bar": i, "entry_price": closes[i], "direction": direction}
            capital -= capital * cost

        peak = max(peak, capital)
        dd = (peak - capital) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    if holding is not None:
        i = len(closes) - 1
        pnl_pct = (closes[i] / holding["entry_price"] - 1) * holding["direction"] * leverage
        realized = capital * abs(pnl_pct) * (1 if pnl_pct > 0 else -1) - capital * cost * 2
        capital += realized
        trades.append({"pnl_pct": pnl_pct, "direction": holding["direction"]})

    total_return = (capital - initial_capital) / initial_capital * 100
    n_trades = len(trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    win_rate = len(wins) / n_trades * 100 if n_trades else 0
    losses = [t for t in trades if t["pnl_pct"] <= 0]

    pf = 999
    if losses and sum(abs(t["pnl_pct"]) for t in losses) > 0:
        pf = sum(t["pnl_pct"] for t in wins) / sum(abs(t["pnl_pct"]) for t in losses)

    returns_arr = [t["pnl_pct"] for t in trades]
    sharpe = (np.mean(returns_arr) / (np.std(returns_arr) + 1e-10)) * np.sqrt(252 / max(1, n_trades)) if returns_arr else 0

    return {
        "total_return_pct": round(total_return, 2),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "sharpe": round(sharpe, 2),
        "final_capital": round(capital, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--weeks", type=int, default=80)
    parser.add_argument("--leverage", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    loader = CryptoDataLoader()
    try:
        bars = loader.load_with_funding(args.symbol, args.timeframe)
    except Exception:
        bars = loader.load(args.symbol, args.timeframe)

    if args.weeks:
        bars = bars.tail(args.weeks * 7 * 24).reset_index(drop=True)

    logger.info(f"Loaded {len(bars)} bars for {args.symbol}")

    closes = bars["close"].values.astype(np.float64)
    features = build_features(bars)

    split = int(len(bars) * 0.7)

    logger.info("Preparing sequences for TFT training...")
    X, y, mean, std = prepare_sequences(features[:split], closes[:split])
    logger.info(f"Training: {X.shape[0]} sequences, {X.shape[2]} features/timestep")

    logger.info("Training Temporal Fusion Transformer...")
    model, acc = train_tft(X, y, epochs=args.epochs)
    logger.info(f"Best validation accuracy: {acc:.3f}")

    logger.info("Running OOS backtest...")
    test_bars = bars.iloc[split:].reset_index(drop=True)
    results = backtest_tft(test_bars, model, mean, std, leverage=args.leverage)

    logger.info(f"\n{'='*60}")
    logger.info(f"TFT DIRECTION STRATEGY — {args.symbol} (OOS)")
    logger.info(f"{'='*60}")
    for k, v in results.items():
        logger.info(f"  {k}: {v}")

    out_path = Path(__file__).resolve().parent.parent.parent / "models" / "tft_direction_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"symbol": args.symbol, "leverage": args.leverage, "accuracy": round(acc, 4), **results}, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
