"""Wavelet-LSTM Trading Strategy — Cross-domain signal processing + deep learning.

Decomposes price into trend (low-freq) and noise (high-freq) via DWT,
trains a small LSTM on wavelet features, generates directional signals.

Usage:
    python scripts/research/wavelet_lstm_strategy.py [--symbol BTCUSDT] [--weeks 80] [--leverage 8]
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from datahub.crypto_loader import CryptoDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import pywt
except ImportError:
    logger.error("pywt required: pip install PyWavelets")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    logger.error("torch required: pip install torch")
    sys.exit(1)


class WaveletFeatureExtractor:
    """Extract multi-scale features using Discrete Wavelet Transform."""

    def __init__(self, wavelet: str = "db6", level: int = 3):
        self.wavelet = wavelet
        self.level = level

    def transform(self, series: np.ndarray, window: int = 60) -> np.ndarray:
        """Sliding-window DWT: for each bar, decompose the last `window` bars.

        Returns array of shape (N, n_features) where n_features = level+1 (approx + details).
        Each feature is the energy/std of that decomposition level.
        """
        n = len(series)
        n_features = self.level + 1
        features = np.zeros((n, n_features + 3))  # +3 for trend slope, denoised diff, noise ratio

        for i in range(window, n):
            segment = series[i - window : i]
            try:
                coeffs = pywt.wavedec(segment, self.wavelet, level=self.level)
            except Exception:
                continue

            for j, c in enumerate(coeffs):
                features[i, j] = np.std(c) if len(c) > 0 else 0.0

            approx = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], self.wavelet)
            approx = approx[: len(segment)]
            if len(approx) >= 2:
                features[i, n_features] = (approx[-1] - approx[-5]) / (abs(approx[-5]) + 1e-10)
            features[i, n_features + 1] = (approx[-1] - segment[-1]) / (abs(segment[-1]) + 1e-10)

            detail_energy = sum(np.sum(c ** 2) for c in coeffs[1:])
            total_energy = sum(np.sum(c ** 2) for c in coeffs)
            features[i, n_features + 2] = detail_energy / (total_energy + 1e-10)

        return features


class SmallLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3 classes: down, neutral, up
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


def prepare_data(bars: pd.DataFrame, seq_len: int = 20, wavelet_window: int = 60):
    """Build wavelet features + OHLCV features, create sequences for LSTM."""
    closes = bars["close"].values.astype(np.float64)
    returns = np.diff(np.log(closes + 1e-10), prepend=np.log(closes[0] + 1e-10))
    volumes = bars["volume"].values.astype(np.float64)

    wfe = WaveletFeatureExtractor()
    wavelet_feats = wfe.transform(closes, window=wavelet_window)

    vol_norm = volumes / (pd.Series(volumes).rolling(20).mean().fillna(1).values + 1e-10)

    rsi_period = 14
    delta = pd.Series(closes).diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rsi = (100 - (100 / (1 + gain / (loss + 1e-10)))).fillna(50).values / 100.0

    all_features = np.column_stack([
        returns,
        vol_norm,
        rsi,
        wavelet_feats,
    ])

    threshold = 0.003
    labels = np.zeros(len(closes), dtype=np.int64)
    for i in range(len(closes) - 1):
        future_ret = (closes[min(i + 5, len(closes) - 1)] - closes[i]) / (closes[i] + 1e-10)
        if future_ret > threshold:
            labels[i] = 2  # up
        elif future_ret < -threshold:
            labels[i] = 0  # down
        else:
            labels[i] = 1  # neutral

    valid_start = wavelet_window + seq_len
    X_seqs, y_seqs = [], []
    for i in range(valid_start, len(closes) - 5):
        X_seqs.append(all_features[i - seq_len : i])
        y_seqs.append(labels[i])

    return np.array(X_seqs), np.array(y_seqs), valid_start


def train_model(X: np.ndarray, y: np.ndarray, epochs: int = 30, lr: float = 1e-3):
    """Train LSTM and return model + accuracy."""
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SmallLSTM(input_size=X.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_test).to(device)).argmax(dim=1).cpu().numpy()
            acc = (preds == y_test).mean()
            if acc > best_acc:
                best_acc = acc

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, test acc={acc:.3f}")

    return model, mean, std, best_acc


def backtest_wavelet_lstm(
    bars: pd.DataFrame,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int = 20,
    wavelet_window: int = 60,
    leverage: int = 8,
    initial_capital: float = 100.0,
    commission_pct: float = 0.0004,
    slippage_pct: float = 0.0003,
) -> dict:
    """Run trading backtest using LSTM predictions."""
    closes = bars["close"].values.astype(np.float64)
    returns = np.diff(np.log(closes + 1e-10), prepend=np.log(closes[0] + 1e-10))
    volumes = bars["volume"].values.astype(np.float64)

    wfe = WaveletFeatureExtractor()
    wavelet_feats = wfe.transform(closes, window=wavelet_window)

    vol_norm = volumes / (pd.Series(volumes).rolling(20).mean().fillna(1).values + 1e-10)
    delta = pd.Series(closes).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = (100 - (100 / (1 + gain / (loss_s + 1e-10)))).fillna(50).values / 100.0

    all_features = np.column_stack([returns, vol_norm, rsi, wavelet_feats])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()

    capital = initial_capital
    peak = capital
    max_dd = 0.0
    holding = None
    trades = []
    cost = commission_pct + slippage_pct
    valid_start = wavelet_window + seq_len

    for i in range(valid_start, len(closes) - 1):
        seq = all_features[i - seq_len : i]
        seq_norm = (seq - mean) / std

        with torch.no_grad():
            pred = model(torch.FloatTensor(seq_norm).unsqueeze(0).to(device))
            signal = pred.argmax(dim=1).item()  # 0=down, 1=neutral, 2=up
            probs = torch.softmax(pred, dim=1).cpu().numpy()[0]

        confidence = max(probs)
        if confidence < 0.45:
            signal = 1  # neutral if low confidence

        if holding is not None:
            bars_held = i - holding["entry_bar"]
            pnl_pct = (closes[i] / holding["entry_price"] - 1) * holding["direction"] * leverage
            liq_threshold = -0.90 / leverage

            should_close = False
            if pnl_pct > 0.05:  # take profit at 5% (leveraged)
                should_close = True
            elif pnl_pct < -0.02:  # stop loss at 2% (leveraged)
                should_close = True
            elif bars_held >= 48:
                should_close = True
            elif (closes[i] / holding["entry_price"] - 1) * holding["direction"] < liq_threshold:
                should_close = True

            if should_close:
                realized = capital * abs(pnl_pct) * (1 if pnl_pct > 0 else -1) - capital * cost * 2
                capital += realized
                trades.append({
                    "entry_bar": holding["entry_bar"],
                    "exit_bar": i,
                    "direction": holding["direction"],
                    "pnl_pct": pnl_pct,
                    "realized": realized,
                })
                holding = None

        if holding is None and signal != 1:
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
        trades.append({
            "entry_bar": holding["entry_bar"],
            "exit_bar": i,
            "direction": holding["direction"],
            "pnl_pct": pnl_pct,
            "realized": realized,
        })

    total_return = (capital - initial_capital) / initial_capital * 100
    n_trades = len(trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0

    avg_win = np.mean([t["pnl_pct"] for t in wins]) * 100 if wins else 0
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    avg_loss = np.mean([abs(t["pnl_pct"]) for t in losses]) * 100 if losses else 0
    pf = (sum(t["pnl_pct"] for t in wins) / abs(sum(t["pnl_pct"] for t in losses))) if losses and sum(t["pnl_pct"] for t in losses) != 0 else 999

    returns_arr = [t["pnl_pct"] for t in trades]
    sharpe = (np.mean(returns_arr) / (np.std(returns_arr) + 1e-10)) * np.sqrt(252 / max(1, n_trades)) if returns_arr else 0

    return {
        "total_return_pct": round(total_return, 2),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "sharpe": round(sharpe, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "final_capital": round(capital, 2),
    }


def main():

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="ml-training", command="wavelet_lstm_strategy.py", requester="wavelet-lstm", estimated_duration_sec=3600)
            _task_id = _tr.get("task_id")
        except Exception:
            pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--weeks", type=int, default=80)
    parser.add_argument("--leverage", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    loader = CryptoDataLoader()
    try:
        bars = loader.load_with_funding(args.symbol, args.timeframe)
    except Exception:
        bars = loader.load(args.symbol, args.timeframe)

    if args.weeks:
        bars = bars.tail(args.weeks * 7 * 24).reset_index(drop=True)

    logger.info(f"Loaded {len(bars)} bars for {args.symbol} {args.timeframe}")

    split_idx = int(len(bars) * 0.7)
    train_bars = bars.iloc[:split_idx]
    test_bars = bars.iloc[split_idx:]

    logger.info("Preparing wavelet features and training LSTM...")
    X, y, _ = prepare_data(train_bars)
    logger.info(f"Training data: {X.shape[0]} sequences, {X.shape[2]} features")

    model, mean, std, acc = train_model(X, y, epochs=args.epochs)
    logger.info(f"Training complete. Test accuracy: {acc:.3f}")

    logger.info("Running backtest on out-of-sample data...")
    results = backtest_wavelet_lstm(test_bars, model, mean, std, leverage=args.leverage)

    logger.info(f"\n{'='*60}")
    logger.info(f"WAVELET-LSTM STRATEGY — {args.symbol} (OOS)")
    logger.info(f"{'='*60}")
    for k, v in results.items():
        logger.info(f"  {k}: {v}")

    out_path = Path(__file__).resolve().parent.parent.parent / "models" / "wavelet_lstm_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"symbol": args.symbol, "leverage": args.leverage, "accuracy": round(acc, 4), **results}, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    return results


    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    main()
