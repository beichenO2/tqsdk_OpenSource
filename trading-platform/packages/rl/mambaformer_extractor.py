"""MambaFormer Feature Extractor for SB3 PPO.

Hybrid architecture combining:
- S4D-style State Space Model (Mamba-like) for long-range dependencies
- Transformer self-attention for short-range pattern capture

Based on CrossMamba (FinTech 2026) and CryptoMamba (arXiv 2025.01).
Pure PyTorch implementation — no mamba-ssm dependency required.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import torch
import torch.nn as nn

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    _SB3_AVAILABLE = True
except ImportError:
    BaseFeaturesExtractor = nn.Module  # type: ignore[misc, assignment]
    _SB3_AVAILABLE = False


class S4DBlock(nn.Module):
    """Simplified S4D (Structured State Space for Sequences, Diagonal) block.

    Implements the diagonal SSM recurrence in the frequency domain for
    efficiency.  At inference the recurrence can also run step-by-step.
    """

    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.ones(d_model))

        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.  x: (batch, seq_len, d_model)."""
        residual = x
        x = self.norm(x)
        x = self.in_proj(x)

        y = self._ssm_scan(x)

        y = y * torch.sigmoid(x)

        y = self.out_proj(y)
        y = self.dropout(y)
        return y + residual

    def _ssm_scan(self, x: torch.Tensor) -> torch.Tensor:
        """SSM scan with chunked processing for better GPU utilisation."""
        B, L, D = x.shape
        A = -torch.exp(self.A_log)

        dt = torch.sigmoid(x.mean(dim=-1, keepdim=True))
        dA = torch.exp(A.unsqueeze(0) * dt.unsqueeze(-1))
        dB = dt.unsqueeze(-1) * self.B.unsqueeze(0).unsqueeze(0)
        Bu = dB * x.unsqueeze(-1)

        chunk_size = min(L, 8)
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        y_chunks: list[torch.Tensor] = []

        for c_start in range(0, L, chunk_size):
            c_end = min(c_start + chunk_size, L)
            chunk_out = []
            for t in range(c_start, c_end):
                h = dA[:, t] * h + Bu[:, t]
                chunk_out.append((h * self.C.unsqueeze(0)).sum(dim=-1))
            y_chunks.append(torch.stack(chunk_out, dim=1))

        y = torch.cat(y_chunks, dim=1)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer encoder block with causal masking."""

    def __init__(
        self, d_model: int, nhead: int = 4, dim_ff: int = 0, dropout: float = 0.1
    ) -> None:
        super().__init__()
        dim_ff = dim_ff or d_model * 4
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=x.device, dtype=x.dtype
        )
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class MambaFormerExtractor(BaseFeaturesExtractor):
    """SB3-compatible feature extractor: S4D (Mamba-like) + Transformer.

    The flat observation vector is reshaped into a sequence of K-line bars.
    Each bar is projected into ``d_model`` dimensions, then processed by
    alternating S4D and Transformer blocks.  The final hidden state (last
    timestep) serves as the feature vector for the PPO policy/value heads.

    Parameters
    ----------
    observation_space
        Gymnasium Box space (flat vector).
    d_model
        Hidden dimension.  Default 64.
    n_s4d_layers
        Number of S4D (Mamba-like) layers.  Default 2.
    n_attn_layers
        Number of Transformer attention layers.  Default 1.
    nhead
        Attention heads.  Default 4.
    d_state
        SSM state dimension.  Default 16.
    features_per_bar
        Number of raw features per bar in the obs vector (OHLCV=5).
    extra_features
        Number of extra features appended after the bar sequence
        (position, balance, hold_duration, unrealized, TA indicators).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        d_model: int = 64,
        n_s4d_layers: int = 2,
        n_attn_layers: int = 1,
        nhead: int = 4,
        d_state: int = 16,
        features_per_bar: int = 5,
        extra_features: int = 16,
        dropout: float = 0.1,
    ) -> None:
        obs_dim = int(observation_space.shape[0])  # type: ignore[index]
        seq_features = obs_dim - extra_features
        if seq_features <= 0 or seq_features % features_per_bar != 0:
            raise ValueError(
                f"obs_dim={obs_dim}, extra={extra_features}, "
                f"seq_features={seq_features} not divisible by {features_per_bar}"
            )
        seq_len = seq_features // features_per_bar

        output_dim = d_model + extra_features

        # latent_dim_* MUST be set before super().__init__() because
        # BasePolicy._build_mlp_extractor() reads them to size policy/value nets.
        self.latent_dim_pi = output_dim
        self.latent_dim_vf = output_dim

        if _SB3_AVAILABLE:
            super().__init__(observation_space, features_dim=output_dim)
        else:
            super().__init__()

        self._seq_len = seq_len
        self._features_per_bar = features_per_bar
        self._extra_features = extra_features
        self._features_dim = output_dim

        self.bar_proj = nn.Sequential(
            nn.Linear(features_per_bar, d_model),
            nn.GELU(),
        )

        self.pos_embed = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02
        )

        layers: list[nn.Module] = []
        for _ in range(n_s4d_layers):
            layers.append(S4DBlock(d_model, d_state, dropout))
        for _ in range(n_attn_layers):
            layers.append(TransformerBlock(d_model, nhead, dropout=dropout))
        self.encoder = nn.Sequential(*layers)

        self.final_norm = nn.LayerNorm(d_model)

        self._target_device: str | None = None

    def _get_device(self) -> torch.device:
        if self._target_device:
            return torch.device(self._target_device)
        if torch.backends.mps.is_available():
            self._target_device = "mps"
        elif torch.cuda.is_available():
            self._target_device = "cuda"
        else:
            self._target_device = "cpu"
        return torch.device(self._target_device)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, **kwargs) -> "MambaFormerExtractor":
        """Load a MambaFormerExtractor from a torch checkpoint or SB3 zip.

        Supports two formats:
        1. Raw torch checkpoint: contains ``observation_space`` and
           ``state_dict`` keys (saved via ``BaseFeaturesExtractor.save()``).
        2. SB3 model zip: contains ``policy.pth`` with the full
           PPO model state dict.  The feature extractor parameters are
           extracted from the ``features_extractor.*`` keys.
        """
        import collections, io, json, zipfile

        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "observation_space" in ckpt and "state_dict" in ckpt:
                obs_space = ckpt["observation_space"]
                extractor = cls(observation_space=obs_space, **kwargs)
                extractor.load_state_dict(ckpt["state_dict"])
                return extractor
        except (RuntimeError, TypeError, AttributeError):
            pass

        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            data_json = json.loads(zf.read("data"))
            policy_state = torch.load(
                io.BytesIO(zf.read("policy.pth")),
                map_location="cpu",
                weights_only=False,
            )

        obs_shape = data_json.get("observation_space", {}).get("shape", [kwargs.get("obs_dim", 166)])
        import gymnasium as gym
        obs_space = gym.spaces.Box(-1.0, 1.0, shape=obs_shape, dtype="float32")

        extractor_sd: dict = {}
        prefix = "features_extractor."
        prefix_len = len(prefix)
        for key, val in policy_state.items():
            if key.startswith(prefix):
                extractor_sd[key[prefix_len:]] = val

        if not extractor_sd:
            raise ValueError(
                f"No keys starting with '{prefix}' found in policy.pth of {checkpoint_path}"
            )

        extractor = cls(observation_space=obs_space, **kwargs)
        extractor.load_state_dict(extractor_sd)
        return extractor

    def save(self, path: str) -> str:
        """Save extractor to a torch checkpoint (SB3-compatible format).

        Saves the state dict with ``features_extractor.`` prefix so it matches
        the format used by ``policy.pth`` inside SB3 model zips.
        """
        import gymnasium as gym
        prefixed_sd = {"features_extractor." + k: v for k, v in self.state_dict().items()}
        torch.save(
            {"observation_space": gym.spaces.box.Box(-1, 1, shape=self.observation_space.shape),  # type: ignore[arg-type]
             "state_dict": prefixed_sd},
            path,
        )
        return path

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass — returns (latent_pi, latent_vf) for SB3 compatibility.

        Both tensors are identical since the MambaFormer shares feature extraction
        between policy and value networks.
        """
        batch = observations.shape[0]

        seq_end = self._seq_len * self._features_per_bar
        seq_flat = observations[:, :seq_end]
        extra = observations[:, seq_end:]

        seq = seq_flat.view(batch, self._seq_len, self._features_per_bar)
        h = self.bar_proj(seq) + self.pos_embed

        h = self.encoder(h)
        h = self.final_norm(h)

        last_hidden = h[:, -1, :]
        features = torch.cat([last_hidden, extra], dim=-1)
        # Return identical features for pi and vf (shared extractor)
        return features, features
