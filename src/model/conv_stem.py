"""Patch embedding layer: unfold mel spectrogram into patches + linear projection."""

import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """ViT-style patch embedding for mel spectrograms.

    Unfolds the mel along the time axis into fixed-size patches, then projects
    each patch to d_model dimensions. Overlap is controlled by hop_frames.

    Args:
        n_mels: Number of mel frequency bins (default 80).
        patch_frames: Temporal frames per patch (4 = 40ms, 8 = 80ms at 10ms hop).
        hop_frames: Stride between patches (== patch_frames for 0% overlap,
                    patch_frames//2 for 50% overlap).
        d_model: Output embedding dimension.
    """

    def __init__(
        self,
        n_mels: int = 80,
        patch_frames: int = 4,
        hop_frames: int = 4,
        d_model: int = 512,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.patch_frames = patch_frames
        self.hop_frames = hop_frames
        self.patch_dim = n_mels * patch_frames
        self.d_model = d_model

        self.projection = nn.Linear(self.patch_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def num_tokens(self, n_frames: int) -> int:
        """Number of output tokens for a given input length."""
        if n_frames < self.patch_frames:
            return 0
        return (n_frames - self.patch_frames) // self.hop_frames + 1

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, n_mels, T] mel spectrogram.
        Returns:
            tokens: [B, N, d_model] patch embeddings.
        """
        # unfold: [B, n_mels, T] -> [B, n_mels, N, patch_frames]
        patches = mel.unfold(2, self.patch_frames, self.hop_frames)
        B, F, N, P = patches.shape
        # -> [B, N, n_mels * patch_frames]
        patches = patches.permute(0, 2, 1, 3).reshape(B, N, F * P)
        # project + normalize
        return self.layer_norm(self.projection(patches))


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding (not learned)."""

    def __init__(self, d_model: int, max_len: int = 4000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: [B, N, d_model]."""
        return x + self.pe[:, : x.shape[1]]
