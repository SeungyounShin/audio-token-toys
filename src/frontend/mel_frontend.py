"""Mel-based frontend: wraps PatchEmbedding as a standard Frontend module.

Phase 1: reuses existing precomputed mel .pt files.
Produces 12.5Hz tokens via patch_frames=8, hop_frames=8 (80ms patches).
"""

import torch
import torch.nn as nn
from src.model.conv_stem import PatchEmbedding


class MelFrontend(nn.Module):
    """Frontend that patchifies a precomputed mel spectrogram.

    Input:  mel [B, 80, T]  (Whisper-style 10ms-hop mel)
    Output: tokens [B, N, d_model]  at 12.5Hz (80ms per token)
    """

    def __init__(
        self,
        n_mels: int = 80,
        patch_frames: int = 8,
        hop_frames: int = 8,
        d_model: int = 512,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            n_mels=n_mels,
            patch_frames=patch_frames,
            hop_frames=hop_frames,
            d_model=d_model,
        )

    @property
    def patch_frames(self) -> int:
        return self.patch_embed.patch_frames

    @property
    def hop_frames(self) -> int:
        return self.patch_embed.hop_frames

    def num_tokens(self, n_frames: int) -> int:
        return self.patch_embed.num_tokens(n_frames)

    def num_tokens_batch(self, n_frames: torch.Tensor) -> torch.Tensor:
        """Vectorized num_tokens for a batch. Returns [B] LongTensor."""
        pe = self.patch_embed
        n = (n_frames - pe.patch_frames) // pe.hop_frames + 1
        return n.clamp(min=1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, n_mels, T]
        Returns:
            tokens: [B, N, d_model]
        """
        return self.patch_embed(mel)
