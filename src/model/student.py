"""Full student audio encoder: PatchEmbedding + Transformer + Projection."""

import torch
import torch.nn as nn
from .conv_stem import PatchEmbedding
from .encoder import AudioTransformerEncoder


class ProjectionHead(nn.Module):
    """2-layer MLP to map student representations to teacher space."""

    def __init__(self, student_dim: int = 512, teacher_dim: int = 512, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, teacher_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StudentAudioEncoder(nn.Module):
    """Student encoder for distillation from Whisper.

    mel [B, 80, T] → PatchEmbedding → Transformer → encoded [B, N, d_model]
                                                   → projected [B, N, teacher_dim]
    """

    def __init__(
        self,
        n_mels: int = 80,
        patch_frames: int = 4,
        hop_frames: int = 4,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1500,
        dropout: float = 0.1,
        teacher_dim: int = 512,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            n_mels=n_mels,
            patch_frames=patch_frames,
            hop_frames=hop_frames,
            d_model=d_model,
        )
        self.encoder = AudioTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.projection = ProjectionHead(
            student_dim=d_model,
            teacher_dim=teacher_dim,
        )

    def forward(self, mel: torch.Tensor, n_frames: torch.Tensor = None):
        """
        Args:
            mel: [B, 80, T] mel spectrogram.
            n_frames: [B] actual mel frame count per sample (for masking).
        Returns:
            encoded: [B, N, d_model]
            projected: [B, N, teacher_dim]
            n_tokens: [B] valid token counts (or None)
        """
        tokens = self.patch_embed(mel)  # [B, N, d_model]
        B, N, _ = tokens.shape

        # Build padding mask from frame lengths
        mask = None
        n_tokens = None
        if n_frames is not None:
            pe = self.patch_embed
            n_tokens = (n_frames - pe.patch_frames) // pe.hop_frames + 1
            n_tokens = n_tokens.clamp(min=1, max=N)
            mask = torch.arange(N, device=tokens.device)[None, :] >= n_tokens[:, None]

        encoded = self.encoder(tokens, src_key_padding_mask=mask)
        projected = self.projection(encoded)

        return encoded, projected, n_tokens

    @classmethod
    def from_config(cls, cfg) -> "StudentAudioEncoder":
        return cls(
            n_mels=cfg.n_mels,
            patch_frames=cfg.patch_frames,
            hop_frames=cfg.hop_frames,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout,
            teacher_dim=cfg.teacher_dim,
        )
