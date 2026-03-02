"""Frozen Whisper encoder for distillation targets."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel


class TeacherWhisperEncoder(nn.Module):
    """Wraps a frozen Whisper encoder. Always pads mel to 30s (3000 frames).

    Output: [B, 1500, hidden_dim] (Whisper-base: hidden_dim=512)
    """

    def __init__(self, model_name: str = "openai/whisper-base"):
        super().__init__()
        whisper = WhisperModel.from_pretrained(model_name)
        self.encoder = whisper.encoder
        self.hidden_dim = whisper.config.d_model  # 512 for base

        # Freeze everything
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def train(self, mode=True):
        # Always stay in eval mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, 80, T] mel spectrogram (any length).
        Returns:
            [B, 1500, hidden_dim] encoder last hidden state.
        """
        # Pad to 3000 frames (Whisper's fixed 30s input)
        if mel.shape[-1] < 3000:
            mel = F.pad(mel, (0, 3000 - mel.shape[-1]))
        else:
            mel = mel[..., :3000]

        out = self.encoder(mel)
        return out.last_hidden_state  # [B, 1500, 512]
