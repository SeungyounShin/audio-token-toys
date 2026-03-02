"""Transformer encoder for audio tokens."""

import torch
import torch.nn as nn
from .conv_stem import SinusoidalPE


class AudioTransformerEncoder(nn.Module):
    """Pre-norm Transformer encoder.

    Args:
        d_model: Hidden dimension (512).
        n_heads: Attention heads (8).
        n_layers: Transformer layers (6).
        d_ff: FFN inner dimension (2048).
        max_seq_len: Maximum sequence length for positional encoding.
        dropout: Dropout rate.
        causal: If True, apply causal (autoregressive) attention mask.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1500,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal
        self.n_heads = n_heads
        self.pos_enc = SinusoidalPE(d_model, max_len=max_seq_len + 100)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, d_model] token embeddings.
            src_key_padding_mask: [B, N] bool, True = padding.
        Returns:
            [B, N, d_model]
        """
        x = self.pos_enc(x)

        if self.causal:
            N = x.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                N, device=x.device
            ).to(x.dtype)

            if src_key_padding_mask is not None:
                # Float padding: 0 for real, -inf for padding [B, N]
                pad_float = torch.zeros_like(
                    src_key_padding_mask, dtype=x.dtype
                ).masked_fill_(src_key_padding_mask, float("-inf"))
                # Broadcast: [N, N] + [B, 1, N] -> [B, N, N]
                combined = causal_mask.unsqueeze(0) + pad_float.unsqueeze(1)
                # Expand to [B * n_heads, N, N] for MHA
                combined = combined.repeat_interleave(self.n_heads, dim=0)
                x = self.transformer(x, mask=combined, is_causal=False)
            else:
                x = self.transformer(x, mask=causal_mask, is_causal=True)
        else:
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        return self.final_norm(x)
