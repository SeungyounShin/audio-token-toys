"""Raw waveform frontend: causal conv stack → 12.5Hz tokens.

wav2vec2-style 7-layer causal conv feature encoder (16kHz → 50Hz)
followed by 2 stride-2 causal conv layers for 4x downsample (50Hz → 12.5Hz).

Total stride: 5*2*2*2*2*2*2 * 2*2 = 1280 samples = 80ms @ 16kHz = 12.5Hz output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Conv1d with left-only padding for causal (streaming) operation."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1,
                 groups: int = 1):
        super().__init__()
        self.pad = kernel_size - stride
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=0, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))
        return self.conv(x)


# (out_channels, kernel_size, stride)
# Total stride = 5*2*2*2*2*2*2 = 320 → 50Hz from 16kHz
CONV_STACK_CONFIG = [
    (512, 10, 5),   # layer 0: 16kHz → 3200Hz
    (512,  3, 2),   # layer 1: → 1600Hz
    (512,  3, 2),   # layer 2: → 800Hz
    (512,  3, 2),   # layer 3: → 400Hz
    (512,  3, 2),   # layer 4: → 200Hz
    (512,  2, 2),   # layer 5: → 100Hz
    (512,  2, 2),   # layer 6: → 50Hz
]


class RawConvFrontend(nn.Module):
    """Causal conv feature encoder for raw 16kHz waveforms.

    Input:  waveform [B, T]  (16kHz raw audio)
    Output: tokens [B, N, d_model]  at 12.5Hz

    Architecture:
        7-layer causal conv stack (stride 320 total → 50Hz)
        → 2 stride-2 causal conv layers (4x downsample → 12.5Hz)
        → Linear projection to d_model

    Total stride: 1280 samples = 80ms @ 16kHz.
    """

    def __init__(
        self,
        d_model: int = 512,
        conv_dim: int = 512,
        conv_config: list = None,
    ):
        super().__init__()
        if conv_config is None:
            conv_config = CONV_STACK_CONFIG

        # 7-layer conv feature encoder (50Hz output)
        layers = []
        in_ch = 1
        for i, (out_ch, k, s) in enumerate(conv_config):
            layers.append(CausalConv1d(in_ch, out_ch, k, stride=s))
            if i == 0:
                layers.append(nn.GroupNorm(out_ch, out_ch))
            layers.append(nn.GELU())
            in_ch = out_ch
        self.conv_stack = nn.Sequential(*layers)
        self._conv_stride = 1
        for _, _, s in conv_config:
            self._conv_stride *= s  # = 320

        # 4x temporal downsample: 50Hz → 12.5Hz
        self.downsample = nn.Sequential(
            CausalConv1d(conv_dim, conv_dim, kernel_size=3, stride=2),
            nn.GELU(),
            CausalConv1d(conv_dim, conv_dim, kernel_size=3, stride=2),
            nn.GELU(),
        )
        self._total_stride = self._conv_stride * 4  # = 1280

        # Project to d_model if different from conv_dim
        self.proj = nn.Linear(conv_dim, d_model) if conv_dim != d_model else nn.Identity()
        self.norm = nn.LayerNorm(d_model)

    @property
    def total_stride(self) -> int:
        """Total stride in samples. 1280 = 80ms at 16kHz."""
        return self._total_stride

    def num_tokens(self, n_samples: int) -> int:
        """Number of output tokens for a given input waveform length."""
        # After conv stack
        n = n_samples
        for mod in self.conv_stack:
            if isinstance(mod, CausalConv1d):
                n = (n + mod.pad) // mod.conv.stride[0]
        # After downsample (2 stride-2 layers)
        for mod in self.downsample:
            if isinstance(mod, CausalConv1d):
                n = (n + mod.pad) // mod.conv.stride[0]
        return n

    def num_tokens_batch(self, n_samples: torch.Tensor) -> torch.Tensor:
        """Vectorized: approximate token count from sample counts."""
        # Approximate: n_tokens ≈ n_samples / total_stride
        # More precise: account for causal padding
        n = n_samples.float()
        for mod in self.conv_stack:
            if isinstance(mod, CausalConv1d):
                n = (n + mod.pad) / mod.conv.stride[0]
        for mod in self.downsample:
            if isinstance(mod, CausalConv1d):
                n = (n + mod.pad) / mod.conv.stride[0]
        return n.long().clamp(min=1)

    def num_tokens_50hz(self, n_samples: torch.Tensor) -> torch.Tensor:
        """Number of 50Hz tokens (before downsample) from sample counts."""
        n = n_samples.float()
        for mod in self.conv_stack:
            if isinstance(mod, CausalConv1d):
                n = (n + mod.pad) / mod.conv.stride[0]
        return n.long().clamp(min=1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: [B, T] raw waveform at 16kHz.
        Returns:
            tokens: [B, N, d_model] at 12.5Hz.
        """
        x = wav.unsqueeze(1)  # [B, 1, T]
        x = self.conv_stack(x)  # [B, conv_dim, T/320]
        self._conv50hz = x  # cache 50Hz features for CTC
        x = self.downsample(x)  # [B, conv_dim, T/1280]
        x = x.transpose(1, 2)  # [B, N, conv_dim]
        x = self.proj(x)  # [B, N, d_model]
        return self.norm(x)
