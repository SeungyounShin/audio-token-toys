"""Top-level audio encoder: Frontend -> CausalEncoder -> Projector -> LLM dim.

Supports swappable frontends: 'mel' (precomputed mel) or 'raw' (raw waveform conv).
Optional CTC auxiliary head at 50Hz (before downsample) for direct speech supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.frontend.mel_frontend import MelFrontend
from src.frontend.raw_conv_frontend import RawConvFrontend
from src.model.encoder import AudioTransformerEncoder


# Character vocab for CTC: a-z(0-25), space(26), apostrophe(27), blank(28)
CTC_VOCAB = list("abcdefghijklmnopqrstuvwxyz '")
CTC_BLANK = len(CTC_VOCAB)  # 28
CTC_VOCAB_SIZE = len(CTC_VOCAB) + 1  # 29


def text_to_ctc_targets(text: str) -> list[int]:
    """Convert lowercase text to CTC character indices."""
    return [CTC_VOCAB.index(c) for c in text if c in CTC_VOCAB]


class AudioEncoder(nn.Module):
    """Audio encoder for frozen-LLM decoder architecture.

    Two modes:
      'mel': mel [B, 80, T] -> PatchEmbedding -> Transformer -> projector
      'raw': wav [B, T]     -> CausalConvStack -> Transformer -> projector

    Optional CTC head at 50Hz conv features (raw frontend only).
    """

    def __init__(
        self,
        frontend_type: str = "raw",
        # Mel frontend params
        n_mels: int = 80,
        patch_frames: int = 8,
        hop_frames: int = 8,
        # Shared params
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 500,
        dropout: float = 0.1,
        llm_dim: int = 1024,
        causal: bool = True,
        # CTC params
        use_ctc: bool = False,
        conv_dim: int = 512,
        # Use 50Hz features directly (skip 4x downsample)
        use_50hz: bool = False,
    ):
        super().__init__()
        self.frontend_type = frontend_type
        self.use_ctc = use_ctc
        self.use_50hz = use_50hz

        if frontend_type == "mel":
            self.frontend = MelFrontend(
                n_mels=n_mels,
                patch_frames=patch_frames,
                hop_frames=hop_frames,
                d_model=d_model,
            )
        elif frontend_type == "raw":
            self.frontend = RawConvFrontend(d_model=d_model)
        else:
            raise ValueError(f"Unknown frontend_type: {frontend_type}")

        # When using 50Hz features, remove unused downsample/proj/norm from frontend
        # to avoid DDP "unused parameters" error
        if use_50hz and frontend_type == "raw":
            del self.frontend.downsample
            del self.frontend.proj
            del self.frontend.norm

        # Project conv output to d_model when they differ (e.g. conv_dim=512, d_model=768)
        self.conv_proj = (
            nn.Linear(conv_dim, d_model)
            if (use_50hz and frontend_type == "raw" and n_layers > 0 and conv_dim != d_model)
            else None
        )

        self.n_layers = n_layers
        if n_layers > 0:
            self.encoder = AudioTransformerEncoder(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                dropout=dropout,
                causal=causal,
            )
        else:
            self.encoder = None

        # Projector: MLP + LayerNorm with small initial gamma.
        # LN gamma=0.03 matches LLM embedding std to prevent bf16 softmax overflow.
        # Gamma is learnable and grows as LLM adapts to audio features.
        proj_in_dim = conv_dim if (use_50hz and n_layers == 0) else d_model
        self.projector = nn.Sequential(
            nn.Linear(proj_in_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )
        with torch.no_grad():
            self.projector[-1].weight.fill_(0.03)

        # CTC head: on transformer output (n_layers>0) or conv output (n_layers=0)
        if use_ctc and frontend_type == "raw":
            ctc_dim = d_model if n_layers > 0 else conv_dim
            self.ctc_head = nn.Linear(ctc_dim, CTC_VOCAB_SIZE)

    def forward(
        self,
        audio: torch.Tensor,
        n_lengths: torch.Tensor = None,
        texts: list[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            audio: [B, 80, T] mel or [B, T] waveform depending on frontend_type.
            n_lengths: [B] actual frame counts (mel) or sample counts (raw).
            texts: list of transcriptions (for CTC loss, optional).

        Returns:
            audio_features: [B, N, llm_dim]
            n_tokens: [B] valid audio token counts (or None)
            ctc_loss: scalar CTC loss (or None if not using CTC)
        """
        if self.use_50hz and self.frontend_type == "raw":
            # Run only conv_stack for 50Hz features (downsample removed)
            x = audio.unsqueeze(1)  # [B, 1, T]
            x = self.frontend.conv_stack(x)  # [B, conv_dim, T_50hz]
            self.frontend._conv50hz = x  # cache for CTC
            conv_out = x.transpose(1, 2)  # [B, T_50hz, conv_dim]
            if self.conv_proj is not None:
                conv_out = self.conv_proj(conv_out)  # [B, T_50hz, d_model]
            B, N, _ = conv_out.shape

            n_tokens = None
            pad_mask = None
            if n_lengths is not None:
                n_tokens = self.frontend.num_tokens_50hz(n_lengths)
                n_tokens = n_tokens.clamp(max=N)
                pad_mask = (
                    torch.arange(N, device=conv_out.device).unsqueeze(0)
                    >= n_tokens.unsqueeze(1)
                )

            if self.encoder is not None:
                encoded = self.encoder(conv_out, src_key_padding_mask=pad_mask)
            else:
                encoded = conv_out
        else:
            tokens = self.frontend(audio)  # [B, N, d_model] at 12.5Hz
            B, N, _ = tokens.shape

            pad_mask = None
            n_tokens = None
            if n_lengths is not None:
                n_tokens = self.frontend.num_tokens_batch(n_lengths)
                n_tokens = n_tokens.clamp(max=N)
                pad_mask = (
                    torch.arange(N, device=tokens.device).unsqueeze(0)
                    >= n_tokens.unsqueeze(1)
                )

            if self.encoder is not None:
                encoded = self.encoder(tokens, src_key_padding_mask=pad_mask)
            else:
                encoded = tokens

        self._encoded = encoded  # cache for eval (CTC decode, feature analysis)
        audio_features = self.projector(encoded)

        # CTC loss on pre-projector features (computed inside forward for DDP).
        # When n_layers>0, CTC on transformer output → conv+transformer both get CTC grad.
        # When n_layers=0, CTC on conv output (same as before).
        ctc_loss = None
        if self.use_ctc and texts is not None and self.frontend_type == "raw":
            # Filter to samples with non-empty CTC targets (non-Latin scripts → empty targets)
            valid_idx = [i for i, t in enumerate(texts) if len(text_to_ctc_targets(t)) > 0]

            if valid_idx:
                valid_idx_t = torch.tensor(valid_idx, device=encoded.device)
                ctc_input = encoded[valid_idx_t].float()  # [V, T_50hz, dim]

                logits = self.ctc_head(ctc_input)  # [V, T_50hz, vocab]
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # [T, V, vocab]

                valid_lengths = n_lengths[valid_idx_t]
                input_lengths = self.frontend.num_tokens_50hz(valid_lengths)
                input_lengths = input_lengths.clamp(max=ctc_input.shape[1])

                targets = []
                target_lengths = []
                for t in [texts[i] for i in valid_idx]:
                    ids = text_to_ctc_targets(t)
                    targets.extend(ids)
                    target_lengths.append(len(ids))

                targets_t = torch.tensor(targets, dtype=torch.long, device=log_probs.device)
                target_lengths_t = torch.tensor(
                    target_lengths, dtype=torch.long, device=log_probs.device
                )

                ctc_loss = F.ctc_loss(
                    log_probs, targets_t, input_lengths, target_lengths_t,
                    blank=CTC_BLANK, zero_infinity=True,
                )

        return audio_features, n_tokens, ctc_loss

    @classmethod
    def from_config(cls, cfg) -> "AudioEncoder":
        frontend_type = getattr(cfg, "frontend_type", "raw")
        use_ctc = getattr(cfg, "use_ctc", False)
        use_50hz = getattr(cfg, "use_50hz", False)
        return cls(
            frontend_type=frontend_type,
            n_mels=cfg.n_mels,
            patch_frames=cfg.patch_frames,
            hop_frames=cfg.hop_frames,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout,
            llm_dim=cfg.llm_dim,
            causal=cfg.causal,
            use_ctc=use_ctc,
            use_50hz=use_50hz,
        )
