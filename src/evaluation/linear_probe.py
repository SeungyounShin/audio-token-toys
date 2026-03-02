"""Linear CTC probe: freeze encoder, train a single linear head for ASR."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearCTCProbe(nn.Module):
    """Single linear layer on top of frozen encoder for CTC-based ASR.

    Vocab: 29 = blank(0) + a-z(1-26) + space(27) + apostrophe(28)
    """

    VOCAB = ["<blank>"] + list("abcdefghijklmnopqrstuvwxyz") + [" ", "'"]

    def __init__(self, encoder: nn.Module, d_model: int = 512, vocab_size: int = 29):
        super().__init__()
        self.encoder = encoder
        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.head = nn.Linear(d_model, vocab_size)
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, mel: torch.Tensor, n_frames: torch.Tensor):
        """
        Returns:
            log_probs: [B, N, vocab_size]
            n_tokens: [B]
        """
        with torch.no_grad():
            encoded, _, n_tokens = self.encoder(mel, n_frames)
        logits = self.head(encoded)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, n_tokens

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        n_tokens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """CTC loss. log_probs: [B, T, V], needs [T, B, V] for CTCLoss."""
        return self.ctc_loss(
            log_probs.permute(1, 0, 2),  # [T, B, V]
            targets,
            n_tokens,
            target_lengths,
        )

    @staticmethod
    def text_to_targets(texts: list) -> tuple:
        """Convert text strings to CTC target tensors.

        Returns:
            targets: [sum(lengths)] flat target tensor
            target_lengths: [B]
        """
        char_to_idx = {c: i for i, c in enumerate(LinearCTCProbe.VOCAB)}
        all_targets = []
        lengths = []
        for text in texts:
            ids = [char_to_idx.get(c, 0) for c in text.lower() if c in char_to_idx]
            all_targets.extend(ids)
            lengths.append(len(ids))
        return torch.tensor(all_targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)
