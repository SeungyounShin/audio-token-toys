"""Greedy CTC decoding."""

import torch
from .linear_probe import LinearCTCProbe


def greedy_ctc_decode(
    log_probs: torch.Tensor,
    n_tokens: torch.Tensor,
) -> list:
    """Greedy CTC decode: collapse repeats, remove blanks.

    Args:
        log_probs: [B, T, V]
        n_tokens: [B] valid lengths
    Returns:
        List of decoded strings.
    """
    idx_to_char = {i: c for i, c in enumerate(LinearCTCProbe.VOCAB)}
    preds = log_probs.argmax(dim=-1)  # [B, T]
    results = []

    for i in range(len(preds)):
        seq = preds[i, : n_tokens[i]]
        decoded = []
        prev = -1
        for idx in seq:
            idx = idx.item()
            if idx != 0 and idx != prev:  # 0 = blank
                decoded.append(idx_to_char.get(idx, ""))
            prev = idx
        results.append("".join(decoded))

    return results
