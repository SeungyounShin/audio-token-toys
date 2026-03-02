"""Distillation loss: align teacher/student sequences via interpolation + MSE."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """MSE loss between student projections and interpolated teacher hidden states.

    The teacher produces 1500 tokens (20ms each) for 30s input.
    The student produces N tokens depending on patch/overlap config.
    We linearly interpolate the teacher to match the student's token count.
    """

    def forward(
        self,
        student_proj: torch.Tensor,
        teacher_hidden: torch.Tensor,
        student_n_tokens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            student_proj: [B, N_s, D] student projected output.
            teacher_hidden: [B, 1500, D] teacher last hidden state.
            student_n_tokens: [B] valid token count per sample.
        Returns:
            Scalar loss.
        """
        B, N_s, D = student_proj.shape

        # Interpolate teacher [B, D, 1500] -> [B, D, N_s]
        teacher_aligned = F.interpolate(
            teacher_hidden.permute(0, 2, 1),  # [B, D, 1500]
            size=N_s,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)  # [B, N_s, D]

        # Per-token MSE
        mse = (student_proj - teacher_aligned).pow(2).mean(dim=-1)  # [B, N_s]

        # Mask padding tokens
        if student_n_tokens is not None:
            mask = torch.arange(N_s, device=mse.device)[None, :] < student_n_tokens[:, None]
            mse = mse * mask.float()
            return mse.sum() / mask.float().sum().clamp(min=1)

        return mse.mean()
