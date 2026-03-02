#!/usr/bin/env python3
"""End-to-end shape verification + 1-batch training sanity test."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import load_config
from src.model.student import StudentAudioEncoder
from src.model.teacher import TeacherWhisperEncoder
from src.training.distill_loss import DistillationLoss
from src.training.flop_counter import FLOPCounter


def test_shapes():
    """Verify output shapes for all 4 configs."""
    configs = [
        ("40ms-0%",  4, 4),
        ("40ms-50%", 4, 2),
        ("80ms-0%",  8, 8),
        ("80ms-50%", 8, 4),
    ]
    B, T = 2, 800  # 800 frames = 8 seconds

    print("=== Shape verification ===\n")
    for name, pf, hf in configs:
        student = StudentAudioEncoder(
            patch_frames=pf, hop_frames=hf,
            d_model=512, n_heads=8, n_layers=2,  # tiny for test
            d_ff=1024, max_seq_len=300,
        )
        mel = torch.randn(B, 80, T)
        n_frames = torch.tensor([T, T - 100])

        encoded, projected, n_tokens = student(mel, n_frames)
        expected_n = (T - pf) // hf + 1

        print(f"  {name:10s}: mel {list(mel.shape)} → "
              f"encoded {list(encoded.shape)}, "
              f"projected {list(projected.shape)}, "
              f"n_tokens {n_tokens.tolist()}, "
              f"expected_N={expected_n}")

        assert encoded.shape == (B, expected_n, 512), f"Bad encoded shape for {name}"
        assert projected.shape == (B, expected_n, 512), f"Bad projected shape for {name}"
    print("\n  All shapes OK!")


def test_distillation_step():
    """One forward+backward step of distillation."""
    print("\n=== 1-batch distillation test ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T = 4, 1600  # 16 seconds

    student = StudentAudioEncoder(
        patch_frames=4, hop_frames=4,
        d_model=512, n_heads=8, n_layers=2,
        d_ff=1024, max_seq_len=500,
    ).to(device)

    teacher = TeacherWhisperEncoder("openai/whisper-base").to(device)
    loss_fn = DistillationLoss()
    flops = FLOPCounter(512, 2, 1024)

    mel = torch.randn(B, 80, T, device=device)
    n_frames = torch.tensor([T, T - 200, T - 400, T - 600], device=device)

    # Forward
    encoded, projected, n_tokens = student(mel, n_frames)
    teacher_hidden = teacher(mel)

    print(f"  student: {list(projected.shape)}, n_tokens: {n_tokens.tolist()}")
    print(f"  teacher: {list(teacher_hidden.shape)}")

    # Loss
    loss = loss_fn(projected, teacher_hidden, n_tokens)
    print(f"  loss: {loss.item():.4f}")

    # Backward
    loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in student.parameters() if p.grad is not None) ** 0.5
    print(f"  grad norm: {grad_norm:.4f}")

    # FLOP count
    step_flops = flops.step(n_tokens.sum().item())
    print(f"  step FLOPs: {step_flops/1e9:.2f} GFLOPs")

    # Param count
    n_params = sum(p.numel() for p in student.parameters())
    print(f"  student params: {n_params/1e6:.1f}M")

    assert loss.item() > 0, "Loss should be positive"
    assert grad_norm > 0, "Gradients should flow"
    print("\n  Distillation step OK!")


def test_linear_probe():
    """Quick linear probe shape test."""
    from src.evaluation.linear_probe import LinearCTCProbe
    from src.evaluation.ctc_decode import greedy_ctc_decode

    print("\n=== Linear probe test ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = StudentAudioEncoder(
        patch_frames=4, hop_frames=4,
        d_model=512, n_heads=8, n_layers=2,
        d_ff=1024, max_seq_len=500,
    ).to(device)

    probe = LinearCTCProbe(encoder, d_model=512, vocab_size=29).to(device)

    mel = torch.randn(2, 80, 800, device=device)
    n_frames = torch.tensor([800, 600], device=device)

    log_probs, n_tokens = probe(mel, n_frames)
    print(f"  log_probs: {list(log_probs.shape)}, n_tokens: {n_tokens.tolist()}")

    # CTC decode
    decoded = greedy_ctc_decode(log_probs, n_tokens)
    print(f"  decoded: {decoded}")

    # CTC loss
    texts = ["hello world", "test"]
    targets, target_lengths = LinearCTCProbe.text_to_targets(texts)
    targets = targets.to(device)
    target_lengths = target_lengths.to(device)

    loss = probe.compute_loss(log_probs, n_tokens, targets, target_lengths)
    print(f"  CTC loss: {loss.item():.4f}")

    assert log_probs.shape[-1] == 29
    print("\n  Linear probe OK!")


if __name__ == "__main__":
    test_shapes()
    test_distillation_step()
    test_linear_probe()
    print("\n" + "=" * 40)
    print("ALL TESTS PASSED!")
