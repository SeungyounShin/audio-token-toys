#!/usr/bin/env python3
"""Shape verification and 1-batch forward pass test for the LLM pipeline.

Uses a mock LLM to test shapes and gradient flow without real model weights.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.encoder import AudioTransformerEncoder
from src.frontend.mel_frontend import MelFrontend
from src.model.audio_encoder import AudioEncoder
from src.training.flop_counter import LLMFLOPCounter


def test_causal_encoder():
    """Verify causal mask doesn't break shapes or introduce NaN."""
    print("=== CausalEncoder ===")
    B, N, d = 4, 100, 512
    enc = AudioTransformerEncoder(
        d_model=d, n_heads=8, n_layers=2, d_ff=1024, max_seq_len=200, causal=True
    )
    x = torch.randn(B, N, d)
    pad_mask = torch.zeros(B, N, dtype=torch.bool)
    pad_mask[:2, 80:] = True
    out = enc(x, src_key_padding_mask=pad_mask)
    assert out.shape == (B, N, d), f"Bad shape: {out.shape}"
    assert not out.isnan().any(), "NaN in causal encoder output"
    print(f"  OK: output {list(out.shape)}, no NaN")


def test_causal_encoder_no_padding():
    """Verify causal encoder works without padding mask."""
    print("=== CausalEncoder (no padding) ===")
    B, N, d = 2, 50, 512
    enc = AudioTransformerEncoder(
        d_model=d, n_heads=8, n_layers=2, d_ff=1024, max_seq_len=200, causal=True
    )
    x = torch.randn(B, N, d)
    out = enc(x)
    assert out.shape == (B, N, d), f"Bad shape: {out.shape}"
    assert not out.isnan().any(), "NaN in causal encoder output"
    print(f"  OK: output {list(out.shape)}, no NaN")


def test_noncausal_encoder():
    """Verify non-causal mode still works (backward compat)."""
    print("=== NonCausalEncoder ===")
    B, N, d = 4, 100, 512
    enc = AudioTransformerEncoder(
        d_model=d, n_heads=8, n_layers=2, d_ff=1024, max_seq_len=200, causal=False
    )
    x = torch.randn(B, N, d)
    out = enc(x)
    assert out.shape == (B, N, d), f"Bad shape: {out.shape}"
    print(f"  OK: output {list(out.shape)}")


def test_mel_frontend():
    """Verify MelFrontend token count formula."""
    print("=== MelFrontend ===")
    B, n_mels, T = 4, 80, 1600
    frontend = MelFrontend(n_mels=n_mels, patch_frames=8, hop_frames=8, d_model=512)
    mel = torch.randn(B, n_mels, T)
    tokens = frontend(mel)
    expected_N = (T - 8) // 8 + 1  # = 200
    assert tokens.shape == (B, expected_N, 512), f"Bad shape: {tokens.shape}"

    # Test batch num_tokens
    n_frames = torch.tensor([1600, 1400, 1200, 800])
    n_tokens = frontend.num_tokens_batch(n_frames)
    expected = torch.tensor([200, 175, 150, 100])
    assert torch.equal(n_tokens, expected), f"Bad n_tokens: {n_tokens} vs {expected}"
    print(f"  OK: mel {list(mel.shape)} -> tokens {list(tokens.shape)}")
    print(f"  OK: n_tokens = {n_tokens.tolist()}")


def test_audio_encoder_mel():
    """Verify AudioEncoder (mel frontend) output shape [B, N, llm_dim]."""
    print("=== AudioEncoder (mel) ===")
    B, T = 4, 1600
    encoder = AudioEncoder(
        frontend_type="mel",
        n_mels=80,
        patch_frames=8,
        hop_frames=8,
        d_model=512,
        n_heads=8,
        n_layers=2,
        d_ff=1024,
        max_seq_len=300,
        llm_dim=1024,
        causal=True,
    )
    mel = torch.randn(B, 80, T)
    n_frames = torch.tensor([1600, 1400, 1200, 1000])
    feats, n_tokens = encoder(mel, n_frames)
    assert feats.shape == (B, 200, 1024), f"Bad shape: {feats.shape}"
    assert n_tokens is not None
    print(f"  OK: features {list(feats.shape)}, n_tokens={n_tokens.tolist()}")

    params = sum(p.numel() for p in encoder.parameters())
    print(f"  OK: {params / 1e6:.1f}M params")


def test_audio_encoder_raw():
    """Verify AudioEncoder (raw frontend) output shape."""
    print("=== AudioEncoder (raw) ===")
    B = 4
    # 1 second = 16000 samples -> 12.5 tokens at 12.5Hz
    n_samples = torch.tensor([16000, 14080, 12800, 10240])
    max_samples = n_samples.max().item()
    # Pad to multiple of 1280
    pad_to = ((max_samples + 1279) // 1280) * 1280
    wav = torch.randn(B, pad_to)

    encoder = AudioEncoder(
        frontend_type="raw",
        d_model=512,
        n_heads=8,
        n_layers=2,
        d_ff=1024,
        max_seq_len=300,
        llm_dim=1024,
        causal=True,
    )
    feats, n_tokens = encoder(wav, n_samples)
    expected_N = pad_to // 1280  # total stride
    assert feats.shape == (B, expected_N, 1024), f"Bad shape: {feats.shape}"
    assert n_tokens is not None
    print(f"  OK: wav {list(wav.shape)} -> features {list(feats.shape)}, n_tokens={n_tokens.tolist()}")

    params = sum(p.numel() for p in encoder.parameters())
    print(f"  OK: {params / 1e6:.1f}M params")


class _MockTokenizer:
    eos_token_id = 2
    pad_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return list(range(10, 10 + min(len(text.split()), 20)))

    def convert_tokens_to_ids(self, token):
        return 3  # fake im_end id

    def decode(self, ids, skip_special_tokens=False):
        return "mock output"


class _MockOutput:
    def __init__(self, loss):
        self.loss = loss


class _MockLLM(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self._embed = nn.Embedding(100, dim)
        self._embed.weight.requires_grad = False

    def get_input_embeddings(self):
        return self._embed

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None, **kw):
        # Fake loss that flows gradients through inputs_embeds
        return _MockOutput(inputs_embeds[:, :, :1].sum())

    def eval(self):
        return self

    def train(self, mode=True):
        return self


def test_bridge_with_mock_llm():
    """Verify EmbeddingInjectionBridge with a tiny mock LLM."""
    print("=== EmbeddingInjectionBridge (mock LLM) ===")
    from src.llm_bridge.injection import EmbeddingInjectionBridge

    B = 4
    llm = _MockLLM(1024)
    tok = _MockTokenizer()
    bridge = EmbeddingInjectionBridge(llm, tok, max_text_len=64)

    # Use raw waveform frontend
    n_samples = torch.tensor([16000, 14080, 12800, 10240])
    pad_to = ((n_samples.max().item() + 1279) // 1280) * 1280
    wav = torch.randn(B, pad_to)

    audio_encoder = AudioEncoder(
        frontend_type="raw",
        d_model=512,
        n_heads=8,
        n_layers=2,
        d_ff=1024,
        max_seq_len=300,
        llm_dim=1024,
        causal=True,
    )
    feats, n_tokens = audio_encoder(wav, n_samples)
    texts = ["hello world", "test sample here", "chapter one two", "the quick fox"]

    loss = bridge(feats, texts, n_audio_tokens=n_tokens)
    loss.backward()

    # Verify gradients flow to audio encoder
    has_grad = False
    for name, p in audio_encoder.named_parameters():
        if p.grad is not None:
            assert not p.grad.isnan().any(), f"NaN grad in {name}"
            has_grad = True
    assert has_grad, "No gradients in audio encoder!"
    print(f"  OK: loss={loss.item():.4f}, gradients flow through encoder")


def test_flop_counter():
    """Verify LLMFLOPCounter produces reasonable values."""
    print("=== LLMFLOPCounter ===")
    counter = LLMFLOPCounter(512, 6, 2048, 1024, 28, 3072)
    flops = counter.step(16 * 150, 16 * 50)
    assert flops > 0
    print(f"  Step FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"  OK: {counter.tflops:.4f} TFLOPs cumulative")


if __name__ == "__main__":
    test_causal_encoder()
    test_causal_encoder_no_padding()
    test_noncausal_encoder()
    test_mel_frontend()
    test_audio_encoder_mel()
    test_audio_encoder_raw()
    test_bridge_with_mock_llm()
    test_flop_counter()
    print("\n" + "=" * 40)
    print("ALL TESTS PASSED!")
