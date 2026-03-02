# Experiment: llm_50hz_ctc_unfreeze

**Date**: 2026-03-02
**Status**: Running (step 44K/50K)

## Summary

From-scratch causal conv encoder (50Hz) feeding directly into fully fine-tuned Qwen3-0.6B, with CTC auxiliary loss. No transformer encoder (n_layers=0).

**Result: WER 20.5% on test-clean** (100 samples, 40K checkpoint)

## Architecture

```
Raw waveform (16kHz)
  → 7-layer CausalConvStack (stride 320 → 50Hz, 512d)
  → MLP Projector (512 → 1024 → 1024 + LayerNorm)
  → Qwen3-0.6B (fully unfrozen, chat template injection)
  → Autoregressive text generation
```

- **Audio encoder**: 5.8M params (conv + projector + CTC head)
- **LLM**: 596M params (Qwen3-0.6B, all layers unfrozen)
- **CTC head**: Linear(512 → 29) on 50Hz conv features (a-z, space, apostrophe, blank)
- **No transformer encoder** — conv features go directly to projector

## Training

| Param | Value |
|-------|-------|
| Data | LibriSpeech 960h (train.100+360+500) |
| Batch size | 4/GPU × 4 GPUs × 8 grad_accum = 128 effective |
| CTC warmup | 5000 steps (CTC-only, no LM loss) |
| Total steps | 50000 (~3 epochs) |
| LR (encoder) | 3e-4 |
| LR (projector) | 1e-3 |
| LR (LLM) | 1e-4 (cosine decay) |
| Precision | bf16 |
| Hardware | 4× NVIDIA A6000 (48GB) |
| Wall time | ~7 hours |

### Loss Trajectory

```
Step   | LM loss | CTC loss | Notes
-------+---------+----------+----------------------------
  1K   |    —    |  2.87    | CTC warmup (LM off)
  5K   |    —    |  2.58    | CTC warmup ends
  5.5K |  4.37   |  2.55    | Joint training begins
 12K   |  4.06   |  2.34    | Apparent plateau
 17K   |  3.18   |  2.27    | ★ Phase transition begins
 19K   |  2.19   |  2.25    | Rapid LM improvement
 21K   |  1.26   |  2.26    | LM loss drops 4x in 4K steps
 25K   |  0.69   |  2.21    |
 30K   |  0.46   |  2.13    |
 35K   |  0.28   |  2.10    |
 40K   |  0.28   |  2.06    | ← eval checkpoint
 44K   |  0.35   |  2.04    | (ongoing)
```

Key observation: **LM loss has a phase transition at step ~17K**. The first 12K steps of joint training (~5K-17K) appear to plateau at LM loss ~4.0, then suddenly the loss drops from 4.0 to 1.2 within 4K steps. CTC loss barely changes throughout (2.5→2.0).

## Evaluation (step 40K, test-clean, 100 samples)

**WER: 20.5%** (384 errors / 1870 words)

### Sample Outputs

| WER | Reference | Hypothesis |
|-----|-----------|------------|
| 0% | you will be frank with me i always am | you will be frank with me i always am |
| 0% | it is you who are mistaken raoul i have read his distress in his eyes in his every gesture and action the whole day | (exact match) |
| 6% | she taught her daughter then by her own affection for it that love for a country where they had both been hospitably received and where a brilliant future opened before them | she **told** her daughter then by her own affection for it that love for a country where they had both been hospitably received and where a brilliant future opened **for** them |
| 12% | the english forwarded to the french baskets of flowers... (43 words) | the english **thought it to be a** french basket of flowers... (minor substitutions at start, rest perfect) |
| 62% | concord returned to its place amidst the tents | conquer the torrent which plays amidst the tents |

### Error Patterns

- **Short utterances**: Higher WER, more prone to hallucination
- **Rare/proper nouns**: "Buckingham" → misrecognized, "Kaffar" → "half her"
- **Long utterances**: Generally lower WER, LLM's language model helps fill in gaps
- **Hallucination**: Some samples generate plausible but incorrect text (the LLM "invents" content)

## Technical Details

### Bugs Fixed During Development

1. **DDP unused parameters**: When `use_50hz=True`, frontend downsample/proj/norm modules are unused. Fix: delete them in `__init__` to avoid DDP errors.

2. **CTC inside forward()**: CTC loss must be computed inside `AudioEncoder.forward()` (not externally) so DDP can track all parameter gradients through the autograd graph.

3. **LLM gradient sync**: The LLM is not DDP-wrapped (only audio_encoder is via `accelerator.prepare`). Manual `torch.distributed.all_reduce` on LLM gradients is required at sync steps.

4. **Non-finite gradient skip**: Initial joint training produces transient gradient explosions (~12% skip rate for first ~200 joint steps). Fix: detect non-finite `grad_norm` after `clip_grad_norm` and skip the optimizer step. Stabilizes quickly.

5. **Bridge re-freeze**: `EmbeddingInjectionBridge.__init__` freezes all LLM params. Must re-unfreeze after bridge creation.

### Design Choices

- **LayerNorm gamma=0.03**: Projector's final LayerNorm initialized with gamma=0.03 to match LLM embedding std (~0.029), preventing bf16 attention overflow at the start of joint training.
- **No conv freezing**: Attempted to freeze conv after CTC warmup, but DDP can't handle mid-training parameter freeze. Removed entirely.
- **50Hz tokens**: 4x more tokens than 12.5Hz, but gives richer temporal info. A 10s utterance = 500 audio tokens.

## Analysis

The conv stack has a receptive field of only ~25ms per frame — barely enough for a single phoneme. CTC on these features plateaus at loss ~2.0 and produces garbled output. Despite this, the LLM learns to decode these low-level features into accurate text.

This suggests the LLM performs the heavy lifting: it learns a mapping from noisy, local acoustic features to language, leveraging its pre-trained language model to resolve ambiguities. The "phase transition" at step 17K may correspond to the LLM developing an internal acoustic model that can parse the conv features.

## Config

```yaml
experiment_name: "llm_50hz_ctc_unfreeze"
frontend_type: "raw"
d_model: 512
n_layers: 0
use_50hz: true
use_ctc: true
ctc_weight: 0.3
ctc_warmup_steps: 5000
llm_model: "Qwen/Qwen3-0.6B"
llm_dim: 1024
unfreeze_llm_layers: 99
encoder_lr: 3.0e-4
projector_lr: 1.0e-3
lr: 1.0e-4
batch_size: 4
grad_accum: 8
total_steps: 50000
mixed_precision: "bf16"
train_split: "train.100+train.360+train.500"
```

## Next Steps

- [ ] Full test-clean + test-other WER at 50K steps
- [ ] Run `llm_50hz_2L_ctc_unfreeze` (2-layer causal transformer) for comparison
- [ ] Investigate hallucination on short utterances
- [ ] Try beam search decoding (currently greedy)
