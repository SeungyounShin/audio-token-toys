# Ablation: CTC Auxiliary Loss vs No CTC (2L Transformer, 20K Steps)

## Setup

Both experiments use **identical architecture and hyperparameters**, differing only in whether CTC auxiliary loss is used.

| Config | Value |
|--------|-------|
| Architecture | raw wav → CausalConvStack (50Hz) → 2L Causal Transformer → Projector → Qwen3-0.6B |
| Encoder params | 12.1M (4.2M conv + 6.3M transformer + 1.6M projector) |
| LLM | Qwen3-0.6B, full fine-tune (596M params) |
| d_model | 512, n_heads=8, d_ff=2048, causal=True |
| LR | encoder 3e-4, projector 1e-3, LLM 1e-4 |
| Batch | 4/GPU × 8 grad_accum × 4 GPUs = 128 effective |
| Data | LibriSpeech 960h |
| Steps | 20,000 |
| Mixed precision | bf16 |

**CTC variant**: `use_ctc=True`, ctc_weight=0.3, ctc_warmup_steps=2000 (CTC-only for first 2K steps)
**No-CTC variant**: `use_ctc=False`, joint LM training from step 0

---

## Results

| Model | WER (test-clean, 100 samples) | LM Loss @ 20K | Phase Transition |
|-------|-------------------------------|---------------|-----------------|
| 2L Transformer + CTC | **35.3%** | **~0.8** | ✅ step ~14-16K |
| 2L Transformer, No CTC | **119.6%** | ~3.5-4.1 | ❌ never |
| Baseline (n_layers=0, CTC) @ 40K | 20.5% | ~0.3 | ✅ step ~17K |

---

## Loss Trajectories

### 2L + CTC
| Step | LM Loss | CTC Loss | Note |
|------|---------|----------|------|
| 2K | - | 2.75 | warmup ends |
| 4K | 4.13 | 2.55 | joint training starts |
| 8K | 4.25 | 2.22 | plateau |
| 12K | 3.70 | 1.91 | transition begins |
| 14K | 3.19 | 1.44 | rapid drop |
| 16K | 1.52 | 1.73 | **phase transition** |
| 18K | 0.80 | 1.60 | |
| 20K | ~0.8 | ~1.4 | |

### 2L + No CTC
| Step | LM Loss | Note |
|------|---------|------|
| 2K | 4.04 | |
| 4K | 4.57 | |
| 8K | 3.75 | |
| 12K | 4.45 | |
| 16K | 3.60 | |
| 18K | 3.50 | |
| 20K | 4.12 | still in plateau |

---

## Analysis

### CTC is essential for inducing the phase transition

Without CTC, the encoder never learns to produce features the LLM can decode — the LM loss stays stuck at ~3.5-4.5 throughout 20K steps with no phase transition. The no-CTC model hallucinates repetitive phrases ("and the other was a little boy...") for virtually every input.

With CTC, the encoder is forced to align its 50Hz conv features with character sequences, effectively learning a phonetic representation. Once this representation becomes sufficiently clean (~step 12-14K), the LLM can suddenly "read" the audio features, triggering the phase transition (LM loss 4.0 → 0.8 within ~4K steps).

### Why CTC matters at 25ms receptive field

The CausalConvStack has only ~25ms receptive field — far too small for phoneme recognition on its own. However, the 2-layer transformer can integrate ~seconds of context at 50Hz. The CTC loss on transformer output forces this context integration to be phonetically meaningful, rather than relying on the LLM to figure out everything from scratch.

### Comparison with baseline (n_layers=0)

The baseline model (no transformer, CTC directly on conv features) still achieved WER 20.5% at 40K steps. The 2L transformer with CTC reaches WER 35.3% at only 20K steps, with LM loss 0.8 vs baseline's 0.3 at 50K steps. The 2L model is learning faster (phase transition at 14K vs 17K) and will likely converge to lower WER by 50K.

---

## Qualitative Samples @ 20K

### CTC model (WER 35.3%)
```
ref: the english forwarded to the french baskets of flowers...
hyp: the english thought it too the french basket of flowers...  (WER 23%)

ref: she taught her daughter then by her own affection for it...
hyp: she tore her daughter then by her own affection for it...   (WER 23%)
```

### No-CTC model (WER 119.6%)
```
ref: the english forwarded to the french baskets of flowers...
hyp: and the two men who had been in the habit of sitting...     (WER 105%)

ref: she taught her daughter then by her own affection for it...
hyp: the first thing i did was to get the money and i did it...  (WER 103%)
```

The no-CTC model completely ignores the audio and generates plausible-sounding LibriSpeech-style text from memory.

---

## Conclusion

**CTC auxiliary loss is critical** for the audio-to-text pipeline with a from-scratch conv encoder. It solves the cold-start problem by providing direct phonetic supervision that the LLM loss alone cannot provide at this scale (only 20K steps, 960h data).
