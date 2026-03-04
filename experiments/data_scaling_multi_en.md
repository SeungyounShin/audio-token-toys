# Data Scaling: Baseline + Multi-English Dataset

## Experiment Summary

Scale the baseline architecture (n_layers=0, conv→projector→LLM) along the **data axis** by adding more English speech datasets, keeping the architecture identical.

## Architecture

- **Encoder**: n_layers=0, conv_stack(50Hz) → MLP projector → LLM (5.8M params)
- **LLM**: Qwen3-0.6B, fully unfrozen (596M params)
- **CTC**: auxiliary loss on 50Hz conv features, warmup 5K steps
- **Total trainable**: ~602M

## Data

| Dataset | Hours (approx) |
|---------|---------------|
| LibriSpeech (train-960) | 960h |
| TED-LIUM | 450h |
| VoxPopuli (en) | 540h |
| People's Speech (clean, 300K samples) | ~1000h |
| **Total** | **~2950h** |

Baseline used LibriSpeech 960h only → **~3x data increase**.

## Training Config

- `configs/llm_direct_ctc_unfreeze_multi_en.yaml`
- Batch: 4/GPU × 8 accum × 4 GPUs = effective 128
- LR: encoder=3e-4, projector=1e-3, LLM=1e-4
- CTC warmup: 5K steps (CTC-only), then joint CTC+LM
- max_grad_norm: 1.0
- Total: 100K steps (resumed from 50K checkpoint)

## Results

### WER (test-clean, 100 samples, greedy decoding)

| Step | WER | Notes |
|------|-----|-------|
| 30K | 60.6% | Pre-phase-transition |
| 40K | 32.9% | Transition in progress |
| 50K | 26.9% | Still converging |
| **100K** | **15.2%** | **Best** |

### Comparison with Baseline

| Experiment | Data | Steps | WER |
|------------|------|-------|-----|
| Baseline (n_layers=0) | 960h LibriSpeech | 40K | 20.5% |
| **Data scaling** | **~2950h Multi-EN** | **100K** | **15.2%** |

**Improvement: 20.5% → 15.2% (26% relative reduction)**

### Sample Outputs (step 100K)

```
ref:  from the respect paid her on all sides she seemed like a queen and from
      the adoration with which she was treated by two or three she appeared an
      object of worship the queen mother gave the french the most affectionate
      reception france was her native country and she had suffered too much
      unhappiness in england for england to have made her forget france
hyp:  from the respect paid her on all sides she seemed like a queen and from
      the adoration with which she was treated by two or three she appeared in
      object of worship the queen mother gave the french the most affectionate
      reception franz was her native country and she had suffered too much in
      happiness in england for england to have made her forget france

ref:  she taught her daughter then by her own affection for it that love for a
      country where they had both been hospitably received and where a brilliant
      future opened before them
hyp:  she taught her daughter then by her own affection for it that love for a
      country where they had both been hospitably received and where a brilliant
      future opened for them
```

## Phase Transition Timing

| Experiment | Data | Phase transition step |
|------------|------|---------------------|
| Baseline | 960h | ~17K |
| 2L transformer | 960h | ~14-16K |
| **Data scaling** | **~2950h** | **~35-40K** |

Phase transition was delayed (~2x) with more data, likely because each epoch takes longer and the model needs sufficient passes over the diverse data to learn the audio→text mapping.

## Failed Approaches (12L Whisper-Small Scale)

Before settling on data scaling, we attempted scaling the encoder to Whisper-Small size (d_model=768, n_layers=12, 91.5M encoder params) with multilingual data. This failed in multiple ways:

1. **Full LLM unfreeze + multilingual**: LLM memorized training data instead of learning audio features. WER 112% — noise test confirmed model completely ignored audio input.
2. **Frozen LLM + LM loss**: Gradient explosions from backprop through 28 frozen LLM layers. Persistent NaN skips (~80% of steps).
3. **CTC-only encoder training**: Stable (CTC loss 0.2-0.5 at 60K steps), but CTC greedy WER was only 51.6% — limited by character-level CTC without language model.

**Key lesson**: Scaling encoder size and adding multilingual data simultaneously introduced too many variables. Data scaling with a proven architecture was more effective.

## Conclusions

1. **Data scaling works**: Same 5.8M encoder, 3x more data → 26% relative WER reduction
2. **Phase transition is delayed but still occurs**: More data = later grokking but ultimately better performance
3. **Architecture scaling requires careful staging**: Naive encoder scaling + multilingual failed; data scaling is the safer first step
4. **CTC+LM joint training remains essential**: CTC alone (51.6%) vs CTC+LM (15.2%) confirms the LLM decoder is critical
