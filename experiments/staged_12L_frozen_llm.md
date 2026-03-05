# Staged 12L Encoder Training (LLM Frozen)

## Experiment Summary

Train a 12-layer transformer encoder with frozen LLM through staged training:
1. Stage 1: CTC-only encoder pre-training
2. Stage 2: Projector-only with frozen encoder + frozen LLM
3. Encoder + projector joint training with frozen LLM

## Architecture

- **Encoder**: 12L transformer (d=768, h=12), 91.5M params
- **Projector**: Linear(768→1024), 1.8M params
- **LLM**: Qwen3-0.6B, fully frozen (596M params)
- **Data**: ~2950h English (LibriSpeech + TED-LIUM + VoxPopuli + People's Speech)

## Stages

### Stage 1: CTC-only encoder (60K steps)
- Config: `configs/llm_whisper_small_ctc_multi.yaml`
- CTC loss: 2.5 → 0.23
- CTC-only WER: ~51.6%

### Stage 2: Projector-only (10K steps)
- Config: `configs/llm_12L_stage2_projector.yaml`
- Encoder frozen, LLM frozen, projector 1.8M training
- LM loss: 6.0 → 1.6
- WER: ~24%

### Stage 3: Encoder + Projector (50K steps)
- Config: `configs/llm_12L_enc_proj_frozen_llm.yaml`
- Resume from Stage 2 checkpoint, LLM frozen
- encoder_lr: 1e-4, projector_lr: 3e-4, max_grad_norm: 0.5

## Results (test-clean, 100 samples, greedy)

| Step | WER |
|------|------|
| 10K  | 45.6% |
| 20K  | 40.4% |
| 30K  | 25.9% |
| 50K  | 23.6% |

## Key Observations

- Phase transition between 20K-30K (40.4% → 25.9%)
- Converges to ~23.6% at 50K — LLM frozen ceiling
- Frozen LLM limits WER floor vs. unfrozen baseline (15.2%)
- Encoder unfreeze temporarily degrades then recovers (Stage 2 24% → 10K 45.6% → 50K 23.6%)
