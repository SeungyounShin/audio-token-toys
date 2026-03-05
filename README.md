# audio-token-toys

From-scratch audio encoder + frozen LLM for speech recognition.

## Architecture

Raw waveform → CausalConvStack (50Hz) → Transformer encoder → MLP projector → Qwen3-0.6B

## Experiments

| Experiment | Encoder | LLM | Data | WER |
|---|---|---|---|---|
| [Baseline (n_layers=0)](experiments/llm_50hz_ctc_unfreeze.md) | Conv only, 5.8M | Unfrozen | 960h | 20.5% |
| [CTC ablation](experiments/ctc_ablation_2L.md) | 2L transformer | Frozen | 960h | - |
| [Data scaling](experiments/data_scaling_multi_en.md) | Conv only, 5.8M | Unfrozen | 2950h | 15.2% |
| [Staged 12L, frozen LLM](experiments/staged_12L_frozen_llm.md) | 12L, 91.5M | Frozen | 2950h | 23.6% |

## Quick Start

```bash
# Train
accelerate launch --num_processes 4 scripts/train_llm.py \
  --base_config configs/llm_12L_enc_proj_frozen_llm.yaml

# Evaluate
python scripts/evaluate_llm.py \
  --base_config configs/llm_12L_enc_proj_frozen_llm.yaml \
  --checkpoint outputs/llm_12L_enc_proj_frozen_llm/ckpt_50000.pt
```
