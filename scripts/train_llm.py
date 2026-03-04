#!/usr/bin/env python3
"""Training script: frozen LLM decoder + trainable audio encoder.

Supports two frontend modes:
  - 'raw': raw waveform -> causal conv stack -> transformer -> LLM
  - 'mel': precomputed mel -> patch embedding -> transformer -> LLM
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import torch.distributed
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import load_config
from src.model.audio_encoder import AudioEncoder
from src.llm_bridge.injection import EmbeddingInjectionBridge
from src.training.flop_counter import LLMFLOPCounter


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Single LambdaLR: linear warmup + cosine decay. No double-step bug."""

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/llm_base.yaml")
    parser.add_argument("--config", default=None, help="Experiment config override")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume_weights", default=None,
                        help="Load model weights but reset optimizer/scheduler (for new data)")
    args = parser.parse_args()

    cfg = load_config(args.base_config, args.config)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Accelerator ──
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.grad_accum,
        log_with="tensorboard",
        project_dir=str(run_dir),
    )
    accelerator.init_trackers(cfg.experiment_name)
    is_main = accelerator.is_main_process

    frontend_type = getattr(cfg, "frontend_type", "raw")
    use_ctc = getattr(cfg, "use_ctc", False)
    ctc_weight = getattr(cfg, "ctc_weight", 0.3)
    ctc_warmup_steps = getattr(cfg, "ctc_warmup_steps", 0)  # CTC-only warmup before joint

    if is_main:
        print(f"=== Training: {cfg.experiment_name} ===")
        print(f"  LLM: {cfg.llm_model}")
        print(f"  Audio encoder: {cfg.n_layers}L, d={cfg.d_model}, causal={cfg.causal}")
        print(f"  Frontend: {frontend_type}")
        if use_ctc:
            print(f"  CTC auxiliary loss: weight={ctc_weight}")
        print(f"  Batch: {cfg.batch_size}/GPU x {cfg.grad_accum} accum x {accelerator.num_processes} GPUs")
        print(f"  Steps: {cfg.total_steps}, LR: {cfg.lr}")

    # ── Data ──
    if frontend_type == "raw":
        from src.data.wav_dataset import LibriSpeechWavDataset, MultiWavDataset, wav_collate_fn
        if hasattr(cfg, "datasets") and cfg.datasets:
            train_ds = MultiWavDataset(cfg.datasets, sr=cfg.sr)
            if is_main:
                print(f"  Multi-dataset: {len(cfg.datasets)} sources, {len(train_ds)} samples")
        else:
            train_ds = LibriSpeechWavDataset(split=cfg.train_split, sr=cfg.sr)
        collate = wav_collate_fn
    else:
        from src.data.dataset import LibriSpeechMelDataset, collate_fn
        train_ds = LibriSpeechMelDataset(str(Path(cfg.data_dir) / cfg.train_split))
        collate = collate_fn

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ── Audio Encoder (trainable) ──
    audio_encoder = AudioEncoder.from_config(cfg)

    # ── LLM ──
    if is_main:
        print(f"\nLoading LLM: {cfg.llm_model}")
    llm_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.llm_model, torch_dtype=llm_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model)

    # Freeze all, then optionally unfreeze first N layers
    for p in llm.parameters():
        p.requires_grad = False

    unfreeze_llm_layers = getattr(cfg, "unfreeze_llm_layers", 0)
    if unfreeze_llm_layers > 0:
        # Unfreeze input embedding
        for p in llm.get_input_embeddings().parameters():
            p.requires_grad = True
        # Unfreeze first N transformer layers
        for i in range(min(unfreeze_llm_layers, len(llm.model.layers))):
            for p in llm.model.layers[i].parameters():
                p.requires_grad = True

    llm_trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    llm_frozen = sum(p.numel() for p in llm.parameters() if not p.requires_grad)

    if is_main:
        enc_params = sum(p.numel() for p in audio_encoder.parameters())
        print(f"  Audio encoder: {enc_params / 1e6:.1f}M trainable params")
        if unfreeze_llm_layers > 0:
            print(f"  LLM: {llm_trainable / 1e6:.1f}M trainable (first {unfreeze_llm_layers} layers + embed)")
            print(f"  LLM: {llm_frozen / 1e6:.1f}M frozen (remaining layers)")
        else:
            print(f"  LLM: {(llm_trainable + llm_frozen) / 1e6:.0f}M frozen params")

    # ── Bridge ──
    bridge = EmbeddingInjectionBridge(llm, tokenizer, max_text_len=cfg.max_text_len)

    # Re-apply unfreeze AFTER bridge creation (bridge.__init__ re-freezes all)
    if unfreeze_llm_layers > 0:
        if unfreeze_llm_layers >= len(llm.model.layers):
            # Unfreeze ALL
            for p in llm.parameters():
                p.requires_grad = True
        else:
            for p in llm.get_input_embeddings().parameters():
                p.requires_grad = True
            for i in range(min(unfreeze_llm_layers, len(llm.model.layers))):
                for p in llm.model.layers[i].parameters():
                    p.requires_grad = True

        llm_trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
        llm_frozen = sum(p.numel() for p in llm.parameters() if not p.requires_grad)
        if is_main:
            print(f"  [verified] LLM trainable: {llm_trainable/1e6:.1f}M, frozen: {llm_frozen/1e6:.1f}M")

    # ── Optimizer + Scheduler ──
    # Separate LRs: encoder (conv+CTC), projector, LLM
    projector_lr = getattr(cfg, "projector_lr", 0.0) or cfg.lr
    encoder_lr = getattr(cfg, "encoder_lr", 0.0) or cfg.lr
    encoder_params = []
    projector_params = []
    for name, p in audio_encoder.named_parameters():
        if "projector" in name:
            projector_params.append(p)
        else:
            encoder_params.append(p)

    param_groups = [
        {"params": encoder_params, "lr": encoder_lr},
        {"params": projector_params, "lr": projector_lr},
    ]
    if unfreeze_llm_layers > 0:
        param_groups.append(
            {"params": [p for p in llm.parameters() if p.requires_grad], "lr": cfg.lr}
        )

    trainable_params = encoder_params + projector_params
    if unfreeze_llm_layers > 0:
        trainable_params += [p for p in llm.parameters() if p.requires_grad]

    if is_main:
        print(f"  LR: encoder={encoder_lr}, projector={projector_lr}, LLM={cfg.lr}")

    optimizer = AdamW(
        param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, cfg.warmup_steps, cfg.total_steps
    )

    # ── Prepare (only trainable model + optimizer + data + scheduler) ──
    audio_encoder, optimizer, train_dl, scheduler = accelerator.prepare(
        audio_encoder, optimizer, train_dl, scheduler
    )
    # Frozen LLM moved to device manually (not via prepare)
    llm = llm.to(accelerator.device)
    bridge.llm = llm

    # Read LLM architecture for FLOP counting
    llm_cfg = llm.config
    flops = LLMFLOPCounter(
        enc_d_model=cfg.d_model,
        enc_n_layers=cfg.n_layers,
        enc_d_ff=cfg.d_ff,
        llm_d_model=llm_cfg.hidden_size,
        llm_n_layers=llm_cfg.num_hidden_layers,
        llm_d_ff=llm_cfg.intermediate_size,
    )

    # ── Resume ──
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        accelerator.unwrap_model(audio_encoder).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt["step"]
        flops.cumulative = ckpt.get("cumulative_flops", 0)
        if "llm_full" in ckpt and unfreeze_llm_layers > 0:
            llm.load_state_dict(ckpt["llm_full"])
        if is_main:
            print(f"  Resumed from step {global_step}")
    elif args.resume_weights:
        # Load model weights only, reset optimizer/scheduler (for new data phase)
        ckpt = torch.load(args.resume_weights, map_location="cpu", weights_only=False)
        accelerator.unwrap_model(audio_encoder).load_state_dict(ckpt["model"])
        if "llm_full" in ckpt and unfreeze_llm_layers > 0:
            llm.load_state_dict(ckpt["llm_full"])
        if is_main:
            print(f"  Loaded weights from {args.resume_weights} (step {ckpt.get('step', '?')})")
            print(f"  Optimizer/scheduler reset for new training phase")

    # ── Training loop ──
    log_history = []
    audio_encoder.train()
    epoch = 0
    t0 = time.time()

    if is_main:
        print(f"\nStarting training from step {global_step}...\n")

    while global_step < cfg.total_steps:
        epoch += 1
        for batch in train_dl:
            if global_step >= cfg.total_steps:
                break

            if frontend_type == "raw":
                audio_input = batch["wav"]
                n_lengths = batch["n_samples"]
            else:
                audio_input = batch["mel"]
                n_lengths = batch["n_frames"]
            texts = batch["texts"]

            with accelerator.accumulate(audio_encoder):
                # CTC loss computed inside forward for DDP compatibility
                audio_features, n_tokens, ctc_loss = audio_encoder(
                    audio_input, n_lengths, texts=texts if use_ctc else None
                )

                # CTC warmup: CTC-only for first N steps, then joint
                in_ctc_warmup = use_ctc and ctc_warmup_steps > 0 and global_step < ctc_warmup_steps

                if not in_ctc_warmup:
                    lm_loss = bridge(audio_features, texts, n_audio_tokens=n_tokens)
                else:
                    lm_loss = torch.tensor(0.0, device=audio_features.device)

                if use_ctc and ctc_loss is not None:
                    if in_ctc_warmup:
                        loss = ctc_loss + 0.0 * audio_features.sum()
                    else:
                        loss = lm_loss + ctc_weight * ctc_loss
                else:
                    loss = lm_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # LLM is not DDP-wrapped: manually all-reduce its gradients
                    if unfreeze_llm_layers > 0:
                        for p in llm.parameters():
                            if p.grad is not None:
                                torch.distributed.all_reduce(
                                    p.grad, op=torch.distributed.ReduceOp.AVG
                                )
                    grad_norm = accelerator.clip_grad_norm_(
                        trainable_params, cfg.max_grad_norm
                    )
                    # Skip step if gradients overflowed (bf16 inf → NaN)
                    if not math.isfinite(grad_norm):
                        optimizer.zero_grad()
                        if is_main:
                            print(f"  [step {global_step}] SKIP: non-finite grad norm")
                        global_step += 1
                        continue
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # FLOP tracking
            if n_tokens is not None:
                total_audio = accelerator.gather(n_tokens.sum()).sum().item()
            else:
                total_audio = (
                    audio_input.shape[0]
                    * audio_features.shape[1]
                    * accelerator.num_processes
                )
            # Estimate text tokens from tokenizer
            total_text = sum(
                len(tokenizer.encode(t, add_special_tokens=False)) + 1 for t in texts
            ) * accelerator.num_processes
            flops.step(int(total_audio), int(total_text))

            global_step += 1

            # ── Logging ──
            if global_step % cfg.log_every == 0 and is_main:
                elapsed = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]
                log_entry = {
                    "step": global_step,
                    "loss": loss.item(),
                    "lr": lr_now,
                    "tflops": flops.tflops,
                    "epoch": epoch,
                    "sec": elapsed,
                }
                if use_ctc:
                    log_entry["lm_loss"] = lm_loss.item()
                    log_entry["ctc_loss"] = ctc_loss.item()
                log_history.append(log_entry)
                accelerator.log(log_entry, step=global_step)
                if use_ctc:
                    feat_std = audio_features.std().item()
                    print(
                        f"  step {global_step:6d} | loss {loss.item():.4f} "
                        f"(lm {lm_loss.item():.3f} + ctc {ctc_loss.item():.3f}) | "
                        f"feat_std {feat_std:.3f} | "
                        f"lr {lr_now:.2e} | TFLOPs {flops.tflops:.1f} | "
                        f"ep {epoch} | {elapsed:.0f}s"
                    )
                else:
                    print(
                        f"  step {global_step:6d} | loss {loss.item():.4f} | "
                        f"lr {lr_now:.2e} | TFLOPs {flops.tflops:.1f} | "
                        f"ep {epoch} | {elapsed:.0f}s"
                    )

            # ── Checkpoint ──
            if global_step % cfg.save_every == 0 and is_main:
                ckpt_path = run_dir / f"ckpt_{global_step}.pt"
                ckpt_data = {
                    "model": accelerator.unwrap_model(audio_encoder).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": global_step,
                    "cumulative_flops": flops.cumulative,
                    "config": vars(cfg),
                }
                if unfreeze_llm_layers > 0:
                    n_model_layers = len(llm.model.layers)
                    if unfreeze_llm_layers >= n_model_layers:
                        # Full LLM fine-tune: save entire state dict
                        ckpt_data["llm_full"] = llm.state_dict()
                    else:
                        ckpt_data["llm_partial"] = {
                            k: v for k, v in llm.state_dict().items()
                            if any(f"layers.{i}." in k for i in range(unfreeze_llm_layers))
                            or "embed_tokens" in k
                        }
                torch.save(ckpt_data, ckpt_path)
                print(f"  -> saved {ckpt_path}")

    # ── Final save ──
    if is_main:
        final_path = run_dir / "final.pt"
        final_data = {
            "model": accelerator.unwrap_model(audio_encoder).state_dict(),
            "step": global_step,
            "cumulative_flops": flops.cumulative,
            "config": vars(cfg),
        }
        if unfreeze_llm_layers > 0:
            n_model_layers = len(llm.model.layers)
            if unfreeze_llm_layers >= n_model_layers:
                final_data["llm_full"] = llm.state_dict()
            else:
                final_data["llm_partial"] = {
                    k: v for k, v in llm.state_dict().items()
                    if any(f"layers.{i}." in k for i in range(unfreeze_llm_layers))
                    or "embed_tokens" in k
                }
        torch.save(final_data, final_path)
        with open(run_dir / "train_log.json", "w") as f:
            json.dump(log_history, f, indent=2)
        print(f"\nDone! {global_step} steps, {flops.tflops:.1f} TFLOPs")
        print(f"Saved: {final_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
