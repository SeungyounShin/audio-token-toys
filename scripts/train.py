#!/usr/bin/env python3
"""Main training script: distill Whisper-base → student audio encoder."""

import argparse
import json
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from accelerate import Accelerator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import load_config
from src.data.dataset import LibriSpeechMelDataset, collate_fn
from src.model.student import StudentAudioEncoder
from src.model.teacher import TeacherWhisperEncoder
from src.training.distill_loss import DistillationLoss
from src.training.flop_counter import FLOPCounter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--config", required=True, help="Experiment config override")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
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

    if is_main:
        print(f"Config: {cfg.experiment_name}")
        print(f"  patch={cfg.patch_frames} frames ({cfg.patch_frames*10}ms), "
              f"hop={cfg.hop_frames} frames, overlap={cfg.overlap_pct:.0%}")
        print(f"  model: {cfg.n_layers}L, d={cfg.d_model}, heads={cfg.n_heads}")
        print(f"  output: {run_dir}")

    # ── Data ──
    train_ds = LibriSpeechMelDataset(str(Path(cfg.data_dir) / cfg.train_split))
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    if is_main:
        print(f"  dataset: {len(train_ds)} utterances, "
              f"batch={cfg.batch_size}/GPU × {accelerator.num_processes} GPUs")

    # ── Models ──
    student = StudentAudioEncoder.from_config(cfg)
    teacher = TeacherWhisperEncoder(cfg.teacher_model)

    if is_main:
        n_params = sum(p.numel() for p in student.parameters())
        print(f"  student params: {n_params/1e6:.1f}M")

    # ── Optimizer + Scheduler ──
    optimizer = AdamW(
        student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    warmup = LinearLR(optimizer, start_factor=1e-2, total_iters=cfg.warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cfg.total_steps - cfg.warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[cfg.warmup_steps])

    # ── Prepare ──
    student, optimizer, train_dl, scheduler = accelerator.prepare(
        student, optimizer, train_dl, scheduler
    )
    teacher = teacher.to(accelerator.device)

    loss_fn = DistillationLoss()
    flops = FLOPCounter(cfg.d_model, cfg.n_layers, cfg.d_ff)

    # ── Resume ──
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        accelerator.unwrap_model(student).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt["step"]
        flops.cumulative = ckpt.get("cumulative_flops", 0)
        if is_main:
            print(f"  resumed from step {global_step}")

    # ── Training loop ──
    log_history = []
    student.train()
    epoch = 0
    t0 = time.time()

    while global_step < cfg.total_steps:
        epoch += 1
        for batch in train_dl:
            if global_step >= cfg.total_steps:
                break

            mel = batch["mel"]          # [B, 80, T]
            n_frames = batch["n_frames"]  # [B]

            with accelerator.accumulate(student):
                # Student forward
                encoded, projected, n_tokens = student(mel, n_frames)

                # Teacher forward (frozen, no grad)
                with torch.no_grad():
                    teacher_hidden = teacher(mel)

                # Loss
                loss = loss_fn(projected, teacher_hidden, n_tokens)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # FLOP tracking
            if n_tokens is not None:
                total_tokens = accelerator.gather(n_tokens.sum()).sum().item()
            else:
                total_tokens = mel.shape[0] * projected.shape[1] * accelerator.num_processes
            flops.step(int(total_tokens))

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
                log_history.append(log_entry)
                accelerator.log(log_entry, step=global_step)
                print(
                    f"  step {global_step:6d} | loss {loss.item():.4f} | "
                    f"lr {lr_now:.2e} | TFLOPs {flops.tflops:.1f} | "
                    f"ep {epoch} | {elapsed:.0f}s"
                )

            # ── Save checkpoint ──
            if global_step % cfg.save_every == 0 and is_main:
                ckpt_path = run_dir / f"ckpt_{global_step}.pt"
                torch.save({
                    "model": accelerator.unwrap_model(student).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": global_step,
                    "cumulative_flops": flops.cumulative,
                    "config": vars(cfg),
                }, ckpt_path)
                print(f"  → saved {ckpt_path}")

    # ── Final save ──
    if is_main:
        final_path = run_dir / "final.pt"
        torch.save({
            "model": accelerator.unwrap_model(student).state_dict(),
            "step": global_step,
            "cumulative_flops": flops.cumulative,
            "config": vars(cfg),
        }, final_path)

        log_path = run_dir / "train_log.json"
        with open(log_path, "w") as f:
            json.dump(log_history, f, indent=2)

        print(f"\nDone! {global_step} steps, {flops.tflops:.1f} TFLOPs")
        print(f"Model: {final_path}")
        print(f"Log:   {log_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
