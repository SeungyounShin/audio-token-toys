#!/usr/bin/env python3
"""Evaluate a trained encoder: train linear CTC probe → decode → WER."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import load_config
from src.data.dataset import LibriSpeechMelDataset, collate_fn
from src.model.student import StudentAudioEncoder
from src.evaluation.linear_probe import LinearCTCProbe
from src.evaluation.ctc_decode import greedy_ctc_decode
from src.evaluation.metrics import compute_wer


def train_probe(probe, train_dl, cfg, device):
    """Train the linear CTC head."""
    optimizer = AdamW(probe.head.parameters(), lr=cfg.probe_lr)
    probe.train()

    for epoch in range(1, cfg.probe_epochs + 1):
        total_loss = 0
        n_batches = 0
        for batch in train_dl:
            mel = batch["mel"].to(device)
            n_frames = batch["n_frames"].to(device)
            texts = batch["texts"]

            log_probs, n_tokens = probe(mel, n_frames)
            targets, target_lengths = LinearCTCProbe.text_to_targets(texts)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            # Skip batch if any target is longer than output
            if (target_lengths > n_tokens.to(target_lengths.device)).any():
                continue

            loss = probe.compute_loss(log_probs, n_tokens, targets, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.head.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if n_batches > 0:
            print(f"  probe epoch {epoch:2d} | loss {total_loss / n_batches:.4f}")


def evaluate_split(probe, test_dl, device):
    """Run CTC decode on a test split and return WER."""
    probe.eval()
    all_preds = []
    all_refs = []

    with torch.no_grad():
        for batch in test_dl:
            mel = batch["mel"].to(device)
            n_frames = batch["n_frames"].to(device)

            log_probs, n_tokens = probe(mel, n_frames)
            decoded = greedy_ctc_decode(log_probs, n_tokens)

            all_preds.extend(decoded)
            all_refs.extend(batch["texts"])

    wer = compute_wer(all_preds, all_refs)
    return wer, all_preds, all_refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.base_config, args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.run_dir) / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder
    encoder = StudentAudioEncoder.from_config(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder.load_state_dict(ckpt["model"])
    encoder = encoder.to(device)
    print(f"Loaded checkpoint: {args.checkpoint} (step {ckpt.get('step', '?')})")

    # Build probe
    probe = LinearCTCProbe(encoder, d_model=cfg.d_model, vocab_size=cfg.vocab_size)
    probe = probe.to(device)

    # Train probe on train split
    print(f"\nTraining linear CTC probe on {cfg.train_split}...")
    train_ds = LibriSpeechMelDataset(str(Path(cfg.data_dir) / cfg.train_split))
    train_dl = DataLoader(
        train_ds, batch_size=cfg.probe_batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=2,
    )
    train_probe(probe, train_dl, cfg, device)

    # Evaluate on test splits
    results = {}
    for split in cfg.test_splits:
        split_dir = Path(cfg.data_dir) / split
        if not split_dir.exists():
            print(f"  Skipping {split} (not found)")
            continue

        test_ds = LibriSpeechMelDataset(str(split_dir))
        test_dl = DataLoader(
            test_ds, batch_size=cfg.probe_batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=2,
        )

        wer, preds, refs = evaluate_split(probe, test_dl, device)
        results[split] = {"wer": wer, "n_samples": len(refs)}
        print(f"\n  {split}: WER = {wer:.2%} ({len(refs)} samples)")

        # Show a few examples
        for i in range(min(3, len(preds))):
            print(f"    ref:  {refs[i]}")
            print(f"    pred: {preds[i]}")
            print()

    # Save results
    out_path = run_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "step": ckpt.get("step"),
            "tflops": ckpt.get("cumulative_flops", 0) / 1e12,
            "config": cfg.experiment_name,
            **{f"wer_{k}": v["wer"] for k, v in results.items()},
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
