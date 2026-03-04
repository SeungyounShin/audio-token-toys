#!/usr/bin/env python3
"""Evaluate trained audio encoder via greedy LLM generation -> WER."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import load_config
from src.model.audio_encoder import AudioEncoder
from src.llm_bridge.injection import EmbeddingInjectionBridge
from src.evaluation.metrics import compute_wer


def evaluate_split(audio_encoder, bridge, split_name, cfg, device):
    """Evaluate one test split. Returns (wer, preds, refs)."""
    frontend_type = getattr(cfg, "frontend_type", "raw")

    if frontend_type == "raw":
        from src.data.wav_dataset import LibriSpeechWavDataset, wav_collate_fn
        ds = LibriSpeechWavDataset(split=split_name, sr=cfg.sr)
        collate = wav_collate_fn
    else:
        from src.data.dataset import LibriSpeechMelDataset, collate_fn
        ds = LibriSpeechMelDataset(str(Path(cfg.data_dir) / split_name))
        collate = collate_fn

    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=2)

    preds, refs = [], []
    audio_encoder.eval()

    for batch in dl:
        if frontend_type == "raw":
            audio_input = batch["wav"].to(device)
            n_lengths = batch["n_samples"].to(device)
        else:
            audio_input = batch["mel"].to(device)
            n_lengths = batch["n_frames"].to(device)
        text_ref = batch["texts"][0]

        with torch.no_grad():
            audio_features, n_tokens, _ = audio_encoder(audio_input, n_lengths)
            transcriptions = bridge.generate(
                audio_features,
                n_audio_tokens=n_tokens,
                max_new_tokens=cfg.max_text_len,
            )
        preds.append(transcriptions[0])
        refs.append(text_ref)

    wer = compute_wer(preds, refs)
    return wer, preds, refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/llm_base.yaml")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.base_config, args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.run_dir) / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load audio encoder
    audio_encoder = AudioEncoder.from_config(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    audio_encoder.load_state_dict(ckpt["model"])
    audio_encoder = audio_encoder.to(device)
    audio_encoder.eval()
    print(f"Loaded encoder: {args.checkpoint} (step {ckpt.get('step', '?')})")

    # Load frozen LLM + bridge
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading LLM: {cfg.llm_model}")
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.llm_model, torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model)
    # Restore fine-tuned LLM weights if saved in checkpoint
    if "llm_full" in ckpt:
        print(f"  Restoring full LLM state ({len(ckpt['llm_full'])} tensors)")
        llm.load_state_dict(ckpt["llm_full"])
    elif "llm_partial" in ckpt:
        print(f"  Restoring {len(ckpt['llm_partial'])} unfrozen LLM param tensors")
        llm.load_state_dict(ckpt["llm_partial"], strict=False)
    llm.eval()
    bridge = EmbeddingInjectionBridge(llm, tokenizer, max_text_len=cfg.max_text_len)

    # Evaluate each test split
    results = {
        "checkpoint": args.checkpoint,
        "step": ckpt.get("step"),
        "tflops": ckpt.get("cumulative_flops", 0) / 1e12,
        "config": cfg.experiment_name,
    }

    for split in cfg.test_splits:
        print(f"\nEvaluating {split}...")
        wer, preds, refs = evaluate_split(audio_encoder, bridge, split, cfg, device)
        results[f"wer_{split}"] = wer
        print(f"  {split}: WER = {wer:.2%} ({len(refs)} samples)")

        for i in range(min(5, len(preds))):
            print(f"    ref:  {refs[i]}")
            print(f"    pred: {preds[i]}")
            print()

    out_path = run_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
