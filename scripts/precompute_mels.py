#!/usr/bin/env python3
"""Precompute Whisper-compatible mel spectrograms from LibriSpeech.

Downloads LibriSpeech via HuggingFace datasets and saves per-utterance .pt files.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.mel_extract import MelExtractor


SPLIT_MAP = {
    "train.100": "train.clean.100",
    "train.360": "train.clean.360",
    "train.other": "train.other.500",
    "dev.clean": "validation.clean",
    "dev.other": "validation.other",
    "test.clean": "test.clean",
    "test.other": "test.other",
}


def precompute_split(split_name: str, out_dir: Path, extractor: MelExtractor):
    hf_split = SPLIT_MAP[split_name]
    print(f"\n{'='*60}")
    print(f"Processing: {split_name} (HF split: {hf_split})")
    print(f"{'='*60}")

    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("openslr/librispeech_asr", split=hf_split)

    count = 0
    skipped = 0
    for i, sample in enumerate(ds):
        audio = sample["audio"]["array"].astype(np.float32)
        sr = sample["audio"]["sampling_rate"]
        text = sample["text"].lower().strip()
        uid = sample["id"]

        # Skip very short clips (< 0.5s)
        if len(audio) < sr * 0.5:
            skipped += 1
            continue

        mel = extractor(audio, sr=sr)
        n_frames = mel.shape[1]

        torch.save(
            {"mel": mel, "n_frames": n_frames, "text": text},
            split_dir / f"{uid}.pt",
        )
        count += 1
        if count % 2000 == 0:
            print(f"  {count} saved ({skipped} skipped)...")

    print(f"  Done: {count} utterances saved, {skipped} skipped")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits", nargs="+",
        default=["train.100", "test.clean", "test.other"],
        help="Splits to precompute",
    )
    parser.add_argument(
        "--out_dir", type=str, default="data/mels",
        help="Output directory for mel .pt files",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    extractor = MelExtractor()
    print(f"Output: {out_dir.resolve()}")

    total = 0
    for split in args.splits:
        if split not in SPLIT_MAP:
            print(f"Unknown split: {split}, skipping. Valid: {list(SPLIT_MAP.keys())}")
            continue
        total += precompute_split(split, out_dir, extractor)

    print(f"\nAll done! Total: {total} utterances in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
