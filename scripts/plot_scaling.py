#!/usr/bin/env python3
"""Generate scaling law plots from training logs and evaluation results."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Korean font (if available)
KR_FONT = "/home/robin/.local/share/fonts/NotoSansCJKkr-Regular.otf"
try:
    from matplotlib.font_manager import FontProperties, fontManager
    if Path(KR_FONT).exists():
        fontManager.addfont(KR_FONT)
        fp = FontProperties(fname=KR_FONT)
        matplotlib.rcParams["font.sans-serif"] = [fp.get_name()] + matplotlib.rcParams["font.sans-serif"]
except Exception:
    pass
matplotlib.rcParams["axes.unicode_minus"] = False

CONFIGS = {
    "40ms_0pct":  {"label": "40ms / 0% overlap",  "color": "#2196F3", "marker": "o"},
    "40ms_50pct": {"label": "40ms / 50% overlap", "color": "#FF9800", "marker": "s"},
    "80ms_0pct":  {"label": "80ms / 0% overlap",  "color": "#4CAF50", "marker": "^"},
    "80ms_50pct": {"label": "80ms / 50% overlap", "color": "#E91E63", "marker": "D"},
}

OUT_DIR = Path("outputs")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def load_train_log(config_name):
    path = OUT_DIR / config_name / "train_log.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_eval_results(config_name):
    path = OUT_DIR / config_name / "eval" / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_loss_vs_tflops():
    """Figure 1: Distillation loss vs cumulative TFLOPs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, style in CONFIGS.items():
        log = load_train_log(name)
        if log is None:
            continue
        tflops = [e["tflops"] for e in log]
        loss = [e["loss"] for e in log]
        ax.plot(tflops, loss, color=style["color"], marker=style["marker"],
                markevery=max(1, len(tflops) // 15), markersize=5,
                label=style["label"], linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Cumulative TFLOPs", fontsize=12)
    ax.set_ylabel("Distillation Loss (MSE)", fontsize=12)
    ax.set_title("Training Efficiency: Loss vs Compute", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "loss_vs_tflops.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'loss_vs_tflops.png'}")


def plot_loss_vs_steps():
    """Figure 2: Loss vs training steps."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, style in CONFIGS.items():
        log = load_train_log(name)
        if log is None:
            continue
        steps = [e["step"] for e in log]
        loss = [e["loss"] for e in log]
        ax.plot(steps, loss, color=style["color"], label=style["label"],
                linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Distillation Loss (MSE)", fontsize=12)
    ax.set_title("Loss Curves by Configuration", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "loss_vs_steps.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'loss_vs_steps.png'}")


def plot_wer_comparison():
    """Figure 3: WER comparison bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, split in enumerate(["test.clean", "test.other"]):
        ax = axes[idx]
        names = []
        wers = []
        colors = []
        for name, style in CONFIGS.items():
            results = load_eval_results(name)
            if results is None:
                continue
            wer_key = f"wer_{split}"
            if wer_key in results:
                names.append(style["label"])
                wers.append(results[wer_key] * 100)
                colors.append(style["color"])

        if names:
            bars = ax.bar(range(len(names)), wers, color=colors, alpha=0.85)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("WER (%)", fontsize=11)
            ax.set_title(split.replace(".", "-"), fontsize=12)
            # Value labels
            for bar, w in zip(bars, wers):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{w:.1f}%", ha="center", fontsize=9)
            ax.grid(axis="y", alpha=0.2)

    fig.suptitle("WER by Configuration (Linear CTC Probe)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "wer_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'wer_comparison.png'}")


def plot_efficiency_summary():
    """Figure 4: TFLOPs at final step vs final loss — scatter."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, style in CONFIGS.items():
        log = load_train_log(name)
        results = load_eval_results(name)
        if log is None:
            continue
        final = log[-1]
        tflops = final["tflops"]
        loss = final["loss"]

        wer_text = ""
        if results and "wer_test.clean" in results:
            wer_text = f"\nWER={results['wer_test.clean']*100:.1f}%"

        ax.scatter(tflops, loss, color=style["color"], marker=style["marker"],
                   s=120, zorder=5, edgecolors="white", linewidths=1)
        ax.annotate(style["label"] + wer_text,
                    (tflops, loss), textcoords="offset points",
                    xytext=(10, 5), fontsize=9)

    ax.set_xlabel("Total TFLOPs", fontsize=12)
    ax.set_ylabel("Final Distillation Loss", fontsize=12)
    ax.set_title("Compute Efficiency: Final Loss vs Total Compute", fontsize=14)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "efficiency_summary.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'efficiency_summary.png'}")


def print_summary_table():
    """Print results summary."""
    print("\n" + "=" * 70)
    print(f"{'Config':<20s} {'Steps':>6s} {'TFLOPs':>8s} {'Loss':>8s} "
          f"{'WER-clean':>10s} {'WER-other':>10s}")
    print("-" * 70)

    for name in CONFIGS:
        log = load_train_log(name)
        results = load_eval_results(name)
        if log is None:
            print(f"  {name:<20s}  (no data)")
            continue

        final = log[-1]
        wer_c = f"{results['wer_test.clean']*100:.1f}%" if results and "wer_test.clean" in results else "N/A"
        wer_o = f"{results['wer_test.other']*100:.1f}%" if results and "wer_test.other" in results else "N/A"

        print(f"  {name:<20s} {final['step']:>6d} {final['tflops']:>8.1f} "
              f"{final['loss']:>8.4f} {wer_c:>10s} {wer_o:>10s}")

    print("=" * 70)


if __name__ == "__main__":
    print("Generating scaling law figures...\n")
    plot_loss_vs_tflops()
    plot_loss_vs_steps()
    plot_wer_comparison()
    plot_efficiency_summary()
    print_summary_table()
