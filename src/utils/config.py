"""YAML config loader with experiment override support."""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Config:
    # Experiment
    experiment_name: str = "default"
    patch_frames: int = 4
    hop_frames: int = 4

    # Frontend
    frontend_type: str = "raw"  # "raw" or "mel"

    # Model
    n_mels: int = 80
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 1500
    dropout: float = 0.1
    teacher_model: str = "openai/whisper-base"
    teacher_dim: int = 512

    # LLM decoder
    llm_model: str = "Qwen/Qwen3-0.6B"
    llm_dim: int = 1024
    causal: bool = True
    max_text_len: int = 128

    # CTC auxiliary loss
    use_ctc: bool = False
    ctc_weight: float = 0.3
    ctc_warmup_steps: int = 0  # CTC-only warmup before joint training

    # Use 50Hz features directly (skip 4x downsample)
    use_50hz: bool = False

    # LLM partial unfreezing
    unfreeze_llm_layers: int = 0  # number of first LLM layers to unfreeze (0=fully frozen)

    # Separate learning rates (0 = use base lr)
    encoder_lr: float = 0.0   # conv + CTC head
    projector_lr: float = 0.0  # audio → LLM projector

    # Data
    data_dir: str = "data/mels"
    train_split: str = "train.100"
    test_splits: List[str] = field(default_factory=lambda: ["test.clean", "test.other"])
    sr: int = 16000
    datasets: list = field(default_factory=list)  # multi-dataset config (list of dicts)
    max_audio_duration: float = 20.0  # max utterance duration in seconds

    # Training
    batch_size: int = 32
    grad_accum: int = 1
    lr: float = 3e-4
    warmup_steps: int = 1000
    total_steps: int = 50000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"

    # Checkpointing
    save_every: int = 5000
    eval_every: int = 2500
    log_every: int = 50
    output_dir: str = "outputs"
    wandb_project: str = "audio-encoder-scaling"

    # Evaluation
    probe_lr: float = 1e-3
    probe_epochs: int = 20
    probe_batch_size: int = 64
    vocab_size: int = 29

    @property
    def overlap_pct(self) -> float:
        return 1.0 - (self.hop_frames / self.patch_frames)

    @property
    def patch_dim(self) -> int:
        return self.n_mels * self.patch_frames

    @property
    def run_dir(self) -> str:
        return str(Path(self.output_dir) / self.experiment_name)


def load_config(base_path: str, override_path: Optional[str] = None) -> Config:
    """Load base config and optionally override with experiment-specific values."""
    with open(base_path) as f:
        base = yaml.safe_load(f)

    if override_path:
        with open(override_path) as f:
            override = yaml.safe_load(f)
        base.update(override)

    return Config(**{k: v for k, v in base.items() if hasattr(Config, k)})
