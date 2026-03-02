"""Lightweight FLOP counter for compute-matched experiment comparison."""


class FLOPCounter:
    """Tracks cumulative training FLOPs (forward + backward).

    Approximation: per token per layer ≈ 6 * (4*d² + 2*d*d_ff)
    (factor 6 = 2x forward ops * 3x for fwd+bwd)
    """

    def __init__(self, d_model: int, n_layers: int, d_ff: int):
        self.flops_per_token_per_layer = 6 * (4 * d_model**2 + 2 * d_model * d_ff)
        self.n_layers = n_layers
        self.cumulative = 0

    def step(self, total_tokens: int) -> int:
        """Record one training step. Returns FLOPs for this step."""
        flops = self.flops_per_token_per_layer * self.n_layers * total_tokens
        self.cumulative += flops
        return flops

    @property
    def tflops(self) -> float:
        """Cumulative TFLOPs."""
        return self.cumulative / 1e12


class LLMFLOPCounter:
    """FLOP counter for audio encoder (trainable, fwd+bwd) + LLM (frozen, fwd only).

    Encoder fwd+bwd: 6 * (4*d^2 + 2*d*d_ff) per token per layer
    LLM fwd only:    2 * (4*d^2 + 2*d*d_ff) per token per layer
    """

    def __init__(
        self,
        enc_d_model: int,
        enc_n_layers: int,
        enc_d_ff: int,
        llm_d_model: int,
        llm_n_layers: int,
        llm_d_ff: int,
    ):
        self.enc_flops_per_token = 6 * (4 * enc_d_model**2 + 2 * enc_d_model * enc_d_ff)
        self.enc_n_layers = enc_n_layers
        self.llm_flops_per_token = 2 * (4 * llm_d_model**2 + 2 * llm_d_model * llm_d_ff)
        self.llm_n_layers = llm_n_layers
        self.cumulative = 0

    def step(self, total_audio_tokens: int, total_text_tokens: int) -> int:
        """Record one step. Returns FLOPs for this step."""
        enc_flops = self.enc_flops_per_token * self.enc_n_layers * total_audio_tokens
        llm_flops = (
            self.llm_flops_per_token
            * self.llm_n_layers
            * (total_audio_tokens + total_text_tokens)
        )
        flops = enc_flops + llm_flops
        self.cumulative += flops
        return flops

    @property
    def tflops(self) -> float:
        return self.cumulative / 1e12
