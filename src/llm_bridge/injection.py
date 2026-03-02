"""Chat-template embedding injection bridge for frozen LLM.

Follows Qwen3 instruct chat format:

  <|im_start|>system
  You are a speech recognition assistant.<|im_end|>
  <|im_start|>user
  [audio_feat × N] Transcribe this audio.<|im_end|>
  <|im_start|>assistant
  {transcription}<|im_end|>

Loss is computed only on the assistant's transcription tokens.

Embedding layout:
  [prefix_embeds | audio_feats | suffix_embeds | target_embeds | pad...]
  prefix = "<|im_start|>system\n....<|im_end|>\n<|im_start|>user\n"
  suffix = "Transcribe this audio.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
  target = "{transcription}<|im_end|>"

Labels:
  [-100 × (prefix + audio + suffix) | target_ids | -100 × pad]
"""

import torch
import torch.nn as nn
from typing import List, Optional

SYSTEM_PROMPT = "You are a speech recognition assistant."
USER_SUFFIX = "Transcribe this audio."


class EmbeddingInjectionBridge(nn.Module):
    """Frozen LLM with chat-template audio injection.

    Args:
        llm: Frozen CausalLM instance.
        tokenizer: Corresponding tokenizer.
        max_text_len: Maximum transcription token length.
        system_prompt: System message content.
    """

    def __init__(
        self,
        llm: nn.Module,
        tokenizer,
        max_text_len: int = 128,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.eval()

        # Pre-tokenize fixed template parts
        # prefix: everything before audio features
        prefix_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n"
        )
        # suffix: everything between audio features and transcription
        # Empty <think> block disables Qwen3 extended thinking
        suffix_text = (
            f"{USER_SUFFIX}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n"
        )
        self._prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        self._suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
        self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def train(self, mode: bool = True):
        super().train(mode)
        self.llm.eval()
        return self

    def _build_sequence(
        self,
        audio_features: torch.Tensor,
        texts: List[str],
        n_audio_tokens: Optional[torch.Tensor],
    ):
        """Build inputs_embeds, attention_mask, labels for the full chat sequence.

        Returns:
            inputs_embeds: [B, L_total, llm_dim]
            attention_mask: [B, L_total]
            labels: [B, L_total]
        """
        B, N_audio, llm_dim = audio_features.shape
        device = audio_features.device
        tok = self.tokenizer
        embed_fn = self.llm.get_input_embeddings()

        # Fixed template token ids → embeddings (shared across batch)
        prefix_ids_t = torch.tensor(self._prefix_ids, dtype=torch.long, device=device)
        suffix_ids_t = torch.tensor(self._suffix_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            prefix_embeds = embed_fn(prefix_ids_t).unsqueeze(0).expand(B, -1, -1)
            suffix_embeds = embed_fn(suffix_ids_t).unsqueeze(0).expand(B, -1, -1)

        N_prefix = len(self._prefix_ids)
        N_suffix = len(self._suffix_ids)

        # Tokenize each transcription target: "{text}<|im_end|>"
        all_target_ids = []
        for text in texts:
            ids = tok.encode(text, add_special_tokens=False)
            ids = ids[: self.max_text_len - 1]
            ids.append(self._im_end_id)
            all_target_ids.append(ids)

        target_lengths = torch.tensor(
            [len(ids) for ids in all_target_ids], device=device
        )
        T_max = target_lengths.max().item()

        # Pad targets
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else self._im_end_id
        target_ids_padded = torch.full(
            (B, T_max), pad_id, dtype=torch.long, device=device
        )
        for i, ids in enumerate(all_target_ids):
            target_ids_padded[i, : len(ids)] = torch.tensor(
                ids, dtype=torch.long, device=device
            )

        with torch.no_grad():
            target_embeds = embed_fn(target_ids_padded)  # [B, T_max, llm_dim]

        # Concatenate: [prefix | audio | suffix | target]
        inputs_embeds = torch.cat(
            [prefix_embeds, audio_features, suffix_embeds, target_embeds], dim=1
        )

        # Total length breakdown
        L_total = N_prefix + N_audio + N_suffix + T_max
        N_context = N_prefix + N_audio + N_suffix  # everything before target

        # Labels: -100 for context, target_ids for transcription, -100 for padding
        labels_context = torch.full(
            (B, N_context), -100, dtype=torch.long, device=device
        )
        labels_target = target_ids_padded.clone()
        idx_t = torch.arange(T_max, device=device).unsqueeze(0)
        labels_target[idx_t >= target_lengths.unsqueeze(1)] = -100
        labels = torch.cat([labels_context, labels_target], dim=1)

        # Attention mask: 1 for real, 0 for padding
        # prefix and suffix are always real
        attn_prefix = torch.ones(B, N_prefix, dtype=torch.long, device=device)
        attn_suffix = torch.ones(B, N_suffix, dtype=torch.long, device=device)

        if n_audio_tokens is not None:
            idx_a = torch.arange(N_audio, device=device).unsqueeze(0)
            attn_audio = (idx_a < n_audio_tokens.unsqueeze(1)).long()
        else:
            attn_audio = torch.ones(B, N_audio, dtype=torch.long, device=device)

        attn_target = (idx_t < target_lengths.unsqueeze(1)).long()
        attention_mask = torch.cat(
            [attn_prefix, attn_audio, attn_suffix, attn_target], dim=1
        )

        return inputs_embeds, attention_mask, labels

    def forward(
        self,
        audio_features: torch.Tensor,
        texts: List[str],
        n_audio_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            audio_features: [B, N_audio, llm_dim] from AudioEncoder.
            texts: List[str] of B transcription strings.
            n_audio_tokens: [B] valid audio token counts.

        Returns:
            loss: scalar CE loss (transcription tokens only).
        """
        inputs_embeds, attention_mask, labels = self._build_sequence(
            audio_features, texts, n_audio_tokens
        )

        llm_dtype = next(self.llm.parameters()).dtype
        outputs = self.llm(
            inputs_embeds=inputs_embeds.to(llm_dtype),
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        audio_features: torch.Tensor,
        n_audio_tokens: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
    ) -> List[str]:
        """Generate transcriptions from audio features.

        Args:
            audio_features: [B, N_audio, llm_dim]
            n_audio_tokens: [B] valid audio token counts.
            max_new_tokens: Max tokens to generate.

        Returns:
            List of transcription strings.
        """
        B, N_audio, llm_dim = audio_features.shape
        device = audio_features.device
        embed_fn = self.llm.get_input_embeddings()

        prefix_ids_t = torch.tensor(self._prefix_ids, dtype=torch.long, device=device)
        suffix_ids_t = torch.tensor(self._suffix_ids, dtype=torch.long, device=device)
        prefix_embeds = embed_fn(prefix_ids_t).unsqueeze(0).expand(B, -1, -1)
        suffix_embeds = embed_fn(suffix_ids_t).unsqueeze(0).expand(B, -1, -1)

        llm_dtype = next(self.llm.parameters()).dtype
        inputs_embeds = torch.cat(
            [prefix_embeds, audio_features, suffix_embeds], dim=1
        ).to(llm_dtype)

        # Attention mask
        N_prefix = len(self._prefix_ids)
        N_suffix = len(self._suffix_ids)
        attn_prefix = torch.ones(B, N_prefix, dtype=torch.long, device=device)
        attn_suffix = torch.ones(B, N_suffix, dtype=torch.long, device=device)
        if n_audio_tokens is not None:
            idx_a = torch.arange(N_audio, device=device).unsqueeze(0)
            attn_audio = (idx_a < n_audio_tokens.unsqueeze(1)).long()
        else:
            attn_audio = torch.ones(B, N_audio, dtype=torch.long, device=device)
        attention_mask = torch.cat(
            [attn_prefix, attn_audio, attn_suffix], dim=1
        )

        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self._im_end_id,
            eos_token_id=self._im_end_id,
        )

        results = []
        for i in range(B):
            text = self.tokenizer.decode(output_ids[i], skip_special_tokens=True)
            results.append(text.strip())
        return results
