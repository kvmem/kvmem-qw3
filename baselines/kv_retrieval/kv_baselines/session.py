"""One persistent text session with explicit, safe prefix reuse accounting."""
import torch

from .infllm_hybrid import forward_text, make_cache
from .session_state import CheckpointRing


class InfLLMSession:
    def __init__(self, model, infllm_config, prefill_chunk=512, checkpoint_capacity=3,
                 tail_guard=32):
        if prefill_chunk < 1 or tail_guard < 1:
            raise ValueError("chunk and tail guard must be positive")
        self.model = model
        self.config = infllm_config
        self.prefill_chunk = prefill_chunk
        self.tail_guard = tail_guard
        self.device = next(model.parameters()).device
        self.cache = make_cache(model.config, infllm_config)
        self.tokens = []
        self.logits = None
        self.checkpoints = CheckpointRing(checkpoint_capacity)

    def _forward(self, tokens):
        ids = torch.tensor([tokens], device=self.device, dtype=torch.long)
        output = forward_text(self.model, ids, self.cache, logits_to_keep=1)
        self.tokens.extend(tokens)
        self.logits = output.logits[:, -1, :]
        if self.cache.get_seq_length() != len(self.tokens):
            raise RuntimeError("authoritative token/cache length mismatch")

    @torch.inference_mode()
    def begin(self, prompt_tokens):
        prompt_tokens = list(prompt_tokens)
        if not prompt_tokens:
            raise ValueError("empty prompt")
        previous_length = len(self.tokens)
        if prompt_tokens[:previous_length] == self.tokens and len(prompt_tokens) >= previous_length:
            origin = previous_length
            reason = "live_prefix" if previous_length else "initial"
        else:
            restored, origin = self.checkpoints.restore_for(prompt_tokens)
            self.cache = restored if restored is not None else make_cache(self.model.config, self.config)
            self.tokens = prompt_tokens[:origin]
            self.logits = None
            reason = "checkpoint" if restored is not None else "no_matching_checkpoint"
        # Preserve a snapshot on an existing prefill-chunk boundary. Capturing
        # state does not split the original InfLLM execution chunk into new ones.
        # Short appends after an unaligned live prefix retain the previous safe
        # snapshot instead of overwriting it with an unsafe prompt-end state.
        safe_end = len(prompt_tokens) - self.tail_guard
        final_boundary = origin + ((safe_end - origin) // self.prefill_chunk) * self.prefill_chunk
        for start in range(origin, len(prompt_tokens), self.prefill_chunk):
            self._forward(prompt_tokens[start:start + self.prefill_chunk])
            if len(self.tokens) == final_boundary and final_boundary > origin:
                self.checkpoints.capture(self.tokens, self.cache)
        if self.logits is None:
            raise RuntimeError("no logits available for resumed prompt")
        return self.logits, {"resume_origin": origin, "prompt_tokens": len(prompt_tokens),
            "prefill_tokens": len(prompt_tokens) - origin, "reuse_reason": reason,
            "cold_prefill": origin == 0, "unexpected_cold_prefill": origin == 0 and previous_length > 0,
            "previous_cache_tokens": previous_length,
            "checkpoint_positions": [len(c.tokens) for c in self.checkpoints.checkpoints]}

    @torch.inference_mode()
    def consume_generated_token(self, token_id):
        """Commit exactly one generated token before obtaining the next logits."""
        self._forward([int(token_id)])
        return self.logits
