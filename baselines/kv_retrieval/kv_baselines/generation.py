"""Explicit non-speculative sampling and the native reasoning-budget policy.

Sampling uses PyTorch RNG, not native std::mt19937_64. Same seed/distribution
parameters do not imply identical random draws across those backends.
"""
from collections import deque
import torch

GUIDANCE = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n"


def sample(logits, generator, *, temperature, top_p, top_k, min_p=0,
           presence_penalty=0, repetition_penalty=1, seen=()):
    scores = logits.reshape(-1).float()
    if presence_penalty or repetition_penalty != 1:
        scores = scores.clone()
        ids = torch.tensor(list(set(seen)), device=scores.device, dtype=torch.long)
        old = scores[ids]
        scores[ids] = torch.where(old >= 0, old / repetition_penalty, old * repetition_penalty) - presence_penalty
    if temperature <= 0:
        return int(scores.argmax().item())
    # Native top-p uses original full-vocabulary mass, even after top-k/min-p.
    # Renormalizing at each intermediate filter would yield a different policy.
    probabilities = torch.softmax(scores / temperature, dim=-1)
    k = min(top_k, probabilities.numel()) if top_k > 0 else probabilities.numel()
    values, indices = torch.topk(probabilities, k, sorted=True)
    keep = torch.ones_like(values, dtype=torch.bool)
    if min_p > 0:
        keep &= values >= values[0] * min_p
    if 0 < top_p < 1:
        keep &= (torch.cumsum(values, 0) - values) < top_p
    values = torch.where(keep, values, 0)
    picked = torch.multinomial(values, 1, generator=generator)
    return int(indices[picked].item())


class ThinkingBudget:
    def __init__(self, bridge, enabled, budget, recover_eos, eos_id):
        self.open = enabled
        self.budget = max(0, budget)
        self.recover_eos = recover_eos
        self.eos_id = eos_id
        ids = bridge.call("tokenize", text="</think>")["input_ids"]
        if len(ids) != 1:
            raise ValueError("thinking close tag must have one token")
        self.close_id = ids[0]
        self.guidance = bridge.call("tokenize", text=GUIDANCE)["input_ids"]
        self.queue = deque()
        self.count = 0
        self.forced = False
        self.forced_tokens = 0

    def apply(self, proposed):
        if self.open and self.recover_eos and proposed == self.eos_id:
            self.open = False
            self.queue.clear()
            return self.close_id
        if self.open and self.budget > 0 and self.count >= self.budget and not self.forced:
            self.queue.extend([t for t in self.guidance if t != self.close_id] + [self.close_id])
            self.forced = True
        if self.queue:
            proposed = self.queue.popleft()
            self.forced_tokens += 1
        if proposed == self.close_id:
            self.open = False
            self.queue.clear()
        elif self.open:
            self.count += 1
        return proposed
