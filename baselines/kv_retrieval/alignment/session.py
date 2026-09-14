"""Release discarded-branch GPU pools before allocating restored pools."""
import gc
import torch
from kv_baselines.session import InfLLMSession as OriginalSession


class AlignedSession(OriginalSession):
    def begin(self,prompt_tokens):
        prompt_tokens=list(prompt_tokens)
        if prompt_tokens[:len(self.tokens)] != self.tokens:
            # Checkpoints own immutable CPU history and independent recurrent
            # state. The old mutable branch is ineligible and must not double
            # the global buffer/cache allocation while restore constructs anew.
            torch.cuda.synchronize()
            self.cache=None
            self.logits=None
            gc.collect()
        return super().begin(prompt_tokens)
