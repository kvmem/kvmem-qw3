"""Isolated Q8_0 + FP8 entry point; reuse the validated Messages server."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import serve_infllm as server
from alignment.q8_model import load_q8_model
from alignment.fp8_kv import install_fp8_storage
from alignment.session import AlignedSession
from run_model_smoke import save

ROOT = Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--gguf',type=Path,required=True)
    extra,rest=p.parse_known_args()
    sys.argv=[sys.argv[0],*rest]
    torch.set_num_threads(4)
    install_fp8_storage()
    server.InfLLMSession=AlignedSession
    server.load_text_model=lambda directory: load_q8_model(directory,extra.gguf)
    original=server.Application

    class AlignedApplication(original):
        def __init__(self,args):
            super().__init__(args)
            self.manifest.update(
                method='InfLLM Q8_0 / FP8 (author algorithm, Qwen hybrid adapter)',
                active_window_capacity=self.config.n_init+self.config.n_local+self.config.topk*self.config.block_size,
                kv_storage_dtype='FP8 E4M3FN, raw scale=1, saturate finite',
                kv_storage_scope='CPU history, GPU block cache, persistent local/remainder/init KV',
                attention_computation_dtype='FP16; assembled global attention buffer is FP16',
                comparison_limitations=['native old arms use MTP chain4; this InfLLM adapter has no MTP',
                    'HF/FLA BF16 intermediate activations versus native FP32 residual',
                    'same numeric seed; sampling RNG implementations differ',
                    'capacity ceiling alignment, not identical per-step working sets'])
            files=[Path(__file__), *sorted((ROOT/'alignment').glob('*.py')),
                   ROOT/'compat_build/native_q8/libaligned_q8.so',args.infllm_config]
            self.manifest['alignment_source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
            save(args.out/'manifest.json',self.manifest)

        def record(self,row):
            row.update(active_window_capacity=self.config.n_init+self.config.n_local+self.config.topk*self.config.block_size,
                       kv_storage_dtype='float8_e4m3fn',weight_dtype='Q8_0')
            return super().record(row)

    server.Application=AlignedApplication
    server.main()


if __name__=='__main__': main()
