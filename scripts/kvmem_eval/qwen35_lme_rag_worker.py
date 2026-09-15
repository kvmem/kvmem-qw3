#!/usr/bin/env python3
"""CPU-only, pinned Jina indexing/query worker for the 9B memory-QA pilot."""
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REFERENCE = Path('/home/chaidi/AgentLongBench-Long/script/compactRag/run_three_round_compact_rag.py')
MODEL = '/data/chaidi/kvmem_eval/models/jinaai/jina-embeddings-v2-small-en'
TOKENIZER = '/home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/Qwen3.6-27B-FP8'


def main():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    spec = importlib.util.spec_from_file_location('lme_pinned_rag', REFERENCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
        retriever = module.RagRetriever(MODEL, TOKENIZER, device='cpu', batch_size=8)
    print(json.dumps({'ready': True, 'reference_sha256': hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
                      'config': retriever.config}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request['op'] == 'quit':
                return
            directory = Path(request['directory'])
            if request['op'] == 'index':
                start = time.perf_counter()
                chunks = []
                for session_index, session in enumerate(request['sessions']):
                    blocks, _ = retriever.chunks(session['text'], 1024, 128)
                    for block in blocks:
                        block['session_index'] = session_index
                        block['date'] = session['date']
                        block['block_id'] = f's{session_index:04d}_c{block["chunk_index"]:04d}'
                        block['embedding_text'] = f"Conversation date: {session['date']}\n{block['text']}"
                        chunks.append(block)
                texts = [block['embedding_text'] for block in chunks]
                lengths = [len(ids) for ids in retriever.model.tokenizer(
                    texts, add_special_tokens=True, truncation=False)['input_ids']]
                assert chunks and max(lengths) <= module.JINA_MAX_SEQ_LENGTH, 'Embedding truncation'
                with contextlib.redirect_stdout(sys.stderr):
                    embeddings = retriever.model.encode(texts, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False, batch_size=8)
                assert np.isfinite(embeddings).all()
                directory.mkdir(parents=True, exist_ok=True)
                np.save(directory / 'embeddings.npy', embeddings)
                result = {'chunks': chunks, 'max_embedding_tokens': max(lengths),
                          'history_index_sec': time.perf_counter() - start,
                          'config_sha256': retriever.config_sha256}
                (directory / 'index.json').write_text(json.dumps(result, ensure_ascii=False) + '\n')
                print(json.dumps({'ok': True, 'chunks': len(chunks),
                                  'history_index_sec': result['history_index_sec']}), flush=True)
            elif request['op'] == 'query':
                start = time.perf_counter()
                embeddings = np.load(directory / 'embeddings.npy')
                length = len(retriever.model.tokenizer(request['question'], truncation=False)['input_ids'])
                assert length <= module.JINA_MAX_SEQ_LENGTH
                with contextlib.redirect_stdout(sys.stderr):
                    query = retriever.model.encode([request['question']], normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False, batch_size=1)[0]
                scores = embeddings @ query
                assert np.isfinite(scores).all()
                ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
                print(json.dumps({'ok': True, 'ranked': ranked, 'scores': scores.tolist(),
                                  'query_sec': time.perf_counter() - start}), flush=True)
            else:
                raise ValueError(request['op'])
        except Exception as exc:
            print(json.dumps({'ok': False, 'error': repr(exc)}), flush=True)


if __name__ == '__main__':
    main()
