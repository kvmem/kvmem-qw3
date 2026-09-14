"""Check real GGUF rows against CPU quantized-dot oracles, within spare VRAM."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from gguf import GGUFReader, GGMLQuantizationType

from alignment.native_q8 import NativeQ8, pack_native_rows

ROOT = Path(__file__).resolve().parent


def reference(raw, x, rows):
    blocks = raw.reshape(raw.shape[0], -1, 34)[rows]
    wd = blocks[..., :2].copy().view(np.float16).reshape(len(rows), -1).astype(np.float32)
    wq = blocks[..., 2:].copy().view(np.int8).astype(np.int32)
    xb = x.reshape(len(x), -1, 32)
    amax = np.max(np.abs(xb), axis=-1)
    if len(x) <= 8:
        d = amax / np.float32(127)
        aq = np.rint(np.divide(xb, d[..., None], out=np.zeros_like(xb),
                               where=d[..., None] != 0)).astype(np.int32)
        ad = d.astype(np.float16).astype(np.float32)
    else:
        inv = np.divide(np.float32(127), amax, out=np.zeros_like(amax), where=amax != 0)
        aq = np.rint(xb * inv[..., None]).astype(np.int32)
        ad = np.divide(np.float32(1), inv, out=np.zeros_like(inv), where=inv != 0)
    dots = np.einsum('bkc,rkc->brk', aq, wq).astype(np.float32)
    return (dots * ad[:, None, :] * wd[None, :, :]).sum(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    report = dict(started=time.time(), status='running', component_only=True,
                  full_model_test=False, performance_eligible=False, shared_gpu=True,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), ROOT / 'alignment/native_q8.py',
                                           ROOT / 'compat_build/native_q8/libaligned_q8.so']}, cases=[])
    try:
        # Check total room BEFORE CUDA context initialization. The 2 GiB covers
        # the driver context plus our <=.25 GiB tensor allocator cap.
        before = float(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True).strip()) / 1024
        report['free_before_cuda_gib'] = before
        if before < 2:
            raise RuntimeError('need 2 GiB free before initializing the bounded probe')
        free, total = torch.cuda.mem_get_info()
        report['free_after_cuda_gib'] = free / 2**30
        if free < .75 * 2**30:
            raise RuntimeError('need .75 GiB after CUDA context initialization')
        torch.cuda.set_per_process_memory_fraction(.25 * 2**30 / total)
        reader = GGUFReader(args.model)
        tensor = next(t for t in reader.tensors if t.name == 'blk.0.ffn_gate.weight')
        assert tensor.tensor_type == GGMLQuantizationType.Q8_0
        cols = int(tensor.shape[0])
        raw = np.array(tensor.data[:128], copy=True)
        report['weight'] = dict(path=str(args.model.resolve()), tensor=tensor.name,
                                first_rows=128, cols=cols,
                                raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest())
        packed = pack_native_rows(raw, cols)
        # Byte-exact round trip: packing must never requantize weights.
        n = cols // 32
        unpacked = np.concatenate((packed[:, :2*n].reshape(128, n, 2),
                                   packed[:, 2*n:].reshape(128, n, 32)), axis=-1)
        np.testing.assert_array_equal(unpacked.reshape(raw.shape), raw)
        api = NativeQ8()
        weight = torch.from_numpy(packed).cuda()
        rng = np.random.default_rng(1271)
        chosen = [0, 1, 63, 127]
        with torch.inference_mode():
            for batch in [1, 8, 11, 17, 64, 65, 512]:
                x = rng.normal(size=(batch, cols)).astype(np.float32)
                x[:, :32] = 0  # Exercise all-zero activation scale.
                observed = api.matmul(weight, torch.from_numpy(x).cuda(), cols).cpu().numpy()
                if batch == 11:  # MMVQ split into batches of 8 and 3.
                    expected = np.concatenate([reference(raw, x[:8], chosen),
                                               reference(raw, x[8:], chosen)])
                else:
                    expected = reference(raw, x, chosen)
                err = np.abs(observed[:, chosen] - expected)
                np.testing.assert_allclose(observed[:, chosen], expected, atol=2e-4, rtol=2e-4)
                report['cases'].append(dict(batch=batch, rows=128,
                                           checked_rows=chosen, max_abs=float(err.max())))
            # Tiny 48-row projections also occur in this Qwen model.
            x = rng.normal(size=(17, cols)).astype(np.float32)
            observed = api.matmul(weight[:48], torch.from_numpy(x).cuda(), cols).cpu().numpy()
            expected = np.concatenate([reference(raw[:48], part, [0, 47])
                                       for part in (x[:8], x[8:16], x[16:])])
            np.testing.assert_allclose(observed[:, [0, 47]], expected, atol=2e-4, rtol=2e-4)
            report['cases'].append(dict(batch=17, rows=48, checked_rows=[0, 47],
                                       max_abs=float(np.abs(observed[:, [0,47]]-expected).max())))
            ids = torch.tensor([127, 0, 63, 127], dtype=torch.int64, device='cuda')
            observed = api.embedding(weight, ids, cols).cpu().numpy()
            blocks = raw.reshape(128, -1, 34)[[127, 0, 63, 127]]
            scales = blocks[..., :2].copy().view(np.float16).astype(np.float32)
            expected = (scales * blocks[..., 2:].copy().view(np.int8)).reshape(4, cols)
            np.testing.assert_array_equal(observed, expected)
            report['embedding_bitwise_equal'] = True
            # Explicit non-default stream: neither bridge assumes stream zero.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                observed = api.matmul(weight, torch.ones((1, cols), device='cuda'), cols)
            torch.cuda.current_stream().wait_stream(stream)
            np.testing.assert_allclose(observed.cpu().numpy()[:, chosen],
                reference(raw, np.ones((1, cols), dtype=np.float32), chosen), atol=2e-4, rtol=2e-4)
            report['nondefault_stream_passed'] = True
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report.update(finished=time.time(), peak_reserved_gib=(
            torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_initialized() else 0))
        args.out.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
