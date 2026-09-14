"""Snapshot the working source and add opt-in prefill attention instrumentation.

No production source/build is modified. The probe uses ordinary dense attention;
its prefill top-k is a diagnostic selector, not the production compact index.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

PATCH = r'''
            // Read-only motivation_v3 probe: retain the production batched
            // prefill and inspect its already-computed, RoPE-applied Q/K.
            const uint32_t probe_tail = std::min<uint32_t>(total,
                env_uint32_or("QW3_ATTN_TRACE_PREFILL_TOKENS", 0));
            if (global_attn_trace_enabled() && compute_logits &&
                !kvmem_enabled_ && use_paged_prefill && !mtp_single_chunk &&
                probe_tail > 0 && chunk_off + batch > total - probe_tail) {
                const uint32_t saved_position = position_;
                const uint64_t saved_sample = global_attn_trace_sample_;
                for (uint32_t row = 0; row < batch; ++row) {
                    if (chunk_off + row < total - probe_tail) continue;
                    require_status(backend_.copy_d2d(*q_, *q_batch_,
                        static_cast<uint64_t>(row) * q_stride_buf,
                        standard_n_heads * 2 * standard_head_dim));
                    position_ = base_pos + row;
                    global_attn_trace_sample_ = position_;
                    global_trace_attention_layer(il, attention_k_cache(il),
                        *q_, 2 * standard_head_dim, attn_pages_dev,
                        attn_pages_count, attn_base_pos + row + 1, scale);
                }
                position_ = saved_position;
                global_attn_trace_sample_ = saved_sample;
            }
'''

TOKEN_PATCH = r'''
        // motivation_v3: archive the exact execution token sequence and RoPE
        // positions after visual preparation, before any target generation.
        if (const char *trace_path = std::getenv("QW3_ATTN_TRACE_INPUT")) {
            std::ofstream trace(trace_path, std::ios::app);
            if (!trace) throw std::runtime_error("cannot open attention input trace");
            trace << "{\"append_base\":" << append_base << ",\"tokens\":[";
            for (size_t i = 0; i < prompt_tokens.size(); ++i) {
                if (i) trace << ',';
                trace << prompt_tokens[i];
            }
            trace << "],\"mrope\":[";
            for (size_t i = 0; i < effective_options.input_mrope_positions.size(); ++i) {
                if (i) trace << ',';
                const auto &p = effective_options.input_mrope_positions[i];
                trace << '[' << p[0] << ',' << p[1] << ',' << p[2] << ']';
            }
            trace << "]}\n";
        }
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--configure-only', action='store_true')
    ap.add_argument('--cuda-architectures',default='120a')
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    src = args.out.resolve()/'source'
    if src.exists():
        raise FileExistsError('Choose a fresh snapshot directory')
    src.mkdir(parents=True)
    for d in ('src', 'include'):
        shutil.copytree(root/d, src/d)
    (src/'third_party').mkdir()
    for f in ('httplib.h', 'json.hpp'):
        shutil.copy2(root/'third_party'/f, src/'third_party'/f)
    shutil.copytree(root/'third_party/yyjson', src/'third_party/yyjson')
    shutil.copy2(root/'CMakeLists.txt', src/'CMakeLists.txt')
    original_hashes = {str(p.relative_to(src)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in src.rglob('*') if p.is_file()}
    p = src/'src/qwen_executor.cpp'
    text = p.read_text()
    anchor = '            const float scale = 1.0f / std::sqrt(static_cast<float>(standard_head_dim));\n            if (use_paged_prefill) {'
    if text.count(anchor) != 1:
        raise ValueError('prefill instrumentation anchor is not unique')
    p.write_text(text.replace(anchor, anchor.replace('            if (use_paged_prefill) {', PATCH+'\n            if (use_paged_prefill) {')))
    (args.out/'probe_patch.txt').write_text(PATCH)
    native=src/'src/qwen_native_backend.cpp'
    text=native.read_text()
    anchor='        if (options_.dump_tokens) {'
    if text.count(anchor)!=1:
        raise ValueError('input trace anchor is not unique')
    native.write_text(text.replace(anchor,TOKEN_PATCH+'\n'+anchor))
    (args.out/'input_trace_patch.txt').write_text(TOKEN_PATCH)
    (args.out/'source_manifest.json').write_text(json.dumps(dict(
        original_source_hashes=original_hashes,
        patched_executor_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        patched_native_backend_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        semantics='dense prefill-tail exact-attention diagnostic, no production selector',
        production_sources_modified=False), indent=2)+'\n')
    build = args.out.resolve()/'build'
    flashinfer=next((root/'.venv/lib').glob('python*/site-packages/flashinfer/data'),None)
    if flashinfer is None:
        raise RuntimeError('FlashInfer headers required for paged dense decode and GPU vision')
    subprocess.run(['cmake','-S',str(src),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
        '-DQW3_ENABLE_CUDA=ON','-DQW3_ENABLE_FLASHINFER=ON','-DQW3_BUILD_TESTS=OFF',
        '-DQW3_FLASHINFER_INCLUDE_DIR='+str(flashinfer/'include'),
        '-DQW3_FLASHINFER_CUTLASS_INCLUDE_DIR='+str(flashinfer/'cutlass/include'),
        '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13/bin/nvcc',
        '-DCMAKE_CUDA_ARCHITECTURES='+args.cuda_architectures], check=True)
    if not args.configure_only:
        subprocess.run(['cmake','--build',str(build),'--target','qw3','-j','4'], check=True)


if __name__ == '__main__':
    main()
