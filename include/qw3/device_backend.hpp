#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace qw3 {

enum class LinearBackend {
    Auto,
    Custom,
    Cublas,
};

LinearBackend parse_linear_backend(const std::string &name);
const char *linear_backend_name(LinearBackend backend);

struct DeviceStatus {
    bool ok = true;
    const char *message = "";
};

struct DeviceTensor {
    virtual ~DeviceTensor() = default;
    uint64_t count = 0;
    // Element size in bytes (4 for FP32, 2 for FP16). Default keeps
    // backwards compatibility with the historical FP32-only behaviour.
    uint32_t elem_size = 4;
};

struct DeviceWeight {
    virtual ~DeviceWeight() = default;
    uint64_t rows = 0;
    uint64_t cols = 0;
};

struct DeviceArgmax {
    int token = -1;
    float logit = 0.0f;
};

struct DeviceArgmaxBuffer {
    virtual ~DeviceArgmaxBuffer() = default;
    uint64_t count = 0;
};

class DeviceBackend {
public:
    virtual ~DeviceBackend() = default;

    virtual const char *name() const = 0;
    virtual DeviceStatus begin() = 0;
    virtual DeviceStatus end() = 0;
    virtual DeviceStatus synchronize() = 0;

    // Free bytes of device memory available for allocation. Returns 0 when
    // the backend cannot report this (CPU / mock). Used by the executor to
    // size prefill chunks so per-prompt scratch fits within available memory.
    virtual uint64_t free_device_bytes() const { return 0; }

    // CUDA-graph capture hooks. Default impls are no-ops so non-CUDA backends
    // pass through. The intended pattern, used for the per-token decode loop:
    //
    //   if (begin_capture()) {
    //       // record kernel launches here
    //       end_capture();         // turns the recorded launches into a graph
    //       replay_graph();        // launches the captured graph
    //   } else {
    //       // record + execute eagerly (capture unsupported / not enabled)
    //   }
    //
    // begin_capture returns false when graph capture is not supported or has
    // been disabled at runtime — caller falls back to the eager path.
    virtual bool begin_capture() { return false; }
    virtual DeviceStatus end_capture() { return {}; }
    virtual DeviceStatus replay_graph() { return {}; }

    virtual std::unique_ptr<DeviceTensor> tensor_f32(uint64_t count, const char *label) = 0;
    virtual std::unique_ptr<DeviceTensor> tensor_i32(uint64_t count, const char *label) {
        return tensor_f32(count, label);
    }
    virtual DeviceStatus copy_i32_from_host(DeviceTensor &dst,
                                           uint64_t dst_offset,
                                           const int32_t *src,
                                           uint64_t count) {
        (void)dst; (void)dst_offset; (void)src; (void)count;
        return {false, "copy_i32_from_host not implemented for this backend"};
    }
    // Optional FP16 tensor. Backends that don't override fall back to FP32 —
    // call sites should be prepared for that and treat the returned tensor
    // as FP32 (i.e. don't blindly assume FP16 storage). The KV cache uses
    // this to halve attention bandwidth on the CUDA backend.
    virtual std::unique_ptr<DeviceTensor> tensor_f16(uint64_t count, const char *label) {
        return tensor_f32(count, label);
    }
    // Optional Q8 KV-cache tensor: `count` int8 elements grouped into rows of
    // `row_elems` that each share one fp16 max-abs scale. Backends that don't
    // override fall back to FP32 (callers must check the returned tensor's
    // actual storage, e.g. via is_q8_kv on the CUDA backend, before assuming
    // quantized layout).
    virtual std::unique_ptr<DeviceTensor> tensor_q8_kv(uint64_t count, uint32_t row_elems,
                                                       const char *label) {
        (void)row_elems;
        return tensor_f32(count, label);
    }
    // Optional FP8 (e4m3) KV-cache tensor: `count` 1-byte e4m3 elements, no
    // scale (raw e4m3). Backends that don't override fall back to FP32 (callers
    // must check the returned tensor's actual storage, e.g. via is_fp8_kv on
    // the CUDA backend, before assuming fp8 layout).
    virtual std::unique_ptr<DeviceTensor> tensor_fp8_kv(uint64_t count,
                                                        const char *label) {
        return tensor_f32(count, label);
    }
    // Transient FP32 workspace whose previous contents are never observed by
    // correct code. Backends may skip initialization to avoid large scratch
    // memset costs; conservative backends can keep zero-initialized behavior.
    virtual std::unique_ptr<DeviceTensor> scratch_f32(uint64_t count, const char *label) {
        return tensor_f32(count, label);
    }
    virtual std::unique_ptr<DeviceWeight> weight_f32(const float *data, uint64_t count, const char *label) = 0;
    virtual std::unique_ptr<DeviceWeight> weight_q8_0(const void *data, uint64_t rows, uint64_t cols, const char *label) = 0;

    virtual DeviceStatus q8_0_get_row(DeviceTensor &out, const DeviceWeight &weight, uint64_t row) = 0;
    virtual DeviceStatus q8_0_get_row_from_argmax(DeviceTensor &out,
                                                  const DeviceWeight &weight,
                                                  const DeviceArgmaxBuffer &argmaxes,
                                                  uint32_t index) {
        (void)out; (void)weight; (void)argmaxes; (void)index;
        return {false, "q8_0_get_row_from_argmax not implemented for this backend"};
    }
    // Batched embedding lookup. `rows` is a host-side array of `batch` token
    // ids. out layout: [batch, cols].
    virtual DeviceStatus q8_0_get_rows_batch(DeviceTensor &out,
                                             const DeviceWeight &weight,
                                             const uint64_t *rows,
                                             uint32_t batch) {
        for (uint32_t i = 0; i < batch; ++i) {
            if (auto st = q8_0_get_row(out, weight, rows[i]); !st.ok) return st;
        }
        return {};
    }
    virtual DeviceStatus q8_0_matvec(DeviceTensor &out, const DeviceWeight &weight, const DeviceTensor &x) = 0;

    // Fused matvec + residual add: out = out + W*x. Used at decode for the
    // attn_output and ffn_down matvecs whose result is immediately summed
    // into the residual stream. Saves one add_kernel launch and the
    // intermediate buffer round-trip per layer (~128 launches/token).
    //
    // Default falls back to separate matvec + add via a temporary tensor;
    // CUDA backend overrides with a fused mmvq kernel.
    virtual DeviceStatus q8_0_matvec_add(DeviceTensor &accum,
                                         const DeviceWeight &weight,
                                         const DeviceTensor &x) {
        (void)accum; (void)weight; (void)x;
        return {false, "q8_0_matvec_add not implemented by this backend"};
    }

    // Fused FFN SwiGLU: out = silu(W_gate * x) * (W_up * x). Used at decode
    // for the FFN block, where the previous gate+up two-weight matvec was
    // followed by a separate silu_mul kernel that read both intermediates
    // back from DRAM. The fused version writes only the n_ff-wide result.
    //
    // Default falls back to running them as separate ops; CUDA overrides.
    virtual DeviceStatus q8_0_matvec_silu_mul(DeviceTensor &out,
                                              const DeviceWeight &weight_gate,
                                              const DeviceWeight &weight_up,
                                              const DeviceTensor &x) {
        (void)out; (void)weight_gate; (void)weight_up; (void)x;
        return {false, "q8_0_matvec_silu_mul not implemented by this backend"};
    }

    // Run several Q8_0 matvecs that share the same input vector x. The MMVQ
    // path quantizes x to Q8_1 once, then runs each (weight -> out) matvec
    // against the cached Q8_1 buffer. Saves K-1 redundant input quantizations
    // in attention QKV (3 projections) and FFN gate+up (2 projections).
    //
    // Default falls back to N independent matvecs; CUDA overrides.
    virtual DeviceStatus q8_0_matvec_fanout(DeviceTensor *const *outs,
                                            const DeviceWeight *const *weights,
                                            uint32_t n,
                                            const DeviceTensor &x) {
        for (uint32_t i = 0; i < n; ++i) {
            if (auto st = q8_0_matvec(*outs[i], *weights[i], x); !st.ok) return st;
        }
        return {};
    }

    // Batched matvec ("matmul") for prefill. The weight (Q8_0, [out_dim x in_dim])
    // is read once and reused across `batch` input vectors stored contiguously
    // in `x`. Output layout: [batch, out_dim], input layout: [batch, in_dim].
    // Default impl loops over batch positions calling q8_0_matvec; backends
    // should override to share the weight read. (CUDA does.)
    virtual DeviceStatus q8_0_matmul(DeviceTensor &out,
                                     const DeviceWeight &weight,
                                     const DeviceTensor &x,
                                     uint32_t batch,
                                     uint32_t in_stride,
                                     uint32_t out_stride) {
        for (uint32_t i = 0; i < batch; ++i) {
            if (auto st = q8_0_matvec(out, weight, x); !st.ok) return st;
            (void)in_stride; (void)out_stride;
            // Default impl can't slice DeviceTensors, so this only works for
            // batch==1; real backends override.
            if (batch > 1) return {false, "q8_0_matmul default impl only supports batch=1"};
        }
        return {};
    }

    // Run several batched Q8_0 matmuls that share the same input batch. Backends
    // can reuse input-side staging/quantization work across projections.
    // `out_strides[i]` is the row stride for `outs[i]`.
    virtual DeviceStatus q8_0_matmul_fanout(DeviceTensor *const *outs,
                                            const DeviceWeight *const *weights,
                                            const uint32_t *out_strides,
                                            uint32_t n,
                                            const DeviceTensor &x,
                                            uint32_t batch,
                                            uint32_t in_stride) {
        for (uint32_t i = 0; i < n; ++i) {
            if (auto st = q8_0_matmul(*outs[i], *weights[i], x,
                                      batch, in_stride, out_strides[i]); !st.ok) {
                return st;
            }
        }
        return {};
    }

    // Batched fused matmul + residual add: out = residual + W*x. `matmul_tmp`
    // is scratch used to hold W*x before the add. Default keeps the two-step
    // path (matmul into tmp, then add); backends may fuse the residual add
    // into the matmul writeback. Used by the batched MTP prefix path.
    virtual DeviceStatus q8_0_matmul_add(DeviceTensor &out,
                                         const DeviceTensor &residual,
                                         DeviceTensor &matmul_tmp,
                                         const DeviceWeight &weight,
                                         const DeviceTensor &x,
                                         uint32_t batch,
                                         uint32_t in_stride,
                                         uint32_t out_stride) {
        if (auto st = q8_0_matmul(matmul_tmp, weight, x, batch, in_stride, out_stride); !st.ok) return st;
        return add(out, residual, matmul_tmp);
    }

    virtual DeviceStatus rms_norm(DeviceTensor &out, const DeviceTensor &x, const DeviceWeight &weight, float eps) = 0;
    // Batched RMS norm. out / x layouts: [batch, n], weight: [n] (shared).
    virtual DeviceStatus rms_norm_batch(DeviceTensor &out,
                                        const DeviceTensor &x,
                                        const DeviceWeight &weight,
                                        uint32_t batch,
                                        uint32_t n,
                                        float eps) {
        // Default: just N synchronous launches if backend hasn't overridden.
        (void)n;
        for (uint32_t i = 0; i < batch; ++i) {
            if (auto st = rms_norm(out, x, weight, eps); !st.ok) return st;
        }
        return {};
    }
    virtual DeviceStatus add(DeviceTensor &out, const DeviceTensor &a, const DeviceTensor &b) = 0;
    // Prefix-limited elementwise add. Batched residual buffers are
    // capacity-sized; small verifier batches must add only `count` elements,
    // not every allocated row.
    virtual DeviceStatus add_n(DeviceTensor &out, const DeviceTensor &a,
                               const DeviceTensor &b, uint64_t count) {
        if (count == out.count) return add(out, a, b);
        return {false, "add_n requires backend override for partial tensors"};
    }
    virtual DeviceStatus silu(DeviceTensor &out, const DeviceTensor &x) = 0;
    virtual DeviceStatus mul(DeviceTensor &out, const DeviceTensor &a, const DeviceTensor &b) = 0;
    // Default fallback: silu(gate) * up done as two separate kernels.
    // Backends should override to do this in a single launch.
    virtual DeviceStatus silu_mul(DeviceTensor &out, const DeviceTensor &gate, const DeviceTensor &up) {
        if (auto st = silu(const_cast<DeviceTensor &>(gate), gate); !st.ok) return st;
        return mul(out, gate, up);
    }
    // Prefix-limited SwiGLU. Batched scratch buffers are capacity-sized, so
    // small verifier batches must not process every allocated row.
    virtual DeviceStatus silu_mul_n(DeviceTensor &out,
                                    const DeviceTensor &gate,
                                    const DeviceTensor &up,
                                    uint64_t count) {
        if (count == out.count) return silu_mul(out, gate, up);
        return {false, "silu_mul_n requires backend override for partial tensors"};
    }
    virtual DeviceStatus recurrent_single_token(DeviceTensor &core,
                                                DeviceTensor &state,
                                                DeviceTensor &conv_state,
                                                DeviceTensor &conv_out,
                                                const DeviceTensor &proj,
                                                const DeviceTensor &gate,
                                                const DeviceTensor &alpha,
                                                const DeviceTensor &beta,
                                                const DeviceWeight &conv,
                                                const DeviceWeight &ssm_a,
                                                const DeviceWeight &dt_bias,
                                                const DeviceWeight &ssm_norm,
                                                uint32_t num_k_heads,
                                                uint32_t num_v_heads,
                                                uint32_t head_k_dim,
                                                uint32_t head_v_dim,
                                                uint32_t conv_kernel_size,
                                                float eps) = 0;

    // Offset-aware variant for batched prefill. The four input tensors and
    // the output core are treated as [batch, *] buffers and the call reads
    // from index `*_off`. Recurrent state / conv_state stay shared (sequential
    // updates across the batch). proj_count = recurrent_qkv_dim;
    // core_count = recurrent_value_dim.
    virtual DeviceStatus recurrent_single_token_at(DeviceTensor &core,
                                                    DeviceTensor &state,
                                                    DeviceTensor &conv_state,
                                                    DeviceTensor &conv_out,
                                                    const DeviceTensor &proj,
                                                    const DeviceTensor &gate,
                                                    const DeviceTensor &alpha,
                                                    const DeviceTensor &beta,
                                                    const DeviceWeight &conv,
                                                    const DeviceWeight &ssm_a,
                                                    const DeviceWeight &dt_bias,
                                                    const DeviceWeight &ssm_norm,
                                                    uint32_t num_k_heads,
                                                    uint32_t num_v_heads,
                                                    uint32_t head_k_dim,
                                                    uint32_t head_v_dim,
                                                    uint32_t conv_kernel_size,
                                                    uint32_t proj_count,
                                                    uint32_t proj_off,
                                                    uint32_t gate_off,
                                                    uint32_t alpha_off,
                                                    uint32_t beta_off,
                                                    uint32_t core_off,
                                                    float eps) {
        if (proj_off || gate_off || alpha_off || beta_off || core_off) {
            return {false, "recurrent_single_token_at with non-zero offsets requires backend override"};
        }
        (void)proj_count;
        return recurrent_single_token(core, state, conv_state, conv_out, proj, gate, alpha, beta,
                                       conv, ssm_a, dt_bias, ssm_norm,
                                       num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                                       conv_kernel_size, eps);
    }

    // Time-batched recurrent layer: processes `batch` tokens in O(1) kernel
    // launches instead of O(batch). Inputs are row-major:
    //   proj  : [batch, proj_count]      stride proj_stride
    //   gate  : [batch, gate_count]      stride gate_stride
    //   alpha : [batch, num_v_heads]     stride alpha_stride
    //   beta  : [batch, num_v_heads]     stride beta_stride
    //   core  : [batch, num_v_heads*head_v_dim] stride core_stride
    // `conv_out_buf` is a scratch buffer sized to [batch, proj_count].
    // The default implementation loops over the per-token variant for
    // backends that don't override it.
    virtual DeviceStatus recurrent_batch(DeviceTensor &core,
                                          DeviceTensor &state,
                                          DeviceTensor &conv_state,
                                          DeviceTensor &conv_out_buf,
                                          const DeviceTensor &proj,
                                          const DeviceTensor &gate,
                                          const DeviceTensor &alpha,
                                          const DeviceTensor &beta,
                                          const DeviceWeight &conv,
                                          const DeviceWeight &ssm_a,
                                          const DeviceWeight &dt_bias,
                                          const DeviceWeight &ssm_norm,
                                          uint32_t batch,
                                          uint32_t num_k_heads,
                                          uint32_t num_v_heads,
                                          uint32_t head_k_dim,
                                          uint32_t head_v_dim,
                                          uint32_t conv_kernel_size,
                                          uint32_t proj_count,
                                          uint32_t proj_stride,
                                          uint32_t gate_stride,
                                          uint32_t alpha_stride,
                                          uint32_t beta_stride,
                                          uint32_t core_stride,
                                          float eps,
                                          DeviceTensor *state_checkpoints = nullptr,
                                          DeviceTensor *conv_state_checkpoints = nullptr,
                                          uint32_t checkpoint_count = 0) {
        (void)state_checkpoints;
        (void)conv_state_checkpoints;
        (void)checkpoint_count;
        for (uint32_t b = 0; b < batch; ++b) {
            if (auto st = recurrent_single_token_at(core, state, conv_state, conv_out_buf,
                                                    proj, gate, alpha, beta,
                                                    conv, ssm_a, dt_bias, ssm_norm,
                                                    num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                                                    conv_kernel_size,
                                                    proj_count,
                                                    b * proj_stride,
                                                    b * gate_stride,
                                                    b * alpha_stride,
                                                    b * beta_stride,
                                                    b * core_stride,
                                                    eps); !st.ok) {
                return st;
            }
        }
        return {};
    }
    virtual DeviceStatus attention_single_token(DeviceTensor &mid,
                                                DeviceTensor &q,
                                                DeviceTensor &k,
                                                const DeviceTensor &v,
                                                const DeviceWeight &q_norm,
                                                const DeviceWeight &k_norm,
                                                uint32_t n_heads,
                                                uint32_t n_kv_heads,
                                                uint32_t head_dim,
                                                float eps) = 0;

    // Per-head RMS norm: normalize the first `head_dim` floats of each unit,
    // scaled by the (shared) head_dim-length weight. Each unit has stride
    // `per_unit_stride`; remaining floats inside each unit are left alone.
    virtual DeviceStatus rmsnorm_per_head(DeviceTensor &x,
                                          const DeviceWeight &weight,
                                          uint32_t n_units,
                                          uint32_t per_unit_stride,
                                          uint32_t head_dim,
                                          float eps) = 0;

    // Batched per-head RMS norm. x layout: [batch, batch_stride] where each
    // row contains `n_units` units of `per_unit_stride` floats.
    virtual DeviceStatus rmsnorm_per_head_batch(DeviceTensor &x,
                                                const DeviceWeight &weight,
                                                uint32_t batch,
                                                uint32_t batch_stride,
                                                uint32_t n_units,
                                                uint32_t per_unit_stride,
                                                uint32_t head_dim,
                                                float eps) {
        for (uint32_t i = 0; i < batch; ++i) {
            if (auto st = rmsnorm_per_head(x, weight, n_units, per_unit_stride, head_dim, eps); !st.ok) return st;
        }
        (void)batch_stride;
        return {};
    }

    // Partial / split-half RoPE on the first `rope_dim` floats of each unit.
    // For i in [0, rope_dim/2): pair (x[i], x[i + rope_dim/2]) is rotated by
    // angle pos * theta^(-2i/rope_dim). Remaining floats unchanged.
    virtual DeviceStatus rope_partial(DeviceTensor &x,
                                      uint32_t n_units,
                                      uint32_t per_unit_stride,
                                      uint32_t rope_dim,
                                      uint32_t pos,
                                      float theta) = 0;

    // Batched partial RoPE. Token i in the batch (0-indexed) is rotated at
    // position `base_pos + i`. x layout: [batch, batch_stride].
    virtual DeviceStatus rope_partial_batch(DeviceTensor &x,
                                            uint32_t batch,
                                            uint32_t batch_stride,
                                            uint32_t n_units,
                                            uint32_t per_unit_stride,
                                            uint32_t rope_dim,
                                            uint32_t base_pos,
                                            float theta) {
        for (uint32_t i = 0; i < batch; ++i) {
            if (auto st = rope_partial(x, n_units, per_unit_stride, rope_dim, base_pos + i, theta); !st.ok) return st;
        }
        (void)batch_stride;
        return {};
    }

    // Append `per_pos_size` floats from src into cache at slot `pos`.
    virtual DeviceStatus kv_append(DeviceTensor &cache,
                                   const DeviceTensor &src,
                                   uint32_t pos,
                                   uint32_t per_pos_size) = 0;

    // Append `batch` consecutive positions starting at `base_pos`. src
    // layout: [batch, per_pos_size].
    virtual DeviceStatus kv_append_batch(DeviceTensor &cache,
                                         const DeviceTensor &src,
                                         uint32_t base_pos,
                                         uint32_t per_pos_size,
                                         uint32_t batch) {
        for (uint32_t i = 0; i < batch; ++i) {
            if (auto st = kv_append(cache, src, base_pos + i, per_pos_size); !st.ok) return st;
        }
        return {};
    }

    // Append into a KV cache addressed as physical pages. `page_indices` maps
    // logical page id -> physical page id, and `page_size` is measured in
    // tokens. The default implementation preserves compatibility for identity
    // page tables; CUDA overrides this with true physical-page writes.
    virtual DeviceStatus kv_append_paged(DeviceTensor &cache,
                                         const DeviceTensor &src,
                                         uint32_t logical_pos,
                                         uint32_t per_pos_size,
                                         const int32_t *page_indices,
                                         uint32_t n_pages,
                                         uint32_t page_size) {
        if (page_size == 0) return {false, "kv_append_paged requires page_size > 0"};
        const uint32_t logical_page = logical_pos / page_size;
        const uint32_t page_offset = logical_pos % page_size;
        if (!page_indices || logical_page >= n_pages) {
            return {false, "kv_append_paged page table is too small"};
        }
        if (page_indices[logical_page] < 0) {
            return {false, "kv_append_paged encountered negative physical page"};
        }
        const uint32_t physical_pos =
            static_cast<uint32_t>(page_indices[logical_page]) * page_size + page_offset;
        return kv_append(cache, src, physical_pos, per_pos_size);
    }

    virtual DeviceStatus kv_append_batch_paged(DeviceTensor &cache,
                                               const DeviceTensor &src,
                                               uint32_t base_logical_pos,
                                               uint32_t per_pos_size,
                                               uint32_t batch,
                                               const int32_t *page_indices,
                                               uint32_t n_pages,
                                               uint32_t page_size) {
        for (uint32_t i = 0; i < batch; ++i) {
            if (auto st = kv_append_paged(cache, src, base_logical_pos + i,
                                          per_pos_size, page_indices, n_pages,
                                          page_size); !st.ok) {
                return st;
            }
        }
        return {};
    }

    virtual DeviceStatus kv_append_batch_paged_device(DeviceTensor &cache,
                                                      const DeviceTensor &src,
                                                      uint32_t base_logical_pos,
                                                      uint32_t per_pos_size,
                                                      uint32_t batch,
                                                      const DeviceTensor &page_indices,
                                                      uint32_t n_pages,
                                                      uint32_t page_size) {
        (void)cache; (void)src; (void)base_logical_pos; (void)per_pos_size;
        (void)batch; (void)page_indices; (void)n_pages; (void)page_size;
        return {false, "kv_append_batch_paged_device requires backend override"};
    }

    // Decode-step attention: causal scaled dot-product over the live KV cache.
    //   out[head, d]     = sum_t softmax(q_head . K[t, kv_head] * scale) * V[t, kv_head, d]
    // where kv_head = head / (n_heads / n_kv_heads).
    // q layout: [n_heads, q_stride], q_stride >= head_dim; only the first
    //           head_dim floats of each head are read (the rest is the gate).
    // k_cache / v_cache layout: [seq_len, n_kv_heads * head_dim].
    // scores_scratch must have at least n_heads*seq_len floats.
    virtual DeviceStatus attention_decode(DeviceTensor &out,
                                          DeviceTensor &scores_scratch,
                                          const DeviceTensor &q,
                                          uint32_t q_stride,
                                          const DeviceTensor &k_cache,
                                          const DeviceTensor &v_cache,
                                          uint32_t n_heads,
                                          uint32_t n_kv_heads,
                                          uint32_t head_dim,
                                          uint32_t seq_len,
                                          float scale) = 0;

    // out[head, d] *= sigmoid(q[head, head_dim + d]) — applies the gated-
    // attention modulation derived from the second-half of each Q head.
    virtual DeviceStatus apply_attn_gate(DeviceTensor &out,
                                         const DeviceTensor &q,
                                         uint32_t q_stride,
                                         uint32_t n_heads,
                                         uint32_t head_dim) = 0;

    // Batched prefill attention. For batch index b (0..batch-1) we compute
    // attention with seq_len = base_seq_len + b + 1 (causal) using the KV
    // cache that was just appended to with kv_append_batch. q layout:
    // [batch, q_batch_stride], out layout: [batch, out_batch_stride].
    virtual DeviceStatus attention_decode_batch(DeviceTensor &out,
                                                const DeviceTensor &q,
                                                uint32_t q_stride,
                                                const DeviceTensor &k_cache,
                                                const DeviceTensor &v_cache,
                                                uint32_t n_heads,
                                                uint32_t n_kv_heads,
                                                uint32_t head_dim,
                                                uint32_t base_seq_len,
                                                uint32_t batch,
                                                uint32_t q_batch_stride,
                                                uint32_t out_batch_stride,
                                                float scale) {
        (void)out; (void)q; (void)q_stride; (void)k_cache; (void)v_cache;
        (void)n_heads; (void)n_kv_heads; (void)head_dim; (void)base_seq_len;
        (void)batch; (void)q_batch_stride; (void)out_batch_stride; (void)scale;
        return {false, "attention_decode_batch requires backend override"};
    }

    // Batched gate apply across `batch` rows. out[b, head, d] *= sigmoid(q[b, head, head_dim + d]).
    virtual DeviceStatus apply_attn_gate_batch(DeviceTensor &out,
                                                const DeviceTensor &q,
                                                uint32_t q_stride,
                                                uint32_t batch,
                                                uint32_t batch_stride_q,
                                                uint32_t batch_stride_out,
                                                uint32_t n_heads,
                                                uint32_t head_dim) {
        for (uint32_t i = 0; i < batch; ++i) {
            if (auto st = apply_attn_gate(out, q, q_stride, n_heads, head_dim); !st.ok) return st;
            (void)batch_stride_q; (void)batch_stride_out;
        }
        return {};
    }

    // Combined prefill/verify attention + Q-gate application. Backends can
    // override to fuse the final gate multiply into attention writeback; the
    // default preserves the existing two-step behaviour.
    virtual DeviceStatus attention_decode_batch_gated(DeviceTensor &out,
                                                      const DeviceTensor &q,
                                                      uint32_t q_stride,
                                                      const DeviceTensor &k_cache,
                                                      const DeviceTensor &v_cache,
                                                      uint32_t n_heads,
                                                      uint32_t n_kv_heads,
                                                      uint32_t head_dim,
                                                      uint32_t base_seq_len,
                                                      uint32_t batch,
                                                      uint32_t q_batch_stride,
                                                      uint32_t out_batch_stride,
                                                      float scale) {
        if (auto st = attention_decode_batch(out, q, q_stride, k_cache, v_cache,
                                             n_heads, n_kv_heads, head_dim,
                                             base_seq_len, batch,
                                             q_batch_stride, out_batch_stride,
                                             scale); !st.ok) {
            return st;
        }
        return apply_attn_gate_batch(out, q, q_stride, batch, q_batch_stride,
                                     out_batch_stride, n_heads, head_dim);
    }

    // Paged-KV variant for one logical sequence. Row b attends to
    // base_seq_len + b + 1 logical tokens, described by the same page table.
    // Backends without native paged attention fall back to the contiguous path
    // when the page table is identity.
    virtual DeviceStatus attention_decode_batch_paged_gated(
                                                      DeviceTensor &out,
                                                      const DeviceTensor &q,
                                                      uint32_t q_stride,
                                                      const DeviceTensor &k_cache,
                                                      const DeviceTensor &v_cache,
                                                      const int32_t *page_indices,
                                                      uint32_t n_pages,
                                                      uint32_t page_size,
                                                      uint32_t n_heads,
                                                      uint32_t n_kv_heads,
                                                      uint32_t head_dim,
                                                      uint32_t base_seq_len,
                                                      uint32_t batch,
                                                      uint32_t q_batch_stride,
                                                      uint32_t out_batch_stride,
                                                      float scale) {
        if (!page_indices || page_size == 0) {
            return {false, "attention_decode_batch_paged_gated requires a page table"};
        }
        const uint32_t need_pages =
            (base_seq_len + batch + page_size - 1) / page_size;
        if (need_pages > n_pages) {
            return {false, "attention_decode_batch_paged_gated page table is too small"};
        }
        for (uint32_t p = 0; p < need_pages; ++p) {
            if (page_indices[p] != static_cast<int32_t>(p)) {
                return {false, "paged attention fallback only supports identity page tables"};
            }
        }
        return attention_decode_batch_gated(out, q, q_stride, k_cache, v_cache,
                                            n_heads, n_kv_heads, head_dim,
                                            base_seq_len, batch,
                                            q_batch_stride, out_batch_stride,
                                            scale);
    }

    virtual DeviceStatus attention_decode_batch_paged_gated_device(
                                                      DeviceTensor &out,
                                                      const DeviceTensor &q,
                                                      uint32_t q_stride,
                                                      const DeviceTensor &k_cache,
                                                      const DeviceTensor &v_cache,
                                                      const DeviceTensor &page_indices,
                                                      uint32_t n_pages,
                                                      uint32_t page_size,
                                                      uint32_t n_heads,
                                                      uint32_t n_kv_heads,
                                                      uint32_t head_dim,
                                                      uint32_t base_seq_len,
                                                      uint32_t batch,
                                                      uint32_t q_batch_stride,
                                                      uint32_t out_batch_stride,
                                                      float scale) {
        (void)out; (void)q; (void)q_stride; (void)k_cache; (void)v_cache;
        (void)page_indices; (void)n_pages; (void)page_size; (void)n_heads;
        (void)n_kv_heads; (void)head_dim; (void)base_seq_len; (void)batch;
        (void)q_batch_stride; (void)out_batch_stride; (void)scale;
        return {false, "attention_decode_batch_paged_gated_device requires backend override"};
    }

    // Zero all floats in tensor. Used to reset KV / recurrent state between
    // generate() calls.
    virtual DeviceStatus zero_tensor(DeviceTensor &x) = 0;

    virtual DeviceArgmax argmax(const DeviceTensor &x) = 0;

    virtual std::unique_ptr<DeviceArgmaxBuffer> argmax_buffer(uint64_t count) {
        (void)count;
        return nullptr;
    }
    virtual DeviceStatus argmax_to_buffer(const DeviceTensor &x,
                                          DeviceArgmaxBuffer &out,
                                          uint32_t index) {
        (void)x; (void)out; (void)index;
        return {false, "argmax_to_buffer not implemented for this backend"};
    }
    virtual DeviceStatus copy_argmax_buffer_to_host(const DeviceArgmaxBuffer &src,
                                                    DeviceArgmax *host,
                                                    uint32_t count) {
        (void)src; (void)host; (void)count;
        return {false, "copy_argmax_buffer_to_host not implemented for this backend"};
    }

    // Batched argmax across `batch` rows of `row_stride` floats each. Default
    // copies to host and scans; CUDA overrides with a device reduction.
    virtual DeviceStatus argmax_batch(const DeviceTensor &x,
                                      uint32_t batch,
                                      uint32_t row_stride,
                                      std::vector<DeviceArgmax> &out) {
        out.assign(batch, DeviceArgmax{});
        std::vector<float> host(static_cast<size_t>(batch) * row_stride);
        if (auto st = copy_to_host(x, host.data(), 0, host.size()); !st.ok) return st;
        for (uint32_t row = 0; row < batch; ++row) {
            DeviceArgmax best;
            best.logit = -std::numeric_limits<float>::infinity();
            const float *base = host.data() + static_cast<size_t>(row) * row_stride;
            for (uint32_t col = 0; col < row_stride; ++col) {
                if (base[col] > best.logit) {
                    best.logit = base[col];
                    best.token = static_cast<int>(col);
                }
            }
            out[row] = best;
        }
        return {};
    }

    // Two-phase argmax for graph capture: launch the device kernel inside
    // the captured stream, then later sync + read the device buffer outside
    // the graph. Default impls fall back to the synchronous argmax above —
    // they're only correct when graph capture is *not* in flight.
    virtual DeviceStatus argmax_launch(const DeviceTensor &x) {
        (void)x;
        return {};
    }
    virtual DeviceArgmax argmax_collect() {
        return {};
    }

    // Copy `count` floats from the device tensor starting at offset to host
    // memory. Used by --dump-logits and other diagnostics. Default no-op
    // implementation keeps the mock backend simple; CUDA overrides it.
    virtual DeviceStatus copy_to_host(const DeviceTensor &x, float *host, uint64_t offset, uint64_t count) {
        (void)x; (void)host; (void)offset; (void)count;
        return {false, "copy_to_host not implemented for this backend"};
    }

    // Device-to-device copy: dst[0..count) = src[src_offset..src_offset+count).
    virtual DeviceStatus copy_d2d(DeviceTensor &dst,
                                  const DeviceTensor &src,
                                  uint64_t src_offset,
                                  uint64_t count) {
        (void)dst; (void)src; (void)src_offset; (void)count;
        return {false, "copy_d2d not implemented for this backend"};
    }

    // Offset-aware device-to-device copy. Counts are in tensor elements.
    virtual DeviceStatus copy_d2d_into(DeviceTensor &dst,
                                       uint64_t dst_offset,
                                       const DeviceTensor &src,
                                       uint64_t src_offset,
                                       uint64_t count) {
        if (dst_offset != 0) {
            return {false, "copy_d2d_into with non-zero dst offset requires backend override"};
        }
        return copy_d2d(dst, src, src_offset, count);
    }

    // Pack MTP prefix hidden inputs:
    //   dst[0]      = first_h
    //   dst[1..T-1] = h_batch[0..T-2]
    // Layout is row-major, with every row using `h_stride` elements.
    virtual DeviceStatus pack_mtp_prefix_hinputs(DeviceTensor &dst,
                                                 const DeviceTensor &first_h,
                                                 const DeviceTensor &h_batch,
                                                 uint32_t batch,
                                                 uint32_t h_stride) {
        if (batch == 0 || h_stride == 0) return {};
        if (auto st = copy_d2d_into(dst, 0, first_h, 0, h_stride); !st.ok) return st;
        if (batch == 1) return {};
        return copy_d2d_into(dst,
                             h_stride,
                             h_batch,
                             0,
                             static_cast<uint64_t>(batch - 1) * h_stride);
    }

    // Pack MTP concat rows:
    //   dst[row] = concat(left[row][0:width], right[row][0:width])
    // Strides and width are in tensor elements.
    virtual DeviceStatus pack_mtp_concat(DeviceTensor &dst,
                                         const DeviceTensor &left,
                                         const DeviceTensor &right,
                                         uint32_t batch,
                                         uint32_t left_stride,
                                         uint32_t right_stride,
                                         uint32_t concat_stride,
                                         uint32_t width) {
        if (batch == 0 || width == 0) return {};
        for (uint32_t i = 0; i < batch; ++i) {
            const uint64_t dst_row = static_cast<uint64_t>(i) * concat_stride;
            if (auto st = copy_d2d_into(dst,
                                        dst_row,
                                        left,
                                        static_cast<uint64_t>(i) * left_stride,
                                        width); !st.ok) {
                return st;
            }
            if (auto st = copy_d2d_into(dst,
                                        dst_row + width,
                                        right,
                                        static_cast<uint64_t>(i) * right_stride,
                                        width); !st.ok) {
                return st;
            }
        }
        return {};
    }
};

bool cuda_device_backend_available();
std::unique_ptr<DeviceBackend> make_cuda_device_backend(LinearBackend linear_backend);

} // namespace qw3
