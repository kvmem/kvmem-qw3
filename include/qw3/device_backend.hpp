#pragma once

#include <cstdint>
#include <mutex>
#include <new>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

enum class DeviceTensorDType {
    F32,
    F16,
    BF16,
    I32,
    Q8_KV,
    FP8_E4M3,
};

struct DeviceTensor {
    virtual ~DeviceTensor() = default;
    uint64_t count = 0;
    // Element size in bytes (4 for FP32, 2 for FP16). Default keeps
    // backwards compatibility with the historical FP32-only behaviour.
    uint32_t elem_size = 4;
    DeviceTensorDType dtype = DeviceTensorDType::F32;
};

enum class DeviceWeightFormat {
    F32,
    BF16,
    FP8_E4M3,
    Q8_0,
    NVFP4_E2M1,
};

struct DeviceWeight {
    virtual ~DeviceWeight() = default;
    uint64_t rows = 0;
    uint64_t cols = 0;
    DeviceWeightFormat format = DeviceWeightFormat::F32;
};

struct DeviceArgmax {
    int token = -1;
    float logit = 0.0f;
};

struct DeviceArgmaxBuffer {
    virtual ~DeviceArgmaxBuffer() = default;
    uint64_t count = 0;
};

struct HostBuffer {
    virtual ~HostBuffer() = default;
    void *data = nullptr;
    uint64_t bytes = 0;
    bool pinned = false;
};

// Opaque completion marker for the dedicated KV transfer stream. CUDA records
// a cudaEvent behind all copies queued so far; synchronous/mock backends use
// the default implementation below, which drains the transfer immediately and
// returns a null marker. Keeping the marker backend-owned lets the executor
// pipeline two pinned slabs without leaking CUDA types into the public header.
struct DeviceTransferFence {
    virtual ~DeviceTransferFence() = default;
};

class DeviceBackend {
public:
    virtual ~DeviceBackend() = default;

    virtual const char *name() const = 0;
    virtual DeviceStatus begin() = 0;
    virtual DeviceStatus end() = 0;
    virtual DeviceStatus synchronize() = 0;

    // Free / total bytes of device memory. Return 0 when the backend cannot
    // report this (CPU / mock). Free bytes are used to size prefill chunks;
    // total bytes are used for KVMem ratio-based KV residency budgeting.
    virtual uint64_t free_device_bytes() const { return 0; }
    virtual uint64_t total_device_bytes() const { return 0; }
    // PCI bus/device/function for topology-aware host-side I/O placement.
    // CUDA returns the canonical domain:bus:device.function string; CPU/mock
    // backends leave it empty so callers can fall back without special cases.
    virtual std::string pci_bus_id() const { return {}; }

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
    virtual bool supports_bf16_activations() const { return false; }
    virtual std::unique_ptr<DeviceTensor> tensor_bf16(uint64_t count,
                                                       const char *label) {
        return tensor_f32(count, label);
    }
    virtual std::unique_ptr<HostBuffer> host_buffer(uint64_t bytes,
                                                    const char *label) {
        (void)label;
        struct HeapHostBuffer final : HostBuffer {
            explicit HeapHostBuffer(uint64_t n) {
                bytes = n;
                pinned = false;
                if (n > 0) data = ::operator new(static_cast<size_t>(n));
            }
            ~HeapHostBuffer() override {
                ::operator delete(data);
            }
        };
        return std::make_unique<HeapHostBuffer>(bytes);
    }
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
    // Optional non-owning view into an existing tensor allocation. Offsets are
    // expressed in bytes so one FP32 arena can host FP32 and BF16 scratch
    // tensors. The backing storage must outlive every returned view.
    virtual bool supports_tensor_views() const { return false; }
    virtual std::unique_ptr<DeviceTensor> tensor_view(
            DeviceTensor &storage,
            uint64_t byte_offset,
            uint64_t count,
            uint32_t elem_size,
            DeviceTensorDType dtype,
            const char *label) {
        (void)storage;
        (void)byte_offset;
        (void)count;
        (void)elem_size;
        (void)dtype;
        (void)label;
        return nullptr;
    }
    virtual std::unique_ptr<DeviceTensor> scratch_like(
            const DeviceTensor &source,
            uint64_t count,
            const char *label) {
        if (source.dtype == DeviceTensorDType::BF16) {
            return tensor_bf16(count, label);
        }
        return scratch_f32(count, label);
    }
    virtual std::unique_ptr<DeviceWeight> weight_f32(const float *data, uint64_t count, const char *label) = 0;
    virtual std::unique_ptr<DeviceWeight> weight_q8_0(const void *data, uint64_t rows, uint64_t cols, const char *label) = 0;
    virtual std::unique_ptr<DeviceWeight> weight_bf16(const void *data,
                                                       uint64_t rows,
                                                       uint64_t cols,
                                                       const char *label) {
        (void)data; (void)rows; (void)cols; (void)label;
        throw std::runtime_error("BF16 weights are not supported by this backend");
    }
    // Host-resident BF16 lookup weight. This is intended for sparse row
    // access (token embeddings), not matmul. CUDA backends can gather the
    // requested rows into pinned staging memory and transfer only those rows.
    virtual std::unique_ptr<DeviceWeight> weight_bf16_host(
            const void *data,
            uint64_t rows,
            uint64_t cols,
            const char *label) {
        (void)data; (void)rows; (void)cols; (void)label;
        throw std::runtime_error(
            "host-resident BF16 weights are not supported by this backend");
    }
    virtual std::unique_ptr<DeviceWeight> weight_fp8_e4m3(const void *data,
                                                           const void *scale_data,
                                                           uint64_t scale_count,
                                                           uint64_t rows,
                                                           uint64_t cols,
                                                           const char *label) {
        (void)data; (void)scale_data; (void)scale_count;
        (void)rows; (void)cols; (void)label;
        throw std::runtime_error("FP8 weights are not supported by this backend");
    }
    virtual std::unique_ptr<DeviceWeight> weight_nvfp4_e2m1(
            const void *packed_data,
            const void *scale_data,
            uint64_t scale_rows,
            uint64_t scale_cols,
            float input_global_scale_divisor,
            float weight_global_scale_divisor,
            uint64_t rows,
            uint64_t cols,
            const char *label) {
        (void)packed_data; (void)scale_data; (void)scale_rows; (void)scale_cols;
        (void)input_global_scale_divisor; (void)weight_global_scale_divisor;
        (void)rows; (void)cols; (void)label;
        throw std::runtime_error("NVFP4 weights are not supported by this backend");
    }

    // Optionally replace two separately allocated FP8 weights with one
    // contiguous allocation while preserving both logical DeviceWeight
    // handles. Backends can then issue one larger GEMM for shared-input
    // fanouts without retaining a duplicate packed copy.
    virtual DeviceStatus prepare_fp8_fanout_pair(DeviceWeight &first,
                                                  DeviceWeight &second) {
        (void)first;
        (void)second;
        return {};
    }

    // Optionally make two NVFP4 weights contiguous so a shared-input fanout
    // can run as one larger Tensor Core GEMM.
    virtual DeviceStatus prepare_nvfp4_fanout_pair(DeviceWeight &first,
                                                    DeviceWeight &second) {
        (void)first;
        (void)second;
        return {};
    }

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

    // Batched counterpart used by MTP verification. Backends may compute the
    // gate/up projections together and write only the final SwiGLU output.
    virtual DeviceStatus q8_0_matmul_silu_mul(DeviceTensor &out,
                                              const DeviceWeight &weight_gate,
                                              const DeviceWeight &weight_up,
                                              const DeviceTensor &x,
                                              uint32_t batch,
                                              uint32_t in_stride,
                                              uint32_t out_stride) {
        (void)out; (void)weight_gate; (void)weight_up; (void)x;
        (void)batch; (void)in_stride; (void)out_stride;
        return {false, "q8_0_matmul_silu_mul not implemented by this backend"};
    }

    // Optimized NVFP4 prefill FFN:
    // out = W_down * (silu(W_gate * x) * (W_up * x)).
    // The CUDA backend keeps the gate/up result in BF16 and directly quantizes
    // SwiGLU into the down projection input. Unsupported formats and small
    // batches return !ok so callers can use the generic three-op path.
    virtual DeviceStatus nvfp4_ffn_prefill(DeviceTensor &out,
                                           const DeviceWeight &weight_gate,
                                           const DeviceWeight &weight_up,
                                           const DeviceWeight &weight_down,
                                           const DeviceWeight &norm_weight,
                                           const DeviceTensor &x,
                                           uint32_t batch,
                                           uint32_t in_stride,
                                           uint32_t ffn_stride,
                                           uint32_t out_stride,
                                           float norm_eps) {
        (void)out; (void)weight_gate; (void)weight_up; (void)weight_down;
        (void)norm_weight; (void)x; (void)batch; (void)in_stride;
        (void)ffn_stride; (void)out_stride; (void)norm_eps;
        return {false, "NVFP4 fused prefill FFN not implemented by this backend"};
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

    // RMSNorm followed by several linear projections that share the normalized
    // activation. CUDA can fuse normalization with input quantization; other
    // backends retain the exact two-operation behavior through this fallback.
    virtual DeviceStatus rms_norm_q8_0_matmul_fanout(
            DeviceTensor &normalized,
            DeviceTensor *const *outs,
            const DeviceWeight *const *weights,
            const uint32_t *out_strides,
            uint32_t fanout,
            const DeviceTensor &x,
            const DeviceWeight &norm_weight,
            uint32_t batch,
            uint32_t in_stride,
            float eps) {
        if (auto st = rms_norm_batch(
                normalized, x, norm_weight, batch, in_stride, eps);
            !st.ok) {
            return st;
        }
        return q8_0_matmul_fanout(
            outs, weights, out_strides, fanout,
            normalized, batch, in_stride);
    }

    // Recurrent projections may leave a compatible packed FP8 QKV/gate pair
    // in backend-owned storage for immediate consumption by recurrent_batch.
    // Other backends and unsupported routes materialize the normal outputs.
    virtual DeviceStatus rms_norm_q8_0_matmul_recurrent_fanout(
            DeviceTensor &normalized,
            DeviceTensor *const *outs,
            const DeviceWeight *const *weights,
            const uint32_t *out_strides,
            uint32_t fanout,
            const DeviceTensor &x,
            const DeviceWeight &norm_weight,
            uint32_t batch,
            uint32_t in_stride,
            float eps) {
        return rms_norm_q8_0_matmul_fanout(
            normalized, outs, weights, out_strides, fanout,
            x, norm_weight, batch, in_stride, eps);
    }

    // True when the recurrent projection fanout will leave `proj` and `gate`
    // as backend-owned lazy results that recurrent_batch consumes without
    // writing either output tensor. Executors may then use overlapping
    // metadata-only views for those two outputs. The default is conservative.
    virtual bool can_defer_recurrent_projection_outputs(
            const DeviceWeight *const *weights,
            uint32_t fanout,
            const DeviceWeight &norm_weight,
            bool input_is_bf16,
            uint32_t input_stride,
            uint32_t batch,
            uint32_t num_k_heads,
            uint32_t num_v_heads,
            uint32_t head_k_dim,
            uint32_t head_v_dim,
            uint32_t proj_count,
            uint32_t proj_stride,
            uint32_t gate_stride,
            uint32_t alpha_stride,
            uint32_t beta_stride,
            bool state_checkpoints,
            bool deltanet_capture) const {
        (void)weights;
        (void)fanout;
        (void)norm_weight;
        (void)input_is_bf16;
        (void)input_stride;
        (void)batch;
        (void)num_k_heads;
        (void)num_v_heads;
        (void)head_k_dim;
        (void)head_v_dim;
        (void)proj_count;
        (void)proj_stride;
        (void)gate_stride;
        (void)alpha_stride;
        (void)beta_stride;
        (void)state_checkpoints;
        (void)deltanet_capture;
        return false;
    }

    // Attention projections can defer FP8 output scaling until Q/K
    // normalization and FP8 V-cache append consume the values.
    virtual DeviceStatus rms_norm_q8_0_matmul_attention_fanout(
            DeviceTensor &normalized,
            DeviceTensor *const *outs,
            const DeviceWeight *const *weights,
            const uint32_t *out_strides,
            uint32_t fanout,
            const DeviceTensor &x,
            const DeviceWeight &norm_weight,
            uint32_t batch,
            uint32_t in_stride,
            float eps) {
        return rms_norm_q8_0_matmul_fanout(
            normalized, outs, weights, out_strides, fanout,
            x, norm_weight, batch, in_stride, eps);
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

    // Format-neutral linear operations. Their defaults preserve the original
    // Q8_0 behavior exactly; mixed-precision backends override only these
    // entry points and dispatch on DeviceWeight::format.
    virtual DeviceStatus linear_get_row(DeviceTensor &out,
                                        const DeviceWeight &weight,
                                        uint64_t row) {
        return q8_0_get_row(out, weight, row);
    }
    virtual DeviceStatus linear_get_row_from_argmax(DeviceTensor &out,
                                                     const DeviceWeight &weight,
                                                     const DeviceArgmaxBuffer &argmaxes,
                                                     uint32_t index) {
        return q8_0_get_row_from_argmax(out, weight, argmaxes, index);
    }
    virtual DeviceStatus linear_get_rows_batch(DeviceTensor &out,
                                                const DeviceWeight &weight,
                                                const uint64_t *rows,
                                                uint32_t batch) {
        return q8_0_get_rows_batch(out, weight, rows, batch);
    }
    virtual DeviceStatus linear_matvec(DeviceTensor &out,
                                       const DeviceWeight &weight,
                                       const DeviceTensor &x) {
        return q8_0_matvec(out, weight, x);
    }
    virtual DeviceStatus linear_matvec_add(DeviceTensor &accum,
                                           const DeviceWeight &weight,
                                           const DeviceTensor &x) {
        return q8_0_matvec_add(accum, weight, x);
    }
    virtual DeviceStatus linear_matvec_silu_mul(DeviceTensor &out,
                                                const DeviceWeight &weight_gate,
                                                const DeviceWeight &weight_up,
                                                const DeviceTensor &x) {
        return q8_0_matvec_silu_mul(out, weight_gate, weight_up, x);
    }
    virtual DeviceStatus linear_matvec_fanout(DeviceTensor *const *outs,
                                              const DeviceWeight *const *weights,
                                              uint32_t n,
                                              const DeviceTensor &x) {
        for (uint32_t i = 0; i < n; ++i) {
            if (auto st = linear_matvec(*outs[i], *weights[i], x); !st.ok) return st;
        }
        return {};
    }
    virtual DeviceStatus linear_matmul(DeviceTensor &out,
                                       const DeviceWeight &weight,
                                       const DeviceTensor &x,
                                       uint32_t batch,
                                       uint32_t in_stride,
                                       uint32_t out_stride) {
        return q8_0_matmul(out, weight, x, batch, in_stride, out_stride);
    }
    virtual DeviceStatus linear_matmul_fanout(DeviceTensor *const *outs,
                                              const DeviceWeight *const *weights,
                                              const uint32_t *out_strides,
                                              uint32_t n,
                                              const DeviceTensor &x,
                                              uint32_t batch,
                                              uint32_t in_stride) {
        for (uint32_t i = 0; i < n; ++i) {
            if (auto st = linear_matmul(*outs[i], *weights[i], x, batch,
                                        in_stride, out_strides[i]); !st.ok) {
                return st;
            }
        }
        return {};
    }
    virtual DeviceStatus linear_matmul_add(DeviceTensor &out,
                                           const DeviceTensor &residual,
                                           DeviceTensor &matmul_tmp,
                                           const DeviceWeight &weight,
                                           const DeviceTensor &x,
                                           uint32_t batch,
                                           uint32_t in_stride,
                                           uint32_t out_stride) {
        if (auto st = linear_matmul(matmul_tmp, weight, x, batch,
                                    in_stride, out_stride); !st.ok) {
            return st;
        }
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
                                          uint32_t checkpoint_count = 0,
                                          // DeltaNet retrieval block-boundary state
                                          // capture (kvmem retrieval_method==deltanet;
                                          // see docs/kvmem_deltanet_retrieval_design.md). When
                                          // dn_state_snap != nullptr the CUDA backend
                                          // captures S_j per block into dn_state_snap
                                          // (offset dn_layer_snap_off) and accumulates
                                          // each block's in-block log-decay into
                                          // dn_decaysum (offset dn_layer_decay_off).
                                          // Forces the qw3 (non-ported) delta kernel.
                                          DeviceTensor *dn_state_snap = nullptr,
                                          DeviceTensor *dn_decaysum = nullptr,
                                          uint64_t dn_layer_snap_off = 0,
                                          uint64_t dn_layer_decay_off = 0,
                                          uint32_t dn_block_tokens = 0,
                                          uint32_t dn_base_pos = 0,
                                          uint32_t dn_n_blocks = 0) {
        (void)state_checkpoints;
        (void)conv_state_checkpoints;
        (void)checkpoint_count;
        (void)dn_state_snap; (void)dn_decaysum; (void)dn_layer_snap_off;
        (void)dn_layer_decay_off; (void)dn_block_tokens; (void)dn_base_pos;
        (void)dn_n_blocks;
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

    // Cross-request recurrent layer: each row has an independent recurrent
    // state and conv state. Inputs and outputs are row-major batch buffers.
    // This is for continuous batching decode, not time-batched prefill.
    virtual DeviceStatus recurrent_batch_independent(DeviceTensor &core,
                                                     DeviceTensor &state_batch,
                                                     DeviceTensor &conv_state_batch,
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
                                                     uint32_t state_stride,
                                                     uint32_t conv_state_stride,
                                                     float eps) {
        (void)core;
        (void)state_batch;
        (void)conv_state_batch;
        (void)conv_out_buf;
        (void)proj;
        (void)gate;
        (void)alpha;
        (void)beta;
        (void)conv;
        (void)ssm_a;
        (void)dt_bias;
        (void)ssm_norm;
        (void)batch;
        (void)num_k_heads;
        (void)num_v_heads;
        (void)head_k_dim;
        (void)head_v_dim;
        (void)conv_kernel_size;
        (void)proj_count;
        (void)proj_stride;
        (void)gate_stride;
        (void)alpha_stride;
        (void)beta_stride;
        (void)core_stride;
        (void)state_stride;
        (void)conv_state_stride;
        (void)eps;
        return {false, "recurrent_batch_independent requires backend override"};
    }

    // Cross-request recurrent prefill: q_indptr describes ragged token ranges
    // in the row-major proj/gate/alpha/beta/core buffers. Each sequence has
    // an independent recurrent state and conv state, and tokens within a
    // sequence are processed sequentially.
    virtual DeviceStatus recurrent_batch_ragged(DeviceTensor &core,
                                                DeviceTensor &state_batch,
                                                DeviceTensor &conv_state_batch,
                                                DeviceTensor &conv_out_buf,
                                                const DeviceTensor &proj,
                                                const DeviceTensor &gate,
                                                const DeviceTensor &alpha,
                                                const DeviceTensor &beta,
                                                const DeviceTensor &q_indptr,
                                                const int32_t *q_indptr_host,
                                                const DeviceWeight &conv,
                                                const DeviceWeight &ssm_a,
                                                const DeviceWeight &dt_bias,
                                                const DeviceWeight &ssm_norm,
                                                uint32_t batch,
                                                uint32_t total_q,
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
                                                uint32_t state_stride,
                                                uint32_t conv_state_stride,
                                                float eps,
                                                DeviceTensor *state_checkpoints = nullptr,
                                                DeviceTensor *conv_state_checkpoints = nullptr,
                                                uint32_t checkpoint_count = 0) {
        (void)core;
        (void)state_batch;
        (void)conv_state_batch;
        (void)conv_out_buf;
        (void)proj;
        (void)gate;
        (void)alpha;
        (void)beta;
        (void)q_indptr;
        (void)q_indptr_host;
        (void)conv;
        (void)ssm_a;
        (void)dt_bias;
        (void)ssm_norm;
        (void)batch;
        (void)total_q;
        (void)num_k_heads;
        (void)num_v_heads;
        (void)head_k_dim;
        (void)head_v_dim;
        (void)conv_kernel_size;
        (void)proj_count;
        (void)proj_stride;
        (void)gate_stride;
        (void)alpha_stride;
        (void)beta_stride;
        (void)core_stride;
        (void)state_stride;
        (void)conv_state_stride;
        (void)eps;
        (void)state_checkpoints;
        (void)conv_state_checkpoints;
        (void)checkpoint_count;
        return {false, "recurrent_batch_ragged requires backend override"};
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

    // Decode-only Q/K preparation. CUDA backends can combine both per-head
    // RMS norms and both RoPE transforms into one launch.
    virtual DeviceStatus rmsnorm_rope_qk(DeviceTensor &q,
                                         DeviceTensor &k,
                                         const DeviceWeight &q_weight,
                                         const DeviceWeight &k_weight,
                                         uint32_t n_q_heads,
                                         uint32_t n_kv_heads,
                                         uint32_t q_head_stride,
                                         uint32_t k_head_stride,
                                         uint32_t head_dim,
                                         uint32_t rope_dim,
                                         uint32_t pos,
                                         float theta,
                                         float eps) {
        if (auto st = rmsnorm_per_head(
                q, q_weight, n_q_heads, q_head_stride, head_dim, eps); !st.ok) {
            return st;
        }
        if (auto st = rmsnorm_per_head(
                k, k_weight, n_kv_heads, k_head_stride, head_dim, eps); !st.ok) {
            return st;
        }
        if (auto st = rope_partial(
                q, n_q_heads, q_head_stride, rope_dim, pos, theta); !st.ok) {
            return st;
        }
        return rope_partial(
            k, n_kv_heads, k_head_stride, rope_dim, pos, theta);
    }

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

    // Batched RoPE where every row has an arbitrary logical position. This is
    // required for cross-request continuous decode: active rows are not
    // consecutive positions from the same sequence.
    virtual DeviceStatus rope_partial_batch_positions(DeviceTensor &x,
                                                      uint32_t batch,
                                                      uint32_t batch_stride,
                                                      uint32_t n_units,
                                                      uint32_t per_unit_stride,
                                                      uint32_t rope_dim,
                                                      const DeviceTensor &positions,
                                                      float theta) {
        (void)x; (void)batch; (void)batch_stride; (void)n_units;
        (void)per_unit_stride; (void)rope_dim; (void)positions; (void)theta;
        return {false, "rope_partial_batch_positions requires backend override"};
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

    // Block-sparse re-RoPE: re-bake one block's stored K in place, from its
    // current bake position (from_base) to a new window slot (to_base). The
    // block occupies logical window positions [win_base..win_base+n_tokens) in
    // the page table. Operates directly on the paged cache (fp16/fp32) where K
    // is stored — no copy. De-rotate(from)+re-rotate(to) with the same __sincosf
    // so the prior bake's range-reduction error cancels (lossless to ~1 ULP).
    virtual DeviceStatus rope_block_remap_paged_device(
                                                      DeviceTensor &k_cache,
                                                      uint32_t n_tokens,
                                                      uint32_t n_kv_heads,
                                                      uint32_t per_pos_size,
                                                      uint32_t head_dim,
                                                      uint32_t rope_dim,
                                                      uint32_t win_base,
                                                      int32_t from_base,
                                                      int32_t to_base,
                                                      const DeviceTensor &page_indices,
                                                      uint32_t page_size,
                                                      float theta) {
        (void)k_cache; (void)n_tokens; (void)n_kv_heads; (void)per_pos_size;
        (void)head_dim; (void)rope_dim; (void)win_base; (void)from_base;
        (void)to_base; (void)page_indices; (void)page_size; (void)theta;
        return {false, "rope_block_remap_paged_device requires backend override"};
    }

    // Batched re-RoPE: move n_blocks blocks in ONE launch. to_base/from_base/
    // n_tokens are device int32 arrays of length n_blocks (window slot, original
    // bake position, and token count per block); max_n_tokens bounds grid.x.
    // Equivalent to calling rope_block_remap_paged_device once per block, but
    // collapses the per-(layer,block) launch storm during window assembly.
    virtual DeviceStatus rope_block_remap_paged_batched_device(
            DeviceTensor &k_cache, uint32_t n_blocks, uint32_t max_n_tokens,
            uint32_t n_kv_heads, uint32_t per_pos_size, uint32_t head_dim,
            uint32_t rope_dim, const DeviceTensor &to_base,
            const DeviceTensor &from_base, const DeviceTensor &n_tokens,
            const DeviceTensor &page_indices, uint32_t page_size, float theta,
            const DeviceTensor *rope_sincos = nullptr,
            uint32_t rope_table_positions = 0) {
        (void)k_cache; (void)n_blocks; (void)max_n_tokens; (void)n_kv_heads;
        (void)per_pos_size; (void)head_dim; (void)rope_dim; (void)to_base;
        (void)from_base; (void)n_tokens; (void)page_indices; (void)page_size;
        (void)theta; (void)rope_sincos; (void)rope_table_positions;
        return {false,
                "rope_block_remap_paged_batched_device requires backend override"};
    }

    // Build a persistent FP32 table laid out as
    // [position, rope_dim / 2, {sin, cos}].  KVMem assembly reuses it across
    // every KV head and standard-attention layer, eliminating billions of
    // duplicate powf/sincosf calls while retaining the legacy rotation math.
    virtual DeviceStatus build_rope_sincos_table_device(
            DeviceTensor &table, uint32_t positions, uint32_t rope_dim,
            float theta) {
        (void)table; (void)positions; (void)rope_dim; (void)theta;
        return {false,
                "build_rope_sincos_table_device requires backend override"};
    }

    // Materialize packed, unrotated raw K into arbitrary paged window slots and
    // apply RoPE exactly once at each destination position. `raw_k` begins at
    // raw_element_offset; raw_block_stride_elements==0 means the compatibility
    // layout [n_blocks,max_n_tokens,per_pos_size]. A non-zero stride supports
    // block-major, layer-interleaved assembly buffers without a CPU transpose.
    // This is the drift-free cold/periodic refresh path for immutable K.
    virtual DeviceStatus raw_k_scatter_rope_paged_batched_device(
            DeviceTensor &k_cache, const DeviceTensor &raw_k,
            uint64_t raw_element_offset, uint32_t n_blocks,
            uint32_t max_n_tokens, uint32_t n_kv_heads,
            uint32_t per_pos_size, uint32_t head_dim, uint32_t rope_dim,
            const DeviceTensor &to_base, const DeviceTensor &n_tokens,
            const DeviceTensor &page_indices, uint32_t page_size, float theta,
            const DeviceTensor *rope_sincos = nullptr,
            uint32_t rope_table_positions = 0,
            uint64_t raw_block_stride_elements = 0) {
        (void)k_cache; (void)raw_k; (void)raw_element_offset; (void)n_blocks;
        (void)max_n_tokens; (void)n_kv_heads; (void)per_pos_size;
        (void)head_dim; (void)rope_dim; (void)to_base; (void)n_tokens;
        (void)page_indices; (void)page_size; (void)theta;
        (void)rope_sincos; (void)rope_table_positions;
        (void)raw_block_stride_elements;
        return {false,
                "raw_k_scatter_rope_paged_batched_device requires backend override"};
    }

    // Block-sparse cumulative-attention selection signal (#40). Both are
    // no-ops on backends without an override (selection falls back to recency).
    //
    // block_kmean_paged_device: compute a per-window-block representative K
    // vector = mean baked K over the block's tokens, for `n_blocks` window
    // blocks. win_base[w]/blk_tokens[w] (device int32) give each block's window
    // position range; the window page table addresses physical slots. kbar
    // output is fp32 [n_blocks, n_kv_heads, head_dim]. Called once per reselect
    // interval (amortized), not per step.
    virtual DeviceStatus block_kmean_paged_device(const DeviceTensor &k_cache,
                                                  DeviceTensor &kbar,
                                                  uint32_t n_blocks,
                                                  uint32_t n_kv_heads,
                                                  uint32_t per_pos_size,
                                                  uint32_t head_dim,
                                                  const DeviceTensor &win_base,
                                                  const DeviceTensor &blk_tokens,
                                                  const DeviceTensor &page_indices,
                                                  uint32_t page_size) {
        (void)k_cache; (void)kbar; (void)n_blocks; (void)n_kv_heads;
        (void)per_pos_size; (void)head_dim; (void)win_base; (void)blk_tokens;
        (void)page_indices; (void)page_size;
        return {false, "block_kmean_paged_device requires backend override"};
    }

    // block_attn_score_step_device: for the current decode token's RoPE-baked Q
    // (layout [n_heads, q_stride], head_dim floats read per head), add to each
    // window block's accumulator slot accum[w] the sum over query heads of
    // max(0, scale * (q_head . kbar_block[kv_head])). GPU-resident accumulator;
    // no D2H. Called once per decode step at a single representative layer.
    virtual DeviceStatus block_attn_score_step_device(DeviceTensor &accum,
                                                      const DeviceTensor &q,
                                                      const DeviceTensor &kbar,
                                                      uint32_t q_stride,
                                                      uint32_t n_blocks,
                                                      uint32_t n_heads,
                                                      uint32_t n_kv_heads,
                                                      uint32_t head_dim,
                                                      float scale) {
        (void)accum; (void)q; (void)kbar; (void)q_stride; (void)n_blocks;
        (void)n_heads; (void)n_kv_heads; (void)head_dim; (void)scale;
        return {false, "block_attn_score_step_device requires backend override"};
    }

    // Diagnostic-only true attention mass tracing. Recomputes the decode
    // softmax for one query token over a paged KV cache, then aggregates the
    // normalized attention weights by kvmem window block. `mass` is fp32
    // [n_window_blocks + 1], where the last slot is the decode tail after the
    // last assembled block. The sum of `mass` is n_heads; callers can normalize
    // by the sum to get a per-layer probability distribution. This is intended
    // for trace/analysis runs and is not on the default inference path.
    virtual DeviceStatus block_attention_mass_paged_device(
                                                      DeviceTensor &mass,
                                                      const DeviceTensor &q,
                                                      uint32_t q_stride,
                                                      const DeviceTensor &k_cache,
                                                      uint32_t n_window_blocks,
                                                      uint32_t n_heads,
                                                      uint32_t n_kv_heads,
                                                      uint32_t per_pos_size,
                                                      uint32_t head_dim,
                                                      const DeviceTensor &win_base,
                                                      const DeviceTensor &blk_tokens,
                                                      const DeviceTensor &page_indices,
                                                      uint32_t n_pages,
                                                      uint32_t page_size,
                                                      uint32_t seq_len,
                                                      float scale) {
        (void)mass; (void)q; (void)q_stride; (void)k_cache; (void)n_window_blocks;
        (void)n_heads; (void)n_kv_heads; (void)per_pos_size; (void)head_dim;
        (void)win_base; (void)blk_tokens; (void)page_indices; (void)n_pages;
        (void)page_size; (void)seq_len; (void)scale;
        return {false, "block_attention_mass_paged_device requires backend override"};
    }

    // block_kmean_content_paged_device: global content-frame mean-Key index
    // (#48). Like block_kmean_paged_device, but reads through the FULL repository
    // page table at each block's TRUE positions and DE-RoPEs every token's K by
    // (orig_base[w]+tok) before averaging, recovering the position-invariant raw
    // key. orig_base[w]/blk_tokens[w] (device int32) give each block's true
    // position range; page_indices is the full repository table. kbar output is
    // fp32 [n_blocks, n_kv_heads, head_dim] indexed by block_id. Built ONCE from
    // the pristine post-prefill cache (blocks baked at true positions), immutable
    // thereafter — never rebuilt as blocks are re-RoPE'd into windows.
    // out_elem_off (default 0) writes kbar starting at that fp32 element offset, so
    // a single [L, n_blocks, n_kv_heads, head_dim] buffer can be filled one normal
    // layer's slice at a time (multi-layer selection, #86).
    virtual DeviceStatus block_kmean_content_paged_device(const DeviceTensor &k_cache,
                                                          DeviceTensor &kbar,
                                                          uint32_t n_blocks,
                                                          uint32_t n_kv_heads,
                                                          uint32_t per_pos_size,
                                                          uint32_t head_dim,
                                                          uint32_t rope_dim,
                                                          const DeviceTensor &orig_base,
                                                          const DeviceTensor &blk_tokens,
                                                          const DeviceTensor &page_indices,
                                                          uint32_t page_size,
                                                          float theta,
                                                          uint64_t out_elem_off = 0) {
        (void)k_cache; (void)kbar; (void)n_blocks; (void)n_kv_heads;
        (void)per_pos_size; (void)head_dim; (void)rope_dim; (void)orig_base;
        (void)blk_tokens; (void)page_indices; (void)page_size; (void)theta;
        (void)out_elem_off;
        return {false, "block_kmean_content_paged_device requires backend override"};
    }

    // block_kmean_content_batch_device (#91): build per-block content mean-Keys for
    // one normal-attention layer directly from the freshly-RoPE'd prefill K batch,
    // so EVERY block is indexed during prefill (the paged builder above can only run
    // once from the pristine cache and misses the tail of histories larger than the
    // GPU page pool). k_batch is the contiguous fp32 [batch, n_kv_heads*head_dim]
    // (row stride k_stride) K buffer RoPE-baked at rope_base; each chunk-local block
    // owns up to blk_tokens contiguous rows, de-RoPE'd at (rope_base + row). Writes
    // kbar (fp32 [.., n_kv_heads, head_dim]) at global block (kbar_block_base + j),
    // so one [L, n_blocks, n_kv_heads, head_dim] buffer is filled a (layer, chunk)
    // slice at a time. Chunks are block-aligned, so blocks never straddle chunks.
    // src_row_off (default 0): skip this many leading rows of k_batch before block 0
    // begins, so an over-budget resume suffix whose first token is mid-block can index
    // only its full trailing blocks (the straddling boundary block keeps its prior mean).
    // n_subblocks (default 1, plain mean-k): capture this many equal, non-overlapping
    // sub-block means per block (sub_tokens = ceil(blk_tokens/n_subblocks), last
    // sub-block absorbs the remainder); kbar is then [.., n_subblocks, n_kv_heads,
    // head_dim] with block b's sub-blocks at slots [b*n_subblocks, (b+1)*n_subblocks).
    virtual DeviceStatus block_kmean_content_batch_device(const DeviceTensor &k_batch,
                                                          DeviceTensor &kbar,
                                                          uint64_t kbar_block_base,
                                                          uint32_t n_blocks_chunk,
                                                          uint32_t k_stride,
                                                          uint32_t batch,
                                                          uint32_t blk_tokens,
                                                          uint32_t n_kv_heads,
                                                          uint32_t head_dim,
                                                          uint32_t rope_dim,
                                                          int32_t rope_base,
                                                          float theta,
                                                          uint32_t src_row_off = 0,
                                                          uint32_t n_subblocks = 1) {
        (void)k_batch; (void)kbar; (void)kbar_block_base; (void)n_blocks_chunk;
        (void)k_stride; (void)batch; (void)blk_tokens; (void)n_kv_heads;
        (void)head_dim; (void)rope_dim; (void)rope_base; (void)theta; (void)src_row_off;
        (void)n_subblocks;
        return {false, "block_kmean_content_batch_device requires backend override"};
    }

    // Hierarchical Fixed-4 non-contiguous Key-direction prototypes. Split each
    // logical block into contiguous 32-token slices; for every slice,
    // normal-attention layer, and KV head, de-RoPE the freshly produced Keys,
    // choose four farthest-first cosine directions, assign every token to its
    // nearest direction, and store the four cluster means in the same layout
    // consumed by the existing sub-block scorer. Experimental CUDA path.
    virtual DeviceStatus block_kdirection_fixed4_batch_device(
            const DeviceTensor &k_batch,
            DeviceTensor &kbar,
            uint64_t kbar_block_base,
            uint32_t n_blocks_chunk,
            uint32_t k_stride,
            uint32_t batch,
            uint32_t blk_tokens,
            uint32_t n_kv_heads,
            uint32_t head_dim,
            uint32_t rope_dim,
            int32_t rope_base,
            float theta,
            uint32_t src_row_off = 0) {
        (void)k_batch; (void)kbar; (void)kbar_block_base;
        (void)n_blocks_chunk; (void)k_stride; (void)batch;
        (void)blk_tokens; (void)n_kv_heads; (void)head_dim;
        (void)rope_dim; (void)rope_base; (void)theta;
        (void)src_row_off;
        return {
            false,
            "block_kdirection_fixed4_batch_device requires backend override"};
    }

    // Bounded dense scratch for Adaptive Key-direction capture. Every slice
    // writes exact P=1, P=2, and P=4 candidates (seven rows total) plus
    // per-head [E1,E2,E4] cosine residuals. The executor performs the threshold
    // decision and persists only selected rows in a packed index.
    virtual DeviceStatus block_kdirection_adaptive_batch_device(
            const DeviceTensor &k_batch,
            DeviceTensor &candidate_scratch,
            DeviceTensor &residual_scratch,
            uint32_t n_blocks_chunk,
            uint32_t k_stride,
            uint32_t batch,
            uint32_t blk_tokens,
            uint32_t n_kv_heads,
            uint32_t head_dim,
            uint32_t rope_dim,
            int32_t rope_base,
            float theta,
            uint32_t src_row_off = 0) {
        (void)k_batch; (void)candidate_scratch; (void)residual_scratch;
        (void)n_blocks_chunk; (void)k_stride; (void)batch;
        (void)blk_tokens; (void)n_kv_heads; (void)head_dim;
        (void)rope_dim; (void)rope_base; (void)theta; (void)src_row_off;
        return {
            false,
            "block_kdirection_adaptive_batch_device requires backend override"};
    }

    // Incremental counterpart used when a logical block is produced by several
    // prefill calls (multi-turn transcript replay / resumed sessions). The first
    // input row belongs at `first_block_token_offset` in the first destination
    // block. Existing means for [0, offset) are merged by their exact row count;
    // later blocks are overwritten from an empty prefix.
    virtual DeviceStatus block_kmean_content_batch_merge_device(
            const DeviceTensor &k_batch,
            DeviceTensor &kbar,
            uint64_t kbar_block_base,
            uint32_t n_blocks_chunk,
            uint32_t k_stride,
            uint32_t batch,
            uint32_t blk_tokens,
            uint32_t first_block_token_offset,
            uint32_t n_kv_heads,
            uint32_t head_dim,
            uint32_t rope_dim,
            int32_t rope_base,
            float theta,
            uint32_t n_subblocks = 1) {
        (void)k_batch; (void)kbar; (void)kbar_block_base;
        (void)n_blocks_chunk; (void)k_stride; (void)batch;
        (void)blk_tokens; (void)first_block_token_offset;
        (void)n_kv_heads; (void)head_dim; (void)rope_dim;
        (void)rope_base; (void)theta; (void)n_subblocks;
        return {false,
                "block_kmean_content_batch_merge_device requires backend override"};
    }

    // derope_store_content_batch_device (#104, raw-key MaxSim): like
    // block_kmean_content_batch_device but STORES each de-RoPE'd K row instead of
    // averaging — fills kraw (fp32 [.., n_kv_heads, head_dim]) so the MaxSim scorer
    // can take a MAX over a block's per-token keys rather than scoring one diluted
    // mean key. Same de-rotate math. out_base_elem indexes the (slot, first-global-
    // token) origin element; row r is de-RoPE'd at (rope_base + r) and written there.
    virtual DeviceStatus derope_store_content_batch_device(const DeviceTensor &k_batch,
                                                           DeviceTensor &kraw,
                                                           uint64_t out_base_elem,
                                                           uint32_t k_stride,
                                                           uint32_t batch,
                                                           uint32_t n_kv_heads,
                                                           uint32_t head_dim,
                                                           uint32_t rope_dim,
                                                           int32_t rope_base,
                                                           float theta) {
        (void)k_batch; (void)kraw; (void)out_base_elem; (void)k_stride; (void)batch;
        (void)n_kv_heads; (void)head_dim; (void)rope_dim; (void)rope_base; (void)theta;
        return {false, "derope_store_content_batch_device requires backend override"};
    }

    // Widen a slab of raw-K rows from the KV cache dtype to fp32 so the
    // prototype builders above can consume them. Attaching a context archive
    // rebuilds the retrieval index from the persisted raw-K authority instead
    // of from live prefill activations, and those builders require fp32 input.
    // `src` holds `rows` rows of `row_elems` elements each, contiguous; `dst`
    // receives the same shape as fp32. Raw K is already in the content frame,
    // so callers pass rope_dim=0 to the builders to make the de-rotate an
    // identity and reproduce byte-identical prototypes.
    virtual DeviceStatus widen_raw_k_to_f32_device(const DeviceTensor &src,
                                                   DeviceTensor &dst,
                                                   uint64_t rows,
                                                   uint32_t row_elems) {
        (void)src; (void)dst; (void)rows; (void)row_elems;
        return {false, "widen_raw_k_to_f32_device requires backend override"};
    }

    // block_attn_score_exactmass_device (ExactMass, AgentKV port): per normal-
    // attention layer (one launch, ACCUMULATES into score), for every block w
    //   score[w] += head_w · Σ_{j<M} Σ_{qh<n_heads}
    //                 Σ_{t : t/bt==w} softmax_t( scale·(q[j,qh] · kraw[t,kvh]) ),
    // kvh = qh/(n_heads/n_kv_heads), head_w = 1/(L·M·n_heads). When group_mean=1
    // (strict AgentKV) the per-CTA query is the MEAN of the KV group's query heads
    // formed BEFORE the softmax; the caller sets head_w = 1/(L·M·n_kv_heads). Unlike
    // block_attn_score_softmax_pages_device (softmax over per-block MEAN keys →
    // exp(mean), needle diluted), the exp here is applied to every RAW token logit
    // and THEN summed into blocks (Σexp) under one global softmax denominator, so a
    // needle token's mass survives and blocks compete globally. q_layer is
    // [M, n_heads, head_dim] fp32; kraw is [total_tokens, n_kv_heads, head_dim] fp32;
    // q_elem_off / kraw_elem_off select this layer's slice. block_tokens (bt) is the
    // uniform per-block token span (t/bt is the block id). score is fp32 [n_blocks]
    // (caller zeros once). Returns an error if the per-block shmem exceeds the cap.
    virtual DeviceStatus block_attn_score_exactmass_device(DeviceTensor &score,
                                                           const DeviceTensor &q_layer,
                                                           const DeviceTensor &kraw,
                                                           uint32_t M,
                                                           uint32_t total_tokens,
                                                           uint32_t n_blocks,
                                                           uint32_t block_tokens,
                                                           uint32_t n_heads,
                                                           uint32_t n_kv_heads,
                                                           uint32_t head_dim,
                                                           float scale,
                                                           float head_w,
                                                           uint64_t q_elem_off = 0,
                                                           uint64_t kraw_elem_off = 0,
                                                           uint32_t group_mean = 0) {
        (void)score; (void)q_layer; (void)kraw; (void)M; (void)total_tokens;
        (void)n_blocks; (void)block_tokens; (void)n_heads; (void)n_kv_heads;
        (void)head_dim; (void)scale; (void)head_w; (void)q_elem_off;
        (void)kraw_elem_off; (void)group_mean;
        return {false, "block_attn_score_exactmass_device requires backend override"};
    }

    // block_attn_score_softmax_pages_device: query-conditioned scorer variant that
    // forms a PROPER per-(layer,token,head) softmax distribution over pages (#102).
    // One CUDA block per (layer,token); for each query head it softmaxes
    // scale·(q·k̄) over the n_blocks pages, averages over heads, and atomicAdds the
    // (1/(n_layers·n_heads))-weighted distribution into score[w]. Summed over the
    // (layer,token) grid this yields
    //   score[w] = Σ_t mean_l mean_h softmax_w( scale · (q[l,t,h]·k̄[l,w,h/group]) ),
    // i.e. accumulated attention mass each page receives from all question tokens,
    // averaged over the L normal-attention layers. score is ZEROED then accumulated.
    // scale should be 1/sqrt(head_dim) (real-attention temperature). q_multi is
    // [L, q_layer_stride, n_heads, head_dim] fp32 (first M token rows/layer read);
    // kbar_multi is [L, kbar_layer_stride, n_kv_heads, head_dim] fp32 with the first
    // n_blocks pages/layer scored. kbar_layer_stride (in blocks) may exceed n_blocks
    // when the index is allocated at a fixed session stride so preserved [0,D) slices
    // stay valid as the block count grows; callers that don't fix-stride pass
    // kbar_layer_stride == n_blocks. Up to 8192 pages use the original one-CTA
    // shared-memory fused kernel; larger requests use an exact tiled two-dot
    // implementation with a small log-sum-exp workspace.
    // excl_lo_end / excl_hi_begin mask the always-kept sink [0,excl_lo_end) and
    // recent [excl_hi_begin,n_blocks) bands out of the per-token softmax so their
    // probability mass redistributes onto the retrievable middle. Defaults
    // (0, UINT32_MAX) mask nothing (byte-identical to the pre-masking path).
    // n_subblocks (default 1): when >1, kbar_multi carries n_subblocks means per
    // block; the softmax runs over all n_blocks*n_subblocks logits and a block's
    // sub-block masses are summed into score[w] (masking bands are per block).
    // reduce_max (default 0): with n_subblocks>1, score each block by its best
    // sub-block's mass (max over sub-blocks) instead of the sum — MaxSim late
    // interaction. No-op at n_subblocks==1 (single sub-block => max == sum).
    virtual DeviceStatus block_attn_score_softmax_pages_device(DeviceTensor &score,
                                                               const DeviceTensor &q_multi,
                                                               const DeviceTensor &kbar_multi,
                                                               uint32_t n_layers,
                                                               uint32_t n_tokens,
                                                               uint32_t q_layer_stride,
                                                               uint32_t n_blocks,
                                                               uint32_t kbar_layer_stride,
                                                               uint32_t n_heads,
                                                               uint32_t n_kv_heads,
                                                               uint32_t head_dim,
                                                               float scale,
                                                               uint32_t excl_lo_end = 0,
                                                               uint32_t excl_hi_begin = UINT32_MAX,
                                                               uint32_t n_subblocks = 1,
                                                               uint32_t reduce_max = 0,
                                                               uint32_t accumulate = 0,
                                                               uint64_t q_elem_off = 0,
                                                               uint64_t kbar_elem_off = 0) {
        (void)score; (void)q_multi; (void)kbar_multi; (void)n_layers; (void)n_tokens;
        (void)q_layer_stride; (void)n_blocks; (void)kbar_layer_stride; (void)n_heads;
        (void)n_kv_heads; (void)head_dim; (void)scale; (void)excl_lo_end; (void)excl_hi_begin;
        (void)n_subblocks; (void)reduce_max; (void)accumulate; (void)q_elem_off;
        (void)kbar_elem_off;
        return {false, "block_attn_score_softmax_pages_device requires backend override"};
    }

    virtual DeviceStatus block_attn_score_softmax_adaptive_device(
        DeviceTensor &score,
        const DeviceTensor &q_multi,
        const DeviceTensor &packed_prototypes,
        const DeviceTensor &layer_offsets,
        const DeviceTensor &block_offsets,
        const DeviceTensor &block_counts,
        const DeviceTensor &prototype_blocks,
        uint32_t n_layers,
        uint32_t n_tokens,
        uint32_t q_layer_stride,
        uint32_t n_blocks,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        float scale,
        uint32_t excl_lo_end = 0,
        uint32_t excl_hi_begin = UINT32_MAX,
        uint32_t max_layer_prototypes = 0,
        uint32_t accumulate = 0) {
        (void)score; (void)q_multi; (void)packed_prototypes;
        (void)layer_offsets; (void)block_offsets; (void)block_counts;
        (void)prototype_blocks; (void)n_layers; (void)n_tokens;
        (void)q_layer_stride; (void)n_blocks; (void)n_heads;
        (void)n_kv_heads; (void)head_dim; (void)scale;
        (void)excl_lo_end; (void)excl_hi_begin;
        (void)max_layer_prototypes; (void)accumulate;
        return {
            false,
            "block_attn_score_softmax_adaptive_device requires backend override"};
    }

    // CPU-offloaded Adaptive index streaming. The executor packs one
    // layer/block-aligned prototype tile into bounded GPU staging. Pass 1
    // incrementally merges the tile into the exact per-distribution LSE; pass
    // 2 MaxSim-reduces the same tile's variable prototype runs into block
    // scores. Replaying all tiles in both passes is mathematically identical
    // to the resident packed scorer while bounding device index storage.
    virtual DeviceStatus block_attn_stream_adaptive_lse_device(
        DeviceTensor &global_max,
        DeviceTensor &global_sum,
        const DeviceTensor &q_multi,
        const DeviceTensor &prototype_tile,
        uint32_t layer,
        uint32_t n_layers,
        uint32_t n_tokens,
        uint32_t q_layer_stride,
        uint32_t prototype_count,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        float scale,
        uint32_t initialize) {
        (void)global_max; (void)global_sum; (void)q_multi;
        (void)prototype_tile; (void)layer; (void)n_layers;
        (void)n_tokens; (void)q_layer_stride; (void)prototype_count;
        (void)n_heads; (void)n_kv_heads; (void)head_dim; (void)scale;
        (void)initialize;
        return {
            false,
            "block_attn_stream_adaptive_lse_device requires backend override"};
    }

    virtual DeviceStatus block_attn_stream_adaptive_score_device(
        DeviceTensor &score,
        const DeviceTensor &q_multi,
        const DeviceTensor &prototype_tile,
        const DeviceTensor &block_offsets,
        const DeviceTensor &block_counts,
        const DeviceTensor &global_max,
        const DeviceTensor &global_sum,
        uint32_t layer,
        uint32_t n_layers,
        uint32_t n_tokens,
        uint32_t q_layer_stride,
        uint32_t tile_blocks,
        uint32_t global_block_base,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        float scale) {
        (void)score; (void)q_multi; (void)prototype_tile;
        (void)block_offsets; (void)block_counts;
        (void)global_max; (void)global_sum;
        (void)layer; (void)n_layers; (void)n_tokens;
        (void)q_layer_stride; (void)tile_blocks;
        (void)global_block_base; (void)n_heads; (void)n_kv_heads;
        (void)head_dim; (void)scale;
        return {
            false,
            "block_attn_stream_adaptive_score_device requires backend override"};
    }

    // Exact one-transfer/one-dot tiled Adaptive scorer. Each streamed tile
    // writes per-(distribution, block) MaxSim logits and LSE pairs into a
    // bounded layer-local workspace. `finalize` reduces those pairs into the
    // exact all-prototype softmax denominator and accumulates the layer score.
    virtual DeviceStatus block_attn_stream_adaptive_block_stats_device(
        DeviceTensor &block_max,
        DeviceTensor &block_sum,
        const DeviceTensor &q_multi,
        const DeviceTensor &prototype_tile,
        const DeviceTensor &block_offsets,
        const DeviceTensor &block_counts,
        uint32_t layer,
        uint32_t n_layers,
        uint32_t n_tokens,
        uint32_t q_layer_stride,
        uint32_t prototype_count,
        uint32_t tile_blocks,
        uint32_t workspace_block_base,
        uint32_t workspace_block_stride,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        float scale) {
        (void)block_max; (void)block_sum; (void)q_multi;
        (void)prototype_tile; (void)block_offsets; (void)block_counts;
        (void)layer; (void)n_layers; (void)n_tokens;
        (void)q_layer_stride; (void)prototype_count; (void)tile_blocks;
        (void)workspace_block_base; (void)workspace_block_stride;
        (void)n_heads; (void)n_kv_heads; (void)head_dim; (void)scale;
        return {
            false,
            "block_attn_stream_adaptive_block_stats_device requires backend override"};
    }

    // Tensor-core implementation of the same exact block-statistics update.
    // `logits` is bounded scratch reused for every KV head and streamed tile;
    // backends that cannot run FP16 GEMM return unavailable and callers retain
    // the scalar one-dot implementation above.
    virtual DeviceStatus block_attn_stream_adaptive_block_stats_gemm_device(
        DeviceTensor &block_max,
        DeviceTensor &block_sum,
        DeviceTensor &logits,
        const DeviceTensor &q_multi,
        const DeviceTensor &prototype_tile,
        const DeviceTensor &block_offsets,
        const DeviceTensor &block_counts,
        uint32_t layer,
        uint32_t n_layers,
        uint32_t n_tokens,
        uint32_t q_layer_stride,
        uint32_t prototype_count,
        uint32_t tile_blocks,
        uint32_t workspace_block_base,
        uint32_t workspace_block_stride,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        float scale) {
        (void)block_max; (void)block_sum; (void)logits; (void)q_multi;
        (void)prototype_tile; (void)block_offsets; (void)block_counts;
        (void)layer; (void)n_layers; (void)n_tokens;
        (void)q_layer_stride; (void)prototype_count; (void)tile_blocks;
        (void)workspace_block_base; (void)workspace_block_stride;
        (void)n_heads; (void)n_kv_heads; (void)head_dim; (void)scale;
        return {
            false,
            "block_attn_stream_adaptive_block_stats_gemm_device requires backend override"};
    }

    // Repack the naturally captured [row, head, dim] Adaptive tensors into
    // head-major matrices.  This makes the GEMM K dimension and successive
    // columns contiguous instead of gathering one small head slice from every
    // wide interleaved row.
    virtual DeviceStatus adaptive_pack_query_head_major_device(
        DeviceTensor &dst, const DeviceTensor &src,
        uint32_t layer, uint32_t n_tokens, uint32_t q_layer_stride,
        uint32_t n_heads, uint32_t head_dim) {
        (void)dst; (void)src; (void)layer; (void)n_tokens;
        (void)q_layer_stride; (void)n_heads; (void)head_dim;
        return {false,
                "adaptive_pack_query_head_major_device requires backend override"};
    }

    virtual DeviceStatus adaptive_pack_prototypes_head_major_device(
        DeviceTensor &dst, const DeviceTensor &src,
        uint32_t prototype_begin, uint32_t prototype_count,
        uint32_t n_kv_heads, uint32_t head_dim) {
        (void)dst; (void)src; (void)prototype_begin;
        (void)prototype_count; (void)n_kv_heads; (void)head_dim;
        return {false,
                "adaptive_pack_prototypes_head_major_device requires backend override"};
    }

    virtual DeviceStatus
    block_attn_stream_adaptive_block_stats_gemm_head_major_device(
        DeviceTensor &block_max, DeviceTensor &block_sum,
        DeviceTensor &logits, const DeviceTensor &query_head_major,
        const DeviceTensor &prototype_head_major,
        const DeviceTensor &block_offsets,
        const DeviceTensor &block_counts,
        uint32_t n_tokens, uint32_t prototype_count,
        uint32_t tile_blocks, uint32_t workspace_block_base,
        uint32_t workspace_block_stride, uint32_t n_heads,
        uint32_t n_kv_heads, uint32_t head_dim, float scale) {
        (void)block_max; (void)block_sum; (void)logits;
        (void)query_head_major; (void)prototype_head_major;
        (void)block_offsets; (void)block_counts; (void)n_tokens;
        (void)prototype_count; (void)tile_blocks;
        (void)workspace_block_base; (void)workspace_block_stride;
        (void)n_heads; (void)n_kv_heads; (void)head_dim; (void)scale;
        return {false,
                "adaptive head-major GEMM requires backend override"};
    }

    virtual DeviceStatus block_attn_stream_adaptive_finalize_device(
        DeviceTensor &score,
        const DeviceTensor &block_max,
        const DeviceTensor &block_sum,
        DeviceTensor &global_max,
        DeviceTensor &global_sum,
        uint32_t layer,
        uint32_t n_layers,
        uint32_t n_tokens,
        uint32_t workspace_block_stride,
        uint32_t block_count,
        uint32_t global_block_base,
        uint32_t n_heads) {
        (void)score; (void)block_max; (void)block_sum;
        (void)global_max; (void)global_sum; (void)layer;
        (void)n_layers; (void)n_tokens; (void)workspace_block_stride;
        (void)block_count; (void)global_block_base; (void)n_heads;
        return {
            false,
            "block_attn_stream_adaptive_finalize_device requires backend override"};
    }

    // One-transfer/one-dot Adaptive scorer for a complete normal-attention
    // layer. `prototype_layer` and the block metadata use layer-local offsets;
    // the result is accumulated into global block scores with normalization by
    // the original all-layer/head distribution count.
    virtual DeviceStatus block_attn_score_adaptive_layer_device(
        DeviceTensor &score,
        const DeviceTensor &q_multi,
        const DeviceTensor &prototype_layer,
        const DeviceTensor &block_offsets,
        const DeviceTensor &block_counts,
        uint32_t layer,
        uint32_t n_layers,
        uint32_t n_tokens,
        uint32_t q_layer_stride,
        uint32_t prototype_row_offset,
        uint32_t prototype_count,
        uint32_t block_count,
        uint32_t global_block_base,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        float scale) {
        (void)score; (void)q_multi; (void)prototype_layer;
        (void)block_offsets; (void)block_counts;
        (void)layer; (void)n_layers; (void)n_tokens;
        (void)q_layer_stride; (void)prototype_row_offset;
        (void)prototype_count;
        (void)block_count; (void)global_block_base;
        (void)n_heads; (void)n_kv_heads; (void)head_dim;
        (void)scale;
        return {
            false,
            "block_attn_score_adaptive_layer_device requires backend override"};
    }


    // Variable-length logical-group variant used by semantic retrieval.
    // Fine-grained logits and the global softmax are identical to
    // block_attn_score_softmax_pages_device. For every query
    // (layer,token,head), however, the device either takes a MaxSim or sums the
    // normalized attention mass over each group's half-open flattened sub-block
    // range [group_begin[g],group_end[g]). Length normalization is applied by
    // the executor after this raw group reduction.
    virtual DeviceStatus block_attn_score_softmax_groups_device(
        DeviceTensor &score,
        const DeviceTensor &q_multi,
        const DeviceTensor &kbar_multi,
        const DeviceTensor &group_begin,
        const DeviceTensor &group_end,
        uint32_t n_groups,
        uint32_t n_layers,
        uint32_t n_tokens,
        uint32_t q_layer_stride,
        uint32_t n_blocks,
        uint32_t kbar_layer_stride,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        float scale,
        uint32_t excl_lo_end = 0,
        uint32_t excl_hi_begin = UINT32_MAX,
        uint32_t n_subblocks = 1,
        uint32_t group_reduce_mass = 0,
        uint32_t accumulate = 0,
        uint64_t q_elem_off = 0,
        uint64_t kbar_elem_off = 0) {
        (void)score; (void)q_multi; (void)kbar_multi;
        (void)group_begin; (void)group_end; (void)n_groups;
        (void)n_layers; (void)n_tokens; (void)q_layer_stride;
        (void)n_blocks; (void)kbar_layer_stride; (void)n_heads;
        (void)n_kv_heads; (void)head_dim; (void)scale;
        (void)excl_lo_end; (void)excl_hi_begin; (void)n_subblocks;
        (void)group_reduce_mass;
        (void)accumulate; (void)q_elem_off; (void)kbar_elem_off;
        return {false,
                "block_attn_score_softmax_groups_device requires backend override"};
    }

    // Exact CPU-streamed mean-K scoring primitives. The full index remains in
    // host memory; callers copy one [L,tile_blocks,n_subblocks,Hkv,D] tile into
    // kbar_tile and invoke these methods in two passes. Pass 1 online-merges
    // per-distribution log-sum-exp state across tiles. Pass 2 recomputes the
    // dots and writes normalized block scores at global_block_base.
    virtual DeviceStatus block_attn_stream_lse_device(
        DeviceTensor &global_max,
        DeviceTensor &global_sum,
        const DeviceTensor &q_multi,
        const DeviceTensor &kbar_tile,
        uint32_t n_layers,
        uint32_t n_tokens,
        uint32_t q_layer_stride,
        uint32_t tile_blocks,
        uint32_t kbar_layer_stride,
        uint32_t global_block_base,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        float scale,
        uint32_t excl_lo_end,
        uint32_t excl_hi_begin,
        uint32_t n_subblocks,
        uint32_t initialize) {
        (void)global_max; (void)global_sum; (void)q_multi; (void)kbar_tile;
        (void)n_layers; (void)n_tokens; (void)q_layer_stride;
        (void)tile_blocks; (void)kbar_layer_stride;
        (void)global_block_base; (void)n_heads; (void)n_kv_heads;
        (void)head_dim; (void)scale; (void)excl_lo_end;
        (void)excl_hi_begin; (void)n_subblocks; (void)initialize;
        return {false,
                "block_attn_stream_lse_device requires backend override"};
    }

    virtual DeviceStatus block_attn_stream_score_device(
        DeviceTensor &score,
        const DeviceTensor &q_multi,
        const DeviceTensor &kbar_tile,
        const DeviceTensor &global_max,
        const DeviceTensor &global_sum,
        uint32_t n_layers,
        uint32_t n_tokens,
        uint32_t q_layer_stride,
        uint32_t tile_blocks,
        uint32_t kbar_layer_stride,
        uint32_t global_block_base,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        float scale,
        uint32_t excl_lo_end,
        uint32_t excl_hi_begin,
        uint32_t n_subblocks,
        uint32_t reduce_max,
        uint32_t accumulate) {
        (void)score; (void)q_multi; (void)kbar_tile;
        (void)global_max; (void)global_sum; (void)n_layers;
        (void)n_tokens; (void)q_layer_stride; (void)tile_blocks;
        (void)kbar_layer_stride; (void)global_block_base;
        (void)n_heads; (void)n_kv_heads; (void)head_dim;
        (void)scale; (void)excl_lo_end; (void)excl_hi_begin;
        (void)n_subblocks; (void)reduce_max; (void)accumulate;
        return {false,
                "block_attn_stream_score_device requires backend override"};
    }

    // Semantic-group pass-2 counterpart. group_dist is
    // [n_groups, n_layers*n_tokens*n_heads] fp32 and is zeroed by the caller
    // before the first tile. Each tile updates either the probability sum or
    // MaxSim value for every (group,distribution); finalize reduces the
    // distribution dimension into the ordinary per-group score vector.
    virtual DeviceStatus block_attn_stream_group_update_device(
        DeviceTensor &group_dist,
        const DeviceTensor &q_multi,
        const DeviceTensor &kbar_tile,
        const DeviceTensor &global_max,
        const DeviceTensor &global_sum,
        const DeviceTensor &group_begin,
        const DeviceTensor &group_end,
        uint32_t n_groups,
        uint32_t n_layers,
        uint32_t n_tokens,
        uint32_t q_layer_stride,
        uint32_t tile_blocks,
        uint32_t kbar_layer_stride,
        uint32_t global_block_base,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        float scale,
        uint32_t excl_lo_end,
        uint32_t excl_hi_begin,
        uint32_t n_subblocks,
        uint32_t group_reduce_mass) {
        (void)group_dist; (void)q_multi; (void)kbar_tile;
        (void)global_max; (void)global_sum;
        (void)group_begin; (void)group_end; (void)n_groups;
        (void)n_layers; (void)n_tokens; (void)q_layer_stride;
        (void)tile_blocks; (void)kbar_layer_stride;
        (void)global_block_base; (void)n_heads; (void)n_kv_heads;
        (void)head_dim; (void)scale; (void)excl_lo_end;
        (void)excl_hi_begin; (void)n_subblocks;
        (void)group_reduce_mass;
        return {false,
                "block_attn_stream_group_update_device requires backend override"};
    }

    virtual DeviceStatus block_attn_stream_group_finalize_device(
        DeviceTensor &score,
        const DeviceTensor &group_dist,
        uint32_t n_groups,
        uint32_t n_distributions,
        float distribution_weight,
        uint32_t accumulate) {
        (void)score; (void)group_dist; (void)n_groups;
        (void)n_distributions; (void)distribution_weight;
        (void)accumulate;
        return {false,
                "block_attn_stream_group_finalize_device requires backend override"};
    }

    // deltanet_pack_query_device: pack one DeltaNet layer's in-span query rows
    // (kvmem retrieval_method==deltanet). The conv output `conv_out` holds the
    // L2-normalized DeltaNet Q in the first num_k_heads*head_k_dim of each row
    // (row stride conv_stride). Copies `cnt` rows starting at chunk-local row r0
    // into q_dst[dst_row_base + r][num_k_heads][head_k_dim]. No RoPE (DeltaNet
    // queries are not rotated; the read-out dots them against the state directly).
    virtual DeviceStatus deltanet_pack_query_device(DeviceTensor &q_dst,
                                                    const DeviceTensor &conv_out,
                                                    uint64_t q_dst_off,
                                                    uint32_t conv_stride,
                                                    uint32_t r0,
                                                    uint32_t dst_row_base,
                                                    uint32_t cnt,
                                                    uint32_t num_k_heads,
                                                    uint32_t head_k_dim) {
        (void)q_dst; (void)conv_out; (void)q_dst_off; (void)conv_stride; (void)r0;
        (void)dst_row_base; (void)cnt; (void)num_k_heads; (void)head_k_dim;
        return {false, "deltanet_pack_query_device requires backend override"};
    }

    // deltanet_block_score_device: DeltaNet-retrieval per-layer block scorer
    // (kvmem retrieval_method==deltanet; see
    // docs/kvmem_deltanet_retrieval_design.md §5-8). For one
    // DeltaNet layer, computes r[blk,vh] = TopKMean_t( d[blk,vh] *
    // ||(S_j - a_j S_{j-1})^T q_t||_2 ) into r_out[blk,vh]. dn_snap holds S_j per
    // block ([blocks, v_heads, d_v, d_k]); decaysum holds each block's in-block
    // log-decay (a_j = exp(decaysum)); decay_d holds the precomputed post-block
    // decay exp(G_M-G_j) (or all-ones when decay is off); q_layer is the layer's
    // L2-normalized DeltaNet queries [M, num_k_heads, d_k]. All pointers are
    // pre-offset to this layer's slice by the caller (via *_off element offsets).
    virtual DeviceStatus deltanet_block_score_device(DeviceTensor &r_out,
                                                     const DeviceTensor &dn_snap,
                                                     const DeviceTensor &decaysum,
                                                     const DeviceTensor &decay_d,
                                                     const DeviceTensor &q_layer,
                                                     uint64_t r_off,
                                                     uint64_t snap_off,
                                                     uint64_t decay_off,
                                                     uint64_t decay_d_off,
                                                     uint64_t q_off,
                                                     uint32_t n_blocks,
                                                     uint32_t num_k_heads,
                                                     uint32_t num_v_heads,
                                                     uint32_t head_v_dim,
                                                     uint32_t head_k_dim,
                                                     uint32_t M,
                                                     uint32_t topk_q) {
        (void)r_out; (void)dn_snap; (void)decaysum; (void)decay_d; (void)q_layer;
        (void)r_off; (void)snap_off; (void)decay_off; (void)decay_d_off; (void)q_off;
        (void)n_blocks; (void)num_k_heads; (void)num_v_heads; (void)head_v_dim;
        (void)head_k_dim; (void)M; (void)topk_q;
        return {false, "deltanet_block_score_device requires backend override"};
    }

    // derope_query_device: de-RoPE the current decode token's RoPE-baked Q (baked
    // at query_pos, layout [n_heads, q_stride] with attn-Q the first head_dim of
    // each unit) into a contiguous content-frame query [n_heads, head_dim], so it
    // can be dotted against the content-frame mean keys above. Same de-rotate math
    // (SAME __sincosf as the bake). Called once per retrieval interval.
    virtual DeviceStatus derope_query_device(DeviceTensor &q_content,
                                             const DeviceTensor &q,
                                             uint32_t q_stride,
                                             uint32_t n_heads,
                                             uint32_t head_dim,
                                             uint32_t rope_dim,
                                             int32_t query_pos,
                                             float theta) {
        (void)q_content; (void)q; (void)q_stride; (void)n_heads; (void)head_dim;
        (void)rope_dim; (void)query_pos; (void)theta;
        return {false, "derope_query_device requires backend override"};
    }

    // derope_query_multi_device: batched de-RoPE of `cnt` contiguous query tokens
    // (the in-span question tokens captured during prefill) into a packed
    // content-frame buffer [cnt, n_heads, head_dim] at out_elem_offset. Input `q`
    // is read from q_elem_offset; token r at q_token_stride*r, head qh at
    // +q_head_stride (attn-Q the first head_dim of each unit), baked at start_pos+r.
    // Same de-rotate math (SAME __sincosf) as derope_query_device.
    virtual DeviceStatus derope_query_multi_device(DeviceTensor &q_multi,
                                                   const DeviceTensor &q,
                                                   uint64_t q_elem_offset,
                                                   uint64_t out_elem_offset,
                                                   uint32_t q_token_stride,
                                                   uint32_t q_head_stride,
                                                   uint32_t cnt,
                                                   uint32_t n_heads,
                                                   uint32_t head_dim,
                                                   uint32_t rope_dim,
                                                   int32_t start_pos,
                                                   float theta) {
        (void)q_multi; (void)q; (void)q_elem_offset; (void)out_elem_offset;
        (void)q_token_stride; (void)q_head_stride; (void)cnt; (void)n_heads;
        (void)head_dim; (void)rope_dim; (void)start_pos; (void)theta;
        return {false, "derope_query_multi_device requires backend override"};
    }

    // Cross-request paged KV append. src layout is [batch, src_stride], each
    // row b writes to logical_positions[b] through the page-table slice
    // page_indices[page_indptr[b]..page_indptr[b+1]).
    virtual DeviceStatus kv_append_batch_paged_ragged_device(
                                                      DeviceTensor &cache,
                                                      const DeviceTensor &src,
                                                      const DeviceTensor &logical_positions,
                                                      uint32_t per_pos_size,
                                                      uint32_t batch,
                                                      uint32_t src_stride,
                                                      const DeviceTensor &page_indices,
                                                      const DeviceTensor &page_indptr,
                                                      uint32_t page_size) {
        (void)cache; (void)src; (void)logical_positions; (void)per_pos_size;
        (void)batch; (void)src_stride; (void)page_indices; (void)page_indptr;
        (void)page_size;
        return {false, "kv_append_batch_paged_ragged_device requires backend override"};
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

    // Paged-KV prefill for one logical sequence. Unlike the decode-shaped
    // paged entry above, this treats `batch` as a contiguous query chunk and
    // uses a causal prefill kernel over q_len=batch, kv_len=base_seq_len+batch.
    virtual DeviceStatus attention_prefill_batch_paged_gated_device(
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
        return {false, "attention_prefill_batch_paged_gated_device requires backend override"};
    }

    // Cross-request paged-KV prefill. q/out rows are concatenated across
    // requests, described by q_indptr[batch+1]. page_indices are flattened
    // page table slices described by page_indptr[batch+1], and last_page_len
    // stores each request's final page occupancy. The host indptr mirrors are
    // required by FlashInfer's prefill planner.
    virtual DeviceStatus attention_prefill_batch_paged_ragged_gated_device(
                                                      DeviceTensor &out,
                                                      const DeviceTensor &q,
                                                      uint32_t q_stride,
                                                      const DeviceTensor &k_cache,
                                                      const DeviceTensor &v_cache,
                                                      const DeviceTensor &page_indices,
                                                      const DeviceTensor &page_indptr,
                                                      const DeviceTensor &last_page_len,
                                                      const DeviceTensor &q_indptr,
                                                      const int32_t *q_indptr_host,
                                                      const int32_t *page_indptr_host,
                                                      uint32_t batch,
                                                      uint32_t total_q,
                                                      uint32_t page_size,
                                                      uint32_t n_heads,
                                                      uint32_t n_kv_heads,
                                                      uint32_t head_dim,
                                                      uint32_t q_batch_stride,
                                                      uint32_t out_batch_stride,
                                                      float scale) {
        (void)out; (void)q; (void)q_stride; (void)k_cache; (void)v_cache;
        (void)page_indices; (void)page_indptr; (void)last_page_len;
        (void)q_indptr; (void)q_indptr_host; (void)page_indptr_host;
        (void)batch; (void)total_q; (void)page_size; (void)n_heads;
        (void)n_kv_heads; (void)head_dim; (void)q_batch_stride;
        (void)out_batch_stride; (void)scale;
        return {false, "attention_prefill_batch_paged_ragged_gated_device requires backend override"};
    }

    // Cross-request ragged paged decode attention. Each row b has its own page
    // table slice page_indices[page_indptr[b]..page_indptr[b+1]), last page
    // length, and sequence length. This is the continuous batching variant;
    // unlike attention_decode_batch_paged_gated_device it does not assume one
    // shared page table or consecutive positions.
    virtual DeviceStatus attention_decode_batch_paged_gated_ragged_device(
                                                      DeviceTensor &out,
                                                      const DeviceTensor &q,
                                                      uint32_t q_stride,
                                                      const DeviceTensor &k_cache,
                                                      const DeviceTensor &v_cache,
                                                      const DeviceTensor &page_indices,
                                                      const DeviceTensor &page_indptr,
                                                      const DeviceTensor &last_page_len,
                                                      const DeviceTensor &seq_lens,
                                                      const int32_t *page_indptr_host,
                                                      const int32_t *last_page_len_host,
                                                      const int32_t *seq_lens_host,
                                                      uint32_t page_size,
                                                      uint32_t n_heads,
                                                      uint32_t n_kv_heads,
                                                      uint32_t head_dim,
                                                      uint32_t batch,
                                                      uint32_t q_batch_stride,
                                                      uint32_t out_batch_stride,
                                                      float scale) {
        (void)out; (void)q; (void)q_stride; (void)k_cache; (void)v_cache;
        (void)page_indices; (void)page_indptr; (void)last_page_len;
        (void)seq_lens; (void)page_indptr_host; (void)last_page_len_host;
        (void)seq_lens_host; (void)page_size; (void)n_heads; (void)n_kv_heads;
        (void)head_dim; (void)batch; (void)q_batch_stride;
        (void)out_batch_stride; (void)scale;
        return {false, "attention_decode_batch_paged_gated_ragged_device requires backend override"};
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

    // Raw byte copies for tiered KV storage. Offsets/counts are bytes, not
    // tensor elements, so this can move fp16/fp32/fp8/q8 KV cache payloads
    // without knowing the dtype at the call site.
    virtual DeviceStatus copy_bytes_to_host(const DeviceTensor &x,
                                            void *host,
                                            uint64_t byte_offset,
                                            uint64_t byte_count) {
        (void)x; (void)host; (void)byte_offset; (void)byte_count;
        return {false, "copy_bytes_to_host not implemented for this backend"};
    }

    // Async raw byte copies for tiered KV storage. The default implementation
    // preserves synchronous semantics so non-CUDA/mock backends do not need to
    // understand streams. CUDA overrides these with a dedicated copy stream;
    // callers must keep host buffers alive until wait_kv_transfer() returns.
    virtual DeviceStatus begin_kv_transfer_from_device() { return {}; }
    virtual DeviceStatus begin_kv_transfer_to_device() { return {}; }
    virtual DeviceStatus wait_kv_transfer() { return {}; }
    virtual DeviceStatus record_kv_transfer_fence(
            std::unique_ptr<DeviceTransferFence> &fence) {
        fence.reset();
        return wait_kv_transfer();
    }
    // Cross-stream ordering used by raw-K materialization. The CUDA backend
    // records an event on exec_stream after scatter/RoPE and lets the copy
    // stream wait before reusing that staging slot. Conversely, exec_stream
    // waits on the H2D event without synchronizing the host.
    virtual DeviceStatus record_execution_fence(
            std::unique_ptr<DeviceTransferFence> &fence) {
        fence.reset();
        return synchronize();
    }
    virtual DeviceStatus execution_wait_for_kv_transfer(
            const std::unique_ptr<DeviceTransferFence> &fence) {
        (void)fence;
        return {};
    }
    virtual DeviceStatus kv_transfer_wait_for_execution(
            std::unique_ptr<DeviceTransferFence> &fence) {
        fence.reset();
        return {};
    }
    virtual DeviceStatus wait_kv_transfer_fence(
            std::unique_ptr<DeviceTransferFence> &fence) {
        fence.reset();
        return {};
    }
    virtual DeviceStatus copy_bytes_to_host_async(const DeviceTensor &x,
                                                  void *host,
                                                  uint64_t byte_offset,
                                                  uint64_t byte_count) {
        return copy_bytes_to_host(x, host, byte_offset, byte_count);
    }

    virtual DeviceStatus copy_bytes_from_host(DeviceTensor &x,
                                              uint64_t byte_offset,
                                              const void *host,
                                              uint64_t byte_count) {
        (void)x; (void)byte_offset; (void)host; (void)byte_count;
        return {false, "copy_bytes_from_host not implemented for this backend"};
    }

    virtual DeviceStatus copy_bytes_from_host_async(DeviceTensor &x,
                                                    uint64_t byte_offset,
                                                    const void *host,
                                                    uint64_t byte_count) {
        return copy_bytes_from_host(x, byte_offset, host, byte_count);
    }

    // CPU-tier stage-in fast path. `host` contains a packed slab of KV pages.
    // One H2D moves the slab into a reusable backend staging allocation, then
    // one scatter kernel per target tensor writes its pages to arbitrary
    // physical cache slots. src/dst_page_indices are target-major arrays with
    // target_count * pages_per_target int32 entries and must remain alive until
    // the following KV-transfer fence completes.
    virtual DeviceStatus copy_packed_pages_from_host_async(
            const void *host, uint64_t host_bytes,
            const std::vector<DeviceTensor *> &targets,
            const int32_t *src_page_indices,
            const int32_t *dst_page_indices,
            uint32_t pages_per_target,
            uint64_t page_bytes) {
        (void)host; (void)host_bytes; (void)targets;
        (void)src_page_indices; (void)dst_page_indices;
        (void)pages_per_target; (void)page_bytes;
        return {false,
                "copy_packed_pages_from_host_async not implemented"};
    }

    // Symmetric CPU-tier stage-out path: gather arbitrary physical cache pages
    // into the backend's packed staging allocation on the KV copy stream, then
    // issue one contiguous D2H into `host`.
    virtual DeviceStatus copy_packed_pages_to_host_async(
            void *host, uint64_t host_bytes,
            const std::vector<DeviceTensor *> &targets,
            const int32_t *src_page_indices,
            const int32_t *dst_page_indices,
            uint32_t pages_per_target,
            uint64_t page_bytes) {
        (void)host; (void)host_bytes; (void)targets;
        (void)src_page_indices; (void)dst_page_indices;
        (void)pages_per_target; (void)page_bytes;
        return {false,
                "copy_packed_pages_to_host_async not implemented"};
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

// Size-keyed recycling pool for host buffers (used for kvmem's pinned CPU tier).
//
// kvmem's CPU tier needs a large cudaHostAlloc'd buffer, and pinned allocation
// runs at ~0.6 s/GB. On the continuous-batching path each request builds a fresh
// executor and configures kvmem, so allocating the tier buffer per request added
// (e.g.) ~5 s to the TTFT of every request with an 8 GiB CPU tier. This pool
// recycles freed buffers keyed by exact byte size: sequential requests reuse one
// buffer (the alloc is paid once), while concurrent requests still receive
// distinct buffers (the pool grows on demand) so no two live executors ever share
// backing bytes. acquire() transfers ownership out; release() takes it back.
// Recycled bytes are stale, which is safe because each executor's tier slot
// bookkeeping starts empty and only reads slots it has itself written. Pooled
// buffers outlive the per-request executors and are freed when the pool is
// destroyed, which must happen while the owning backend (CUDA context) is alive.
class HostTierBufferPool {
public:
    explicit HostTierBufferPool(DeviceBackend &backend) : backend_(backend) {}

    std::unique_ptr<HostBuffer> acquire(uint64_t bytes, const char *label) {
        if (bytes == 0) return nullptr;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = free_.find(bytes);
            if (it != free_.end() && !it->second.empty()) {
                std::unique_ptr<HostBuffer> buf = std::move(it->second.back());
                it->second.pop_back();
                return buf;
            }
        }
        return backend_.host_buffer(bytes, label);
    }

    void release(std::unique_ptr<HostBuffer> buf) {
        if (!buf) return;
        const uint64_t bytes = buf->bytes;
        std::lock_guard<std::mutex> lk(mu_);
        free_[bytes].push_back(std::move(buf));
    }

private:
    DeviceBackend &backend_;
    std::mutex mu_;
    std::unordered_map<uint64_t, std::vector<std::unique_ptr<HostBuffer>>> free_;
};

bool cuda_device_backend_available();
std::unique_ptr<DeviceBackend> make_cuda_device_backend(LinearBackend linear_backend);

} // namespace qw3
