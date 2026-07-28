#include "qw3/device_backend.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kHidden = 5120;
constexpr uint32_t kIntermediate = 17408;
constexpr uint32_t kRecurrentQkv = 10240;
constexpr uint32_t kRecurrentValue = 6144;
constexpr uint32_t kRecurrentSmall = 48;
constexpr uint32_t kAttentionQ = 12288;
constexpr uint32_t kAttentionKv = 1024;

void require(const qw3::DeviceStatus &status, const char *operation) {
    if (!status.ok) {
        std::fprintf(stderr, "%s failed: %s\n", operation, status.message);
        std::exit(1);
    }
}

struct Weights {
    std::unique_ptr<qw3::DeviceWeight> gate;
    std::unique_ptr<qw3::DeviceWeight> up;
    std::unique_ptr<qw3::DeviceWeight> down;
};

uint32_t xorshift32(uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

std::unique_ptr<qw3::DeviceWeight> make_fp8_weight(
        qw3::DeviceBackend &backend,
        uint32_t rows,
        uint32_t cols,
        uint32_t seed,
        const char *label) {
    const uint64_t elements = static_cast<uint64_t>(rows) * cols;
    std::vector<uint8_t> packed(elements);
    for (uint8_t &value : packed) {
        const float sample =
            static_cast<float>(
                static_cast<int>(xorshift32(seed) % 31U) - 15) /
            32.0f;
        value = __nv_fp8_e4m3(sample).__x;
    }
    std::vector<float> scales(rows);
    for (uint32_t row = 0; row < rows; ++row) {
        scales[row] =
            0.5f + static_cast<float>(row % 17U) / 32.0f;
    }
    return backend.weight_fp8_e4m3(
        packed.data(), scales.data(), scales.size(),
        rows, cols, label);
}

std::unique_ptr<qw3::DeviceWeight> make_q8_weight(
        qw3::DeviceBackend &backend,
        uint32_t rows,
        uint32_t cols,
        uint32_t seed,
        const char *label) {
    const uint64_t blocks =
        static_cast<uint64_t>(rows) * cols / 32;
    std::vector<uint8_t> packed(blocks * 34);
    for (uint64_t block = 0; block < blocks; ++block) {
        uint8_t *dst = packed.data() + block * 34;
        const __half scale = __float2half(
            (0.5f + static_cast<float>(xorshift32(seed) & 31U) / 32.0f) /
            64.0f);
        std::memcpy(dst, &scale, sizeof(scale));
        for (uint32_t i = 0; i < 32; ++i) {
            const int value =
                static_cast<int>(xorshift32(seed) % 63U) - 31;
            dst[2 + i] =
                static_cast<uint8_t>(static_cast<int8_t>(value));
        }
    }
    return backend.weight_q8_0(packed.data(), rows, cols, label);
}

void initialize_tensor(qw3::DeviceBackend &backend,
                       qw3::DeviceTensor &tensor,
                       uint32_t count,
                       uint32_t period,
                       float divisor,
                       const char *operation) {
    std::vector<float> values(count);
    for (uint32_t i = 0; i < count; ++i) {
        values[i] =
            static_cast<float>(
                static_cast<int>(i % period) -
                static_cast<int>(period / 2)) /
            divisor;
    }
    require(backend.copy_bytes_from_host(
                tensor, 0, values.data(),
                values.size() * sizeof(float)),
            operation);
}

uint64_t tensor_hash(qw3::DeviceBackend &backend,
                     const qw3::DeviceTensor &tensor,
                     uint32_t count) {
    std::vector<float> values(count);
    require(backend.copy_to_host(
                tensor, values.data(), 0, values.size()),
            "copy benchmark hash output");
    uint64_t hash = 1469598103934665603ULL;
    for (float value : values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (uint32_t byte = 0; byte < sizeof(bits); ++byte) {
            hash ^= static_cast<uint8_t>(bits >> (byte * 8));
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

Weights make_nvfp4_weights(qw3::DeviceBackend &backend) {
    const uint64_t elements =
        static_cast<uint64_t>(kHidden) * kIntermediate;
    std::vector<uint8_t> packed(elements / 2);
    uint32_t random_state = 0x4f3a2b1cU;
    for (uint8_t &value : packed) {
        value = static_cast<uint8_t>(xorshift32(random_state));
    }
    uint8_t scale_values[16];
    for (uint32_t i = 0; i < 16; ++i) {
        scale_values[i] =
            __nv_fp8_e4m3(0.5f + static_cast<float>(i) / 16.0f).__x;
    }
    std::vector<uint8_t> scales(elements / 16);
    for (uint8_t &value : scales) {
        value = scale_values[xorshift32(random_state) & 15U];
    }

    Weights result;
    result.gate = backend.weight_nvfp4_e2m1(
        packed.data(), scales.data(), kIntermediate, kHidden / 16,
        1.0f, 1.0f, kIntermediate, kHidden, "bench.gate");
    result.up = backend.weight_nvfp4_e2m1(
        packed.data(), scales.data(), kIntermediate, kHidden / 16,
        1.0f, 1.0f, kIntermediate, kHidden, "bench.up");
    require(backend.prepare_nvfp4_fanout_pair(
                *result.gate, *result.up),
            "prepare NVFP4 FFN gate/up pair");
    result.down = backend.weight_nvfp4_e2m1(
        packed.data(), scales.data(), kHidden, kIntermediate / 16,
        1.0f, 1.0f, kHidden, kIntermediate, "bench.down");
    return result;
}

Weights make_q8_weights(qw3::DeviceBackend &backend) {
    const uint64_t elements =
        static_cast<uint64_t>(kHidden) * kIntermediate;
    const uint64_t blocks = elements / 32;
    std::vector<uint8_t> packed(blocks * 34);
    uint32_t random_state = 0x7c6d5e4fU;
    for (uint64_t block = 0; block < blocks; ++block) {
        uint8_t *dst = packed.data() + block * 34;
        const __half scale = __float2half(
            (0.5f + static_cast<float>(xorshift32(random_state) & 31U) / 32.0f) /
            64.0f);
        std::memcpy(dst, &scale, sizeof(scale));
        for (uint32_t i = 0; i < 32; ++i) {
            const int value =
                static_cast<int>(xorshift32(random_state) % 63U) - 31;
            dst[2 + i] = static_cast<uint8_t>(static_cast<int8_t>(value));
        }
    }

    Weights result;
    result.gate = backend.weight_q8_0(
        packed.data(), kIntermediate, kHidden, "bench.gate");
    result.up = backend.weight_q8_0(
        packed.data(), kIntermediate, kHidden, "bench.up");
    result.down = backend.weight_q8_0(
        packed.data(), kHidden, kIntermediate, "bench.down");
    return result;
}

Weights make_fp8_weights(qw3::DeviceBackend &backend) {
    const uint64_t elements =
        static_cast<uint64_t>(kHidden) * kIntermediate;
    std::vector<uint8_t> packed(elements);
    uint32_t random_state = 0x29d83a71U;
    for (uint8_t &value : packed) {
        const float sample =
            static_cast<float>(static_cast<int>(xorshift32(random_state) % 31U) - 15) /
            32.0f;
        value = __nv_fp8_e4m3(sample).__x;
    }
    std::vector<float> intermediate_scales(kIntermediate);
    std::vector<float> hidden_scales(kHidden);
    for (uint32_t i = 0; i < kIntermediate; ++i) {
        intermediate_scales[i] =
            0.5f + static_cast<float>(i % 17U) / 32.0f;
    }
    for (uint32_t i = 0; i < kHidden; ++i) {
        hidden_scales[i] =
            0.5f + static_cast<float>(i % 13U) / 32.0f;
    }

    Weights result;
    result.gate = backend.weight_fp8_e4m3(
        packed.data(), intermediate_scales.data(), intermediate_scales.size(),
        kIntermediate, kHidden, "bench.gate");
    result.up = backend.weight_fp8_e4m3(
        packed.data(), intermediate_scales.data(), intermediate_scales.size(),
        kIntermediate, kHidden, "bench.up");
    require(backend.prepare_fp8_fanout_pair(
                *result.gate, *result.up),
            "prepare FP8 FFN gate/up pair");
    result.down = backend.weight_fp8_e4m3(
        packed.data(), hidden_scales.data(), hidden_scales.size(),
        kHidden, kIntermediate, "bench.down");
    return result;
}

struct Buffers {
    std::unique_ptr<qw3::DeviceTensor> input;
    std::unique_ptr<qw3::DeviceTensor> gate;
    std::unique_ptr<qw3::DeviceTensor> up;
    std::unique_ptr<qw3::DeviceTensor> intermediate;
    std::unique_ptr<qw3::DeviceTensor> residual;
    std::unique_ptr<qw3::DeviceTensor> down_tmp;
};

Buffers make_buffers(qw3::DeviceBackend &backend, uint32_t batch) {
    Buffers result;
    result.input = backend.tensor_f32(
        static_cast<uint64_t>(batch) * kHidden, "bench.input");
    result.gate = backend.tensor_f32(
        static_cast<uint64_t>(batch) * kIntermediate, "bench.gate");
    result.up = backend.tensor_f32(
        static_cast<uint64_t>(batch) * kIntermediate, "bench.up");
    result.intermediate =
        backend.tensor_f32(
            static_cast<uint64_t>(batch) * kIntermediate,
            "bench.intermediate");
    result.residual = backend.tensor_f32(
        static_cast<uint64_t>(batch) * kHidden, "bench.residual");
    result.down_tmp = backend.tensor_f32(
        static_cast<uint64_t>(batch) * kHidden, "bench.down_tmp");

    std::vector<float> input(static_cast<size_t>(batch) * kHidden);
    for (uint32_t item = 0; item < batch; ++item) {
        for (uint32_t i = 0; i < kHidden; ++i) {
            input[static_cast<size_t>(item) * kHidden + i] =
                static_cast<float>(static_cast<int>(i % 17) - 8) /
                (64.0f + static_cast<float>(item));
        }
    }
    std::vector<float> intermediate(
        static_cast<size_t>(batch) * kIntermediate);
    for (uint32_t item = 0; item < batch; ++item) {
        for (uint32_t i = 0; i < kIntermediate; ++i) {
            intermediate[
                static_cast<size_t>(item) * kIntermediate + i] =
                static_cast<float>(static_cast<int>(i % 13) - 6) /
                (128.0f + static_cast<float>(item));
        }
    }
    std::vector<float> residual(
        static_cast<size_t>(batch) * kHidden, 0.0f);
    require(backend.copy_bytes_from_host(
                *result.input, 0, input.data(), input.size() * sizeof(float)),
            "copy benchmark input");
    require(backend.copy_bytes_from_host(
                *result.intermediate, 0, intermediate.data(),
                intermediate.size() * sizeof(float)),
            "copy benchmark intermediate");
    require(backend.copy_bytes_from_host(
                *result.residual, 0, residual.data(),
                residual.size() * sizeof(float)),
            "copy benchmark residual");
    require(backend.synchronize(), "initialize benchmark buffers");
    return result;
}

void run_gate_up(qw3::DeviceBackend &backend,
                 const Weights &weights,
                 Buffers &buffers,
                 uint32_t batch) {
    require(backend.begin(), "begin gate/up");
    if (batch == 1) {
        auto fused = backend.q8_0_matvec_silu_mul(
            *buffers.intermediate, *weights.gate, *weights.up,
            *buffers.input);
        if (fused.ok) return;

        qw3::DeviceTensor *outputs[2] = {
            buffers.gate.get(), buffers.up.get()};
        const qw3::DeviceWeight *fanout[2] = {
            weights.gate.get(), weights.up.get()};
        require(backend.q8_0_matvec_fanout(
                    outputs, fanout, 2, *buffers.input),
                "gate/up fanout fallback");
        require(backend.silu_mul(
                    *buffers.intermediate, *buffers.gate, *buffers.up),
                "gate/up SwiGLU fallback");
        return;
    }
    auto fused = backend.q8_0_matmul_silu_mul(
        *buffers.intermediate, *weights.gate, *weights.up,
        *buffers.input, batch, kHidden, kIntermediate);
    if (fused.ok) return;
    qw3::DeviceTensor *outputs[2] = {
        buffers.gate.get(), buffers.up.get()};
    const qw3::DeviceWeight *fanout[2] = {
        weights.gate.get(), weights.up.get()};
    const uint32_t strides[2] = {kIntermediate, kIntermediate};
    require(backend.q8_0_matmul_fanout(
                outputs, fanout, strides, 2, *buffers.input,
                batch, kHidden),
            "gate/up batch fanout fallback");
    require(backend.silu_mul_n(
                *buffers.intermediate, *buffers.gate, *buffers.up,
                static_cast<uint64_t>(batch) * kIntermediate),
            "gate/up batch SwiGLU fallback");
}

void run_down(qw3::DeviceBackend &backend,
              const Weights &weights,
              Buffers &buffers,
              uint32_t batch) {
    require(backend.begin(), "begin down");
    if (batch > 1) {
        require(backend.q8_0_matmul_add(
                    *buffers.residual, *buffers.residual,
                    *buffers.down_tmp, *weights.down,
                    *buffers.intermediate, batch,
                    kIntermediate, kHidden),
                "down batch residual add");
        return;
    }
    auto fused = backend.q8_0_matvec_add(
        *buffers.residual, *weights.down, *buffers.intermediate);
    if (fused.ok) return;

    require(backend.q8_0_matvec(
                *buffers.down_tmp, *weights.down, *buffers.intermediate),
            "down matvec fallback");
    require(backend.add(
                *buffers.residual, *buffers.residual, *buffers.down_tmp),
            "down residual add fallback");
}

template <typename Operation>
double measure_ms(qw3::DeviceBackend &backend,
                  Operation operation,
                  uint32_t warmup,
                  uint32_t iterations) {
    for (uint32_t i = 0; i < warmup; ++i) operation();
    require(backend.synchronize(), "benchmark warmup");

    const auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < iterations; ++i) operation();
    require(backend.synchronize(), "benchmark timing");
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    return elapsed_ms / static_cast<double>(iterations);
}

const char *env_or_default(const char *name, const char *fallback) {
    const char *value = std::getenv(name);
    return value ? value : fallback;
}

struct RecurrentWeights {
    std::unique_ptr<qw3::DeviceWeight> qkv;
    std::unique_ptr<qw3::DeviceWeight> gate;
    std::unique_ptr<qw3::DeviceWeight> alpha;
    std::unique_ptr<qw3::DeviceWeight> beta;
};

RecurrentWeights make_recurrent_weights(qw3::DeviceBackend &backend,
                                         bool fp8_mixed) {
    RecurrentWeights result;
    if (!fp8_mixed) {
        result.qkv = make_q8_weight(
            backend, kRecurrentQkv, kHidden,
            0x31415926U, "bench.recurrent_qkv");
        result.gate = make_q8_weight(
            backend, kRecurrentValue, kHidden,
            0x27182818U, "bench.recurrent_gate");
        result.alpha = make_q8_weight(
            backend, kRecurrentSmall, kHidden,
            0x16180339U, "bench.recurrent_alpha");
        result.beta = make_q8_weight(
            backend, kRecurrentSmall, kHidden,
            0x14142135U, "bench.recurrent_beta");
        return result;
    }
    const uint64_t elements =
        static_cast<uint64_t>(kRecurrentQkv) * kHidden;
    std::vector<uint8_t> packed(elements);
    uint32_t random_state = 0x31415926U;
    for (uint8_t &value : packed) {
        const float sample =
            static_cast<float>(
                static_cast<int>(xorshift32(random_state) % 31U) - 15) /
            32.0f;
        value = __nv_fp8_e4m3(sample).__x;
    }
    std::vector<float> scales(kRecurrentQkv);
    for (uint32_t i = 0; i < kRecurrentQkv; ++i) {
        scales[i] = 0.5f + static_cast<float>(i % 17U) / 32.0f;
    }
    std::vector<__nv_bfloat16> small(
        static_cast<size_t>(kRecurrentSmall) * kHidden,
        __float2bfloat16(1.0f / 1024.0f));

    result.qkv = backend.weight_fp8_e4m3(
        packed.data(), scales.data(), kRecurrentQkv,
        kRecurrentQkv, kHidden, "bench.recurrent_qkv");
    result.gate = backend.weight_fp8_e4m3(
        packed.data(), scales.data(), kRecurrentValue,
        kRecurrentValue, kHidden, "bench.recurrent_gate");
    require(backend.prepare_fp8_fanout_pair(
                *result.qkv, *result.gate),
            "prepare FP8 recurrent QKV/gate pair");
    result.alpha = backend.weight_bf16(
        small.data(), kRecurrentSmall, kHidden, "bench.recurrent_alpha");
    result.beta = backend.weight_bf16(
        small.data(), kRecurrentSmall, kHidden, "bench.recurrent_beta");
    return result;
}

struct RecurrentBuffers {
    std::unique_ptr<qw3::DeviceTensor> input;
    std::unique_ptr<qw3::DeviceTensor> qkv;
    std::unique_ptr<qw3::DeviceTensor> gate;
    std::unique_ptr<qw3::DeviceTensor> alpha;
    std::unique_ptr<qw3::DeviceTensor> beta;
};

RecurrentBuffers make_recurrent_buffers(qw3::DeviceBackend &backend,
                                        uint32_t batch) {
    RecurrentBuffers result;
    result.input = backend.tensor_f32(
        static_cast<uint64_t>(batch) * kHidden, "bench.recurrent_input");
    result.qkv = backend.tensor_f32(
        static_cast<uint64_t>(batch) * kRecurrentQkv,
        "bench.recurrent_qkv_out");
    result.gate =
        backend.tensor_f32(
            static_cast<uint64_t>(batch) * kRecurrentValue,
            "bench.recurrent_gate_out");
    result.alpha =
        backend.tensor_f32(
            static_cast<uint64_t>(batch) * kRecurrentSmall,
            "bench.recurrent_alpha_out");
    result.beta =
        backend.tensor_f32(
            static_cast<uint64_t>(batch) * kRecurrentSmall,
            "bench.recurrent_beta_out");
    std::vector<float> input(static_cast<size_t>(batch) * kHidden);
    for (uint32_t item = 0; item < batch; ++item) {
        for (uint32_t i = 0; i < kHidden; ++i) {
            input[static_cast<size_t>(item) * kHidden + i] =
                static_cast<float>(static_cast<int>(i % 17U) - 8) /
                (64.0f + static_cast<float>(item));
        }
    }
    require(backend.copy_bytes_from_host(
                *result.input, 0, input.data(), input.size() * sizeof(float)),
            "copy recurrent benchmark input");
    require(backend.synchronize(), "initialize recurrent benchmark buffers");
    return result;
}

void run_recurrent_fanout(qw3::DeviceBackend &backend,
                          const RecurrentWeights &weights,
                          RecurrentBuffers &buffers,
                          uint32_t batch) {
    require(backend.begin(), "begin recurrent fanout");
    qw3::DeviceTensor *outputs[4] = {
        buffers.qkv.get(), buffers.gate.get(),
        buffers.alpha.get(), buffers.beta.get()};
    const qw3::DeviceWeight *fanout[4] = {
        weights.qkv.get(), weights.gate.get(),
        weights.alpha.get(), weights.beta.get()};
    if (batch == 1) {
        require(backend.q8_0_matvec_fanout(
                    outputs, fanout, 4, *buffers.input),
                "recurrent mixed fanout");
        return;
    }
    const uint32_t strides[4] = {
        kRecurrentQkv, kRecurrentValue,
        kRecurrentSmall, kRecurrentSmall};
    require(backend.q8_0_matmul_fanout(
                outputs, fanout, strides, 4, *buffers.input,
                batch, kHidden),
            "recurrent mixed batch fanout");
}

struct AttentionWeights {
    std::unique_ptr<qw3::DeviceWeight> q;
    std::unique_ptr<qw3::DeviceWeight> k;
    std::unique_ptr<qw3::DeviceWeight> v;
};

AttentionWeights make_attention_weights(qw3::DeviceBackend &backend,
                                        bool fp8) {
    AttentionWeights result;
    if (fp8) {
        result.q = make_fp8_weight(
            backend, kAttentionQ, kHidden, 0x12345678U, "bench.attn_q");
        result.k = make_fp8_weight(
            backend, kAttentionKv, kHidden, 0x23456789U, "bench.attn_k");
        result.v = make_fp8_weight(
            backend, kAttentionKv, kHidden, 0x3456789aU, "bench.attn_v");
        require(backend.prepare_fp8_fanout_pair(
                    *result.k, *result.v),
                "prepare FP8 attention K/V pair");
    } else {
        result.q = make_q8_weight(
            backend, kAttentionQ, kHidden, 0x12345678U, "bench.attn_q");
        result.k = make_q8_weight(
            backend, kAttentionKv, kHidden, 0x23456789U, "bench.attn_k");
        result.v = make_q8_weight(
            backend, kAttentionKv, kHidden, 0x3456789aU, "bench.attn_v");
    }
    return result;
}

struct AttentionBuffers {
    std::unique_ptr<qw3::DeviceTensor> input;
    std::unique_ptr<qw3::DeviceTensor> q;
    std::unique_ptr<qw3::DeviceTensor> k;
    std::unique_ptr<qw3::DeviceTensor> v;
};

AttentionBuffers make_attention_buffers(qw3::DeviceBackend &backend,
                                        uint32_t batch) {
    AttentionBuffers result;
    result.input = backend.tensor_f32(
        static_cast<uint64_t>(batch) * kHidden, "bench.attn_input");
    result.q = backend.tensor_f32(
        static_cast<uint64_t>(batch) * kAttentionQ,
        "bench.attn_q_out");
    result.k = backend.tensor_f32(
        static_cast<uint64_t>(batch) * kAttentionKv,
        "bench.attn_k_out");
    result.v = backend.tensor_f32(
        static_cast<uint64_t>(batch) * kAttentionKv,
        "bench.attn_v_out");
    initialize_tensor(
        backend, *result.input, batch * kHidden, 17, 64.0f,
        "copy attention benchmark input");
    require(backend.synchronize(), "initialize attention benchmark buffers");
    return result;
}

void run_attention_fanout(qw3::DeviceBackend &backend,
                          const AttentionWeights &weights,
                          AttentionBuffers &buffers,
                          uint32_t batch) {
    require(backend.begin(), "begin attention fanout");
    qw3::DeviceTensor *outputs[3] = {
        buffers.q.get(), buffers.k.get(), buffers.v.get()};
    const qw3::DeviceWeight *fanout[3] = {
        weights.q.get(), weights.k.get(), weights.v.get()};
    if (batch == 1) {
        require(backend.q8_0_matvec_fanout(
                    outputs, fanout, 3, *buffers.input),
                "attention fanout");
        return;
    }
    const uint32_t strides[3] = {
        kAttentionQ, kAttentionKv, kAttentionKv};
    require(backend.q8_0_matmul_fanout(
                outputs, fanout, strides, 3, *buffers.input,
                batch, kHidden),
            "attention batch fanout");
}

struct OutputProjection {
    std::unique_ptr<qw3::DeviceWeight> weight;
    std::unique_ptr<qw3::DeviceTensor> input;
    std::unique_ptr<qw3::DeviceTensor> residual;
    std::unique_ptr<qw3::DeviceTensor> temporary;
};

OutputProjection make_output_projection(qw3::DeviceBackend &backend,
                                        bool fp8) {
    OutputProjection result;
    if (fp8) {
        result.weight = make_fp8_weight(
            backend, kHidden, kRecurrentValue,
            0x456789abU, "bench.recurrent_output");
    } else {
        result.weight = make_q8_weight(
            backend, kHidden, kRecurrentValue,
            0x456789abU, "bench.recurrent_output");
    }
    result.input =
        backend.tensor_f32(kRecurrentValue, "bench.recurrent_core");
    result.residual =
        backend.tensor_f32(kHidden, "bench.recurrent_residual");
    result.temporary =
        backend.tensor_f32(kHidden, "bench.recurrent_output_tmp");
    initialize_tensor(
        backend, *result.input, kRecurrentValue, 17, 64.0f,
        "copy output benchmark input");
    initialize_tensor(
        backend, *result.residual, kHidden, 13, 128.0f,
        "copy output benchmark residual");
    require(backend.synchronize(), "initialize output benchmark buffers");
    return result;
}

void run_output_projection(qw3::DeviceBackend &backend,
                           OutputProjection &projection) {
    require(backend.begin(), "begin output projection");
    auto fused = backend.q8_0_matvec_add(
        *projection.residual, *projection.weight, *projection.input);
    if (fused.ok) return;
    require(backend.q8_0_matvec(
                *projection.temporary, *projection.weight,
                *projection.input),
            "output projection fallback");
    require(backend.add(
                *projection.residual, *projection.residual,
                *projection.temporary),
            "output residual fallback");
}

} // namespace

int main(int argc, char **argv) {
    const std::string format = argc > 1 ? argv[1] : "nvfp4";
    const uint32_t iterations =
        argc > 2 ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 100;
    const uint32_t warmup =
        argc > 3 ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 10;
    if ((format != "nvfp4" && format != "fp8" && format != "q8" &&
         format != "fp8-batch4" && format != "q8-batch4" &&
         format != "fp8-recurrent" && format != "q8-recurrent" &&
         format != "fp8-recurrent-batch4" &&
         format != "q8-recurrent-batch4" &&
         format != "fp8-attention" && format != "q8-attention" &&
         format != "fp8-attention-batch4" &&
         format != "q8-attention-batch4" &&
         format != "fp8-output" && format != "q8-output") ||
        iterations == 0) {
        std::fprintf(stderr,
                     "usage: %s [nvfp4|fp8|q8|fp8-batch4|q8-batch4|"
                     "q8-recurrent|fp8-recurrent-batch4|"
                     "q8-recurrent-batch4|fp8-attention|q8-attention|"
                     "fp8-attention-batch4|q8-attention-batch4|"
                     "fp8-output|q8-output] "
                     "[iterations] [warmup]\n",
                     argv[0]);
        return 2;
    }

    auto backend = qw3::make_cuda_device_backend(qw3::LinearBackend::Auto);
    require(backend->begin(), "backend begin");
    if (format == "fp8-recurrent" || format == "q8-recurrent" ||
        format == "fp8-recurrent-batch4" ||
        format == "q8-recurrent-batch4") {
        const bool fp8_mixed =
            format == "fp8-recurrent" ||
            format == "fp8-recurrent-batch4";
        const uint32_t batch =
            format == "fp8-recurrent-batch4" ||
            format == "q8-recurrent-batch4" ? 4U : 1U;
        RecurrentWeights weights =
            make_recurrent_weights(*backend, fp8_mixed);
        RecurrentBuffers buffers =
            make_recurrent_buffers(*backend, batch);
        auto measure_projection = [&](qw3::DeviceTensor &out,
                                      const qw3::DeviceWeight &weight,
                                      const char *label) {
            return measure_ms(
                *backend,
                [&]() {
                    require(backend->begin(), label);
                    if (batch == 1) {
                        require(backend->q8_0_matvec(
                                    out, weight, *buffers.input),
                                label);
                    } else {
                        require(backend->q8_0_matmul(
                                    out, weight, *buffers.input,
                                    batch, kHidden,
                                    static_cast<uint32_t>(weight.rows)),
                                label);
                    }
                },
                warmup, iterations);
        };
        const double qkv_ms = measure_projection(
            *buffers.qkv, *weights.qkv, "recurrent qkv projection");
        const double gate_ms = measure_projection(
            *buffers.gate, *weights.gate, "recurrent gate projection");
        const double alpha_ms = measure_projection(
            *buffers.alpha, *weights.alpha, "recurrent alpha projection");
        const double beta_ms = measure_projection(
            *buffers.beta, *weights.beta, "recurrent beta projection");
        const double fanout_ms = measure_ms(
            *backend,
            [&]() {
                run_recurrent_fanout(
                    *backend, weights, buffers, batch);
            },
            warmup, iterations);
        std::printf(
            "format=%s batch=%u hidden=%u outputs=%u,%u,%u,%u "
            "iterations=%u warmup=%u\n",
            format.c_str(), batch, kHidden,
            kRecurrentQkv, kRecurrentValue,
            kRecurrentSmall, kRecurrentSmall, iterations, warmup);
        if (fp8_mixed) {
            std::printf(
                "policy fp8_small_cublaslt=%s fp8_mixed_fanout=%s "
                "fp8_outer_scale=%s fp8_fanout_scale_fusion=%s "
                "fp8_packed_pair=%s bf16_small_pair_max_rows=%s\n",
                env_or_default("QW3_FP8_SMALL_CUBLASLT", "default"),
                env_or_default("QW3_FP8_MIXED_FANOUT", "default"),
                env_or_default("QW3_FP8_CUBLASLT_OUTER_SCALE", "default"),
                env_or_default("QW3_FP8_FANOUT_SCALE_FUSION", "default"),
                env_or_default("QW3_FP8_PACKED_PAIR", "default"),
                env_or_default("QW3_BF16_SMALL_PAIR_MAX_ROWS", "default"));
        }
        std::printf(
            "qkv_ms=%.6f gate_ms=%.6f alpha_ms=%.6f beta_ms=%.6f "
            "recurrent_fanout_ms=%.6f\n",
            qkv_ms, gate_ms, alpha_ms, beta_ms, fanout_ms);
        auto print_checksum = [&](const char *name,
                                  const qw3::DeviceTensor &tensor,
                                  uint32_t count) {
            std::vector<float> values(count);
            require(backend->copy_to_host(
                        tensor, values.data(), 0, values.size()),
                    "copy recurrent checksum output");
            uint64_t hash = 1469598103934665603ULL;
            double sum = 0.0;
            for (float value : values) {
                uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                for (uint32_t byte = 0; byte < sizeof(bits); ++byte) {
                    hash ^= static_cast<uint8_t>(bits >> (byte * 8));
                    hash *= 1099511628211ULL;
                }
                sum += value;
            }
            std::printf(
                "%s_hash=%016llx %s_sum=%.9e\n",
                name, static_cast<unsigned long long>(hash), name, sum);
        };
        print_checksum(
            "qkv", *buffers.qkv, batch * kRecurrentQkv);
        print_checksum(
            "gate", *buffers.gate, batch * kRecurrentValue);
        print_checksum(
            "alpha", *buffers.alpha, batch * kRecurrentSmall);
        print_checksum(
            "beta", *buffers.beta, batch * kRecurrentSmall);
        require(backend->end(), "backend end");
        return 0;
    }
    if (format == "fp8-attention" || format == "q8-attention" ||
        format == "fp8-attention-batch4" ||
        format == "q8-attention-batch4") {
        const bool fp8 =
            format == "fp8-attention" ||
            format == "fp8-attention-batch4";
        const uint32_t batch =
            format == "fp8-attention-batch4" ||
            format == "q8-attention-batch4" ? 4U : 1U;
        AttentionWeights weights =
            make_attention_weights(*backend, fp8);
        AttentionBuffers buffers =
            make_attention_buffers(*backend, batch);
        auto measure_projection = [&](qw3::DeviceTensor &out,
                                      const qw3::DeviceWeight &weight,
                                      const char *label) {
            return measure_ms(
                *backend,
                [&]() {
                    require(backend->begin(), label);
                    if (batch == 1) {
                        require(backend->q8_0_matvec(
                                    out, weight, *buffers.input),
                                label);
                    } else {
                        require(backend->q8_0_matmul(
                                    out, weight, *buffers.input,
                                    batch, kHidden,
                                    static_cast<uint32_t>(weight.rows)),
                                label);
                    }
                },
                warmup, iterations);
        };
        const double q_ms = measure_projection(
            *buffers.q, *weights.q, "attention q projection");
        const double k_ms = measure_projection(
            *buffers.k, *weights.k, "attention k projection");
        const double v_ms = measure_projection(
            *buffers.v, *weights.v, "attention v projection");
        const double fanout_ms = measure_ms(
            *backend,
            [&]() {
                run_attention_fanout(
                    *backend, weights, buffers, batch);
            },
            warmup, iterations);
        std::printf(
            "format=%s batch=%u hidden=%u outputs=%u,%u,%u "
            "iterations=%u warmup=%u\n",
            format.c_str(), batch, kHidden, kAttentionQ,
            kAttentionKv, kAttentionKv, iterations, warmup);
        std::printf(
            "policy fp8_small_cublaslt=%s "
            "fp8_fanout_scale_fusion=%s "
            "fp8_fanout_custom_max_rows=%s "
            "fp8_packed_pair=%s\n",
            env_or_default("QW3_FP8_SMALL_CUBLASLT", "default"),
            env_or_default(
                "QW3_FP8_FANOUT_SCALE_FUSION", "default"),
            env_or_default(
                "QW3_FP8_FANOUT_CUSTOM_MAX_ROWS", "default"),
            env_or_default("QW3_FP8_PACKED_PAIR", "default"));
        std::printf(
            "q_ms=%.6f k_ms=%.6f v_ms=%.6f "
            "attention_fanout_ms=%.6f q_hash=%016llx "
            "k_hash=%016llx v_hash=%016llx\n",
            q_ms, k_ms, v_ms, fanout_ms,
            static_cast<unsigned long long>(
                tensor_hash(
                    *backend, *buffers.q, batch * kAttentionQ)),
            static_cast<unsigned long long>(
                tensor_hash(
                    *backend, *buffers.k, batch * kAttentionKv)),
            static_cast<unsigned long long>(
                tensor_hash(
                    *backend, *buffers.v, batch * kAttentionKv)));
        require(backend->end(), "backend end");
        return 0;
    }
    if (format == "fp8-output" || format == "q8-output") {
        const bool fp8 = format == "fp8-output";
        OutputProjection projection =
            make_output_projection(*backend, fp8);
        const double projection_ms = measure_ms(
            *backend,
            [&]() {
                run_output_projection(*backend, projection);
            },
            warmup, iterations);
        std::printf(
            "format=%s rows=%u cols=%u iterations=%u warmup=%u\n",
            format.c_str(), kHidden, kRecurrentValue,
            iterations, warmup);
        std::printf(
            "policy fp8_small_cublaslt=%s "
            "fp8_matvec_add_custom_max_cols=%s\n",
            env_or_default("QW3_FP8_SMALL_CUBLASLT", "default"),
            env_or_default(
                "QW3_FP8_MATVEC_ADD_CUSTOM_MAX_COLS", "default"));
        std::printf(
            "policy fp8_small_cublaslt=%s fp8_outer_scale=%s\n",
            env_or_default("QW3_FP8_SMALL_CUBLASLT", "default"),
            env_or_default(
                "QW3_FP8_CUBLASLT_OUTER_SCALE", "default"));
        std::printf(
            "output_projection_ms=%.6f output_hash=%016llx\n",
            projection_ms,
            static_cast<unsigned long long>(
                tensor_hash(*backend, *projection.residual, kHidden)));
        require(backend->end(), "backend end");
        return 0;
    }

    Weights weights;
    if (format == "nvfp4") {
        weights = make_nvfp4_weights(*backend);
    } else if (format == "fp8" || format == "fp8-batch4") {
        weights = make_fp8_weights(*backend);
    } else {
        weights = make_q8_weights(*backend);
    }
    const uint32_t batch =
        format == "fp8-batch4" || format == "q8-batch4" ? 4U : 1U;
    Buffers buffers = make_buffers(*backend, batch);

    const double gate_up_ms = measure_ms(
        *backend,
        [&]() { run_gate_up(*backend, weights, buffers, batch); },
        warmup, iterations);
    const double down_ms = measure_ms(
        *backend,
        [&]() { run_down(*backend, weights, buffers, batch); },
        warmup, iterations);
    const double full_ms = measure_ms(
        *backend,
        [&]() {
            run_gate_up(*backend, weights, buffers, batch);
            run_down(*backend, weights, buffers, batch);
        },
        warmup, iterations);

    std::printf(
        "format=%s batch=%u hidden=%u intermediate=%u "
        "iterations=%u warmup=%u\n",
        format.c_str(), batch, kHidden, kIntermediate,
        iterations, warmup);
    std::printf(
        "policy global=%s single=%s pair=%s tile_n=%s output_bf16=%s "
        "fused_cutlass_pair=%s vllm_tactics=%s fp8_small_cublaslt=%s\n",
        env_or_default("QW3_NVFP4_SMALL_MMVQ", "default"),
        env_or_default("QW3_NVFP4_SMALL_SINGLE", "default"),
        env_or_default("QW3_NVFP4_SMALL_PAIR", "default"),
        env_or_default("QW3_NVFP4_CUTLASS_TILE_N", "default"),
        env_or_default("QW3_NVFP4_CUTLASS_OUTPUT_BF16", "default"),
        env_or_default("QW3_NVFP4_FUSED_CUTLASS_PAIR", "default"),
        env_or_default("QW3_NVFP4_VLLM_TACTICS", "default"),
        env_or_default("QW3_FP8_SMALL_CUBLASLT", "default"));
    std::printf(
        "gate_up_ms=%.6f down_ms=%.6f full_ffn_ms=%.6f\n",
        gate_up_ms, down_ms, full_ms);
    return 0;
}
