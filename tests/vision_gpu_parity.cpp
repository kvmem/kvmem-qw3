#include "qw3/device_backend.hpp"
#include "vision_cpu_frontend.hpp"
#include "vision_embedding_storage.hpp"
#include "vision_gpu_frontend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string base64_encode(const std::vector<uint8_t> &data) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((data.size() + 2) / 3 * 4);
    for (size_t i = 0; i < data.size(); i += 3) {
        const uint32_t a = data[i];
        const uint32_t b = i + 1 < data.size() ? data[i + 1] : 0;
        const uint32_t c = i + 2 < data.size() ? data[i + 2] : 0;
        const uint32_t value = (a << 16) | (b << 8) | c;
        out.push_back(alphabet[(value >> 18) & 63]);
        out.push_back(alphabet[(value >> 12) & 63]);
        out.push_back(i + 1 < data.size() ? alphabet[(value >> 6) & 63] : '=');
        out.push_back(i + 2 < data.size() ? alphabet[value & 63] : '=');
    }
    return out;
}

float bf16_to_float(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t float_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

void require_ok(qw3::DeviceStatus status) {
    if (!status.ok) throw std::runtime_error(status.message);
}

void test_vision_rounding(qw3::DeviceBackend &backend) {
    // Dot product falls exactly on a BF16 midpoint. The bias must be added
    // before rounding; a BF16 GEMM followed by a separate bias add returns 1.
    const uint16_t input[] = {float_to_bf16(1.0f), float_to_bf16(0.0625f)};
    const uint16_t bias[] = {float_to_bf16(0.001953125f)};
    auto x = backend.tensor_bf16(2, "vision test input");
    auto out = backend.tensor_bf16(3, "vision test padded output");
    auto w = backend.weight_bf16(input, 1, 2, "vision test weight");
    auto b = backend.weight_bf16(bias, 1, 1, "vision test bias");
    require_ok(backend.copy_bytes_from_host(*x, 0, input, sizeof(input)));
    require_ok(backend.vision_bf16_linear(*out, *w, *x, b.get(), nullptr,
                                        1, 2, 3, false, false));
    uint16_t actual[3];
    require_ok(backend.copy_bytes_to_host(*out, actual, 0, sizeof(actual)));
    if (actual[0] != float_to_bf16(1.0078125f)) {
        throw std::runtime_error("vision linear rounded before adding bias");
    }
    require_ok(backend.vision_bf16_linear(*out, *w, *x, b.get(), nullptr,
                                        1, 2, 3, false, false, true));
    require_ok(backend.copy_bytes_to_host(*out, actual, 0, sizeof(actual)));
    if (actual[0] != float_to_bf16(1.0f)) {
        throw std::runtime_error("vision patch Conv3d must round before bias");
    }
    auto residual = backend.tensor_bf16(3, "vision test residual");
    const uint16_t residual_values[] = {float_to_bf16(-1.0f), 0, 0};
    require_ok(backend.copy_bytes_from_host(*residual, 0, residual_values,
                                           sizeof(residual_values)));
    require_ok(backend.vision_bf16_linear(*out, *w, *x, b.get(), residual.get(),
                                        1, 2, 3, false, false));
    require_ok(backend.copy_bytes_to_host(*out, actual, 0, sizeof(actual)));
    if (actual[0] != float_to_bf16(0.0078125f)) {
        throw std::runtime_error("vision residual skipped Linear BF16 boundary");
    }

    // Values intentionally make FP32 vs BF16 sin/cos differ after the final
    // BF16 cast. Also exercise the padded head lanes and unchanged V values.
    const uint16_t packed[] = {float_to_bf16(1.0f), float_to_bf16(-1.0f),
                              float_to_bf16(1.0f), float_to_bf16(-1.0f),
                              float_to_bf16(2.0f), float_to_bf16(3.0f)};
    const float cosine[] = {std::cos(257.0f), std::cos(257.0f)};
    const float sine[] = {std::sin(257.0f), std::sin(257.0f)};
    auto qkv = backend.tensor_bf16(6, "vision test QKV");
    auto q = backend.tensor_bf16(4, "vision test Q");
    auto k = backend.tensor_bf16(4, "vision test K");
    auto v = backend.tensor_bf16(4, "vision test V");
    auto c = backend.tensor_f32(2, "vision test cos");
    auto s = backend.tensor_f32(2, "vision test sin");
    require_ok(backend.copy_bytes_from_host(*qkv, 0, packed, sizeof(packed)));
    require_ok(backend.copy_bytes_from_host(*c, 0, cosine, sizeof(cosine)));
    require_ok(backend.copy_bytes_from_host(*s, 0, sine, sizeof(sine)));
    require_ok(backend.vision_bf16_qkv_rope_pad(*q, *k, *v, *qkv, *c, *s,
                                              1, 1, 2, 4));
    const uint16_t expected[] = {float_to_bf16(cosine[0] + sine[0]),
                                float_to_bf16(-cosine[1] + sine[1]), 0, 0};
    for (const auto *tensor : {q.get(), k.get()}) {
        uint16_t result[4];
        require_ok(backend.copy_bytes_to_host(*tensor, result, 0, sizeof(result)));
        if (!std::equal(std::begin(result), std::end(result), expected)) {
            throw std::runtime_error("vision RoPE FP32/padding regression");
        }
    }
    uint16_t result_v[4];
    require_ok(backend.copy_bytes_to_host(*v, result_v, 0, sizeof(result_v)));
    if (result_v[0] != packed[4] || result_v[1] != packed[5] ||
        result_v[2] != 0 || result_v[3] != 0) {
        throw std::runtime_error("vision RoPE changed V or padding");
    }
    std::cout << "vision linear/RoPE rounding regression passed\n";
}

qw3::VisionImage load_image(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open image: " + path.string());
    const std::vector<uint8_t> bytes(
        std::istreambuf_iterator<char>(input), {});
    qw3::VisionImage image;
    image.media_type = path.extension() == ".png" ? "image/png" : "image/jpeg";
    image.base64_data = base64_encode(bytes);
    return image;
}

std::vector<uint16_t> copy_device_encoding(
        qw3::DeviceBackend &backend,
        const qw3::detail::DeviceInputEmbeddingStorage &storage) {
    const uint64_t count = static_cast<uint64_t>(storage.rows()) * storage.dim();
    std::vector<uint16_t> result(static_cast<size_t>(count));
    for (const auto &segment : storage.segments()) {
        auto status = backend.copy_bytes_to_host(
            *storage.resolve(segment.logical_row).tensor,
            result.data() + static_cast<uint64_t>(segment.logical_row) *
                                storage.dim(),
            static_cast<uint64_t>(segment.source_row) * storage.dim() *
                sizeof(uint16_t),
            static_cast<uint64_t>(segment.rows) * storage.dim() *
                sizeof(uint16_t));
        if (!status.ok) throw std::runtime_error(status.message);
    }
    return result;
}

int test_batch_parity(int argc, char **argv) {
    std::vector<qw3::VisionImage> images;
    for (int i = 5; i < argc; ++i) images.push_back(load_image(argv[i]));
    if (images.size() < 9) throw std::runtime_error("batch test requires at least 9 images");
    // Duplicates cross chunk boundaries and must preserve their logical rows.
    images.insert(images.begin() + 5, images.front());
    images.push_back(images.front());
    auto backend = qw3::make_cuda_device_backend(qw3::LinearBackend::Cublas);
    require_ok(backend->begin());
    test_vision_rounding(*backend);
    std::vector<uint16_t> reference;
    std::vector<qw3::detail::CpuVisionEncoding::Grid> reference_grids;
    for (const char *limit : {"0", "0", "4", "4"}) {
        setenv("QW3_VISION_MAX_BATCH_IMAGES", limit, 1);
        qw3::detail::GpuVisionFrontend gpu(*backend, argv[2], argv[3], argv[4], 8);
        const auto start = std::chrono::steady_clock::now();
        auto result = gpu.encode(images);
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        const auto &storage = dynamic_cast<const qw3::detail::DeviceInputEmbeddingStorage &>(*result.storage);
        auto values = copy_device_encoding(*backend, storage);
        if (reference.empty()) {
            reference = values;
            reference_grids = result.grids;
        }
        if (values.size() != reference.size() || result.grids.size() != reference_grids.size())
            throw std::runtime_error("batching changed the output shape");
        for (size_t i = 0; i < result.grids.size(); ++i) {
            const auto &a = result.grids[i], &b = reference_grids[i];
            if (a.temporal != b.temporal || a.height != b.height || a.width != b.width || a.rows != b.rows)
                throw std::runtime_error("batching changed image order or grids");
        }
        double dot = 0, a2 = 0, b2 = 0, abs_sum = 0, max_abs = 0;
        size_t differences = 0;
        for (size_t i = 0; i < values.size(); ++i) {
            const double a = bf16_to_float(reference[i]), b = bf16_to_float(values[i]);
            if (!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error("non-finite batch output");
            differences += reference[i] != values[i];
            dot += a * b; a2 += a * a; b2 += b * b;
            abs_sum += std::abs(a - b); max_abs = std::max(max_abs, std::abs(a - b));
        }
        const double cosine = dot / std::sqrt(a2 * b2), mean_abs = abs_sum / values.size();
        std::cout << std::setprecision(12) << "batch_limit=" << limit << " images=" << images.size()
                  << " encoded=" << result.cache_misses << " encode_ms=" << elapsed
                  << " elements=" << values.size() << " bitwise_differences=" << differences
                  << " cosine=" << cosine << " mean_abs=" << mean_abs << " max_abs=" << max_abs << std::endl;
        // Retain the existing BF16 CPU/GPU parity tolerance. Report exact
        // differences too; batching can select a different GEMM shape.
        if (!(cosine > 0.999 && mean_abs < 0.02)) throw std::runtime_error("batch numerical parity failed");
        auto reversed = images;
        std::reverse(reversed.begin(), reversed.end());
        auto cached = gpu.encode(reversed);
        if (cached.cache_misses) throw std::runtime_error("batch cache lost an image");
        auto reordered = copy_device_encoding(*backend,
            dynamic_cast<const qw3::detail::DeviceInputEmbeddingStorage &>(*cached.storage));
        size_t source = values.size(), target = 0;
        for (size_t i = images.size(); i-- > 0;) {
            const size_t count = static_cast<size_t>(result.grids[i].rows) * storage.dim();
            source -= count;
            if (!std::equal(values.begin() + source, values.begin() + source + count,
                            reordered.begin() + target)) throw std::runtime_error("cached batch reorder changed features");
            target += count;
        }
    }
    require_ok(backend->end());
    return 0;
}

} // namespace

int main(int argc, char **argv) try {
    if (argc >= 6 && std::string(argv[1]) == "--batch-parity") return test_batch_parity(argc, argv);
    if (argc != 5 && argc != 6) {
        std::cerr << "usage: qw3-vision-gpu-parity MODEL_DIR PYTHON WORKER "
                     "IMAGE [IMAGE2]\n";
        return 2;
    }
    const qw3::VisionImage image = load_image(argv[4]);
    const bool test_per_image_cache = argc == 6;
    const qw3::VisionImage image2 = test_per_image_cache
        ? load_image(argv[5]) : qw3::VisionImage{};

    auto backend = qw3::make_cuda_device_backend(qw3::LinearBackend::Cublas);
    if (!backend) throw std::runtime_error("CUDA backend unavailable");
    auto status = backend->begin();
    if (!status.ok) throw std::runtime_error(status.message);
    test_vision_rounding(*backend);
    qw3::detail::GpuVisionFrontend gpu(
        *backend, argv[1], argv[2], argv[3], 8);
    const char *reference_device = std::getenv("QW3_VISION_REFERENCE_DEVICE");
    qw3::detail::CpuVisionFrontend cpu(
        argv[1], argv[2], argv[3], 16, false,
        reference_device && std::string(reference_device) == "cuda"
            ? "cuda" : "cpu");

    uint32_t repeats = 1;
    if (const char *value = std::getenv("QW3_VISION_PARITY_REPEATS")) {
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (end != value && *end == '\0' && parsed > 0 && parsed <= 20) {
            repeats = static_cast<uint32_t>(parsed);
        }
    }
    qw3::detail::CpuVisionEncoding gpu_result;
    qw3::detail::CpuVisionEncoding cpu_result;
    double gpu_total_ms = 0.0;
    double cpu_total_ms = 0.0;
    double gpu_min_ms = std::numeric_limits<double>::infinity();
    double cpu_min_ms = std::numeric_limits<double>::infinity();
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        const auto gpu_begin = std::chrono::steady_clock::now();
        gpu_result = gpu.encode({image});
        const auto gpu_end = std::chrono::steady_clock::now();
        const auto cpu_begin = std::chrono::steady_clock::now();
        cpu_result = cpu.encode({image});
        const auto cpu_end = std::chrono::steady_clock::now();
        const double gpu_ms = std::chrono::duration<double, std::milli>(
            gpu_end - gpu_begin).count();
        const double cpu_ms = std::chrono::duration<double, std::milli>(
            cpu_end - cpu_begin).count();
        gpu_total_ms += gpu_ms;
        cpu_total_ms += cpu_ms;
        gpu_min_ms = std::min(gpu_min_ms, gpu_ms);
        cpu_min_ms = std::min(cpu_min_ms, cpu_ms);
    }
    double partial_hit_ms = 0.0;
    double reordered_hit_ms = 0.0;
    if (test_per_image_cache) {
        const auto partial_begin = std::chrono::steady_clock::now();
        gpu_result = gpu.encode({image, image2});
        const auto partial_end = std::chrono::steady_clock::now();
        partial_hit_ms = std::chrono::duration<double, std::milli>(
            partial_end - partial_begin).count();
        if (gpu_result.cache_hit || gpu_result.cache_hits != 1 ||
            gpu_result.cache_misses != 1) {
            throw std::runtime_error(
                "per-image cache did not report one hit and one miss");
        }
        const auto reordered_begin = std::chrono::steady_clock::now();
        gpu_result = gpu.encode({image2, image});
        const auto reordered_end = std::chrono::steady_clock::now();
        reordered_hit_ms = std::chrono::duration<double, std::milli>(
            reordered_end - reordered_begin).count();
        if (!gpu_result.cache_hit || gpu_result.cache_hits != 2 ||
            gpu_result.cache_misses != 0) {
            throw std::runtime_error(
                "per-image cache did not fully hit reordered images");
        }
        cpu_result = cpu.encode({image2, image});
        if (cpu_result.cache_hit || cpu_result.cache_hits != 1 ||
            cpu_result.cache_misses != 1) {
            throw std::runtime_error(
                "CPU per-image cache did not report one hit and one miss");
        }
    }
    if (!gpu_result.storage || gpu_result.embedding_dim != cpu_result.embedding_dim ||
        gpu_result.grids.size() != cpu_result.grids.size()) {
        throw std::runtime_error("CPU/GPU vision result metadata mismatch");
    }
    for (size_t i = 0; i < gpu_result.grids.size(); ++i) {
        const auto &a = gpu_result.grids[i];
        const auto &b = cpu_result.grids[i];
        if (a.temporal != b.temporal || a.height != b.height ||
            a.width != b.width || a.rows != b.rows) {
            throw std::runtime_error("CPU/GPU vision grid mismatch");
        }
    }
    const auto *storage = dynamic_cast<
        const qw3::detail::DeviceInputEmbeddingStorage *>(gpu_result.storage.get());
    if (!storage) throw std::runtime_error("unexpected GPU storage type");
    const uint64_t count = static_cast<uint64_t>(storage->rows()) * storage->dim();
    if (count != cpu_result.embeddings.size()) {
        throw std::runtime_error("CPU/GPU vision result shape mismatch");
    }
    std::vector<uint16_t> gpu_bf16 = copy_device_encoding(*backend, *storage);
    std::cout << std::setprecision(9);
    uint64_t offset = 0;
    for (size_t image_index = 0; image_index < cpu_result.grids.size(); ++image_index) {
        const auto &grid = cpu_result.grids[image_index];
        const uint64_t end = offset + static_cast<uint64_t>(grid.rows) * storage->dim();
        double dot = 0, a2 = 0, b2 = 0, abs_error = 0;
        for (uint64_t i = offset; i < end; ++i) {
            const double a = bf16_to_float(gpu_bf16[i]);
            const double b = cpu_result.embeddings[i];
            dot += a * b; a2 += a * a; b2 += b * b;
            abs_error += std::abs(a - b);
        }
        std::cout << "image=" << image_index << " grid=" << grid.temporal << ","
                  << grid.height << "," << grid.width << " rows=" << grid.rows
                  << " cosine=" << dot / std::sqrt(a2 * b2)
                  << " mean_abs=" << abs_error / (end - offset) << "\n";
        offset = end;
    }

    double abs_sum = 0.0;
    double sq_error = 0.0;
    double dot = 0.0;
    double norm_cpu = 0.0;
    double norm_gpu = 0.0;
    float max_abs = 0.0f;
    for (uint64_t i = 0; i < count; ++i) {
        const float expected = cpu_result.embeddings[static_cast<size_t>(i)];
        const float actual = bf16_to_float(gpu_bf16[static_cast<size_t>(i)]);
        if (!std::isfinite(expected) || !std::isfinite(actual)) {
            throw std::runtime_error("non-finite vision embedding");
        }
        const float error = std::abs(actual - expected);
        max_abs = std::max(max_abs, error);
        abs_sum += error;
        sq_error += static_cast<double>(error) * error;
        dot += static_cast<double>(actual) * expected;
        norm_cpu += static_cast<double>(expected) * expected;
        norm_gpu += static_cast<double>(actual) * actual;
    }
    const double cosine = dot / std::sqrt(norm_cpu * norm_gpu);
    std::cout << "rows=" << storage->rows()
              << " dim=" << storage->dim()
              << " weights_mib=" << (gpu.weight_bytes() / 1048576.0)
              << " repeats=" << repeats
              << " gpu_mean_ms=" << (gpu_total_ms / repeats)
              << " gpu_min_ms=" << gpu_min_ms
              << " cpu_mean_ms=" << (cpu_total_ms / repeats)
              << " cpu_min_ms=" << cpu_min_ms
              << " partial_hit_ms=" << partial_hit_ms
              << " reordered_hit_ms=" << reordered_hit_ms
              << " mean_abs=" << (abs_sum / count)
              << " rmse=" << std::sqrt(sq_error / count)
              << " max_abs=" << max_abs
              << " cosine=" << cosine << "\n";
    status = backend->end();
    if (!status.ok) throw std::runtime_error(status.message);
    if (!(cosine > 0.999 && abs_sum / count < 0.02)) return 1;
    return 0;
} catch (const std::exception &e) {
    std::cerr << "vision parity failed: " << e.what() << "\n";
    return 1;
}
