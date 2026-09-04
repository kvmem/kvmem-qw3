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

} // namespace

int main(int argc, char **argv) try {
    if (argc != 5) {
        std::cerr << "usage: qw3-vision-gpu-parity MODEL_DIR PYTHON WORKER IMAGE\n";
        return 2;
    }
    const std::filesystem::path image_path = argv[4];
    std::ifstream image_in(image_path, std::ios::binary);
    if (!image_in) throw std::runtime_error("cannot open image");
    const std::vector<uint8_t> image_bytes(
        std::istreambuf_iterator<char>(image_in), {});
    qw3::VisionImage image;
    image.media_type = image_path.extension() == ".png" ? "image/png" : "image/jpeg";
    image.base64_data = base64_encode(image_bytes);

    auto backend = qw3::make_cuda_device_backend(qw3::LinearBackend::Cublas);
    if (!backend) throw std::runtime_error("CUDA backend unavailable");
    auto status = backend->begin();
    if (!status.ok) throw std::runtime_error(status.message);
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
    if (!gpu_result.storage || gpu_result.embedding_dim != cpu_result.embedding_dim ||
        gpu_result.grids.size() != cpu_result.grids.size()) {
        throw std::runtime_error("CPU/GPU vision result metadata mismatch");
    }
    const auto *storage = dynamic_cast<
        const qw3::detail::DeviceInputEmbeddingStorage *>(gpu_result.storage.get());
    if (!storage) throw std::runtime_error("unexpected GPU storage type");
    const uint64_t count = static_cast<uint64_t>(storage->rows()) * storage->dim();
    if (count != cpu_result.embeddings.size()) {
        throw std::runtime_error("CPU/GPU vision result shape mismatch");
    }
    std::vector<uint16_t> gpu_bf16(static_cast<size_t>(count));
    status = backend->copy_bytes_to_host(storage->tensor(), gpu_bf16.data(), 0,
                                         count * sizeof(uint16_t));
    if (!status.ok) throw std::runtime_error(status.message);

    double abs_sum = 0.0;
    double sq_error = 0.0;
    double dot = 0.0;
    double norm_cpu = 0.0;
    double norm_gpu = 0.0;
    float max_abs = 0.0f;
    for (uint64_t i = 0; i < count; ++i) {
        const float expected = cpu_result.embeddings[static_cast<size_t>(i)];
        const float actual = bf16_to_float(gpu_bf16[static_cast<size_t>(i)]);
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
