#include "vision_cpu_frontend.hpp"

#include "json.hpp"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace qw3::detail {
namespace {

using json = nlohmann::json;

void close_fd(int &fd) {
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
}

void write_all(int fd, const void *data, size_t size) {
    const auto *p = static_cast<const uint8_t *>(data);
    while (size > 0) {
        const ssize_t n = ::write(fd, p, size);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            throw std::runtime_error(
                std::string("vision worker write failed: ") + std::strerror(errno));
        }
        p += n;
        size -= static_cast<size_t>(n);
    }
}

void read_all(int fd, void *data, size_t size) {
    auto *p = static_cast<uint8_t *>(data);
    while (size > 0) {
        const ssize_t n = ::read(fd, p, size);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            throw std::runtime_error("vision worker closed while returning embeddings");
        }
        p += n;
        size -= static_cast<size_t>(n);
    }
}

std::string read_line(int fd) {
    std::string line;
    for (;;) {
        char c = 0;
        const ssize_t n = ::read(fd, &c, 1);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) throw std::runtime_error("vision worker exited unexpectedly");
        if (c == '\n') return line;
        if (line.size() >= (1u << 20)) {
            throw std::runtime_error("vision worker response header is too large");
        }
        line.push_back(c);
    }
}

uint64_t image_fingerprint(const CpuVisionImage &image) {
    uint64_t hash = 1469598103934665603ULL;
    const auto mix = [&](std::string_view value) {
        for (unsigned char byte : value) {
            hash ^= static_cast<uint64_t>(byte);
            hash *= 1099511628211ULL;
        }
        hash ^= 0xffu;
        hash *= 1099511628211ULL;
    };
    mix(image.media_type);
    mix(image.base64_data);
    return hash != 0 ? hash : 1;
}

bool same_image(const CpuVisionImage &a, const CpuVisionImage &b) {
    return a.media_type == b.media_type && a.base64_data == b.base64_data;
}

} // namespace

struct CpuVisionFrontend::Impl {
    struct CacheEntry {
        CpuVisionImage image;
        uint64_t fingerprint = 0;
        CpuVisionEncoding::Grid grid;
        uint32_t dim = 0;
        std::shared_ptr<std::vector<float>> embeddings;
        uint64_t bytes = 0;
    };

    struct ImageResult {
        CpuVisionEncoding::Grid grid;
        uint32_t dim = 0;
        std::shared_ptr<std::vector<float>> embeddings;
    };

    pid_t pid = -1;
    int request_fd = -1;
    int response_fd = -1;
    std::mutex mutex;
    uint64_t cache_limit_bytes = 512ULL << 20;
    uint64_t cache_bytes = 0;
    std::vector<CacheEntry> cache;
    bool preprocess_only = false;

    Impl(const std::string &model_directory,
         const std::string &python_executable,
         const std::string &worker_script,
         uint32_t threads,
         bool preprocess_only_arg,
         const std::string &runtime_device) : preprocess_only(preprocess_only_arg) {
        if (const char *value = std::getenv("QW3_VISION_CPU_CACHE_MIB")) {
            char *end = nullptr;
            errno = 0;
            const unsigned long long mib = std::strtoull(value, &end, 10);
            if (errno == 0 && end != value && *end == '\0' &&
                mib <= std::numeric_limits<uint64_t>::max() / (1ULL << 20)) {
                cache_limit_bytes = static_cast<uint64_t>(mib) << 20;
            }
        }
        int to_child[2] = {-1, -1};
        int from_child[2] = {-1, -1};
        if (::pipe(to_child) != 0 || ::pipe(from_child) != 0) {
            close_fd(to_child[0]); close_fd(to_child[1]);
            close_fd(from_child[0]); close_fd(from_child[1]);
            throw std::runtime_error(
                std::string("cannot create vision worker pipes: ") +
                std::strerror(errno));
        }
        pid = ::fork();
        if (pid < 0) {
            close_fd(to_child[0]); close_fd(to_child[1]);
            close_fd(from_child[0]); close_fd(from_child[1]);
            throw std::runtime_error(
                std::string("cannot fork vision worker: ") + std::strerror(errno));
        }
        if (pid == 0) {
            // Interactive Ctrl-C is intended for the serving parent. Let the
            // worker observe pipe EOF/the explicit shutdown frame instead of
            // emitting an unrelated Python KeyboardInterrupt traceback.
            (void)::signal(SIGINT, SIG_IGN);
            ::dup2(to_child[0], STDIN_FILENO);
            ::dup2(from_child[1], STDOUT_FILENO);
            close_fd(to_child[0]); close_fd(to_child[1]);
            close_fd(from_child[0]); close_fd(from_child[1]);
            const std::string thread_arg = std::to_string(threads);
            if (preprocess_only) {
                ::execl(python_executable.c_str(), python_executable.c_str(),
                        worker_script.c_str(), "--model", model_directory.c_str(),
                        "--threads", thread_arg.c_str(), "--preprocess-only",
                        static_cast<char *>(nullptr));
            } else {
                ::execl(python_executable.c_str(), python_executable.c_str(),
                        worker_script.c_str(), "--model", model_directory.c_str(),
                        "--threads", thread_arg.c_str(), "--device",
                        runtime_device.c_str(), static_cast<char *>(nullptr));
            }
            std::fprintf(stderr, "failed to exec vision worker: %s\n",
                         std::strerror(errno));
            _exit(127);
        }
        close_fd(to_child[0]);
        close_fd(from_child[1]);
        request_fd = to_child[1];
        response_fd = from_child[0];
        // A worker-side exception must become a request error rather than a
        // process-wide SIGPIPE termination on the next encode call. This is
        // installed only when the opt-in vision frontend is constructed.
        (void)::signal(SIGPIPE, SIG_IGN);

        try {
            const json ready = json::parse(read_line(response_fd));
            if (!ready.value("ok", false) ||
                ready.value("event", "") != "ready") {
                throw std::runtime_error(
                    "vision worker failed to initialize: " +
                    ready.value("error", "unknown error"));
            }
        } catch (...) {
            close_fd(request_fd);
            close_fd(response_fd);
            int status = 0;
            if (pid > 0) {
                ::kill(pid, SIGTERM);
                (void)::waitpid(pid, &status, 0);
                pid = -1;
            }
            throw;
        }
    }

    ~Impl() {
        if (request_fd >= 0) {
            try {
                const std::string shutdown = "{\"op\":\"shutdown\"}\n";
                write_all(request_fd, shutdown.data(), shutdown.size());
            } catch (...) {
            }
        }
        close_fd(request_fd);
        close_fd(response_fd);
        if (pid > 0) {
            int status = 0;
            for (int i = 0; i < 20; ++i) {
                const pid_t result = ::waitpid(pid, &status, WNOHANG);
                if (result == pid || result < 0) return;
                ::usleep(10000);
            }
            ::kill(pid, SIGTERM);
            (void)::waitpid(pid, &status, 0);
        }
    }
};

CpuVisionFrontend::CpuVisionFrontend(std::string model_directory,
                                     std::string python_executable,
                                     std::string worker_script,
                                     uint32_t threads,
                                     bool preprocess_only,
                                     std::string runtime_device)
    : impl_(std::make_unique<Impl>(model_directory, python_executable,
                                   worker_script, std::max<uint32_t>(1, threads),
                                   preprocess_only, runtime_device)) {}

CpuVisionFrontend::~CpuVisionFrontend() = default;

CpuVisionEncoding CpuVisionFrontend::encode(
        const std::vector<CpuVisionImage> &images) {
    if (images.empty()) return {};
    if (impl_->preprocess_only) {
        throw std::runtime_error("encode called on preprocess-only vision worker");
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::vector<Impl::ImageResult> ordered(images.size());
    std::vector<uint64_t> fingerprints(images.size());
    std::vector<size_t> unique_miss_requests;
    std::vector<uint32_t> request_to_miss(
        images.size(), std::numeric_limits<uint32_t>::max());
    uint32_t cache_hits = 0;

    for (size_t request_index = 0; request_index < images.size();
         ++request_index) {
        fingerprints[request_index] = image_fingerprint(images[request_index]);
        bool found = false;
        for (size_t cache_index = 0; cache_index < impl_->cache.size();
             ++cache_index) {
            if (impl_->cache[cache_index].fingerprint !=
                    fingerprints[request_index] ||
                !same_image(impl_->cache[cache_index].image,
                            images[request_index])) {
                continue;
            }
            Impl::CacheEntry hit = std::move(impl_->cache[cache_index]);
            impl_->cache.erase(impl_->cache.begin() +
                               static_cast<std::ptrdiff_t>(cache_index));
            impl_->cache.push_back(std::move(hit));
            const auto &entry = impl_->cache.back();
            ordered[request_index] = {
                entry.grid, entry.dim, entry.embeddings};
            ++cache_hits;
            found = true;
            break;
        }
        if (found) continue;
        for (uint32_t miss = 0; miss < unique_miss_requests.size(); ++miss) {
            const size_t prior = unique_miss_requests[miss];
            if (fingerprints[prior] == fingerprints[request_index] &&
                same_image(images[prior], images[request_index])) {
                request_to_miss[request_index] = miss;
                ++cache_hits;
                found = true;
                break;
            }
        }
        if (!found) {
            request_to_miss[request_index] =
                static_cast<uint32_t>(unique_miss_requests.size());
            unique_miss_requests.push_back(request_index);
        }
    }

    if (!unique_miss_requests.empty()) {
        json request{{"op", "encode"}, {"images", json::array()}};
        for (size_t request_index : unique_miss_requests) {
            const auto &image = images[request_index];
            request["images"].push_back(
                json{{"media_type", image.media_type},
                     {"data", image.base64_data}});
        }
        const std::string wire = request.dump() + "\n";
        write_all(impl_->request_fd, wire.data(), wire.size());
        const json response = json::parse(read_line(impl_->response_fd));
        if (!response.value("ok", false)) {
            throw std::runtime_error(
                "CPU vision encoding failed: " +
                response.value("error", "unknown error"));
        }
        if (response.value("dtype", "") != "f32") {
            throw std::runtime_error(
                "CPU vision worker returned an unsupported dtype");
        }
        const uint64_t rows = response.at("rows").get<uint64_t>();
        const uint64_t dim = response.at("dim").get<uint64_t>();
        const uint64_t bytes = response.at("bytes").get<uint64_t>();
        if (rows == 0 || dim == 0 ||
            dim > std::numeric_limits<uint32_t>::max() ||
            rows > std::numeric_limits<uint64_t>::max() / dim ||
            rows * dim > std::numeric_limits<size_t>::max() / sizeof(float) ||
            bytes != rows * dim * sizeof(float)) {
            throw std::runtime_error(
                "CPU vision worker returned an invalid embedding shape");
        }
        std::vector<float> flat(static_cast<size_t>(rows * dim));
        read_all(impl_->response_fd, flat.data(), static_cast<size_t>(bytes));

        const json &grids = response.at("grids");
        const json &counts = response.at("counts");
        if (!grids.is_array() || !counts.is_array() ||
            grids.size() != unique_miss_requests.size() ||
            counts.size() != unique_miss_requests.size()) {
            throw std::runtime_error(
                "CPU vision worker returned inconsistent image grids");
        }
        std::vector<Impl::ImageResult> miss_results(
            unique_miss_requests.size());
        uint64_t source_row = 0;
        for (size_t miss = 0; miss < unique_miss_requests.size(); ++miss) {
            if (!grids[miss].is_array() || grids[miss].size() != 3) {
                throw std::runtime_error(
                    "CPU vision worker returned an invalid grid");
            }
            CpuVisionEncoding::Grid grid;
            grid.temporal = grids[miss][0].get<uint32_t>();
            grid.height = grids[miss][1].get<uint32_t>();
            grid.width = grids[miss][2].get<uint32_t>();
            grid.rows = counts[miss].get<uint32_t>();
            if (grid.rows == 0 || grid.rows > rows ||
                source_row > rows - grid.rows) {
                throw std::runtime_error(
                    "CPU vision grid rows exceed embeddings");
            }
            auto image_embeddings = std::make_shared<std::vector<float>>(
                flat.begin() + static_cast<std::ptrdiff_t>(source_row * dim),
                flat.begin() + static_cast<std::ptrdiff_t>(
                    (source_row + grid.rows) * dim));
            Impl::ImageResult part{
                grid, static_cast<uint32_t>(dim), image_embeddings};
            const uint64_t result_bytes = static_cast<uint64_t>(
                image_embeddings->size()) * sizeof(float);
            const size_t request_index = unique_miss_requests[miss];
            if (impl_->cache_limit_bytes > 0 &&
                result_bytes <= impl_->cache_limit_bytes) {
                while (!impl_->cache.empty() &&
                       impl_->cache_bytes >
                           impl_->cache_limit_bytes - result_bytes) {
                    impl_->cache_bytes -= impl_->cache.front().bytes;
                    impl_->cache.erase(impl_->cache.begin());
                }
                Impl::CacheEntry entry;
                entry.image = images[request_index];
                entry.fingerprint = fingerprints[request_index];
                entry.grid = grid;
                entry.dim = static_cast<uint32_t>(dim);
                entry.embeddings = image_embeddings;
                entry.bytes = result_bytes;
                impl_->cache.push_back(std::move(entry));
                impl_->cache_bytes += result_bytes;
            }
            miss_results[miss] = std::move(part);
            source_row += grid.rows;
        }
        if (source_row != rows) {
            throw std::runtime_error(
                "CPU vision grid rows do not match embeddings");
        }
        for (size_t request_index = 0; request_index < images.size();
             ++request_index) {
            const uint32_t miss = request_to_miss[request_index];
            if (miss != std::numeric_limits<uint32_t>::max()) {
                ordered[request_index] = miss_results[miss];
            }
        }
    }

    CpuVisionEncoding result;
    result.cache_hits = cache_hits;
    result.cache_misses = static_cast<uint32_t>(unique_miss_requests.size());
    result.cache_hit = result.cache_misses == 0;
    uint64_t total_values = 0;
    for (const auto &part : ordered) {
        if (!part.embeddings || part.grid.rows == 0 || part.dim == 0 ||
            part.embeddings->size() !=
                static_cast<uint64_t>(part.grid.rows) * part.dim ||
            (result.embedding_dim != 0 && result.embedding_dim != part.dim) ||
            total_values > std::numeric_limits<size_t>::max() -
                               part.embeddings->size()) {
            throw std::runtime_error("invalid cached CPU vision image result");
        }
        result.embedding_dim = part.dim;
        result.grids.push_back(part.grid);
        total_values += part.embeddings->size();
    }
    result.embeddings.reserve(static_cast<size_t>(total_values));
    for (const auto &part : ordered) {
        result.embeddings.insert(result.embeddings.end(),
                                 part.embeddings->begin(),
                                 part.embeddings->end());
    }
    return result;
}

CpuVisionPatches CpuVisionFrontend::preprocess(
        const std::vector<CpuVisionImage> &images) {
    if (images.empty()) return {};
    if (!impl_->preprocess_only) {
        throw std::runtime_error("preprocess called on encoding vision worker");
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    json request{{"op", "preprocess"}, {"images", json::array()}};
    for (const CpuVisionImage &image : images) {
        request["images"].push_back(
            json{{"media_type", image.media_type}, {"data", image.base64_data}});
    }
    const std::string wire = request.dump() + "\n";
    write_all(impl_->request_fd, wire.data(), wire.size());
    const json response = json::parse(read_line(impl_->response_fd));
    if (!response.value("ok", false)) {
        throw std::runtime_error(
            "vision preprocessing failed: " + response.value("error", "unknown error"));
    }
    if (response.value("dtype", "") != "bf16") {
        throw std::runtime_error("vision preprocessor returned an unsupported dtype");
    }
    const uint64_t rows = response.at("rows").get<uint64_t>();
    const uint64_t dim = response.at("dim").get<uint64_t>();
    const uint64_t bytes = response.at("bytes").get<uint64_t>();
    if (rows == 0 || dim == 0 || dim > std::numeric_limits<uint32_t>::max() ||
        rows > std::numeric_limits<uint64_t>::max() / dim ||
        rows * dim > std::numeric_limits<size_t>::max() / sizeof(uint16_t) ||
        bytes != rows * dim * sizeof(uint16_t)) {
        throw std::runtime_error("vision preprocessor returned an invalid patch shape");
    }
    CpuVisionPatches result;
    result.patch_dim = static_cast<uint32_t>(dim);
    result.patches.resize(static_cast<size_t>(rows * dim));
    read_all(impl_->response_fd, result.patches.data(), static_cast<size_t>(bytes));

    const json &grids = response.at("grids");
    const json &counts = response.at("counts");
    const json &patch_counts = response.at("patch_counts");
    if (!grids.is_array() || !counts.is_array() || !patch_counts.is_array() ||
        grids.size() != images.size() || counts.size() != images.size() ||
        patch_counts.size() != images.size()) {
        throw std::runtime_error("vision preprocessor returned inconsistent image grids");
    }
    uint64_t counted_patches = 0;
    for (size_t i = 0; i < grids.size(); ++i) {
        if (!grids[i].is_array() || grids[i].size() != 3) {
            throw std::runtime_error("vision preprocessor returned an invalid grid");
        }
        CpuVisionPatches::Grid grid;
        grid.temporal = grids[i][0].get<uint32_t>();
        grid.height = grids[i][1].get<uint32_t>();
        grid.width = grids[i][2].get<uint32_t>();
        grid.rows = counts[i].get<uint32_t>();
        const uint64_t expected_patches = static_cast<uint64_t>(grid.temporal) *
                                          grid.height * grid.width;
        if (patch_counts[i].get<uint64_t>() != expected_patches ||
            grid.rows != expected_patches / 4) {
            throw std::runtime_error("vision preprocessor grid/count mismatch");
        }
        counted_patches += expected_patches;
        result.grids.push_back(grid);
    }
    if (counted_patches != rows) {
        throw std::runtime_error("vision grids do not match preprocessed patches");
    }
    return result;
}

} // namespace qw3::detail
