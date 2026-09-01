#include "vision_cpu_frontend.hpp"

#include "json.hpp"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
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

} // namespace

struct CpuVisionFrontend::Impl {
    pid_t pid = -1;
    int request_fd = -1;
    int response_fd = -1;
    std::mutex mutex;

    Impl(const std::string &model_directory,
         const std::string &python_executable,
         const std::string &worker_script,
         uint32_t threads) {
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
            ::dup2(to_child[0], STDIN_FILENO);
            ::dup2(from_child[1], STDOUT_FILENO);
            close_fd(to_child[0]); close_fd(to_child[1]);
            close_fd(from_child[0]); close_fd(from_child[1]);
            const std::string thread_arg = std::to_string(threads);
            ::execl(python_executable.c_str(), python_executable.c_str(),
                    worker_script.c_str(), "--model", model_directory.c_str(),
                    "--threads", thread_arg.c_str(), static_cast<char *>(nullptr));
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
                                     uint32_t threads)
    : impl_(std::make_unique<Impl>(model_directory, python_executable,
                                   worker_script, std::max<uint32_t>(1, threads))) {}

CpuVisionFrontend::~CpuVisionFrontend() = default;

CpuVisionEncoding CpuVisionFrontend::encode(
        const std::vector<CpuVisionImage> &images) {
    if (images.empty()) return {};
    std::lock_guard<std::mutex> lock(impl_->mutex);
    json request{{"op", "encode"}, {"images", json::array()}};
    for (const CpuVisionImage &image : images) {
        request["images"].push_back(
            json{{"media_type", image.media_type}, {"data", image.base64_data}});
    }
    const std::string wire = request.dump() + "\n";
    write_all(impl_->request_fd, wire.data(), wire.size());
    const json response = json::parse(read_line(impl_->response_fd));
    if (!response.value("ok", false)) {
        throw std::runtime_error(
            "CPU vision encoding failed: " + response.value("error", "unknown error"));
    }
    if (response.value("dtype", "") != "f32") {
        throw std::runtime_error("CPU vision worker returned an unsupported dtype");
    }
    const uint64_t rows = response.at("rows").get<uint64_t>();
    const uint64_t dim = response.at("dim").get<uint64_t>();
    const uint64_t bytes = response.at("bytes").get<uint64_t>();
    if (rows == 0 || dim == 0 || dim > std::numeric_limits<uint32_t>::max() ||
        rows > std::numeric_limits<uint64_t>::max() / dim ||
        rows * dim > std::numeric_limits<size_t>::max() / sizeof(float) ||
        bytes != rows * dim * sizeof(float)) {
        throw std::runtime_error("CPU vision worker returned an invalid embedding shape");
    }
    CpuVisionEncoding result;
    result.embedding_dim = static_cast<uint32_t>(dim);
    result.embeddings.resize(static_cast<size_t>(rows * dim));
    read_all(impl_->response_fd, result.embeddings.data(), static_cast<size_t>(bytes));

    const json &grids = response.at("grids");
    const json &counts = response.at("counts");
    if (!grids.is_array() || !counts.is_array() ||
        grids.size() != images.size() || counts.size() != images.size()) {
        throw std::runtime_error("CPU vision worker returned inconsistent image grids");
    }
    uint64_t counted_rows = 0;
    for (size_t i = 0; i < grids.size(); ++i) {
        if (!grids[i].is_array() || grids[i].size() != 3) {
            throw std::runtime_error("CPU vision worker returned an invalid grid");
        }
        CpuVisionEncoding::Grid grid;
        grid.temporal = grids[i][0].get<uint32_t>();
        grid.height = grids[i][1].get<uint32_t>();
        grid.width = grids[i][2].get<uint32_t>();
        grid.rows = counts[i].get<uint32_t>();
        counted_rows += grid.rows;
        result.grids.push_back(grid);
    }
    if (counted_rows != rows) {
        throw std::runtime_error("CPU vision grid rows do not match embeddings");
    }
    return result;
}

} // namespace qw3::detail
