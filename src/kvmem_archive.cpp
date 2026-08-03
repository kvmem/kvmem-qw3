#include "qw3/kvmem_archive.hpp"

#include "json.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef QW3_HAVE_OPENSSL
#include <openssl/evp.h>
#endif

namespace qw3 {
namespace {

using nlohmann::json;

constexpr uint32_t kStateMagic = 0x41434b51u;  // "QKCA"
constexpr uint32_t kStateVersion = 1;

// Compact streaming SHA-256 fallback used for model identity when OpenSSL is
// unavailable.
class Sha256 {
public:
    void update(const void *data, size_t bytes) {
        const auto *p = static_cast<const uint8_t *>(data);
        total_bytes_ += bytes;
        while (bytes > 0) {
            const size_t n = std::min(bytes, block_.size() - buffered_);
            std::memcpy(block_.data() + buffered_, p, n);
            buffered_ += n;
            p += n;
            bytes -= n;
            if (buffered_ == block_.size()) {
                transform(block_.data());
                buffered_ = 0;
            }
        }
    }

    std::string finish_hex() {
        const uint64_t bit_count = total_bytes_ * 8ull;
        const uint8_t one = 0x80;
        update(&one, 1);
        const uint8_t zero = 0;
        while (buffered_ != 56) update(&zero, 1);
        uint8_t length[8];
        for (uint32_t i = 0; i < 8; ++i) {
            length[7 - i] = static_cast<uint8_t>(bit_count >> (i * 8));
        }
        update(length, sizeof(length));
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (uint32_t v : state_) out << std::setw(8) << v;
        return out.str();
    }

private:
    static uint32_t rotr(uint32_t x, uint32_t n) {
        return (x >> n) | (x << (32 - n));
    }

    void transform(const uint8_t *p) {
        static constexpr uint32_t k[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
            0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
            0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
            0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
            0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
        };
        uint32_t w[64];
        for (uint32_t i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(p[i * 4]) << 24) |
                   (static_cast<uint32_t>(p[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(p[i * 4 + 2]) << 8) |
                   static_cast<uint32_t>(p[i * 4 + 3]);
        }
        for (uint32_t i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^
                                rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^
                                rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (uint32_t i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + s1 + ch + k[i] + w[i];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<uint32_t, 8> state_{{
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    }};
    std::array<uint8_t, 64> block_{};
    uint64_t total_bytes_ = 0;
    size_t buffered_ = 0;
};

std::string sha256_string(const std::string &text) {
    Sha256 sha;
    sha.update(text.data(), text.size());
    return sha.finish_hex();
}

bool is_sha256_hex(const std::string &text) {
    if (text.size() != 64) return false;
    return std::all_of(text.begin(), text.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') ||
               (c >= 'a' && c <= 'f') ||
               (c >= 'A' && c <= 'F');
    });
}

int64_t unix_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void ensure_dir(const std::string &path) {
    struct stat st {};
    if (::stat(path.c_str(), &st) == 0) {
        if (!S_ISDIR(st.st_mode)) {
            throw std::runtime_error(
                "KVMem archive path is not a directory: " + path);
        }
        return;
    }
    // An archive is normally addressed by a full path on a dedicated volume,
    // so create the intermediate components rather than requiring the operator
    // to pre-create them.
    const size_t slash = path.find_last_of('/');
    if (slash != std::string::npos && slash > 0) {
        ensure_dir(path.substr(0, slash));
    }
    if (::mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        throw std::runtime_error("failed to create KVMem archive directory: " +
                                 path + ": " + std::strerror(errno));
    }
}

bool path_exists(const std::string &path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0;
}

uint64_t file_size(const std::string &path) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) return 0;
    return static_cast<uint64_t>(st.st_size);
}

void write_all(int fd, const void *data, uint64_t bytes, uint64_t offset) {
    const uint8_t *src = static_cast<const uint8_t *>(data);
    uint64_t done = 0;
    while (done < bytes) {
        const ssize_t n = ::pwrite(fd, src + done,
                                   static_cast<size_t>(bytes - done),
                                   static_cast<off_t>(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            throw std::runtime_error("KVMem archive write failed: " +
                                     std::string(std::strerror(errno)));
        }
        done += static_cast<uint64_t>(n);
    }
}

void read_all(int fd, void *data, uint64_t bytes, uint64_t offset) {
    uint8_t *dst = static_cast<uint8_t *>(data);
    uint64_t done = 0;
    while (done < bytes) {
        const ssize_t n = ::pread(fd, dst + done,
                                  static_cast<size_t>(bytes - done),
                                  static_cast<off_t>(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            throw std::runtime_error(
                "KVMem archive read failed or hit unwritten data");
        }
        done += static_cast<uint64_t>(n);
    }
}

void write_file(const std::string &path, const void *data, uint64_t bytes) {
    const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC,
                          0644);
    if (fd < 0) {
        throw std::runtime_error("failed to open KVMem archive file for write: " +
                                 path + ": " + std::strerror(errno));
    }
    try {
        if (bytes > 0) write_all(fd, data, bytes, 0);
    } catch (...) {
        ::close(fd);
        throw;
    }
    ::close(fd);
}

std::vector<uint8_t> read_file(const std::string &path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return {};
    const uint64_t bytes = file_size(path);
    std::vector<uint8_t> out(static_cast<size_t>(bytes));
    try {
        if (bytes > 0) read_all(fd, out.data(), bytes, 0);
    } catch (...) {
        ::close(fd);
        throw;
    }
    ::close(fd);
    return out;
}

json layout_to_json(const KvMemArchiveLayout &l) {
    return json{
        {"architecture", l.architecture},
        {"model_name", l.model_name},
        {"model_bytes", l.model_bytes},
        {"model_sha256", l.model_sha256},
        {"n_layers", l.n_layers},
        {"full_attention_interval", l.full_attention_interval},
        {"n_standard_layers", l.n_standard_layers},
        {"n_kv_heads", l.n_kv_heads},
        {"head_dim", l.head_dim},
        {"head_v_dim", l.head_v_dim},
        {"rope_dim", l.rope_dim},
        {"rope_theta", l.rope_theta},
        {"kv_dtype", l.kv_dtype},
        {"block_tokens", l.block_tokens},
        {"raw_chunk_tokens", l.raw_chunk_tokens},
        {"kv_page_size", l.kv_page_size},
        {"immutable_source_k", l.immutable_source_k},
        {"raw_k_block_major", l.raw_k_block_major},
        {"mtp_archived", l.mtp_archived},
        {"raw_k_row_bytes", l.raw_k_row_bytes},
        {"raw_chunk_bytes", l.raw_chunk_bytes},
        {"mtp_chunk_bytes", l.mtp_chunk_bytes},
        {"v_block_bytes", l.v_block_bytes},
    };
}

KvMemArchiveLayout layout_from_json(const json &j) {
    KvMemArchiveLayout l;
    l.architecture = j.value("architecture", std::string());
    l.model_name = j.value("model_name", std::string());
    l.model_bytes = j.value("model_bytes", 0ull);
    l.model_sha256 = j.value("model_sha256", std::string());
    l.n_layers = j.value("n_layers", 0u);
    l.full_attention_interval = j.value("full_attention_interval", 0u);
    l.n_standard_layers = j.value("n_standard_layers", 0u);
    l.n_kv_heads = j.value("n_kv_heads", 0u);
    l.head_dim = j.value("head_dim", 0u);
    l.head_v_dim = j.value("head_v_dim", 0u);
    l.rope_dim = j.value("rope_dim", 0u);
    l.rope_theta = j.value("rope_theta", 0.0);
    l.kv_dtype = j.value("kv_dtype", std::string());
    l.block_tokens = j.value("block_tokens", 0u);
    l.raw_chunk_tokens = j.value("raw_chunk_tokens", 0u);
    l.kv_page_size = j.value("kv_page_size", 0u);
    l.immutable_source_k = j.value("immutable_source_k", true);
    l.raw_k_block_major = j.value("raw_k_block_major", true);
    l.mtp_archived = j.value("mtp_archived", false);
    l.raw_k_row_bytes = j.value("raw_k_row_bytes", 0ull);
    l.raw_chunk_bytes = j.value("raw_chunk_bytes", 0ull);
    l.mtp_chunk_bytes = j.value("mtp_chunk_bytes", 0ull);
    l.v_block_bytes = j.value("v_block_bytes", 0ull);
    return l;
}

} // namespace

std::string KvMemArchiveLayout::key() const {
    std::ostringstream out;
    out << architecture << '|' << model_bytes << '|' << model_sha256 << '|'
        << n_layers << '|'
        << full_attention_interval << '|' << n_standard_layers << '|'
        << n_kv_heads << '|' << head_dim << '|' << head_v_dim << '|' << rope_dim
        << '|' << rope_theta << '|' << kv_dtype << '|' << block_tokens << '|'
        << raw_chunk_tokens << '|' << kv_page_size << '|'
        << (immutable_source_k ? 1 : 0) << '|' << (raw_k_block_major ? 1 : 0)
        << '|' << (mtp_archived ? 1 : 0) << '|' << raw_k_row_bytes << '|'
        << raw_chunk_bytes << '|' << mtp_chunk_bytes << '|' << v_block_bytes;
    return out.str();
}

std::string KvMemArchiveLayout::explain_mismatch(
        const KvMemArchiveLayout &other) const {
    auto diff_str = [](const char *name, const std::string &a,
                       const std::string &b) {
        return a == b ? std::string()
                      : std::string(name) + " archive=" + a + " engine=" + b;
    };
    auto diff_u64 = [](const char *name, uint64_t a, uint64_t b) {
        return a == b ? std::string()
                      : std::string(name) + " archive=" + std::to_string(a) +
                            " engine=" + std::to_string(b);
    };
    const std::string checks[] = {
        diff_str("architecture", architecture, other.architecture),
        diff_u64("model_bytes", model_bytes, other.model_bytes),
        diff_str("model_sha256", model_sha256, other.model_sha256),
        diff_u64("n_layers", n_layers, other.n_layers),
        diff_u64("full_attention_interval", full_attention_interval,
                 other.full_attention_interval),
        diff_u64("n_standard_layers", n_standard_layers,
                 other.n_standard_layers),
        diff_u64("n_kv_heads", n_kv_heads, other.n_kv_heads),
        diff_u64("head_dim", head_dim, other.head_dim),
        diff_u64("head_v_dim", head_v_dim, other.head_v_dim),
        diff_u64("rope_dim", rope_dim, other.rope_dim),
        diff_str("kv_dtype", kv_dtype, other.kv_dtype),
        diff_u64("block_tokens", block_tokens, other.block_tokens),
        diff_u64("raw_chunk_tokens", raw_chunk_tokens, other.raw_chunk_tokens),
        diff_u64("kv_page_size", kv_page_size, other.kv_page_size),
        diff_u64("immutable_source_k", immutable_source_k ? 1 : 0,
                 other.immutable_source_k ? 1 : 0),
        diff_u64("raw_k_block_major", raw_k_block_major ? 1 : 0,
                 other.raw_k_block_major ? 1 : 0),
        diff_u64("mtp_archived", mtp_archived ? 1 : 0,
                 other.mtp_archived ? 1 : 0),
        diff_u64("raw_k_row_bytes", raw_k_row_bytes, other.raw_k_row_bytes),
        diff_u64("raw_chunk_bytes", raw_chunk_bytes, other.raw_chunk_bytes),
        diff_u64("mtp_chunk_bytes", mtp_chunk_bytes, other.mtp_chunk_bytes),
        diff_u64("v_block_bytes", v_block_bytes, other.v_block_bytes),
    };
    for (const std::string &c : checks) {
        if (!c.empty()) return c;
    }
    if (rope_theta != other.rope_theta) {
        return "rope_theta archive=" + std::to_string(rope_theta) +
               " engine=" + std::to_string(other.rope_theta);
    }
    return {};
}

uint64_t kvmem_archive_state_bytes(
        const std::vector<KvMemArchiveStateEntry> &entries) {
    uint64_t total = 0;
    for (const KvMemArchiveStateEntry &e : entries) total += e.bytes;
    return total;
}

std::string kvmem_archive_model_sha256(const std::string &model_path) {
    struct stat before {};
    if (::stat(model_path.c_str(), &before) != 0 ||
        !S_ISREG(before.st_mode)) {
        throw std::runtime_error(
            "cannot stat GGUF for KVMem archive identity: " + model_path +
            ": " + std::strerror(errno));
    }

    std::string canonical = model_path;
    if (char *resolved = ::realpath(model_path.c_str(), nullptr)) {
        canonical = resolved;
        std::free(resolved);
    }

    std::string cache_root;
    if (const char *p = std::getenv("QW3_MODEL_DIGEST_CACHE_DIR")) {
        cache_root = p;
    } else if (const char *p = std::getenv("XDG_CACHE_HOME")) {
        cache_root = std::string(p) + "/qw3/model-digests";
    } else if (const char *p = std::getenv("HOME")) {
        cache_root = std::string(p) + "/.cache/qw3/model-digests";
    } else {
        cache_root = "/tmp/qw3-model-digests-" +
                     std::to_string(static_cast<unsigned long long>(::getuid()));
    }
    const std::string cache_path =
        cache_root + "/" + sha256_string(canonical) + ".json";

    auto stat_matches = [](const json &j, const struct stat &st,
                           const std::string &path) {
        return j.value("path", std::string()) == path &&
               j.value("size", 0ull) == static_cast<uint64_t>(st.st_size) &&
               j.value("dev", 0ull) == static_cast<uint64_t>(st.st_dev) &&
               j.value("ino", 0ull) == static_cast<uint64_t>(st.st_ino) &&
               j.value("mtime_sec", static_cast<int64_t>(-1)) ==
                   static_cast<int64_t>(st.st_mtim.tv_sec) &&
               j.value("mtime_nsec", static_cast<int64_t>(-1)) ==
                   static_cast<int64_t>(st.st_mtim.tv_nsec) &&
               j.value("ctime_sec", static_cast<int64_t>(-1)) ==
                   static_cast<int64_t>(st.st_ctim.tv_sec) &&
               j.value("ctime_nsec", static_cast<int64_t>(-1)) ==
                   static_cast<int64_t>(st.st_ctim.tv_nsec);
    };

    try {
        std::ifstream in(cache_path);
        if (in) {
            json cached;
            in >> cached;
            const std::string digest =
                cached.value("sha256", std::string());
            if (is_sha256_hex(digest) &&
                stat_matches(cached, before, canonical)) {
                return digest;
            }
        }
    } catch (...) {
        // A corrupt or stale cache is merely a miss; the full-file digest
        // remains authoritative.
    }

    const int fd = ::open(model_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error("cannot open GGUF for SHA-256: " +
                                 model_path + ": " + std::strerror(errno));
    }
#ifdef QW3_HAVE_OPENSSL
    EVP_MD_CTX *evp = EVP_MD_CTX_new();
    if (!evp || EVP_DigestInit_ex(evp, EVP_sha256(), nullptr) != 1) {
        if (evp) EVP_MD_CTX_free(evp);
        ::close(fd);
        throw std::runtime_error("failed to initialize accelerated SHA-256");
    }
#else
    Sha256 sha;
#endif
    std::vector<uint8_t> buffer(8u * 1024u * 1024u);
    try {
        for (;;) {
            const ssize_t n = ::read(fd, buffer.data(), buffer.size());
            if (n < 0 && errno == EINTR) continue;
            if (n < 0) {
                throw std::runtime_error("failed to hash GGUF: " + model_path +
                                         ": " + std::strerror(errno));
            }
            if (n == 0) break;
#ifdef QW3_HAVE_OPENSSL
            if (EVP_DigestUpdate(evp, buffer.data(),
                                 static_cast<size_t>(n)) != 1) {
                throw std::runtime_error(
                    "accelerated SHA-256 update failed for GGUF: " +
                    model_path);
            }
#else
            sha.update(buffer.data(), static_cast<size_t>(n));
#endif
        }
    } catch (...) {
#ifdef QW3_HAVE_OPENSSL
        EVP_MD_CTX_free(evp);
#endif
        ::close(fd);
        throw;
    }
    ::close(fd);

    struct stat after {};
    if (::stat(model_path.c_str(), &after) != 0 ||
        before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
        before.st_size != after.st_size ||
        before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
        before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
        before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
        before.st_ctim.tv_nsec != after.st_ctim.tv_nsec) {
#ifdef QW3_HAVE_OPENSSL
        EVP_MD_CTX_free(evp);
#endif
        throw std::runtime_error(
            "GGUF changed while computing KVMem archive identity: " +
            model_path);
    }
    std::string digest;
#ifdef QW3_HAVE_OPENSSL
    std::array<unsigned char, EVP_MAX_MD_SIZE> out{};
    unsigned int out_bytes = 0;
    if (EVP_DigestFinal_ex(evp, out.data(), &out_bytes) != 1 ||
        out_bytes != 32) {
        EVP_MD_CTX_free(evp);
        throw std::runtime_error("accelerated SHA-256 finalize failed");
    }
    EVP_MD_CTX_free(evp);
    std::ostringstream hex;
    hex << std::hex << std::setfill('0');
    for (uint32_t i = 0; i < out_bytes; ++i) {
        hex << std::setw(2) << static_cast<unsigned>(out[i]);
    }
    digest = hex.str();
#else
    digest = sha.finish_hex();
#endif

    try {
        ensure_dir(cache_root);
        json cached{
            {"path", canonical},
            {"size", static_cast<uint64_t>(after.st_size)},
            {"dev", static_cast<uint64_t>(after.st_dev)},
            {"ino", static_cast<uint64_t>(after.st_ino)},
            {"mtime_sec", static_cast<int64_t>(after.st_mtim.tv_sec)},
            {"mtime_nsec", static_cast<int64_t>(after.st_mtim.tv_nsec)},
            {"ctime_sec", static_cast<int64_t>(after.st_ctim.tv_sec)},
            {"ctime_nsec", static_cast<int64_t>(after.st_ctim.tv_nsec)},
            {"sha256", digest},
        };
        const std::string text = cached.dump(2);
        const std::string tmp = cache_path + ".tmp." +
                                std::to_string(static_cast<unsigned long long>(
                                    ::getpid()));
        write_file(tmp, text.data(), text.size());
        if (::rename(tmp.c_str(), cache_path.c_str()) != 0) {
            (void)::unlink(tmp.c_str());
        }
    } catch (...) {
        // Read-only HOME/model deployments are valid.  They pay the one-time
        // full scan again on the next process rather than failing inference.
    }
    return digest;
}

KvMemArchive::~KvMemArchive() {
    if (tokens_fd_ >= 0) ::close(tokens_fd_);
}

std::string KvMemArchive::raw_k_path() const {
    return dir_ + "/" + kvmem_archive_files::kRawK;
}

std::string KvMemArchive::v_path() const {
    return dir_ + "/" + kvmem_archive_files::kV;
}

std::string KvMemArchive::state_path(uint64_t position) const {
    return dir_ + "/" + kvmem_archive_files::kStateDir + "/" +
           std::to_string(position) + ".bin";
}

KvMemArchiveManifest KvMemArchive::read_manifest(const std::string &dir) {
    const std::string path = dir + "/" + kvmem_archive_files::kManifest;
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("KVMem archive manifest not found: " + path);
    }
    json j;
    in >> j;
    KvMemArchiveManifest m;
    m.format_version = j.value("format_version", 0u);
    if (m.format_version != 1 && m.format_version != 2 &&
        m.format_version != 3) {
        throw std::runtime_error(
            "unsupported KVMem archive format_version " +
            std::to_string(m.format_version) + " in " + path);
    }
    m.layout = layout_from_json(j.at("layout"));
    if (m.format_version >= 3 &&
        !is_sha256_hex(m.layout.model_sha256)) {
        throw std::runtime_error(
            "KVMem archive format v3 has a missing or invalid model_sha256 in " +
            path);
    }
    m.policy_snapshot = j.value("policy_snapshot", std::string());
    m.total_tokens = j.value("total_tokens", 0ull);
    m.total_blocks = j.value("total_blocks", 0u);
    m.raw_chunks = j.value("raw_chunks", 0u);
    m.sealed = j.value("sealed", false);
    m.created_at = j.value("created_at", static_cast<int64_t>(0));
    m.sealed_at = j.value("sealed_at", static_cast<int64_t>(0));
    if (j.contains("ladder")) {
        for (const auto &v : j.at("ladder")) m.ladder.push_back(v.get<uint64_t>());
    }
    std::sort(m.ladder.begin(), m.ladder.end());
    return m;
}

void KvMemArchive::write_manifest() {
    json j;
    j["format_version"] = manifest_.format_version;
    j["layout"] = layout_to_json(manifest_.layout);
    j["layout_key"] = manifest_.layout.key();
    j["policy_snapshot"] = manifest_.policy_snapshot;
    j["total_tokens"] = manifest_.total_tokens;
    j["total_blocks"] = manifest_.total_blocks;
    j["raw_chunks"] = manifest_.raw_chunks;
    j["ladder"] = manifest_.ladder;
    j["sealed"] = manifest_.sealed;
    j["created_at"] = manifest_.created_at;
    j["sealed_at"] = manifest_.sealed_at;
    const std::string text = j.dump(2);
    const std::string tmp = dir_ + "/" + kvmem_archive_files::kManifestTmp;
    write_file(tmp, text.data(), text.size());
    // Rename last so a reader never observes a partially written manifest.
    // The manifest is the archive's commit point.
    const std::string final_path = dir_ + "/" + kvmem_archive_files::kManifest;
    if (::rename(tmp.c_str(), final_path.c_str()) != 0) {
        throw std::runtime_error("failed to commit KVMem archive manifest: " +
                                 std::string(std::strerror(errno)));
    }
}

void KvMemArchive::write_bitmaps() {
    write_file(dir_ + "/" + kvmem_archive_files::kChunkValid,
               chunk_valid_.data(), chunk_valid_.size());
    write_file(dir_ + "/" + kvmem_archive_files::kBlockValid,
               block_valid_.data(), block_valid_.size());
}

void KvMemArchive::load_bitmaps() {
    chunk_valid_ = read_file(dir_ + "/" + kvmem_archive_files::kChunkValid);
    block_valid_ = read_file(dir_ + "/" + kvmem_archive_files::kBlockValid);
}

std::unique_ptr<KvMemArchive> KvMemArchive::open_for_build(
        const std::string &dir, const KvMemArchiveLayout &layout,
        const std::string &policy_snapshot) {
    ensure_dir(dir);
    ensure_dir(dir + "/" + kvmem_archive_files::kStateDir);
    std::unique_ptr<KvMemArchive> a(new KvMemArchive());
    a->dir_ = dir;
    a->writable_ = true;

    const std::string manifest_path =
        dir + "/" + kvmem_archive_files::kManifest;
    if (path_exists(manifest_path)) {
        KvMemArchiveManifest existing = read_manifest(dir);
        if (existing.sealed) {
            throw std::runtime_error(
                "KVMem archive is already sealed; build into a new directory: " +
                dir);
        }
        KvMemArchiveLayout compatible = layout;
        if (existing.format_version < 3 &&
            existing.layout.model_sha256.empty()) {
            // v1/v2 used file size as the model identity.  Preserve resumable
            // compatibility for those archives, but never remove the digest
            // from a newly-created v3 archive.
            compatible.model_sha256.clear();
        }
        const std::string why =
            existing.layout.explain_mismatch(compatible);
        if (!why.empty()) {
            throw std::runtime_error(
                "cannot resume KVMem archive build, layout differs: " + why);
        }
        a->manifest_ = std::move(existing);
        a->manifest_.policy_snapshot = policy_snapshot;
        a->load_bitmaps();
    } else {
        if (!is_sha256_hex(layout.model_sha256)) {
            throw std::runtime_error(
                "new KVMem archives require a complete GGUF SHA-256");
        }
        a->manifest_.format_version = 3;
        a->manifest_.layout = layout;
        a->manifest_.policy_snapshot = policy_snapshot;
        a->manifest_.created_at = unix_now();
        a->write_manifest();
    }

    const std::string tokens_path = dir + "/" + kvmem_archive_files::kTokens;
    a->tokens_fd_ = ::open(tokens_path.c_str(),
                           O_CREAT | O_RDWR | O_CLOEXEC, 0644);
    if (a->tokens_fd_ < 0) {
        throw std::runtime_error("failed to open KVMem archive token stream: " +
                                 tokens_path + ": " + std::strerror(errno));
    }
    a->token_count_ = file_size(tokens_path) / sizeof(uint32_t);
    return a;
}

std::unique_ptr<KvMemArchive> KvMemArchive::attach(const std::string &dir) {
    std::unique_ptr<KvMemArchive> a(new KvMemArchive());
    a->dir_ = dir;
    a->writable_ = false;
    a->manifest_ = read_manifest(dir);
    if (!a->manifest_.sealed) {
        throw std::runtime_error(
            "KVMem archive is not sealed and cannot be attached: " + dir);
    }
    a->load_bitmaps();
    const std::string tokens_path = dir + "/" + kvmem_archive_files::kTokens;
    a->tokens_fd_ = ::open(tokens_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (a->tokens_fd_ < 0) {
        throw std::runtime_error("failed to open KVMem archive token stream: " +
                                 tokens_path + ": " + std::strerror(errno));
    }
    a->token_count_ = file_size(tokens_path) / sizeof(uint32_t);
    if (a->token_count_ < a->manifest_.total_tokens) {
        throw std::runtime_error(
            "KVMem archive token stream is shorter than its manifest claims");
    }
    return a;
}

void KvMemArchive::append_tokens(const uint32_t *tokens, size_t count) {
    if (!writable_) {
        throw std::runtime_error("KVMem archive is attached read-only");
    }
    if (count == 0) return;
    write_all(tokens_fd_, tokens, count * sizeof(uint32_t),
              token_count_ * sizeof(uint32_t));
    token_count_ += count;
}

void KvMemArchive::truncate_tokens(uint64_t count) {
    if (!writable_) {
        throw std::runtime_error("KVMem archive is attached read-only");
    }
    if (count > token_count_) {
        throw std::runtime_error(
            "cannot extend KVMem archive token stream by truncation");
    }
    const uint64_t bytes = count * sizeof(uint32_t);
    if (bytes > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
        ::ftruncate(tokens_fd_, static_cast<off_t>(bytes)) != 0) {
        throw std::runtime_error(
            "failed to truncate KVMem archive token stream: " +
            std::string(std::strerror(errno)));
    }
    token_count_ = count;
}

std::vector<uint32_t> KvMemArchive::read_tokens(uint64_t offset,
                                                uint64_t count) const {
    if (offset + count > token_count_) {
        throw std::runtime_error("KVMem archive token read out of range");
    }
    std::vector<uint32_t> out(static_cast<size_t>(count));
    if (count == 0) return out;
    read_all(tokens_fd_, out.data(), count * sizeof(uint32_t),
             offset * sizeof(uint32_t));
    return out;
}

void KvMemArchive::reserve_chunks(uint32_t chunks) {
    if (chunk_valid_.size() < chunks) chunk_valid_.resize(chunks, 0);
}

void KvMemArchive::reserve_blocks(uint32_t blocks) {
    if (block_valid_.size() < blocks) block_valid_.resize(blocks, 0);
}

void KvMemArchive::mark_chunk_valid(uint32_t chunk) {
    reserve_chunks(chunk + 1);
    chunk_valid_[chunk] = 1;
}

void KvMemArchive::mark_block_valid(uint32_t block) {
    reserve_blocks(block + 1);
    block_valid_[block] = 1;
}

bool KvMemArchive::chunk_valid(uint32_t chunk) const {
    return chunk < chunk_valid_.size() && chunk_valid_[chunk] != 0;
}

bool KvMemArchive::block_valid(uint32_t block) const {
    return block < block_valid_.size() && block_valid_[block] != 0;
}

void KvMemArchive::write_state(const KvMemArchiveState &state) {
    if (!writable_) {
        throw std::runtime_error("KVMem archive is attached read-only");
    }
    const uint64_t expect = kvmem_archive_state_bytes(state.entries);
    if (expect != state.payload.size()) {
        throw std::runtime_error(
            "KVMem archive state payload does not match its entry table");
    }
    std::vector<uint8_t> buffer;
    const uint64_t header_bytes = 4 * sizeof(uint32_t) + 2 * sizeof(uint64_t);
    const uint64_t table_bytes =
        state.entries.size() * (2 * sizeof(uint32_t) + sizeof(uint64_t));
    buffer.resize(static_cast<size_t>(header_bytes + table_bytes));
    uint8_t *p = buffer.data();
    auto put_u32 = [&p](uint32_t v) {
        std::memcpy(p, &v, sizeof(v));
        p += sizeof(v);
    };
    auto put_u64 = [&p](uint64_t v) {
        std::memcpy(p, &v, sizeof(v));
        p += sizeof(v);
    };
    put_u32(kStateMagic);
    put_u32(kStateVersion);
    put_u32(state.mtp_prefix_len);
    put_u32(static_cast<uint32_t>(state.entries.size()));
    put_u64(state.position);
    put_u64(state.kvmem_registered_pos);
    for (const KvMemArchiveStateEntry &e : state.entries) {
        put_u32(e.kind);
        put_u32(e.index);
        put_u64(e.bytes);
    }

    const std::string path = state_path(state.position);
    const std::string tmp = path + ".tmp";
    const int fd =
        ::open(tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
    if (fd < 0) {
        throw std::runtime_error("failed to open KVMem archive state file: " +
                                 tmp + ": " + std::strerror(errno));
    }
    try {
        write_all(fd, buffer.data(), buffer.size(), 0);
        if (!state.payload.empty()) {
            write_all(fd, state.payload.data(), state.payload.size(),
                      buffer.size());
        }
    } catch (...) {
        ::close(fd);
        throw;
    }
    ::close(fd);
    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        throw std::runtime_error("failed to commit KVMem archive state file: " +
                                 std::string(std::strerror(errno)));
    }
    if (std::find(manifest_.ladder.begin(), manifest_.ladder.end(),
                  state.position) == manifest_.ladder.end()) {
        manifest_.ladder.push_back(state.position);
        std::sort(manifest_.ladder.begin(), manifest_.ladder.end());
    }
}

KvMemArchiveState KvMemArchive::read_state(uint64_t position) const {
    const std::string path = state_path(position);
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error("KVMem archive state file not found: " + path);
    }
    KvMemArchiveState state;
    try {
        const uint64_t header_bytes =
            4 * sizeof(uint32_t) + 2 * sizeof(uint64_t);
        std::vector<uint8_t> header(static_cast<size_t>(header_bytes));
        read_all(fd, header.data(), header_bytes, 0);
        const uint8_t *p = header.data();
        auto get_u32 = [&p]() {
            uint32_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            p += sizeof(v);
            return v;
        };
        auto get_u64 = [&p]() {
            uint64_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            p += sizeof(v);
            return v;
        };
        if (get_u32() != kStateMagic) {
            throw std::runtime_error("bad KVMem archive state magic: " + path);
        }
        if (get_u32() != kStateVersion) {
            throw std::runtime_error("unsupported KVMem archive state version: " +
                                     path);
        }
        state.mtp_prefix_len = get_u32();
        const uint32_t entry_count = get_u32();
        state.position = get_u64();
        state.kvmem_registered_pos = get_u64();

        const uint64_t table_bytes =
            static_cast<uint64_t>(entry_count) *
            (2 * sizeof(uint32_t) + sizeof(uint64_t));
        std::vector<uint8_t> table(static_cast<size_t>(table_bytes));
        if (table_bytes > 0) read_all(fd, table.data(), table_bytes, header_bytes);
        p = table.data();
        state.entries.resize(entry_count);
        for (uint32_t i = 0; i < entry_count; ++i) {
            state.entries[i].kind = get_u32();
            state.entries[i].index = get_u32();
            state.entries[i].bytes = get_u64();
        }
        const uint64_t payload_bytes =
            kvmem_archive_state_bytes(state.entries);
        state.payload.resize(static_cast<size_t>(payload_bytes));
        if (payload_bytes > 0) {
            read_all(fd, state.payload.data(), payload_bytes,
                     header_bytes + table_bytes);
        }
    } catch (...) {
        ::close(fd);
        throw;
    }
    ::close(fd);
    return state;
}

uint64_t KvMemArchive::ladder_at_or_below(uint64_t position) const {
    uint64_t best = UINT64_MAX;
    for (uint64_t p : manifest_.ladder) {
        if (p <= position) best = p;
    }
    return best;
}

uint64_t KvMemArchive::resume_position() const {
    const uint32_t bt = manifest_.layout.block_tokens;
    const uint32_t ct = manifest_.layout.raw_chunk_tokens;
    if (bt == 0 || ct == 0) return 0;
    uint64_t best = 0;
    for (uint64_t p : manifest_.ladder) {
        if (p == 0) continue;
        const uint32_t chunks = static_cast<uint32_t>((p + ct - 1) / ct);
        const uint32_t blocks = static_cast<uint32_t>((p + bt - 1) / bt);
        bool ok = true;
        for (uint32_t c = 0; c < chunks && ok; ++c) ok = chunk_valid(c);
        for (uint32_t b = 0; b < blocks && ok; ++b) ok = block_valid(b);
        if (ok && p > best) best = p;
    }
    return best;
}

void KvMemArchive::checkpoint_metadata(uint64_t total_tokens,
                                       uint32_t total_blocks,
                                       uint32_t raw_chunks) {
    if (!writable_) return;
    manifest_.total_tokens = total_tokens;
    manifest_.total_blocks = total_blocks;
    manifest_.raw_chunks = raw_chunks;
    write_bitmaps();
    write_manifest();
}

void KvMemArchive::seal(uint64_t total_tokens, uint32_t total_blocks,
                        uint32_t raw_chunks) {
    if (!writable_) {
        throw std::runtime_error("KVMem archive is attached read-only");
    }
    if (token_count_ < total_tokens) {
        throw std::runtime_error(
            "cannot seal KVMem archive: token stream holds " +
            std::to_string(token_count_) + " of " +
            std::to_string(total_tokens) + " tokens");
    }
    const uint32_t bt = manifest_.layout.block_tokens;
    const uint32_t ct = manifest_.layout.raw_chunk_tokens;
    const uint32_t need_chunks =
        ct == 0 ? 0 : static_cast<uint32_t>((total_tokens + ct - 1) / ct);
    const uint32_t need_blocks =
        bt == 0 ? 0 : static_cast<uint32_t>((total_tokens + bt - 1) / bt);
    for (uint32_t c = 0; c < need_chunks; ++c) {
        if (!chunk_valid(c)) {
            throw std::runtime_error(
                "cannot seal KVMem archive: raw-K chunk " + std::to_string(c) +
                " was never persisted");
        }
    }
    for (uint32_t b = 0; b < need_blocks; ++b) {
        if (!block_valid(b)) {
            throw std::runtime_error(
                "cannot seal KVMem archive: V block " + std::to_string(b) +
                " was never persisted");
        }
    }
    manifest_.total_tokens = total_tokens;
    manifest_.total_blocks = total_blocks;
    manifest_.raw_chunks = raw_chunks == 0 ? need_chunks : raw_chunks;
    manifest_.sealed = true;
    manifest_.sealed_at = unix_now();
    write_bitmaps();
    write_manifest();
}

} // namespace qw3
