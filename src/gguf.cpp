#include "qw3/gguf.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace qw3 {
namespace {

constexpr uint32_t kGgufMagic = 0x46554747u;

uint64_t align_up(uint64_t v, uint64_t alignment) {
    if (alignment == 0) return v;
    const uint64_t rem = v % alignment;
    return rem == 0 ? v : v + (alignment - rem);
}

template <typename T>
T read_unaligned(const uint8_t *p) {
    T value{};
    std::memcpy(&value, p, sizeof(T));
    return value;
}

class Cursor {
public:
    Cursor(const uint8_t *data, uint64_t size) : data_(data), size_(size) {}

    template <typename T>
    T read_scalar() {
        require(sizeof(T));
        T value = read_unaligned<T>(data_ + pos_);
        pos_ += sizeof(T);
        return value;
    }

    std::string read_string() {
        const uint64_t len = read_scalar<uint64_t>();
        if (len > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("GGUF string is too large");
        }
        require(len);
        std::string s(reinterpret_cast<const char *>(data_ + pos_), static_cast<size_t>(len));
        pos_ += len;
        return s;
    }

    void skip(uint64_t n) {
        require(n);
        pos_ += n;
    }

    uint64_t pos() const {
        return pos_;
    }

private:
    void require(uint64_t n) const {
        if (n > size_ - pos_) throw std::runtime_error("truncated GGUF file");
    }

    const uint8_t *data_ = nullptr;
    uint64_t size_ = 0;
    uint64_t pos_ = 0;
};

uint64_t scalar_size(GgufValueType type) {
    switch (type) {
    case GgufValueType::Uint8:
    case GgufValueType::Int8:
    case GgufValueType::Bool:
        return 1;
    case GgufValueType::Uint16:
    case GgufValueType::Int16:
        return 2;
    case GgufValueType::Uint32:
    case GgufValueType::Int32:
    case GgufValueType::Float32:
        return 4;
    case GgufValueType::Uint64:
    case GgufValueType::Int64:
    case GgufValueType::Float64:
        return 8;
    case GgufValueType::String:
    case GgufValueType::Array:
        return 0;
    }
    throw std::runtime_error("unknown GGUF metadata type");
}

void skip_array(Cursor &c);

void skip_array_payload(Cursor &c, GgufValueType elem_type, uint64_t count) {
    if (elem_type == GgufValueType::String) {
        for (uint64_t i = 0; i < count; ++i) (void)c.read_string();
        return;
    }
    if (elem_type == GgufValueType::Array) {
        for (uint64_t i = 0; i < count; ++i) skip_array(c);
        return;
    }
    const uint64_t elem_size = scalar_size(elem_type);
    if (count > std::numeric_limits<uint64_t>::max() / elem_size) {
        throw std::runtime_error("GGUF array size overflow");
    }
    c.skip(count * elem_size);
}

void skip_array(Cursor &c) {
    const auto elem_type = static_cast<GgufValueType>(c.read_scalar<uint32_t>());
    const uint64_t count = c.read_scalar<uint64_t>();
    skip_array_payload(c, elem_type, count);
}

GgufValue read_value(Cursor &c, GgufValueType type) {
    GgufValue v;
    v.type = type;
    switch (type) {
    case GgufValueType::Uint8:
        v.unsigned_value = c.read_scalar<uint8_t>();
        break;
    case GgufValueType::Uint16:
        v.unsigned_value = c.read_scalar<uint16_t>();
        break;
    case GgufValueType::Uint32:
        v.unsigned_value = c.read_scalar<uint32_t>();
        break;
    case GgufValueType::Uint64:
        v.unsigned_value = c.read_scalar<uint64_t>();
        break;
    case GgufValueType::Int8:
        v.signed_value = c.read_scalar<int8_t>();
        break;
    case GgufValueType::Int16:
        v.signed_value = c.read_scalar<int16_t>();
        break;
    case GgufValueType::Int32:
        v.signed_value = c.read_scalar<int32_t>();
        break;
    case GgufValueType::Int64:
        v.signed_value = c.read_scalar<int64_t>();
        break;
    case GgufValueType::Float32:
        v.float_value = c.read_scalar<float>();
        break;
    case GgufValueType::Float64:
        v.float_value = c.read_scalar<double>();
        break;
    case GgufValueType::Bool:
        v.bool_value = c.read_scalar<uint8_t>() != 0;
        break;
    case GgufValueType::String:
        v.string_value = c.read_string();
        break;
    case GgufValueType::Array:
        {
            const auto elem_type = static_cast<GgufValueType>(c.read_scalar<uint32_t>());
            const uint64_t count = c.read_scalar<uint64_t>();
            if (elem_type == GgufValueType::String) {
                v.string_array.reserve(static_cast<size_t>(count));
                for (uint64_t i = 0; i < count; ++i) v.string_array.push_back(c.read_string());
            } else {
                skip_array_payload(c, elem_type, count);
            }
        }
        break;
    }
    return v;
}

uint32_t get_u32(const std::unordered_map<std::string, GgufValue> &kv, const std::string &key) {
    const auto it = kv.find(key);
    if (it == kv.end()) return 0;
    if (it->second.type == GgufValueType::Uint32 || it->second.type == GgufValueType::Uint64 ||
        it->second.type == GgufValueType::Uint16 || it->second.type == GgufValueType::Uint8) {
        return static_cast<uint32_t>(it->second.unsigned_value);
    }
    return static_cast<uint32_t>(std::max<int64_t>(0, it->second.signed_value));
}

uint32_t get_alignment(const std::unordered_map<std::string, GgufValue> &kv) {
    const auto it = kv.find("general.alignment");
    if (it == kv.end()) return 32;
    return static_cast<uint32_t>(it->second.unsigned_value);
}

uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::runtime_error("GGUF tensor size overflow");
    }
    return a * b;
}

} // namespace

struct GgufFile::Impl {
    std::string path;
    int fd = -1;
    uint64_t size = 0;
    const uint8_t *data = nullptr;
    uint32_t version = 0;
    uint32_t alignment = 32;
    uint64_t tensor_data_offset = 0;
    std::unordered_map<std::string, GgufValue> metadata;
    std::vector<GgufTensorInfo> tensors;
    std::unordered_map<std::string, size_t> tensor_index;
};

GgufFile::GgufFile(const std::string &path) : impl_(std::make_unique<Impl>()) {
    impl_->path = path;
    impl_->fd = open(path.c_str(), O_RDONLY);
    if (impl_->fd < 0) {
        throw std::runtime_error("failed to open GGUF: " + path + ": " + std::strerror(errno));
    }

    struct stat st {};
    if (fstat(impl_->fd, &st) != 0) {
        throw std::runtime_error("failed to stat GGUF: " + path + ": " + std::strerror(errno));
    }
    if (st.st_size <= 0) throw std::runtime_error("empty GGUF: " + path);
    impl_->size = static_cast<uint64_t>(st.st_size);
    void *mapped = mmap(nullptr, static_cast<size_t>(impl_->size), PROT_READ, MAP_PRIVATE, impl_->fd, 0);
    if (mapped == MAP_FAILED) {
        throw std::runtime_error("failed to mmap GGUF: " + path + ": " + std::strerror(errno));
    }
    impl_->data = static_cast<const uint8_t *>(mapped);

    Cursor c(impl_->data, impl_->size);
    const uint32_t magic = c.read_scalar<uint32_t>();
    if (magic != kGgufMagic) throw std::runtime_error("not a GGUF file: " + path);
    impl_->version = c.read_scalar<uint32_t>();
    if (impl_->version != 3) throw std::runtime_error("only GGUF v3 is supported");
    const uint64_t n_tensors = c.read_scalar<uint64_t>();
    const uint64_t n_metadata = c.read_scalar<uint64_t>();

    impl_->metadata.reserve(static_cast<size_t>(n_metadata));
    for (uint64_t i = 0; i < n_metadata; ++i) {
        const std::string key = c.read_string();
        const auto type = static_cast<GgufValueType>(c.read_scalar<uint32_t>());
        impl_->metadata.emplace(key, read_value(c, type));
    }
    impl_->alignment = get_alignment(impl_->metadata);

    impl_->tensors.reserve(static_cast<size_t>(n_tensors));
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensorInfo t;
        t.name = c.read_string();
        const uint32_t n_dims = c.read_scalar<uint32_t>();
        if (n_dims > 8) throw std::runtime_error("unsupported GGUF tensor rank");
        t.dims.reserve(n_dims);
        t.elements = 1;
        for (uint32_t d = 0; d < n_dims; ++d) {
            const uint64_t dim = c.read_scalar<uint64_t>();
            t.dims.push_back(dim);
            t.elements = checked_mul(t.elements, dim);
        }
        t.type = c.read_scalar<uint32_t>();
        t.rel_offset = c.read_scalar<uint64_t>();
        t.bytes = gguf_tensor_type_size(t.type, t.elements);
        impl_->tensor_index.emplace(t.name, impl_->tensors.size());
        impl_->tensors.push_back(std::move(t));
    }

    impl_->tensor_data_offset = align_up(c.pos(), impl_->alignment);
    for (GgufTensorInfo &t : impl_->tensors) {
        t.abs_offset = impl_->tensor_data_offset + t.rel_offset;
        if (t.abs_offset > impl_->size || t.bytes > impl_->size - t.abs_offset) {
            throw std::runtime_error("GGUF tensor points outside file: " + t.name);
        }
    }
}

GgufFile::~GgufFile() {
    if (impl_) {
        if (impl_->data) munmap(const_cast<uint8_t *>(impl_->data), static_cast<size_t>(impl_->size));
        if (impl_->fd >= 0) close(impl_->fd);
    }
}

const std::string &GgufFile::path() const { return impl_->path; }
uint64_t GgufFile::size() const { return impl_->size; }
const uint8_t *GgufFile::data() const { return impl_->data; }
uint32_t GgufFile::version() const { return impl_->version; }
uint32_t GgufFile::alignment() const { return impl_->alignment; }
uint64_t GgufFile::tensor_data_offset() const { return impl_->tensor_data_offset; }
const std::unordered_map<std::string, GgufValue> &GgufFile::metadata() const { return impl_->metadata; }
const std::vector<GgufTensorInfo> &GgufFile::tensors() const { return impl_->tensors; }

const GgufTensorInfo *GgufFile::find_tensor(const std::string &name) const {
    const auto it = impl_->tensor_index.find(name);
    if (it == impl_->tensor_index.end()) return nullptr;
    return &impl_->tensors[it->second];
}

std::string GgufFile::token_text(uint32_t token_id) const {
    const auto it = impl_->metadata.find("tokenizer.ggml.tokens");
    if (it == impl_->metadata.end() || token_id >= it->second.string_array.size()) return {};
    return it->second.string_array[token_id];
}

ModelInfo GgufFile::model_info() const {
    ModelInfo info;
    info.tensor_count = impl_->tensors.size();
    info.metadata_count = impl_->metadata.size();
    const auto arch_it = impl_->metadata.find("general.architecture");
    if (arch_it != impl_->metadata.end()) info.architecture = arch_it->second.string_value;
    const std::string prefix = info.architecture.empty() ? std::string{} : info.architecture + ".";
    info.block_count = get_u32(impl_->metadata, prefix + "block_count");
    info.nextn_predict_layers = get_u32(impl_->metadata, prefix + "nextn_predict_layers");
    info.embedding_length = get_u32(impl_->metadata, prefix + "embedding_length");
    info.head_count = get_u32(impl_->metadata, prefix + "attention.head_count");
    info.head_count_kv = get_u32(impl_->metadata, prefix + "attention.head_count_kv");
    info.context_length = get_u32(impl_->metadata, prefix + "context_length");
    return info;
}

std::string gguf_tensor_type_name(uint32_t type) {
    switch (type) {
    case 0: return "F32";
    case 1: return "F16";
    case 2: return "Q4_0";
    case 3: return "Q4_1";
    case 6: return "Q5_0";
    case 7: return "Q5_1";
    case 8: return "Q8_0";
    case 9: return "Q8_1";
    case 10: return "Q2_K";
    case 11: return "Q3_K";
    case 12: return "Q4_K";
    case 13: return "Q5_K";
    case 14: return "Q6_K";
    case 15: return "Q8_K";
    case 16: return "IQ2_XXS";
    case 17: return "IQ2_XS";
    case 18: return "IQ3_XXS";
    case 19: return "IQ1_S";
    case 20: return "IQ4_NL";
    case 21: return "IQ3_S";
    case 22: return "IQ2_S";
    case 23: return "IQ4_XS";
    case 24: return "I8";
    case 25: return "I16";
    case 26: return "I32";
    case 27: return "I64";
    case 28: return "F64";
    case 29: return "IQ1_M";
    case 30: return "BF16";
    case 31: return "TQ1_0";
    case 32: return "TQ2_0";
    default: return "UNKNOWN";
    }
}

uint64_t gguf_tensor_type_size(uint32_t type, uint64_t elements) {
    auto block = [&](uint64_t block_size, uint64_t type_size) -> uint64_t {
        if (elements % block_size != 0) {
            throw std::runtime_error("quantized GGUF tensor elements are not block aligned");
        }
        return checked_mul(elements / block_size, type_size);
    };
    switch (type) {
    case 0: return checked_mul(elements, 4);
    case 1: return checked_mul(elements, 2);
    case 2: return block(32, 18);
    case 3: return block(32, 20);
    case 6: return block(32, 22);
    case 7: return block(32, 24);
    case 8: return block(32, 34);
    case 9: return block(32, 36);
    case 10: return block(256, 84);
    case 11: return block(256, 110);
    case 12: return block(256, 144);
    case 13: return block(256, 176);
    case 14: return block(256, 210);
    case 15: return block(256, 292);
    case 24: return elements;
    case 25: return checked_mul(elements, 2);
    case 26: return checked_mul(elements, 4);
    case 27: return checked_mul(elements, 8);
    case 28: return checked_mul(elements, 8);
    case 30: return checked_mul(elements, 2);
    default:
        return 0;
    }
}

ModelInfo inspect_gguf(const std::string &path) {
    return GgufFile(path).model_info();
}

} // namespace qw3
