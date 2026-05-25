#pragma once

#include "qw3/qw3.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace qw3 {

enum class GgufValueType : uint32_t {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
};

struct GgufValue {
    GgufValueType type{};
    std::string string_value;
    std::vector<std::string> string_array;
    uint64_t unsigned_value = 0;
    int64_t signed_value = 0;
    double float_value = 0.0;
    bool bool_value = false;
};

struct GgufTensorInfo {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type = 0;
    uint64_t rel_offset = 0;
    uint64_t abs_offset = 0;
    uint64_t elements = 0;
    uint64_t bytes = 0;
};

class GgufFile {
public:
    explicit GgufFile(const std::string &path);
    ~GgufFile();

    GgufFile(const GgufFile &) = delete;
    GgufFile &operator=(const GgufFile &) = delete;

    const std::string &path() const;
    uint64_t size() const;
    const uint8_t *data() const;
    uint32_t version() const;
    uint32_t alignment() const;
    uint64_t tensor_data_offset() const;

    const std::unordered_map<std::string, GgufValue> &metadata() const;
    const std::vector<GgufTensorInfo> &tensors() const;
    const GgufTensorInfo *find_tensor(const std::string &name) const;
    std::string token_text(uint32_t token_id) const;
    ModelInfo model_info() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::string gguf_tensor_type_name(uint32_t type);
uint64_t gguf_tensor_type_size(uint32_t type, uint64_t elements);

} // namespace qw3
