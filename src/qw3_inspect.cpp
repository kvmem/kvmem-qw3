#include "qw3/gguf.hpp"
#include "qw3/qw3.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_value(const qw3::GgufValue &v) {
    using T = qw3::GgufValueType;
    switch (v.type) {
    case T::String:
        std::cout << '"' << v.string_value << '"';
        break;
    case T::Bool:
        std::cout << (v.bool_value ? "true" : "false");
        break;
    case T::Float32:
    case T::Float64:
        std::cout << v.float_value;
        break;
    case T::Uint8:
    case T::Uint16:
    case T::Uint32:
    case T::Uint64:
        std::cout << v.unsigned_value;
        break;
    case T::Int8:
    case T::Int16:
    case T::Int32:
    case T::Int64:
        std::cout << v.signed_value;
        break;
    case T::Array:
        std::cout << "[array len=" << v.string_array.size();
        if (!v.string_array.empty()) {
            std::cout << " sample=";
            for (size_t i = 0; i < std::min<size_t>(v.string_array.size(), 3); ++i) {
                if (i) std::cout << ',';
                std::cout << '"' << v.string_array[i] << '"';
            }
        }
        std::cout << ']';
        break;
    }
}

} // namespace

int main(int argc, char **argv) {
    bool dump_meta = false;
    std::string path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--meta" || a == "--dump-metadata") {
            dump_meta = true;
        } else {
            path = a;
        }
    }
    if (path.empty()) {
        std::cerr << "Usage: qw3-inspect [--meta] MODEL.gguf\n";
        return 2;
    }
    try {
        if (dump_meta) {
            const qw3::GgufFile gguf(path);
            std::vector<std::string> keys;
            keys.reserve(gguf.metadata().size());
            for (const auto &kv : gguf.metadata()) keys.push_back(kv.first);
            std::sort(keys.begin(), keys.end());
            for (const std::string &k : keys) {
                std::cout << k << " = ";
                print_value(gguf.metadata().at(k));
                std::cout << "\n";
            }
            return 0;
        }
        const qw3::ModelInfo info = qw3::inspect_gguf(path);
        std::cout << "architecture: " << info.architecture << "\n"
                  << "metadata: " << info.metadata_count << "\n"
                  << "tensors: " << info.tensor_count << "\n"
                  << "blocks: " << info.block_count << "\n"
                  << "embedding: " << info.embedding_length << "\n"
                  << "heads: " << info.head_count << "\n"
                  << "kv_heads: " << info.head_count_kv << "\n"
                  << "context: " << info.context_length << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "qw3-inspect: " << e.what() << "\n";
        return 1;
    }
}
