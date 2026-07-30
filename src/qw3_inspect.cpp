#include "qw3/gguf.hpp"
#include "qw3/qw3.hpp"
#include "qw3/tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_hex(const std::string &text) {
    static constexpr char kHex[] = "0123456789abcdef";
    for (const unsigned char byte : text) {
        std::cout << kHex[byte >> 4] << kHex[byte & 0x0f];
    }
}

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
    bool tokenize_stdin = false;
    bool count_tokens_stdin = false;
    bool tokenize_pieces_stdin = false;
    std::string path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--meta" || a == "--dump-metadata") {
            dump_meta = true;
        } else if (a == "--tokenize-stdin") {
            tokenize_stdin = true;
        } else if (a == "--count-tokens-stdin") {
            count_tokens_stdin = true;
        } else if (a == "--tokenize-pieces-stdin") {
            tokenize_pieces_stdin = true;
        } else {
            path = a;
        }
    }
    if (path.empty()) {
        std::cerr << "Usage: qw3-inspect "
                     "[--meta|--tokenize-stdin|--count-tokens-stdin|"
                     "--tokenize-pieces-stdin] "
                     "MODEL.gguf\n";
        return 2;
    }
    try {
        if (tokenize_stdin || count_tokens_stdin || tokenize_pieces_stdin) {
            const qw3::GgufFile gguf(path);
            const qw3::QwenTokenizer tokenizer(gguf);
            const std::string input((std::istreambuf_iterator<char>(std::cin)),
                                    std::istreambuf_iterator<char>());
            const std::vector<int32_t> ids = tokenizer.encode(input);
            if (count_tokens_stdin) {
                std::cout << ids.size() << '\n';
                return 0;
            }
            if (tokenize_pieces_stdin) {
                size_t byte_offset = 0;
                for (size_t i = 0; i < ids.size(); ++i) {
                    const std::string piece = tokenizer.decode_one(ids[i]);
                    std::cout << i << '\t' << ids[i] << '\t'
                              << byte_offset << '\t'
                              << byte_offset + piece.size() << '\t';
                    print_hex(piece);
                    std::cout << '\n';
                    byte_offset += piece.size();
                }
                if (byte_offset != input.size()) {
                    std::cerr << "qw3-inspect: decoded token pieces cover "
                              << byte_offset << " bytes, input has "
                              << input.size() << " bytes\n";
                    return 1;
                }
                return 0;
            }
            std::cout << '[';
            for (size_t i = 0; i < ids.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << ids[i];
            }
            std::cout << "]\n";
            return 0;
        }
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
                  << "nextn_predict_layers: " << info.nextn_predict_layers << "\n"
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
