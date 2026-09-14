#include "input_embedding_override_index.hpp"

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using qw3::detail::InputEmbeddingOverrideIndex;

static void check(bool ok) {
    if (!ok) throw std::runtime_error("visual override index differs from reference");
}

static void compare(InputEmbeddingOverrideIndex &index,
                    const std::vector<uint32_t> &tokens) {
    index.clear();
    std::unordered_map<uint32_t, size_t> reference;
    const auto lookup = [&](uint32_t token) {
        const auto it = reference.find(token);
        check(index.find(token) == (it == reference.end()
            ? InputEmbeddingOverrideIndex::missing : it->second));
    };
    lookup(0x80000001u);
    for (uint32_t token : tokens) {
        const bool expected = reference.emplace(token, reference.size()).second;
        check(index.append(token) == expected);
        lookup(token);
        lookup(token - 1);
        lookup(token + 1);
        lookup(0);
        lookup(0x80000000u);
        lookup(0xffffffffu);
    }
    for (const auto &entry : reference) lookup(entry.first);
}

int main() {
    InputEmbeddingOverrideIndex index;
    compare(index, {});
    compare(index, {0x80000001u, 0x80000002u, 0x80000003u});
    // Duplicates at the dense/sparse transition must not overwrite row 0.
    compare(index, {0x80000001u, 0x80000002u, 0x80000001u, 0x80000003u});
    compare(index, {0xffffffffu, 0x80000000u, 0xfffffffeu, 0xffffffffu});
    compare(index, {0x80000003u, 0x80000001u, 0x80000002u});
    compare(index, {0, 1, 0x7fffffffu, 0x80000000u, 0x80000001u});

    std::vector<uint32_t> dense(8192);
    for (size_t i = 0; i < dense.size(); ++i) dense[i] = 0x80000001u + i;
    compare(index, dense);
    // A long valid prefix followed by a gap or duplicate exercises migration.
    dense.push_back(0xffffffffu);
    dense.push_back(0x80000001u);
    compare(index, dense);
    // Clear must remove the sparse mapping before returning to dense inputs.
    compare(index, {0x80000001u});
    check(index.find(0xffffffffu) == InputEmbeddingOverrideIndex::missing);

    std::mt19937 random(20260909);
    for (size_t trial = 0; trial < 64; ++trial) {
        std::vector<uint32_t> tokens;
        for (size_t i = 0; i < 2048; ++i) {
            tokens.push_back(i < trial * 8
                ? 0x80000001u + i : 0x80000000u + random() % 3072);
        }
        compare(index, tokens);
        std::shuffle(tokens.begin(), tokens.end(), random);
        compare(index, tokens);
    }
    std::cout << "PASS: dense, sparse, reordered, duplicate, boundary and reset parity\n";
}
