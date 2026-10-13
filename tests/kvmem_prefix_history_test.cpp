#include "kvmem_prefix_history.hpp"

#include <cstdlib>
#include <iostream>
#include <numeric>

using namespace qw3::detail;

static void require(bool ok, const char *message) {
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}

int main() {
    Sha256 hash;
    hash.update("abc", 3);
    require(hash.finish_hex() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "shared streaming SHA-256 changed");
    Sha256 million;
    const std::string part(1000, 'a');
    for (unsigned i = 0; i < 1000; ++i) million.update(part.data(), part.size());
    require(million.finish_hex() == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
            "SHA-256 multi-block/length padding changed");

    KvMemPrefixHistory history;
    std::vector<uint32_t> tokens;
    const std::vector<uint32_t> no_output;
    const std::vector<MultimodalTokenIdentity> text;
    constexpr size_t keep = 96;
    for (uint32_t round = 0; round < 1000; ++round) {
        const auto previous = tokens.size();
        for (uint32_t i = 0; i < 73; ++i) tokens.push_back(round * 73 + i);
        const auto proof = history.match(tokens, text, 0);
        require(proof.common_prefix == previous, "regular append lost a reusable prefix");
        history.assign(tokens, no_output, tokens.size(), text, 0, keep, &proof);
        require(history.size() == tokens.size() && history.retained_tokens() <= keep,
                "prefix history grew past its fixed tail bound");
        require(history.capacity_bytes() <= keep * 48, "allocation grew with cumulative history");
        if (round % 13 == 0) {
            tokens[tokens.size() - 5] ^= 1;
            const auto changed = history.match(tokens, text, 0);
            require(changed.common_prefix == tokens.size() - 5 && !changed.outside_retained_tail,
                    "five-token tail rewrite was not located exactly");
            history.assign(tokens, no_output, tokens.size(), text, 0, keep, &changed);
        }
    }
    auto old_change = tokens;
    old_change[5] ^= 1;
    const auto old_match = history.match(old_change, text, 0);
    require(old_match.common_prefix == 0 && old_match.outside_retained_tail,
            "changed discarded prefix was accepted");
    old_change.resize(12);
    require(history.match(old_change, text, 0).outside_retained_tail,
            "shortened discarded prefix was accepted");

    // A committed decode prefix excludes the final unforwarded emitted token.
    auto proof = history.match(tokens, text, 0);
    const std::vector<uint32_t> emitted{1, 2, 3, 4};
    history.assign(tokens, emitted, tokens.size() + 3, text, 0, keep, &proof);
    auto next = tokens;
    next.insert(next.end(), emitted.begin(), emitted.end());
    require(history.match(next, text, 0).common_prefix == next.size() - 1,
            "unforwarded output entered checkpoint authority");
    const auto before_error = history.size();
    try { history.assign(tokens, emitted, next.size() + 1, text, 0, keep); require(false, "bad capture accepted"); }
    catch (const std::invalid_argument &) {}
    require(history.size() == before_error, "bad capture destroyed previous authority");
    history.clear();
    history.assign(tokens, no_output, tokens.size(), text, 0, keep, &proof);
    require(history.match(tokens, text, 0).common_prefix == tokens.size(),
            "stale proof after clear corrupted hash authority");

    // Synthetic IDs describe layout; identical IDs must not hide image edits.
    tokens.resize(200);
    std::iota(tokens.begin(), tokens.end(), 0);
    std::vector<MultimodalTokenIdentity> visuals{
        {0x80000001u, 100, {1, 2, 3}, 12345},
        {0x80000002u, 100, {4, 5, 6}, 54321}};
    tokens[10] = visuals[0].token_id;
    tokens[190] = visuals[1].token_id;
    history.assign(tokens, no_output, tokens.size(), visuals, 77, keep);
    require(history.match(tokens, visuals, 99).common_prefix == tokens.size(),
            "per-row visual identity was replaced by whole-request fingerprint");
    visuals[1].cache_identity++;
    require(history.match(tokens, visuals, 77).common_prefix == 190,
            "changed recent image was not detected");
    visuals[1].cache_identity--;
    visuals[0].position[0]++;
    require(history.match(tokens, visuals, 77).outside_retained_tail,
            "changed old image coordinate was not detected");
    visuals[0].position[0]--;
    visuals[0].cache_identity = 0;
    history.assign(tokens, no_output, tokens.size(), visuals, 77, keep);
    require(history.match(tokens, visuals, 77).common_prefix == tokens.size(),
            "legacy image identity fallback failed");
    require(history.match(tokens, visuals, 78).outside_retained_tail,
            "changed legacy image fingerprint was accepted");
    require(history.match(tokens, text, 77).outside_retained_tail,
            "missing image identity was accepted");
    history.assign(tokens, no_output, tokens.size(), text, 77, keep);
    require(history.match(tokens, text, 77).common_prefix == 0,
            "missing old and new image identities became cache authority");

    std::cout << "bounded prefix: 1000 append rounds, tail rewrites, visual identity and SHA-256 passed\n";
}
