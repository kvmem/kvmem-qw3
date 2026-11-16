#include "kvmem_prefix_state.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using qw3::detail::KvmemQueryHistoryState;
using qw3::detail::KvMemQueryHistory;
using qw3::detail::MultimodalTokenIdentity;

static void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

static double score(const KvmemQueryHistoryState &state, uint64_t key) {
    const auto &scores = state.scores.committed_scores();
    const auto found = scores.find(key);
    return found == scores.end() ? 0 : found->second;
}

static void accept(KvmemQueryHistoryState &state,
        const std::vector<uint32_t> &input,
        const std::vector<MultimodalTokenIdentity> &images,
        const KvmemQueryHistoryState::Positions &positions,
        const KvMemQueryHistory::Scores &scores) {
    auto next = state.prepare_input(input, images, positions, 42);
    state.scores.begin(next.input_end, next.verified_prefix, .5);
    state.scores.preview(scores);
    state.scores.commit();
    state.accept_input(std::move(next));
}

int main() {
    // Accepted output and publication have distinct boundaries. A callback
    // may cancel after the first accepted draft while target KV already owns
    // all of them; a sampled bonus remains pending until it is forwarded.
    qw3::detail::KvmemAcceptedOutput output(100);
    const std::vector<uint32_t> accepted_batch{20, 21, 22, 23};
    output.record(100, accepted_batch.data(), accepted_batch.size());
    check(output.tokens() == accepted_batch,
          "accepted verifier suffix depended on how much text was published");
    const uint32_t pending_bonus = 24;
    check(output.tokens().back() == 23, "unforwarded bonus entered accepted identity");
    output.record(104, &pending_bonus, 1);
    const std::vector<uint32_t> accepted_replay{25, 26};
    output.record(105, accepted_replay.data(), accepted_replay.size());
    check(output.tokens() == std::vector<uint32_t>({20, 21, 22, 23, 24, 25, 26}),
          "plain bridge and accepted rejection replay did not join one prefix");
    bool duplicate_rejected = false;
    try { output.record(105, accepted_replay.data(), accepted_replay.size()); }
    catch (const std::logic_error &) { duplicate_rejected = true; }
    check(duplicate_rejected && output.tokens().size() == 7,
          "speculative replay recorded one committed prefix twice");

    KvmemQueryHistoryState state;
    const std::vector<uint32_t> first{10, 11, 12, 13};
    accept(state, first, {}, {}, {{0, 1}});
    // The next complete input includes only model-accepted output tokens. An
    // emitted final token that was never forwarded is intentionally absent.
    const std::vector<uint32_t> second{10, 11, 12, 13, 20, 21, 30, 31};
    accept(state, second, {}, {}, {{6, 1}});
    check(state.scores.round_count() == 2 && score(state, 0) == .5,
          "append containing accepted output did not inherit its input parent");
    const KvmemQueryHistoryState frozen = state;
    accept(state, second, {}, {}, {{6, 1}});
    check(state.scores.round_count() == 2 && score(state, 0) == .5,
          "same input regeneration accumulated duplicate credit");
    auto third = second;
    third.insert(third.end(), {40, 41});
    accept(state, third, {}, {}, {{8, 1}});
    check(state.scores.round_count() == 3 && score(state, 0) == .25,
          "third input did not continue the same logical history");
    state = frozen;
    check(state.scores.round_count() == 2 && state.identity.size() == second.size(),
          "named-cache restore did not restore scores and input proof together");
    auto branch = second;
    branch[6] = 88;
    accept(state, branch, {}, {}, {{6, 1}});
    check(state.scores.round_count() == 2 && score(state, 0) == .5,
          "edited suffix failed to select the strictly earlier verified input");
    branch[1] = 99;
    accept(state, branch, {}, {}, {{6, 1}});
    check(state.scores.round_count() == 1 && score(state, 0) == 0,
          "early edit inherited an unverified ancestor");
    check(frozen.scores.round_count() == 2 && score(frozen, 0) == .5,
          "editing a restored session mutated the frozen cache");

    KvmemQueryHistoryState positioned;
    KvmemQueryHistoryState::Positions positions{{0,0,0},{1,1,1},{2,2,2},{3,3,3}};
    accept(positioned, first, {}, positions, {{0, 1}});
    auto extended_positions = positions;
    extended_positions.insert(extended_positions.end(), {{4,4,4},{5,5,5},{6,6,6},{7,7,7}});
    auto changed_positions = extended_positions;
    changed_positions[1] = {100, 100, 100};
    auto prepared = positioned.prepare_input(second, {}, changed_positions, 42);
    check(prepared.verified_prefix == 1, "changed text M-RoPE position retained prefix authority");
    check(positioned.prepare_input(second, {}, {}, 42).verified_prefix == 0,
          "missing position identity silently reused supplied positions");
    accept(positioned, second, {}, changed_positions, {{6, 1}});
    check(positioned.scores.round_count() == 1 && score(positioned, 0) == 0,
          "changed text positions inherited incompatible Query evidence");

    KvmemQueryHistoryState visual;
    const std::vector<uint32_t> visual_first{10, 0x80000000u, 12, 13};
    std::vector<MultimodalTokenIdentity> images{{0x80000000u, 99, {1,0,0}, 1234}};
    accept(visual, visual_first, images, positions, {{0, 1}});
    auto visual_next = visual_first;
    visual_next.insert(visual_next.end(), {14, 15, 16, 17});
    check(visual.prepare_input(visual_next, images, extended_positions, 42).verified_prefix == 4,
          "unchanged saved image identity was not inherited on append");
    images[0].cache_identity++;
    check(visual.prepare_input(visual_next, images, extended_positions, 42).verified_prefix == 1,
          "changed image bytes inherited incompatible Query evidence");
    images[0].cache_identity--;
    images[0].position[0]++;
    check(visual.prepare_input(visual_next, images, extended_positions, 42).verified_prefix == 1,
          "changed image coordinates retained prefix authority");

    bool rejected = false;
    try { (void)visual.prepare_input(visual_next, images, positions, 42); }
    catch (const std::invalid_argument &) { rejected = true; }
    check(rejected && visual.identity.size() == visual_first.size(),
          "incomplete input identity changed committed state");
    std::cout << "query history identity: append, regeneration, branches, restored sessions and token/media positions passed\n";
}
