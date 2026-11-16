#include "kvmem_query_history.hpp"
#include "kvmem_prefix_history.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

using qw3::detail::KvMemQueryHistory;

static void require(bool ok, const char *message) {
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}

static double score(const KvMemQueryHistory::Scores &scores, uint64_t start) {
    const auto found = scores.find(start);
    return found == scores.end() ? 0 : found->second;
}

static bool near(double a, double b) { return std::abs(a - b) < 1e-12; }

static void verified_input_lineage() {
    using qw3::detail::KvMemPrefixHistory;
    using qw3::detail::MultimodalTokenIdentity;
    KvMemQueryHistory history;
    KvMemPrefixHistory identity;
    std::vector<uint32_t> first{11, 12, 0x80000000u, 14};
    std::vector<MultimodalTokenIdentity> media{{0x80000000u, 99, {2, 0, 0}, 1234}};
    auto accept = [&](const std::vector<uint32_t> &input,
                      const std::vector<MultimodalTokenIdentity> &images,
                      const KvMemQueryHistory::Scores &raw) {
        const auto proof = identity.match(input, images, 7);
        history.begin(input.size(), proof.common_prefix, .5);
        history.preview(raw);
        history.commit();
        identity.assign(input, {}, input.size(), images, 7, input.size());
    };
    accept(first, media, {{0, 1}});
    accept(first, media, {{0, 1}});
    require(history.round_count() == 1 && near(score(history.scores(), 0), 1),
            "verified same-input regeneration rewarded itself");
    auto appended = first;
    appended.insert(appended.end(), {15, 16});
    accept(appended, media, {{4, 1}});
    require(history.round_count() == 2 && near(score(history.scores(), 0), .5),
            "verified append failed to inherit its parent");
    const auto frozen_history = history.snapshot();
    const auto frozen_identity = identity;
    auto changed_image = media;
    ++changed_image[0].cache_identity;
    accept(appended, changed_image, {{4, 1}});
    require(history.round_count() == 1 && score(history.scores(), 0) == 0,
            "changed media inherited another input's query history");
    history.restore(frozen_history);
    identity = frozen_identity;
    auto edited = appended;
    edited[1] = 88;
    accept(edited, media, {{4, 1}});
    require(history.round_count() == 1 && score(history.scores(), 0) == 0,
            "edited prefix inherited unverified query history");
    history.restore(frozen_history);
    identity = frozen_identity;
    accept(appended, media, {{4, 1}});
    require(history.round_count() == 2 && near(score(history.scores(), 0), .5),
            "named-cache restore lost its immutable input parent");
}

int main() {
    verified_input_lineage();
    KvMemQueryHistory history;
    require(history.snapshot().empty() && !history.open(), "new history was not empty");
    history.begin(128, 0, .5);
    history.preview({{0, 3}, {32, 1}, {128, 100}, {160, 100}});
    require(near(score(history.scores(), 0), .75) &&
            near(score(history.scores(), 32), .25) &&
            history.scores().size() == 2,
            "query normalization included header-only/future blocks");
    require(history.committed_scores().empty(), "preview committed its observation");
    require(history.commit() && !history.commit() && history.round_count() == 1,
            "round commit was not exactly once");
    const auto first = history.snapshot();

    history.begin(256, 128, .5);
    history.preview({{32, 1}, {192, 3}});
    const auto once = history.scores();
    for (unsigned replay = 0; replay < 8; ++replay)
        require(history.preview({{32, 1}, {192, 3}}) == once,
                "replay repeatedly decayed/rewarded one input");
    require(near(score(once, 0), .375) && near(score(once, 32), .375) &&
            near(score(once, 192), .75), "parent history/current query formula changed");
    const auto during_preview = history.snapshot();
    history.preview({{64, 1}});
    require(near(score(history.scores(), 0), .375) &&
            near(score(history.scores(), 32), .125) &&
            near(score(history.scores(), 64), 1) && score(history.scores(), 192) == 0,
            "replacement preview used its predecessor as the parent");
    history.preview({{32, 1}, {192, 3}});
    history.commit();
    const auto second = history.snapshot();
    require(history.round_count() == 2, "append did not extend history");

    // Full identity matches on regeneration still select a STRICTLY earlier
    // input node. The same scores must reproduce, not accumulate the old turn.
    history.begin(256, 256, .5);
    require(history.preview({{32, 1}, {192, 3}}) == once,
            "same-input regeneration inherited its own reward");
    history.commit();
    require(history.round_count() == 2 && history.scores() == once,
            "regeneration appended an extra logical round");

    // Only complete, verified input boundaries may be inherited. A rollback
    // within the first input has no matching parent, even with live old KV.
    history.begin(300, 100, .5);
    require(history.scores().empty(), "unverified input node became a parent");
    history.preview({{64, 1}});
    history.abort();
    require(history.scores() == once && history.input_end() == 256,
            "aborted branch destroyed the prior committed future");

    history.begin(192, 128, .5);
    history.preview({{64, 1}, {192, 999}});
    history.commit();
    const auto branch_scores = history.scores();
    require(history.round_count() == 2 && score(branch_scores, 192) == 0,
            "edited input retained stale future blocks");
    history.begin(320, 256, .5);
    require(history.scores() == branch_scores,
            "new append resurrected the abandoned future branch");
    history.abort();

    KvMemQueryHistory other;
    other.restore(second);
    require(other.scores() == once && other.input_end() == 256,
            "snapshot was mutated by another conversation branch");
    other.begin(384, 256, .5);
    other.preview({{288, 1}});
    other.commit();
    require(history.scores() == branch_scores,
            "mutating a restored session changed another session");
    other.restore(during_preview);
    require(other.input_end() == 128 && other.round_count() == 1,
            "snapshot captured an uncommitted replay");

    // A failed/cold/empty query is absence of an observation, not a negative
    // observation. Recording a boundary must neither reward nor decay history.
    history.restore(first);
    const auto no_observation_parent = history.scores();
    const auto *shared_map = &history.scores();
    history.begin(256, 128, .1);
    history.preview({{0, 1}});
    require(history.has_observation(), "valid query was not an observation");
    history.preview({{0, 0}, {32, 0}, {256, 4}});
    require(!history.has_observation() && history.scores() == no_observation_parent,
            "empty replacement query implicitly decayed history");
    history.commit();
    require(history.input_end() == 256 && history.round_count() == 2 &&
            &history.scores() == shared_map, "unscored boundary copied or changed scores");
    history.begin(320, 256, 0);
    history.commit();
    require(history.scores() == no_observation_parent,
            "commit without preview erased parent history");
    history.begin(384, 320, 0);
    history.preview({});
    history.commit();
    require(history.scores() == no_observation_parent,
            "empty query with zero decay erased parent history");

    // Invalid input is transactional: it cannot poison an already valid
    // preview, and invalid decay cannot replace the current open round.
    history.begin(448, 384, .5);
    history.preview({{64, 1}});
    const auto valid_candidate = history.scores();
    for (double invalid : {-1., std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        bool caught = false;
        try { history.preview({{0, invalid}}); }
        catch (const std::invalid_argument &) { caught = true; }
        require(caught && history.scores() == valid_candidate && history.has_observation(),
                "invalid scores poisoned or silently replaced a valid preview");
    }
    for (double invalid : {-1., 1.01, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        bool caught = false;
        try { history.begin(512, 448, invalid); }
        catch (const std::invalid_argument &) { caught = true; }
        require(caught && history.input_end() == 448 && history.scores() == valid_candidate,
                "invalid decay replaced an open round");
    }
    history.abort();
    bool closed_preview_caught = false;
    try { history.preview({{0, 1}}); }
    catch (const std::logic_error &) { closed_preview_caught = true; }
    require(closed_preview_caught, "decode preview opened a new logical observation");

    history.begin(448, 384, 0);
    const double huge = std::numeric_limits<double>::max();
    history.preview({{0, huge}, {32, huge}});
    require(near(score(history.scores(), 0), .5) &&
            near(score(history.scores(), 32), .5) && history.scores().size() == 2,
            "finite large-score normalization overflowed");
    history.preview({{64, std::numeric_limits<double>::denorm_min()}});
    require(history.has_observation() && score(history.scores(), 64) == 1 &&
            history.scores().size() == 1, "small positive query was treated as empty");
    history.commit();

    history.clear();
    require(history.scores().empty() && history.snapshot().empty() &&
            history.round_count() == 0 && !history.open(), "clear retained a prior branch");
    history.restore(first);
    require(history.scores() == no_observation_parent,
            "clearing one session destroyed a saved snapshot");
    history.restore({});
    history.begin(64, 0, 1);
    history.commit();
    require(history.scores().empty() && history.round_count() == 1,
            "cold zero-input observation manufactured score history");

    std::cout << "query history: normalized rounds, regeneration, replay, rollback, "
                 "shared snapshots and invalid/empty observations passed\n";
}
