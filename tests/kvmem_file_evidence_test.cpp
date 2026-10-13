#ifdef NDEBUG
#undef NDEBUG
#endif
#include "kvmem_file_evidence.hpp"
#include "kvmem_refresh_policy.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace qw3::detail;
    using json = nlohmann::json;
    const json messages = json::array({
        {{"role", "system"}, {"content", "tools"}},
        {{"role", "assistant"}, {"content", "call"}},
        {{"role", "tool"}, {"content", "<path>a</path><content>1: x</content>"}}});
    json metadata = {{"scope", "session1:/workspace"}, {"changed", true},
                     {"excluded_messages", json::array({1})},
                     {"pinned_messages", json::array({2})}};
    assert(parse_kvmem_file_evidence(metadata, messages).changed);
    json source_messages=json::array({{{"role","tool"},{"content","<path>a</path><content>\n1: old\n2: kept\n</content>"}}});
    const auto body=source_messages[0]["content"].get<std::string>();
    const size_t second=body.find("2: kept");
    json partial={{"scope","partial"},{"changed",true},{"excluded_messages",{0}},
        {"pinned_messages",json::array()},{"read_spans",json::array({
            {{"message",0},{"begin",second},{"end",second+7},{"pin",true}}})}};
    assert(parse_kvmem_file_evidence(partial,source_messages).read_spans.size()==1);
    auto invalid_span=partial;invalid_span["read_spans"][0]["begin"]=0;
    bool rejected_span=false;
    try { parse_kvmem_file_evidence(invalid_span,source_messages); }
    catch(const std::invalid_argument &) { rejected_span=true; }
    assert(rejected_span);
    for (const json &invalid : {
            json{{"scope", ""}, {"changed", true}},
            json{{"scope", "x"}, {"changed", true}, {"excluded_messages", {0}}, {"pinned_messages", json::array()}},
            json{{"scope", "x"}, {"changed", true}, {"excluded_messages", {2}}, {"pinned_messages", {2}}}}) {
        bool rejected = false;
        try { parse_kvmem_file_evidence(invalid, messages); }
        catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected);
    }
    std::vector<uint32_t> kept{0, 1, 7};
    assert(kvmem_admit_complete_evidence_blocks(kept, 2, 5, 20, 10, 2));
    const auto before = kept;
    assert(!kvmem_admit_complete_evidence_blocks(kept, 8, 12, 20, 10, 2));
    assert(kept == before); // Atomic rejection, no partial Read or displacement.
    assert(kvmem_admit_complete_evidence_blocks(kept, 2, 5, 20, 3, 0)); // Already resident.
    const auto before_ranges=kept;
    assert(!kvmem_admit_complete_evidence_ranges(kept,{{6,9},{12,15}},20,10,2));
    assert(kept==before_ranges); // Header/body/footer admission stays atomic.
    assert(kvmem_admit_complete_evidence_ranges(kept,{{2,4},{3,5}},20,10,2));
    KvmemHarnessRefreshInput input;
    input.known_conversation = input.same_user_query = input.selection_started = true;
    input.prompt_tokens = 100000; input.selection_budget = 36864;
    input.trigger_tokens = 28672; input.evidence_changed = true;
    auto decision = kvmem_harness_refresh_decision(input);
    assert(decision.reselect && decision.private_query &&
           decision.reason == KvmemHarnessRefreshReason::EvidenceChanged);
    input.evidence_changed = false;
    assert(!kvmem_harness_refresh_decision(input).reselect);
    input.prompt_tokens = 100; input.evidence_changed = true;
    assert(!kvmem_harness_refresh_decision(input).reselect); // Dense context needs no selection.
    std::cout << "file evidence policy passed\n";
}
