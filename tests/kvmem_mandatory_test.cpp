#include "kvmem_mandatory_request.hpp"

#include <iostream>
#include <stdexcept>

using qw3::KvMemMandatoryConfig;
using qw3::detail::parse_kvmem_mandatory_parts;
using nlohmann::json;

static void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

template<class F> static void rejected(F action) {
    bool threw = false;
    try { action(); } catch (const std::invalid_argument &) { threw = true; }
    check(threw, "invalid mandatory configuration was silently accepted");
}

int main() {
    const KvMemMandatoryConfig defaults;
    check(defaults.parts() == "system,root,query,policy", "default retention changed");
    check(KvMemMandatoryConfig::parse("default").parts() == defaults.parts(), "default preset changed");
    check(KvMemMandatoryConfig::parse("all").parts() ==
          "system,root,query,policy,tool-call,tool-result", "all omitted a category");
    check(KvMemMandatoryConfig::parse("none").parts() == "none", "none retained a category");
    check(KvMemMandatoryConfig::parse(" tool-result , system ").parts() ==
          "system,tool-result", "list did not replace default or canonicalize order");
    for (const char *bad : {"", "system,", ",query", "system,,query", "tool",
                            "system,system", "none,query", "default,query"})
        rejected([&] { (void)KvMemMandatoryConfig::parse(bad); });

    const auto result_only = parse_kvmem_mandatory_parts(
        json::array({"tool-result"}), defaults);
    check(result_only.tool_result && !result_only.tool_call && !result_only.query,
          "result-only request also protected call/query");
    check(parse_kvmem_mandatory_parts(json::array(), defaults).parts() == "none",
          "empty array failed to disable categories");
    check(parse_kvmem_mandatory_parts(json::object(), result_only).parts() == "tool-result" &&
          parse_kvmem_mandatory_parts(json("default"), result_only).parts() == "tool-result" &&
          parse_kvmem_mandatory_parts(json(" default "), result_only).parts() == "tool-result",
          "empty override/default failed to inherit configured server policy");
    const auto switched = parse_kvmem_mandatory_parts(
        json{{"policy", false}, {"tool-call", true}}, defaults);
    check(switched.parts() == "system,root,query,tool-call", "partial switches replaced unrelated defaults");
    check(defaults.parts() == "system,root,query,policy", "request override mutated server defaults");
    for (const auto &bad : {json(nullptr), json(32), json(false), json{{"query", 0}},
                           json{{"unknown", true}}, json::array({"system", "system"}),
                           json::array({"query", 3}), json::array({"system,query"})})
        rejected([&] { (void)parse_kvmem_mandatory_parts(bad, defaults); });

    const auto root_only = KvMemMandatoryConfig::parse("root");
    check(root_only.protects_current_query(true) && !root_only.protects_current_query(false),
          "shared root/current instruction had the wrong lifetime");
    std::cout << "mandatory parts: defaults, masks, partial overrides and rejection passed\n";
}
