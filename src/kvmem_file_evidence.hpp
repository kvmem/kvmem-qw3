#pragma once

#include "json.hpp"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace qw3::detail {

// Request metadata from an observing tool adapter. It never rewrites model
// messages or bypasses a tool's read/version checks. Disabled unless the
// server explicitly enables QW3_KVMEM_FILE_EVIDENCE.
struct KvmemFileEvidencePlan {
    struct ReadSpan { size_t message, begin, end; bool pin; };
    std::string scope;
    bool changed = false;
    std::vector<size_t> excluded_messages;
    // Newest first: admit each complete Read atomically within the KV budget.
    std::vector<size_t> pinned_messages;
    // Byte coordinates in the original tool content, never replacement text.
    std::vector<ReadSpan> read_spans;
};

inline KvmemFileEvidencePlan parse_kvmem_file_evidence(
        const nlohmann::json &value, const nlohmann::json &messages) {
    KvmemFileEvidencePlan out;
    if (!value.is_object() || !value.contains("scope") ||
        !value["scope"].is_string() ||
        value["scope"].get_ref<const std::string &>().empty() ||
        value["scope"].get_ref<const std::string &>().size() > 512 ||
        !value.contains("changed") || !value["changed"].is_boolean()) {
        throw std::invalid_argument("invalid kvmem_file_evidence scope/changed");
    }
    out.scope = value["scope"].get<std::string>();
    out.changed = value["changed"].get<bool>();
    for (const char *key : {"excluded_messages", "pinned_messages"}) {
        if (!value.contains(key) || !value[key].is_array() ||
            value[key].size() > messages.size()) {
            throw std::invalid_argument("invalid kvmem_file_evidence message list");
        }
        auto &indices = std::string(key) == "pinned_messages"
            ? out.pinned_messages : out.excluded_messages;
        for (const auto &index : value[key]) {
            if (!index.is_number_integer())
                throw std::invalid_argument("evidence message index must be integer");
            const int64_t i = index.get<int64_t>();
            if (i < 0 || static_cast<uint64_t>(i) >= messages.size())
                throw std::invalid_argument("evidence message index out of range");
            const auto &message = messages[static_cast<size_t>(i)];
            const std::string role = message.value("role", "");
            if (role != "tool" && role != "assistant")
                throw std::invalid_argument("evidence cannot exclude user/system messages");
            if (std::string(key) == "pinned_messages" &&
                (role != "tool" || !message.contains("content") ||
                 !message["content"].is_string() ||
                 message["content"].get_ref<const std::string &>().find("<path>") == std::string::npos ||
                 message["content"].get_ref<const std::string &>().find("<content>") == std::string::npos)) {
                throw std::invalid_argument("evidence pin must reference a complete file Read result");
            }
            if (std::find(indices.begin(), indices.end(), static_cast<size_t>(i)) != indices.end())
                throw std::invalid_argument("duplicate evidence message index");
            indices.push_back(static_cast<size_t>(i));
        }
    }
    std::sort(out.excluded_messages.begin(), out.excluded_messages.end());
    for (size_t i : out.pinned_messages) {
        if (std::binary_search(out.excluded_messages.begin(), out.excluded_messages.end(), i))
            throw std::invalid_argument("evidence message cannot be pinned and excluded");
    }
    if (value.contains("read_spans")) {
        if (!value["read_spans"].is_array() || value["read_spans"].size() > 8192)
            throw std::invalid_argument("invalid evidence read_spans");
        for (const auto &span : value["read_spans"]) {
            for (const char *key : {"message", "begin", "end"})
                if (!span.is_object() || !span.contains(key) ||
                    !span[key].is_number_integer() || span[key].get<int64_t>() < 0)
                    throw std::invalid_argument("invalid evidence source span coordinate");
            if (!span.contains("pin") || !span["pin"].is_boolean())
                throw std::invalid_argument("invalid evidence source span pin");
            const size_t i=span["message"].get<size_t>();
            const size_t begin=span["begin"].get<size_t>(), end=span["end"].get<size_t>();
            if (i >= messages.size() || messages[i].value("role", "") != "tool" ||
                !messages[i].contains("content") || !messages[i]["content"].is_string())
                throw std::invalid_argument("evidence source span must reference a Read");
            const auto &text=messages[i]["content"].get_ref<const std::string &>();
            const size_t body=text.find("<content>\n"), close=text.rfind("</content>");
            if (text.find("<path>")==std::string::npos || body==std::string::npos ||
                close==std::string::npos || begin < body+10 || end > close || begin>=end ||
                (static_cast<unsigned char>(text[begin]) & 0xc0)==0x80 ||
                (end<text.size() && (static_cast<unsigned char>(text[end]) & 0xc0)==0x80))
                throw std::invalid_argument("evidence source span must stay inside original Read content");
            if (std::find(out.pinned_messages.begin(),out.pinned_messages.end(),i)!=out.pinned_messages.end())
                throw std::invalid_argument("whole Read and partial Read pins cannot overlap");
            out.read_spans.push_back({i,begin,end,span["pin"].get<bool>()});
        }
        auto sorted=out.read_spans;
        std::sort(sorted.begin(),sorted.end(),[](const auto &a,const auto &b){
            return a.message!=b.message ? a.message<b.message : a.begin<b.begin; });
        for (size_t i=1;i<sorted.size();++i)
            if (sorted[i].message==sorted[i-1].message && sorted[i].begin<sorted[i-1].end)
                throw std::invalid_argument("evidence source spans overlap");
    }
    return out;
}

inline bool kvmem_admit_complete_evidence_ranges(
        std::vector<uint32_t> &kept,
        const std::vector<std::pair<uint32_t,uint32_t>> &ranges,
        uint32_t prompt_blocks, uint32_t budget_blocks, uint32_t reserve) {
    auto combined=kept;
    for (const auto &[first,end] : ranges) {
        if (first>=end || end>prompt_blocks) return false;
        for (uint32_t id=first;id<end;++id) combined.push_back(id);
    }
    std::sort(combined.begin(),combined.end());
    combined.erase(std::unique(combined.begin(),combined.end()),combined.end());
    if (combined.size()==kept.size()) return true;
    const uint32_t limit=budget_blocks>reserve ? budget_blocks-reserve : 0;
    if (combined.size()>limit) return false;
    kept=std::move(combined);
    return true;
}

// Reuse the already fitted control/live anchors. Never let evidence pins
// enlarge H, displace these anchors, or partially admit a Read.
inline bool kvmem_admit_complete_evidence_blocks(
        std::vector<uint32_t> &kept, uint32_t first, uint32_t end,
        uint32_t prompt_blocks, uint32_t budget_blocks,
        uint32_t retrieval_reserve) {
    if (first >= end || end > prompt_blocks) return false;
    uint32_t extra = 0;
    for (uint32_t i = first; i < end; ++i)
        extra += !std::binary_search(kept.begin(), kept.end(), i);
    if (extra == 0) return true;
    const uint32_t limit = budget_blocks > retrieval_reserve
        ? budget_blocks - retrieval_reserve : 0;
    if (kept.size() > limit || extra > limit - kept.size()) return false;
    for (uint32_t i = first; i < end; ++i) kept.push_back(i);
    std::sort(kept.begin(), kept.end());
    kept.erase(std::unique(kept.begin(), kept.end()), kept.end());
    return true;
}

} // namespace qw3::detail
