#include "vision_request_preparation.hpp"
#include "qw3/gguf.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>

using namespace qw3;
using namespace qw3::detail;
using J = nlohmann::json;
static void check(bool v, const char *message) { if (!v) throw std::runtime_error(message); }

static VisionEncoding fake_encode(const std::vector<VisionImage> &images) {
    VisionEncoding out;
    out.embedding_dim = 4;
    for (const auto &image : images) {
        const uint32_t width = image.base64_data.front() == 'B' ? 6 : 4;
        out.grids.push_back({1, 4, width, width});
        for (uint32_t i = 0; i < width * 4; ++i) out.embeddings.push_back(float(i + image.base64_data.front()));
    }
    return out;
}
static VisionEncoding fake_sources(const std::vector<VisionImageSourcePtr> &images) {
    std::vector<VisionImage> values;
    for (const auto &image : images) values.push_back(image->image());
    return fake_encode(values);
}
static J picture(std::string payload, std::string text = "Inspect this image.") {
    return J{{"role","user"},{"content",J::array({J{{"type","text"},{"text",text}},
        J{{"type","image_url"},{"image_url",{{"url","data:image/png;base64," + payload}}}}})}};
}
static std::string render(const J &messages) {
    std::string out;
    for (const auto &m : messages) {
        out += "<|im_start|>" + m["role"].get<std::string>() + "\n";
        if (m["content"].is_string()) out += m["content"].get<std::string>();
        else for (const auto &b : m["content"]) out += b.value("text","");
        out += "<|im_end|>\n";
    }
    return out + "<|im_start|>assistant\n";
}
static void compare(const GenerationOptions &a, const GenerationOptions &b) {
    check(a.input_embedding_fingerprint == b.input_embedding_fingerprint,"image fingerprint differs");
    check(a.input_mrope_positions == b.input_mrope_positions,"M-RoPE differs");
    check(a.input_embedding_overrides.size() == b.input_embedding_overrides.size(),"row count differs");
    for (size_t i = 0; i < a.input_embedding_overrides.size(); ++i) {
        const auto &x = a.input_embedding_overrides[i], &y = b.input_embedding_overrides[i];
        check(x.token_id == y.token_id && x.source_token_id == y.source_token_id &&
              x.position == y.position && x.cache_identity == y.cache_identity &&
              x.embedding == y.embedding && x.embedding_row == y.embedding_row,"visual row differs");
    }
}

static void text_only_parity(QwenTokenizer &tokenizer) {
    PreparedMessageCache cache(8u << 20, 2);
    J history = J::array({J{{"role", "system"}, {"content", "Keep exact text."}},
                         J{{"role", "user"}, {"content", "Inspect appended records."}}});
    for (size_t turn = 0; turn < 100; ++turn) {
        if (turn && turn % 13 == 0) history.front()["content"] = "Changed system " + std::to_string(turn);
        if (turn && turn % 7 == 0) history.back()["content"] = "rewritten 中文 👩‍💻\n";
        history.push_back(J{{"role", "assistant"}, {"content", ""},
                            {"tool_calls", J::array({J{{"id", std::to_string(turn)}}})}});
        history.push_back(J{{"role", "tool"}, {"tool_call_id", std::to_string(turn)},
                            {"content", turn % 3 ? J("text 中文\n<|image_pad|> café") : J::array()}});
        size_t matched = 0;
        auto previous = cache.match_prefix(history, matched);
        J suffix = J::array();
        for (size_t i = matched; i < history.size(); ++i) suffix.push_back(history[i]);
        J reference = history;
        auto candidate = std::make_shared<PreparedMessageCache::Entry>();
        PreparedVisionRequest actual, full;
        std::string error;
        // A text-only request must work without any vision encoder installed.
        check(prepare_vision_messages_incremental(suffix, {}, previous, matched, *candidate, actual, error), error.c_str());
        check(prepare_vision_messages(reference, {}, full, error), error.c_str());
        check(suffix == reference && reference == history, "text-only preparation changed messages");
        check(!actual.active() && actual.sources.empty() && actual.fingerprint == 0, "text request activated vision");
        auto tokens = tokenizer.encode(render(suffix)), expected = tokenizer.encode(render(history));
        GenerationOptions options;
        check(finalize_vision_tokens(tokens, tokenizer, actual, options, error), error.c_str());
        check(tokens == expected && options.input_embedding_overrides.empty() && options.input_mrope_positions.empty(),
              "text-only preparation changed tokens or positions");
        for (size_t i = 0; i < history.size(); ++i) check(candidate->messages[i]->raw == history[i], "text source changed");
        check(cache.store("text-" + std::to_string(turn), candidate), "text cache store failed");
    }
    std::cout << "100 pure-text histories: exact messages/tokens, tool IDs, append/rewrite/system changes, no vision encoder passed\n";
}

int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: qw3-vision-request-preparation MODEL.gguf");
        GgufFile gguf(argv[1]);
        QwenTokenizer tokenizer(gguf);
        text_only_parity(tokenizer);
        PreparedMessageCache cache(128u << 20, 2);
        std::shared_ptr<const PreparedMessageCache::Entry> previous;
        J full = J::array();
        std::string id;
        uint64_t reused_tokens = 0, reused_images = 0;
        for (int iteration = 0; iteration < 180; ++iteration) {
            size_t count = full.size();
            J suffix = J::array();
            if (!iteration || iteration % 17 == 0) {
                count = 0;
                suffix.push_back(J{{"role","system"},{"content","System changed " + std::to_string(iteration) + " 中文 👩‍💻"}});
            } else if (iteration % 7 == 0) {
                count = full.size() > 2 ? full.size() - 2 : 1;
            }
            if (count) suffix.push_back(J{{"role","assistant"},{"content","previous answer / rewritten tail " + std::to_string(iteration)}});
            suffix.push_back(picture(std::string(1,iteration % 3 ? 'A' : 'B') + std::to_string(iteration)));
            if (iteration % 11 == 0) suffix.push_back(picture("A-duplicate"));
            J expected = J::array();
            for (size_t i=0;i<count;++i) expected.push_back(full[i]);
            for (const auto &m : suffix) expected.push_back(m);
            if (!id.empty()) check(cache.restore(id,count,previous)==PreparedMessageCache::Restore::Ok,"prefix unexpectedly expired");
            auto candidate = std::make_shared<PreparedMessageCache::Entry>();
            PreparedVisionRequest actual, reference;
            std::string error;
            J processed = suffix, original = expected;
            check(prepare_vision_messages_incremental(processed,fake_sources,previous,count,*candidate,actual,error),error.c_str());
            check(prepare_vision_messages(original,fake_encode,reference,error),error.c_str());
            check(processed==original,"rendered message prefix differs");
            check(actual.fingerprint==reference.fingerprint && actual.image_fingerprints==reference.image_fingerprints,"source hash differs");
            for (size_t i=0;i<count;++i) check(candidate->messages[i]==previous->messages[i],"copied immutable prefix record");
            for (size_t i=0;i<expected.size();++i) check(candidate->messages[i]->raw==expected[i],"raw source changed");
            auto a=tokenizer.encode(render(processed)), b=tokenizer.encode(render(original));
            GenerationOptions ga,gb;
            check(finalize_vision_tokens(a,tokenizer,actual,ga,error,previous?previous->mapping:nullptr,&candidate->mapping),error.c_str());
            check(finalize_vision_tokens(b,tokenizer,reference,gb,error),error.c_str());
            check(a==b && actual.token_spans==reference.token_spans,"mapped token/span differs");
            compare(ga,gb);
            reused_tokens+=actual.mapping_tokens_reused; reused_images+=actual.prefix_images_reused;
            // Explicit oracle for the first image's spatial coordinates.
            const size_t begin=actual.token_spans.front().first+1;
            const uint32_t base=ga.input_mrope_positions[begin-1][0]+1;
            check(ga.input_mrope_positions[begin]==std::array<uint32_t,3>{base,base,base},"first image origin wrong");
            check(ga.input_mrope_positions[begin+1]==std::array<uint32_t,3>{base,base,base+1},"first image column wrong");
            id="request-"+std::to_string(iteration);
            check(cache.store(id,candidate),"candidate does not fit test cache");
            check(!cache.store(id,candidate),"immutable handle overwritten");
            full=std::move(expected);
        }
        check(reused_tokens>0 && reused_images>0,"incremental path never exercised");
        size_t matched = 0;
        auto implicit = cache.match_prefix(full, matched);
        check(implicit && matched == full.size(), "native full-history match failed");
        J changed = full;
        changed.back() = picture("changed-image");
        implicit = cache.match_prefix(changed, matched);
        check(implicit && matched == full.size() - 1, "native image rewrite reused changed record");
        changed.front() = J{{"role", "system"}, {"content", "unrelated system"}};
        implicit = cache.match_prefix(changed, matched);
        check(!implicit && matched == 0, "native unrelated history reused prefix");
        check(cache.restore(id,full.size()+1,previous)==PreparedMessageCache::Restore::Invalid,"invalid count accepted");
        check(cache.restore("request-0",1,previous)==PreparedMessageCache::Restore::Missing,"LRU eviction failed");
        check(cache.restore(id,0,previous)==PreparedMessageCache::Restore::Ok,"zero-prefix restore failed");
        PreparedMessageCache tiny(128,2);
        check(!tiny.store("too-large",previous),"oversize handle acknowledged");
        auto inflight=previous;
        check(cache.store("replace-1",previous) && cache.store("replace-2",previous),"replace failed");
        check(cache.restore(id,0,previous)==PreparedMessageCache::Restore::Missing,"expected eviction missing");
        check(!inflight->messages.empty(),"in-flight snapshot lost on eviction");
        J bad=J::array({picture("A")});bad[0]["content"][1]["image_url"]["url"]="https://invalid/image.png";
        PreparedMessageCache::Entry candidate;PreparedVisionRequest request;std::string error;
        check(!prepare_vision_messages_incremental(bad,fake_sources,{},0,candidate,request,error),"unvalidated image accepted");
        std::cout<<"180 real-tokenizer histories: raw/rendered/token/feature/M-RoPE parity, append/rewrite/image/grid/system changes, expiry/capacity/in-flight lifetime passed; reused_tokens="<<reused_tokens<<" reused_images="<<reused_images<<'\n';
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
