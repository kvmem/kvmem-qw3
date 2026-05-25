#include "backend.hpp"
#include "qwen_executor.hpp"
#include "qwen_native.hpp"
#include "qwen_weights.hpp"
#include "qw3/device_backend.hpp"
#include "qw3/tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace qw3 {
namespace {

double wall_seconds() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

std::string escape_text(const std::string &s) {
    std::ostringstream out;
    out << "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"':  out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out << buf;
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    out << "\"";
    return out.str();
}

// Streams one JSON object per forward step into a file. Each line records
// the (input token, decoded text, argmax token, top-k logits + token strings)
// so it can be diffed against a llama.cpp logit dump or used to localize
// where qw3 diverges from a reference.
class DumpStream {
public:
    DumpStream(const std::string &path, int top_k,
               const std::vector<uint32_t> &prompt_tokens,
               const QwenTokenizer &tok)
        : top_k_(top_k > 0 ? top_k : 1) {
        out_.open(path);
        if (!out_) throw std::runtime_error("failed to open dump-logits path: " + path);
        out_ << "{\"event\":\"prompt\",\"tokens\":[";
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            if (i) out_ << ",";
            out_ << prompt_tokens[i];
        }
        out_ << "],\"strings\":[";
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            if (i) out_ << ",";
            out_ << escape_text(tok.decode_one(static_cast<int32_t>(prompt_tokens[i])));
        }
        out_ << "]}\n";
    }

    void record(int step_idx, const char *phase, int32_t input_token,
                const QwenExecutor &exec, const QwenTokenizer &tok) {
        std::vector<float> logits;
        if (!exec.copy_last_logits(logits)) return;
        const int K = std::min<int>(top_k_, static_cast<int>(logits.size()));
        std::vector<int> idx(K, -1);
        std::vector<float> val(K, -INFINITY);
        for (size_t i = 0; i < logits.size(); ++i) {
            const float v = logits[i];
            int slot = -1;
            for (int j = 0; j < K; ++j) if (v > val[j]) { slot = j; break; }
            if (slot < 0) continue;
            for (int j = K - 1; j > slot; --j) {
                val[j] = val[j - 1];
                idx[j] = idx[j - 1];
            }
            val[slot] = v;
            idx[slot] = static_cast<int>(i);
        }
        out_ << "{\"event\":\"step\",\"step\":" << step_idx
             << ",\"phase\":\"" << phase << "\""
             << ",\"input_token\":" << input_token
             << ",\"input_text\":" << escape_text(tok.decode_one(input_token))
             << ",\"argmax_token\":" << idx[0]
             << ",\"argmax_logit\":" << val[0]
             << ",\"argmax_text\":" << escape_text(tok.decode_one(idx[0]))
             << ",\"top\":[";
        for (int j = 0; j < K; ++j) {
            if (j) out_ << ",";
            out_ << "{\"id\":" << idx[j]
                 << ",\"logit\":" << val[j]
                 << ",\"text\":" << escape_text(tok.decode_one(idx[j])) << "}";
        }
        out_ << "]}\n";
    }

private:
    std::ofstream out_;
    int top_k_ = 16;
};

class QwenNativeBackend final : public Backend {
public:
    std::string name() const override {
        return "qwen-native";
    }

    void load(const EngineOptions &options) override {
        if (options.model_path.empty()) {
            throw std::invalid_argument("qwen-native backend requires --model");
        }
        options_ = options;

        const double t0 = wall_seconds();
        model_ = std::make_unique<QwenNativeModel>(std::make_unique<GgufFile>(options.model_path));
        const double t_gguf = wall_seconds();

        // Device backend + weight uploads are now part of load(), not
        // generate(). Subsequent generate() calls reuse the same DeviceBackend
        // and the same on-GPU weight buffers.
        if (options_.native_kernels != "cuda") {
            // mock/cpu kernels are no longer wired here; we still let load()
            // complete so callers that just want to inspect the plan can do so.
            log("native load: gguf=" + fmt_seconds(t_gguf - t0) +
                " (skipped device init: native-kernels=" + options_.native_kernels + ")");
            return;
        }
        if (!cuda_device_backend_available()) {
            throw std::runtime_error("CUDA kernels were not built; configure with -DQW3_ENABLE_CUDA=ON");
        }
        const LinearBackend linear_backend = parse_linear_backend(options_.native_linear_backend);
        device_ = make_cuda_device_backend(linear_backend);
        if (!device_) throw std::runtime_error("CUDA device backend is unavailable");

        DeviceStatus st = device_->begin();
        if (!st.ok) throw std::runtime_error(std::string("device begin failed: ") + st.message);

        const double t_begin = wall_seconds();
        weights_ = std::make_unique<QwenWeights>(*model_, *device_);
        st = device_->synchronize();
        if (!st.ok) throw std::runtime_error(std::string("weight upload sync failed: ") + st.message);
        const double t_weights = wall_seconds();

        const uint32_t ctx_size = options_.ctx_size > 0 ? static_cast<uint32_t>(options_.ctx_size) : 4096u;
        executor_ = std::make_unique<QwenExecutor>(*model_, *weights_, *device_, ctx_size);
        executor_->reset_state();

        st = device_->end();
        if (!st.ok) throw std::runtime_error(std::string("device end failed: ") + st.message);

        const double mib = static_cast<double>(weights_->total_bytes_uploaded()) / (1024.0 * 1024.0);
        std::ostringstream msg;
        msg << "native load: gguf=" << fmt_seconds(t_gguf - t0)
            << " device_init=" << fmt_seconds(t_begin - t_gguf)
            << " weights_upload=" << fmt_seconds(t_weights - t_begin)
            << " tensors=" << weights_->tensor_count()
            << " size=" << std::fixed << std::setprecision(1) << mib << " MiB"
            << " backend=" << linear_backend_name(linear_backend);
        log(msg.str());
    }

    std::string generate(const std::string &prompt,
                         const GenerationOptions &options,
                         const TokenCallback &on_text) override {
        if (!model_) throw std::runtime_error("qwen-native backend is not loaded");
        if (options_.native_kernels != "cuda") {
            throw std::runtime_error("qwen-native now uses a device-resident executor; use --native-kernels cuda");
        }
        if (!device_ || !weights_ || !executor_) {
            throw std::runtime_error("qwen-native backend was not fully initialized in load()");
        }

        if (!tokenizer_) tokenizer_ = std::make_unique<QwenTokenizer>(model_->gguf());
        const std::vector<int32_t> ids = tokenizer_->encode(prompt);
        std::vector<uint32_t> prompt_tokens(ids.begin(), ids.end());

        if (options_.dump_tokens) {
            std::ostringstream out;
            out << "tokens=" << prompt_tokens.size() << "\n";
            for (size_t i = 0; i < prompt_tokens.size(); ++i) {
                const std::string text = tokenizer_->decode_one(static_cast<int32_t>(prompt_tokens[i]));
                out << i << "\t" << prompt_tokens[i] << "\t" << escape_text(text) << "\n";
            }
            const std::string text = out.str();
            if (on_text) on_text(text);
            return text;
        }

        std::unique_ptr<DumpStream> dump;
        if (!options_.dump_logits_path.empty()) {
            dump = std::make_unique<DumpStream>(options_.dump_logits_path,
                                                options_.dump_logits_top_k,
                                                prompt_tokens, *tokenizer_);
        }

        if (!options_.native_heavy) {
            DeviceStatus st = device_->begin();
            if (!st.ok) throw std::runtime_error(st.message);
            NativeExecutorReport dry = executor_->dry_run_token(
                prompt_tokens.empty() ? 0u : prompt_tokens.front(), false);
            st = device_->end();
            if (!st.ok) throw std::runtime_error(st.message);
            std::ostringstream out;
            out << "executor_dry_run: " << (dry.ok ? "complete" : "blocked") << "\n";
            const std::string text = out.str();
            if (on_text) on_text(text);
            return text;
        }

        DeviceStatus st = device_->begin();
        if (!st.ok) throw std::runtime_error(st.message);
        executor_->reset_state();

        const double t_prefill_start = wall_seconds();
        uint64_t prefill_ops = 0;
        NativeExecutorReport step;
        for (size_t pi = 0; pi < prompt_tokens.size(); ++pi) {
            step = executor_->forward_one_token(prompt_tokens[pi]);
            if (!step.ok) throw std::runtime_error("prefill failed");
            prefill_ops += step.ops_executed;
            if (dump) dump->record(static_cast<int>(pi), "prefill",
                                   static_cast<int32_t>(prompt_tokens[pi]),
                                   *executor_, *tokenizer_);
        }
        const double t_prefill_end = wall_seconds();

        std::string generated;
        const int32_t eos = tokenizer_->eos_id();
        uint32_t next_token = step.argmax_token >= 0 ? static_cast<uint32_t>(step.argmax_token)
                                                     : static_cast<uint32_t>(eos);
        uint64_t decode_ops = 0;
        int decoded = 0;
        for (int i = 0; i < options.max_tokens; ++i) {
            const uint32_t feed = next_token;
            step = executor_->forward_one_token(feed);
            if (!step.ok) throw std::runtime_error("decode failed");
            decode_ops += step.ops_executed;
            const int32_t new_argmax = step.argmax_token >= 0 ? step.argmax_token : eos;
            if (dump) dump->record(static_cast<int>(prompt_tokens.size() + i),
                                   "decode", static_cast<int32_t>(feed),
                                   *executor_, *tokenizer_);
            next_token = static_cast<uint32_t>(new_argmax);
            ++decoded;
            if (next_token == static_cast<uint32_t>(eos)) break;
            const std::string piece = tokenizer_->decode_one(new_argmax);
            generated += piece;
            if (on_text && !piece.empty()) on_text(piece);
        }
        const double t_decode_end = wall_seconds();

        st = device_->end();
        if (!st.ok) throw std::runtime_error(st.message);

        const double prefill_s = std::max(t_prefill_end - t_prefill_start, 1e-9);
        const double decode_s = std::max(t_decode_end - t_prefill_end, 1e-9);
        std::ostringstream msg;
        msg << "native generate:"
            << " prompt_tokens=" << prompt_tokens.size()
            << " prefill=" << fmt_seconds(prefill_s);
        if (!prompt_tokens.empty()) {
            msg << " (" << std::fixed << std::setprecision(2)
                << (prompt_tokens.size() / prefill_s) << " tok/s)";
        }
        msg << " decoded=" << decoded
            << " decode=" << fmt_seconds(decode_s);
        if (decoded > 0) {
            msg << " (" << std::fixed << std::setprecision(2)
                << (decoded / decode_s) << " tok/s)";
        }
        msg << " prefill_ops=" << prefill_ops << " decode_ops=" << decode_ops;
        log(msg.str());

        return generated;
    }

private:
    static std::string fmt_seconds(double s) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3) << s << "s";
        return ss.str();
    }

    void log(const std::string &line) const {
        std::cerr << "[qw3] " << line << "\n";
    }

    EngineOptions options_;
    std::unique_ptr<QwenNativeModel> model_;
    std::unique_ptr<DeviceBackend> device_;
    std::unique_ptr<QwenWeights> weights_;
    std::unique_ptr<QwenExecutor> executor_;
    std::unique_ptr<QwenTokenizer> tokenizer_;
};

} // namespace

std::unique_ptr<Backend> make_qwen_native_backend() {
    return std::make_unique<QwenNativeBackend>();
}

} // namespace qw3
