#pragma once

#include "qw3/qw3.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace qw3 {

// Ingest a corpus once into a durable context archive. The harness prefills in
// append-only turns, snapshots recurrent state onto the ladder, and seals the
// archive at the last complete raw-K chunk. Interrupted builds resume from the
// highest ladder point whose payload is fully durable.
struct KvMemArchiveBuildConfig {
    enum class PrefillWindow {
        Pressure,
        SemanticChunk,
    };

    // UTF-8 text to ingest. Empty uses token_input_path when supplied,
    // otherwise a deterministic synthetic corpus used by end-to-end tests.
    std::string input_path;
    // Optional output of `qw3 tokenize --token-output FILE`. When present the
    // archive builder consumes the exact uint32 token stream and avoids
    // tokenizing a multi-million-token corpus a second time.
    std::string token_input_path;
    // Stop after this many tokens. 0 ingests the whole input.
    uint64_t tokens = 0;
    // Tokens between ladder points. 0 derives one from the raw-K chunk stride.
    uint64_t ladder_tokens = 0;
    // Preserve a non-aligned corpus tail by appending newline tokens through
    // the next complete raw-K chunk.  This is useful for real benchmark rows;
    // the default remains truncation for archive-format compatibility tests.
    bool pad_final_chunk = false;
    // Optional one-shot semantic-chunk construction. Pressure remains the
    // archive compatibility default. SemanticChunk runs every physical
    // prefill chunk after semantic_start_tokens provisionally, selects a
    // bounded historical window with that complete chunk as the query, then
    // replays the chunk before the archive is sealed.
    PrefillWindow prefill_window = PrefillWindow::Pressure;
    uint32_t semantic_start_tokens = 0;
    uint32_t semantic_query_tokens = 0;
};

// Attach a sealed archive and ask questions against it without recomputing the
// context. Every question runs from the same attached prefix, which is the
// point: it isolates the cost and the answer quality of the question itself.
struct KvMemArchiveRunConfig {
    struct Question {
        std::string content;
        // Optional UTF-8 byte offsets into content.  When present, only this
        // substring is the retrieval score query; the complete content remains
        // the answer-producing replay span.  Absence preserves the legacy
        // behavior where the complete content is both score and replay span.
        uint64_t query_content_start = 0;
        uint64_t query_content_end = 0;
        bool has_query_content_span = false;
    };

    enum class QuestionFormat {
        // Backward-compatible diagnostic mode: append the supplied bytes as-is
        // and use every appended token as the retrieval query.
        Raw,
        // Append a complete Qwen user turn plus an assistant generation header.
        // Only the user content (not the role/control tokens) is scored as the
        // retrieval query.  The second variant leaves the think block closed.
        QwenChatThinking,
        QwenChatNoThinking,
        // The archived prefix already contains `<|im_start|>user\n` and the
        // context but deliberately leaves that user turn open. Append the
        // question, close the turn, and open a non-thinking assistant reply.
        // This preserves benchmarks whose canonical full-context prompt puts
        // context and question in one user message while still allowing many
        // frozen branches from the same archived context.
        QwenUserContinuationNoThinking,
    };

    // Attach only the first N tokens. 0 attaches the whole archive. Any N is
    // allowed; the nearest ladder point at or below it is restored and the
    // residual is re-prefilled from the archived token stream.
    uint64_t tokens = 0;
    std::vector<Question> questions;
    int max_tokens = 128;
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 0;
    int thinking_budget = 0;
    QuestionFormat question_format = QuestionFormat::Raw;
    // Default off. The mutually-exclusive CLI selectors either retain top
    // original-query rows from a reversible attention probe, or retain only a
    // compact query generated after reversible private reasoning.
    bool query_attention_probe = false;
    bool query_guided_query = false;
    uint32_t query_probe_tokens = 100;
    uint32_t query_score_tokens = 64;
    // Optional machine-readable result stream. One JSON object is flushed
    // after every completed question so long benchmark rows remain auditable
    // even if a later question fails.
    std::string results_path;
    // Emit each answer plus a per-question timing line.
    bool verbose = true;
};

int run_kvmem_archive_build(EngineOptions engine,
                            const KvMemArchiveBuildConfig &cfg);
int run_kvmem_archive_query(EngineOptions engine,
                            const KvMemArchiveRunConfig &cfg);
int run_kvmem_archive_info(const std::string &dir);

} // namespace qw3
