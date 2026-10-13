#include "qwen_executor.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <sys/resource.h>

using namespace qw3;

// Explicit, model-dependent diagnostic; never registered with CTest. It uses
// real attention/recurrent kernels but synthetic teacher-forced tokens, so its
// output is a storage/continuation check, not a task-quality measurement.
namespace qw3 {
struct KvPageTableTestAccess {
    static void decode(QwenExecutor &e, uint32_t current, uint32_t count,
                       uint32_t &cycles, uint32_t &accepted_drafts, uint32_t &rejected_rows) {
        QwenExecutor::StateCheckpointSet checkpoints;
        uint32_t generated = 0;
        while (generated < count) {
            const auto base = e.position();
            const auto rope = e.kvmem_active_
                ? (e.input_mrope_positions_.empty() ? e.window_query_pos_ : e.window_mrope_next_) : base;
            const auto drafts = e.forward_mtp_draft_chain_with_prefix_device(
                current, std::min(4u, count - generated - 1));
            std::vector<uint32_t> input{current};
            for (const auto &draft : drafts) {
                if (!draft.ok || draft.argmax_token < 0) throw std::runtime_error("MTP decode draft failed");
                input.push_back(draft.argmax_token);
            }
            NativeExecutorReport result, prefix;
            if (input.size() == 1) {
                result = e.forward_one_token(current);
                prefix = e.prime_mtp_prefix_from_current_at(current, base, rope);
            } else {
                std::vector<DeviceArgmax> next;
                result = e.forward_n_tokens(input, true, &next, &checkpoints, input.size() - 1, false);
                if (!result.ok || next.size() != input.size()) throw std::runtime_error("MTP decode verify failed");
                if (!cycles) {
                    // Identical append serials in two sessions must not make
                    // their recurrent/KV rollback checkpoints interchangeable.
                    const auto owner = checkpoints.repository_owner;
                    const auto position = e.position();
                    if (owner != e.block_store_->repository())
                        throw std::runtime_error("verification checkpoint has no repository owner");
                    checkpoints.repository_owner = std::make_shared<KvMemRepository>(
                        KvMemRepository::Config{2, 32});
                    bool rejected = false;
                    try { e.restore_state_checkpoint(checkpoints, 0); }
                    catch (const std::runtime_error &error) {
                        rejected = std::string(error.what()).find("does not belong") != std::string::npos;
                    }
                    checkpoints.repository_owner = owner;
                    if (!rejected || e.position() != position)
                        throw std::runtime_error("foreign repository checkpoint was not rejected before restore");
                    std::cout << "repository_checkpoint foreign_owner_rejected=1\n";
                }
                uint32_t accepted = 1;
                while (accepted < input.size() && next[accepted - 1].token == int32_t(input[accepted])) ++accepted;
                if (accepted < input.size()) e.restore_state_checkpoint(checkpoints, accepted - 1);
                rejected_rows += input.size() - accepted;
                accepted_drafts += accepted - 1;
                result.argmax_token = next[accepted - 1].token;
                input.resize(accepted);
                prefix = e.prime_mtp_prefix_from_last_batch_at(input, base, rope);
            }
            if (!result.ok || result.argmax_token < 0 || !prefix.ok)
                throw std::runtime_error("MTP accepted decode/prefix rebuild failed");
            e.kvmem_register_append(input.size());
            generated += input.size(); ++cycles;
            current = result.argmax_token;
            if (!e.input_mrope_positions_.empty() && e.kvmem_active_ &&
                e.window_mrope_next_ != rope + input.size())
                throw std::runtime_error("MTP draft/priming advanced the target visual tail twice");
            if (e.mtp_prefix_len_ != e.position() || e.block_store_->history_end() != e.position())
                throw std::runtime_error("MTP decode acceptance lost its committed position");
        }
        if (!e.kvmem_finalize_content_index_from_raw_k(e.position()))
            throw std::runtime_error("MTP decode could not publish the accepted raw-K index");
    }
    using Bytes = std::vector<std::pair<std::string, std::vector<uint8_t>>>;
    static Bytes replay_bytes(QwenExecutor &e, uint64_t begin) {
        Bytes out;
        const auto tensor = [&](const std::string &name, const DeviceTensor *value) {
            if (!value) return;
            auto &bytes = out.emplace_back(name, std::vector<uint8_t>(value->count * value->elem_size)).second;
            const auto s = e.backend_.copy_bytes_to_host(*value, bytes.data(), 0, bytes.size());
            if (!s.ok) throw std::runtime_error(s.message);
        };
        tensor("hidden", e.h_.get()); tensor("mtp-prefix-hidden", e.mtp_prefix_h_.get());
        for (size_t i = 0; i < e.recurrent_states_.size(); ++i) {
            tensor("recurrent-" + std::to_string(i), e.recurrent_states_[i].get());
            tensor("conv-" + std::to_string(i), e.conv_states_[i].get());
        }
        const auto ids = e.block_store_->history_span_blocks(begin, begin + 32, true);
        if (ids.size() != 1) throw std::runtime_error("replay byte oracle expects one aligned suffix block");
        const auto storage = e.block_store_->blocks()[ids[0]].orig_pos_start;
        for (const bool mtp : {false, true}) {
            const auto size = (mtp ? 1 : e.kvmem_raw_layers_.size()) * 32 * e.kvmem_raw_k_row_bytes_;
            auto &raw = out.emplace_back(mtp ? "mtp-raw-K" : "main-raw-K", std::vector<uint8_t>(size)).second;
            e.kvmem_read_raw_k_block(storage, 32, raw.data(), mtp);
        }
        const auto cache = [&](const std::string &name, DeviceTensor &value, const QwenExecutor::KvPageTable &pages) {
            const auto page_bytes = uint64_t(pages.page_size) * e.model_.config().n_kv_heads *
                                    e.model_.config().head_dim * value.elem_size;
            auto &bytes = out.emplace_back(name, std::vector<uint8_t>(32 / pages.page_size * page_bytes)).second;
            for (uint32_t i = 0; i < 32 / pages.page_size; ++i) {
                const auto page = pages.pages.at(storage / pages.page_size + i);
                if (page < 0) throw std::runtime_error("replay byte oracle lost suffix GPU page");
                const auto s = e.backend_.copy_bytes_to_host(value, bytes.data() + i * page_bytes,
                                                            page * page_bytes, page_bytes);
                if (!s.ok) throw std::runtime_error(s.message);
            }
        };
        for (auto il : e.kvmem_raw_layers_) {
            cache("main-K-" + std::to_string(il), e.k_cache(il), e.kv_pages_);
            cache("main-V-" + std::to_string(il), e.v_cache(il), e.kv_pages_);
        }
        cache("mtp-K", e.mtp_k_cache(), e.mtp_kv_pages_);
        cache("mtp-V", e.mtp_v_cache(), e.mtp_kv_pages_);
        return out;
    }
    static void identity_checkpoint(const QwenNativeModel &model, const QwenWeights &weights,
                                    DeviceBackend &backend) {
        for (const uint32_t boundary : {64u, 416u, 800u}) {
            QwenExecutor e(model, weights, backend, 8192);
            KvMemStoreConfig cfg;
            cfg.block_tokens = 32; cfg.repository_blocks = 256;
            cfg.select_budget = cfg.prefill_budget = 1024; cfg.gen_budget = 512;
            cfg.sink_blocks = 1; cfg.recent_blocks = 4;
            cfg.immutable_source_k = true; cfg.index_placement = KvMemIndexPlacement::CPU;
            cfg.mtp_enabled = true; cfg.cpu_tier_bytes = 1ull << 30;
            cfg.gpu_memory_ratio = 0; cfg.estimated_gpu_block_capacity = 64;
            e.configure_kvmem(cfg);
            e.set_kvmem_enabled(true);
            e.kvmem_gc_begin_step();
            e.kvmem_set_query_span(0, 0, boundary + 46, 0, false, true);
            const auto forward = [&](uint32_t count, uint32_t value) {
                std::vector<uint32_t> input(count, value);
                const auto base = e.position();
                e.kvmem_set_defer_prefill_pressure(true);
                auto result = e.forward_n_tokens(input, true);
                if (!result.ok || result.argmax_token < 0)
                    throw std::runtime_error("identity oracle main forward failed");
                auto mtp = e.prime_mtp_prefix_from_last_batch_at(input, base, e.last_forward_rope_base());
                if (!mtp.ok) throw std::runtime_error("identity oracle MTP prime failed");
                e.kvmem_finish_deferred_prefill_pressure();
                e.kvmem_set_defer_prefill_pressure(false);
                e.kvmem_register_append(count);
                if (!e.kvmem_finalize_content_index_from_raw_k(e.position()))
                    throw std::runtime_error("identity oracle index not finalized");
                return result;
            };
            for (uint32_t done = 0; done < boundary;) {
                const uint32_t count = std::min(128u, boundary - done);
                forward(count, 100 + done); done += count;
            }
            e.kvmem_reselect_prefill_pressure();
            auto checkpoint = e.snapshot_state();
            const auto prefix_bytes = replay_bytes(e, boundary - 32);
            const auto expected = forward(46, 401);
            e.kvmem_reselect_prefill_pressure();
            forward(15, 500); // a short generated tail crossing no new safe P
            if (checkpoint.kvmem_selection_generation == e.kvmem_selection_generation())
                throw std::runtime_error("identity oracle did not exercise an epoch change");
            const auto prefix_position = checkpoint.window_query_pos;
            checkpoint.window_query_pos -= 32;
            if (e.kvmem_rebase_identity_checkpoint(checkpoint))
                throw std::runtime_error("sparse checkpoint was treated as dense");
            checkpoint.window_query_pos = prefix_position;
            if (!e.kvmem_rebase_identity_checkpoint(checkpoint))
                throw std::runtime_error("intact identity prefix could not rebase");
            e.restore_state(checkpoint);
            const auto restored_bytes = replay_bytes(e, boundary - 32);
            if (restored_bytes.size() != prefix_bytes.size())
                throw std::runtime_error("identity oracle state shape changed");
            for (size_t i = 0; i < prefix_bytes.size(); ++i)
                if (prefix_bytes[i] != restored_bytes[i])
                    throw std::runtime_error("identity prefix restore changed " + prefix_bytes[i].first);
            const auto actual = forward(46, 401);
            if (actual.argmax_token != expected.argmax_token || actual.argmax_logit != expected.argmax_logit)
                throw std::runtime_error("identity prefix restore changed continuation logits");
            e.kvmem_gc_end_step();
            std::cout << "identity_checkpoint boundary=" << boundary
                      << " state_bytes=exact continuation=exact sparse_rejected=1\n";
        }
    }

    static void run(const QwenNativeModel &model, const QwenWeights &weights,
                    DeviceBackend &backend, uint32_t rounds, bool mtp, bool guided, bool replay, bool generated_decode, uint64_t history_origin) {
        constexpr uint32_t capacity = 8192, batch = 128;
        QwenExecutor e(model, weights, backend, capacity);
        KvMemStoreConfig cfg;
        cfg.block_tokens = 32; cfg.repository_blocks = capacity / cfg.block_tokens;
        cfg.select_budget = 1024; cfg.prefill_budget = 1024; cfg.gen_budget = 512;
        cfg.sink_blocks = 1; cfg.recent_blocks = 4;
        cfg.immutable_source_k = true; cfg.index_placement = KvMemIndexPlacement::CPU;
        cfg.mtp_enabled = mtp;
        cfg.cpu_tier_bytes = 1ull << 30; cfg.gpu_memory_ratio = 0;
        cfg.estimated_gpu_block_capacity = 64;
        if (const char *layout = std::getenv("QW3_TEST_REPOSITORY_INDEX")) {
            const std::string mode(layout);
            if (mode == "subblock4") {
                cfg.retrieval_method = KvMemRetrievalMethod::SubBlockMeanK;
                cfg.n_subblocks = 4;
            } else if (mode == "fixed4" || mode == "adaptive") {
                cfg.prototype_mode = mode == "fixed4" ? KvMemPrototypeMode::KeyDirectionFixed4
                                                      : KvMemPrototypeMode::KeyDirectionAdaptive;
                cfg.n_subblocks = 4;
            } else throw std::invalid_argument("unknown repository probe index layout");
        }
        e.configure_kvmem(cfg);
        e.set_kvmem_enabled(true);
        if (history_origin) {
            if (history_origin % cfg.block_tokens || history_origin < cfg.block_tokens ||
                uint64_t(rounds) * 160 > UINT64_MAX - history_origin)
                throw std::invalid_argument("probe origin must be aligned and leave room for all rounds");
            e.prepare_runtime_state();
            if (mtp) e.ensure_mtp_scratch();
            // Simulate a discarded old history, without pretending to have
            // computed it. Only the subsequently appended tokens are tested.
            auto repo = e.block_store_->repository();
            auto ghost = repo->try_reserve(history_origin - cfg.block_tokens, cfg.block_tokens).commit();
            const auto handle = ghost.handle();
            e.block_store_->register_repository_block(std::move(ghost), 0);
            e.block_store_->set_block_tier(handle.slot, KvTier::CPU, -1, -1, handle.generation);
            if (!repo->retire_if_unpinned(handle)) throw std::runtime_error("probe origin marker stayed pinned");
            e.kvmem_repository_gc_->notify_progress();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (repo->ready_slots() < cfg.repository_blocks && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (repo->ready_slots() != cfg.repository_blocks) throw std::runtime_error("probe origin marker was not reclaimed");
            e.position_ = e.kvmem_registered_pos_ = history_origin;
            if (mtp) e.mtp_prefix_len_ = history_origin;
            e.kvmem_active_ = true;
        }
        const bool multimodal = std::getenv("QW3_TEST_MULTIMODAL") != nullptr;
        if (multimodal && history_origin) throw std::invalid_argument("visual probe uses real zero-origin inputs");
        std::vector<uint32_t> visual_tokens;
        std::vector<std::array<uint32_t, 3>> visual_positions;
        std::vector<GenerationOptions::InputEmbeddingOverride> visual_rows;
        uint32_t replay_count = 0, replay_byte_checks = 0;
        uint32_t decode_tokens = 0, decode_cycles = 0, accepted_drafts = 0, rejected_rows = 0;
        std::cout << "round,history,ready,reclaimed,physical_blocks,cpu_bytes,index_bytes,gpu_kv_bytes,rss_kib,elapsed_ms,gc_ticks,gc_protected,gc_races,gc_max_tick_ns,gc_policy_fallback,gc_incomplete_backing,gc_reader_conflicts,replays,replay_byte_checks,decode_tokens,decode_cycles,accepted_drafts,rejected_rows\n";
        for (uint32_t round = 0; round < rounds; ++round) {
            const auto start = std::chrono::steady_clock::now();
            const uint64_t begin = e.position(), end = begin + batch;
            const bool do_guided = guided && round >= 8 && round % 4 == 0;
            const bool do_replay = replay && do_guided;
            e.kvmem_set_pin_from_block(UINT64_MAX);
            e.kvmem_gc_begin_step();
            e.kvmem_set_query_span(end - 16, end, end, 0, round != 0);
            if (round == 0 && history_origin) e.kvmem_qc_captured_tokens_ = history_origin;
            std::vector<uint32_t> tokens(batch);
            for (uint32_t i = 0; i < batch; ++i) tokens[i] = 100 + (begin - history_origin + i) % 1000;
            if (multimodal) {
                visual_tokens.resize(end, 100); visual_positions.resize(end);
                for (size_t i = 0; i < end; ++i) visual_positions[i] = {uint32_t(2*i), uint32_t(2*i+i%4), uint32_t(2*i+i%2)};
                for (uint32_t offset : {40u, 41u, 42u, 43u, 112u, 113u, 114u, 115u}) {
                    GenerationOptions::InputEmbeddingOverride row;
                    row.token_id = 0x80000001u + visual_rows.size(); row.source_token_id = 100;
                    row.position = visual_positions[begin+offset];
                    row.embedding.resize(model.config().n_embd);
                    for (size_t i = 0; i < row.embedding.size(); ++i) row.embedding[i] = 0.05f * std::sin(float(i+offset));
                    tokens[offset] = row.token_id; visual_rows.push_back(std::move(row));
                }
                std::copy(tokens.begin(), tokens.end(), visual_tokens.begin()+begin);
                e.set_input_embedding_overrides(visual_rows, visual_tokens, visual_positions);
            }
            const auto forward = [&](const std::vector<uint32_t> &input) {
                const auto base = e.position();
                e.kvmem_set_defer_prefill_pressure(mtp);
                auto result = e.forward_n_tokens(input, true);
                if (!result.ok || result.argmax_token < 0 || !std::isfinite(result.argmax_logit))
                    throw std::runtime_error("model probe target forward failed or produced invalid logits");
                if (mtp) {
                    auto prefix = e.prime_mtp_prefix_from_last_batch_at(input, base, e.last_forward_rope_base());
                    if (!prefix.ok) {
                        for (const auto &message : prefix.missing_kernels) std::cerr << message << '\n';
                        throw std::runtime_error("model probe MTP prefix failed");
                    }
                    if (e.mtp_prefix_len_ != base + input.size())
                        throw std::runtime_error("model probe MTP prefix lost its historical coordinate");
                    e.kvmem_finish_deferred_prefill_pressure();
                    e.kvmem_set_defer_prefill_pressure(false);
                }
                e.kvmem_register_append(input.size());
                return result;
            };
            QwenExecutor::StateSnapshot boundary;
            NativeExecutorReport result;
            if (do_replay) {
                forward({tokens.begin(), tokens.end() - 32});
                if (!e.kvmem_finalize_content_index_from_raw_k(end - 32))
                    throw std::runtime_error("model probe cannot finalize replay boundary");
                e.capture_state(boundary);
                result = forward({tokens.end() - 32, tokens.end()});
            } else {
                result = forward(tokens);
            }
            if (!e.kvmem_finalize_content_index_from_raw_k(end))
                throw std::runtime_error("model probe cannot publish the actual raw-K index");
            if (!e.kvmem_query_capture_complete())
                throw std::runtime_error("model probe lost query capture");
            if (do_guided) {
                QwenExecutor::StateSnapshot snapshot;
                e.capture_transient_state(snapshot);
                if (!e.kvmem_begin_guided_query_probe(8))
                    throw std::runtime_error("model probe cannot begin a guided query");
                e.kvmem_set_guided_query_capture(true);
                QwenExecutor::StateCheckpointSet checkpoints;
                uint32_t generated = 0, next = static_cast<uint32_t>(result.argmax_token);
                while (generated < 8) {
                    std::vector<uint32_t> accepted;
                    auto step = e.forward_guided_query_mtp(next, 4, 8 - generated, {-1, -1, -1}, checkpoints, accepted);
                    if (!step.ok || step.argmax_token < 0 || accepted.empty())
                        throw std::runtime_error("model probe guided MTP failed");
                    generated += accepted.size(); next = static_cast<uint32_t>(step.argmax_token);
                }
                if (e.kvmem_end_guided_query_probe() != generated)
                    throw std::runtime_error("model probe guided Q publication lost rows");
                e.restore_state(snapshot);
                if (e.position() != end || e.mtp_prefix_len_ != end ||
                    !e.kvmem_query_selection_ready() || e.kvmem_query_captured_tokens() != generated ||
                    !e.kvmem_content_index_covers(end))
                    throw std::runtime_error("model probe guided rollback lost query/index/model state");
            }
            if (do_replay) {
                e.kvmem_set_pin_from_block(boundary.position / cfg.block_tokens);
                if (!e.kvmem_publish_captured_prefix(boundary.position))
                    throw std::runtime_error("model probe cannot publish replay context index");
            }
            e.kvmem_reselect();
            if (do_replay) {
                std::vector<uint32_t> context;
                const auto ids = e.block_store_->history_span_blocks(0, end, false);
                for (auto id : ids) {
                    const auto &b = e.block_store_->blocks()[id];
                    if (b.history_pos_start < boundary.position) {
                        if (b.in_working_set) context.push_back(id);
                    } else if (!b.in_working_set) {
                        throw std::runtime_error("model probe query suffix lost selection protection");
                    }
                }
                QwenExecutor::StateSnapshot discarded_suffix;
                e.capture_transient_state(discarded_suffix);
                e.kvmem_begin_query_replay(boundary, context, false, true, true);
                if (e.position() != boundary.position || e.mtp_prefix_len_ != boundary.position)
                    throw std::runtime_error("model probe query replay did not rewind main/MTP state");
                result = forward({tokens.end() - 32, tokens.end()});
                e.kvmem_end_query_replay();
                if (!e.kvmem_query_selection_ready() || e.kvmem_query_captured_tokens() != 8 ||
                    !e.kvmem_content_index_covers(boundary.position) ||
                    e.block_store_->repository_checkpoint_valid(discarded_suffix.repository_checkpoint))
                    throw std::runtime_error("model probe replay lost Q/index or revived obsolete checkpoint");
                if (!e.kvmem_finalize_content_index_from_raw_k(end))
                    throw std::runtime_error("model probe cannot rebuild replayed raw-K suffix index");
                ++replay_count;
                if (round % 64 == 8) {
                    // Repeat identical computation from R with the SAME final
                    // context, but newly allocated suffix slots/pages. This
                    // checks actual state/K/V, not just successful execution.
                    const auto expected = replay_bytes(e, boundary.position);
                    const auto expected_result = result;
                    e.kvmem_begin_query_replay(boundary, context, false, true, true);
                    result = forward({tokens.end() - 32, tokens.end()});
                    e.kvmem_end_query_replay();
                    if (!e.kvmem_finalize_content_index_from_raw_k(end))
                        throw std::runtime_error("repeated replay could not publish its raw-K index");
                    const auto actual = replay_bytes(e, boundary.position);
                    if (expected.size() != actual.size()) throw std::runtime_error("replay oracle shape changed");
                    for (size_t i = 0; i < expected.size(); ++i)
                        if (expected[i] != actual[i])
                            throw std::runtime_error("replay byte mismatch in " + expected[i].first);
                    if (result.argmax_token != expected_result.argmax_token || result.argmax_logit != expected_result.argmax_logit)
                        throw std::runtime_error("replay changed the next-token prediction");
                    ++replay_count; ++replay_byte_checks;
                }
            }
            if (generated_decode) {
                decode(e, result.argmax_token, 32, decode_cycles, accepted_drafts, rejected_rows);
                decode_tokens += 32;
            }
            if (multimodal) {
                if (e.input_compact_mrope_positions_.size() != capacity || e.input_compact_mrope_valid_.size() != capacity)
                    throw std::runtime_error("visual coordinate arena grew with cumulative history");
                e.release_bounded_input_metadata();
                if (e.input_mrope_positions_.capacity() || e.input_multimodal_prompt_tokens_.capacity() || e.input_embedding_overrides_.capacity())
                    throw std::runtime_error("executor retained request-sized visual inputs");
            }
            e.kvmem_gc_end_step();
            auto status = backend.synchronize();
            if (!status.ok) throw std::runtime_error(status.message);
            const auto usage = e.kvmem_tier_usage();
            const auto index = e.kvmem_index_usage();
            const auto gc = e.kvmem_repository_gc_->stats();
            if (gc.failures) throw std::runtime_error("model probe encountered a background GC failure");
            struct rusage resource{}; getrusage(RUSAGE_SELF, &resource);
            std::cout << round << ',' << e.position() << ','
                      << e.block_store_->repository()->ready_slots() << ',' << gc.reclaimed << ','
                      << e.block_store_->block_count() << ',' << usage.external_physical_bytes << ','
                      << index.capacity_bytes << ',' << usage.gpu_used_bytes << ',' << resource.ru_maxrss << ','
                      << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()
                      << ',' << gc.ticks << ',' << gc.protected_blocks << ',' << gc.raced_updates << ',' << gc.max_tick_ns
                      << ',' << gc.policy_fallback_retired
                      << ',' << gc.incomplete_backing << ',' << gc.reader_conflicts << ',' << replay_count << ',' << replay_byte_checks
                      << ',' << decode_tokens << ',' << decode_cycles << ',' << accepted_drafts << ',' << rejected_rows
                      << std::endl;
            if (e.position() != end + (generated_decode ? 32 : 0) || e.block_store_->block_count() != cfg.repository_blocks)
                throw std::runtime_error("model probe lost position or grew its repository");
        }
    }
};
}

int main(int argc, char **argv) {
    try {
        if (argc < 2 || argc > 5) throw std::invalid_argument("usage: qw3-kvmem-repository-model-probe MODEL [ROUNDS=16] [mtp|guided-mtp|guided-mtp-replay|guided-mtp-replay-decode|identity-checkpoint] [HISTORY_ORIGIN=0]");
        const uint32_t rounds = argc >= 3 ? std::stoul(argv[2]) : 16;
        const bool identity = argc >= 4 && std::string(argv[3]) == "identity-checkpoint";
        const bool generated_decode = argc >= 4 && std::string(argv[3]) == "guided-mtp-replay-decode";
        const bool replay = generated_decode || (argc >= 4 && std::string(argv[3]) == "guided-mtp-replay");
        const bool guided = replay || (argc >= 4 && std::string(argv[3]) == "guided-mtp");
        const bool mtp = identity || guided || (argc >= 4 && std::string(argv[3]) == "mtp");
        if (argc >= 4 && !mtp) throw std::invalid_argument("unknown probe mode");
        setenv("QW3_PAGED_KV_PAGE_SIZE", "16", 1);
        setenv("QW3_KV_DTYPE", "fp8", 1);
        auto backend = make_cuda_device_backend(LinearBackend::Auto);
        if (!backend) throw std::runtime_error("CUDA backend unavailable");
        auto status = backend->begin();
        if (!status.ok) throw std::runtime_error(status.message);
        {
            QwenNativeModel model(open_model_source(argv[1]));
            QwenWeights weights(model, *backend, false, mtp);
            if (identity) KvPageTableTestAccess::identity_checkpoint(model, weights, *backend);
            else KvPageTableTestAccess::run(model, weights, *backend, rounds, mtp, guided, replay, generated_decode, argc == 5 ? std::stoull(argv[4]) : 0);
        }
        status = backend->end();
        if (!status.ok) throw std::runtime_error(status.message);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "model probe failed: " << e.what() << '\n'; return 1;
    }
}
