#include "continuous_batch_utils.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

using namespace qw3::detail;

static void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

template <class F> static void rejects(F f) {
    try { f(); } catch (const std::runtime_error &) { return; }
    throw std::runtime_error("invalid batch metadata was accepted");
}

int main() {
    try {
        check(batch_needs_logits(1.0f, 0, 1), "sampling lost its logits");
        check(batch_needs_logits(0, -0.2f, 1), "presence penalty lost its logits");
        check(batch_needs_logits(0, 0, 1.1f), "repetition penalty lost its logits");
        check(!batch_needs_logits(0, 0, 1), "greedy should not copy logits");
        check(!batch_needs_logits(0, 0, 0), "disabled repetition penalty");

        // Two speculative jobs of unequal lengths: 2 and 3 target rows. Job
        // order need not match active request indices, and the bonus row must
        // remain attached to its own request after a different row is consumed.
        const std::vector<float> matrix = {
            9, 1, 0,  1, 8, 0,
            0, 0, 7,  0, 6, 0,  5, 0, 0};
        auto first_bonus = batch_logits_row(matrix, 3, 1);
        auto second_first = batch_logits_row(matrix, 3, 2);
        auto second_bonus = batch_logits_row(matrix, 3, 4);
        check(first_bonus == std::vector<float>({1, 8, 0}), "first bonus row");
        check(second_first == std::vector<float>({0, 0, 7}), "second request start");
        check(second_bonus == std::vector<float>({5, 0, 0}), "second bonus row");
        first_bonus[0] = -100; // Penalties must not mutate another row/the source.
        check(matrix[3] == 1 && second_first[0] == 0, "logits ownership");
        rejects([&] { batch_logits_row(matrix, 3, 5); });
        rejects([&] { batch_logits_row(matrix, 3, UINT64_MAX); });
        rejects([&] { batch_logits_row(matrix, 0, 0); });
        rejects([&] { batch_logits_row(matrix, 4, 0); });

        const int32_t history[] = {11, 4, 27, 8};
        const int32_t window[] = {27, 11, 8};
        auto dense = decode_page_span(31, 16, history, 4, false, 0, nullptr, 0);
        check(dense.position == 31 && dense.seq_len == 32 && dense.pages == 2 &&
              dense.last_page_len == 16 && dense.page_indices[1] == 4,
              "dense page boundary");
        auto sparse = decode_page_span(1ULL << 40, 16, history, 4, true, 32, window, 3);
        check(sparse.position == 32 && sparse.seq_len == 33 && sparse.pages == 3 &&
              sparse.last_page_len == 1 && sparse.page_indices[0] == 27 &&
              sparse.page_indices[2] == 8, "compact window used historical positions/pages");
        // Different request budget => different selected page count, even in
        // the same batch and with the same physical arena.
        auto short_row = decode_page_span(1024, 16, history, 4, true, 7, window, 1);
        check(short_row.pages == 1 && short_row.seq_len == 8, "per-request window length");
        rejects([&] { decode_page_span(1ULL << 40, 16, history, 4, false, 0, nullptr, 0); });
        rejects([&] { decode_page_span(0, 16, history, 4, true, 32, window, 2); });
        rejects([&] { decode_page_span(0, 0, history, 4, false, 0, nullptr, 0); });
        const int32_t missing[] = {-1};
        rejects([&] { decode_page_span(0, 16, missing, 1, false, 0, nullptr, 0); });
        std::cout << "continuous batch routing and page metadata passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
