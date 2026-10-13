#include "flashinfer_decode_plan.hpp"
#include <iostream>
#include <stdexcept>

using qw3::detail::FlashinferDecodePlan;
void require(bool v, const char *why) { if (!v) throw std::runtime_error(why); }

int main() try {
    FlashinferDecodePlan p;
    require(p.update(31, 3, 32, 32, true), "first plan not built");
    // Causal lengths 32,33,34 consume 1,2,2 pages and the same split counts.
    require(p.page_indptr_offset == 5 && p.total_tiles == 5, "boundary layout wrong");
    auto at = [&](uint64_t offset, unsigned i) { return p.metadata.at(offset - p.page_indptr_offset + i); };
    require(at(p.page_indptr_offset, 0) == 0 && at(p.page_indptr_offset, 1) == 1 &&
            at(p.page_indptr_offset, 2) == 3 && at(p.page_indptr_offset, 3) == 5,
            "causal page spans wrong");
    require(at(p.last_page_len_offset, 0) == 32 && at(p.last_page_len_offset, 1) == 1 &&
            at(p.last_page_len_offset, 2) == 2, "partial last page wrong");
    require(p.output_indptr == std::vector<int32_t>({0, 1, 3, 5}), "split merge boundaries wrong");
    require(at(p.request_indices_offset, 4) == 2 && at(p.kv_tile_indices_offset, 4) == 1,
            "last split references wrong causal row");
    const auto original = p.metadata;
    require(!p.update(31, 3, 32, 32, true) && p.metadata == original, "same shape changed");
    require(p.update(31, 3, 32, 32, false) && p.total_tiles == 3,
            "partition mode reused old plan");
    require(p.update(32, 3, 32, 32, false), "position advance reused old plan");
    require(p.update(31, 3, 32, 32, true) && p.metadata == original, "rollback layout changed");
    require(p.update(31, 1, 32, 32, true) && p.page_indptr_offset == 0,
            "single row duplicated physical pages");
    require(p.update(131071, 8, 32, 512, true) && p.page_indptr_offset == 32775 &&
            p.output_indptr.front() == 0 && p.total_tiles == 2055,
            "128K boundary plan wrong");
    for (auto key : {std::array<uint32_t, 4>{0, 0, 32, 256}, {0, 1, 0, 256},
                     {0, 1, 32, 0}, {UINT32_MAX, 8, 32, 256},
                     {2147483640u, 7, 1, 1}}) {
        bool rejected = false;
        try { p.update(key[0], key[1], key[2], key[3], true); }
        catch (const std::exception &) { rejected = true; }
        require(rejected && !p.valid, "invalid/overflowing plan was reusable");
    }
    require(p.update(31, 3, 32, 32, true) && p.metadata == original, "error recovery changed plan");
    std::cout << "FlashInfer plan: causal pages, split boundaries, rollback, invalidation and overflow PASS\n";
    return 0;
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
