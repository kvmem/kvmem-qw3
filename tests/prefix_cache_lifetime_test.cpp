#include "global_kv_page_pool.hpp"
#include "request_cancellation.hpp"
#include <iostream>
#include <stdexcept>

static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
int main() {
    // A 144K checkpoint, active reader and 16K reserve fit in a 256K pool
    // because the checkpoint/reader alias pages. New checkpoints pin the same
    // pages; deleting an old checkpoint must not free the new one's prefix.
    qw3::GlobalKvPagePool pool(16384, 16);
    std::vector<int32_t> old, suffix;
    for (int i = 0; i < 9000; ++i) old.push_back(pool.allocate_physical_page());
    pool.pin_pages(old);
    pool.pin_pages(old);
    require(pool.pinned_pages() == 9000, "count physical pages, not references");
    for (int i = 0; i < 1024; ++i) suffix.push_back(pool.allocate_physical_page());
    pool.unpin_pages(old); // evict old checkpoint
    pool.release_physical_pages(old);
    require(pool.used_pages() == 10024, "eviction freed active/shared prefix");
    pool.release_physical_pages(suffix);
    pool.release_physical_pages(suffix); // idempotent cleanup
    require(pool.used_pages() == 9000, "duplicate release corrupted free list");
    pool.unpin_pages(old);
    pool.release_physical_pages(old);
    require(pool.free_pages() == 16384, "last owner did not release pages");
    // Eviction may make progress without freeing any physical page.
    qw3::GlobalKvPagePool tiny(1, 16);
    std::vector<int32_t> page{tiny.allocate_physical_page()};
    tiny.pin_pages(page);
    tiny.pin_pages(page);
    int evicted = 0;
    tiny.set_evict_callback([&] {
        ++evicted;
        tiny.unpin_pages(page);
        tiny.release_physical_pages(page);
        return 1u;
    });
    require(tiny.allocate_physical_page() == page[0] && evicted == 2,
            "shared checkpoint eviction stopped before reclaiming page");
    qw3::detail::RequestCancellationRegistry registry;
    auto lease = registry.begin("cancel-test");
    auto gpu_lease = lease;
    auto state = registry.cancel("cancel-test");
    require(state && state->cancelled.load(), "cancel flag missing");
    lease.reset();
    require(!state->completed.load(), "acknowledged before GPU teardown");
    gpu_lease.reset();
    require(state->completed.load(), "completion not acknowledged");
    require(!registry.cancel("unknown"), "unknown cancellation falsely acknowledged");
    std::cout << "prefix pool and cancellation lifetime checks passed\n";
}
