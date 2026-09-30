#include "host_io_buffer_pool.hpp"
#include <atomic>
#include <cstring>
#include <future>
#include <iostream>
#include <thread>

using qw3::detail::HostIoBufferPool;
static void check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
struct Buffer : qw3::HostBuffer {
    std::atomic<unsigned> &live;
    Buffer(uint64_t n, std::atomic<unsigned> &counter) : live(counter) {
        bytes = n; data = ::operator new(n); ++live;
    }
    ~Buffer() override { ::operator delete(data); --live; }
};
int main() try {
    std::atomic<unsigned> live{0}, drains{0};
    auto allocate = [&](uint64_t n, const char *) { return std::make_unique<Buffer>(n, live); };
    {
        HostIoBufferPool pool(allocate, 128, [&] { ++drains; return true; });
        auto a = pool.acquire(64, "a"), b = pool.acquire(64, "b");
        check(a->data != b->data, "live requests share storage");
        std::memset(a->data, 0x35, 64); const void *address = a->data; a.reset();
        check(drains == 1, "storage returned before transfer drain");
        a = pool.acquire(64, "reuse");
        check(a->data == address && static_cast<unsigned char *>(a->data)[63] == 0x35,
              "lease failed to preserve and recycle backing bytes");
        auto extra = pool.acquire(64, "extra");
        a.reset(); b.reset(); extra.reset();
        check(live == 2 && pool.stats().cached_bytes == 128, "idle cap was exceeded");
        auto oversized = pool.acquire(256, "oversized"); oversized.reset();
        check(live == 2, "oversized idle buffer retained");
        check(pool.stats().allocations == 4 && pool.stats().reuses == 1, "pool counters wrong");
        check(!pool.acquire(0, "zero"), "zero allocation returned a lease");
    }
    check(live == 0, "pool destruction leaked idle storage");
    {
        std::unique_ptr<qw3::HostBuffer> surviving;
        { HostIoBufferPool pool(allocate, 64); surviving = pool.acquire(64, "live"); }
        check(live == 1, "pool destroyed a live lease"); surviving.reset();
        check(live == 0, "orphaned lease leaked");
    }
    {
        HostIoBufferPool pool(allocate, 64, [] { return false; });
        pool.acquire(64, "failed drain").reset();
        check(live == 0 && pool.stats().cached_bytes == 0, "failed drain buffer was recycled");
    }
    {
        HostIoBufferPool pool(allocate, 256);
        std::vector<std::future<void>> work;
        for (unsigned i = 0; i < 8; ++i) work.push_back(std::async(std::launch::async, [&, i] {
            for (unsigned j = 0; j < 1000; ++j) {
                auto b = pool.acquire(64, "concurrent");
                std::memset(b->data, i, 64);
                std::this_thread::yield();
                for (unsigned k = 0; k < 64; ++k)
                    check(static_cast<unsigned char *>(b->data)[k] == i, "concurrent lease alias");
            }
        }));
        for (auto &f : work) f.get();
        check(pool.stats().cached_bytes <= 256, "concurrent releases exceeded cap");
    }
    check(live == 0, "lease test leaked storage");
    std::cout << "host I/O pool: ownership, bounded reuse, transfer drain, teardown, concurrency PASS\n";
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
