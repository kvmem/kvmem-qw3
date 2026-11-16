#include "kvmem_session_cache.hpp"

#include <iostream>
#include <type_traits>
#include <vector>

static void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Exception = std::exception, class Action>
static void rejects(Action &&action, const char *message) {
    bool rejected = false;
    try { action(); } catch (const Exception &) { rejected = true; }
    require(rejected, message);
}

struct Payload {
    int id;
    std::vector<int> &destroyed;
    std::vector<int> history;
    std::unique_ptr<int> gpu;
    int checkpoint = 0;
    Payload(int value, std::vector<int> &retired)
        : id(value), destroyed(retired), history{value, value + 1},
          gpu(std::make_unique<int>(value + 17)) {}
    ~Payload() { destroyed.push_back(id); }
};
using Cache = qw3::detail::KvmemSessionCache<Payload>;
static_assert(!std::is_copy_constructible<Cache::Active>::value);
static_assert(std::is_nothrow_move_constructible<Cache::Active>::value);
static_assert(std::is_nothrow_move_assignable<Cache::Active>::value);

static bool suspend(Payload &p) {
    require(bool(p.gpu), "suspending a payload that is already cold");
    p.checkpoint = *p.gpu;
    p.gpu.reset();
    return true;
}
static bool restore(Payload &p) {
    require(!p.gpu, "restoring a payload that is already warm");
    p.gpu = std::make_unique<int>(p.checkpoint);
    return true;
}

static void disabled_and_factory_failure() {
    std::vector<int> destroyed;
    Cache disabled;
    int factories = 0;
    auto factory = [&] { ++factories; return std::make_unique<Payload>(1, destroyed); };
    rejects<std::logic_error>([&] { disabled.create("A", factory, 0); }, "default cache was enabled");
    rejects<std::logic_error>([&] { disabled.acquire("A", restore, 0); }, "disabled acquire accepted");
    rejects<std::logic_error>([&] { disabled.cool_to(0, 0, suspend); }, "disabled suspend accepted");
    require(factories == 0 && disabled.size() == 0, "disabled cache ran a factory");
    rejects<std::invalid_argument>([] { Cache invalid({true, 0, 0}); }, "enabled unlimited histories accepted");

    Cache cache({true, 1, 0});
    rejects<std::runtime_error>([&] {
        cache.create("A", []() -> std::unique_ptr<Payload> { throw std::runtime_error("allocation failed"); }, 0);
    }, "factory exception was swallowed");
    rejects<std::invalid_argument>([&] {
        cache.create("A", [] { return std::unique_ptr<Payload>{}; }, 0);
    }, "null factory payload accepted");
    rejects<std::invalid_argument>([&] { cache.create("", factory, 0); }, "empty key accepted");
    require(cache.size() == 0 && factories == 0, "failed factory reserved history capacity");
    auto active = cache.create("A", factory, 0);
    require(active && factories == 1, "factory failure prevented later admission");
}

static void active_ownership_and_capacity() {
    std::vector<int> destroyed;
    Cache cache({true, 2, 0});
    auto a = cache.create("tenant-1/A", [&] { return std::make_unique<Payload>(1, destroyed); }, 0);
    auto b = cache.create("tenant-2/A", [&] { return std::make_unique<Payload>(2, destroyed); }, 0);
    auto *a_address = a.get();
    b.park(1);
    require(cache.stats().histories == 2 && cache.stats().active == 1,
            "active history was not included in total capacity");
    int factories = 0;
    rejects<std::runtime_error>([&] {
        cache.create("C", [&] { ++factories; return std::make_unique<Payload>(3, destroyed); }, 2);
    }, "full cache silently admitted or evicted history");
    rejects<std::logic_error>([&] {
        cache.create("tenant-1/A", [&] { ++factories; return std::make_unique<Payload>(3, destroyed); }, 2);
    }, "duplicate active owner admitted");
    rejects<std::logic_error>([&] { cache.acquire("tenant-1/A", restore, 2); }, "active history acquired twice");
    rejects<std::logic_error>([&] { cache.erase("tenant-1/A"); }, "active history erased");
    rejects<std::logic_error>([&] { cache.clear(); }, "clear erased an active history");
    require(!cache.peek_idle("tenant-1/A") && cache.peek_idle("tenant-2/A"), "peek exposed active ownership");
    require(cache.cool_to(0, 2, suspend) == 1 && a->gpu && a.get() == a_address,
            "pressure suspended or moved an active payload");
    require(cache.is_cold("tenant-2/A") && destroyed.empty() && factories == 0,
            "capacity refusal deleted a history or executed its factory");
    a.park(3);
    auto reused = cache.acquire("tenant-1/A", restore, 4);
    require(reused.get() == a_address, "warm resume replaced history owner");
    reused.park(5);
    require(cache.erase("tenant-2/A") && !cache.erase("missing"), "explicit idle deletion failed");
    require(destroyed == std::vector<int>{2}, "explicit deletion freed the wrong owner");
    cache.clear();
    require(destroyed == std::vector<int>({2, 1}), "history teardown leaked or doubled ownership");
}

static void many_histories_share_one_warm_slot() {
    std::vector<int> destroyed;
    Cache cache({true, 4, 0});
    for (int id = 0; id < 4; ++id) {
        require(cache.cool_to(0, id, suspend) == (id ? 1u : 0u), "old working set was not suspended");
        auto active = cache.create(std::to_string(id), [&] { return std::make_unique<Payload>(id, destroyed); }, id);
        active.park(id);
    }
    require(cache.size() == 4 && cache.stats().idle_cold == 3 && destroyed.empty(),
            "history count remained tied to warm slots");
    for (int round = 0; round < 3; ++round) {
        for (int id = 0; id < 4; ++id) {
            cache.cool_to(0, 10 + round * 4 + id, suspend);
            auto active = cache.acquire(std::to_string(id), restore, 20 + round * 4 + id);
            require(active->history == std::vector<int>({id, id + 1}) && *active->gpu == id + 17,
                    "cold resume lost history, continuation or tenant isolation");
            active.park(30 + round * 4 + id);
        }
    }
    require(cache.size() == 4 && destroyed.empty(), "rotation silently deleted an idle history");
    cache.clear();
    require(destroyed.size() == 4, "rotation leaked histories");
}

static void transition_failures_and_poison() {
    std::vector<int> destroyed;
    Cache cache({true, 2, 0});
    auto active = cache.create("A", [&] { return std::make_unique<Payload>(1, destroyed); }, 0);
    auto *address = active.get();
    active.park(1);
    require(cache.cool_to(0, 2, [](Payload &) { return false; }) == 0 && !cache.is_cold("A"),
            "suspend refusal changed residency");
    rejects<std::runtime_error>([&] {
        cache.cool_to(0, 2, [](Payload &) -> bool { throw std::runtime_error("I/O failure"); });
    }, "suspend exception was swallowed");
    require(cache.peek_idle("A") == address && address->gpu && cache.stats().idle_warm == 1,
            "suspend failure lost original warm owner");
    cache.cool_to(0, 2, suspend);
    rejects<std::runtime_error>([&] {
        cache.acquire("A", [](Payload &) { return false; }, 3);
    }, "restore refusal accepted an active owner");
    rejects<std::runtime_error>([&] {
        cache.acquire("A", [](Payload &) -> bool { throw std::runtime_error("GPU quota unavailable"); }, 3);
    }, "restore exception was swallowed");
    require(cache.peek_idle("A") == address && cache.stats().active == 0 && cache.is_cold("A"),
            "failed restoration lost or activated cold history");
    active = cache.acquire("A", restore, 4);
    require(active.get() == address && *active->gpu == 18, "retry did not restore the original history");
    active.invalidate();
    require(active.poisoned(), "invalid continuation was not marked");
    active.park(5);
    rejects<std::logic_error>([&] { cache.acquire("A", restore, 6); }, "poisoned history resumed");
    rejects<std::logic_error>([&] {
        cache.create("A", [&] { return std::make_unique<Payload>(2, destroyed); }, 6);
    }, "poisoned key was silently replaced");
    require(cache.is_poisoned("A") && cache.peek_idle("A") == address && destroyed.empty(),
            "poisoning silently discarded history");
    require(cache.erase("A") && destroyed == std::vector<int>{1}, "explicit poison cleanup failed");
}

static void ttl_lru_and_partial_success() {
    std::vector<int> destroyed;
    Cache cache({true, 3, 10});
    auto a = cache.create("A", [&] { return std::make_unique<Payload>(1, destroyed); }, 1);
    auto b = cache.create("B", [&] { return std::make_unique<Payload>(2, destroyed); }, 1);
    a.park(20); b.park(25);
    require(cache.cool_to(3, 19, suspend) == 0, "clock reversal underflowed TTL");
    require(cache.cool_to(3, 29, suspend) == 0, "TTL expired early");
    require(cache.cool_to(3, 30, suspend) == 1 && cache.is_cold("A") && !cache.is_cold("B"),
            "TTL did not suspend exactly the expired owner");
    a = cache.acquire("A", restore, 30);
    a.park(31); // B is now the oldest idle warm owner.
    require(cache.cool_to(1, 31, suspend) == 1 && cache.is_cold("B") && !cache.is_cold("A"),
            "pressure did not use idle return order");
    b = cache.acquire("B", restore, 32); b.park(32);
    rejects<std::runtime_error>([&] {
        cache.cool_to(0, 32, [](Payload &p) {
            if (p.id == 2) throw std::runtime_error("second owner cannot suspend");
            return suspend(p);
        });
    }, "second suspension exception was swallowed");
    require(cache.is_cold("A") && !cache.is_cold("B") && cache.stats().idle_warm == 1,
            "partial transition failure corrupted per-owner commits");
    require(destroyed.empty() && cache.size() == 2, "TTL or pressure deleted history");
    require(!cache.acquire("missing", restore, 33), "cache miss was not empty");
}

static void lease_moves_and_cache_lifetime() {
    std::vector<int> destroyed;
    auto cache = std::make_unique<Cache>(Cache::Config{true, 2, 0});
    auto a = cache->create("A", [&] { return std::make_unique<Payload>(1, destroyed); }, 1);
    auto b = cache->create("B", [&] { return std::make_unique<Payload>(2, destroyed); }, 2);
    auto *b_address = b.get();
    a = std::move(b);
    require(!b && a.key() == "B" && a.get() == b_address && cache->stats().active == 1 &&
            cache->peek_idle("A") && destroyed.empty(), "lease move failed to return the old owner");
    {
        auto moved = std::move(a);
        require(!a && moved.get() == b_address, "lease move constructor duplicated ownership");
    }
    require(cache->stats().active == 0 && cache->peek_idle("B") == b_address,
            "lease destructor did not return the owner to idle");
    a = cache->acquire("B", restore, 3);
    cache.reset();
    require(a.get() == b_address && destroyed.empty(), "cache shell destruction invalidated an active owner");
    a.park(4);
    require(destroyed.size() == 2, "last lease leaked registry ownership after cache teardown");
}

int main() try {
    disabled_and_factory_failure();
    active_ownership_and_capacity();
    many_histories_share_one_warm_slot();
    transition_failures_and_poison();
    ttl_lru_and_partial_success();
    lease_moves_and_cache_lifetime();
    std::cout << "kvmem_sessions disabled=pass history_capacity=pass active_isolation=pass "
                 "cold_resume=pass rollback=pass poison=pass ttl_lru=pass ownership=pass\n";
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}
