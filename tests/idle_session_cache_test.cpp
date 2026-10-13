#include "idle_session_cache.hpp"
#include <iostream>
#include <vector>

static void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

int main() try {
    struct Resident {
        int id;
        std::vector<int> &retired;
        Resident(int i, std::vector<int> &r) : id(i), retired(r) {}
        ~Resident() { retired.push_back(id); }
    };
    std::vector<int> retired;
    qw3::detail::IdleSessionCache<Resident> cache;
    auto make = [&](int id) { return std::make_unique<Resident>(id, retired); };
    cache.park("A", make(1)); cache.park("B", make(2)); cache.park("C", make(3));
    auto active = cache.take("A");
    require(active && active->id == 1 && cache.size() == 2, "active owner was not detached");
    auto *stable = active.get();
    // Admission needs two free slots. A is active and must survive eviction.
    require(cache.trim(1) == 1 && retired == std::vector<int>{2}, "wrong idle LRU retired");
    cache.park("A", std::move(active));
    require(cache.trim(1) == 1 && retired == std::vector<int>({2, 3}), "returned owner did not become most recent");
    active = cache.take("A");
    require(active.get() == stable && cache.size() == 0, "executor address changed across lease");
    cache.park("D", make(4));
    require(cache.trim(0) == 1 && active->id == 1, "cold admission evicted active owner");
    cache.park("A", std::move(active));
    bool duplicate = false;
    try { cache.park("A", make(5)); } catch (const std::logic_error &) { duplicate = true; }
    require(duplicate && cache.size() == 1, "duplicate key replaced a live idle lineage");
    active = cache.take("A");
    require(active.get() == stable, "duplicate insertion corrupted original lineage");
    active.reset();
    require(!cache.take("missing") && cache.trim(0) == 0, "cache miss changed ownership");
    cache.park("E", make(6)); cache.clear();
    require(retired == std::vector<int>({2, 3, 4, 5, 1, 6}), "resident leaked or retired twice");
    std::cout << "idle_sessions admission=pass active_isolation=pass lru=pass duplicate=pass teardown=pass\n";
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
