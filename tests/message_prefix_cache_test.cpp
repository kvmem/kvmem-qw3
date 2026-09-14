#include "message_prefix_cache.hpp"
#include <iostream>
#include <stdexcept>

using Cache = qw3::detail::MessagePrefixCache;
using J = nlohmann::json;
void check(bool value) { if (!value) throw std::runtime_error("message cache test failed"); }

int main() {
    Cache cache(1 << 20, 2);
    J first = J::array({J{{"role", "system"}, {"content", "original"}},
                       J{{"role", "user"}, {"content", J::array({J{{"type", "image_url"},
                             {"image_url", {{"url", "data:image/png;base64,ABC"}}}}})}},
                       J{{"role", "assistant"}, {"content", "old tail"}}});
    check(cache.store("first", first));
    J next, suffix = J::array({J{{"role", "assistant"}, {"content", "rewritten tail"}},
                               J{{"role", "user"}, {"content", "new turn"}}});
    check(cache.restore("first", 2, suffix, next) == Cache::Restore::Ok);
    J expected = J::array({first[0], first[1], suffix[0], suffix[1]});
    check(next == expected);
    next[0]["content"] = "mutated";
    check(cache.restore("first", 3, J::array(), next) == Cache::Restore::Ok);
    check(next == first); // Restore and request preparation cannot corrupt history.
    check(cache.restore("first", 4, suffix, next) == Cache::Restore::Invalid);
    check(cache.restore("absent", 1, suffix, next) == Cache::Restore::Missing);
    check(!cache.store("first", J::array())); // Handles are immutable.
    check(cache.restore("first", 0, suffix, next) == Cache::Restore::Ok && next == suffix);
    check(cache.store("second", expected));
    check(cache.store("third", first));
    check(cache.restore("first", 1, suffix, next) == Cache::Restore::Missing);
    Cache tiny(128, 2);
    check(!tiny.store("x", first));
    Cache limited(4096, 100);
    for (int i = 0; i < 100; ++i) {
        limited.store(std::to_string(i), J::array({J{{"content", std::string(500, 'a')}}}));
        check(limited.accounted_bytes() <= 4096);
    }
    std::cout << "message cache: append/rewrite/image identity/immutability/eviction/capacity checks passed\n";
}
