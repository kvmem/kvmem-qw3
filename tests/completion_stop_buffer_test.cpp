#include "completion_stop_buffer.hpp"

#include <iostream>
#include <stdexcept>

int main() try {
    size_t cases = 0;
    // Every possible three-chunk partition, including splits within a UTF-8
    // code point and overlapping stop prefixes, must preserve the byte result.
    for (const std::string text : {"hello<stop>tail", "aaaaab-rest", "中文停止后面", "prefix<sto"}) {
        const std::vector<std::string> stops{"", "<stop>", "aaaab", "停止"};
        size_t cut = std::string::npos;
        for (const auto &stop : stops) if (!stop.empty()) cut = std::min(cut, text.find(stop));
        const auto expected = text.substr(0, cut);
        for (size_t a = 0; a <= text.size(); ++a) {
            for (size_t b = a; b <= text.size(); ++b) {
                qw3::detail::CompletionStopBuffer buffer(stops);
                std::string actual = buffer.push(text.substr(0, a));
                actual += buffer.push(text.substr(a, b - a));
                actual += buffer.push(text.substr(b));
                actual += buffer.finish();
                if (actual != expected || buffer.stopped() != (cut != std::string::npos))
                    throw std::runtime_error("stop leaked across chunk boundaries");
                ++cases;
            }
        }
    }
    std::cout << "completion_stop_buffer partitions=" << cases << " status=pass\n";
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
