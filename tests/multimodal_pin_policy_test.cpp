#include "multimodal_pin_policy.hpp"

#include <iostream>
#include <stdexcept>

using qw3::detail::VisualTokenSpan;
using qw3::detail::mandatory_visual_spans;

static void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    // More than 62 full-size screenshots must not consume the entire 128K
    // mandatory budget. This models the failed OSWorld step 63 and later steps.
    std::vector<VisualTokenSpan> history;
    for (uint32_t i = 0; i < 100; ++i) {
        history.emplace_back(100 + i * 2200, 2142 + i * 2200);
    }
    require(mandatory_visual_spans(history, {}, false) == history,
            "default pin-all compatibility changed");
    auto selected = mandatory_visual_spans(history, {{100, 2200}}, true);
    require(selected == std::vector<VisualTokenSpan>{history.front(), history.back()},
            "old screenshots were still all pinned or task/latest image lost");
    require(history.size() == 100, "source history was truncated");
    // A live message can contain multiple images. Explicit pins or queries
    // touching only part of an image must protect the complete image span.
    selected = mandatory_visual_spans(history,
        {{history[97].first + 5, history[98].second - 5},
         {history[20].first + 10, history[20].first + 11}}, true);
    require(selected == std::vector<VisualTokenSpan>{
                history[20], history[97], history[98], history[99]},
            "partial or multi-image protected span lost visual rows");
    selected = mandatory_visual_spans(history,
        {{history[4].second, history[5].first}, {0, 0}}, true);
    require(selected == std::vector<VisualTokenSpan>{history.back()},
            "half-open/empty spans pinned unrelated screenshots");
    require(mandatory_visual_spans({}, {}, true).empty(), "empty history failed");
    std::cout << "multimodal pin policy: OK\n";
}
