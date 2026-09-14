#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace qw3::detail {

using VisualTokenSpan = std::pair<uint32_t, uint32_t>;

// Keeping a visual input in the source history is independent of forcing all
// its KV rows into every working set. Sparse visual history keeps the latest
// observation and any image touched by a protected task/query/live span whole;
// other historical image blocks remain eligible for the normal KV selector.
inline std::vector<VisualTokenSpan> mandatory_visual_spans(
        const std::vector<VisualTokenSpan> &images,
        const std::vector<VisualTokenSpan> &protected_spans,
        bool retrievable_history) {
    if (!retrievable_history) return images;
    std::vector<VisualTokenSpan> result;
    for (size_t i = 0; i < images.size(); ++i) {
        const auto &image = images[i];
        const bool protected_image = std::any_of(
            protected_spans.begin(), protected_spans.end(),
            [&](const VisualTokenSpan &span) {
                return span.first < span.second &&
                       image.first < span.second && span.first < image.second;
            });
        if (i + 1 == images.size() || protected_image) {
            result.push_back(image);
        }
    }
    return result;
}

} // namespace qw3::detail
