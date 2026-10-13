#pragma once

#include <string>
#include <vector>

namespace qw3::detail {

std::vector<std::string> qwen_pre_tokenize(const std::string &text);
std::string qwen_normalize_utf8(const std::string &text);

} // namespace qw3::detail
