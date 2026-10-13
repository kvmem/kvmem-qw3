#include "dflash_archive.hpp"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>

using namespace qw3;
namespace fs = std::filesystem;
void require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
void rejected(const std::function<void()> &fn) {
    try { fn(); } catch (const std::exception &) { return; }
    throw std::runtime_error("invalid archive state accepted");
}
KvMemArchiveState draft_state(nlohmann::json overrides = nlohmann::json::object()) {
    const uint64_t position = (1ull << 40) + 7;
    auto j = nlohmann::json{{"version", 1u}, {"identity_sha256", std::string(64, 'a')},
        {"next_position", position}, {"rows", 2u}, {"rope_next", 4u}, {"layers", 1u},
        {"kv_heads", 1u}, {"head_dim", 4u}, {"window", 4u}, {"max_position", 16u}, {"block_size", 2u}};
    j.update(overrides);
    const auto text = j.dump();
    KvMemArchiveState s;
    s.position = s.kvmem_registered_pos = position;
    s.entries = {{KvMemArchiveStateEntry::DFlashMetadata, 0, text.size()},
                 {KvMemArchiveStateEntry::DFlashRawK, 0, 32}, {KvMemArchiveStateEntry::DFlashV, 0, 32}};
    s.payload.assign(text.begin(), text.end());
    // Arbitrary bits, including NaN/subnormal patterns, must survive unchanged.
    const uint32_t bits[] = {0, 0x80000000u, 1, 0x7f800000u, 0x7fc00001u, 0x3f800000u, 0xff800000u, 0xffffffffu};
    for (unsigned i = 0; i < 2; ++i) {
        const auto *p = reinterpret_cast<const uint8_t *>(bits);
        s.payload.insert(s.payload.end(), p, p + sizeof(bits));
    }
    return s;
}
int main() {
    const auto dir = fs::temp_directory_path() / ("qw3-dflash-archive-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        auto state = draft_state();
        const auto view = detail::read_dflash_archive(state);
        require(view && view->rows == 2 && view->tensor_bytes() == 32, "metadata round trip");
        KvMemArchiveLayout layout;
        layout.model_sha256 = std::string(64, 'b');
        auto archive = KvMemArchive::open_for_build(dir.string(), layout, "test");
        archive->write_state(state);
        require(archive->read_state(state.position).payload == state.payload, "disk round trip changed bits");
        KvMemArchiveState legacy;
        legacy.position = 19; legacy.entries = {{KvMemArchiveStateEntry::Hidden, 0, 4}};
        legacy.payload = {1, 2, 3, 4};
        archive->write_state(legacy);
        require(!detail::read_dflash_archive(archive->read_state(19)), "legacy v1 rejected");
        const auto empty = draft_state({{"rows", 0u}, {"rope_next", 0u}});
        auto zero = empty; zero.entries.resize(1); zero.payload.resize(zero.entries[0].bytes);
        archive->write_state(zero);
        require(detail::read_dflash_archive(archive->read_state(zero.position))->keys.empty(), "empty cache");
        for (const auto &patch : std::vector<nlohmann::json>{
                {{"version", 2u}}, {{"rows", 5u}}, {{"rope_next", 15u}}, {{"rows", -1}},
                {{"layers", 0u}}, {{"head_dim", 1025u}}, {{"identity_sha256", "wrong"}},
                {{"next_position", 2u}}, {{"next_position", 1.5}}, {{"max_position", 1u}}}) {
            rejected([&] { archive->write_state(draft_state(patch)); });
        }
        auto malformed = state;
        KvMemArchiveState overflow;
        overflow.entries = {{KvMemArchiveStateEntry::Hidden, 0, UINT64_MAX},
                            {KvMemArchiveStateEntry::Recurrent, 0, 1}};
        rejected([&] { archive->write_state(overflow); });
        rejected([&] { detail::read_dflash_archive(overflow); });
        malformed.entries[2].kind = KvMemArchiveStateEntry::DFlashRawK;
        rejected([&] { archive->write_state(malformed); });
        malformed = state; malformed.entries[2].index = 1;
        rejected([&] { archive->write_state(malformed); });
        malformed = state; malformed.entries.pop_back(); malformed.payload.resize(malformed.payload.size() - 32);
        rejected([&] { archive->write_state(malformed); });
        malformed = state; malformed.entries[0].kind = KvMemArchiveStateEntry::Hidden;
        rejected([&] { archive->write_state(malformed); });

        const auto path = dir / "state" / (std::to_string(state.position) + ".bin");
        auto corrupt = [&](uint64_t offset, uint64_t value, size_t size) {
            archive->write_state(state);
            std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
            f.seekp(offset); f.write(reinterpret_cast<const char *>(&value), size); f.close();
            rejected([&] { archive->read_state(state.position); });
        };
        corrupt(4, 1, 4); // Extended entries may not be disguised as v1.
        corrupt(4, 3, 4); corrupt(12, UINT32_MAX, 4); // Future version / truncated table.
        corrupt(16, 8, 8); corrupt(32, 99, 4); // Wrong position / unknown kind.
        corrupt(40, UINT64_MAX, 8); // Payload overflow without allocation.
        archive->write_state(state);
        fs::resize_file(path, fs::file_size(path) - 1);
        rejected([&] { archive->read_state(state.position); });
        archive.reset(); fs::remove_all(dir);
        std::cout << "DFlash archive v1/v2, bit preservation, uint64 positions, bounded metadata and corruption checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        fs::remove_all(dir); std::cerr << e.what() << '\n'; return 1;
    }
}
