#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>

namespace qw3::detail {

// Copyable streaming SHA-256 state for model and bounded-prefix identity.
class Sha256 {
public:
    void update(const void *data, size_t bytes) {
        const auto *p = static_cast<const uint8_t *>(data);
        total_bytes_ += bytes;
        while (bytes > 0) {
            const size_t n = std::min(bytes, block_.size() - buffered_);
            std::memcpy(block_.data() + buffered_, p, n);
            buffered_ += n;
            p += n;
            bytes -= n;
            if (buffered_ == block_.size()) {
                transform(block_.data());
                buffered_ = 0;
            }
        }
    }

    std::string finish_hex() {
        const uint64_t bit_count = total_bytes_ * 8ull;
        const uint8_t one = 0x80;
        update(&one, 1);
        const uint8_t zero = 0;
        while (buffered_ != 56) update(&zero, 1);
        uint8_t length[8];
        for (uint32_t i = 0; i < 8; ++i) {
            length[7 - i] = static_cast<uint8_t>(bit_count >> (i * 8));
        }
        update(length, sizeof(length));
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (uint32_t v : state_) out << std::setw(8) << v;
        return out.str();
    }

private:
    static uint32_t rotr(uint32_t x, uint32_t n) {
        return (x >> n) | (x << (32 - n));
    }

    void transform(const uint8_t *p) {
        static constexpr uint32_t k[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
            0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
            0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
            0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
            0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
        };
        uint32_t w[64];
        for (uint32_t i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(p[i * 4]) << 24) |
                   (static_cast<uint32_t>(p[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(p[i * 4 + 2]) << 8) |
                   static_cast<uint32_t>(p[i * 4 + 3]);
        }
        for (uint32_t i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^
                                rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^
                                rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (uint32_t i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + s1 + ch + k[i] + w[i];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<uint32_t, 8> state_{{
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    }};
    std::array<uint8_t, 64> block_{};
    uint64_t total_bytes_ = 0;
    size_t buffered_ = 0;
};


} // namespace qw3::detail
