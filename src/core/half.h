#ifndef TRIDENT_CORE_HALF_H
#define TRIDENT_CORE_HALF_H

#include <cstdint>
#include <cstring>

namespace trident {

// --- float16 (IEEE 754 half precision) ---
// 1 sign bit, 5 exponent bits, 10 mantissa bits
struct float16 {
    uint16_t raw{0};

    float16() = default;

    explicit float16(float f) noexcept {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));

        const uint32_t sign = (bits >> 16) & 0x8000;
        const uint32_t exp = (bits >> 23) & 0xFF;
        const uint32_t mant = bits & 0x7FFFFF;

        if (exp == 0xFF) {
            // Inf or NaN
            raw = static_cast<uint16_t>(sign | 0x7C00 | (mant ? 1 : 0));
        } else if (exp >= 113) {
            // Overflow or normal range
            if (exp > 142) {
                raw = static_cast<uint16_t>(sign | 0x7C00);  // Inf
            } else {
                const uint32_t new_exp = exp - 112;
                const uint32_t new_mant = mant >> 13;
                raw = static_cast<uint16_t>(sign | (new_exp << 10) | new_mant);
            }
        } else if (exp >= 103) {
            // Subnormal range
            const uint32_t sub_exp = exp - 102;
            const uint32_t sub_mant = (mant | 0x800000) >> (24 - sub_exp);
            raw = static_cast<uint16_t>(sign | sub_mant);
        } else {
            // Underflow to zero
            raw = static_cast<uint16_t>(sign);
        }
    }

    [[nodiscard]] float to_float() const noexcept {
        uint32_t sign = (raw & 0x8000) << 16;
        uint32_t exp = (raw & 0x7C00) >> 10;
        uint32_t mant = raw & 0x3FF;

        if (exp == 0x1F) {
            // Inf or NaN
            uint32_t bits = sign | 0x7F800000 | (mant ? 0x400000 : 0);
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            return f;
        } else if (exp == 0) {
            if (mant == 0) {
                float f;
                uint32_t bits = sign;
                std::memcpy(&f, &bits, sizeof(f));
                return f;
            }
            // Subnormal — normalize by shifting mant left until bit 10 is set
            uint32_t shifts = 0;
            while (!(mant & 0x400)) {
                mant <<= 1;
                ++shifts;
            }
            // float16 subnormal value = 2^(-14) * (original_mant / 2^10)
            // After normalization: exponent = -14 - shifts
            // float32 biased exp = 127 + (-14 - shifts) = 113 - shifts
            const uint32_t f32_exp = 113 - shifts;
            mant &= 0x3FF;  // remove implicit bit
            uint32_t bits = sign | (f32_exp << 23) | (mant << 13);
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            return f;
        } else {
            uint32_t bits = sign | ((exp + 112) << 23) | (mant << 13);
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            return f;
        }
    }

    [[nodiscard]] operator float() const noexcept { return to_float(); }
};

// --- bfloat16 (Brain Float 16) ---
// 1 sign bit, 8 exponent bits, 7 mantissa bits
// Simple truncation of float32's lower 16 bits
struct bfloat16 {
    uint16_t raw{0};

    bfloat16() = default;

    explicit bfloat16(float f) noexcept {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        // Round to nearest even, then truncate
        const uint32_t rounding_bias = 0x7FFF + ((bits >> 16) & 1);
        bits += rounding_bias;
        raw = static_cast<uint16_t>(bits >> 16);
    }

    [[nodiscard]] float to_float() const noexcept {
        uint32_t bits = static_cast<uint32_t>(raw) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    [[nodiscard]] operator float() const noexcept { return to_float(); }
};

}  // namespace trident

#endif  // TRIDENT_CORE_HALF_H
