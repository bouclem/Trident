#ifndef TRIDENT_CORE_DTYPE_H
#define TRIDENT_CORE_DTYPE_H

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace trident {

enum class DType {
    float32,
    float64,
    int32,
    int64,
    bool8,
};

// TODO: Add bfloat16, float16, uint8, int8, int16, complex64

constexpr std::size_t dtype_size(DType dt) noexcept {
    switch (dt) {
        case DType::float32: return 4;
        case DType::float64: return 8;
        case DType::int32:   return 4;
        case DType::int64:   return 8;
        case DType::bool8:   return 1;
    }
    return 0;
}

constexpr std::string_view dtype_name(DType dt) noexcept {
    switch (dt) {
        case DType::float32: return "float32";
        case DType::float64: return "float64";
        case DType::int32:   return "int32";
        case DType::int64:   return "int64";
        case DType::bool8:   return "bool8";
    }
    return "unknown";
}

}  // namespace trident

#endif  // TRIDENT_CORE_DTYPE_H
