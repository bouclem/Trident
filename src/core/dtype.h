#ifndef TRIDENT_CORE_DTYPE_H
#define TRIDENT_CORE_DTYPE_H

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace trident {

enum class DType {
    float16,
    bfloat16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    bool8,
    complex64,
};

constexpr std::size_t dtype_size(DType dt) noexcept {
    switch (dt) {
        case DType::float16:   return 2;
        case DType::bfloat16:  return 2;
        case DType::float32:   return 4;
        case DType::float64:   return 8;
        case DType::int8:      return 1;
        case DType::int16:     return 2;
        case DType::int32:     return 4;
        case DType::int64:     return 8;
        case DType::uint8:     return 1;
        case DType::bool8:     return 1;
        case DType::complex64: return 8;
    }
    return 0;
}

constexpr std::string_view dtype_name(DType dt) noexcept {
    switch (dt) {
        case DType::float16:   return "float16";
        case DType::bfloat16:  return "bfloat16";
        case DType::float32:   return "float32";
        case DType::float64:   return "float64";
        case DType::int8:      return "int8";
        case DType::int16:     return "int16";
        case DType::int32:     return "int32";
        case DType::int64:     return "int64";
        case DType::uint8:     return "uint8";
        case DType::bool8:     return "bool8";
        case DType::complex64: return "complex64";
    }
    return "unknown";
}

}  // namespace trident

#endif  // TRIDENT_CORE_DTYPE_H
