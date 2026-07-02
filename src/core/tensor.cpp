#include "core/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>

namespace trident {

Tensor::Tensor(std::vector<std::size_t> shape, DType dtype, Device device)
    : shape_(std::move(shape)),
      dtype_(dtype),
      device_(device) {
    compute_strides();
    allocate();
}

void Tensor::compute_strides() {
    if (shape_.empty()) {
        strides_.clear();
        numel_ = 0;
        return;
    }

    strides_.resize(shape_.size());
    strides_.back() = 1;
    for (std::size_t i = shape_.size() - 1; i > 0; --i) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    numel_ = strides_[0] * shape_[0];
}

void Tensor::allocate() {
    if (numel_ == 0) {
        data_.clear();
        return;
    }

    const std::size_t bytes = numel_ * dtype_size(dtype_);
    data_.resize(bytes);

    // FIXME: Use memory pool allocator from src/memory/ when available
    // TODO: Handle GPU allocation when device != Device::cpu
}

void Tensor::reshape(std::vector<std::int64_t> new_shape) {
    std::size_t wildcard_count = 0;
    std::size_t known_numel = 1;

    for (const auto dim : new_shape) {
        if (dim == -1) {
            ++wildcard_count;
        } else if (dim < 0) {
            throw std::invalid_argument("reshape: dimensions must be non-negative or -1");
        } else {
            known_numel *= static_cast<std::size_t>(dim);
        }
    }

    if (wildcard_count > 1) {
        throw std::invalid_argument("reshape: only one dimension can be -1");
    }

    if (wildcard_count == 1) {
        if (known_numel == 0 || numel_ % known_numel != 0) {
            throw std::invalid_argument("reshape: cannot infer -1 dimension from total elements");
        }
        const std::size_t inferred = numel_ / known_numel;
        for (auto& dim : new_shape) {
            if (dim == -1) {
                dim = static_cast<std::int64_t>(inferred);
            }
        }
    } else {
        if (known_numel != numel_) {
            throw std::invalid_argument("reshape: total elements must match");
        }
    }

    shape_.clear();
    shape_.reserve(new_shape.size());
    for (const auto dim : new_shape) {
        shape_.push_back(static_cast<std::size_t>(dim));
    }
    compute_strides();
}

std::size_t Tensor::compute_offset(std::initializer_list<std::size_t> indices) const {
#ifndef NDEBUG
    check_multi_bounds(indices);
#endif
    std::size_t offset = 0;
    std::size_t i = 0;
    for (const auto idx : indices) {
        offset += idx * strides_[i];
        ++i;
    }
    return offset;
}

void Tensor::check_flat_bounds(std::size_t flat_index) const {
    if (flat_index >= numel_) {
        throw std::out_of_range("Tensor: flat index " + std::to_string(flat_index)
                                + " out of range for numel " + std::to_string(numel_));
    }
}

void Tensor::check_multi_bounds(std::initializer_list<std::size_t> indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Tensor: expected " + std::to_string(shape_.size())
                                    + " indices but got " + std::to_string(indices.size()));
    }
    std::size_t i = 0;
    for (const auto idx : indices) {
        if (idx >= shape_[i]) {
            throw std::out_of_range("Tensor: index " + std::to_string(idx)
                                    + " out of range for dimension " + std::to_string(i)
                                    + " (size " + std::to_string(shape_[i]) + ")");
        }
        ++i;
    }
}

// --- Factory methods ---

Tensor Tensor::zeros(std::vector<std::size_t> shape, DType dtype, Device device) {
    Tensor t(std::move(shape), dtype, device);
    std::fill(t.data_.begin(), t.data_.end(), std::byte{0});
    return t;
}

Tensor Tensor::ones(std::vector<std::size_t> shape, DType dtype, Device device) {
    Tensor t(std::move(shape), dtype, device);
    switch (dtype) {
        case DType::float32: t.fill<float>(1.0f); break;
        case DType::float64: t.fill<double>(1.0); break;
        case DType::int32:   t.fill<int32_t>(1); break;
        case DType::int64:   t.fill<int64_t>(1); break;
        case DType::int16:   t.fill<int16_t>(1); break;
        case DType::int8:    t.fill<int8_t>(1); break;
        case DType::uint8:   t.fill<uint8_t>(1); break;
        case DType::bool8:   t.fill<uint8_t>(1); break;
        // TODO: float16, bfloat16, complex64 ones
        default:
            throw std::invalid_argument("ones: unsupported dtype "
                                        + std::string(dtype_name(dtype)));
    }
    return t;
}

Tensor Tensor::arange(std::size_t n, DType dtype, Device device) {
    Tensor t({n}, dtype, device);
    switch (dtype) {
        case DType::float32:
            for (std::size_t i = 0; i < n; ++i) t.at<float>(i) = static_cast<float>(i);
            break;
        case DType::float64:
            for (std::size_t i = 0; i < n; ++i) t.at<double>(i) = static_cast<double>(i);
            break;
        case DType::int32:
            for (std::size_t i = 0; i < n; ++i) t.at<int32_t>(i) = static_cast<int32_t>(i);
            break;
        case DType::int64:
            for (std::size_t i = 0; i < n; ++i) t.at<int64_t>(i) = static_cast<int64_t>(i);
            break;
        // TODO: int16, int8, uint8 arange
        default:
            throw std::invalid_argument("arange: unsupported dtype "
                                        + std::string(dtype_name(dtype)));
    }
    return t;
}

Tensor Tensor::rand(std::vector<std::size_t> shape, DType dtype, Device device) {
    Tensor t(std::move(shape), dtype, device);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    switch (dtype) {
        case DType::float32:
            for (std::size_t i = 0; i < t.numel_; ++i) t.at<float>(i) = dist(gen);
            break;
        case DType::float64: {
            std::uniform_real_distribution<double> dist64(0.0, 1.0);
            for (std::size_t i = 0; i < t.numel_; ++i) t.at<double>(i) = dist64(gen);
            break;
        }
        // TODO: int dtypes rand, float16, bfloat16, complex64
        default:
            throw std::invalid_argument("rand: unsupported dtype "
                                        + std::string(dtype_name(dtype)));
    }
    return t;
}

Tensor Tensor::randn(std::vector<std::size_t> shape, DType dtype, Device device) {
    Tensor t(std::move(shape), dtype, device);
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    switch (dtype) {
        case DType::float32:
            for (std::size_t i = 0; i < t.numel_; ++i) t.at<float>(i) = dist(gen);
            break;
        case DType::float64: {
            std::normal_distribution<double> dist64(0.0, 1.0);
            for (std::size_t i = 0; i < t.numel_; ++i) t.at<double>(i) = dist64(gen);
            break;
        }
        // TODO: int dtypes randn (error?), float16, bfloat16, complex64
        default:
            throw std::invalid_argument("randn: unsupported dtype "
                                        + std::string(dtype_name(dtype)));
    }
    return t;
}

// --- Arithmetic operators (element-wise, same shape) ---

namespace {

void check_same_shape(const Tensor& a, const Tensor& b, const char* op) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument(std::string(op) + ": shape mismatch");
    }
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument(std::string(op) + ": dtype mismatch");
    }
}

}  // anonymous namespace

Tensor Tensor::elementwise(const Tensor& a, const Tensor& b, char op) {
    check_same_shape(a, b, "elementwise");
    Tensor result({}, a.dtype(), a.device());
    result.shape_ = a.shape_;
    result.compute_strides();
    result.allocate();

    switch (a.dtype()) {
        case DType::float32:
            for (std::size_t i = 0; i < a.numel(); ++i) {
                switch (op) {
                    case '+': result.at<float>(i) = a.at<float>(i) + b.at<float>(i); break;
                    case '-': result.at<float>(i) = a.at<float>(i) - b.at<float>(i); break;
                    case '*': result.at<float>(i) = a.at<float>(i) * b.at<float>(i); break;
                    case '/': result.at<float>(i) = a.at<float>(i) / b.at<float>(i); break;
                }
            }
            break;
        case DType::float64:
            for (std::size_t i = 0; i < a.numel(); ++i) {
                switch (op) {
                    case '+': result.at<double>(i) = a.at<double>(i) + b.at<double>(i); break;
                    case '-': result.at<double>(i) = a.at<double>(i) - b.at<double>(i); break;
                    case '*': result.at<double>(i) = a.at<double>(i) * b.at<double>(i); break;
                    case '/': result.at<double>(i) = a.at<double>(i) / b.at<double>(i); break;
                }
            }
            break;
        case DType::int32:
            for (std::size_t i = 0; i < a.numel(); ++i) {
                switch (op) {
                    case '+': result.at<int32_t>(i) = a.at<int32_t>(i) + b.at<int32_t>(i); break;
                    case '-': result.at<int32_t>(i) = a.at<int32_t>(i) - b.at<int32_t>(i); break;
                    case '*': result.at<int32_t>(i) = a.at<int32_t>(i) * b.at<int32_t>(i); break;
                    case '/': result.at<int32_t>(i) = a.at<int32_t>(i) / b.at<int32_t>(i); break;
                }
            }
            break;
        case DType::int64:
            for (std::size_t i = 0; i < a.numel(); ++i) {
                switch (op) {
                    case '+': result.at<int64_t>(i) = a.at<int64_t>(i) + b.at<int64_t>(i); break;
                    case '-': result.at<int64_t>(i) = a.at<int64_t>(i) - b.at<int64_t>(i); break;
                    case '*': result.at<int64_t>(i) = a.at<int64_t>(i) * b.at<int64_t>(i); break;
                    case '/': result.at<int64_t>(i) = a.at<int64_t>(i) / b.at<int64_t>(i); break;
                }
            }
            break;
        // TODO: float16, bfloat16, int8, int16, uint8, complex64 arithmetic
        default:
            throw std::invalid_argument("arithmetic: unsupported dtype "
                                        + std::string(dtype_name(a.dtype())));
    }
    return result;
}

Tensor Tensor::operator+(const Tensor& other) const {
    return elementwise(*this, other, '+');
}

Tensor Tensor::operator-(const Tensor& other) const {
    return elementwise(*this, other, '-');
}

Tensor Tensor::operator*(const Tensor& other) const {
    return elementwise(*this, other, '*');
}

Tensor Tensor::operator/(const Tensor& other) const {
    return elementwise(*this, other, '/');
}

}  // namespace trident
