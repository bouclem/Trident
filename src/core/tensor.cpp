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

void Tensor::check_dtype_size(std::size_t type_size) const {
    if (type_size != dtype_size(dtype_)) {
        throw std::invalid_argument("Tensor: type size " + std::to_string(type_size)
                                    + " does not match dtype " + std::string(dtype_name(dtype_))
                                    + " (size " + std::to_string(dtype_size(dtype_)) + ")");
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

// --- Broadcasting ---

std::vector<std::size_t> Tensor::broadcast_shapes(
        const std::vector<std::size_t>& a,
        const std::vector<std::size_t>& b) {
    const std::size_t result_ndim = std::max(a.size(), b.size());
    std::vector<std::size_t> result(result_ndim);

    for (std::size_t i = 0; i < result_ndim; ++i) {
        const std::size_t a_dim = (i < result_ndim - a.size())
            ? 1 : a[i - (result_ndim - a.size())];
        const std::size_t b_dim = (i < result_ndim - b.size())
            ? 1 : b[i - (result_ndim - b.size())];

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw std::invalid_argument("broadcast: shapes are not compatible");
        }
        result[i] = std::max(a_dim, b_dim);
    }
    return result;
}

namespace {

std::vector<std::size_t> compute_broadcast_strides(
        const std::vector<std::size_t>& shape,
        const std::vector<std::size_t>& strides,
        const std::vector<std::size_t>& target_shape) {
    const std::size_t ndim = target_shape.size();
    std::vector<std::size_t> result(ndim, 0);

    const std::size_t offset = ndim - shape.size();
    for (std::size_t i = 0; i < shape.size(); ++i) {
        const std::size_t target_idx = offset + i;
        if (shape[i] == target_shape[target_idx]) {
            result[target_idx] = strides[i];
        } else {
            result[target_idx] = 0;
        }
    }
    return result;
}

}  // anonymous namespace

// --- Arithmetic operators (element-wise, with broadcasting) ---

Tensor Tensor::elementwise(const Tensor& a, const Tensor& b, char op) {
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("elementwise: dtype mismatch");
    }
    if (a.device() != b.device()) {
        throw std::invalid_argument("elementwise: device mismatch");
    }

    const auto result_shape = broadcast_shapes(a.shape_, b.shape_);

    const auto a_strides = compute_broadcast_strides(a.shape_, a.strides_, result_shape);
    const auto b_strides = compute_broadcast_strides(b.shape_, b.strides_, result_shape);

    Tensor result(result_shape, a.dtype(), a.device());
    const std::size_t ndim = result_shape.size();
    const std::size_t numel = result.numel_;

    std::vector<std::size_t> idx(ndim, 0);

    auto dispatch = [&]<typename T>(T) {
        for (std::size_t flat = 0; flat < numel; ++flat) {
            std::size_t a_off = 0, b_off = 0;
            for (std::size_t d = 0; d < ndim; ++d) {
                a_off += idx[d] * a_strides[d];
                b_off += idx[d] * b_strides[d];
            }
            switch (op) {
                case '+': result.at<T>(flat) = a.at<T>(a_off) + b.at<T>(b_off); break;
                case '-': result.at<T>(flat) = a.at<T>(a_off) - b.at<T>(b_off); break;
                case '*': result.at<T>(flat) = a.at<T>(a_off) * b.at<T>(b_off); break;
                case '/': result.at<T>(flat) = a.at<T>(a_off) / b.at<T>(b_off); break;
            }
            for (std::size_t d = ndim; d > 0; --d) {
                const std::size_t dim = d - 1;
                ++idx[dim];
                if (idx[dim] < result_shape[dim]) break;
                idx[dim] = 0;
            }
        }
    };

    switch (a.dtype()) {
        case DType::float32: dispatch(float{}); break;
        case DType::float64: dispatch(double{}); break;
        case DType::int32:   dispatch(int32_t{}); break;
        case DType::int64:   dispatch(int64_t{}); break;
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

// --- Shape operations ---

void Tensor::squeeze() {
    std::vector<std::size_t> new_shape;
    std::vector<std::size_t> new_strides;
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (shape_[i] != 1) {
            new_shape.push_back(shape_[i]);
            new_strides.push_back(strides_[i]);
        }
    }
    shape_ = std::move(new_shape);
    strides_ = std::move(new_strides);
    if (shape_.empty()) {
        numel_ = (data_.empty()) ? 0 : 1;
    }
}

void Tensor::squeeze(std::size_t dim) {
    if (dim >= shape_.size()) {
        throw std::out_of_range("squeeze: dimension " + std::to_string(dim)
                                + " out of range for ndim " + std::to_string(shape_.size()));
    }
    if (shape_[dim] != 1) {
        throw std::invalid_argument("squeeze: dimension " + std::to_string(dim)
                                    + " has size " + std::to_string(shape_[dim])
                                    + ", not 1");
    }
    shape_.erase(shape_.begin() + static_cast<std::ptrdiff_t>(dim));
    strides_.erase(strides_.begin() + static_cast<std::ptrdiff_t>(dim));
}

void Tensor::unsqueeze(std::size_t dim) {
    if (dim > shape_.size()) {
        throw std::out_of_range("unsqueeze: dimension " + std::to_string(dim)
                                + " out of range for ndim " + std::to_string(shape_.size()));
    }
    std::size_t new_stride = 1;
    if (dim < shape_.size()) {
        new_stride = strides_[dim] * shape_[dim];
    }
    shape_.insert(shape_.begin() + static_cast<std::ptrdiff_t>(dim), 1);
    strides_.insert(strides_.begin() + static_cast<std::ptrdiff_t>(dim), new_stride);
}

void Tensor::transpose(std::size_t dim0, std::size_t dim1) {
    if (dim0 >= shape_.size() || dim1 >= shape_.size()) {
        throw std::out_of_range("transpose: dimension index out of range for ndim "
                                + std::to_string(shape_.size()));
    }
    std::swap(shape_[dim0], shape_[dim1]);
    std::swap(strides_[dim0], strides_[dim1]);
}

void Tensor::permute(std::vector<std::size_t> perm) {
    if (perm.size() != shape_.size()) {
        throw std::invalid_argument("permute: permutation size "
                                    + std::to_string(perm.size())
                                    + " does not match ndim " + std::to_string(shape_.size()));
    }
    std::vector<std::size_t> new_shape(shape_.size());
    std::vector<std::size_t> new_strides(shape_.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] >= shape_.size()) {
            throw std::out_of_range("permute: index " + std::to_string(perm[i])
                                    + " out of range for ndim " + std::to_string(shape_.size()));
        }
        new_shape[i] = shape_[perm[i]];
        new_strides[i] = strides_[perm[i]];
    }
    shape_ = std::move(new_shape);
    strides_ = std::move(new_strides);
}

}  // namespace trident
