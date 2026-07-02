#include "core/tensor.h"

#include <algorithm>
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

}  // namespace trident
