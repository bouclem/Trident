#include "core/tensor.h"

#include <algorithm>
#include <stdexcept>

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

void Tensor::reshape(std::vector<std::size_t> new_shape) {
    std::size_t new_numel = 1;
    for (const auto dim : new_shape) {
        new_numel *= dim;
    }

    // Support single -1 dimension for inference
    // TODO: Add -1 wildcard dimension support

    if (new_numel != numel_) {
        throw std::invalid_argument("reshape: total elements must match");
    }

    shape_ = std::move(new_shape);
    compute_strides();
}

}  // namespace trident
