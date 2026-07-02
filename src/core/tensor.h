#ifndef TRIDENT_CORE_TENSOR_H
#define TRIDENT_CORE_TENSOR_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/dtype.h"

namespace trident {

enum class Device {
    cpu,
    // TODO: cuda, rocm, intel
};

class Tensor {
public:
    // --- Constructors ---

    Tensor() = default;

    Tensor(std::vector<std::size_t> shape, DType dtype = DType::float32,
           Device device = Device::cpu);

    // Rule of Zero — compiler-generated copy/move are correct for our members
    // TODO: Revisit when we add GPU memory or shared storage

    // --- Accessors ---

    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] const std::vector<std::size_t>& strides() const noexcept { return strides_; }
    [[nodiscard]] DType dtype() const noexcept { return dtype_; }
    [[nodiscard]] Device device() const noexcept { return device_; }
    [[nodiscard]] std::size_t ndim() const noexcept { return shape_.size(); }
    [[nodiscard]] std::size_t numel() const noexcept { return numel_; }
    [[nodiscard]] bool is_empty() const noexcept { return numel_ == 0; }

    // --- Element access (CPU only) ---

    template<typename T>
    T& at(std::size_t flat_index) {
#ifndef NDEBUG
        check_flat_bounds(flat_index);
#endif
        return *reinterpret_cast<T*>(data_.data() + flat_index * dtype_size(dtype_));
    }

    template<typename T>
    const T& at(std::size_t flat_index) const {
#ifndef NDEBUG
        check_flat_bounds(flat_index);
#endif
        return *reinterpret_cast<const T*>(data_.data() + flat_index * dtype_size(dtype_));
    }

    // Multi-dimensional indexing: at<float>(row, col) for a 2D tensor
    template<typename T, typename... Indices>
    T& at(std::size_t first, Indices... rest) {
        static_assert(sizeof...(Indices) > 0, "Use at<T>(flat_index) for 1D access");
        const auto idx = {first, static_cast<std::size_t>(rest)...};
        const std::size_t offset = compute_offset(idx);
        return at<T>(offset);
    }

    template<typename T, typename... Indices>
    const T& at(std::size_t first, Indices... rest) const {
        static_assert(sizeof...(Indices) > 0, "Use at<T>(flat_index) for 1D access");
        const auto idx = {first, static_cast<std::size_t>(rest)...};
        const std::size_t offset = compute_offset(idx);
        return at<T>(offset);
    }

    // TODO: operator() overload for indexing: t(row, col)

    // --- Raw data access ---

    [[nodiscard]] void* data() noexcept { return data_.data(); }
    [[nodiscard]] const void* data() const noexcept { return data_.data(); }
    [[nodiscard]] std::size_t nbytes() const noexcept { return data_.size(); }

    // --- Shape operations ---

    void reshape(std::vector<std::int64_t> new_shape);
    // TODO: squeeze, unsqueeze, permute, transpose, expand, broadcast_to

    // --- Fill operations ---

    template<typename T>
    void fill(T value) {
        for (std::size_t i = 0; i < numel_; ++i) {
            at<T>(i) = value;
        }
    }

    // --- Factory methods ---

    static Tensor zeros(std::vector<std::size_t> shape,
                        DType dtype = DType::float32,
                        Device device = Device::cpu);

    static Tensor ones(std::vector<std::size_t> shape,
                       DType dtype = DType::float32,
                       Device device = Device::cpu);

    static Tensor arange(std::size_t n, DType dtype = DType::float32,
                         Device device = Device::cpu);

    static Tensor rand(std::vector<std::size_t> shape,
                       DType dtype = DType::float32,
                       Device device = Device::cpu);

    static Tensor randn(std::vector<std::size_t> shape,
                        DType dtype = DType::float32,
                        Device device = Device::cpu);

    // --- Arithmetic (element-wise, same shape only) ---

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    // TODO: add(), sub(), mul(), div() with broadcasting

    // --- Autograd ---

    // TODO: requires_grad, grad(), backward()
    // FIXME: Autograd integration requires computation graph node linkage

    // --- Serialization ---

    // TODO: save(), load() — integrate with src/serialization/

private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::size_t numel_{0};
    DType dtype_{DType::float32};
    Device device_{Device::cpu};
    std::vector<std::byte> data_;

    void compute_strides();
    void allocate();

    std::size_t compute_offset(std::initializer_list<std::size_t> indices) const;
    void check_flat_bounds(std::size_t flat_index) const;
    void check_multi_bounds(std::initializer_list<std::size_t> indices) const;

    static Tensor elementwise(const Tensor& a, const Tensor& b, char op);
};

}  // namespace trident

#endif  // TRIDENT_CORE_TENSOR_H
