# Changelog

All notable changes to Trident will be documented in this file.

## [Unreleased]

_Nothing yet._

## [0.0.1] - 2026-07-02

### Added
- Project directory structure created with `.gitkeep` in all empty directories
- `.gitignore` for C++ build artifacts, IDE files, and binding artifacts
- Initial documentation (README, TODO, PROJECT, LESSONS, METRICS)
- CMake build system (C++20, warnings as error, sanitizer options, backend toggles)
- `DType` enum with `dtype_size()` and `dtype_name()` helpers (`src/core/dtype.h`)
- `float16` and `bfloat16` types with float conversion (`src/core/half.h`)
- Additional dtypes: float16, bfloat16, int8, int16, uint8, complex64
- `Tensor` class with shape/strides, CPU memory allocation, element access (`src/core/tensor.h` / `tensor.cpp`)
- Multi-dimensional variadic indexing: `at<T>(row, col, ...)`
- Debug bounds checking (compiled out with `NDEBUG`)
- `reshape()` with `-1` wildcard dimension inference
- `fill<T>()` for uniform initialization
- `Device` enum (CPU placeholder, GPU backends TODO)
- Test runner with per-test and total timing (`tests/test_tensor.cpp`)
- 23 assert-based tests covering DType, half precision, construction, strides, element access, reshape, fill, and bounds checking
