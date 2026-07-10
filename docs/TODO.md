# TODO

## Infrastructure
- [x] Set up CMake build system
- [x] Configure compiler flags (C++20, -Wall -Wextra -Wpedantic)
- [x] Set up testing framework (assert-based, no external deps)
- [ ] Set up CI/CD pipeline
- [ ] Configure sanitizers (ASan, UBSan)

## Core
- [x] Tensor class and dtype system
  - [x] DType enum (float32, float64, int32, int64, bool8)
  - [x] Shape, strides, numel
  - [x] Element access (flat + multi-dimensional)
  - [x] Debug bounds checking (#ifndef NDEBUG)
  - [x] reshape with -1 wildcard inference
  - [x] fill<T>()
  - [x] Additional dtypes (bfloat16, float16, int8, int16, uint8, complex64)
  - [x] Factory methods (zeros, ones, arange, randn, rand)
  - [x] Arithmetic operators (+, -, *, /) — element-wise, with broadcasting
  - [x] Broadcasting for arithmetic operators
  - [x] Shape ops (squeeze, unsqueeze, transpose, permute)
  - [x] Slicing
  - [x] operator() overload for indexing
- [ ] Memory allocator and pool
- [ ] Device abstraction layer
- [ ] Computation graph node/edge structures

## Backends
- [ ] CPU backend with SIMD support
- [ ] CUDA backend
- [ ] ROCm backend
- [ ] Intel (oneAPI/SYCL) backend
- [ ] Intel Arc GPU backend

## Autograd
- [ ] Reverse-mode autograd engine
- [ ] Gradient accumulation
- [ ] Higher-order gradients

## Compiler
- [ ] IR definition
- [ ] Optimization passes
- [ ] JIT compilation

## NN & Optim
- [ ] Layer abstractions (Linear, Conv, Norm, etc.)
- [ ] Loss functions
- [ ] Optimizers (SGD, Adam, AdamW)

## Bindings
- [ ] Python bindings (pybind11)
- [ ] Go bindings (cgo)
- [ ] Zig bindings (C interop)

## Tooling
- [ ] Profiler
- [ ] Plotting utilities
- [ ] Quantization support
- [ ] Distributed training
