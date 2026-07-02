# TODO

## Infrastructure
- [ ] Set up CMake build system
- [ ] Configure compiler flags (C++20, -Wall -Wextra -Wpedantic)
- [ ] Set up testing framework (GoogleTest)
- [ ] Set up CI/CD pipeline
- [ ] Configure sanitizers (ASan, UBSan)

## Core
- [ ] Tensor class and dtype system
- [ ] Memory allocator and pool
- [ ] Device abstraction layer
- [ ] Computation graph node/edge structures

## Backends
- [ ] CPU backend with SIMD support
- [ ] CUDA backend
- [ ] ROCm backend
- [ ] Intel (oneAPI/SYCL) backend

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
