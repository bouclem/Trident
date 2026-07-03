# Project — Trident

**Version:** 0.0.3  
**Last updated:** 2026-07-03

## What It Is

Trident is a C++ deep learning and tensor computation framework built from scratch. It aims to provide a complete training and inference stack with multi-backend hardware support, a compiled graph runtime, and language bindings for Python, Go, and Zig.

## Architecture

### Layered Design

```
┌─────────────────────────────────────┐
│         Language Bindings           │  Python / Go / Zig
├─────────────────────────────────────┤
│            Public API               │  src/api/
├─────────────────────────────────────┤
│    NN Layers   │   Optimizers       │  src/nn/  src/optim/
├─────────────────────────────────────┤
│         Autograd Engine             │  src/autograd/
├─────────────────────────────────────┤
│      Computation Graph              │  src/graph/
├─────────────────────────────────────┤
│   Compiler (IR → Optimize → JIT)    │  src/compiler/
├─────────────────────────────────────┤
│     Runtime    │   Scheduler        │  src/runtime/  src/scheduler/
├─────────────────────────────────────┤
│         Operations / Kernels        │  src/ops/  src/kernels/
├─────────────────────────────────────┤
│    CPU   │   CUDA   │   ROCm   │ Intel  │  src/backend/
├─────────────────────────────────────┤
│       Memory Management             │  src/memory/
├─────────────────────────────────────┤
│            Core Types               │  src/core/
└─────────────────────────────────────┘
```

### Key Subsystems

- **Core** ✅ — `Tensor` class, `DType` (11 types), `float16`/`bfloat16`, `Device`, shape/strides, element access, reshape, fill, factory methods, arithmetic operators with broadcasting, shape ops (squeeze/unsqueeze/transpose/permute)
- **Memory** — Allocators, pools, pinned memory, caching, ref counting _(TODO)_
- **Backend** — Per-vendor kernel implementations behind a common interface _(TODO)_
- **Autograd** — Reverse-mode automatic differentiation over the computation graph _(TODO)_
- **Compiler** — IR lowering, optimization passes, JIT code generation _(TODO)_
- **Runtime** — Execution engine that schedules and runs operations _(TODO)_
- **Distributed** — Multi-node coordination for data and model parallelism _(TODO)_

## Design Principles

- **Zero-overhead abstractions** — What you don't use, you don't pay for
- **Modern C++20/23** — Concepts, ranges, constexpr where applicable
- **RAII everywhere** — No manual resource management
- **Backend-agnostic API** — Same code runs on CPU, CUDA, ROCm, Intel
- **Compile-time safety** — Strong types, concepts, const-correctness

## Direction

1. ✅ ~~Establish core types~~ — Tensor, DType, half precision, factory methods, arithmetic
2. Build CPU backend with SIMD support
3. Build autograd and computation graph
4. Add CUDA backend
5. Implement compiler (IR + JIT)
6. Add NN layers and optimizers
7. Python bindings
8. Distributed training
9. ROCm and Intel backends
10. Quantization and deployment tooling

## Current Progress (v0.0.3)

| Area | Status |
|------|--------|
| CMake build system | ✅ Done |
| Testing framework | ✅ 52 tests, assert-based |
| DType system | ✅ 11 dtypes |
| Half precision | ✅ float16, bfloat16 (subnormal fixed) |
| Tensor class | ✅ Shape, strides, indexing, reshape |
| Factory methods | ✅ zeros, ones, arange, rand, randn |
| Arithmetic | ✅ +, -, *, / (element-wise + broadcasting) |
| Broadcasting | ✅ Done |
| Shape ops | ✅ squeeze, unsqueeze, transpose, permute |
| Slicing | ⬜ TODO |
| Memory allocator | ⬜ TODO |
| Device abstraction | ⬜ TODO |
| Autograd | ⬜ TODO |
| Backends (CPU/CUDA/ROCm/Intel) | ⬜ TODO |
