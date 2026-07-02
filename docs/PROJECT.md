# Project — Trident

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

- **Core** — Tensor, DType, Shape, Device abstractions
- **Memory** — Allocators, pools, pinned memory, caching, ref counting
- **Backend** — Per-vendor kernel implementations behind a common interface
- **Autograd** — Reverse-mode automatic differentiation over the computation graph
- **Compiler** — IR lowering, optimization passes, JIT code generation
- **Runtime** — Execution engine that schedules and runs operations
- **Distributed** — Multi-node coordination for data and model parallelism

## Design Principles

- **Zero-overhead abstractions** — What you don't use, you don't pay for
- **Modern C++20/23** — Concepts, ranges, constexpr where applicable
- **RAII everywhere** — No manual resource management
- **Backend-agnostic API** — Same code runs on CPU, CUDA, ROCm, Intel
- **Compile-time safety** — Strong types, concepts, const-correctness

## Direction

1. Establish core types and CPU backend
2. Build autograd and computation graph
3. Add CUDA backend
4. Implement compiler (IR + JIT)
5. Add NN layers and optimizers
6. Python bindings
7. Distributed training
8. ROCm and Intel backends
9. Quantization and deployment tooling
