# Trident

A C++ deep learning and tensor computation framework designed for multi-backend support, automatic differentiation, and a compiled graph runtime.

## Overview

Trident is a from-scratch C++ framework aiming to provide:

- **Multi-backend execution** — CPU, NVIDIA CUDA, AMD ROCm, Intel
- **Automatic differentiation** — reverse-mode autograd
- **Graph compiler** — IR, optimization passes, JIT compilation
- **Neural network modules** — layers, losses, model abstractions
- **Distributed training** — multi-node data & model parallelism
- **Language bindings** — Python, Go, Zig
- **Quantization** — INT8/INT4 model compression
- **Profiling & visualization** — built-in profiler and plotting tools

## Project Structure

```
Trident/
├── assets/        # Static resources
├── benchmarks/    # Performance benchmarks
├── docs/          # Documentation
├── examples/      # Usage examples
├── scripts/       # Build & utility scripts
├── src/           # Framework source code
│   ├── api/           # Public API
│   ├── autograd/      # Automatic differentiation
│   ├── backend/       # Hardware backends (CPU, CUDA, ROCm, Intel)
│   ├── bindings/      # Language bindings (Python, Go, Zig)
│   ├── compiler/      # Graph compiler (IR, JIT, optimizer, passes)
│   ├── config/        # Configuration
│   ├── core/          # Core types & foundations
│   ├── data/          # Data loading & pipelines
│   ├── distributed/   # Distributed training
│   ├── graph/         # Computation graph
│   ├── io/            # I/O operations
│   ├── kernels/       # Compute kernels
│   ├── logging/       # Logging
│   ├── memory/        # Memory management
│   ├── nn/            # Neural network layers
│   ├── ops/           # Tensor operations
│   ├── optim/         # Optimizers
│   ├── plotting/      # Visualization
│   ├── profiler/      # Performance profiling
│   ├── quantization/  # Model quantization
│   ├── random/        # RNG & sampling
│   ├── runtime/       # Execution runtime
│   ├── scheduler/     # Operation scheduling
│   ├── serialization/ # Save/load
│   ├── test_utils/    # Testing utilities
│   └── utils/         # General utilities
├── tests/         # Unit & integration tests
├── third_party/   # External dependencies
└── tools/         # Development tools
```

## Status

**v0.0.2** — core Tensor class implemented with:
- 11 dtypes (float16, bfloat16, float32, float64, int8, int16, int32, int64, uint8, bool8, complex64)
- `float16` / `bfloat16` conversion types
- Multi-dimensional indexing with debug bounds checking
- `reshape()` with `-1` wildcard inference
- Factory methods: `zeros()`, `ones()`, `arange()`, `rand()`, `randn()`
- Element-wise arithmetic: `+`, `-`, `*`, `/`
- 38 assert-based tests (all passing, ~5 ms)
- CMake build system (C++20, MSVC / GCC / Clang)

## Build

```bash
cmake -B build -S .
cmake --build build --config Debug
```

## Test

```bash
./build/bin/Debug/test_trident.exe       # Windows
./build/bin/test_trident                  # Linux/macOS
```

## License

TBD
