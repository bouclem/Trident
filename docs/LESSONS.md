# Lessons

Cumulative mistakes, patterns, and insights. Read before starting work; add when learning something new.

---

## v0.0.1 — CMake + Tensor Core

### Reshape with -1 wildcard
- **Pattern:** Use `std::int64_t` for shape parameters when wildcard support is needed. `std::size_t` can't represent `-1`.
- **Validation:** Count wildcards (max 1), compute known numel, infer missing dim via `numel / known_numel`. Check divisibility.

### Debug bounds checking
- **Pattern:** Guard with `#ifndef NDEBUG` so checks are zero-cost in release builds. Both flat and multi-dimensional variants.
- **Lesson:** Always include the index/dimension in error messages — saves debugging time.

### float16 conversion
- **Gotcha:** IEEE 754 half precision has subnormal range, inf, NaN cases. Simple truncation doesn't work.
- **Pattern:** Handle all exponent ranges explicitly: overflow → inf, normal → remap exponent, subnormal → shift mantissa, underflow → zero.

### bfloat16 conversion
- **Pattern:** Much simpler than float16 — just round-to-nearest-even then truncate upper 16 bits of float32. No subnormal handling needed.

---

## v0.0.2 — Factory Methods + Arithmetic

### Private member access from free functions
- **Issue:** Anonymous namespace `elementwise()` couldn't access `Tensor::shape_`, `compute_strides()`, `allocate()` — MSVC error C2248.
- **Fix:** Made `elementwise` a `private static` member of `Tensor` instead of a free function in an anonymous namespace.
- **Lesson:** When a helper needs private access, prefer a private static member over `friend` — cleaner encapsulation.

### Factory method design
- **Pattern:** Static methods returning `Tensor` by value. RVO ensures no copy. Constructor does the heavy lifting, factory fills data.
- **Lesson:** `zeros()` can use `std::fill` on raw bytes since zero is zero regardless of dtype. `ones()` needs per-type fill since representation differs.

### Arithmetic operators
- **Pattern:** Single `elementwise()` static helper dispatching on `char op` ('+', '-', '*', '/'). Operators just forward to it. Avoids code duplication.
- **Constraint:** Same shape + same dtype only for now. Broadcasting is a separate concern — don't mix into initial implementation.
- **Lesson:** Validate shape AND dtype mismatch before computing. Throw `std::invalid_argument` with clear messages.

### Test runner timing
- **Pattern:** `std::chrono::high_resolution_clock` per test (microseconds) + total (milliseconds). Helps spot regressions.
- **Observation:** `rand()` and `randn()` tests are ~1ms each due to `std::random_device` seeding cost. Other tests are <100us.

---

<!-- Entries will be added as development progresses -->
