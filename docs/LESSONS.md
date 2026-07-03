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

## v0.0.3 — Fixes + Broadcasting + Shape Ops

### float16 subnormal conversion — both directions were broken
- **Forward (float→float16):** Shift formula `(mant | 0x800000) >> (13 + 14 - sub_exp)` was wrong. Correct: `>> (24 - sub_exp)`. The 24-bit significand needs to be shifted down to fit the 10-bit subnormal mantissa, and the shift amount depends on how far below the normal range the exponent is.
- **Backward (float16→float):** Normalization loop had `if (exp == 0) break;` which triggered immediately since `exp` starts at 0 for subnormals. Fix: count shifts in a separate variable, then compute float32 exponent as `113 - shifts` (i.e., `127 + (-14 - shifts)`).
- **Lesson:** Subnormal conversion is the hardest part of float16. Test with explicit subnormal values (2^-16, 2^-20, 2^-24), not just normal-range values.

### Debug type safety for reinterpret_cast
- **Pattern:** `at<T>()` uses `reinterpret_cast<T*>` on raw bytes. Added `check_dtype_size(sizeof(T))` in debug builds to catch type mismatches.
- **Lesson:** `reinterpret_cast` is unavoidable for type-erased storage, but debug-time size checks catch the most common misuse (wrong type passed).

### elementwise() constructor bypass
- **Issue:** Old code created `Tensor({}, ...)` then manually set `shape_` and called `compute_strides()` + `allocate()`. This bypassed constructor validation.
- **Fix:** `Tensor(a.shape_, a.dtype(), a.device())` — single constructor call, proper initialization.
- **Lesson:** Never bypass your own constructors. If the constructor does validation, manual field assignment skips it.

### Broadcasting implementation
- **Pattern:** Right-align shapes, prepend 1-dims to shorter shape, each dim must match or be 1, result = max. Compute broadcast strides (0 for broadcasted dims) and iterate with multi-dim index counter.
- **Lesson:** Broadcast strides (0 for broadcasted dims) let you read the same element multiple times without copying data. The index counter increment loop (right-to-right carry) is the standard way to iterate multi-dim indices.

### Shape ops — zero-copy stride manipulation
- **Pattern:** `squeeze`, `unsqueeze`, `transpose`, `permute` only modify `shape_` and `strides_` vectors. No data movement.
- **Lesson:** Most shape operations are metadata-only if you maintain strides. The data stays in the same memory; only the interpretation changes.

---

<!-- Entries will be added as development progresses -->
