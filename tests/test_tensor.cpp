#include <cassert>
#include <chrono>
#include <cstdio>
#include <stdexcept>

#include "core/dtype.h"
#include "core/tensor.h"

using namespace trident;

static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(fn)                                                       \
    do {                                                                   \
        ++tests_run;                                                       \
        std::printf("  [RUN] %-30s ... ", #fn);                           \
        const auto _t0 = std::chrono::high_resolution_clock::now();        \
        try {                                                              \
            fn();                                                          \
            ++tests_passed;                                                \
            const auto _t1 = std::chrono::high_resolution_clock::now();    \
            const auto _us = std::chrono::duration_cast<std::chrono::microseconds>(_t1 - _t0).count(); \
            std::printf("OK  (%lld us)\n", static_cast<long long>(_us));   \
        } catch (const std::exception& e) {                                \
            const auto _t1 = std::chrono::high_resolution_clock::now();    \
            const auto _us = std::chrono::duration_cast<std::chrono::microseconds>(_t1 - _t0).count(); \
            std::printf("FAIL (%lld us): %s\n", static_cast<long long>(_us), e.what()); \
        } catch (...) {                                                    \
            std::printf("FAIL: unknown exception\n");                      \
        }                                                                  \
    } while (0)

// --- DType tests ---

static void test_dtype_size() {
    assert(dtype_size(DType::float32) == 4);
    assert(dtype_size(DType::float64) == 8);
    assert(dtype_size(DType::int32) == 4);
    assert(dtype_size(DType::int64) == 8);
    assert(dtype_size(DType::bool8) == 1);
}

static void test_dtype_name() {
    assert(dtype_name(DType::float32) == "float32");
    assert(dtype_name(DType::float64) == "float64");
    assert(dtype_name(DType::int32) == "int32");
    assert(dtype_name(DType::int64) == "int64");
    assert(dtype_name(DType::bool8) == "bool8");
}

// --- Tensor construction tests ---

static void test_default_constructor() {
    Tensor t;
    assert(t.numel() == 0);
    assert(t.is_empty());
    assert(t.ndim() == 0);
    assert(t.nbytes() == 0);
}

static void test_shape_constructor() {
    Tensor t({2, 3, 4});
    assert(t.numel() == 24);
    assert(!t.is_empty());
    assert(t.ndim() == 3);
    assert(t.shape() == std::vector<std::size_t>({2, 3, 4}));
    assert(t.dtype() == DType::float32);
    assert(t.device() == Device::cpu);
}

static void test_dtype_constructor() {
    Tensor t({4, 4}, DType::int64);
    assert(t.dtype() == DType::int64);
    assert(t.nbytes() == 4 * 4 * 8);
}

// --- Strides tests ---

static void test_strides() {
    Tensor t({2, 3, 4});
    assert(t.strides() == std::vector<std::size_t>({12, 4, 1}));
}

static void test_strides_1d() {
    Tensor t({10});
    assert(t.strides() == std::vector<std::size_t>({1}));
}

static void test_strides_scalar() {
    Tensor t({1});
    assert(t.strides() == std::vector<std::size_t>({1}));
    assert(t.numel() == 1);
}

// --- Element access tests ---

static void test_flat_access() {
    Tensor t({6}, DType::float32);
    t.fill<float>(3.14f);
    for (std::size_t i = 0; i < 6; ++i) {
        assert(t.at<float>(i) == 3.14f);
    }
}

static void test_multi_dim_access() {
    Tensor t({2, 3}, DType::float32);
    t.fill<float>(0.0f);
    t.at<float>(0, 0) = 1.0f;
    t.at<float>(0, 1) = 2.0f;
    t.at<float>(1, 2) = 9.0f;
    assert(t.at<float>(0, 0) == 1.0f);
    assert(t.at<float>(0, 1) == 2.0f);
    assert(t.at<float>(0, 2) == 0.0f);
    assert(t.at<float>(1, 0) == 0.0f);
    assert(t.at<float>(1, 2) == 9.0f);
}

static void test_multi_dim_3d() {
    Tensor t({2, 3, 4}, DType::int32);
    t.fill<int32_t>(0);
    t.at<int32_t>(1, 2, 3) = 42;
    assert(t.at<int32_t>(1, 2, 3) == 42);
    assert(t.at<int32_t>(0, 0, 0) == 0);
}

// --- Reshape tests ---

static void test_reshape_basic() {
    Tensor t({2, 3, 4});
    t.reshape({6, 4});
    assert(t.shape() == std::vector<std::size_t>({6, 4}));
    assert(t.numel() == 24);
}

static void test_reshape_wildcard() {
    Tensor t({2, 3, 4});
    t.reshape({6, -1});
    assert(t.shape() == std::vector<std::size_t>({6, 4}));

    t.reshape({-1, 12});
    assert(t.shape() == std::vector<std::size_t>({2, 12}));

    t.reshape({-1});
    assert(t.shape() == std::vector<std::size_t>({24}));
}

static void test_reshape_wildcard_throws() {
    Tensor t({2, 3, 4});
    bool threw = false;
    try {
        t.reshape({-1, -1});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

static void test_reshape_mismatch_throws() {
    Tensor t({2, 3});
    bool threw = false;
    try {
        t.reshape({5, 1});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

// --- Fill tests ---

static void test_fill_int() {
    Tensor t({4, 4}, DType::int32);
    t.fill<int32_t>(7);
    for (std::size_t i = 0; i < 16; ++i) {
        assert(t.at<int32_t>(i) == 7);
    }
}

static void test_fill_float() {
    Tensor t({3, 3}, DType::float64);
    t.fill<double>(1.5);
    for (std::size_t i = 0; i < 9; ++i) {
        assert(t.at<double>(i) == 1.5);
    }
}

// --- Bounds checking tests (debug only) ---

#ifndef NDEBUG
static void test_flat_bounds_check() {
    Tensor t({4}, DType::float32);
    bool threw = false;
    try {
        (void)t.at<float>(10);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

static void test_multi_dim_bounds_check() {
    Tensor t({2, 3}, DType::float32);
    bool threw = false;
    try {
        (void)t.at<float>(5, 0);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

static void test_wrong_index_count() {
    Tensor t({2, 3}, DType::float32);
    bool threw = false;
    try {
        (void)t.at<float>(0, 0, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}
#endif

int main() {
    const auto total_t0 = std::chrono::high_resolution_clock::now();
    std::printf("=== Trident Tests v0.0.1 ===\n\n");

    std::printf("[DType]\n");
    RUN_TEST(test_dtype_size);
    RUN_TEST(test_dtype_name);

    std::printf("\n[Tensor Construction]\n");
    RUN_TEST(test_default_constructor);
    RUN_TEST(test_shape_constructor);
    RUN_TEST(test_dtype_constructor);

    std::printf("\n[Strides]\n");
    RUN_TEST(test_strides);
    RUN_TEST(test_strides_1d);
    RUN_TEST(test_strides_scalar);

    std::printf("\n[Element Access]\n");
    RUN_TEST(test_flat_access);
    RUN_TEST(test_multi_dim_access);
    RUN_TEST(test_multi_dim_3d);

    std::printf("\n[Reshape]\n");
    RUN_TEST(test_reshape_basic);
    RUN_TEST(test_reshape_wildcard);
    RUN_TEST(test_reshape_wildcard_throws);
    RUN_TEST(test_reshape_mismatch_throws);

    std::printf("\n[Fill]\n");
    RUN_TEST(test_fill_int);
    RUN_TEST(test_fill_float);

#ifndef NDEBUG
    std::printf("\n[Bounds Checking (debug)]\n");
    RUN_TEST(test_flat_bounds_check);
    RUN_TEST(test_multi_dim_bounds_check);
    RUN_TEST(test_wrong_index_count);
#endif

    const auto total_t1 = std::chrono::high_resolution_clock::now();
    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_t1 - total_t0).count();

    std::printf("\n=== Results: %d/%d passed in %lld ms ===\n",
                tests_passed, tests_run, static_cast<long long>(total_ms));
    return (tests_passed == tests_run) ? 0 : 1;
}
