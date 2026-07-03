#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "core/dtype.h"
#include "core/half.h"
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
    assert(dtype_size(DType::float16) == 2);
    assert(dtype_size(DType::bfloat16) == 2);
    assert(dtype_size(DType::float32) == 4);
    assert(dtype_size(DType::float64) == 8);
    assert(dtype_size(DType::int8) == 1);
    assert(dtype_size(DType::int16) == 2);
    assert(dtype_size(DType::int32) == 4);
    assert(dtype_size(DType::int64) == 8);
    assert(dtype_size(DType::uint8) == 1);
    assert(dtype_size(DType::bool8) == 1);
    assert(dtype_size(DType::complex64) == 8);
}

static void test_dtype_name() {
    assert(dtype_name(DType::float16) == "float16");
    assert(dtype_name(DType::bfloat16) == "bfloat16");
    assert(dtype_name(DType::float32) == "float32");
    assert(dtype_name(DType::float64) == "float64");
    assert(dtype_name(DType::int8) == "int8");
    assert(dtype_name(DType::int16) == "int16");
    assert(dtype_name(DType::int32) == "int32");
    assert(dtype_name(DType::int64) == "int64");
    assert(dtype_name(DType::uint8) == "uint8");
    assert(dtype_name(DType::bool8) == "bool8");
    assert(dtype_name(DType::complex64) == "complex64");
}

// --- Half precision tests ---

static void test_float16_conversion() {
    float16 a(1.0f);
    assert(static_cast<float>(a) == 1.0f);

    float16 b(3.14f);
    const float bf = static_cast<float>(b);
    assert(bf > 3.1f && bf < 3.2f);

    float16 c(0.0f);
    assert(static_cast<float>(c) == 0.0f);

    float16 d(-2.5f);
    assert(static_cast<float>(d) == -2.5f);

    float16 e(65504.0f);  // max float16
    assert(static_cast<float>(e) == 65504.0f);
}

static void test_bfloat16_conversion() {
    bfloat16 a(1.0f);
    assert(static_cast<float>(a) == 1.0f);

    bfloat16 b(3.14f);
    const float bf = static_cast<float>(b);
    assert(bf > 3.1f && bf < 3.2f);

    bfloat16 c(0.0f);
    assert(static_cast<float>(c) == 0.0f);

    bfloat16 d(-2.5f);
    assert(static_cast<float>(d) == -2.5f);
}

static void test_tensor_with_new_dtypes() {
    Tensor t8({4}, DType::int8);
    t8.fill<int8_t>(-5);
    assert(t8.at<int8_t>(0) == -5);
    assert(t8.at<int8_t>(3) == -5);
    assert(t8.nbytes() == 4);

    Tensor u8({4}, DType::uint8);
    u8.fill<uint8_t>(200);
    assert(u8.at<uint8_t>(0) == 200);
    assert(u8.nbytes() == 4);

    Tensor i16({4}, DType::int16);
    i16.fill<int16_t>(-1000);
    assert(i16.at<int16_t>(0) == -1000);
    assert(i16.nbytes() == 8);
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

// --- Factory method tests ---

static void test_zeros() {
    Tensor t = Tensor::zeros({3, 4}, DType::float32);
    assert(t.numel() == 12);
    for (std::size_t i = 0; i < 12; ++i) {
        assert(t.at<float>(i) == 0.0f);
    }
}

static void test_zeros_int() {
    Tensor t = Tensor::zeros({2, 2}, DType::int32);
    for (std::size_t i = 0; i < 4; ++i) {
        assert(t.at<int32_t>(i) == 0);
    }
}

static void test_ones() {
    Tensor t = Tensor::ones({3, 4}, DType::float32);
    assert(t.numel() == 12);
    for (std::size_t i = 0; i < 12; ++i) {
        assert(t.at<float>(i) == 1.0f);
    }
}

static void test_ones_int() {
    Tensor t = Tensor::ones({2, 2}, DType::int64);
    for (std::size_t i = 0; i < 4; ++i) {
        assert(t.at<int64_t>(i) == 1);
    }
}

static void test_arange() {
    Tensor t = Tensor::arange(5, DType::float32);
    assert(t.numel() == 5);
    for (std::size_t i = 0; i < 5; ++i) {
        assert(t.at<float>(i) == static_cast<float>(i));
    }
}

static void test_arange_int() {
    Tensor t = Tensor::arange(10, DType::int32);
    assert(t.numel() == 10);
    for (std::size_t i = 0; i < 10; ++i) {
        assert(t.at<int32_t>(i) == static_cast<int32_t>(i));
    }
}

static void test_rand_range() {
    Tensor t = Tensor::rand({100}, DType::float32);
    assert(t.numel() == 100);
    for (std::size_t i = 0; i < 100; ++i) {
        const float v = t.at<float>(i);
        assert(v >= 0.0f && v < 1.0f);
    }
}

static void test_randn_shape() {
    Tensor t = Tensor::randn({4, 5}, DType::float32);
    assert(t.numel() == 20);
    assert(t.shape() == std::vector<std::size_t>({4, 5}));
}

// --- Arithmetic operator tests ---

static void test_add() {
    Tensor a = Tensor::arange(4, DType::float32);
    Tensor b = Tensor::ones({4}, DType::float32);
    Tensor c = a + b;
    assert(c.at<float>(0) == 1.0f);
    assert(c.at<float>(1) == 2.0f);
    assert(c.at<float>(2) == 3.0f);
    assert(c.at<float>(3) == 4.0f);
}

static void test_sub() {
    Tensor a = Tensor::arange(4, DType::float32);
    Tensor b = Tensor::ones({4}, DType::float32);
    Tensor c = a - b;
    assert(c.at<float>(0) == -1.0f);
    assert(c.at<float>(1) == 0.0f);
    assert(c.at<float>(2) == 1.0f);
    assert(c.at<float>(3) == 2.0f);
}

static void test_mul() {
    Tensor a({3}, DType::float32);
    a.fill<float>(2.0f);
    Tensor b({3}, DType::float32);
    b.fill<float>(3.0f);
    Tensor c = a * b;
    for (std::size_t i = 0; i < 3; ++i) {
        assert(c.at<float>(i) == 6.0f);
    }
}

static void test_div() {
    Tensor a({3}, DType::float32);
    a.fill<float>(6.0f);
    Tensor b({3}, DType::float32);
    b.fill<float>(2.0f);
    Tensor c = a / b;
    for (std::size_t i = 0; i < 3; ++i) {
        assert(c.at<float>(i) == 3.0f);
    }
}

static void test_arithmetic_int() {
    Tensor a({3}, DType::int32);
    a.fill<int32_t>(10);
    Tensor b({3}, DType::int32);
    b.fill<int32_t>(3);
    Tensor sum = a + b;
    Tensor diff = a - b;
    Tensor prod = a * b;
    Tensor quot = a / b;
    for (std::size_t i = 0; i < 3; ++i) {
        assert(sum.at<int32_t>(i) == 13);
        assert(diff.at<int32_t>(i) == 7);
        assert(prod.at<int32_t>(i) == 30);
        assert(quot.at<int32_t>(i) == 3);
    }
}

static void test_shape_mismatch_throws() {
    Tensor a({2, 3}, DType::float32);
    Tensor b({3, 2}, DType::float32);
    bool threw = false;
    try {
        (void)(a + b);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

static void test_dtype_mismatch_throws() {
    Tensor a({4}, DType::float32);
    Tensor b({4}, DType::int32);
    bool threw = false;
    try {
        (void)(a + b);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

// --- float16 subnormal fix test ---

static void test_float16_subnormal() {
    // 2^-20 is in float16 subnormal range (2^-24 to 2^-15)
    const float val = std::pow(2.0f, -20.0f);
    float16 h(val);
    const float back = static_cast<float>(h);
    assert(back == val);

    // 2^-16 is also subnormal
    const float val2 = std::pow(2.0f, -16.0f);
    float16 h2(val2);
    assert(static_cast<float>(h2) == val2);

    // Smallest positive subnormal: 2^-24
    const float val3 = std::pow(2.0f, -24.0f);
    float16 h3(val3);
    assert(static_cast<float>(h3) == val3);
}

// --- Broadcasting tests ---

static void test_broadcast_scalar() {
    Tensor a = Tensor::ones({1}, DType::float32);
    Tensor b = Tensor::arange(3, DType::float32);  // [0, 1, 2]
    Tensor c = a + b;
    assert(c.shape() == std::vector<std::size_t>({3}));
    assert(c.at<float>(0) == 1.0f);
    assert(c.at<float>(1) == 2.0f);
    assert(c.at<float>(2) == 3.0f);
}

static void test_broadcast_2d() {
    Tensor a = Tensor::ones({3, 1}, DType::float32);
    Tensor b = Tensor::arange(4, DType::float32);
    b.reshape({1, 4});
    Tensor c = a + b;
    assert(c.shape() == std::vector<std::size_t>({3, 4}));
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            assert(c.at<float>(i, j) == static_cast<float>(j + 1));
        }
    }
}

static void test_broadcast_3d() {
    Tensor a = Tensor::ones({2, 1, 3}, DType::float32);
    Tensor b = Tensor::ones({1, 4, 3}, DType::float32);
    Tensor c = a * b;
    assert(c.shape() == std::vector<std::size_t>({2, 4, 3}));
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            for (std::size_t k = 0; k < 3; ++k) {
                assert(c.at<float>(i, j, k) == 1.0f);
            }
        }
    }
}

static void test_broadcast_incompatible() {
    Tensor a({2, 3}, DType::float32);
    Tensor b({3, 2}, DType::float32);
    bool threw = false;
    try {
        (void)(a + b);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

static void test_broadcast_preserves_same_shape() {
    Tensor a({3, 4}, DType::float32);
    a.fill<float>(2.0f);
    Tensor b({3, 4}, DType::float32);
    b.fill<float>(3.0f);
    Tensor c = a + b;
    assert(c.shape() == std::vector<std::size_t>({3, 4}));
    for (std::size_t i = 0; i < 12; ++i) {
        assert(c.at<float>(i) == 5.0f);
    }
}

// --- Shape operation tests ---

static void test_squeeze_all() {
    Tensor t({1, 3, 1, 4});
    t.squeeze();
    assert(t.shape() == std::vector<std::size_t>({3, 4}));
    assert(t.strides() == std::vector<std::size_t>({4, 1}));
}

static void test_squeeze_dim() {
    Tensor t({1, 3, 4});
    t.squeeze(0);
    assert(t.shape() == std::vector<std::size_t>({3, 4}));
}

static void test_squeeze_dim_throws() {
    Tensor t({2, 3});
    bool threw = false;
    try {
        t.squeeze(0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

static void test_unsqueeze() {
    Tensor t({3, 4});
    t.unsqueeze(0);
    assert(t.shape() == std::vector<std::size_t>({1, 3, 4}));
    assert(t.strides() == std::vector<std::size_t>({12, 4, 1}));

    t.unsqueeze(3);
    assert(t.shape() == std::vector<std::size_t>({1, 3, 4, 1}));
}

static void test_transpose() {
    Tensor t({2, 3}, DType::float32);
    t.at<float>(0, 0) = 1.0f;
    t.at<float>(0, 1) = 2.0f;
    t.at<float>(0, 2) = 3.0f;
    t.at<float>(1, 0) = 4.0f;
    t.at<float>(1, 1) = 5.0f;
    t.at<float>(1, 2) = 6.0f;
    t.transpose(0, 1);
    assert(t.shape() == std::vector<std::size_t>({3, 2}));
    assert(t.strides() == std::vector<std::size_t>({1, 3}));
    assert(t.at<float>(0, 0) == 1.0f);
    assert(t.at<float>(0, 1) == 4.0f);
    assert(t.at<float>(1, 0) == 2.0f);
    assert(t.at<float>(2, 1) == 6.0f);
}

static void test_permute() {
    Tensor t({2, 3, 4});
    t.permute({2, 0, 1});
    assert(t.shape() == std::vector<std::size_t>({4, 2, 3}));
    assert(t.strides() == std::vector<std::size_t>({1, 12, 4}));
}

static void test_permute_invalid() {
    Tensor t({2, 3, 4});
    bool threw = false;
    try {
        t.permute({0, 1});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

// --- Bounds checking tests (debug only) ---

#ifndef NDEBUG
static void test_dtype_size_check() {
    Tensor t({4}, DType::float32);
    bool threw = false;
    try {
        (void)t.at<double>(0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

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
    std::printf("=== Trident Tests v0.0.3 ===\n\n");

    std::printf("[DType]\n");
    RUN_TEST(test_dtype_size);
    RUN_TEST(test_dtype_name);

    std::printf("\n[Half Precision]\n");
    RUN_TEST(test_float16_conversion);
    RUN_TEST(test_float16_subnormal);
    RUN_TEST(test_bfloat16_conversion);
    RUN_TEST(test_tensor_with_new_dtypes);

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

    std::printf("\n[Factory Methods]\n");
    RUN_TEST(test_zeros);
    RUN_TEST(test_zeros_int);
    RUN_TEST(test_ones);
    RUN_TEST(test_ones_int);
    RUN_TEST(test_arange);
    RUN_TEST(test_arange_int);
    RUN_TEST(test_rand_range);
    RUN_TEST(test_randn_shape);

    std::printf("\n[Arithmetic]\n");
    RUN_TEST(test_add);
    RUN_TEST(test_sub);
    RUN_TEST(test_mul);
    RUN_TEST(test_div);
    RUN_TEST(test_arithmetic_int);
    RUN_TEST(test_shape_mismatch_throws);
    RUN_TEST(test_dtype_mismatch_throws);

    std::printf("\n[Broadcasting]\n");
    RUN_TEST(test_broadcast_scalar);
    RUN_TEST(test_broadcast_2d);
    RUN_TEST(test_broadcast_3d);
    RUN_TEST(test_broadcast_incompatible);
    RUN_TEST(test_broadcast_preserves_same_shape);

    std::printf("\n[Shape Ops]\n");
    RUN_TEST(test_squeeze_all);
    RUN_TEST(test_squeeze_dim);
    RUN_TEST(test_squeeze_dim_throws);
    RUN_TEST(test_unsqueeze);
    RUN_TEST(test_transpose);
    RUN_TEST(test_permute);
    RUN_TEST(test_permute_invalid);

#ifndef NDEBUG
    std::printf("\n[Bounds Checking (debug)]\n");
    RUN_TEST(test_dtype_size_check);
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
