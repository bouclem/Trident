cmake_minimum_required(VERSION 3.20)

project(Trident
    VERSION 0.1.0
    DESCRIPTION "C++ deep learning and tensor computation framework"
    LANGUAGES CXX
)

# --- C++ Standard ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Build Options ---
option(TRIDENT_BUILD_TESTS "Build tests" ON)
option(TRIDENT_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(TRIDENT_BUILD_EXAMPLES "Build examples" OFF)
option(TRIDENT_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(TRIDENT_ENABLE_CUDA "Enable CUDA backend" OFF)
option(TRIDENT_ENABLE_ROCM "Enable ROCm backend" OFF)
option(TRIDENT_ENABLE_INTEL "Enable Intel backend" OFF)
option(TRIDENT_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(TRIDENT_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

# --- Output directories ---
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Compiler Warnings ---
add_library(trident_warnings INTERFACE)
add_library(Trident::Warnings ALIAS trident_warnings)

if(MSVC)
    target_compile_options(trident_warnings INTERFACE
        /W4
        /permissive-
        /WX
    )
else()
    target_compile_options(trident_warnings INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
    )
endif()

# --- Sanitizers ---
if(TRIDENT_ENABLE_ASAN)
    if(MSVC)
        message(WARNING "AddressSanitizer not fully supported on MSVC. Use Clang or GCC.")
    else()
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()
endif()

if(TRIDENT_ENABLE_UBSAN)
    if(MSVC)
        message(WARNING "UndefinedBehaviorSanitizer not supported on MSVC. Use Clang or GCC.")
    else()
        add_compile_options(-fsanitize=undefined)
        add_link_options(-fsanitize=undefined)
    endif()
endif()

# --- Export compile commands (for IDE / clangd) ---
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Subdirectories ---
add_subdirectory(src)

if(TRIDENT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TRIDENT_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(TRIDENT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
