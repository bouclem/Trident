# Trident core library

add_library(trident STATIC)

add_library(Trident::Trident ALIAS trident)

target_sources(trident PRIVATE
    core/tensor.cpp
)

target_include_directories(trident
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(trident PUBLIC cxx_std_20)

target_link_libraries(trident
    PUBLIC
        Trident::Warnings
)

# --- Backend selection ---
if(TRIDENT_ENABLE_CUDA)
    enable_language(CUDA)
    target_compile_definitions(trident PUBLIC TRIDENT_HAS_CUDA)
    message(STATUS "CUDA backend enabled")
endif()

if(TRIDENT_ENABLE_ROCM)
    target_compile_definitions(trident PUBLIC TRIDENT_HAS_ROCM)
    message(STATUS "ROCm backend enabled")
endif()

if(TRIDENT_ENABLE_INTEL)
    target_compile_definitions(trident PUBLIC TRIDENT_HAS_INTEL)
    message(STATUS "Intel backend enabled")
endif()
