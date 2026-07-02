# Tests

add_executable(test_trident
    test_tensor.cpp
)

target_link_libraries(test_trident PRIVATE
    Trident::Trident
    Trident::Warnings
)

add_test(NAME test_trident COMMAND test_trident)
