cmake_minimum_required(VERSION 3.25)

project(micrograd
    VERSION 0.1.0
    LANGUAGES C
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(MICROGRAD_BUILD_EXAMPLES "Build micrograd examples" ON)
option(MICROGRAD_BUILD_TESTS "Build micrograd tests" ON)

function(micrograd_set_warnings target)
    target_compile_options(${target}
        PRIVATE
            $<$<C_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wextra -Wpedantic>
            $<$<C_COMPILER_ID:MSVC>:/W4>
    )
endfunction()

add_library(micrograd STATIC
    src/value.c
    src/neuron.c
    src/layer.c
    src/mlp.c
)

target_compile_features(micrograd PUBLIC c_std_23)

set_target_properties(micrograd PROPERTIES
    C_STANDARD 23
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO
)

micrograd_set_warnings(micrograd)

target_include_directories(micrograd
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(micrograd PUBLIC ${MATH_LIBRARY})
endif()

if(MICROGRAD_BUILD_EXAMPLES)
    add_executable(micrograd_basic examples/basic.c)
    target_link_libraries(micrograd_basic PRIVATE micrograd)
    micrograd_set_warnings(micrograd_basic)
endif()

if(MICROGRAD_BUILD_TESTS)
    enable_testing()

    add_executable(micrograd_tests test/test.c)
    target_link_libraries(micrograd_tests PRIVATE micrograd)
    micrograd_set_warnings(micrograd_tests)

    add_test(NAME micrograd_tests COMMAND micrograd_tests)
endif()
