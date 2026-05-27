cmake_minimum_required(VERSION 3.16)

project(qw3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QW3_BUILD_TESTS "Build qw3 smoke tests" ON)
option(QW3_ENABLE_CUDA "Build CUDA native kernels" OFF)

if(QW3_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Default to Ampere+ — INT8 MMA m16n8k32.s8.s8.s32 (mmq_q8.cu) needs
    # sm_80. Override with -DCMAKE_CUDA_ARCHITECTURES=...
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80)
    endif()
    set(QW3_CUDA_SRCS src/kernels_cuda.cu src/gated_delta_net.cu src/mmvq_q8.cu src/fattn_vec_decode.cu src/mmq_q8.cu)
else()
    set(QW3_CUDA_SRCS src/kernels_cuda_stub.cpp)
endif()

add_library(qw3_core
    src/backend.cpp
    src/engine.cpp
    src/gguf.cpp
    ${QW3_CUDA_SRCS}
    src/llama_cli_backend.cpp
    src/mock_backend.cpp
    src/qwen_native.cpp
    src/qwen_native_backend.cpp
    src/qwen_executor.cpp
    src/qwen_weights.cpp
    src/qwen_config.cpp
    src/tokenizer.cpp
)

target_include_directories(qw3_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(qw3_core PRIVATE
    -Wall -Wextra -Wpedantic
)

if(QW3_ENABLE_CUDA)
    target_compile_definitions(qw3_core PRIVATE QW3_ENABLE_CUDA=1)
    target_link_libraries(qw3_core PUBLIC CUDA::cudart CUDA::cublas)
    set_property(TARGET qw3_core PROPERTY CUDA_STANDARD 17)
    set_property(TARGET qw3_core PROPERTY CUDA_STANDARD_REQUIRED ON)
    if(QW3_CUDA_VERBOSE)
        target_compile_options(qw3_core PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
    endif()
endif()

add_executable(qw3 src/qw3_cli.cpp)
target_link_libraries(qw3 PRIVATE qw3_core)

add_executable(qw3-inspect src/qw3_inspect.cpp)
target_link_libraries(qw3-inspect PRIVATE qw3_core)

if(QW3_BUILD_TESTS)
    add_executable(qw3-smoke tests/smoke.cpp)
    target_link_libraries(qw3-smoke PRIVATE qw3_core)
    enable_testing()
    add_test(NAME qw3-smoke COMMAND qw3-smoke)

    if(QW3_ENABLE_CUDA)
        add_executable(qw3-mmq-parity tests/mmq_parity.cu)
        target_link_libraries(qw3-mmq-parity PRIVATE qw3_core)
        set_property(TARGET qw3-mmq-parity PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-mmq-parity PROPERTY CUDA_STANDARD_REQUIRED ON)
        add_test(NAME qw3-mmq-parity COMMAND qw3-mmq-parity)
    endif()
endif()
