cmake_minimum_required(VERSION 3.16)

project(qw3 LANGUAGES CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QW3_BUILD_TESTS "Build qw3 smoke tests" ON)
option(QW3_ENABLE_CUDA "Build CUDA native kernels" OFF)
option(QW3_ENABLE_FLASHINFER "Build optional FlashInfer prefill adapter" OFF)
set(QW3_GDN_SM120_AOT_DIR "" CACHE PATH
    "Directory containing generated gdn_sm120_0.h and gdn_sm120_0.o")
set(QW3_RMS_FP4_SM120_AOT_DIR "" CACHE PATH
    "Directory containing generated rms_fp4_sm120_0.h and rms_fp4_sm120_0.o")
set(QW3_CUTE_DSL_RUNTIME_DIR "" CACHE PATH
    "Directory containing libcute_dsl_runtime.so for SM120 CuTe DSL AOT objects")
set(QW3_FLASHINFER_INCLUDE_DIR "" CACHE PATH "FlashInfer include directory (e.g. /path/to/flashinfer/include)")
set(QW3_FLASHINFER_CUTLASS_INCLUDE_DIR "" CACHE PATH "CUTLASS include directory used by FlashInfer")
set(QW3_NVFP4_TEST_MODEL "" CACHE PATH "Optional local NVFP4 model for checkpoint parity testing")

if(QW3_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Default to Ampere+ — INT8 MMA m16n8k32.s8.s8.s32 (mmq_q8.cu) needs
    # sm_80. Override with -DCMAKE_CUDA_ARCHITECTURES=...
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80)
    endif()
    set(QW3_CUDA_SRCS src/kernels_cuda.cu src/gated_delta_net.cu src/mmvq_q8.cu src/fattn_vec_decode.cu src/mmq_q8.cu src/kernel_legacy.cu)
    set(QW3_BUILD_NVFP4 OFF)
    set(QW3_FLASHINFER_SM120_SWAP_AB OFF)
    if(QW3_ENABLE_FLASHINFER)
        if(NOT QW3_FLASHINFER_INCLUDE_DIR)
            message(FATAL_ERROR "QW3_ENABLE_FLASHINFER=ON requires -DQW3_FLASHINFER_INCLUDE_DIR=/path/to/flashinfer/include")
        endif()
        get_filename_component(QW3_FLASHINFER_DATA_DIR
            "${QW3_FLASHINFER_INCLUDE_DIR}" DIRECTORY)
        set(QW3_FLASHINFER_CCCL_INCLUDE_DIRS)
        foreach(QW3_FLASHINFER_CCCL_DIR
                "${QW3_FLASHINFER_DATA_DIR}/cccl/cub"
                "${QW3_FLASHINFER_DATA_DIR}/cccl/libcudacxx/include"
                "${QW3_FLASHINFER_DATA_DIR}/cccl/thrust")
            if(EXISTS "${QW3_FLASHINFER_CCCL_DIR}")
                list(APPEND QW3_FLASHINFER_CCCL_INCLUDE_DIRS
                    "${QW3_FLASHINFER_CCCL_DIR}")
            endif()
        endforeach()
        if(QW3_FLASHINFER_CCCL_INCLUDE_DIRS)
            message(STATUS "Using FlashInfer bundled CCCL headers")
        endif()
        if("${CMAKE_CUDA_ARCHITECTURES}" MATCHES "(^|;)120a(-real|-virtual)?(;|$)")
            set(QW3_BUILD_NVFP4 ON)
            if(NOT QW3_FLASHINFER_CUTLASS_INCLUDE_DIR)
                set(QW3_FLASHINFER_CUTLASS_INCLUDE_DIR
                    "${QW3_FLASHINFER_DATA_DIR}/cutlass/include"
                    CACHE PATH "CUTLASS include directory used by FlashInfer" FORCE)
            endif()
            if(NOT EXISTS "${QW3_FLASHINFER_CUTLASS_INCLUDE_DIR}/cutlass/arch/arch.h")
                message(FATAL_ERROR
                    "FlashInfer CUTLASS headers not found under ${QW3_FLASHINFER_CUTLASS_INCLUDE_DIR}; "
                    "set -DQW3_FLASHINFER_CUTLASS_INCLUDE_DIR=/path/to/cutlass/include")
            endif()
            get_filename_component(QW3_FLASHINFER_CUTLASS_DIR
                "${QW3_FLASHINFER_CUTLASS_INCLUDE_DIR}" DIRECTORY)
            set(QW3_FLASHINFER_CUTLASS_UTIL_INCLUDE_DIR
                "${QW3_FLASHINFER_CUTLASS_DIR}/tools/util/include")
            if(NOT EXISTS "${QW3_FLASHINFER_CUTLASS_UTIL_INCLUDE_DIR}/cutlass/util/command_line.h")
                message(FATAL_ERROR
                    "FlashInfer CUTLASS utility headers not found under "
                    "${QW3_FLASHINFER_CUTLASS_UTIL_INCLUDE_DIR}")
            endif()
            set(QW3_FLASHINFER_SM120_HEADER
                "${QW3_FLASHINFER_INCLUDE_DIR}/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h")
            file(READ "${QW3_FLASHINFER_SM120_HEADER}" QW3_FLASHINFER_SM120_HEADER_CONTENT)
            string(FIND "${QW3_FLASHINFER_SM120_HEADER_CONTENT}"
                "bool SwapAB" QW3_FLASHINFER_SM120_SWAP_AB_INDEX)
            if(NOT QW3_FLASHINFER_SM120_SWAP_AB_INDEX EQUAL -1)
                set(QW3_FLASHINFER_SM120_SWAP_AB ON)
                message(STATUS "FlashInfer SM120 SwapAB API detected")
            endif()
            list(APPEND QW3_CUDA_SRCS
                src/nvfp4_linear_adapter.cu
                src/fp8_scaled_mm_adapter.cu)
            message(STATUS "QW3 native NVFP4 enabled for ${CMAKE_CUDA_ARCHITECTURES}")
        endif()
        list(APPEND QW3_CUDA_SRCS
            src/flashinfer_prefill_adapter.cu
            src/flashinfer_decode_adapter.cu)
    endif()
else()
    if(QW3_ENABLE_FLASHINFER)
        message(FATAL_ERROR "QW3_ENABLE_FLASHINFER requires QW3_ENABLE_CUDA=ON")
    endif()
    set(QW3_CUDA_SRCS src/kernels_cuda_stub.cpp)
endif()

add_library(qw3_core
    src/backend.cpp
    src/engine.cpp
    src/gguf.cpp
    src/model_source.cpp
    ${QW3_CUDA_SRCS}
    src/llama_cli_backend.cpp
    src/mock_backend.cpp
    src/qwen_native.cpp
    src/qwen_native_backend.cpp
    src/qwen_executor.cpp
    src/qwen_weights.cpp
    src/qwen_config.cpp
    src/kvmem_store.cpp
    src/kvmem_archive.cpp
    src/tool_call_stream.cpp
    src/tool_structure_constraint.cpp
    src/tokenizer.cpp
)

target_include_directories(qw3_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(qw3_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_compile_options(qw3_core PRIVATE
    -Wall -Wextra -Wpedantic
)
target_link_libraries(qw3_core PUBLIC Threads::Threads)

# Archive manifests identify the complete GGUF by SHA-256. OpenSSL's optimized
# implementation makes the first uncached multi-GiB scan much faster; the
# source retains a portable implementation for minimal deployments.
find_package(OpenSSL QUIET COMPONENTS Crypto)
if(OpenSSL_FOUND)
    target_compile_definitions(qw3_core PRIVATE QW3_HAVE_OPENSSL=1)
    target_link_libraries(qw3_core PUBLIC OpenSSL::Crypto)
endif()

if(QW3_ENABLE_CUDA)
    target_compile_definitions(qw3_core PRIVATE QW3_ENABLE_CUDA=1)
    target_link_libraries(qw3_core PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt)
    set_property(TARGET qw3_core PROPERTY CUDA_STANDARD 17)
    set_property(TARGET qw3_core PROPERTY CUDA_STANDARD_REQUIRED ON)
    if(QW3_ENABLE_FLASHINFER)
        target_compile_definitions(qw3_core PRIVATE QW3_ENABLE_FLASHINFER=1)
        target_include_directories(qw3_core SYSTEM PRIVATE
            ${QW3_FLASHINFER_INCLUDE_DIR}
            ${QW3_FLASHINFER_CCCL_INCLUDE_DIRS})
    endif()
    if(QW3_BUILD_NVFP4)
        target_compile_definitions(qw3_core PRIVATE QW3_ENABLE_NVFP4=1)
        target_compile_options(qw3_core PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        if(QW3_FLASHINFER_SM120_SWAP_AB)
            target_compile_definitions(qw3_core PRIVATE
                QW3_FLASHINFER_SM120_SWAP_AB=1)
        endif()
        target_include_directories(qw3_core SYSTEM PRIVATE
            ${QW3_FLASHINFER_CUTLASS_INCLUDE_DIR}
            ${QW3_FLASHINFER_CUTLASS_UTIL_INCLUDE_DIR})
    endif()
    if(QW3_CUDA_VERBOSE)
        target_compile_options(qw3_core PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
    endif()
    if(QW3_GDN_SM120_AOT_DIR)
        if(NOT EXISTS "${QW3_GDN_SM120_AOT_DIR}/gdn_sm120_0.h" OR
           NOT EXISTS "${QW3_GDN_SM120_AOT_DIR}/gdn_sm120_0.o")
            message(FATAL_ERROR
                "QW3_GDN_SM120_AOT_DIR must contain gdn_sm120_0.h and gdn_sm120_0.o")
        endif()
        if(NOT QW3_CUTE_DSL_RUNTIME_DIR)
            message(FATAL_ERROR
                "QW3_GDN_SM120_AOT_DIR requires QW3_CUTE_DSL_RUNTIME_DIR")
        endif()
        find_library(QW3_CUTE_DSL_RUNTIME_LIBRARY
            NAMES cute_dsl_runtime
            PATHS "${QW3_CUTE_DSL_RUNTIME_DIR}"
            NO_DEFAULT_PATH
            REQUIRED)
        set_source_files_properties(
            "${QW3_GDN_SM120_AOT_DIR}/gdn_sm120_0.o"
            PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        target_sources(qw3_core PRIVATE
            src/gdn_sm120_aot_adapter.cpp
            "${QW3_GDN_SM120_AOT_DIR}/gdn_sm120_0.o")
        target_include_directories(qw3_core PRIVATE
            "${QW3_GDN_SM120_AOT_DIR}")
        target_compile_definitions(qw3_core PRIVATE QW3_ENABLE_GDN_SM120_AOT=1)
        target_link_libraries(qw3_core PUBLIC
            "${QW3_CUTE_DSL_RUNTIME_LIBRARY}")
        target_link_options(qw3_core PUBLIC
            "-Wl,-rpath,${QW3_CUTE_DSL_RUNTIME_DIR}")
        message(STATUS "QW3 SM120 GDN AOT backend enabled")
    endif()
    if(QW3_RMS_FP4_SM120_AOT_DIR)
        if(NOT EXISTS "${QW3_RMS_FP4_SM120_AOT_DIR}/rms_fp4_sm120_0.h" OR
           NOT EXISTS "${QW3_RMS_FP4_SM120_AOT_DIR}/rms_fp4_sm120_0.o")
            message(FATAL_ERROR
                "QW3_RMS_FP4_SM120_AOT_DIR must contain "
                "rms_fp4_sm120_0.h and rms_fp4_sm120_0.o")
        endif()
        if(NOT QW3_CUTE_DSL_RUNTIME_DIR)
            message(FATAL_ERROR
                "QW3_RMS_FP4_SM120_AOT_DIR requires QW3_CUTE_DSL_RUNTIME_DIR")
        endif()
        find_library(QW3_CUTE_DSL_RUNTIME_LIBRARY
            NAMES cute_dsl_runtime
            PATHS "${QW3_CUTE_DSL_RUNTIME_DIR}"
            NO_DEFAULT_PATH
            REQUIRED)
        set_source_files_properties(
            "${QW3_RMS_FP4_SM120_AOT_DIR}/rms_fp4_sm120_0.o"
            PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        target_sources(qw3_core PRIVATE
            src/rms_fp4_sm120_aot_adapter.cpp
            "${QW3_RMS_FP4_SM120_AOT_DIR}/rms_fp4_sm120_0.o")
        target_include_directories(qw3_core PRIVATE
            "${QW3_RMS_FP4_SM120_AOT_DIR}")
        target_compile_definitions(qw3_core PRIVATE
            QW3_ENABLE_RMS_FP4_SM120_AOT=1)
        target_link_libraries(qw3_core PUBLIC
            "${QW3_CUTE_DSL_RUNTIME_LIBRARY}")
        target_link_options(qw3_core PUBLIC
            "-Wl,-rpath,${QW3_CUTE_DSL_RUNTIME_DIR}")
        message(STATUS "QW3 SM120 RMSNorm+NVFP4 AOT backend enabled")
    endif()
endif()

add_executable(qw3 src/qw3_cli.cpp src/qw3_server.cpp)
target_include_directories(qw3 SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(qw3 PRIVATE qw3_core)

add_executable(qw3-inspect src/qw3_inspect.cpp)
target_link_libraries(qw3-inspect PRIVATE qw3_core)

if(QW3_BUILD_TESTS)
    add_executable(qw3-smoke tests/smoke.cpp)
    target_link_libraries(qw3-smoke PRIVATE qw3_core)
    enable_testing()
    add_test(NAME qw3-smoke COMMAND qw3-smoke)

    add_executable(qw3-kvmem-store tests/kvmem_store_test.cpp)
    target_link_libraries(qw3-kvmem-store PRIVATE qw3_core)
    add_test(NAME qw3-kvmem-store COMMAND qw3-kvmem-store)

    add_executable(qw3-tool-call-stream tests/tool_call_stream_test.cpp)
    target_include_directories(qw3-tool-call-stream PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(qw3-tool-call-stream PRIVATE qw3_core)
    add_test(NAME qw3-tool-call-stream COMMAND qw3-tool-call-stream)

    add_executable(qw3-tool-structure-constraint
        tests/tool_structure_constraint_test.cpp)
    target_include_directories(qw3-tool-structure-constraint PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(qw3-tool-structure-constraint PRIVATE qw3_core)
    add_test(NAME qw3-tool-structure-constraint
        COMMAND qw3-tool-structure-constraint)

    add_executable(qw3-server-tool-parser
        tests/server_tool_parser_test.cpp)
    target_include_directories(qw3-server-tool-parser PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_include_directories(qw3-server-tool-parser SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_link_libraries(qw3-server-tool-parser PRIVATE qw3_core)
    add_test(NAME qw3-server-tool-parser COMMAND qw3-server-tool-parser)

    add_executable(qw3-chat-template-policy
        tests/chat_template_policy_test.cpp)
    target_include_directories(qw3-chat-template-policy PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(qw3-chat-template-policy PRIVATE qw3_core)
    add_test(NAME qw3-chat-template-policy COMMAND qw3-chat-template-policy)

    add_executable(qw3-tokenizer-pretokenizer
        tests/tokenizer_pretokenizer_test.cpp)
    target_include_directories(qw3-tokenizer-pretokenizer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(qw3-tokenizer-pretokenizer PRIVATE qw3_core)
    add_test(NAME qw3-tokenizer-pretokenizer
        COMMAND qw3-tokenizer-pretokenizer)

    add_executable(qw3-pinned-kv-tier tests/pinned_kv_tier_test.cpp)
    target_link_libraries(qw3-pinned-kv-tier PRIVATE qw3_core)
    add_test(NAME qw3-pinned-kv-tier COMMAND qw3-pinned-kv-tier)

    add_executable(qw3-nvme-kv-tier tests/nvme_kv_tier_test.cpp)
    target_link_libraries(qw3-nvme-kv-tier PRIVATE qw3_core)
    add_test(NAME qw3-nvme-kv-tier COMMAND qw3-nvme-kv-tier)

    add_executable(qw3-kvmem-archive tests/kvmem_archive_test.cpp)
    target_link_libraries(qw3-kvmem-archive PRIVATE qw3_core)
    add_test(NAME qw3-kvmem-archive COMMAND qw3-kvmem-archive)

    if(QW3_ENABLE_CUDA)
        add_executable(qw3-mmq-parity tests/mmq_parity.cu)
        target_link_libraries(qw3-mmq-parity PRIVATE qw3_core)
        set_property(TARGET qw3-mmq-parity PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-mmq-parity PROPERTY CUDA_STANDARD_REQUIRED ON)
        add_test(NAME qw3-mmq-parity COMMAND qw3-mmq-parity)

        add_executable(qw3-rope-remap-parity tests/rope_remap_parity.cu)
        target_link_libraries(qw3-rope-remap-parity PRIVATE qw3_core)
        set_property(TARGET qw3-rope-remap-parity PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-rope-remap-parity PROPERTY CUDA_STANDARD_REQUIRED ON)
        add_test(NAME qw3-rope-remap-parity COMMAND qw3-rope-remap-parity)

        add_executable(qw3-kvmem-assembly tests/kvmem_assembly.cu)
        target_link_libraries(qw3-kvmem-assembly PRIVATE qw3_core)
        set_property(TARGET qw3-kvmem-assembly PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-kvmem-assembly PROPERTY CUDA_STANDARD_REQUIRED ON)
        add_test(NAME qw3-kvmem-assembly COMMAND qw3-kvmem-assembly)

        add_executable(qw3-kvmem-immutable-k tests/kvmem_immutable_k.cu)
        target_link_libraries(qw3-kvmem-immutable-k PRIVATE qw3_core)
        set_property(TARGET qw3-kvmem-immutable-k PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-kvmem-immutable-k PROPERTY CUDA_STANDARD_REQUIRED ON)
        add_test(NAME qw3-kvmem-immutable-k COMMAND qw3-kvmem-immutable-k)

        add_executable(qw3-rope-remap-batched-parity tests/rope_remap_batched_parity.cu)
        target_link_libraries(qw3-rope-remap-batched-parity PRIVATE qw3_core)
        set_property(TARGET qw3-rope-remap-batched-parity PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-rope-remap-batched-parity PROPERTY CUDA_STANDARD_REQUIRED ON)
        add_test(NAME qw3-rope-remap-batched-parity COMMAND qw3-rope-remap-batched-parity)

        add_executable(qw3-rope-remap-drift tests/rope_remap_drift.cu)
        target_link_libraries(qw3-rope-remap-drift PRIVATE qw3_core)
        set_property(TARGET qw3-rope-remap-drift PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-rope-remap-drift PROPERTY CUDA_STANDARD_REQUIRED ON)

        add_executable(qw3-kvmem-deltanet-score tests/kvmem_deltanet_score.cu)
        target_link_libraries(qw3-kvmem-deltanet-score PRIVATE qw3_core)
        set_property(TARGET qw3-kvmem-deltanet-score PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-kvmem-deltanet-score PROPERTY CUDA_STANDARD_REQUIRED ON)
        add_test(NAME qw3-kvmem-deltanet-score COMMAND qw3-kvmem-deltanet-score)

        add_executable(qw3-kvmem-softmax-pages tests/kvmem_softmax_pages.cu)
        target_link_libraries(qw3-kvmem-softmax-pages PRIVATE qw3_core)
        set_property(TARGET qw3-kvmem-softmax-pages PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-kvmem-softmax-pages PROPERTY CUDA_STANDARD_REQUIRED ON)
        add_test(NAME qw3-kvmem-softmax-pages COMMAND qw3-kvmem-softmax-pages)

        add_executable(qw3-kvmem-kmean-merge tests/kvmem_kmean_merge.cu)
        target_link_libraries(qw3-kvmem-kmean-merge PRIVATE qw3_core)
        set_property(TARGET qw3-kvmem-kmean-merge PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-kvmem-kmean-merge PROPERTY CUDA_STANDARD_REQUIRED ON)
        add_test(NAME qw3-kvmem-kmean-merge COMMAND qw3-kvmem-kmean-merge)
        if(QW3_BUILD_NVFP4)
            add_executable(qw3-mixed-linear-parity tests/mixed_linear_parity.cu)
            target_link_libraries(qw3-mixed-linear-parity PRIVATE qw3_core)
            set_property(TARGET qw3-mixed-linear-parity PROPERTY CUDA_STANDARD 17)
            set_property(TARGET qw3-mixed-linear-parity PROPERTY CUDA_STANDARD_REQUIRED ON)
            add_test(NAME qw3-mixed-linear-parity COMMAND qw3-mixed-linear-parity)
            add_executable(qw3-nvfp4-decode-bench tests/nvfp4_decode_bench.cu)
            target_link_libraries(qw3-nvfp4-decode-bench PRIVATE qw3_core)
            set_property(TARGET qw3-nvfp4-decode-bench PROPERTY CUDA_STANDARD 17)
            set_property(TARGET qw3-nvfp4-decode-bench PROPERTY CUDA_STANDARD_REQUIRED ON)
            if(QW3_NVFP4_TEST_MODEL)
                add_test(NAME qw3-nvfp4-checkpoint-parity
                    COMMAND qw3-mixed-linear-parity ${QW3_NVFP4_TEST_MODEL})
            endif()
        endif()
    endif()
endif()
