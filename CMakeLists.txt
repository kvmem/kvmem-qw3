cmake_minimum_required(VERSION 3.16)

project(qw3 LANGUAGES C CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QW3_BUILD_TESTS "Build QW3 host and CUDA tests" ON)
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
            src/flashinfer_decode_adapter.cu
            src/flashinfer_vision_adapter.cu)
    endif()
else()
    if(QW3_ENABLE_FLASHINFER)
        message(FATAL_ERROR "QW3_ENABLE_FLASHINFER requires QW3_ENABLE_CUDA=ON")
    endif()
    set(QW3_CUDA_SRCS src/kernels_cuda_stub.cpp)
endif()

add_library(qw3_core
    src/engine.cpp
    src/gguf.cpp
    src/model_source.cpp
    ${QW3_CUDA_SRCS}
    src/qwen_native.cpp
    src/qwen_native_backend.cpp
    src/qwen_executor.cpp
    src/qwen_weights.cpp
    src/qwen_config.cpp
    src/kvmem_store.cpp
    src/kvmem_gc.cpp
    src/kvmem_archive.cpp
    src/anthropic_adapter.cpp
    src/vision_cpu_frontend.cpp
    src/vision_gpu_frontend.cpp
    src/harness_semantics.cpp
    src/tool_call_stream.cpp
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
target_compile_definitions(qw3_core PRIVATE
    QW3_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
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

add_library(qw3_request_json STATIC
    src/native_request_json.cpp
    third_party/yyjson/yyjson.c)
option(QW3_JSON_ASCII_SIMD "Scan long JSON ASCII strings with SSE2 when available" ON)
target_compile_definitions(qw3_request_json PRIVATE
    $<$<BOOL:${QW3_JSON_ASCII_SIMD}>:QW3_YYJSON_ASCII_SIMD=1>)
target_include_directories(qw3_request_json SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

add_executable(qw3 src/qw3_cli.cpp src/qw3_server.cpp)
target_include_directories(qw3 SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_compile_definitions(qw3 PRIVATE
    QW3_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(qw3 PRIVATE qw3_core qw3_request_json)

add_executable(qw3-inspect src/qw3_inspect.cpp)
target_link_libraries(qw3-inspect PRIVATE qw3_core)

if(QW3_BUILD_TESTS)
    add_executable(qw3-prompt-render tests/prompt_render_test.cpp)
    target_link_libraries(qw3-prompt-render PRIVATE qw3_core)
    enable_testing()
    add_executable(qw3-small-model-config tests/qwen_small_model_config_test.cpp)
    target_include_directories(qw3-small-model-config SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_link_libraries(qw3-small-model-config PRIVATE qw3_core)
    add_test(NAME qw3-small-model-config COMMAND qw3-small-model-config)
    add_executable(qw3-native-request-json tests/native_request_json_test.cpp)
    target_include_directories(qw3-native-request-json PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(qw3-native-request-json PRIVATE qw3_request_json)
    add_test(NAME qw3-native-request-json COMMAND qw3-native-request-json)
    add_executable(qw3-native-message-prefix tests/native_message_prefix_test.cpp src/anthropic_adapter.cpp)
    target_include_directories(qw3-native-message-prefix PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(qw3-native-message-prefix PRIVATE qw3_request_json Threads::Threads)
    add_test(NAME qw3-native-message-prefix COMMAND qw3-native-message-prefix)
    add_executable(qw3-kvmem-gc tests/kvmem_gc_test.cpp src/kvmem_gc.cpp src/kvmem_store.cpp)
    target_include_directories(qw3-kvmem-gc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(qw3-kvmem-gc PRIVATE Threads::Threads)
    add_test(NAME qw3-kvmem-gc COMMAND qw3-kvmem-gc)
    add_executable(qw3-prefix-cache-lifetime tests/prefix_cache_lifetime_test.cpp)
    target_include_directories(qw3-prefix-cache-lifetime PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(qw3-prefix-cache-lifetime PRIVATE Threads::Threads)
    add_test(NAME qw3-prefix-cache-lifetime COMMAND qw3-prefix-cache-lifetime)
    add_executable(qw3-request-disconnect tests/request_disconnect_test.cpp)
    target_link_libraries(qw3-request-disconnect PRIVATE Threads::Threads)
    add_test(NAME qw3-request-disconnect COMMAND qw3-request-disconnect)

    add_test(NAME qw3-prompt-render COMMAND qw3-prompt-render)

    add_executable(qw3-kvmem-store tests/kvmem_store_test.cpp)
    target_link_libraries(qw3-kvmem-store PRIVATE qw3_core)
    add_test(NAME qw3-kvmem-store COMMAND qw3-kvmem-store)

    add_executable(qw3-kvmem-prefix-reuse-policy
        tests/kvmem_prefix_reuse_policy_test.cpp)
    target_include_directories(qw3-kvmem-prefix-reuse-policy PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME qw3-kvmem-prefix-reuse-policy
        COMMAND qw3-kvmem-prefix-reuse-policy)

    add_executable(qw3-multimodal-pin-policy
        tests/multimodal_pin_policy_test.cpp)
    target_include_directories(qw3-multimodal-pin-policy PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME qw3-multimodal-pin-policy COMMAND qw3-multimodal-pin-policy)

    add_executable(qw3-message-prefix-cache tests/message_prefix_cache_test.cpp)
    target_include_directories(qw3-message-prefix-cache PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_include_directories(qw3-message-prefix-cache SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_link_libraries(qw3-message-prefix-cache PRIVATE Threads::Threads)
    add_test(NAME qw3-message-prefix-cache COMMAND qw3-message-prefix-cache)

    # Real-tokenizer parity and timing; invoke with a local GGUF model path.
    add_executable(qw3-chat-token-cache tests/chat_token_cache_test.cpp)
    target_include_directories(qw3-chat-token-cache PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(qw3-chat-token-cache PRIVATE qw3_core)

    # Run with the local GGUF tokenizer, like qw3-chat-token-cache above.
    add_executable(qw3-vision-request-preparation tests/vision_request_preparation_test.cpp)
    target_include_directories(qw3-vision-request-preparation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_include_directories(qw3-vision-request-preparation SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_link_libraries(qw3-vision-request-preparation PRIVATE qw3_core)

    add_executable(qw3-input-embedding-override-index
        tests/input_embedding_override_index_test.cpp)
    target_include_directories(qw3-input-embedding-override-index PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME qw3-input-embedding-override-index
        COMMAND qw3-input-embedding-override-index)

    add_executable(qw3-kvmem-refresh-policy
        tests/kvmem_refresh_policy_test.cpp)
    target_include_directories(qw3-kvmem-refresh-policy PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME qw3-kvmem-refresh-policy
        COMMAND qw3-kvmem-refresh-policy)

    add_executable(qw3-kvmem-request-plan
        tests/kvmem_request_plan_test.cpp)
    target_include_directories(qw3-kvmem-request-plan PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME qw3-kvmem-request-plan
        COMMAND qw3-kvmem-request-plan)

    add_executable(qw3-tool-call-stream tests/tool_call_stream_test.cpp)
    target_include_directories(qw3-tool-call-stream PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(qw3-tool-call-stream PRIVATE qw3_core)
    add_test(NAME qw3-tool-call-stream COMMAND qw3-tool-call-stream)

    add_executable(qw3-anthropic-adapter tests/anthropic_adapter_test.cpp)
    target_include_directories(qw3-anthropic-adapter PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_include_directories(qw3-anthropic-adapter SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_link_libraries(qw3-anthropic-adapter PRIVATE qw3_core)
    add_test(NAME qw3-anthropic-adapter COMMAND qw3-anthropic-adapter)

    add_executable(qw3-harness-semantics tests/harness_semantics_test.cpp)
    target_include_directories(qw3-harness-semantics PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(qw3-harness-semantics PRIVATE qw3_core)
    add_test(NAME qw3-harness-semantics COMMAND qw3-harness-semantics)

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
        if(QW3_ENABLE_FLASHINFER)
            add_executable(qw3-flashinfer-decode-gqa tests/flashinfer_decode_gqa.cu)
            target_link_libraries(qw3-flashinfer-decode-gqa PRIVATE qw3_core)
            set_property(TARGET qw3-flashinfer-decode-gqa PROPERTY CUDA_STANDARD 17)
            add_test(NAME qw3-flashinfer-decode-gqa COMMAND qw3-flashinfer-decode-gqa)
            add_executable(qw3-vision-attention-oracle tests/vision_attention_oracle.cpp)
            target_link_libraries(qw3-vision-attention-oracle PRIVATE qw3_core)
            add_test(NAME qw3-vision-attention-oracle COMMAND qw3-vision-attention-oracle)
        endif()
        add_executable(qw3-vision-cpu-cache tests/vision_cpu_cache_test.cpp)
        target_include_directories(qw3-vision-cpu-cache PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(qw3-vision-cpu-cache PRIVATE qw3_core)

        add_executable(qw3-vision-gpu-parity tests/vision_gpu_parity.cpp)
        target_include_directories(qw3-vision-gpu-parity PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(qw3-vision-gpu-parity PRIVATE qw3_core)

        add_executable(qw3-prefix-state-storage tests/prefix_state_storage_test.cpp)
        target_include_directories(qw3-prefix-state-storage PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(qw3-prefix-state-storage PRIVATE qw3_core)
        add_test(NAME qw3-prefix-state-storage COMMAND qw3-prefix-state-storage)

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
