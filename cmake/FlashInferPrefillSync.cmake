# Keep installed FlashInfer headers untouched. Overlay only the affected header
# in this build, and fail closed if its store/gather boundary is unrecognized.
function(qw3_prepare_flashinfer_prefill include_dir overlay_dir)
    set(header "${include_dir}/flashinfer/attention/prefill.cuh")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${header}")
    file(READ "${header}" content)
    set(boundary [=[      uint32_t o_smem_offset_w = o_smem->template get_permuted_offset<UPCAST_STRIDE_O>(
          warp_idx_x * KTraits::NUM_MMA_Q * 16 + lane_idx / 8, lane_idx % 8);]=])
    string(FIND "${content}" "${boundary}" position)
    if(position LESS 0)
        message(FATAL_ERROR "Unrecognized FlashInfer prefill output layout; review its shared-memory synchronization before updating the compatibility overlay")
    endif()
    string(LENGTH "${boundary}" boundary_length)
    math(EXPR after_boundary "${position} + ${boundary_length}")
    string(SUBSTRING "${content}" ${after_boundary} -1 remainder)
    string(FIND "${remainder}" "${boundary}" duplicate)
    if(NOT duplicate LESS 0)
        message(FATAL_ERROR "Ambiguous FlashInfer prefill output layout")
    endif()
    set(replacement [=[#ifndef FLASHINFER_STMATRIX_M8N8X4_ENABLED
      // QW3: publish scalar shared stores before other lanes gather output.
      // SM90+ uses stmatrix and keeps its existing instruction path unchanged.
      __syncwarp();
#endif
]=])
    string(REPLACE "${boundary}" "${replacement}${boundary}" content "${content}")
    # prefill.cuh uses relative includes such as ../cp_async.cuh. Preserve the
    # include tree in the private overlay; only prefill.cuh is modified.
    file(GLOB_RECURSE dependency_headers "${include_dir}/flashinfer/*")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${dependency_headers})
    file(COPY "${include_dir}/flashinfer" DESTINATION "${overlay_dir}"
        PATTERN "*/attention/prefill.cuh" EXCLUDE)
    # configure_file avoids touching the output timestamp on unchanged runs.
    file(WRITE "${overlay_dir}/prefill.cuh.in" "${content}")
    configure_file("${overlay_dir}/prefill.cuh.in"
        "${overlay_dir}/flashinfer/attention/prefill.cuh" COPYONLY)
    file(SHA256 "${header}" header_sha256)
    message(STATUS "FlashInfer scalar-output sync overlay (source SHA256 ${header_sha256})")
endfunction()
