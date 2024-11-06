# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Target Sources
################################################################################

set(fbgemm_sources_include_directories
  # FBGEMM
  ${FBGEMM}/include
  # FBGEMM_GPU
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/src/quantize
  # PyTorch
  ${TORCH_INCLUDE_DIRS}
  # Third-party
  ${THIRDPARTY}/asmjit/src
  ${THIRDPARTY}/cpuinfo/include
  ${THIRDPARTY}/cutlass/include
  ${THIRDPARTY}/cutlass/tools/util/include
  ${THIRDPARTY}/json/include
  ${NCCL_INCLUDE_DIRS})

set(attention_ops_sources
    src/attention/attention.cpp
    src/attention/gqa_attn_splitk.cu)

if(USE_ROCM)
  set(quantize_ops_sources
    src/quantize/ck_extensions/fp8_tensorwise_gemm.hip
    src/quantize/ck_extensions/fp8_rowwise_gemm.hip
    src/quantize/ck_extensions/fp8_blockwise_gemm.hip
    src/quantize/quantize.cu
    src/quantize/quantize.cpp)
else()
  set(quantize_ops_sources
    src/quantize/cutlass_extensions/f8f8bf16.cu
    src/quantize/cutlass_extensions/f8f8bf16_blockwise.cu
    src/quantize/cutlass_extensions/f8f8bf16_cublas.cu
    src/quantize/cutlass_extensions/f8f8bf16_grouped.cu
    src/quantize/cutlass_extensions/f8f8bf16_rowwise.cu
    src/quantize/cutlass_extensions/f8f8bf16_rowwise_batched.cu
    src/quantize/cutlass_extensions/f8f8bf16_tensorwise.cu
    src/quantize/cutlass_extensions/i8i8bf16.cu
    src/quantize/cutlass_extensions/f8i4bf16_rowwise.cu
    src/quantize/cutlass_extensions/i8i8bf16_dynamic.cu
    src/quantize/cutlass_extensions/bf16i4bf16_rowwise.cu
    src/quantize/cutlass_extensions/bf16i4bf16_rowwise_batched.cu
    src/quantize/quantize.cu
    src/quantize/quantize.cpp)
endif()

set(comm_ops_sources
    src/comm/car.cu
    src/comm/car.cpp)

set(kv_cache_ops_sources
    src/kv_cache/kv_cache.cu
    src/kv_cache/kv_cache.cpp)

set(experimental_gen_ai_cpp_source_files
    ${attention_ops_sources}
    ${quantize_ops_sources}
    ${comm_ops_sources}
    ${kv_cache_ops_sources})

# Set the source file for FB only CPP
if(USE_FB_ONLY)
  file(GLOB fb_only_ops_sources
      fb/src/*/*.cu
      fb/src/*/*.cpp)
  list(APPEND experimental_gen_ai_cpp_source_files ${fb_only_ops_sources})
endif()

set_source_files_properties(${experimental_gen_ai_cpp_source_files}
    PROPERTIES INCLUDE_DIRECTORIES
    "${fbgemm_sources_include_directories}")

file(GLOB_RECURSE experimental_gen_ai_python_source_files
  RELATIVE gen_ai
  *.py)

################################################################################
# FBGEMM_GPU HIP Code Generation
################################################################################

if(USE_ROCM)
  # HIPify CUDA code
  set(header_include_dir
      ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      ${CMAKE_CURRENT_SOURCE_DIR}/../../src
      ${CMAKE_CURRENT_SOURCE_DIR}/../..
      ${CMAKE_CURRENT_SOURCE_DIR})

  hipify(CUDA_SOURCE_DIR ${PROJECT_SOURCE_DIR}
         HEADER_INCLUDE_DIR ${header_include_dir})

  # HIPify source files
  get_hipified_list("${experimental_gen_ai_cpp_source_files}"
    experimental_gen_ai_cpp_source_files_hip)

  set_source_files_properties(${experimental_gen_ai_cpp_source_files_hip}
                              PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

  # Add include directories
  hip_include_directories("${fbgemm_sources_include_directories}")
endif()


################################################################################
# Build Shared Library
################################################################################

if(USE_ROCM)
  # Create a HIP library if using ROCm
  hip_add_library(fbgemm_gpu_experimental_gen_ai_py SHARED
    ${experimental_gen_ai_cpp_source_files_hip}
    ${FBGEMM_HIP_HCC_LIBRARIES}
    HIPCC_OPTIONS
    ${HIP_HCC_FLAGS})

  target_include_directories(fbgemm_gpu_experimental_gen_ai_py PUBLIC
    ${FBGEMM_HIP_INCLUDE}
    ${ROCRAND_INCLUDE}
    ${ROCM_SMI_INCLUDE})

else()
  # Else create a CUDA library
  add_library(fbgemm_gpu_experimental_gen_ai_py MODULE
      ${experimental_gen_ai_cpp_source_files})
endif()

target_include_directories(fbgemm_gpu_experimental_gen_ai_py PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${NCCL_INCLUDE_DIRS})

target_link_libraries(fbgemm_gpu_experimental_gen_ai_py
  ${TORCH_LIBRARIES}
  ${NCCL_LIBRARIES}
  ${CUDA_DRIVER_LIBRARIES})

# Remove `lib` from the output artifact name
set_target_properties(fbgemm_gpu_experimental_gen_ai_py PROPERTIES PREFIX "")


################################################################################
# Install Shared Library and Python Files
################################################################################

install(TARGETS fbgemm_gpu_experimental_gen_ai_py
        DESTINATION fbgemm_gpu/experimental/gen_ai)

install(FILES ${experimental_gen_ai_python_source_files}
        DESTINATION fbgemm_gpu/experimental/gen_ai)

add_custom_target(fbgemm_gpu_experimental_gen_ai_py_clean_rpath ALL
  WORKING_DIRECTORY ${OUTPUT_DIR}
  COMMAND bash ${FBGEMM}/.github/scripts/fbgemm_gpu_postbuild.bash)

add_dependencies(fbgemm_gpu_experimental_gen_ai_py_clean_rpath
  fbgemm_gpu_experimental_gen_ai_py)
