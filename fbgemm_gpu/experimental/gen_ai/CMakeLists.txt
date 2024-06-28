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
  # PyTorch
  ${TORCH_INCLUDE_DIRS}
  # Third-party
  ${THIRDPARTY}/asmjit/src
  ${THIRDPARTY}/cpuinfo/include
  ${THIRDPARTY}/cutlass/include
  ${THIRDPARTY}/cutlass/tools/util/include
  ${NCCL_INCLUDE_DIRS})

set(attention_ops_sources
    src/attention/attention.cpp
    src/attention/gqa_attn_splitk.cu)

set(quantize_ops_sources
    src/quantize/cutlass_extensions.cu
    src/quantize/quantize.cu
    src/quantize/quantize.cpp)

set(comm_ops_sources
    src/comm/car.cu
    src/comm/car.cpp)

set(experimental_gen_ai_cpp_source_files
    ${attention_ops_sources}
    ${quantize_ops_sources}
    ${comm_ops_sources})

set_source_files_properties(${experimental_gen_ai_cpp_source_files}
    PROPERTIES INCLUDE_DIRECTORIES
    "${fbgemm_sources_include_directories}")

set(experimental_gen_ai_python_source_files
    gen_ai/__init__.py)


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
