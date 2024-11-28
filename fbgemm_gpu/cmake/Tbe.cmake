# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Optimizer Group Definitions
################################################################################

set(WEIGHT_OPTIONS
    weighted
    unweighted_nobag
    unweighted)


################################################################################
# Split Embeddings Cache
################################################################################

gpu_cpp_library(
  PREFIX
    split_embeddings_cache
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  CPU_SRCS
    ${FBGEMM_GPU}/src/split_embeddings_cache/lfu_cache_populate_byte.cpp
    ${FBGEMM_GPU}/src/split_embeddings_cache/linearize_cache_indices.cpp
    ${FBGEMM_GPU}/src/split_embeddings_cache/lru_cache_populate_byte.cpp
    ${FBGEMM_GPU}/src/split_embeddings_cache/lxu_cache.cpp
    ${FBGEMM_GPU}/src/split_embeddings_cache/split_embeddings_cache_ops.cpp
  GPU_SRCS
    ${FBGEMM_GPU}/src/split_embeddings_cache/lfu_cache_find.cu
    ${FBGEMM_GPU}/src/split_embeddings_cache/lfu_cache_populate.cu
    ${FBGEMM_GPU}/src/split_embeddings_cache/lfu_cache_populate_byte.cu
    ${FBGEMM_GPU}/src/split_embeddings_cache/linearize_cache_indices.cu
    ${FBGEMM_GPU}/src/split_embeddings_cache/lru_cache_find.cu
    ${FBGEMM_GPU}/src/split_embeddings_cache/lru_cache_populate.cu
    ${FBGEMM_GPU}/src/split_embeddings_cache/lru_cache_populate_byte.cu
    ${FBGEMM_GPU}/src/split_embeddings_cache/lxu_cache.cu
    ${FBGEMM_GPU}/src/split_embeddings_cache/reset_weight_momentum.cu
    ${FBGEMM_GPU}/src/split_embeddings_cache/split_embeddings_cache_ops.cu
  GPU_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DESTINATION
    fbgemm_gpu)


################################################################################
# TBE Inference
################################################################################

set(static_cpu_files_inference
  ${FBGEMM_GPU}/codegen/inference/embedding_forward_quantized_host_cpu.cpp)

set(static_gpu_files_inference
  ${FBGEMM_GPU}/codegen/inference/embedding_forward_quantized_host.cpp
  ${FBGEMM_GPU}/codegen/inference/embedding_forward_quantized_split_lookup.cu)

set(gen_cpu_files_inference
  gen_embedding_forward_quantized_unweighted_codegen_cpu.cpp
  gen_embedding_forward_quantized_weighted_codegen_cpu.cpp)

foreach(wdesc ${WEIGHT_OPTIONS})
  foreach(etype fp32 fp16 fp8 int8 int4 int2)
    list(APPEND gen_gpu_files_inference "gen_embedding_forward_quantized_split_nbit_kernel_${wdesc}_${etype}_codegen_cuda.cu")
  endforeach()

  list(APPEND gen_gpu_files_inference "gen_embedding_forward_quantized_split_nbit_host_${wdesc}_codegen_cuda.cu")
endforeach()

if(USE_ROCM)
  prepend_filepaths(
    PREFIX ${CMAKE_BINARY_DIR}
    INPUT ${gen_cpu_files_inference}
    OUTPUT gen_cpu_files_inference)

  prepend_filepaths(
    PREFIX ${CMAKE_BINARY_DIR}
    INPUT ${gen_gpu_files_inference}
    OUTPUT gen_gpu_files_inference)
endif()

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_tbe_inference
  TYPE
    MODULE
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  CPU_SRCS
    ${static_cpu_files_inference}
    ${gen_cpu_files_inference}
  GPU_SRCS
    ${static_gpu_files_inference}
    ${gen_gpu_files_inference}
  GPU_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DEPS
    asmjit
    fbgemm
    split_embeddings_cache
  DESTINATION
    fbgemm_gpu)
