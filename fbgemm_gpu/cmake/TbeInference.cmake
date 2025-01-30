# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


################################################################################
# Split Embeddings Cache
################################################################################

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_tbe_cache
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
  NVCC_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DESTINATION
    fbgemm_gpu)


################################################################################
# TBE Inference
################################################################################

get_tbe_sources_list(static_cpu_files_inference)
get_tbe_sources_list(static_gpu_files_inference)
get_tbe_sources_list(gen_cpu_files_inference)
get_tbe_sources_list(gen_gpu_files_inference)
handle_genfiles_rocm(gen_cpu_files_inference)
handle_genfiles_rocm(gen_gpu_files_inference)

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_tbe_inference
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  CPU_SRCS
    ${static_cpu_files_inference}
    ${gen_cpu_files_inference}
  GPU_SRCS
    ${static_gpu_files_inference}
    ${gen_gpu_files_inference}
  NVCC_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DEPS
    asmjit
    fbgemm
    fbgemm_gpu_tbe_cache
  DESTINATION
    fbgemm_gpu)
