# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Optimizer Group Definitions
################################################################################

# set(COMMON_OPTIMIZERS
#     adagrad
#     rowwise_adagrad
#     sgd)

# # To be populated in the subsequent diffs
# set(CPU_ONLY_OPTIMIZERS "")

# set(GPU_ONLY_OPTIMIZERS
#     adam
#     lamb
#     partial_rowwise_adam
#     partial_rowwise_lamb
#     lars_sgd
#     none
#     rowwise_adagrad_with_counter)

# set(DEPRECATED_OPTIMIZERS
#     approx_sgd
#     approx_rowwise_adagrad
#     approx_rowwise_adagrad_with_counter
#     approx_rowwise_adagrad_with_weight_decay
#     rowwise_adagrad_with_weight_decay
#     rowwise_weighted_adagrad)

# set(ALL_OPTIMIZERS
#     ${COMMON_OPTIMIZERS}
#     ${CPU_ONLY_OPTIMIZERS}
#     ${GPU_ONLY_OPTIMIZERS}
#     ${DEPRECATED_OPTIMIZERS})

# set(CPU_OPTIMIZERS ${COMMON_OPTIMIZERS} ${CPU_ONLY_OPTIMIZERS})

# set(GPU_OPTIMIZERS ${COMMON_OPTIMIZERS} ${GPU_ONLY_OPTIMIZERS})

# # Optimizers with the VBE support
# set(VBE_OPTIMIZERS
#     rowwise_adagrad
#     rowwise_adagrad_with_counter
#     sgd
#     dense)

# # Optimizers with the GWD support
# set(GWD_OPTIMIZERS
#     rowwise_adagrad)

# # Individual optimizers (not fused with SplitTBE backward)
# set(DEFUSED_OPTIMIZERS
#     rowwise_adagrad)

# # Optimizers with the SSD support
# set(SSD_OPTIMIZERS
#     rowwise_adagrad)

set(WEIGHT_OPTIONS
    weighted
    unweighted_nobag
    unweighted)


################################################################################
# C++ Inference Code
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
