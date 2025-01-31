# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# CMake Prelude
################################################################################

include(${CMAKEMODULES}/Utilities.cmake)


################################################################################
# FBGEMM_GPU Static Sources
################################################################################

set(fbgemm_gpu_sources_cpu_static
    src/memory_utils/memory_utils.cpp
    src/memory_utils/memory_utils_ops.cpp
    src/merge_pooled_embedding_ops/merge_pooled_embedding_ops_cpu.cpp
    src/permute_multi_embedding_ops/permute_multi_embedding_function.cpp
    src/permute_multi_embedding_ops/permute_multi_embedding_ops_cpu.cpp
    src/permute_pooled_embedding_ops/permute_pooled_embedding_function.cpp
    src/permute_pooled_embedding_ops/permute_pooled_embedding_ops_cpu.cpp
    src/permute_pooled_embedding_ops/permute_pooled_embedding_ops_split_cpu.cpp
    src/jagged_tensor_ops/jagged_tensor_ops_autograd.cpp
    src/jagged_tensor_ops/jagged_tensor_ops_meta.cpp
    src/jagged_tensor_ops/jagged_tensor_ops_cpu.cpp
    src/input_combine_ops/input_combine_cpu.cpp
    src/layout_transform_ops/layout_transform_ops_cpu.cpp
    src/quantize_ops/quantize_ops_cpu.cpp
    src/quantize_ops/quantize_ops_meta.cpp
    src/sparse_ops/sparse_ops_cpu.cpp
    src/sparse_ops/sparse_ops_meta.cpp)

if(NOT FBGEMM_CPU_ONLY)
  list(APPEND fbgemm_gpu_sources_cpu_static
    src/intraining_embedding_pruning_ops/intraining_embedding_pruning_gpu.cpp
    src/layout_transform_ops/layout_transform_ops_gpu.cpp
    src/permute_pooled_embedding_ops/permute_pooled_embedding_ops_gpu.cpp
    src/permute_pooled_embedding_ops/permute_pooled_embedding_ops_split_gpu.cpp
    src/quantize_ops/quantize_ops_gpu.cpp
    src/sparse_ops/sparse_ops_gpu.cpp
    src/metric_ops/metric_ops_host.cpp
    src/input_combine_ops/input_combine_gpu.cpp)

  if(NVML_LIB_PATH OR USE_ROCM)
    message(STATUS "Adding merge_pooled_embeddings sources")
    list(APPEND fbgemm_gpu_sources_cpu_static
      src/merge_pooled_embedding_ops/merge_pooled_embedding_ops_gpu.cpp
      src/topology_utils.cpp)
  else()
    message(STATUS "Skipping merge_pooled_embeddings sources")
  endif()
endif()

if(NOT FBGEMM_CPU_ONLY)
  set(fbgemm_gpu_sources_gpu_static
      src/histogram_binning_calibration_ops.cu
      src/input_combine_ops/input_combine.cu
      src/intraining_embedding_pruning_ops/intraining_embedding_pruning.cu
      src/memory_utils/memory_utils.cu
      src/memory_utils/memory_utils_ops.cu
      src/jagged_tensor_ops/batched_dense_vec_jagged_2d_mul_backward.cu
      src/jagged_tensor_ops/batched_dense_vec_jagged_2d_mul_forward.cu
      src/jagged_tensor_ops/dense_to_jagged_forward.cu
      src/jagged_tensor_ops/jagged_dense_bmm_forward.cu
      src/jagged_tensor_ops/jagged_dense_dense_elementwise_add_jagged_output_forward.cu
      src/jagged_tensor_ops/jagged_dense_elementwise_mul_backward.cu
      src/jagged_tensor_ops/jagged_dense_elementwise_mul_forward.cu
      src/jagged_tensor_ops/jagged_index_add_2d_forward.cu
      src/jagged_tensor_ops/jagged_index_select_2d_forward.cu
      src/jagged_tensor_ops/jagged_jagged_bmm_forward.cu
      src/jagged_tensor_ops/jagged_softmax_backward.cu
      src/jagged_tensor_ops/jagged_softmax_forward.cu
      src/jagged_tensor_ops/jagged_tensor_ops.cu
      src/jagged_tensor_ops/jagged_to_padded_dense_backward.cu
      src/jagged_tensor_ops/jagged_to_padded_dense_forward.cu
      src/jagged_tensor_ops/jagged_unique_indices.cu
      src/jagged_tensor_ops/keyed_jagged_index_select_dim1.cu
      src/layout_transform_ops/layout_transform_ops.cu
      src/metric_ops/metric_ops.cu
      src/permute_pooled_embedding_ops/permute_pooled_embedding_ops_split.cu
      src/permute_pooled_embedding_ops/permute_pooled_embedding_ops.cu
      src/permute_multi_embedding_ops/permute_multi_embedding_ops.cu
      src/quantize_ops/quantize_bfloat16.cu
      src/quantize_ops/quantize_fp8_rowwise.cu
      src/quantize_ops/quantize_fused_8bit_rowwise.cu
      src/quantize_ops/quantize_fused_nbit_rowwise.cu
      src/quantize_ops/quantize_hfp8.cu
      src/quantize_ops/quantize_msfp.cu
      src/quantize_ops/quantize_padded_fp8_rowwise.cu
      src/quantize_ops/quantize_mx.cu
      src/sparse_ops/sparse_block_bucketize_features.cu
      src/sparse_ops/sparse_bucketize_features.cu
      src/sparse_ops/sparse_batched_unary_embeddings.cu
      src/sparse_ops/sparse_compute_frequency_sequence.cu
      src/sparse_ops/sparse_expand_into_jagged_permute.cu
      src/sparse_ops/sparse_group_index.cu
      src/sparse_ops/sparse_index_add.cu
      src/sparse_ops/sparse_index_select.cu
      src/sparse_ops/sparse_invert_permute.cu
      src/sparse_ops/sparse_pack_segments_backward.cu
      src/sparse_ops/sparse_pack_segments_forward.cu
      src/sparse_ops/sparse_permute_1d.cu
      src/sparse_ops/sparse_permute_2d.cu
      src/sparse_ops/sparse_permute102.cu
      src/sparse_ops/sparse_permute_embeddings.cu
      src/sparse_ops/sparse_range.cu
      src/sparse_ops/sparse_reorder_batched_ad.cu
      src/sparse_ops/sparse_segment_sum_csr.cu
      src/sparse_ops/sparse_zipf.cu)
endif()


################################################################################
# FBGEMM_GPU C++ Modules
################################################################################

# Build TBE targets
include(${FBGEMM_GPU}/cmake/TbeInference.cmake)
include(${FBGEMM_GPU}/cmake/TbeTraining.cmake)

# Test target to demonstrate that target deps works as intended
gpu_cpp_library(
  PREFIX
    fbgemm_gpu_embedding_inplace_ops
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  CPU_SRCS
    src/embedding_inplace_ops/embedding_inplace_update_cpu.cpp
  GPU_SRCS
    src/embedding_inplace_ops/embedding_inplace_update_gpu.cpp
    src/embedding_inplace_ops/embedding_inplace_update.cu
  NVCC_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DESTINATION
    fbgemm_gpu)


gpu_cpp_library(
  PREFIX
    fbgemm_gpu_py
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  CPU_SRCS
    ${fbgemm_gpu_sources_cpu_static}
  GPU_SRCS
    ${fbgemm_gpu_sources_gpu_static}
  NVCC_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DEPS
    fbgemm
    fbgemm_gpu_sparse_async_cumsum
    fbgemm_gpu_embedding_inplace_ops
    fbgemm_gpu_tbe_index_select
    fbgemm_gpu_tbe_cache
    fbgemm_gpu_tbe_optimizers
    fbgemm_gpu_tbe_utils
  DESTINATION
    fbgemm_gpu)
