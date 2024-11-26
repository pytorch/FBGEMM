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
# Optimizer Group Definitions
################################################################################

set(COMMON_OPTIMIZERS
    adagrad
    rowwise_adagrad
    sgd)

# To be populated in the subsequent diffs
set(CPU_ONLY_OPTIMIZERS "")

set(GPU_ONLY_OPTIMIZERS
    adam
    lamb
    partial_rowwise_adam
    partial_rowwise_lamb
    lars_sgd
    none
    rowwise_adagrad_with_counter)

set(DEPRECATED_OPTIMIZERS
    approx_sgd
    approx_rowwise_adagrad
    approx_rowwise_adagrad_with_counter
    approx_rowwise_adagrad_with_weight_decay
    rowwise_adagrad_with_weight_decay
    rowwise_weighted_adagrad)

set(ALL_OPTIMIZERS
    ${COMMON_OPTIMIZERS}
    ${CPU_ONLY_OPTIMIZERS}
    ${GPU_ONLY_OPTIMIZERS}
    ${DEPRECATED_OPTIMIZERS})

set(CPU_OPTIMIZERS ${COMMON_OPTIMIZERS} ${CPU_ONLY_OPTIMIZERS})

set(GPU_OPTIMIZERS ${COMMON_OPTIMIZERS} ${GPU_ONLY_OPTIMIZERS})

# Optimizers with the VBE support
set(VBE_OPTIMIZERS
    rowwise_adagrad
    rowwise_adagrad_with_counter
    sgd
    dense)

# Optimizers with the GWD support
set(GWD_OPTIMIZERS
    rowwise_adagrad)

# Individual optimizers (not fused with SplitTBE backward)
set(DEFUSED_OPTIMIZERS
    rowwise_adagrad)

# Optimizers with the SSD support
set(SSD_OPTIMIZERS
    rowwise_adagrad)

set(WEIGHT_OPTIONS
    weighted
    unweighted_nobag
    unweighted)


################################################################################
# Optimizer Groups
################################################################################

set(gen_gpu_kernel_source_files
    "gen_embedding_forward_dense_weighted_codegen_cuda.cu"
    "gen_embedding_forward_dense_unweighted_codegen_cuda.cu"
    "gen_embedding_forward_split_weighted_codegen_cuda.cu"
    "gen_embedding_forward_split_unweighted_codegen_cuda.cu"
    "gen_embedding_backward_dense_indice_weights_codegen_cuda.cu"
    "gen_embedding_backward_split_indice_weights_codegen_cuda.cu"
    "gen_embedding_backward_ssd_indice_weights_codegen_cuda.cu"
    "gen_embedding_forward_dense_weighted_vbe_codegen_cuda.cu"
    "gen_embedding_forward_dense_unweighted_vbe_codegen_cuda.cu"
    "gen_embedding_forward_split_weighted_vbe_codegen_cuda.cu"
    "gen_embedding_forward_split_unweighted_vbe_codegen_cuda.cu"
    "gen_embedding_forward_split_weighted_vbe_gwd_codegen_cuda.cu"
    "gen_embedding_forward_split_unweighted_vbe_gwd_codegen_cuda.cu"
    "gen_batch_index_select_dim0_forward_codegen_cuda.cu"
    "gen_batch_index_select_dim0_forward_kernel.cu"
    "gen_batch_index_select_dim0_forward_kernel_small.cu"
    "gen_batch_index_select_dim0_backward_codegen_cuda.cu"
    "gen_batch_index_select_dim0_backward_kernel_cta.cu"
    "gen_batch_index_select_dim0_backward_kernel_warp.cu"
    "gen_embedding_backward_split_grad_embedding_ops.cu"
    "gen_embedding_backward_split_grad_index_select.cu"
    "gen_embedding_backward_split_common_device_kernel.cuh"
    "gen_embedding_backward_split_batch_index_select_device_kernel.cuh"
    "gen_embedding_forward_split_weighted_gwd_codegen_cuda.cu"
    "gen_embedding_forward_split_unweighted_gwd_codegen_cuda.cu"
    "gen_embedding_forward_ssd_weighted_codegen_cuda.cu"
    "gen_embedding_forward_ssd_unweighted_codegen_cuda.cu"
    "gen_embedding_forward_ssd_unweighted_nobag_kernel_small.cu"
    "gen_embedding_forward_ssd_weighted_vbe_codegen_cuda.cu"
    "gen_embedding_forward_ssd_unweighted_vbe_codegen_cuda.cu"
)

list(APPEND gen_gpu_host_source_files
    "gen_embedding_forward_split_unweighted_vbe_codegen_meta.cpp"
    "gen_embedding_forward_split_weighted_vbe_codegen_meta.cpp"
    "gen_embedding_forward_ssd_unweighted_vbe_codegen_meta.cpp"
    "gen_embedding_forward_ssd_weighted_vbe_codegen_meta.cpp"
  )

list(APPEND gen_gpu_kernel_source_files
  "gen_embedding_forward_split_weighted_v2_kernel.cu"
  "gen_embedding_forward_split_unweighted_v2_kernel.cu"
  )

foreach(wdesc dense split)
  list(APPEND gen_gpu_kernel_source_files
    "gen_embedding_forward_${wdesc}_unweighted_nobag_kernel_small.cu")
endforeach()

foreach(wdesc ${WEIGHT_OPTIONS})
  list(APPEND gen_gpu_kernel_source_files
      # "gen_embedding_forward_quantized_split_nbit_host_${wdesc}_codegen_cuda.cu"
      "gen_embedding_forward_dense_${wdesc}_kernel.cu"
      "gen_embedding_backward_dense_split_${wdesc}_cuda.cu"
      "gen_embedding_backward_dense_split_${wdesc}_kernel_cta.cu"
      "gen_embedding_backward_dense_split_${wdesc}_kernel_warp.cu"
      "gen_embedding_forward_split_${wdesc}_kernel.cu"
      "gen_embedding_forward_ssd_${wdesc}_kernel.cu"
      "gen_embedding_backward_split_${wdesc}_device_kernel.cuh")

  # foreach(etype fp32 fp16 fp8 int8 int4 int2)
  #   list(APPEND gen_gpu_kernel_source_files
  #      "gen_embedding_forward_quantized_split_nbit_kernel_${wdesc}_${etype}_codegen_cuda.cu")
  # endforeach()
endforeach()

# Generate VBE files
foreach(wdesc weighted unweighted)
  list(APPEND gen_gpu_kernel_source_files
      "gen_embedding_forward_split_${wdesc}_vbe_kernel.cu"
      "gen_embedding_backward_split_${wdesc}_vbe_device_kernel.cuh"
      "gen_embedding_forward_dense_${wdesc}_vbe_kernel.cu"
      "gen_embedding_forward_ssd_${wdesc}_vbe_kernel.cu")

endforeach()

# Generate GWD files
foreach(wdesc weighted unweighted)
  list(APPEND gen_gpu_kernel_source_files
      "gen_embedding_forward_split_${wdesc}_vbe_gwd_kernel.cu"
      "gen_embedding_forward_split_${wdesc}_gwd_kernel.cu")
endforeach()

set(gen_cpu_source_files
    # "gen_embedding_forward_quantized_unweighted_codegen_cpu.cpp"
    # "gen_embedding_forward_quantized_weighted_codegen_cpu.cpp"
    "gen_embedding_backward_dense_split_cpu.cpp")

set(gen_python_source_files
  ${CMAKE_BINARY_DIR}/__init__.py
  ${CMAKE_BINARY_DIR}/lookup_args.py
  ${CMAKE_BINARY_DIR}/lookup_args_ssd.py
)

# For each of the optimizers, generate the backward split variant by adding
# the Python, CPU-only, GPU host, and GPU kernel source files

# Generate the Python functions only if there is the backend support (for all
# optimizers)
foreach(optimizer
    ${COMMON_OPTIMIZERS}
    ${CPU_ONLY_OPTIMIZERS}
    ${GPU_ONLY_OPTIMIZERS})
  list(APPEND gen_python_source_files
    "${CMAKE_BINARY_DIR}/lookup_${optimizer}.py"
    "${CMAKE_BINARY_DIR}/lookup_${optimizer}_pt2.py")
endforeach()

# Generate the Python functions only if there is the backend support (for SSD
# optimizers)
foreach(optimizer ${SSD_OPTIMIZERS})
  list(APPEND gen_python_source_files
    "${CMAKE_BINARY_DIR}/lookup_${optimizer}_ssd.py")
endforeach()

# Generate the backend API for all optimizers to preserve the backward
# compatibility
list(APPEND gen_cpu_source_files
    "gen_embedding_forward_split_pt2_cpu_wrapper.cpp")
list(APPEND gen_gpu_host_source_files
     "gen_embedding_forward_split_pt2_cuda_wrapper.cpp")

foreach(optimizer ${ALL_OPTIMIZERS})
  list(APPEND gen_cpu_source_files
    "gen_embedding_backward_split_${optimizer}_cpu.cpp"
    "gen_embedding_backward_split_${optimizer}_pt2_cpu_wrapper.cpp"
    "gen_embedding_split_${optimizer}_pt2_autograd.cpp")
  list(APPEND gen_gpu_host_source_files
    "gen_embedding_backward_split_${optimizer}.cpp"
    "gen_embedding_backward_split_${optimizer}_pt2_cuda_wrapper.cpp")
endforeach()

foreach(optimizer ${GPU_OPTIMIZERS})
  list(APPEND gen_gpu_host_source_files
    "gen_embedding_backward_${optimizer}_split_unweighted_meta.cpp"
    "gen_embedding_backward_${optimizer}_split_weighted_meta.cpp")
endforeach()

list(APPEND gen_gpu_host_source_files
    "gen_embedding_backward_split_dense.cpp")

foreach(optimizer ${CPU_OPTIMIZERS})
  list(APPEND gen_cpu_source_files
    "gen_embedding_backward_${optimizer}_split_cpu.cpp")
endforeach()

foreach(optimizer ${GPU_OPTIMIZERS})
  list(APPEND gen_gpu_kernel_source_files
    "gen_embedding_optimizer_${optimizer}_split_device_kernel.cuh")
  foreach(wdesc ${WEIGHT_OPTIONS})
    list(APPEND gen_gpu_kernel_source_files
      "gen_embedding_backward_${optimizer}_split_${wdesc}_cuda.cu"
      "gen_embedding_backward_${optimizer}_split_${wdesc}_kernel_cta.cu"
      "gen_embedding_backward_${optimizer}_split_${wdesc}_kernel_warp.cu")
  endforeach()
endforeach()

foreach(optimizer ${VBE_OPTIMIZERS})
  # vbe is not supported in nobag
  foreach(wdesc weighted unweighted)
    list(APPEND gen_gpu_kernel_source_files
      "gen_embedding_backward_${optimizer}_split_${wdesc}_vbe_cuda.cu"
      "gen_embedding_backward_${optimizer}_split_${wdesc}_vbe_kernel_cta.cu"
      "gen_embedding_backward_${optimizer}_split_${wdesc}_vbe_kernel_warp.cu")
  endforeach()
endforeach()

foreach(optimizer ${GWD_OPTIMIZERS})
  # GWD is not supported in nobag
  foreach(wdesc weighted unweighted)
    list(APPEND gen_gpu_kernel_source_files
      "gen_embedding_backward_${optimizer}_split_${wdesc}_gwd_cuda.cu"
      "gen_embedding_backward_${optimizer}_split_${wdesc}_gwd_kernel_cta.cu"
      "gen_embedding_backward_${optimizer}_split_${wdesc}_gwd_kernel_warp.cu")
    if(";${VBE_OPTIMIZERS};" MATCHES ";${optimizer};")
      list(APPEND gen_gpu_kernel_source_files
        "gen_embedding_backward_${optimizer}_split_${wdesc}_vbe_gwd_cuda.cu"
        "gen_embedding_backward_${optimizer}_split_${wdesc}_vbe_gwd_kernel_cta.cu"
        "gen_embedding_backward_${optimizer}_split_${wdesc}_vbe_gwd_kernel_warp.cu")
    endif()
  endforeach()
endforeach()

foreach(optimizer ${DEFUSED_OPTIMIZERS})
  list(APPEND gen_defused_optim_source_files
    "gen_embedding_optimizer_${optimizer}_split.cpp"
    "gen_embedding_optimizer_${optimizer}_split_cuda.cu"
    "gen_embedding_optimizer_${optimizer}_split_kernel.cu")
  list(APPEND gen_defused_optim_py_files
    "${CMAKE_BINARY_DIR}/split_embedding_optimizer_${optimizer}.py")
endforeach()

foreach(optimizer ${SSD_OPTIMIZERS})
  list(APPEND gen_gpu_kernel_source_files
    "gen_embedding_optimizer_${optimizer}_ssd_device_kernel.cuh"
  )

  list(APPEND gen_gpu_host_source_files
    "gen_embedding_backward_ssd_${optimizer}.cpp"
  )

  foreach(wdesc weighted unweighted unweighted_nobag)
    list(APPEND gen_gpu_kernel_source_files
      "gen_embedding_backward_${optimizer}_ssd_${wdesc}_cuda.cu"
      "gen_embedding_backward_${optimizer}_ssd_${wdesc}_kernel_cta.cu"
      "gen_embedding_backward_${optimizer}_ssd_${wdesc}_kernel_warp.cu")
  endforeach()
  foreach(wdesc weighted unweighted)
    list(APPEND gen_gpu_kernel_source_files
      "gen_embedding_backward_${optimizer}_ssd_${wdesc}_vbe_cuda.cu"
      "gen_embedding_backward_${optimizer}_ssd_${wdesc}_vbe_kernel_cta.cu"
      "gen_embedding_backward_${optimizer}_ssd_${wdesc}_vbe_kernel_warp.cu")
  endforeach()

endforeach()

list(APPEND gen_defused_optim_py_files
    ${CMAKE_BINARY_DIR}/optimizer_args.py)


################################################################################
# FBGEMM_GPU Static Sources
################################################################################

set(fbgemm_gpu_sources_cpu_static
    codegen/training/forward/embedding_forward_split_cpu.cpp
    # codegen/inference/embedding_forward_quantized_host_cpu.cpp
    codegen/training/backward/embedding_backward_dense_host_cpu.cpp
    codegen/training/pt2/pt2_autograd_utils.cpp
    codegen/utils/embedding_bounds_check_host_cpu.cpp
    src/config/feature_gates.cpp
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
    src/sparse_ops/sparse_async_cumsum.cpp
    src/sparse_ops/sparse_ops_cpu.cpp
    src/sparse_ops/sparse_ops_meta.cpp
    # src/embedding_inplace_ops/embedding_inplace_update_cpu.cpp
    # src/split_embeddings_cache/linearize_cache_indices.cpp
    # src/split_embeddings_cache/lfu_cache_populate_byte.cpp
    # src/split_embeddings_cache/lru_cache_populate_byte.cpp
    # src/split_embeddings_cache/lxu_cache.cpp
    # src/split_embeddings_cache/split_embeddings_cache_ops.cpp
    src/split_embeddings_utils/split_embeddings_utils_cpu.cpp
    codegen/training/index_select/batch_index_select_dim0_ops.cpp
    codegen/training/index_select/batch_index_select_dim0_cpu_host.cpp)

if(NOT FBGEMM_CPU_ONLY)
  list(APPEND fbgemm_gpu_sources_cpu_static
    # codegen/inference/embedding_forward_quantized_host.cpp
    codegen/utils/embedding_bounds_check_host.cpp
    src/intraining_embedding_pruning_ops/intraining_embedding_pruning_gpu.cpp
    src/layout_transform_ops/layout_transform_ops_gpu.cpp
    src/permute_pooled_embedding_ops/permute_pooled_embedding_ops_gpu.cpp
    src/permute_pooled_embedding_ops/permute_pooled_embedding_ops_split_gpu.cpp
    src/quantize_ops/quantize_ops_gpu.cpp
    src/sparse_ops/sparse_ops_gpu.cpp
    src/split_embeddings_utils/split_embeddings_utils.cpp
    src/metric_ops/metric_ops_host.cpp
    # src/embedding_inplace_ops/embedding_inplace_update_gpu.cpp
    src/input_combine_ops/input_combine_gpu.cpp
    codegen/training/index_select/batch_index_select_dim0_host.cpp)

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
      codegen/utils/embedding_bounds_check_v1.cu
      codegen/utils/embedding_bounds_check_v2.cu
      # codegen/inference/embedding_forward_quantized_split_lookup.cu
      # src/embedding_inplace_ops/embedding_inplace_update.cu
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
      src/sparse_ops/sparse_async_cumsum.cu
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
      src/sparse_ops/sparse_zipf.cu
      # src/split_embeddings_cache/lfu_cache_find.cu
      # src/split_embeddings_cache/lfu_cache_populate.cu
      # src/split_embeddings_cache/lfu_cache_populate_byte.cu
      # src/split_embeddings_cache/lru_cache_find.cu
      # src/split_embeddings_cache/lru_cache_populate.cu
      # src/split_embeddings_cache/lru_cache_populate_byte.cu
      # src/split_embeddings_cache/lxu_cache.cu
      # src/split_embeddings_cache/linearize_cache_indices.cu
      # src/split_embeddings_cache/reset_weight_momentum.cu
      # src/split_embeddings_cache/split_embeddings_cache_ops.cu
      src/split_embeddings_utils/generate_vbe_metadata.cu
      src/split_embeddings_utils/get_infos_metadata.cu
      src/split_embeddings_utils/radix_sort_pairs.cu
      src/split_embeddings_utils/transpose_embedding_input.cu)
endif()


################################################################################
# FBGEMM_GPU Generated Sources Organized
################################################################################

set(fbgemm_gpu_sources_cpu_gen
  ${gen_cpu_source_files})

set(fbgemm_gpu_sources_gpu_gen
  ${gen_gpu_kernel_source_files}
  ${gen_gpu_host_source_files}
  ${gen_defused_optim_source_files})

if(USE_ROCM)
  prepend_filepaths(
    PREFIX ${CMAKE_BINARY_DIR}
    INPUT ${fbgemm_gpu_sources_cpu_gen}
    OUTPUT fbgemm_gpu_sources_cpu_gen)

  prepend_filepaths(
    PREFIX ${CMAKE_BINARY_DIR}
    INPUT ${fbgemm_gpu_sources_gpu_gen}
    OUTPUT fbgemm_gpu_sources_gpu_gen)
endif()


################################################################################
# FBGEMM_GPU C++ Modules
################################################################################

include(${FBGEMM_GPU}/cmake/Tbe.cmake)


# Test target to demonstrate that target deps works as intended
gpu_cpp_library(
  PREFIX
    embedding_inplace_ops
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  CPU_SRCS
    src/embedding_inplace_ops/embedding_inplace_update_cpu.cpp
  GPU_SRCS
    src/embedding_inplace_ops/embedding_inplace_update_gpu.cpp
    src/embedding_inplace_ops/embedding_inplace_update.cu
  GPU_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DESTINATION
    fbgemm_gpu)


gpu_cpp_library(
  PREFIX
    fbgemm_gpu_py
  TYPE
    MODULE
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  CPU_SRCS
    ${fbgemm_gpu_sources_cpu_static}
    ${fbgemm_gpu_sources_cpu_gen}
  GPU_SRCS
    ${fbgemm_gpu_sources_gpu_static}
    ${fbgemm_gpu_sources_gpu_gen}
  GPU_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DEPS
    asmjit
    fbgemm
    embedding_inplace_ops
    split_embeddings_cache
  DESTINATION
    fbgemm_gpu)


################################################################################
# FBGEMM_GPU Package
################################################################################

install(FILES ${gen_python_source_files}
  DESTINATION fbgemm_gpu/split_embedding_codegen_lookup_invokers)

install(FILES ${gen_defused_optim_py_files}
  DESTINATION fbgemm_gpu/split_embedding_optimizer_codegen)
