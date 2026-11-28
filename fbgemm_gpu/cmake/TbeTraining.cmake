# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Fetch All Sources
################################################################################

# Common
get_tbe_sources_list(static_cpu_files_common)
get_tbe_sources_list(static_gpu_files_common)

# Optimizers
get_tbe_sources_list(gen_defused_optim_src_files)
handle_genfiles_rocm(gen_defused_optim_src_files)

# Forward Split
get_tbe_sources_list(gen_cpu_files_forward_split)
get_tbe_sources_list(gen_gpu_files_forward_split)
handle_genfiles_rocm(gen_cpu_files_forward_split)
handle_genfiles_rocm(gen_gpu_files_forward_split)

# Backward Split
get_tbe_sources_list(static_cpu_files_training)
get_tbe_sources_list(gen_cpu_files_training)
get_tbe_sources_list(gen_gpu_files_training)
get_tbe_sources_list(gen_cpu_files_training_pt2)
get_tbe_sources_list(gen_gpu_files_training_pt2)
get_tbe_sources_list(gen_gpu_files_training_dense)
get_tbe_sources_list(gen_gpu_files_training_dense_helpers)
get_tbe_sources_list(gen_gpu_files_training_gwd_helpers)
get_tbe_sources_list(gen_gpu_files_training_split_host)
get_tbe_sources_list(gen_gpu_files_training_gwd)
get_tbe_sources_list(gen_gpu_files_training_vbe)
get_tbe_sources_list(gen_gpu_files_training_vbe_helpers)
handle_genfiles_rocm(gen_cpu_files_training)
handle_genfiles_rocm(gen_gpu_files_training)
handle_genfiles_rocm(gen_cpu_files_training_pt2)
handle_genfiles_rocm(gen_gpu_files_training_pt2)
handle_genfiles_rocm(gen_gpu_files_training_dense)
handle_genfiles_rocm(gen_gpu_files_training_dense_helpers)
handle_genfiles_rocm(gen_gpu_files_training_gwd_helpers)
handle_genfiles_rocm(gen_gpu_files_training_split_host)
handle_genfiles_rocm(gen_gpu_files_training_gwd)
handle_genfiles_rocm(gen_gpu_files_training_vbe)
handle_genfiles_rocm(gen_gpu_files_training_vbe_helpers)

# Index Select
get_tbe_sources_list(static_cpu_files_index_select)
get_tbe_sources_list(static_gpu_files_index_select)
get_tbe_sources_list(gen_gpu_files_index_select)
handle_genfiles_rocm(gen_gpu_files_index_select)

# Generated Python sources
get_tbe_sources_list(gen_py_files_training)
get_tbe_sources_list(gen_py_files_defused_optim)
handle_genfiles(gen_py_files_training)
handle_genfiles(gen_py_files_defused_optim)


################################################################################
# FBGEMM_GPU Generated HIP-Specific Sources
################################################################################

get_tbe_sources_list(gen_hip_files_training)
handle_genfiles_rocm(gen_hip_files_training)


################################################################################
# TBE C++ Training Targets
################################################################################

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_config
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  SYSTEM_INCLUDE_DIRS
    ${fbgemm_thirdparty_include_directories}
  CPU_SRCS
    src/config/feature_gates.cpp
    src/config/feature_gates_torch_op.cpp
  DESTINATION
    fbgemm_gpu)

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_tbe_utils
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  SYSTEM_INCLUDE_DIRS
    ${fbgemm_thirdparty_include_directories}
  CPU_SRCS
    src/split_embeddings_utils/split_embeddings_utils_cpu.cpp
    src/split_embeddings_utils/split_embeddings_utils_meta.cpp
  GPU_SRCS
    src/split_embeddings_utils/split_embeddings_utils.cpp
    src/split_embeddings_utils/generate_vbe_metadata.cu
    src/split_embeddings_utils/get_infos_metadata.cu
    src/split_embeddings_utils/radix_sort_pairs.cu
    src/split_embeddings_utils/transpose_embedding_input.cu
  DESTINATION
    fbgemm_gpu)

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_sparse_async_cumsum
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  SYSTEM_INCLUDE_DIRS
    ${fbgemm_thirdparty_include_directories}
  CPU_SRCS
    src/sparse_ops/sparse_async_cumsum.cpp
  GPU_SRCS
    src/sparse_ops/sparse_async_cumsum.cu
  DEPS
    fbgemm_gpu_tbe_utils
  DESTINATION
    fbgemm_gpu)

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_tbe_common
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  SYSTEM_INCLUDE_DIRS
    ${fbgemm_thirdparty_include_directories}
  CPU_SRCS
    ${static_cpu_files_common}
  GPU_SRCS
    ${static_gpu_files_common}
  NVCC_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DEPS
    fbgemm
    fbgemm_gpu_config
  DESTINATION
    fbgemm_gpu)

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_tbe_optimizers
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  SYSTEM_INCLUDE_DIRS
    ${fbgemm_thirdparty_include_directories}
  GPU_SRCS
    ${gen_defused_optim_src_files}
  NVCC_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DESTINATION
    fbgemm_gpu)

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_tbe_training_forward
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  SYSTEM_INCLUDE_DIRS
    ${fbgemm_thirdparty_include_directories}
  CPU_SRCS
    ${gen_cpu_files_forward_split}
  GPU_SRCS
    ${gen_gpu_files_forward_split}
  NVCC_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DEPS
    fbgemm_gpu_tbe_common
  DESTINATION
    fbgemm_gpu)

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_tbe_training_backward_pt2
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  SYSTEM_INCLUDE_DIRS
    ${fbgemm_thirdparty_include_directories}
  CPU_SRCS
    ${gen_cpu_files_training_pt2}
  GPU_SRCS
    ${gen_gpu_files_training_pt2}
  NVCC_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DEPS
    fbgemm
    fbgemm_gpu_config
    fbgemm_gpu_tbe_cache
    fbgemm_gpu_tbe_common
    fbgemm_gpu_tbe_utils
    fbgemm_gpu_sparse_async_cumsum
  DESTINATION
    fbgemm_gpu)

set(tbe_backward_gpu_sources ${gen_gpu_files_training})
set(tbe_backward_gpu_deps)

if(FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_CUDA)
  get_tbe_sources_list(GPU_OPTIMIZERS)
  # Kernel templates inherit the hidden visibility of PyTorch packed-accessor types.
  set(tbe_backward_gpu_helpers ${gen_gpu_files_training})
  list(FILTER tbe_backward_gpu_helpers INCLUDE REGEX
    "(^|/)gen_embedding_backward_split_grad_embedding_ops\\.cu$")
  list(REMOVE_ITEM tbe_backward_gpu_sources ${tbe_backward_gpu_helpers})
  foreach(optimizer IN LISTS GPU_OPTIMIZERS)
    set(optimizer_sources ${gen_gpu_files_training})
    list(FILTER optimizer_sources INCLUDE REGEX
      "(^|/)gen_embedding_backward_${optimizer}_(split|ssd)_")
    list(REMOVE_ITEM tbe_backward_gpu_sources ${optimizer_sources})

    gpu_cpp_library(
      PREFIX
        fbgemm_gpu_tbe_training_backward_${optimizer}
      TYPE
        SHARED
      INCLUDE_DIRS
        ${fbgemm_sources_include_directories}
      SYSTEM_INCLUDE_DIRS
        ${fbgemm_thirdparty_include_directories}
      GPU_SRCS
        ${optimizer_sources}
        ${tbe_backward_gpu_helpers}
      NVCC_FLAGS
        ${TORCH_CUDA_OPTIONS}
      DEPS
        fbgemm
        fbgemm_gpu_config
        fbgemm_gpu_tbe_cache
        fbgemm_gpu_tbe_common
        fbgemm_gpu_tbe_utils
        fbgemm_gpu_sparse_async_cumsum
        fbgemm_gpu_tbe_training_backward_common
      DESTINATION
        fbgemm_gpu)
    list(APPEND tbe_backward_gpu_deps
      fbgemm_gpu_tbe_training_backward_${optimizer})
  endforeach()

  gpu_cpp_library(
    PREFIX
      fbgemm_gpu_tbe_training_backward_common
    TYPE
      SHARED
    INCLUDE_DIRS
      ${fbgemm_sources_include_directories}
    SYSTEM_INCLUDE_DIRS
      ${fbgemm_thirdparty_include_directories}
    GPU_SRCS
      ${tbe_backward_gpu_sources}
    NVCC_FLAGS
      ${TORCH_CUDA_OPTIONS}
    DEPS
      fbgemm
      fbgemm_gpu_config
      fbgemm_gpu_tbe_cache
      fbgemm_gpu_tbe_common
      fbgemm_gpu_tbe_utils
      fbgemm_gpu_sparse_async_cumsum
    DESTINATION
      fbgemm_gpu)
  list(APPEND tbe_backward_gpu_deps
    fbgemm_gpu_tbe_training_backward_common)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(PREPEND tbe_backward_gpu_deps "-Wl,--no-as-needed")
    list(APPEND tbe_backward_gpu_deps "-Wl,--as-needed")
  endif()
  set(tbe_backward_gpu_sources)
endif()

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_tbe_training_backward
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  SYSTEM_INCLUDE_DIRS
    ${fbgemm_thirdparty_include_directories}
  CPU_SRCS
    ${static_cpu_files_training}
    ${gen_cpu_files_training}
  GPU_SRCS
    ${tbe_backward_gpu_sources}
  HIP_SPECIFIC_SRCS
    ${gen_hip_files_training}
  NVCC_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DEPS
    fbgemm
    fbgemm_gpu_config
    fbgemm_gpu_tbe_cache
    fbgemm_gpu_tbe_common
    fbgemm_gpu_tbe_utils
    fbgemm_gpu_sparse_async_cumsum
    ${tbe_backward_gpu_deps}
  DESTINATION
    fbgemm_gpu)

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_tbe_training_backward_gwd
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  SYSTEM_INCLUDE_DIRS
    ${fbgemm_thirdparty_include_directories}
  GPU_SRCS
    ${gen_gpu_files_training_gwd_helpers}
    ${gen_gpu_files_training_gwd}
  NVCC_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DEPS
    fbgemm
    fbgemm_gpu_tbe_cache
    fbgemm_gpu_tbe_common
    fbgemm_gpu_tbe_training_backward
    fbgemm_gpu_config
    fbgemm_gpu_tbe_utils
    fbgemm_gpu_sparse_async_cumsum
  DESTINATION
    fbgemm_gpu)

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_tbe_training_backward_vbe
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  SYSTEM_INCLUDE_DIRS
    ${fbgemm_thirdparty_include_directories}
  GPU_SRCS
    ${gen_gpu_files_training_vbe_helpers}
    ${gen_gpu_files_training_vbe}
  NVCC_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DEPS
    fbgemm
    fbgemm_gpu_tbe_cache
    fbgemm_gpu_tbe_common
    fbgemm_gpu_tbe_training_backward
    fbgemm_gpu_config
    fbgemm_gpu_tbe_utils
    fbgemm_gpu_sparse_async_cumsum
  DESTINATION
    fbgemm_gpu)

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_tbe_training_backward_dense
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  SYSTEM_INCLUDE_DIRS
    ${fbgemm_thirdparty_include_directories}
  GPU_SRCS
    ${gen_gpu_files_training_dense_helpers}
    ${gen_gpu_files_training_dense}
  NVCC_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DEPS
    fbgemm
    fbgemm_gpu_tbe_cache
    fbgemm_gpu_tbe_common
    fbgemm_gpu_tbe_training_backward
    fbgemm_gpu_config
    fbgemm_gpu_tbe_utils
    fbgemm_gpu_sparse_async_cumsum
  DESTINATION
    fbgemm_gpu)

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_tbe_training_backward_split_host
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  SYSTEM_INCLUDE_DIRS
    ${fbgemm_thirdparty_include_directories}
  GPU_SRCS
    ${gen_gpu_files_training_split_host}
  NVCC_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DEPS
    fbgemm_gpu_config
    fbgemm_gpu_tbe_utils
  DESTINATION
    fbgemm_gpu)

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_tbe_index_select
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  SYSTEM_INCLUDE_DIRS
    ${fbgemm_thirdparty_include_directories}
  CPU_SRCS
    ${static_cpu_files_index_select}
  GPU_SRCS
    ${static_gpu_files_index_select}
    ${gen_gpu_files_index_select}
  NVCC_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DEPS
    fbgemm_gpu_config
    fbgemm_gpu_sparse_async_cumsum
    fbgemm_gpu_tbe_utils
  DESTINATION
    fbgemm_gpu)


################################################################################
# TBE Python Targets
################################################################################

install(FILES ${gen_py_files_training}
  DESTINATION fbgemm_gpu/split_embedding_codegen_lookup_invokers)

install(FILES ${gen_py_files_defused_optim}
  DESTINATION fbgemm_gpu/split_embedding_optimizer_codegen)

# Install the TBE Python subpackages that are not auto-discovered by
# setuptools.find_packages() in certain scikit-build configurations.
# This ensures tbe/monitoring/ (and other tbe/ subpackages) are included
# in the OSS wheel.
install(FILES
  fbgemm_gpu/tbe/__init__.py
  DESTINATION fbgemm_gpu/tbe)

install(DIRECTORY
  fbgemm_gpu/tbe/monitoring
  DESTINATION fbgemm_gpu/tbe
  FILES_MATCHING PATTERN "*.py")
