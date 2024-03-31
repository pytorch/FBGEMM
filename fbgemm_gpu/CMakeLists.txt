# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# CMake Prelude
################################################################################

cmake_minimum_required(VERSION 3.25.0 FATAL_ERROR)

function(BLOCK_PRINT)
  message("================================================================================")
  foreach(ARG IN LISTS ARGN)
     message("${ARG}")
  endforeach()
  message("================================================================================")
  message("")
endfunction()

set(CMAKEMODULES ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules)
set(FBGEMM ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(THIRDPARTY ${FBGEMM}/third_party)


################################################################################
# FBGEMM_GPU Build Options
################################################################################

option(FBGEMM_CPU_ONLY  "Build FBGEMM_GPU without GPU support" OFF)
option(USE_ROCM         "Build FBGEMM_GPU for ROCm" OFF)

if((NOT FBGEMM_CPU_ONLY) AND
   ((EXISTS "/opt/rocm/") OR (EXISTS $ENV{ROCM_PATH})) AND
   (NOT EXISTS "/bin/nvcc"))
  message(
    "CMake has been set to build a non-CPU variant"
    "and AMD GPU has been detected; "
    "will default to ROCm build"
  )
  set(USE_ROCM ON)
endif()

if(FBGEMM_CPU_ONLY)
  BLOCK_PRINT("Building the CPU-only variant of FBGEMM-GPU")
elseif(USE_ROCM)
  BLOCK_PRINT("Building the ROCm variant of FBGEMM-GPU")
else()
  BLOCK_PRINT("Building the CUDA variant of FBGEMM-GPU")
endif()


################################################################################
# FBGEMM_GPU C++ Setup
################################################################################

# Set the default C++ standard to C++20 if CMAKE_CXX_STANDARD is not supplied
# by CMake command invocation.
# Individual targets can have this value overridden; see
# https://cmake.org/cmake/help/latest/variable/CMAKE_CXX_STANDARD.html
# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
# https://cmake.org/cmake/help/latest/prop_tgt/HIP_STANDARD.html
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_HIP_STANDARD 20)
  set(CXX_STANDARD 20)
  set(HIP_STANDARD 20)
endif()
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(HIP_STANDARD_REQUIRED ON)

# Set the default C standard to C17
# Individual targets can have this value overridden; see
# https://cmake.org/cmake/help/latest/variable/CMAKE_C_STANDARD.html
# https://cmake.org/cmake/help/latest/prop_tgt/C_STANDARD.html
set(C_STANDARD 20)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(DEFINED GLIBCXX_USE_CXX11_ABI)
  if(${GLIBCXX_USE_CXX11_ABI} EQUAL 1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
  endif()
endif()

BLOCK_PRINT(
  "Default C compiler flags"
  "(values may be overridden by CMAKE_CXX_STANDARD and CXX_STANDARD):"
  ""
  "${CMAKE_C_FLAGS}"
)

BLOCK_PRINT(
  "Default C++ compiler flags"
  "(values may be overridden by CMAKE_CXX_STANDARD and CXX_STANDARD):"
  ""
  "${CMAKE_CXX_FLAGS}"
)

# Strip all symbols from the .SO file after building
add_link_options($<$<CONFIG:RELEASE>:-s>)

# Set flags for AVX2
set(AVX2_FLAGS "-mavx2;-mf16c;-mfma;-fopenmp")
if(NOT FBGEMM_CPU_ONLY AND WSL_MODE)
  # NVCC in WSL complains about unknown -mavx options
  # https://github.com/pytorch/FBGEMM/issues/2135
  set(AVX2_FLAGS "-Xcompiler;-mavx;-Xcompiler;-mavx2;-Xcompiler;-mf16c;-Xcompiler;-mfma;-fopenmp")
endif()

# Set flags for AVX512
set(AVX512_FLAGS "-mavx2;-mf16c;-mfma;-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl;-fopenmp")
if(NOT FBGEMM_CPU_ONLY AND WSL_MODE)
  set(AVX512_FLAGS "-Xcompiler;-mavx2;-Xcompiler;-mf16c;-Xcompiler;-mfma;-Xcompiler;-mavx512f;-Xcompiler;-mavx512bw;-Xcompiler;-mavx512dq;-Xcompiler;-mavx512vl;-fopenmp")
endif()

set(CMAKE_CODEGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/codegen)


################################################################################
# FBGEMM_GPU Build Kickstart
################################################################################

if(SKBUILD)
  BLOCK_PRINT("The project is built using scikit-build")
endif()

if(FBGEMM_CPU_ONLY OR USE_ROCM)
  project(
    fbgemm_gpu
    VERSION 0.3.1
    LANGUAGES CXX C)
else()
  project(
    fbgemm_gpu
    VERSION 0.3.1
    LANGUAGES CXX C CUDA)
endif()

include(${CMAKEMODULES}/FindAVX.cmake)


################################################################################
# PyTorch Dependencies Setup
################################################################################

find_package(Torch REQUIRED)

#
# Toch Cuda Extensions are normally compiled with the flags below. However we
# disabled -D__CUDA_NO_HALF_CONVERSIONS__ here as it caused "error: no suitable
# constructor exists to convert from "int" to "__half" errors in
# gen_embedding_forward_quantized_split_[un]weighted_codegen_cuda.cu
#

set(TORCH_CUDA_OPTIONS
    --expt-relaxed-constexpr -D__CUDA_NO_HALF_OPERATORS__
    # -D__CUDA_NO_HALF_CONVERSIONS__
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__)


################################################################################
# CUDA Setup
################################################################################

# Set NVML_LIB_PATH if provided, or detect the default lib path
if(NOT NVML_LIB_PATH)
  set(DEFAULT_NVML_LIB_PATH
      "${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libnvidia-ml.so")

  if(EXISTS ${DEFAULT_NVML_LIB_PATH})
    message(STATUS "Setting NVML_LIB_PATH: \
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libnvidia-ml.so")
    set(NVML_LIB_PATH "${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libnvidia-ml.so")
  endif()
endif()

if(NVML_LIB_PATH)
  message(STATUS "Found NVML_LIB_PATH: ${NVML_LIB_PATH}")
endif()


################################################################################
# ROCm and HIPify Setup
################################################################################

if(USE_ROCM)
  # Load CMake modules
  list(APPEND CMAKE_MODULE_PATH
    "${PROJECT_SOURCE_DIR}/cmake"
    "${THIRDPARTY}/hipify_torch/cmake")
  include(Hip)
  include(Hipify)

  # Configure compiler for HIP
  list(APPEND HIP_HCC_FLAGS
    " \"-Wno-#pragma-messages\" "
    " \"-Wno-#warnings\" "
    -Wno-cuda-compat
    -Wno-deprecated-declarations
    -Wno-format
    -Wno-ignored-attributes
    -Wno-unused-result)

  BLOCK_PRINT(
    "HIP found: ${HIP_FOUND}"
    "HIPCC compiler flags:"
    ""
    "${HIP_HCC_FLAGS}"
  )
endif()


################################################################################
# Third Party Sources
################################################################################

file(GLOB_RECURSE asmjit_sources
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/asmjit/src/asmjit/*/*.cpp")


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
    sgd)

# Individual optimizers (not fused with SplitTBE backward)
set(DEFUSED_OPTIMIZERS
    rowwise_adagrad)

set(WEIGHT_OPTIONS
    weighted
    unweighted_nobag
    unweighted)


################################################################################
# TBE Code Generation
################################################################################

macro(RUN_GEN_SCRIPT SCRIPT)
  set(rocm_flag "")
  if(USE_ROCM)
    set(rocm_flag --is_rocm)
  endif()

  BLOCK_PRINT(
    "Running code generation script ..."
    "${PYTHON_EXECUTABLE} ${SCRIPT} --opensource ${rocm_flag}"
  )

  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" ${SCRIPT} "--opensource" ${rocm_flag})
endmacro()

foreach(script
    "${CMAKE_CODEGEN_DIR}/embedding_backward_code_generator.py"
    "${CMAKE_CODEGEN_DIR}/genscript/generate_embedding_optimizer.py"
    "${CMAKE_CODEGEN_DIR}/genscript/generate_forward_quantized.py")
    RUN_GEN_SCRIPT(${script})
endforeach()


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
    "gen_embedding_forward_split_weighted_vbe_codegen_cuda.cu"
    "gen_embedding_forward_split_unweighted_vbe_codegen_cuda.cu"
    "gen_batch_index_select_dim0_forward_codegen_cuda.cu"
    "gen_batch_index_select_dim0_forward_kernel.cu"
    "gen_batch_index_select_dim0_forward_kernel_small.cu"
    "gen_batch_index_select_dim0_backward_codegen_cuda.cu"
    "gen_batch_index_select_dim0_backward_kernel_cta.cu"
    "gen_batch_index_select_dim0_backward_kernel_warp.cu"
    "gen_embedding_backward_split_grad_embedding_ops.cu"
    "gen_embedding_backward_split_grad_index_select.cu"
    "gen_embedding_backward_common_split_device_kernel.cuh"
    "gen_embedding_backward_batch_index_select_split_device_kernel.cuh"
)

if(NOT USE_ROCM)
  list(APPEND gen_gpu_kernel_source_files
    "gen_embedding_forward_split_weighted_v2_kernel.cu"
    "gen_embedding_forward_split_unweighted_v2_kernel.cu")
endif()

foreach(wdesc dense split)
  list(APPEND gen_gpu_kernel_source_files
    "gen_embedding_forward_${wdesc}_unweighted_nobag_kernel_small.cu")
endforeach()

foreach(wdesc ${WEIGHT_OPTIONS})
  list(APPEND gen_gpu_kernel_source_files
      "gen_embedding_forward_quantized_split_nbit_host_${wdesc}_codegen_cuda.cu"
      "gen_embedding_forward_dense_${wdesc}_kernel.cu"
      "gen_embedding_backward_dense_split_${wdesc}_cuda.cu"
      "gen_embedding_backward_dense_split_${wdesc}_kernel_cta.cu"
      "gen_embedding_backward_dense_split_${wdesc}_kernel_warp.cu"
      "gen_embedding_forward_split_${wdesc}_kernel.cu"
      "gen_embedding_backward_${wdesc}_split_device_kernel.cuh")

  foreach(etype fp32 fp16 fp8 int8 int4 int2)
    list(APPEND gen_gpu_kernel_source_files
       "gen_embedding_forward_quantized_split_nbit_kernel_${wdesc}_${etype}_codegen_cuda.cu")
  endforeach()
endforeach()

# Generate VBE files
foreach(wdesc weighted unweighted)
  list(APPEND gen_gpu_kernel_source_files
      "gen_embedding_forward_split_${wdesc}_vbe_kernel.cu"
      "gen_embedding_backward_${wdesc}_vbe_split_device_kernel.cuh")
endforeach()

set(gen_cpu_source_files
    "gen_embedding_forward_quantized_unweighted_codegen_cpu.cpp"
    "gen_embedding_forward_quantized_weighted_codegen_cpu.cpp"
    "gen_embedding_backward_dense_split_cpu.cpp")

set(gen_python_source_files ${CMAKE_BINARY_DIR}/__init__.py)

# For each of the optimizers, generate the backward split variant by adding
# the Python, CPU-only, GPU host, and GPU kernel source files

# Generate the Python functions only if there is the backend support
foreach(optimizer
    ${COMMON_OPTIMIZERS}
    ${CPU_ONLY_OPTIMIZERS}
    ${GPU_ONLY_OPTIMIZERS})
  list(APPEND gen_python_source_files
    "${CMAKE_BINARY_DIR}/lookup_${optimizer}.py")
  list(APPEND gen_python_source_files
    "${CMAKE_BINARY_DIR}/lookup_${optimizer}_pt2.py")
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
    "gen_embedding_backward_split_${optimizer}_pt2_cpu_wrapper.cpp")
  list(APPEND gen_gpu_host_source_files
    "gen_embedding_backward_split_${optimizer}.cpp"
    "gen_embedding_split_${optimizer}_pt2_autograd.cpp"
    "gen_embedding_backward_split_${optimizer}_pt2_cuda_wrapper.cpp")
endforeach()

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

foreach(optimizer ${DEFUSED_OPTIMIZERS})
  list(APPEND gen_defused_optim_source_files
    "gen_embedding_optimizer_${optimizer}_split.cpp"
    "gen_embedding_optimizer_${optimizer}_split_cuda.cu"
    "gen_embedding_optimizer_${optimizer}_split_kernel.cu")
  list(APPEND gen_defused_optim_py_files
    "${CMAKE_BINARY_DIR}/split_embedding_optimizer_${optimizer}.py")
endforeach()


################################################################################
# FBGEMM_GPU Generated Sources
################################################################################

if(CXX_AVX2_FOUND)
  set_source_files_properties(${gen_cpu_source_files}
    PROPERTIES COMPILE_OPTIONS "${AVX2_FLAGS}")
else()
  set_source_files_properties(${gen_cpu_source_files}
    PROPERTIES COMPILE_OPTIONS "-fopenmp")
endif()

set_source_files_properties(${gen_cpu_source_files}
  PROPERTIES INCLUDE_DIRECTORIES
  "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_SOURCE_DIR}/../include;${THIRDPARTY}/asmjit/src"
)

set_source_files_properties(${gen_gpu_host_source_files}
  PROPERTIES INCLUDE_DIRECTORIES
  "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_SOURCE_DIR}/../include"
)

set_source_files_properties(${gen_gpu_kernel_source_files}
  PROPERTIES INCLUDE_DIRECTORIES
  "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/include")

set_source_files_properties(${gen_gpu_kernel_source_files}
  PROPERTIES COMPILE_OPTIONS
  "${TORCH_CUDA_OPTIONS}")

set_source_files_properties(${gen_defused_optim_source_files}
  PROPERTIES INCLUDE_DIRECTORIES
  "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_SOURCE_DIR}/../include"
)

if(NOT FBGEMM_CPU_ONLY)
  set(fbgemm_gpu_sources_gen
    ${gen_gpu_kernel_source_files}
    ${gen_gpu_host_source_files}
    ${gen_cpu_source_files}
    ${gen_defused_optim_source_files})
else()
  set(fbgemm_gpu_sources_gen
    ${gen_cpu_source_files}
    # To force generate_embedding_optimizer to generate Python files
    ${gen_defused_optim_py_files}
  )
endif()


################################################################################
# FBGEMM (not FBGEMM_GPU) Sources
################################################################################

set(fbgemm_sources_normal
  "${FBGEMM}/src/EmbeddingSpMDM.cc"
  "${FBGEMM}/src/EmbeddingSpMDMAutovec.cc"
  "${FBGEMM}/src/EmbeddingSpMDMNBit.cc"
  "${FBGEMM}/src/QuantUtils.cc"
  "${FBGEMM}/src/RefImplementations.cc"
  "${FBGEMM}/src/RowWiseSparseAdagradFused.cc"
  "${FBGEMM}/src/SparseAdagrad.cc"
  "${FBGEMM}/src/Utils.cc")

set(fbgemm_sources_avx2
  "${FBGEMM}/src/EmbeddingSpMDMAvx2.cc"
  "${FBGEMM}/src/QuantUtilsAvx2.cc")

set(fbgemm_sources_avx512
  "${FBGEMM}/src/EmbeddingSpMDMAvx512.cc")

if(CXX_AVX2_FOUND)
  set_source_files_properties(${fbgemm_sources_avx2}
    PROPERTIES COMPILE_OPTIONS
    "${AVX2_FLAGS}")
endif()

if(CXX_AVX512_FOUND)
  set_source_files_properties(${fbgemm_sources_avx512}
    PROPERTIES COMPILE_OPTIONS
    "${AVX512_FLAGS}")
endif()

set(fbgemm_sources ${fbgemm_sources_normal})
if(CXX_AVX2_FOUND)
  set(fbgemm_sources
    ${fbgemm_sources}
    ${fbgemm_sources_avx2})
endif()
if(NOT USE_ROCM AND CXX_AVX512_FOUND)
  set(fbgemm_sources
    ${fbgemm_sources}
    ${fbgemm_sources_avx2}
    ${fbgemm_sources_avx512})
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_AVX512=1")
endif()

set(fbgemm_sources_include_directories
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${FBGEMM}/include
  ${THIRDPARTY}/asmjit/src
  ${THIRDPARTY}/cpuinfo/include)

set_source_files_properties(${fbgemm_sources}
  PROPERTIES INCLUDE_DIRECTORIES
  "${fbgemm_sources_include_directories}")


################################################################################
# FBGEMM_GPU Static Sources
################################################################################

set(fbgemm_gpu_sources_static_cpu
    codegen/embedding_forward_split_cpu.cpp
    codegen/inference/embedding_forward_quantized_host_cpu.cpp
    codegen/embedding_backward_dense_host_cpu.cpp
    codegen/embedding_bounds_check_host_cpu.cpp
    src/merge_pooled_embedding_ops/merge_pooled_embedding_ops_cpu.cpp
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
    src/sparse_ops/sparse_ops_meta.cpp
    src/embedding_inplace_ops/embedding_inplace_update_cpu.cpp
    src/split_embeddings_cache/linearize_cache_indices.cpp
    src/split_embeddings_cache/lfu_cache_populate_byte.cpp
    src/split_embeddings_cache/lru_cache_populate_byte.cpp
    src/split_embeddings_cache/lxu_cache.cpp
    src/split_embeddings_cache/split_embeddings_cache_ops.cpp
    codegen/batch_index_select_dim0_cpu_host.cpp)

if(NOT FBGEMM_CPU_ONLY)
  list(APPEND fbgemm_gpu_sources_static_cpu
    codegen/inference/embedding_forward_quantized_host.cpp
    codegen/embedding_backward_dense_host.cpp
    codegen/embedding_bounds_check_host.cpp
    src/memory_utils/memory_utils.cpp
    src/memory_utils/memory_utils_ops.cpp
    src/memory_utils/memory_utils_ops_cpu.cpp
    src/layout_transform_ops/layout_transform_ops_gpu.cpp
    src/permute_pooled_embedding_ops/permute_pooled_embedding_ops_gpu.cpp
    src/permute_pooled_embedding_ops/permute_pooled_embedding_ops_split_gpu.cpp
    src/quantize_ops/quantize_ops_gpu.cpp
    src/sparse_ops/sparse_ops_gpu.cpp
    src/split_embeddings_utils/split_embeddings_utils.cpp
    src/split_embeddings_cache/split_embeddings_cache_ops.cu
    src/metric_ops/metric_ops_host.cpp
    src/embedding_inplace_ops/embedding_inplace_update_gpu.cpp
    src/input_combine_ops/input_combine_gpu.cpp
    codegen/batch_index_select_dim0_host.cpp)

  if(NVML_LIB_PATH OR USE_ROCM)
    message(STATUS "Adding merge_pooled_embeddings sources")
    list(APPEND fbgemm_gpu_sources_static_cpu
      src/merge_pooled_embedding_ops/merge_pooled_embedding_ops_gpu.cpp
      src/topology_utils.cpp)
  else()
    message(STATUS "Skipping merge_pooled_embeddings sources")
  endif()
endif()

if(CXX_AVX2_FOUND)
  set_source_files_properties(${fbgemm_gpu_sources_static_cpu}
    PROPERTIES COMPILE_OPTIONS
    "${AVX2_FLAGS}")
else()
  set_source_files_properties(${fbgemm_gpu_sources_static_cpu}
    PROPERTIES COMPILE_OPTIONS
    "-fopenmp")
endif()

if(NOT FBGEMM_CPU_ONLY)
  set(fbgemm_gpu_sources_static_gpu
      codegen/embedding_bounds_check.cu
      codegen/inference/embedding_forward_quantized_split_lookup.cu
      src/memory_utils/memory_utils.cu
      src/memory_utils/memory_utils_ops.cu
      src/embedding_inplace_ops/embedding_inplace_update.cu
      src/histogram_binning_calibration_ops.cu
      src/input_combine_ops/input_combine.cu
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
      src/quantize_ops/quantize_bfloat16.cu
      src/quantize_ops/quantize_fp8_rowwise.cu
      src/quantize_ops/quantize_fused_8bit_rowwise.cu
      src/quantize_ops/quantize_fused_nbit_rowwise.cu
      src/quantize_ops/quantize_hfp8.cu
      src/quantize_ops/quantize_msfp.cu
      src/quantize_ops/quantize_padded_fp8_rowwise.cu
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
      src/split_embeddings_cache/lfu_cache_find.cu
      src/split_embeddings_cache/lfu_cache_populate.cu
      src/split_embeddings_cache/lfu_cache_populate_byte.cu
      src/split_embeddings_cache/lru_cache_find.cu
      src/split_embeddings_cache/lru_cache_populate.cu
      src/split_embeddings_cache/lru_cache_populate_byte.cu
      src/split_embeddings_cache/lxu_cache.cu
      src/split_embeddings_cache/linearize_cache_indices.cu
      src/split_embeddings_cache/reset_weight_momentum.cu
      src/split_embeddings_utils/generate_vbe_metadata.cu
      src/split_embeddings_utils/get_infos_metadata.cu
      src/split_embeddings_utils/radix_sort_pairs.cu
      src/split_embeddings_utils/transpose_embedding_input.cu)

  set_source_files_properties(${fbgemm_gpu_sources_static_gpu}
    PROPERTIES COMPILE_OPTIONS
    "${TORCH_CUDA_OPTIONS}")

  set_source_files_properties(${fbgemm_gpu_sources_static_gpu}
    PROPERTIES INCLUDE_DIRECTORIES
    "${fbgemm_sources_include_directories}")
endif()

set_source_files_properties(${fbgemm_gpu_sources_static_cpu}
  PROPERTIES INCLUDE_DIRECTORIES
  "${fbgemm_sources_include_directories}")

if(NOT FBGEMM_CPU_ONLY)
  set(fbgemm_gpu_sources_static
    ${fbgemm_gpu_sources_static_gpu}
    ${fbgemm_gpu_sources_static_cpu})
else()
  set(fbgemm_gpu_sources_static
    ${fbgemm_gpu_sources_static_cpu})
endif()


################################################################################
# FBGEMM_GPU HIP Code Generation
################################################################################

if(USE_ROCM)
  # HIPify CUDA code
  set(header_include_dir
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR})

  hipify(CUDA_SOURCE_DIR ${PROJECT_SOURCE_DIR}
        HEADER_INCLUDE_DIR ${header_include_dir})

  # Get the absolute paths of all generated sources
  set(fbgemm_gpu_sources_gen_abs)
  foreach(source_gen_filename ${fbgemm_gpu_sources_gen})
    list(APPEND fbgemm_gpu_sources_gen_abs
      "${CMAKE_BINARY_DIR}/${source_gen_filename}")
  endforeach()

  # HIPify FBGEMM, FBGEMM_GPU static, and FBGEMM_GPU generated sources
  get_hipified_list("${fbgemm_gpu_sources_static}" fbgemm_gpu_sources_static)
  get_hipified_list("${fbgemm_gpu_sources_gen_abs}" fbgemm_gpu_sources_gen_abs)
  get_hipified_list("${fbgemm_sources}" fbgemm_sources)

  # Combine all HIPified sources
  set(fbgemm_gpu_sources_hip
    ${fbgemm_sources}
    ${fbgemm_gpu_sources_static}
    ${fbgemm_gpu_sources_gen_abs})

  set_source_files_properties(${fbgemm_gpu_sources_hip}
                              PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

  # Add FBGEMM include/
  hip_include_directories("${fbgemm_sources_include_directories}")
endif()


################################################################################
# FBGEMM_GPU Full Python Module
################################################################################

if(USE_ROCM)
  # Create a HIP library if using ROCm
  hip_add_library(fbgemm_gpu_py SHARED
    ${asmjit_sources}
    ${fbgemm_gpu_sources_hip}
    ${FBGEMM_HIP_HCC_LIBRARIES}
    HIPCC_OPTIONS
    ${HIP_HCC_FLAGS})

  target_include_directories(fbgemm_gpu_py PUBLIC
    ${FBGEMM_HIP_INCLUDE}
    ${ROCRAND_INCLUDE}
    ${ROCM_SMI_INCLUDE})

  list(GET TORCH_INCLUDE_DIRS 0 TORCH_PATH)

else()
  # Else create a regular library
  add_library(fbgemm_gpu_py MODULE
    ${asmjit_sources}
    ${fbgemm_sources}
    ${fbgemm_gpu_sources_static}
    ${fbgemm_gpu_sources_gen})
endif()

# Add PyTorch include/
target_include_directories(fbgemm_gpu_py PRIVATE ${TORCH_INCLUDE_DIRS})

# Remove `lib` from the output artifact name `libfbgemm_gpu_py.so`
set_target_properties(fbgemm_gpu_py PROPERTIES PREFIX "")

# Link to PyTorch
target_link_libraries(fbgemm_gpu_py ${TORCH_LIBRARIES})

# Link to NVML
if(NVML_LIB_PATH)
  target_link_libraries(fbgemm_gpu_py ${NVML_LIB_PATH})
endif()


################################################################################
# FBGEMM_GPU Install
################################################################################

install(TARGETS fbgemm_gpu_py
        DESTINATION fbgemm_gpu)

install(FILES ${gen_python_source_files}
        DESTINATION fbgemm_gpu/split_embedding_codegen_lookup_invokers)

install(FILES ${CMAKE_CODEGEN_DIR}/lookup_args.py
        DESTINATION fbgemm_gpu/split_embedding_codegen_lookup_invokers)

install(FILES ${gen_defused_optim_py_files}
        DESTINATION fbgemm_gpu/split_embedding_optimizer_codegen)

install(FILES ${CMAKE_CODEGEN_DIR}/optimizer_args.py
        DESTINATION fbgemm_gpu/split_embedding_optimizer_codegen)
