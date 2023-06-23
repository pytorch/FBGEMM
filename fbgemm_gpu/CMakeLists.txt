# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# CMake Prelude
################################################################################

cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

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

if(((EXISTS "/opt/rocm/") OR (EXISTS $ENV{ROCM_PATH}))
   AND NOT (EXISTS "/bin/nvcc"))
  message("AMD GPU detected; setting to ROCm build")
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

# Set the default C++ standard to C++17
# Individual targets can have this value overridden; see
# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the default C standard to C17
# Individual targets can have this value overridden; see
# https://cmake.org/cmake/help/latest/prop_tgt/C_STANDARD.html
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
  "Default C++ compiler flags"
  "(values may be overridden by CMAKE_CXX_STANDARD and CXX_STANDARD):"
  ""
  "${CMAKE_CXX_FLAGS}"
)

# Strip all symbols from the .SO file after building
add_link_options($<$<CONFIG:RELEASE>:-s>)


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
# ROCm and HIPify Setup
################################################################################

if(USE_ROCM)
  list(APPEND CMAKE_MODULE_PATH
    "${PROJECT_SOURCE_DIR}/cmake"
    "${THIRDPARTY}/hipify_torch/cmake")
  include(Hip)
  include(Hipify)

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
# FBGEMM_GPU Generated Sources
################################################################################

set(COMMON_OPTIMIZERS
    adagrad
    rowwise_adagrad
    rowwise_adagrad_with_counter
    rowwise_weighted_adagrad
    sgd)

# To be populated in the subsequent diffs
set(CPU_ONLY_OPTIMIZERS "")

set(GPU_ONLY_OPTIMIZERS
  adam
  lamb
  partial_rowwise_adam
  partial_rowwise_lamb
  lars_sgd
  rowwise_adagrad_with_weight_decay
  approx_rowwise_adagrad_with_weight_decay
  none)

set(DEPRECATED_OPTIMIZERS
  approx_sgd
  approx_rowwise_adagrad
  approx_rowwise_adagrad_with_counter)

set(ALL_OPTIMIZERS
  ${COMMON_OPTIMIZERS}
  ${CPU_ONLY_OPTIMIZERS}
  ${GPU_ONLY_OPTIMIZERS}
  ${DEPRECATED_OPTIMIZERS})

set(CPU_OPTIMIZERS ${COMMON_OPTIMIZERS} ${CPU_ONLY_OPTIMIZERS})

set(GPU_OPTIMIZERS ${COMMON_OPTIMIZERS} ${GPU_ONLY_OPTIMIZERS})

# Optimizers with the VBE support
set(VBE_OPTIMIZERS rowwise_adagrad)

# Individual optimizers (not fused with SplitTBE backward)
set(DEFUSED_OPTIMIZERS rowwise_adagrad)

# Individual optimizers (not fused with SplitTBE backward)
set(DEFUSED_OPTIMIZERS rowwise_adagrad)

set(gen_gpu_kernel_source_files
    "gen_embedding_forward_dense_weighted_codegen_cuda.cu"
    "gen_embedding_forward_dense_unweighted_codegen_cuda.cu"
    "gen_embedding_forward_split_weighted_codegen_cuda.cu"
    "gen_embedding_forward_split_unweighted_codegen_cuda.cu"
    "gen_embedding_backward_dense_indice_weights_codegen_cuda.cu"
    "gen_embedding_backward_split_indice_weights_codegen_cuda.cu"
    "gen_embedding_backward_split_grad.cu"
    "gen_embedding_forward_split_weighted_vbe_codegen_cuda.cu"
    "gen_embedding_forward_split_unweighted_vbe_codegen_cuda.cu")

if(NOT USE_ROCM)
  list(APPEND gen_gpu_kernel_source_files
    "gen_embedding_forward_split_weighted_v2_kernel.cu"
    "gen_embedding_forward_split_unweighted_v2_kernel.cu")
endif()

foreach(wdesc dense split)
  list(APPEND gen_gpu_kernel_source_files
    "gen_embedding_forward_${wdesc}_unweighted_nobag_kernel_small.cu")
endforeach()

foreach(wdesc weighted unweighted_nobag unweighted)
  list(APPEND gen_gpu_kernel_source_files
      "gen_embedding_forward_quantized_split_nbit_host_${wdesc}_codegen_cuda.cu"
      "gen_embedding_forward_dense_${wdesc}_kernel.cu"
      "gen_embedding_backward_dense_split_${wdesc}_cuda.cu"
      "gen_embedding_backward_dense_split_${wdesc}_kernel_cta.cu"
      "gen_embedding_backward_dense_split_${wdesc}_kernel_warp.cu")

  list(APPEND gen_gpu_kernel_source_files
      "gen_embedding_forward_split_${wdesc}_kernel.cu")

  foreach(etype fp32 fp16 fp8 int8 int4 int2)
    list(APPEND gen_gpu_kernel_source_files
       "gen_embedding_forward_quantized_split_nbit_kernel_${wdesc}_${etype}_codegen_cuda.cu")
  endforeach()
endforeach()

list(APPEND gen_gpu_kernel_source_files
     "gen_embedding_forward_split_weighted_vbe_kernel.cu"
     "gen_embedding_forward_split_unweighted_vbe_kernel.cu")

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
endforeach()

# Generate the backend API for all optimizers to preserve the backward
# compatibility
foreach(optimizer ${ALL_OPTIMIZERS})
  list(APPEND gen_cpu_source_files
    "gen_embedding_backward_split_${optimizer}_cpu.cpp")
  list(APPEND gen_gpu_host_source_files
    "gen_embedding_backward_split_${optimizer}.cpp")
endforeach()

foreach(optimizer ${CPU_OPTIMIZERS})
  list(APPEND gen_cpu_source_files
    "gen_embedding_backward_${optimizer}_split_cpu.cpp")
endforeach()

foreach(optimizer ${GPU_OPTIMIZERS})
  list(APPEND gen_gpu_kernel_source_files
    "gen_embedding_optimizer_${optimizer}_split_device_kernel.cuh")
  foreach(wdesc weighted unweighted_nobag unweighted)
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

set(CMAKE_CODEGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/codegen)

set(embedding_codegen_dependencies
    ${CMAKE_CODEGEN_DIR}/embedding_backward_code_generator.py
    ${CMAKE_CODEGEN_DIR}/embedding_common_code_generator.py
    ${CMAKE_CODEGEN_DIR}/embedding_backward_dense_host.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_backward_dense_host_cpu.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_backward_split_cpu_approx_template.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_backward_split_cpu_template.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_backward_split_host_cpu_template.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_backward_split_host_template.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_backward_split_indice_weights_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_backward_split_grad_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_backward_split_kernel_cta_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_backward_split_kernel_warp_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_backward_split_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_forward_quantized_cpu_template.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_forward_quantized_host.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_forward_quantized_host_cpu.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_forward_quantized_split_nbit_host_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_forward_quantized_split_nbit_kernel_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_forward_split_cpu.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_forward_split_cpu.h
    ${CMAKE_CODEGEN_DIR}/embedding_forward_split_kernel_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_forward_split_kernel_v2_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_forward_split_kernel_nobag_small_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_forward_split_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_forward_template_helpers.cuh
    ${CMAKE_CODEGEN_DIR}/__init__.template
    ${CMAKE_CODEGEN_DIR}/lookup_args.py
    ${CMAKE_CODEGEN_DIR}/split_embedding_codegen_lookup_invoker.template
    ${CMAKE_CODEGEN_DIR}/embedding_optimizer_split_device_kernel_template.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/cpu_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/cub_namespace_prefix.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/cub_namespace_postfix.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/dispatch_macros.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/embedding_backward_template_helpers.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/embedding_common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/embedding_inplace_update.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/fbgemm_cuda_utils.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/quantize_ops_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/split_embeddings_utils.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/sparse_ops_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/split_embeddings_cache_cuda.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/fbgemm_tensor_accessor.h)

set(optimizer_codegen_dependencies
    ${CMAKE_CODEGEN_DIR}/embedding_optimizer_code_generator.py
    ${CMAKE_CODEGEN_DIR}/embedding_common_code_generator.py
    ${CMAKE_CODEGEN_DIR}/embedding_optimizer_split_host_template.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_optimizer_split_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_optimizer_split_device_kernel_template.cuh
    ${CMAKE_CODEGEN_DIR}/embedding_optimizer_split_kernel_template.cu
    ${CMAKE_CODEGEN_DIR}/optimizer_args.py
    ${CMAKE_CODEGEN_DIR}/split_embedding_optimizer_codegen.template
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/embedding_common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/sparse_ops_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/split_embeddings_utils.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/embedding_backward_template_helpers.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/dispatch_macros.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/fbgemm_cuda_utils.cuh)

if(USE_ROCM)
  message(STATUS "${PYTHON_EXECUTABLE}" "${CMAKE_CODEGEN_DIR}/embedding_backward_code_generator.py" "--opensource --is_rocm")

  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${CMAKE_CODEGEN_DIR}/embedding_backward_code_generator.py"
      "--opensource" "--is_rocm" DEPENDS "${embedding_codegen_dependencies}")

  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${CMAKE_CODEGEN_DIR}/embedding_optimizer_code_generator.py"
      "--opensource" "--is_rocm" DEPENDS "${optimizer_codegen_dependencies}")

  set(header_include_dir
      ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR})

  hipify(CUDA_SOURCE_DIR ${PROJECT_SOURCE_DIR} HEADER_INCLUDE_DIR
         ${header_include_dir})
else()
  add_custom_command(
    OUTPUT
      ${gen_cpu_source_files}
      ${gen_gpu_kernel_source_files}
      ${gen_gpu_host_source_files}
      ${gen_python_source_files}
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${CMAKE_CODEGEN_DIR}/embedding_backward_code_generator.py" "--opensource"
    DEPENDS "${embedding_codegen_dependencies}")

  add_custom_command(
    OUTPUT
      ${gen_defused_optim_source_files}
      ${gen_defused_optim_py_files}
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${CMAKE_CODEGEN_DIR}/embedding_optimizer_code_generator.py"
      "--opensource"
    DEPENDS "${optimizer_codegen_dependencies}")
endif()

if(CXX_AVX2_FOUND)
  set_source_files_properties(${gen_cpu_source_files}
    PROPERTIES COMPILE_OPTIONS
    "-mavx2;-mf16c;-mfma;-fopenmp")
else()
  set_source_files_properties(${gen_cpu_source_files}
    PROPERTIES COMPILE_OPTIONS
    "-fopenmp")
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
    # To force embedding_optimizer_code_generator to generate Python
    # files
    ${gen_defused_optim_py_files}
  )
endif()


################################################################################
# FBGEMM (not FBGEMM_GPU) Sources
################################################################################

set(fbgemm_sources_normal
  "${FBGEMM}/src/EmbeddingSpMDM.cc"
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
    "-mavx2;-mf16c;-mfma")
endif()

if(CXX_AVX512_FOUND)
  set_source_files_properties(${fbgemm_sources_avx512}
    PROPERTIES COMPILE_OPTIONS
    "-mavx2;-mf16c;-mfma;-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl")
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

set(fbgemm_gpu_sources_static_cpu
    codegen/embedding_forward_split_cpu.cpp
    codegen/embedding_forward_quantized_host_cpu.cpp
    codegen/embedding_backward_dense_host_cpu.cpp
    codegen/embedding_bounds_check_host_cpu.cpp
    src/permute_pooled_embedding_ops_split_cpu.cpp
    src/jagged_tensor_ops_autograd.cpp
    src/jagged_tensor_ops_meta.cpp
    src/jagged_tensor_ops_cpu.cpp
    src/input_combine_cpu.cpp
    src/layout_transform_ops_cpu.cpp
    src/quantize_ops_cpu.cpp
    src/sparse_ops/sparse_ops_cpu.cpp
    src/sparse_ops/sparse_ops_meta.cpp
    src/embedding_inplace_update_cpu.cpp)

if(NOT FBGEMM_CPU_ONLY)
  list(APPEND fbgemm_gpu_sources_static_cpu
    codegen/embedding_forward_quantized_host.cpp
    codegen/embedding_backward_dense_host.cpp
    codegen/embedding_bounds_check_host.cpp
    src/cumem_utils_host.cpp
    src/layout_transform_ops_gpu.cpp
    src/permute_pooled_embedding_ops_gpu.cpp
    src/permute_pooled_embedding_ops_split_gpu.cpp
    src/quantize_ops_gpu.cpp
    src/sparse_ops/sparse_ops_gpu.cpp
    src/split_embeddings_utils.cpp
    src/split_table_batched_embeddings.cpp
    src/metric_ops_host.cpp
    src/embedding_inplace_update_gpu.cpp
    src/input_combine_gpu.cpp)

  if(NVML_LIB_PATH)
    message(STATUS "Found NVML_LIB_PATH: ${NVML_LIB_PATH}")
  endif()

  if(NVML_LIB_PATH OR USE_ROCM)
    message(STATUS "Adding merge_pooled_embeddings sources")
    list(APPEND fbgemm_gpu_sources_static_cpu
      src/merge_pooled_embeddings_cpu.cpp
      src/merge_pooled_embeddings_gpu.cpp
      src/topology_utils.cpp)
  else()
    message(STATUS "Skipping merge_pooled_embeddings sources")
  endif()
endif()

if(CXX_AVX2_FOUND)
  set_source_files_properties(${fbgemm_gpu_sources_static_cpu}
    PROPERTIES COMPILE_OPTIONS
    "-mavx;-mf16c;-mfma;-mavx2;-fopenmp")
else()
  set_source_files_properties(${fbgemm_gpu_sources_static_cpu}
    PROPERTIES COMPILE_OPTIONS
    "-fopenmp")
endif()

if(NOT FBGEMM_CPU_ONLY)
  set(fbgemm_gpu_sources_static_gpu
      codegen/embedding_bounds_check.cu
      codegen/embedding_forward_quantized_split_lookup.cu
      src/cumem_utils.cu
      src/embedding_inplace_update.cu
      src/histogram_binning_calibration_ops.cu
      src/input_combine.cu
      src/jagged_tensor_ops/batched_dense_vec_jagged_2d_mul_backward.cu
      src/jagged_tensor_ops/batched_dense_vec_jagged_2d_mul_forward.cu
      src/jagged_tensor_ops/dense_to_jagged_forward.cu
      src/jagged_tensor_ops/jagged_dense_bmm_forward.cu
      src/jagged_tensor_ops/jagged_dense_dense_elementwise_add_jagged_output_forward.cu
      src/jagged_tensor_ops/jagged_dense_elementwise_add_jagged_output.cu
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
      src/jagged_tensor_ops/stacked_jagged_1d_to_dense.cu
      src/jagged_tensor_ops/stacked_jagged_2d_to_dense.cu
      src/layout_transform_ops.cu
      src/metric_ops.cu
      src/permute_pooled_embedding_ops_split.cu
      src/permute_pooled_embedding_ops.cu
      src/quantize_ops.cu
      src/sparse_ops/sparse_ops.cu
      src/split_embeddings_cache_cuda.cu
      src/split_embeddings_utils.cu)

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
target_include_directories(fbgemm_gpu_py PRIVATE
  ${TORCH_INCLUDE_DIRS})

# Remove `lib` from the output artifact name `libfbgemm_gpu_py.so`
set_target_properties(fbgemm_gpu_py
  PROPERTIES PREFIX
  "")

# Link to PyTorch
target_link_libraries(fbgemm_gpu_py
  ${TORCH_LIBRARIES})

# Link to NVML
if(NVML_LIB_PATH)
  target_link_libraries(fbgemm_gpu_py
    ${NVML_LIB_PATH})
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
