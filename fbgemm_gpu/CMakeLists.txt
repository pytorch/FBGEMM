cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

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

function(BLOCK_PRINT)
  message("================================================================================")
  foreach(ARG IN LISTS ARGN)
     message("${ARG}")
  endforeach()
  message("================================================================================")
  message("")
endfunction()

if(SKBUILD)
  BLOCK_PRINT("The project is built using scikit-build")
endif()

# Build options
option(FBGEMM_CPU_ONLY "Build FBGEMM_GPU without GPU support" OFF)
option(USE_CUDA "Use CUDA" ON)
option(USE_ROCM "Use ROCm" OFF)

if(((EXISTS "/opt/rocm/") OR (EXISTS $ENV{ROCM_PATH}))
   AND NOT (EXISTS "/bin/nvcc"))
  message("AMD GPU detected.")
  set(USE_CUDA OFF)
  set(USE_ROCM ON)
endif()

if(FBGEMM_CPU_ONLY)
  BLOCK_PRINT("Building the CPU-only variant of FBGEMM-GPU")
endif()

BLOCK_PRINT("USE_ROCM: ${USE_ROCM}")

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

find_package(Torch REQUIRED)

set(FBGEMM ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(THIRDPARTY ${FBGEMM}/third_party)

if(DEFINED GLIBCXX_USE_CXX11_ABI)
  if(${GLIBCXX_USE_CXX11_ABI} EQUAL 1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
  endif()
  BLOCK_PRINT(
    "Default C++ compiler flags"
    "(values may be overridden by CMAKE_CXX_STANDARD and CXX_STANDARD):"
    ""
    "${CMAKE_CXX_FLAGS}"
  )
endif()

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

if(USE_ROCM)
  list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake"
       "${THIRDPARTY}/hipify_torch/cmake")
  include(Hip)
  include(Hipify)

  BLOCK_PRINT("HIP found: ${HIP_FOUND}")
endif()

#
# GENERATED CUDA, CPP and Python code
#

set(OPTIMIZERS
    adagrad
    adam
    approx_rowwise_adagrad
    approx_rowwise_adagrad_with_weight_decay
    approx_rowwise_adagrad_with_counter
    approx_sgd
    lamb
    lars_sgd
    partial_rowwise_adam
    partial_rowwise_lamb
    rowwise_adagrad
    rowwise_adagrad_with_weight_decay
    rowwise_adagrad_with_counter
    rowwise_weighted_adagrad
    sgd)

set(gen_gpu_source_files
    "gen_embedding_forward_dense_weighted_codegen_cuda.cu"
    "gen_embedding_forward_dense_unweighted_codegen_cuda.cu"
    "gen_embedding_forward_split_weighted_codegen_cuda.cu"
    "gen_embedding_forward_split_unweighted_codegen_cuda.cu"
    "gen_embedding_backward_dense_indice_weights_codegen_cuda.cu"
    "gen_embedding_backward_split_indice_weights_codegen_cuda.cu"
    "gen_embedding_backward_split_grad.cu")

foreach(wdesc weighted unweighted_nobag unweighted)
  list(APPEND gen_gpu_source_files
      "gen_embedding_forward_quantized_split_nbit_host_${wdesc}_codegen_cuda.cu")

  foreach(etype fp32 fp16 fp8 int8 int4 int2)
    list(APPEND gen_gpu_source_files
       "gen_embedding_forward_quantized_split_nbit_kernel_${wdesc}_${etype}_codegen_cuda.cu")
  endforeach()

  list(APPEND gen_gpu_source_files
    "gen_embedding_backward_dense_split_${wdesc}_cuda.cu")
endforeach()

set(gen_cpu_source_files
    "gen_embedding_forward_quantized_unweighted_codegen_cpu.cpp"
    "gen_embedding_forward_quantized_weighted_codegen_cpu.cpp"
    "gen_embedding_backward_dense_split_cpu.cpp")

set(gen_python_files ${CMAKE_BINARY_DIR}/__init__.py)

foreach(optimizer ${OPTIMIZERS})
  list(APPEND gen_gpu_host_source_files
    "gen_embedding_backward_split_${optimizer}.cpp")

  list(APPEND gen_cpu_source_files
    "gen_embedding_backward_split_${optimizer}_cpu.cpp")

  list(APPEND gen_cpu_source_files
    "gen_embedding_backward_${optimizer}_split_cpu.cpp")

  list(APPEND gen_python_files
    "${CMAKE_BINARY_DIR}/lookup_${optimizer}.py")

  foreach(wdesc weighted unweighted_nobag unweighted)
    list(APPEND gen_gpu_source_files
      "gen_embedding_backward_${optimizer}_split_${wdesc}_cuda.cu")
  endforeach()
endforeach()

set(CMAKE_CODEGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/codegen)

set(codegen_dependencies
    ${CMAKE_CODEGEN_DIR}/embedding_backward_code_generator.py
    ${CMAKE_CODEGEN_DIR}/embedding_backward_dense_host.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_backward_dense_host_cpu.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_backward_split_cpu_approx_template.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_backward_split_cpu_template.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_backward_split_host_cpu_template.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_backward_split_host_template.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_backward_split_indice_weights_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_backward_split_grad_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_backward_split_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_forward_quantized_cpu_template.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_forward_quantized_host.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_forward_quantized_host_cpu.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_forward_quantized_split_nbit_host_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_forward_quantized_split_nbit_kernel_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_forward_split_cpu.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_forward_split_cpu.h
    ${CMAKE_CODEGEN_DIR}/embedding_forward_split_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_forward_template_helpers.cuh
    ${CMAKE_CODEGEN_DIR}/__init__.template
    ${CMAKE_CODEGEN_DIR}/lookup_args.py
    ${CMAKE_CODEGEN_DIR}/split_embedding_codegen_lookup_invoker.template
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

if(USE_ROCM)
message(STATUS "${PYTHON_EXECUTABLE}" "${CMAKE_CODEGEN_DIR}/embedding_backward_code_generator.py" "--opensource --is_rocm")
  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${CMAKE_CODEGEN_DIR}/embedding_backward_code_generator.py"
      "--opensource" "--is_rocm" DEPENDS "${codegen_dependencies}")

  set(header_include_dir
      ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR})

  hipify(CUDA_SOURCE_DIR ${PROJECT_SOURCE_DIR} HEADER_INCLUDE_DIR
         ${header_include_dir})
else()
  add_custom_command(
    OUTPUT ${gen_cpu_source_files} ${gen_gpu_source_files}
           ${gen_gpu_host_source_files} ${gen_python_files}
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${CMAKE_CODEGEN_DIR}/embedding_backward_code_generator.py" "--opensource"
    DEPENDS "${codegen_dependencies}")
endif()

set_source_files_properties(
  ${gen_cpu_source_files} PROPERTIES COMPILE_OPTIONS
                                     "-mavx2;-mf16c;-mfma;-fopenmp")
set_source_files_properties(
  ${gen_cpu_source_files}
  PROPERTIES
    INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_SOURCE_DIR}/../include;${THIRDPARTY}/asmjit/src"
)

set_source_files_properties(
  ${gen_gpu_host_source_files}
  PROPERTIES
    INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_SOURCE_DIR}/../include"
)

set_source_files_properties(
  ${gen_gpu_source_files}
  PROPERTIES INCLUDE_DIRECTORIES
             "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/include")
set_source_files_properties(${gen_gpu_source_files}
                            PROPERTIES COMPILE_OPTIONS "${TORCH_CUDA_OPTIONS}")

if(NOT FBGEMM_CPU_ONLY)
  set(gen_source_files ${gen_gpu_source_files} ${gen_gpu_host_source_files}
                       ${gen_cpu_source_files})
else()
  set(gen_source_files ${gen_cpu_source_files})
endif()

#
# CPP FBGEMM support
#

file(GLOB_RECURSE cpp_asmjit_files
     "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/asmjit/src/asmjit/*/*.cpp")

set(cpp_fbgemm_files_normal
    "../src/EmbeddingSpMDM.cc"
    "../src/EmbeddingSpMDMNBit.cc"
    "../src/QuantUtils.cc"
    "../src/RefImplementations.cc"
    "../src/RowWiseSparseAdagradFused.cc"
    "../src/SparseAdagrad.cc"
    "../src/Utils.cc")

set(cpp_fbgemm_files_avx2 "../src/EmbeddingSpMDMAvx2.cc"
                          "../src/QuantUtilsAvx2.cc")

set_source_files_properties(${cpp_fbgemm_files_avx2}
                            PROPERTIES COMPILE_OPTIONS "-mavx2;-mf16c;-mfma")

set(cpp_fbgemm_files_avx512 "../src/EmbeddingSpMDMAvx512.cc")

set_source_files_properties(
  ${cpp_fbgemm_files_avx512}
  PROPERTIES COMPILE_OPTIONS
             "-mavx2;-mf16c;-mfma;-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl")

if(USE_ROCM)
  set(cpp_fbgemm_files ${cpp_fbgemm_files_normal} ${cpp_fbgemm_files_avx2})
else()
  set(cpp_fbgemm_files ${cpp_fbgemm_files_normal} ${cpp_fbgemm_files_avx2}
                       ${cpp_fbgemm_files_avx512})
endif()

set(cpp_fbgemm_files_include_directories
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${FBGEMM}/include ${THIRDPARTY}/asmjit/src ${THIRDPARTY}/cpuinfo/include)

set_source_files_properties(
  ${cpp_fbgemm_files} PROPERTIES INCLUDE_DIRECTORIES
                                 "${cpp_fbgemm_files_include_directories}")

#
# Actual static SOURCES
#

# Ensure NVML_LIB_PATH is empty if it wasn't set and if the default lib path
# doesn't exist.
if(NOT NVML_LIB_PATH)
  set(DEFAULT_NVML_LIB_PATH
      "${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libnvidia-ml.so")

  if(EXISTS ${DEFAULT_NVML_LIB_PATH})
    message(STATUS "Setting NVML_LIB_PATH: \
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libnvidia-ml.so")
    set(NVML_LIB_PATH "${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libnvidia-ml.so")
  endif()
endif()

set(fbgemm_gpu_sources_cpu
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
    src/sparse_ops_cpu.cpp
    src/embedding_inplace_update_cpu.cpp)

if(NOT FBGEMM_CPU_ONLY)
  list(
    APPEND
    fbgemm_gpu_sources_cpu
    codegen/embedding_forward_quantized_host.cpp
    codegen/embedding_backward_dense_host.cpp
    codegen/embedding_bounds_check_host.cpp
    src/cumem_utils_host.cpp
    src/layout_transform_ops_gpu.cpp
    src/permute_pooled_embedding_ops_gpu.cpp
    src/permute_pooled_embedding_ops_split_gpu.cpp
    src/quantize_ops_gpu.cpp
    src/sparse_ops_gpu.cpp
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
    list(
      APPEND
      fbgemm_gpu_sources_cpu
      src/merge_pooled_embeddings_cpu.cpp
      src/merge_pooled_embeddings_gpu.cpp
      src/topology_utils.cpp)
  else()
    message(STATUS "Skipping merge_pooled_embeddings sources")
  endif()
endif()

set_source_files_properties(
  ${fbgemm_gpu_sources_cpu} PROPERTIES COMPILE_OPTIONS
                                       "-mavx;-mf16c;-mfma;-mavx2;-fopenmp")

if(NOT FBGEMM_CPU_ONLY)
  set(fbgemm_gpu_sources_gpu
      codegen/embedding_bounds_check.cu
      codegen/embedding_forward_quantized_split_lookup.cu
      src/cumem_utils.cu
      src/histogram_binning_calibration_ops.cu
      src/jagged_tensor_ops.cu
      src/layout_transform_ops.cu
      src/permute_pooled_embedding_ops.cu
      src/permute_pooled_embedding_ops_split.cu
      src/quantize_ops.cu
      src/sparse_ops.cu
      src/split_embeddings_cache_cuda.cu
      src/split_embeddings_utils.cu
      src/metric_ops.cu
      src/embedding_inplace_update.cu
      src/input_combine.cu)

  set_source_files_properties(
    ${fbgemm_gpu_sources_gpu} PROPERTIES COMPILE_OPTIONS
                                         "${TORCH_CUDA_OPTIONS}")

  # XXXUPS!!! Replace with real
  set_source_files_properties(
    ${fbgemm_gpu_sources_gpu}
    PROPERTIES INCLUDE_DIRECTORIES "${cpp_fbgemm_files_include_directories}")
endif()

set_source_files_properties(
  ${fbgemm_gpu_sources_cpu}
  PROPERTIES INCLUDE_DIRECTORIES "${cpp_fbgemm_files_include_directories}")

if(NOT FBGEMM_CPU_ONLY)
  set(fbgemm_gpu_sources ${fbgemm_gpu_sources_gpu} ${fbgemm_gpu_sources_cpu})
else()
  set(fbgemm_gpu_sources ${fbgemm_gpu_sources_cpu})
endif()

if(USE_ROCM)
  set(abspath_gen_source_files)
  foreach(filename_gen_source_file ${gen_source_files})
    list(APPEND abspath_gen_source_files
         "${CMAKE_BINARY_DIR}/${filename_gen_source_file}")
  endforeach()
endif()

#
# MODULE
#

if(USE_ROCM)
  get_hipified_list("${fbgemm_gpu_sources}" fbgemm_gpu_sources)
  get_hipified_list("${abspath_gen_source_files}" abspath_gen_source_files)
  get_hipified_list("${cpp_fbgemm_files}" cpp_fbgemm_files)

  set(FBGEMM_ALL_HIP_FILES ${fbgemm_gpu_sources} ${abspath_gen_source_files}
                           ${cpp_fbgemm_files})
  set_source_files_properties(${FBGEMM_ALL_HIP_FILES}
                              PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  hip_include_directories("${cpp_fbgemm_files_include_directories}")

  hip_add_library(
    fbgemm_gpu_py
    SHARED
    ${cpp_asmjit_files}
    ${FBGEMM_ALL_HIP_FILES}
    ${FBGEMM_HIP_HCC_LIBRARIES}
    HIPCC_OPTIONS
    ${HIP_HCC_FLAGS})
  target_include_directories(
    fbgemm_gpu_py PUBLIC ${FBGEMM_HIP_INCLUDE} ${ROCRAND_INCLUDE}
                         ${ROCM_SMI_INCLUDE})
  list(GET TORCH_INCLUDE_DIRS 0 TORCH_PATH)
else()
  add_library(fbgemm_gpu_py MODULE ${fbgemm_gpu_sources} ${gen_source_files}
                                   ${cpp_asmjit_files} ${cpp_fbgemm_files})
endif()

set_target_properties(fbgemm_gpu_py PROPERTIES PREFIX "")

target_link_libraries(fbgemm_gpu_py ${TORCH_LIBRARIES})
if(NVML_LIB_PATH)
  target_link_libraries(fbgemm_gpu_py ${NVML_LIB_PATH})
endif()
target_include_directories(fbgemm_gpu_py PRIVATE ${TORCH_INCLUDE_DIRS})

install(TARGETS fbgemm_gpu_py DESTINATION fbgemm_gpu)

# Python

install(FILES ${gen_python_files}
        DESTINATION fbgemm_gpu/split_embedding_codegen_lookup_invokers)
install(FILES ${CMAKE_CODEGEN_DIR}/lookup_args.py
        DESTINATION fbgemm_gpu/split_embedding_codegen_lookup_invokers)
