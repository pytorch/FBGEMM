cmake_minimum_required(VERSION 3.11.0 FATAL_ERROR)

option(FBGEMM_CPU_ONLY "Build fbgemm_gpu without GPU support" OFF)

set(message_line
    "-------------------------------------------------------------")
message("${message_line}")

if(SKBUILD)
  message("The project is built using scikit-build")
endif()
if(FBGEMM_CPU_ONLY)
  message("Building for CPU-only")
endif()

message("${message_line}")

if(FBGEMM_CPU_ONLY)
  project(
    fbgemm_gpu
    VERSION 0.0.1
    LANGUAGES CXX C)
else()
  project(
    fbgemm_gpu
    VERSION 0.0.1
    LANGUAGES CXX C CUDA)
endif()

find_package(Torch REQUIRED)
find_package(PythonExtensions REQUIRED)

set(FBGEMM ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(THIRDPARTY ${FBGEMM}/third_party)

#
# Toch Cuda Extensions are normally compiled with the flags below. However we
# disabled -D__CUDA_NO_HALF_CONVERSIONS__ here as it caused "error: no suitable
# constructor exists to convert from "int" to "__half" errors in
# gen_embedding_forward_quantized_split_[un]weighted_codegen_cuda.cu
#

set(TORCH_CUDA_OPTIONS
    --expt-relaxed-constexpr
    -D__CUDA_NO_HALF_OPERATORS__
    # -D__CUDA_NO_HALF_CONVERSIONS__
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__
    -D__CUDA_NO_HALF2_OPERATORS__)

#
# GENERATED CUDA, CPP and Python code
#

set(OPTIMIZERS
    adagrad
    adam
    approx_rowwise_adagrad
    approx_sgd
    lamb
    lars_sgd
    partial_rowwise_adam
    partial_rowwise_lamb
    rowwise_adagrad
    rowwise_weighted_adagrad
    sgd)

set(gen_gpu_source_files
    "gen_embedding_forward_dense_weighted_codegen_cuda.cu"
    "gen_embedding_forward_dense_unweighted_codegen_cuda.cu"
    "gen_embedding_forward_quantized_split_unweighted_codegen_cuda.cu"
    "gen_embedding_forward_quantized_split_weighted_codegen_cuda.cu"
    "gen_embedding_forward_split_weighted_codegen_cuda.cu"
    "gen_embedding_forward_split_unweighted_codegen_cuda.cu"
    "gen_embedding_backward_split_indice_weights_codegen_cuda.cu"
    "gen_embedding_backward_dense_indice_weights_codegen_cuda.cu"
    "gen_embedding_backward_dense_split_unweighted_cuda.cu"
    "gen_embedding_backward_dense_split_weighted_cuda.cu")

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

  list(APPEND gen_python_files "${CMAKE_BINARY_DIR}/lookup_${optimizer}.py")

  foreach(weight weighted unweighted)
    list(APPEND gen_gpu_source_files
         "gen_embedding_backward_${optimizer}_split_${weight}_cuda.cu")
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
    ${CMAKE_CODEGEN_DIR}/embedding_backward_split_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_forward_quantized_cpu_template.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_forward_quantized_host.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_forward_quantized_host_cpu.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_forward_quantized_split_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_forward_split_cpu.cpp
    ${CMAKE_CODEGEN_DIR}/embedding_forward_split_cpu.h
    ${CMAKE_CODEGEN_DIR}/embedding_forward_split_template.cu
    ${CMAKE_CODEGEN_DIR}/embedding_forward_template_helpers.cuh
    ${CMAKE_CODEGEN_DIR}/__init__.template
    ${CMAKE_CODEGEN_DIR}/lookup_args.py
    ${CMAKE_CODEGEN_DIR}/split_embedding_codegen_lookup_invoker.template
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/cpu_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/cub_namespace_postfix.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/dispatch_macros.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/embedding_backward_template_helpers.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/embedding_common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/fbgemm_cuda_utils.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/split_embeddings_utils.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fbgemm_gpu/sparse_ops_utils.h
)

add_custom_command(
  OUTPUT ${gen_cpu_source_files} ${gen_gpu_source_files}
         ${gen_gpu_host_source_files} ${gen_python_files}
  COMMAND
    "${PYTHON_EXECUTABLE}"
    "${CMAKE_CODEGEN_DIR}/embedding_backward_code_generator.py"
    "--opensource"
  DEPENDS "${codegen_dependencies}")

set_source_files_properties(
  ${gen_cpu_source_files} PROPERTIES COMPILE_OPTIONS
                                     "-mavx2;-mf16c;-mfma;-fopenmp")
set_source_files_properties(
  ${gen_cpu_source_files}
  PROPERTIES
    INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_SOURCE_DIR}/../include"
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
  set(gen_source_files ${gen_gpu_source_files}
      ${gen_gpu_host_source_files} ${gen_cpu_source_files})
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

set(cpp_fbgemm_files ${cpp_fbgemm_files_normal} ${cpp_fbgemm_files_avx2}
                     ${cpp_fbgemm_files_avx512})

set(cpp_fbgemm_files_include_directories
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${FBGEMM}/include ${THIRDPARTY}/asmjit/src ${THIRDPARTY}/cpuinfo/include)

set_source_files_properties(
  ${cpp_fbgemm_files} PROPERTIES INCLUDE_DIRECTORIES
                                 "${cpp_fbgemm_files_include_directories}")

#
# Actual static SOURCES
#

set(fbgemm_gpu_sources_cpu
    codegen/embedding_forward_split_cpu.cpp
    codegen/embedding_forward_quantized_host_cpu.cpp
    codegen/embedding_backward_dense_host_cpu.cpp
    codegen/embedding_bounds_check_host_cpu.cpp
    src/cpu_utils.cpp
    src/input_combine_cpu.cpp
    src/layout_transform_ops_cpu.cpp
    src/permute_pooled_embedding_ops_cpu.cpp
    src/quantize_ops_cpu.cpp
    src/sparse_ops_cpu.cpp)

if(NOT FBGEMM_CPU_ONLY)
  list(APPEND fbgemm_gpu_sources_cpu
    codegen/embedding_forward_quantized_host.cpp
    codegen/embedding_backward_dense_host.cpp
    codegen/embedding_bounds_check_host.cpp
    src/cumem_utils_host.cpp
    src/layout_transform_ops_gpu.cpp
    # src/merge_pooled_embeddings_cpu.cpp src/merge_pooled_embeddings_gpu.cpp
    src/permute_pooled_embedding_ops_gpu.cpp
    src/quantize_ops_gpu.cpp
    src/sparse_ops_gpu.cpp
    src/split_table_batched_embeddings.cpp)
endif()

set_source_files_properties(
  ${fbgemm_gpu_sources_cpu} PROPERTIES COMPILE_OPTIONS
                                       "-mavx;-mf16c;-mfma;-mavx2;-fopenmp")

if(NOT FBGEMM_CPU_ONLY)
  set(fbgemm_gpu_sources_gpu
    codegen/embedding_bounds_check.cu src/cumem_utils.cu
    src/histogram_binning_calibration_ops.cu src/jagged_tensor_ops.cu
    src/layout_transform_ops.cu src/permute_pooled_embedding_ops.cu
    src/quantize_ops.cu src/sparse_ops.cu src/split_embeddings_cache_cuda.cu
    src/split_embeddings_utils.cu)

  set_source_files_properties(${fbgemm_gpu_sources_gpu}
                            PROPERTIES COMPILE_OPTIONS "${TORCH_CUDA_OPTIONS}")

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

#
# MODULE
#

add_library(fbgemm_gpu_py MODULE ${fbgemm_gpu_sources} ${gen_source_files}
                                 ${cpp_asmjit_files} ${cpp_fbgemm_files})

if(NOT FBGEMM_CPU_ONLY)
  target_compile_definitions(fbgemm_gpu_py PRIVATE FBGEMM_CUB_USE_NAMESPACE)
endif()

set_target_properties(fbgemm_gpu_py PROPERTIES PREFIX "")

target_link_libraries(fbgemm_gpu_py ${TORCH_LIBRARIES})
target_include_directories(fbgemm_gpu_py PRIVATE ${TORCH_INCLUDE_DIRS})
set_property(TARGET fbgemm_gpu_py PROPERTY CXX_STANDARD 17)

install(TARGETS fbgemm_gpu_py DESTINATION fbgemm_gpu)

# Python

install(FILES ${gen_python_files}
        DESTINATION fbgemm_gpu/split_embedding_codegen_lookup_invokers)
install(FILES ${CMAKE_CODEGEN_DIR}/lookup_args.py
        DESTINATION fbgemm_gpu/split_embedding_codegen_lookup_invokers)
