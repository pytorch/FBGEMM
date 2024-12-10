# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


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

# Individual optimizers (not fused with SplitTBE backward)
set(DEFUSED_OPTIMIZERS
    rowwise_adagrad)

# Optimizers with the GWD support
set(GWD_OPTIMIZERS
    rowwise_adagrad)

# Optimizers with the SSD support
set(SSD_OPTIMIZERS
    rowwise_adagrad)

# Optimizers with the VBE support
set(VBE_OPTIMIZERS
    rowwise_adagrad
    rowwise_adagrad_with_counter
    sgd
    dense)

set(WEIGHT_OPTIONS
    weighted
    unweighted_nobag
    unweighted)

set(PARTIAL_WEIGHT_OPTIONS
    weighted,
    unweighted)

set(DENSE_OPTIONS
    split
    dense
    ssd)


################################################################################
# TBE Training - Optimizers
################################################################################

foreach(optimizer ${DEFUSED_OPTIMIZERS})
  list(APPEND gen_defused_optim_src_files
    "gen_embedding_optimizer_${optimizer}_split.cpp"
    "gen_embedding_optimizer_${optimizer}_split_cuda.cu"
    "gen_embedding_optimizer_${optimizer}_split_kernel.cu")
endforeach()


################################################################################
# TBE Training - Forward Split
################################################################################

set(gen_cpu_files_forward_split 
  gen_embedding_forward_split_pt2_cpu_wrapper.cpp)

set(gen_gpu_files_forward_split
  gen_embedding_forward_split_pt2_cuda_wrapper.cpp
  gen_embedding_forward_ssd_pt2_cuda_wrapper.cpp)

foreach(wdesc ${WEIGHT_OPTIONS})
  list(APPEND gen_gpu_files_forward_split
    "gen_embedding_forward_split_${wdesc}_kernel.cu"
    "gen_embedding_forward_dense_${wdesc}_kernel.cu"
    "gen_embedding_forward_ssd_${wdesc}_kernel.cu")
endforeach()

foreach(desc ${DENSE_OPTIONS})
  foreach(wdesc ${PARTIAL_WEIGHT_OPTIONS})
    list(APPEND gen_gpu_files_forward_split
      "gen_embedding_forward_${desc}_${wdesc}_codegen_cuda.cu"
      "gen_embedding_forward_${desc}_${wdesc}_codegen_meta.cpp")
  endforeach()
endforeach()

foreach(wdesc ${PARTIAL_WEIGHT_OPTIONS})
  list(APPEND gen_gpu_files_forward_split
    "gen_embedding_forward_split_${wdesc}_vbe_codegen_cuda.cu"
    "gen_embedding_forward_split_${wdesc}_vbe_codegen_meta.cpp"
    "gen_embedding_forward_split_${wdesc}_vbe_kernel.cu"
    "gen_embedding_forward_split_${wdesc}_v2_kernel.cu"
    "gen_embedding_forward_split_${wdesc}_gwd_codegen_cuda.cu"
    "gen_embedding_forward_split_${wdesc}_vbe_gwd_codegen_cuda.cu"
    "gen_embedding_forward_split_${wdesc}_gwd_kernel.cu"
    "gen_embedding_forward_dense_${wdesc}_vbe_codegen_cuda.cu"
    "gen_embedding_forward_dense_${wdesc}_vbe_kernel.cu"
    "gen_embedding_forward_split_${wdesc}_vbe_gwd_kernel.cu"
    "gen_embedding_forward_ssd_${wdesc}_vbe_codegen_cuda.cu"
    "gen_embedding_forward_ssd_${wdesc}_vbe_codegen_meta.cpp"
    "gen_embedding_forward_ssd_${wdesc}_vbe_kernel.cu")
endforeach()

foreach(desc ${DENSE_OPTIONS})
  list(APPEND gen_gpu_files_forward_split
    "gen_embedding_forward_${desc}_unweighted_nobag_kernel_small.cu")
endforeach()


################################################################################
# TBE Training - Index Select
################################################################################

set(static_index_select_src_files
    ${FBGEMM_GPU}/codegen/training/index_select/batch_index_select_dim0_cpu_host.cpp
    ${FBGEMM_GPU}/codegen/training/index_select/batch_index_select_dim0_host.cpp
    ${FBGEMM_GPU}/codegen/training/index_select/batch_index_select_dim0_ops.cpp)

set(gen_index_select_src_files
    gen_batch_index_select_dim0_forward_codegen_cuda.cu
    gen_batch_index_select_dim0_forward_kernel.cu
    gen_batch_index_select_dim0_forward_kernel_small.cu
    gen_batch_index_select_dim0_backward_codegen_cuda.cu
    gen_batch_index_select_dim0_backward_kernel_cta.cu
    gen_batch_index_select_dim0_backward_kernel_warp.cu
    gen_embedding_backward_split_grad_index_select.cu)


















gpu_cpp_library(
  PREFIX
    fbgemm_gpu_tbe_optimizers
  TYPE
    MODULE
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  CPU_SRCS

  GPU_SRCS
    ${gen_defused_optim_src_files}
  GPU_FLAGS
    ${TORCH_CUDA_OPTIONS}
  DEPS
    asmjit
    fbgemm
    split_embeddings_cache
  DESTINATION
    fbgemm_gpu)
