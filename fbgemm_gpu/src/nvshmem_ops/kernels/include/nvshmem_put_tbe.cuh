#ifndef NVSHMEM_PUT_TBE
#define NVSHMEM_PUT_TBE
#include "codegen/embedding_forward_template_helpers.cuh"
#include <nvshmem.h>
#include <nvshmemx.h>
#include <iostream>

#define kWarpSize 32

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

// ==========================================================================================
// ===================================== NVSHMEM KERNEL =====================================
// ==========================================================================================
template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    bool use_lxu_cache,
    typename index_t,
    size_t kMaxVecsPerThread,
    size_t kThreadGroupSize = kWarpSize
    >
__launch_bounds__(kForwardMaxThreads) __global__
void nvshmem_split_embedding_codegen_forward_unweighted_kernel(
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    const pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    FixedDivisor fd_B,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    int64_t pooling_mode,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    pta::PackedTensorAccessor64<output_t, 2, at::RestrictPtrTraits> output, // [B][total_D]
    // all-to-all information:
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> dim_sum_per_rank_data,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> dim_offset_per_rank_data,
    int32_t max_D,
    int32_t total_dim_output,
    float* nvshmem_output_buffer,
    float* output_ptr,
    int32_t nranks,
    int32_t rank,
    int32_t local_dim_output,
    int32_t local_batch_size,
    int32_t put_type
    );

template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    bool use_lxu_cache,
    typename index_t,
    size_t kMaxVecsPerThread,
    size_t kThreadGroupSize = kWarpSize
    >
__launch_bounds__(kForwardMaxThreads) __global__
void nvshmem_split_embedding_codegen_forward_unweighted_kernel_block(
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    const pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    FixedDivisor fd_B,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    int64_t pooling_mode,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    pta::PackedTensorAccessor64<output_t, 2, at::RestrictPtrTraits> output, // [B][total_D]
    // all-to-all info:
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> dim_sum_per_rank_data,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> dim_offset_per_rank_data,
    int32_t max_D,
    int32_t total_dim_output,
    float* nvshmem_output_buffer,
    float* output_ptr,
    int32_t nranks,
    int32_t rank,
    int32_t local_dim_output,
    int32_t local_batch_size,
    int32_t put_type
    );


__launch_bounds__(kForwardMaxThreads) __global__
void block_layout_transform(
    float* output,
    int32_t* D_offsets,
    int32_t* global_D_offset,
    int32_t* dim_offset_per_rank_data,
    FixedDivisor fd_B,
    float* nvshmem_buffer,
    int32_t local_batch_size,
    int32_t total_dim_output,
    int32_t local_dim_output,
    int32_t kThreadGroupSize,
    int32_t total_T,
    int32_t nranks,
    int32_t rank
);

// Host function
Tensor nvshmem_split_embedding_codegen_forward_unweighted_cuda(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor lxu_cache_locations,
    int64_t output_dtype,
    bool is_experimental,
    // all-to-all information:
    Tensor dim_sum_per_rank_data,
    Tensor dim_offset_per_rank_data,
    int32_t total_dim_output,
    float* nvshmem_output_buffer,
    int32_t nranks,
    int32_t rank,
    int32_t put_type
);

Tensor nvshmem_split_embedding_codegen_forward_unweighted_cuda_block(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor lxu_cache_locations,
    int64_t output_dtype,
    bool is_experimental,
    // all-to-all info:
    Tensor dim_sum_per_rank_data,
    Tensor dim_offset_per_rank_data,
    Tensor global_D_offset,
    int32_t total_dim_output,
    float* nvshmem_output_buffer,
    int32_t nranks,
    int32_t rank,
    int32_t put_type,
    float* d_buffer
);

#endif //NVSHMEM_PUT_TBE
