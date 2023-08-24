#ifndef NVSHMEM_PUT_TBE_BACKWARD_UNSORTING
#define NVSHMEM_PUT_TBE_BACKWARD_UNSORTING


#include "backward_template_helper.cuh"
#include "fbgemm_gpu/fbgemm_tensor_accessor.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

#include "fbgemm_gpu/cub_namespace_prefix.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include "fbgemm_gpu/cub_namespace_postfix.cuh"

#include <nvshmem.h>
#include <nvshmemx.h>

#define kWarpSize 32
#define kForwardMaxThreads 512
using Tensor = at::Tensor;
using namespace fbgemm_gpu;


// ========================================================================================
// ======================================= Design-2 =======================================
// ========================================================================================

// TODOs: implementing NCCL AlltoAll Function

template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    bool use_lxu_cache,
    typename index_t,
    size_t kMaxVecsPerThread,
    size_t kThreadGroupSize = kWarpSize
    >
__launch_bounds__(512) __global__
void nvshmem_unsorting_backward_kernel(
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
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
    // pta::PackedTensorAccessor64<output_t, 2, at::RestrictPtrTraits> output, // [B][total_D]
    // all-to-all information:
    float* nvshmem_grad,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> dim_sum_per_rank_data,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> dim_offset_per_rank_data,
    int32_t max_D,
    int32_t total_dim_output,
    int32_t nranks,
    int32_t rank,
    int32_t local_dim_output,
    int32_t local_batch_size,
    float learning_rate
);

// Host function
void nvshmem_unsorting_backward_host_function(
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
    float* nvshmem_grad,
    Tensor dim_sum_per_rank_data,
    Tensor dim_offset_per_rank_data,
    int32_t total_dim_output,
    int32_t nranks,
    int32_t rank,
    float learning_rate
);

#endif //NVSHMEM_PUT_TBE_BACKWARD_UNSORTING
