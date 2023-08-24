#ifndef NVSHMEM_DECOUPLING_BACKWARD
#define NVSHMEM_DECOUPLING_BACKWARD


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

template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    size_t kThreadGroupSize
    >
__launch_bounds__(512) __global__
void nvshmem_unsorting_backward_kernel_decoupling(
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    // const pta::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits> grad_output,
    const pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    FixedDivisor fd_B,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> indices,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,
    int64_t pooling_mode,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    // all-to-all information:
    float* nvshmem_grad,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> dim_sum_per_rank_data,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> dim_offset_per_rank_data,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> grad_dev_weights,
    pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> inverse,
    int64_t max_D,
    int32_t total_dim_output,
    int32_t nranks,
    int32_t rank,
    int64_t local_dim_output,
    int32_t local_batch_size
);

template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    size_t kThreadGroupSize
    >
__launch_bounds__(512) __global__
void nvshmem_unsorting_backward_kernel_decoupling_signal(
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    // const pta::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits> grad_output,
    const pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    FixedDivisor fd_B,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> indices,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,
    int64_t pooling_mode,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    // all-to-all information:
    float* nvshmem_grad,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> dim_sum_per_rank_data,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> dim_offset_per_rank_data,
    pta::PackedTensorAccessor64<float, 1, at::RestrictPtrTraits> grad_dev_weights,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> grad_dev_signal,
    pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> inverse,
    int64_t max_D,
    int32_t total_dim_output,
    int32_t nranks,
    int32_t rank,
    int64_t local_dim_output,
    int32_t local_batch_size
);

void sgd_decoupling_update_host(
    Tensor& dev_weights,
    Tensor& uvm_weights,
    Tensor& lxu_cache_weights,
    const Tensor& grad_dev_weights,
    const Tensor& grad_dev_indices,
    const Tensor& weights_placements,
    const Tensor& weights_offsets,
    const int64_t max_D,
    const bool stochastic_rounding,
    const int64_t max_hash_size,
    Tensor D_offsets,
    const float learning_rate
);

Tensor nvshmem_unsorting_backward_host_function_decoupling(
    Tensor grad_output,
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t max_D,
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,

    Tensor offsets,
    int64_t pooling_mode,
    Tensor lxu_cache_locations,
    int64_t unused_,

    int64_t max_segment_length_per_warp,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,

    Tensor unique_linear_indices,
    Tensor inverse,

    // This is acutally passed via args.split_function_args but explicitly list
    // it here for code readability
    int64_t total_hash_size,
    int64_t total_unique_indices,

    // all-to-all information:
    float* nvshmem_grad,
    Tensor dim_sum_per_rank_data,
    Tensor dim_offset_per_rank_data,
    int32_t total_dim_output,
    int64_t total_D,
    int32_t nranks,
    int32_t rank
);




#endif //NVSHMEM_DECOUPLING_BACKWARD
