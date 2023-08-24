#ifndef FBGEMM_DECOUPLING_BACKWARD
#define FBGEMM_DECOUPLING_BACKWARD

// #include "codegen/embedding_forward_template_helpers.cuh"
// #include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "backward_template_helper.cuh"
#include "fbgemm_gpu/fbgemm_tensor_accessor.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

#include "fbgemm_gpu/cub_namespace_prefix.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include "fbgemm_gpu/cub_namespace_postfix.cuh"

#define kWarpSize 32
using Tensor = at::Tensor;
using namespace fbgemm_gpu;


template <
    typename emb_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize = kWarpSize,
    int32_t VEC_WIDTH
>
__global__ __launch_bounds__(kMaxThreads)
void sgd_decoupling_update_kernel(
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const at::PackedTensorAccessor32<emb_t, 1, at::RestrictPtrTraits> grad_dev_weights,
    // grad_dev_indices is equivalent to sorted_linear_indices_run
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> grad_dev_indices,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_lxu_cache_locations,
    const int32_t max_D,
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    float learning_rate
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

template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize >
__global__ __launch_bounds__(kMaxThreads) void
split_embedding_backward_codegen_none_unweighted_kernel_cta_per_row_1(
    const pta::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits> grad_output, // if optimizer != "none"
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_ids,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> num_long_run_ids,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_lxu_cache_locations,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> grad_dev_weights,
    const int32_t max_D, // if not dense and optimizer != "none"
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_id_to_really_long_run_ids,
    pta::PackedTensorAccessor32<at::acc_type<cache_t, true>, 2, at::RestrictPtrTraits> temp_grad_accum,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> grad_accum_counter,
    const int32_t max_segment_length_per_cta,
    const bool use_deterministic_algorithms,
    int64_t total_hash_size = 0,
    int64_t total_unique_indices = 0
);

template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize = kWarpSize>
__global__ __launch_bounds__(kBackwardMaxThreads) void
split_embedding_backward_codegen_none_unweighted_kernel_warp_per_row_1(
    const pta::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits> grad_output,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_lxu_cache_locations,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_num_runs,
    int32_t max_segment_length_per_warp,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> grad_dev_weights,
    const int32_t max_D, // if not dense and optimizer != "none"
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    int64_t total_hash_size = 0,
    int64_t total_unique_indices = 0
);

Tensor split_embedding_backward_codegen_none_unweighted_exact_cuda(

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

    // This is acutally passed via args.split_function_args but explicitly list
    // it here for code readability
    int64_t total_hash_size,
    int64_t total_unique_indices
);

#endif //FBGEMM_DECOUPLING_BACKWARD
