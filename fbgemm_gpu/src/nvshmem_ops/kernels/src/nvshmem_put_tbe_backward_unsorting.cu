#include "fbgemm_put_tbe_backward.cuh"
#include "nvshmem_put_tbe_backward_unsorting.cuh"


template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    bool use_lxu_cache,
    typename index_t,
    size_t kMaxVecsPerThread,
    size_t kThreadGroupSize
    >
__launch_bounds__(512) __global__
void nvshmem_unsorting_backward_kernel_baseline(
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
){
// shfl_sync_mask is implicitly used by SHFL_SYNC
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
    const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
    const unsigned int shfl_sync_mask = 0xffffffffu;
#endif

    // Shared memory test
    int32_t warp_idx_in_block = threadIdx.y;
    extern __shared__ float shared_output_buffer[]; // nWarp in block * max_D
    int32_t shared_d_offset = warp_idx_in_block * max_D;

    constexpr int VEC_WIDTH = 4;

    // Determine the linearized warp ID, and exit early if needed
    int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    if (b_t >= offsets.size(0) - 1) {
        return;
    }

    // Determine the Table and Training Example IDs
    int32_t t;  // Table ID, which table
    int32_t b;  // Training Example ID, row offset in this table (max val of b == local_batch_size * nDev)
    fd_B.DivMod(b_t, &t, &b); // t = b_t / (local_batch_size * nDev); b = b_t % (local_batch_size * nDev)

    // Get total number of tables
    int64_t weights_offset = weights_offsets[t];
    int32_t T = weights_offsets.size(0);

    // interleave thread block
    // t = b_t % T;
    // b = b_t / T;

    // Determine the number of indices (pooling factor) to look up within the bag
    index_t indices_start = offsets[b_t];
    index_t indices_end = offsets[b_t + 1];
    int32_t L = indices_end - indices_start;

    // Get the offsets of the embedding dimensions of the tables and determine D
    int32_t D_start = D_offsets[t];
    int32_t D_end = D_offsets[t + 1];
    int32_t D = D_end - D_start;

    // D is computed in the bag case or provided as function arg in the nobag case
    // (nobag only supports the case where the embedding dimensions are the same for all tables)
    int32_t D_emb = D;
    if (std::is_same<emb_t, uint8_t>::value) {
        D_emb += kINT8QparamsBytes;
    }
    // Determine if we're doing mean pooling
    const bool mean_pooling = static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN;

    // Compute 1/L - this is used to compute the mean later on
    const float inv_L = (mean_pooling && L != 0) ? static_cast<float>(1.0) / L: static_cast<float>(1.0);

    // Set up the accumulator buffer
    // Each thread works on (D / warpSize) columns; 4 consecutive columns at a time
    Vec4T<cache_t> grad_vals[kMaxVecsPerThread];

    // =============================== Get Gradients from the remote GPU ====================================
    int32_t _local_batch_size = fd_B.D() / nranks;
    int32_t target_gpu = b / _local_batch_size; // target gpu _id
    int32_t b_local_offset = b % _local_batch_size; // row offset in the nvshmem output buffer
    int32_t grad_offset = b_local_offset * total_dim_output + dim_offset_per_rank_data[rank] + D_start;

    nvshmemx_float_get_warp(shared_output_buffer + shared_d_offset, nvshmem_grad + grad_offset, D, target_gpu); // copy from shared memory

    // Load gradients from shared memory
    #pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
        ++i) {
        // Figure out the position in the embedding table row to load
        int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;

        grad_vals[i].load(shared_output_buffer + shared_d_offset + d);
        grad_vals[i].mul_(-1.0 * learning_rate);
    }

    // Iterate over each kThreadGroupSize-sized subset of L indices in the bag
    for (int32_t l_start = 0; l_start < L; l_start += kThreadGroupSize) {
        // Determine the L index that this thread will load data from in cooperative load
        int32_t l = l_start + threadIdx.x;
        // Cooperatively load the indices
        int64_t idx = l < L ? indices[indices_start + l] : 0;
        // Cooperatively load the cache's indices
        // int32_t cache_idx = (use_lxu_cache && placement == PlacementType::MANAGED_CACHING && l < L) ? lxu_cache_locations[indices_start + l] : 0;

        // Iterate over kThreadGroupSize indices ()
        for (auto j = 0; j < kThreadGroupSize && l_start + j < L; ++j) {
            // Load index from thread j in the group
            int64_t idx_j = SHFL_SYNC(idx, j);
            // Load cache's index from thread j in the group
            // int32_t cache_idx_j = use_lxu_cache ? SHFL_SYNC(cache_idx, j) : 0;

            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
                // Figure out the position in the embedding table row to load
                int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;


                // atomic update embedding table parameters:
                gpuAtomicAdd(&dev_weights[weights_offset + idx_j * D_emb + d], grad_vals[i].acc.x);
                gpuAtomicAdd(&dev_weights[weights_offset + idx_j * D_emb + d + 1], grad_vals[i].acc.y);
                gpuAtomicAdd(&dev_weights[weights_offset + idx_j * D_emb + d + 2], grad_vals[i].acc.z);
                gpuAtomicAdd(&dev_weights[weights_offset + idx_j * D_emb + d + 3], grad_vals[i].acc.w);
            }
        }
    }
}


template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    bool use_lxu_cache,
    typename index_t,
    size_t kMaxVecsPerThread,
    size_t kThreadGroupSize
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
){
// shfl_sync_mask is implicitly used by SHFL_SYNC
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
    const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
    const unsigned int shfl_sync_mask = 0xffffffffu;
#endif

    // Shared memory test
    int32_t warp_idx_in_block = threadIdx.y;
    extern __shared__ float shared_output_buffer[]; // nWarp in block * max_D
    int32_t shared_d_offset = warp_idx_in_block * max_D;

    constexpr int VEC_WIDTH = 4;

    // Determine the linearized warp ID, and exit early if needed
    int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    if (b_t >= offsets.size(0) - 1) {
        return;
    }

    // Determine the Table and Training Example IDs
    int32_t t;  // Table ID, which table
    int32_t b;  // Training Example ID, row offset in this table (max val of b == local_batch_size * nDev)
    fd_B.DivMod(b_t, &t, &b); // t = b_t / (local_batch_size * nDev); b = b_t % (local_batch_size * nDev)

    // Get total number of tables
    int64_t weights_offset = weights_offsets[t];
    int32_t T = weights_offsets.size(0);

    // interleave thread block
    // t = b_t % T;
    // b = b_t / T;

    // Determine the number of indices (pooling factor) to look up within the bag
    index_t indices_start = offsets[b_t];
    index_t indices_end = offsets[b_t + 1];
    int32_t L = indices_end - indices_start;

    // Get the offsets of the embedding dimensions of the tables and determine D
    int32_t D_start = D_offsets[t];
    int32_t D_end = D_offsets[t + 1];
    int32_t D = D_end - D_start;

    // D is computed in the bag case or provided as function arg in the nobag case
    // (nobag only supports the case where the embedding dimensions are the same for all tables)
    int32_t D_emb = D;
    if (std::is_same<emb_t, uint8_t>::value) {
        D_emb += kINT8QparamsBytes;
    }
    // Determine if we're doing mean pooling
    const bool mean_pooling = static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN;

    // Compute 1/L - this is used to compute the mean later on
    const float inv_L = (mean_pooling && L != 0) ? static_cast<float>(1.0) / L: static_cast<float>(1.0);

    // Set up the accumulator buffer
    // Each thread works on (D / warpSize) columns; 4 consecutive columns at a time
    Vec4T<cache_t> grad_vals[kMaxVecsPerThread];

    // =============================== Get Gradients from the remote GPU ====================================
    int32_t _local_batch_size = fd_B.D() / nranks;
    int32_t target_gpu = b / _local_batch_size; // target gpu _id
    int32_t b_local_offset = b % _local_batch_size; // row offset in the nvshmem output buffer
    int32_t grad_offset = b_local_offset * total_dim_output + dim_offset_per_rank_data[rank] + D_start;

    nvshmemx_float_get_warp(shared_output_buffer + shared_d_offset, nvshmem_grad + grad_offset, D, target_gpu); // copy from shared memory
    for(int32_t i = threadIdx.x; i < D_emb; i+=kThreadGroupSize){
        shared_output_buffer[shared_d_offset + i] = shared_output_buffer[shared_d_offset + i] * learning_rate * (-1);
    }

    // Iterate over each kThreadGroupSize-sized subset of L indices in the bag
    for (int32_t l_start = 0; l_start < L; l_start += kThreadGroupSize) {
        // Determine the L index that this thread will load data from in cooperative load
        int32_t l = l_start + threadIdx.x;
        // Cooperatively load the indices
        int64_t idx = l < L ? indices[indices_start + l] : 0;
        // Cooperatively load the cache's indices
        // int32_t cache_idx = (use_lxu_cache && placement == PlacementType::MANAGED_CACHING && l < L) ? lxu_cache_locations[indices_start + l] : 0;

        // Iterate over kThreadGroupSize indices ()
        for (auto j = 0; j < kThreadGroupSize && l_start + j < L; ++j) {
            // Load index from thread j in the group
            int64_t idx_j = SHFL_SYNC(idx, j);
            // Load cache's index from thread j in the group
            int64_t weight_offset_emb =  weights_offset + idx_j * D_emb;

            for(int32_t i = threadIdx.x; i < D_emb; i+=kThreadGroupSize){
                gpuAtomicAdd(&dev_weights[weight_offset_emb + i], shared_output_buffer[shared_d_offset + i]);
                // dev_weights[weight_offset_emb + i] = shared_output_buffer[shared_d_offset + i];
            }
        }
    }
}


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
){
    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        weights_offsets,
        D_offsets,
        indices,
        offsets,
        lxu_cache_locations,
        dev_weights
    );

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(dev_weights.get_device());
    int32_t T = D_offsets.numel() - 1; // n_local_Table
    TORCH_CHECK_GT(T, 0);
    // offsets = [B x T  + 1]
    const auto total_B = offsets.size(0) - 1; // global_batch_size = local_batch_size * nDev * n_local_Table
    const int32_t B = (total_B) / T; // local_batch_size_after_dist = local_batch_size * nDev
    int32_t local_batch_size = B / nranks;
    TORCH_CHECK_GE(B, 0);
    TORCH_CHECK_GT(total_D, 0);
    TORCH_CHECK_EQ(total_D % 4, 0);
    TORCH_CHECK_LE(max_D, 1024);

    Tensor output;
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                o_dtype == SparseType::BF16 || o_dtype == SparseType::INT8);
    int64_t total_adjusted_D = total_D;
    if (o_dtype == SparseType::INT8) {
        total_adjusted_D += T * kINT8QparamsBytes;
    }

    if (B == 0) {
        return;
    }

    // =================================================================================================================================
    DISPATCH_EMB_CACHE_OUTPUT_TYPES(
        dev_weights.scalar_type(),
        lxu_cache_weights.scalar_type(),
        dev_weights.scalar_type(),
        "nvshmem_unsorting_backward_kernel", [&] {
        // Check if LXU cache is used
        bool use_lxu_cache = lxu_cache_weights.numel() > 0;
        if (is_experimental) {
          if (std::is_same<emb_t, uint8_t>() || std::is_same<output_t, uint8_t>()) {
            is_experimental = false;
          }
        }

    if (!is_experimental) { // if has_experimental
        // The dense case does not have cache so we have to generate code for
        // only one case (value of use_cache/vbe does not matter)
        if (use_lxu_cache == false) {
            // kMaxElemPerThread is # of elements handled by thread if we use a full warp for a row
            // We consider kMaxElemPerThread 1 and 2, and then a multiple of 4.
            if (max_D <= 128) {
                // hipcc can't use max in constexpr
                constexpr int kMaxVecsPerThread = 4 / 4 >= 1 ? 4 / 4 : 1;
                // If max_D is small, use fewer number of threads than kWarpSize.

#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
                constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 4, 1);
#else
                constexpr int kThreadGroupSize = kWarpSize;
#endif

#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name = "nvshmem_unsorting_backward_kernel";
#endif
                // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
                nvshmem_unsorting_backward_kernel<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    max_D * (512/kThreadGroupSize) * sizeof(float),
                    at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
                    FixedDivisor(B),
                    MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),
                    pooling_mode,
                    MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32), // if not dense

                    nvshmem_grad,
                    MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
                    max_D,
                    total_dim_output,
                    nranks,
                    rank,
                    total_D,
                    local_batch_size,
                    learning_rate
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return;
            }

            if (max_D <= 256) {
                // hipcc can't use max in constexpr
                constexpr int kMaxVecsPerThread = 8 / 4 >= 1 ? 8 / 4 : 1;
                // If max_D is small, use fewer number of threads than kWarpSize.

#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
                constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 8, 1);
#else
                constexpr int kThreadGroupSize = kWarpSize;
#endif

#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name = "nvshmem_unsorting_backward_kernel";
#endif
                // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
                nvshmem_unsorting_backward_kernel<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    max_D * (512/kThreadGroupSize) * sizeof(float),
                    at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
                    FixedDivisor(B),
                    MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),
                    pooling_mode,
                    MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32), // if not dense

                    nvshmem_grad,
                    MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
                    max_D,
                    total_dim_output,
                    nranks,
                    rank,
                    total_D,
                    local_batch_size,
                    learning_rate
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return;
            }
            if (max_D <= 384) {
                // hipcc can't use max in constexpr
                constexpr int kMaxVecsPerThread = 12 / 4 >= 1 ? 12 / 4 : 1;
                // If max_D is small, use fewer number of threads than kWarpSize.

#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
                constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 12, 1);
#else
                constexpr int kThreadGroupSize = kWarpSize;
#endif

#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name = "nvshmem_unsorting_backward_kernel";
#endif
                // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
                nvshmem_unsorting_backward_kernel<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    max_D * (512/kThreadGroupSize) * sizeof(float),
                    at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
                    FixedDivisor(B),
                    MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),
                    pooling_mode,
                    MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32), // if not dense

                    nvshmem_grad,
                    MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
                    max_D,
                    total_dim_output,
                    nranks,
                    rank,
                    total_D,
                    local_batch_size,
                    learning_rate
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return;
            }
            if (max_D <= 512) {
                // hipcc can't use max in constexpr
                constexpr int kMaxVecsPerThread = 16 / 4 >= 1 ? 16 / 4 : 1;
                // If max_D is small, use fewer number of threads than kWarpSize.

#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
                constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 16, 1);
#else
                constexpr int kThreadGroupSize = kWarpSize;
#endif

#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name = "nvshmem_unsorting_backward_kernel";
#endif
                // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
                nvshmem_unsorting_backward_kernel<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    max_D * (512/kThreadGroupSize) * sizeof(float),
                    at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
                    FixedDivisor(B),
                    MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),
                    pooling_mode,
                    MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32), // if not dense

                    nvshmem_grad,
                    MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
                    max_D,
                    total_dim_output,
                    nranks,
                    rank,
                    total_D,
                    local_batch_size,
                    learning_rate
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return;
            }
            if (max_D <= 640) {
                // hipcc can't use max in constexpr
                constexpr int kMaxVecsPerThread = 20 / 4 >= 1 ? 20 / 4 : 1;
                // If max_D is small, use fewer number of threads than kWarpSize.

#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
                constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 20, 1);
#else
                constexpr int kThreadGroupSize = kWarpSize;
#endif

#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name = "nvshmem_unsorting_backward_kernel";
#endif
                // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
                nvshmem_unsorting_backward_kernel<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    max_D * (512/kThreadGroupSize) * sizeof(float),
                    at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
                    FixedDivisor(B),
                    MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),
                    pooling_mode,
                    MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32), // if not dense

                    nvshmem_grad,
                    MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
                    max_D,
                    total_dim_output,
                    nranks,
                    rank,
                    total_D,
                    local_batch_size,
                    learning_rate
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return;
            }
            if (max_D <= 768) {
                // hipcc can't use max in constexpr
                constexpr int kMaxVecsPerThread = 24 / 4 >= 1 ? 24 / 4 : 1;
                // If max_D is small, use fewer number of threads than kWarpSize.

#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
                constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 24, 1);
#else
                constexpr int kThreadGroupSize = kWarpSize;
#endif

#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name = "nvshmem_unsorting_backward_kernel";
#endif
                // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
                nvshmem_unsorting_backward_kernel<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    max_D * (512/kThreadGroupSize) * sizeof(float),
                    at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
                    FixedDivisor(B),
                    MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),
                    pooling_mode,
                    MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32), // if not dense

                    nvshmem_grad,
                    MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
                    max_D,
                    total_dim_output,
                    nranks,
                    rank,
                    total_D,
                    local_batch_size,
                    learning_rate
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return;
            }
            if (max_D <= 896) {
                // hipcc can't use max in constexpr
                constexpr int kMaxVecsPerThread = 28 / 4 >= 1 ? 28 / 4 : 1;
                // If max_D is small, use fewer number of threads than kWarpSize.

#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
                constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 28, 1);
#else
                constexpr int kThreadGroupSize = kWarpSize;
#endif

#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name = "nvshmem_unsorting_backward_kernel";
#endif
                // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
                nvshmem_unsorting_backward_kernel<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    max_D * (512/kThreadGroupSize) * sizeof(float),
                    at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
                    FixedDivisor(B),
                    MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),
                    pooling_mode,
                    MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32), // if not dense

                    nvshmem_grad,
                    MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
                    max_D,
                    total_dim_output,
                    nranks,
                    rank,
                    total_D,
                    local_batch_size,
                    learning_rate
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return;
            }
            if (max_D <= 1024) {
                // hipcc can't use max in constexpr
                constexpr int kMaxVecsPerThread = 32 / 4 >= 1 ? 32 / 4 : 1;
                // If max_D is small, use fewer number of threads than kWarpSize.

#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
                constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 32, 1);
#else
                constexpr int kThreadGroupSize = kWarpSize;
#endif

#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name = "nvshmem_unsorting_backward_kernel";
#endif
                // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
                nvshmem_unsorting_backward_kernel<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    max_D * (512/kThreadGroupSize) * sizeof(float),
                    at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
                    FixedDivisor(B),
                    MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),
                    pooling_mode,
                    MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32), // if not dense

                    nvshmem_grad,
                    MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
                    max_D,
                    total_dim_output,
                    nranks,
                    rank,
                    total_D,
                    local_batch_size,
                    learning_rate
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return;
            }

        }
        } // if (!is_experimental)
    }
);

  cudaDeviceSynchronize();
  return;
}
