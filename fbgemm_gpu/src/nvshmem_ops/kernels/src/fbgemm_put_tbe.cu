#include "fbgemm_put_tbe.cuh"

// ===============================================================================================================
// ======================================== FBGEMM KERNEL ========================================================
// ===============================================================================================================
template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    bool use_lxu_cache,
    typename index_t,
    size_t kMaxVecsPerThread,
    size_t kThreadGroupSize >
__launch_bounds__(kForwardMaxThreads) __global__
void split_embedding_codegen_forward_unweighted_kernel_local(
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
    pta::PackedTensorAccessor64<output_t, 2, at::RestrictPtrTraits> output // [B][total_D]
    ) {

// shfl_sync_mask is implicitly used by SHFL_SYNC
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
    const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
    const unsigned int shfl_sync_mask = 0xffffffffu;
#endif

    // Elements are processed 4 at a time through fbgemm_gpu::Vec4 (CUDA float4, 16 bytes)
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

    // From the Table ID, fetch its weight tensor offset, locate that position
    // in the input weights tensor, and set the weights table pointer
    const emb_t* __restrict__ weights;
    int64_t weights_offset = weights_offsets[t];
    const auto placement = static_cast<PlacementType>(weights_placements[t]);
    if (placement == PlacementType::DEVICE) {
        weights = &dev_weights[weights_offset];
    } else {
        weights = &uvm_weights[weights_offset];
    }

    // Get total number of tables
    int32_t T = weights_offsets.size(0);

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
    Vec4T<cache_t> accumulators[kMaxVecsPerThread];

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

            // Load the embedding table row from memory to the buffer
            auto weight_row_emb = WeightRow<emb_t, cache_t, cache_t>(
                const_cast<emb_t*>(&weights[idx_j * D_emb]),
                nullptr,
                D,
                nullptr);

            // Load the two quantization params (scale and bias) from the end of the embedding table row (2 floats)
            [[maybe_unused]] float2 qparams_emb;
            if (std::is_same<emb_t, uint8_t>::value) {
                qparams_emb = weight_row_emb.load_qparams();
            }
            // Iterate over the row of elements in the weights table, in 4-element strides between adjacent threads
            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
                // Figure out the position in the embedding table row to load
                int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;

                // Fused load-and-dequantize from the buffer
                Vec4T<cache_t> weight = weight_row_emb.load(d, qparams_emb);
                // Accumulate the weight
                accumulators[i].add_(weight);
            }
        }
    }

    // If weight type is FP32/16
    if constexpr (!std::is_same_v<output_t, uint8_t>) {
        output_t* output_ = &output[b][D_start];

        #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
             i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
             ++i) {
            // Compute the mean (for mean pooling) and store directly to memory as is
            accumulators[i].mul_(inv_L);
            int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
            accumulators[i].store(output_ + d);
        }

    }
}



// Host function
Tensor split_embedding_codegen_forward_unweighted_cuda_local(
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
    bool is_experimental
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
    TORCH_CHECK_GE(B, 0);
    TORCH_CHECK_GT(total_D, 0);
    TORCH_CHECK_EQ(total_D % 4, 0);
    TORCH_CHECK_LE(max_D, 1024);
    // std::cout << "total_B:" << total_B << ", B:" << B << ", total_D:" << total_D << ", max_D:" << max_D << std::endl;

    Tensor output;
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                o_dtype == SparseType::BF16 || o_dtype == SparseType::INT8);
    int64_t total_adjusted_D = total_D;
    if (o_dtype == SparseType::INT8) {
        total_adjusted_D += T * kINT8QparamsBytes;
    }
    output = at::empty( // [(local_batch_size * nDev, total_D)]
        {B, total_adjusted_D},
        dev_weights.options().dtype(getScalarType(o_dtype))
    ); // if nobag

    if (B == 0) {
        return output;
    }

    // =================================================================================================================================
    DISPATCH_EMB_CACHE_OUTPUT_TYPES(
        dev_weights.scalar_type(),
        lxu_cache_weights.scalar_type(),
        output.scalar_type(),
        "batched_embedding_forward_kernel_2", [&] {
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
            if (max_D <= 32) {
                // hipcc can't use max in constexpr
                constexpr int kMaxVecsPerThread = 1 / 4 >= 1 ? 1 / 4 : 1;
                // If max_D is small, use fewer number of threads than kWarpSize.

#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
                constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 1, 1);
#else
                constexpr int kThreadGroupSize = kWarpSize;
#endif

#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name = "split_embedding_codegen_forward_unweighted_kernel";
#endif
                // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
                split_embedding_codegen_forward_unweighted_kernel_local<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    0,
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
                    MAKE_PTA_WITH_NAME(func_name, output, output_t, 2, 64)
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return;
            }
            if (max_D <= 64) {
                // hipcc can't use max in constexpr
                constexpr int kMaxVecsPerThread = 2 / 4 >= 1 ? 2 / 4 : 1;
                // If max_D is small, use fewer number of threads than kWarpSize.

#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
                constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 2, 1);
#else
                constexpr int kThreadGroupSize = kWarpSize;
#endif

#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name = "split_embedding_codegen_forward_unweighted_kernel";
#endif
                // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
                split_embedding_codegen_forward_unweighted_kernel_local<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    0,
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
                    MAKE_PTA_WITH_NAME(func_name, output, output_t, 2, 64)
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return;
            }
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
                const auto func_name = "split_embedding_codegen_forward_unweighted_kernel";
#endif
                // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
                split_embedding_codegen_forward_unweighted_kernel_local<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    0,
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
                    MAKE_PTA_WITH_NAME(func_name, output, output_t, 2, 64)
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
                const auto func_name = "split_embedding_codegen_forward_unweighted_kernel";
#endif
                // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
                split_embedding_codegen_forward_unweighted_kernel_local<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    0,
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
                    MAKE_PTA_WITH_NAME(func_name, output, output_t, 2, 64)
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
                const auto func_name = "split_embedding_codegen_forward_unweighted_kernel";
#endif
                // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
                split_embedding_codegen_forward_unweighted_kernel_local<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    0,
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
                    MAKE_PTA_WITH_NAME(func_name, output, output_t, 2, 64)
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
                const auto func_name = "split_embedding_codegen_forward_unweighted_kernel";
#endif
                // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
                split_embedding_codegen_forward_unweighted_kernel_local<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    0,
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
                    MAKE_PTA_WITH_NAME(func_name, output, output_t, 2, 64)
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
                const auto func_name = "split_embedding_codegen_forward_unweighted_kernel";
#endif
                // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
                split_embedding_codegen_forward_unweighted_kernel_local<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    0,
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
                    MAKE_PTA_WITH_NAME(func_name, output, output_t, 2, 64)
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
                const auto func_name = "split_embedding_codegen_forward_unweighted_kernel";
#endif
                // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
                split_embedding_codegen_forward_unweighted_kernel_local<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    0,
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
                    MAKE_PTA_WITH_NAME(func_name, output, output_t, 2, 64)
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
                const auto func_name = "split_embedding_codegen_forward_unweighted_kernel";
#endif
                // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
                split_embedding_codegen_forward_unweighted_kernel_local<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    0,
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
                    MAKE_PTA_WITH_NAME(func_name, output, output_t, 2, 64)
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
                const auto func_name = "split_embedding_codegen_forward_unweighted_kernel";
#endif
                // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
                split_embedding_codegen_forward_unweighted_kernel_local<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    0,
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
                    MAKE_PTA_WITH_NAME(func_name, output, output_t, 2, 64)
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return;
            }
        }
        } // if (!is_experimental)
    }
);

  return output;
}
