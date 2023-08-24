#include "fbgemm_put_tbe_backward.cuh"
#include "nvshmem_put_tbe_backward.cuh"

template <
    typename emb_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize = kWarpSize,
    int32_t VEC_WIDTH
>
DEVICE_INLINE void split_sgd_table_update_kernel(
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits>& dev_weights,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits>& uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>& lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& weights_placements,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>& weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& sorted_lxu_cache_locations,
    Vec4T<at::acc_type<cache_t, true>>* grad_sum,
    const bool stochastic_rounding,
    const at::PhiloxCudaState& stochastic_rounding_philox_args,
    const uint32_t run_id,
    const int32_t D,
    const int32_t t,
    const int64_t idx,
    const int32_t segment_start,
    const uint32_t shfl_sync_mask,
    const int32_t shared_weight_offset,
    float learning_rate = 0
) {
    constexpr auto is_int8 = std::is_same<emb_t, uint8_t>::value;
    const int64_t weights_offset = weights_offsets[t];
    emb_t* __restrict__ weights {nullptr};
    cache_t* __restrict__ cache_weights {nullptr};
    int32_t D_emb = D;
    if (is_int8) {
        D_emb += kINT8QparamsBytes;
    }
    const auto weights_placement = static_cast<PlacementType>(weights_placements[t]);
    if (weights_placement == PlacementType::DEVICE) {
        weights = &dev_weights[weights_offset + idx * D_emb];
    } else {
        weights = &uvm_weights[weights_offset + idx * D_emb];
    }

    struct SharedMemory<Vec4T<at::acc_type<cache_t, true>>> weight_update_buffer;
    Vec4T<at::acc_type<cache_t, true>>* shared_weight_update_row =
        is_int8 ? weight_update_buffer.getPointer() : nullptr;
    auto weight_row_template =
        WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(
            weights, cache_weights, D, nullptr);

    weight_row_template.set_stochastic_rounding(
      stochastic_rounding,
      stochastic_rounding_philox_args,
      threadIdx.x + run_id * blockDim.x
    );

    float2 qparams_template;
    if (is_int8 && !cache_weights) {
        qparams_template = weight_row_template.load_qparams();
    }



    float2 qparams_new;
#pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
        ++i) {
        int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
        Vec4T<at::acc_type<cache_t, true>> weight_new = weight_row_template.load(d, qparams_template);
        auto& grad = grad_sum[i];

      weight_new.fma_(grad, -learning_rate);

        if (is_int8 && !cache_weights) {
            shared_weight_update_row[
                threadIdx.x + (i * kThreadGroupSize) + shared_weight_offset] = weight_new;
        } else {
            // qparams_new not used if type is not int8
            weight_row_template.store(weight_new, d, qparams_new);
        }
    }

    if (is_int8 && !cache_weights) {
        // Calculate new qparams after row update
        qparams_new = thrust_find_qparams<at::acc_type<cache_t, true>>(
            &shared_weight_update_row[shared_weight_offset], D);
        weight_row_template.store_qparams(qparams_new);

        // Fetch cached updated row from shared mem and quantize on-the-fly
        // when saving to lowp embedding
#pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
            i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
            ++i) {
            const int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
            weight_row_template.store(
                shared_weight_update_row[threadIdx.x + (i * kThreadGroupSize) + shared_weight_offset],
                d,
                qparams_new);
        }
    }


}

using namespace fbgemm_gpu;

template <typename index_t, typename info_acc_t, bool nobag, bool vbe>
__global__ __launch_bounds__(kMaxThreads) void linearize_index_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    at::PackedTensorAccessor32<info_acc_t, 1, at::RestrictPtrTraits> infos,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_indices,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    const uint32_t max_T,
    const uint32_t max_B,
    // Use a raw pointer to avoid creating dummy PackedTensorAccessor
    const uint32_t* const __restrict__ vbe_b_t_map,
    FixedDivisor fd) {
  const int32_t T = hash_size_cumsum.size(0) - 1;
  auto b_t = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t b;
  int32_t t;
  const auto total_B = offsets.size(0) - 1;
  bool valid = b_t < total_B;
  // info must be uint32_t (using auto will assign int32_t to info)
  uint32_t info = 0;

  if (vbe && valid) {
    info = vbe_b_t_map[b_t];
    reinterpret_cast<uint32_t*>(&t)[0] = info >> info_B_num_bits;
    reinterpret_cast<uint32_t*>(&b)[0] = info & info_B_mask;
  } else {
    fd.DivMod(b_t, &t, &b);
  }

  const index_t hash_offset = valid ? hash_size_cumsum[t] : -1;
  const index_t indices_start = valid ? offsets[b_t] : -1;
  const int32_t L = valid ? offsets[b_t + 1] - indices_start : 0;
  const int32_t lane_id = threadIdx.x % kWarpSize;

  // Compile-time conditional
  if (nobag) {
    for (int32_t j = 0; j < kWarpSize; ++j) {
      const index_t indices_start_warp =
          fbgemm_gpu::shfl_sync(indices_start, j);
      const int32_t t_warp = fbgemm_gpu::shfl_sync(t, j);
      const int32_t L_warp = fbgemm_gpu::shfl_sync(L, j);
      const index_t hash_offset_warp = fbgemm_gpu::shfl_sync(hash_offset, j);
      for (int32_t i = lane_id; i < L_warp; i += kWarpSize) {
        const index_t idx = __ldg(&indices[indices_start_warp + i]);
        const int64_t l_t = (indices_start_warp + i) * T + t_warp;
        infos[indices_start_warp + i] = l_t;
        linear_indices[indices_start_warp + i] = hash_offset_warp + idx;
      }
    }
  } else {
    // Store t in upper (32 - DEFAULT_INFO_B_NUM_BITS).
    // Store b in lower (DEFAULT_INFO_B_NUM_BITS).
    if (!vbe && valid) {
      info = (reinterpret_cast<uint32_t*>(&t)[0] << info_B_num_bits) |
          reinterpret_cast<uint32_t*>(&b)[0];
    }
    for (int32_t j = 0; j < kWarpSize; ++j) {
      const index_t indices_start_warp =
          fbgemm_gpu::shfl_sync(indices_start, j);
      const uint32_t info_warp = fbgemm_gpu::shfl_sync(info, j);
      const int32_t L_warp = fbgemm_gpu::shfl_sync(L, j);
      const index_t hash_offset_warp = fbgemm_gpu::shfl_sync(hash_offset, j);
      for (int32_t i = lane_id; i < L_warp; i += kWarpSize) {
        const index_t idx = __ldg(&indices[indices_start_warp + i]);
        reinterpret_cast<uint32_t*>(&infos[0])[indices_start_warp + i] =
            info_warp;
        linear_indices[indices_start_warp + i] = hash_offset_warp + idx;
      }
    }
  }
}

template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize >
__global__ __launch_bounds__(kMaxThreads) void
nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1(
    const pta::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits> grad_output,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements, // if optimizer != "none"
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_ids,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> num_long_run_ids,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_lxu_cache_locations,
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args, // if not dense and optimizer != "none"
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_id_to_really_long_run_ids,
    pta::PackedTensorAccessor32<at::acc_type<cache_t, true>, 2, at::RestrictPtrTraits> temp_grad_accum,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> grad_accum_counter,
    const int32_t max_segment_length_per_cta,
    const bool use_deterministic_algorithms,
    // nvshmem parameters
    float* nvshmem_buffer,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> dim_offset_per_rank,
    int32_t local_batch_size,
    int32_t total_dim_output,
    int32_t rank,
    int32_t nranks,
    float learning_rate) {
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
  const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
  const unsigned int shfl_sync_mask = 0xffffffffu;
#endif
  constexpr int VEC_WIDTH = 4;
  int32_t T = weights_offsets.size(0);
  const int32_t num_long_runs = num_long_run_ids[0];

  int32_t max_D = 256;
  __shared__ float grad_get_buffer[256 * 32]; //currently only support embedding_dim <= 256
  int32_t shared_d_offset = threadIdx.y * max_D;

  for (int32_t long_run_id = blockIdx.x; long_run_id < num_long_runs; long_run_id += gridDim.x) {
        // The first thread block in the really long run has run_id in long_run_ids
        // and the rest have the negative of its offset (see find_long_segments kernel).
        int32_t cta_rank_on_current_run = 0;
        int32_t current_run_id = long_run_ids[long_run_id];
        if (current_run_id < 0) {
            cta_rank_on_current_run = -long_run_ids[long_run_id];
            current_run_id = long_run_ids[long_run_id - cta_rank_on_current_run];
        }
        const int32_t run_length =
            sorted_linear_indices_cumulative_run_lengths[current_run_id + 1] -
            sorted_linear_indices_cumulative_run_lengths[current_run_id];
        // This computation must agree with how we compute num_ctas_for_run in
        // find_long_segments kernel!
        const int32_t num_ctas_on_current_run =
            use_deterministic_algorithms ? 1 : div_round_up(run_length, max_segment_length_per_cta);


        const int64_t linear_index = sorted_linear_indices_run[current_run_id];
        const int32_t segment_start =
            sorted_linear_indices_cumulative_run_lengths[current_run_id] +
            cta_rank_on_current_run * max_segment_length_per_cta;
        // const int32_t segment_end = std::min(
        const int32_t segment_end = min(
            use_deterministic_algorithms ? INT_MAX : segment_start + max_segment_length_per_cta,
            sorted_linear_indices_cumulative_run_lengths[current_run_id + 1]);
        const int32_t SL = segment_end - segment_start;
        const int32_t warp_id = threadIdx.y;
        const int32_t lane_id = threadIdx.x;

        // Note that with shared embedding tables we can have multiple tables
        // (i.e. different values of `t` sharing the same segment).
        //
        const auto info_0 = reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start];
        const auto t_0 = info_0 >> info_B_num_bits;

        int64_t hash_size = hash_size_cumsum[t_0];
        int32_t D = D_offsets[t_0 + 1] - D_offsets[t_0];
        int64_t idx = linear_index - hash_size;

        const int32_t SL_per_warp = div_round_up(SL, blockDim.y);
        const int32_t sl_start = SL_per_warp * warp_id;
        const int32_t sl_end = min(SL_per_warp * (warp_id + 1), SL);
        Vec4T<at::acc_type<cache_t, true>> grad_sum[kMaxVecsPerThread];
        for (int32_t sl = sl_start; sl < sl_end; sl += kThreadGroupSize) {
            int32_t sl_j = sl + threadIdx.x;
            const auto b_t = sl_j < sl_end ? reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start + sl_j] : 0;
            const auto b = b_t & info_B_mask;
            const auto t = b_t >> info_B_num_bits; // if vbe
            int32_t D_start = sl_j < sl_end ? D_offsets[t] : 0; // if vbe // if not nobag
            for (int32_t j = 0; j < kThreadGroupSize && sl + j < sl_end; ++j) {
                int32_t b_j = SHFL_SYNC(b, j);
                int32_t D_start_j = SHFL_SYNC(D_start, j);

                // nvshmem_get
                int32_t target_gpu = b_j / local_batch_size; // target gpu _id
                int32_t b_local_offset = b_j % local_batch_size; // row offset in the nvshmem output buffer
                int32_t get_offset = b_local_offset * total_dim_output + dim_offset_per_rank[rank] + D_start_j;
                nvshmemx_float_get_warp(grad_get_buffer + shared_d_offset, nvshmem_buffer + get_offset, D, target_gpu); // copy from shared memory

                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                    i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                    int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                    Vec4T<at::acc_type<grad_t, true>> grad_out_vec(
                        // &grad_output[b_j][0] + D_start_j + d
                        grad_get_buffer + shared_d_offset + d
                    );
                    grad_sum[i].add_(grad_out_vec);
                }
            }
        }
        // do shared memory reduction only if we used multiple warps.
        if (SL > SL_per_warp) {
            struct SharedMemory<Vec4T<at::acc_type<cache_t, true>>> smem;
            Vec4T<at::acc_type<cache_t, true>>* shared_grad_sums = smem.getPointer();

            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
            shared_grad_sums
                [lane_id + i * kThreadGroupSize +
                warp_id * kMaxVecsPerThread * kThreadGroupSize] = grad_sum[i];
            }
            __syncthreads();
            if (blockDim.y >= 32) {
            if (warp_id < 16) {
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kThreadGroupSize +
                    warp_id * kMaxVecsPerThread * kThreadGroupSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                warp_id * kMaxVecsPerThread * kThreadGroupSize],
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                (warp_id + 16) * kMaxVecsPerThread * kThreadGroupSize]);
                }
            }
            __syncthreads();
            }
            if (blockDim.y >= 16) {
            if (warp_id < 8) {
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kThreadGroupSize +
                    warp_id * kMaxVecsPerThread * kThreadGroupSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                warp_id * kMaxVecsPerThread * kThreadGroupSize],
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                (warp_id + 8) * kMaxVecsPerThread * kThreadGroupSize]);
                }
            }
            __syncthreads();
            }
            if (blockDim.y >= 8) {
            if (warp_id < 4) {
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kThreadGroupSize +
                    warp_id * kMaxVecsPerThread * kThreadGroupSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                warp_id * kMaxVecsPerThread * kThreadGroupSize],
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                (warp_id + 4) * kMaxVecsPerThread * kThreadGroupSize]);
                }
            }
            __syncthreads();
            }
            if (blockDim.y >= 4) {
            if (warp_id < 2) {
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kThreadGroupSize +
                    warp_id * kMaxVecsPerThread * kThreadGroupSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                warp_id * kMaxVecsPerThread * kThreadGroupSize],
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                (warp_id + 2) * kMaxVecsPerThread * kThreadGroupSize]);
                }
            }
            __syncthreads();
            }
            if (warp_id == 0) {
            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
                grad_sum[i] = vec4_acc(
                    shared_grad_sums
                        [lane_id + i * kThreadGroupSize +
                        warp_id * kMaxVecsPerThread * kThreadGroupSize],
                    shared_grad_sums
                        [lane_id + i * kThreadGroupSize +
                        (warp_id + 1) * kMaxVecsPerThread * kThreadGroupSize]);
            }
            }
        }

        if (warp_id != 0) {
            continue;
        }

        if (num_ctas_on_current_run > 1) {
            int really_long_run_id = long_run_id_to_really_long_run_ids[long_run_id];
            Vec4T<at::acc_type<cache_t, true>> *temp_grad_accum_ptr =
                reinterpret_cast<Vec4T<at::acc_type<cache_t, true>>*>(&temp_grad_accum[really_long_run_id][0]);
            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
                gpuAtomicAdd(&temp_grad_accum_ptr[lane_id + i * kThreadGroupSize].acc.x, grad_sum[i].acc.x);
                gpuAtomicAdd(&temp_grad_accum_ptr[lane_id + i * kThreadGroupSize].acc.y, grad_sum[i].acc.y);
                gpuAtomicAdd(&temp_grad_accum_ptr[lane_id + i * kThreadGroupSize].acc.z, grad_sum[i].acc.z);
                gpuAtomicAdd(&temp_grad_accum_ptr[lane_id + i * kThreadGroupSize].acc.w, grad_sum[i].acc.w);
            }
            int counter;
            if (threadIdx.x == 0) {
                __threadfence();
                counter = gpuAtomicAdd(&grad_accum_counter[really_long_run_id], -1);
            }
            counter = SHFL_SYNC(counter, 0);
            // Only the thread block accumulated the gradient last does the weight update.
            if (counter > 1) {
                continue;
            }
            CUDA_KERNEL_ASSERT(counter == 1 && "Invalid grad_accum_counter. Race condition?");
            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
                grad_sum[i] = temp_grad_accum_ptr[lane_id + i * kThreadGroupSize];
            }
        }
        split_sgd_table_update_kernel
          <emb_t, cache_t, kMaxVecsPerThread, kThreadGroupSize, VEC_WIDTH>(
              dev_weights,
              uvm_weights,
              lxu_cache_weights,
              weights_placements,
              weights_offsets,
              sorted_lxu_cache_locations,
              grad_sum,
              stochastic_rounding,
              stochastic_rounding_philox_args,
              current_run_id,
              D,
              t_0,
              idx,
              segment_start,
              shfl_sync_mask,
              0, // shared_weight_offset
              learning_rate);
    } // for each run
}


template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize >
__global__ __launch_bounds__(kBackwardMaxThreads) void
nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1(
    const pta::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits> grad_output,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_lxu_cache_locations,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_num_runs,
    int32_t max_segment_length_per_warp,
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args, // if not dense and optimizer != "none"
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    // nvshmem parameters
    float* nvshmem_buffer,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> dim_offset_per_rank,
    int32_t local_batch_size,
    int32_t total_dim_output,
    int32_t rank,
    int32_t nranks,
    float learning_rate
    ) {
    int32_t T = D_offsets.size(0) - 1;
    const int32_t start_run_id = blockIdx.x * blockDim.y + threadIdx.y;

#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
    const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
    const unsigned int shfl_sync_mask = 0xffffffffu;
#endif
    constexpr int VEC_WIDTH = 4;

    int32_t max_D = 512;
    __shared__ float grad_get_buffer[512 * 16]; //currently only support embedding_dim <= 512
    int32_t shared_d_offset = threadIdx.y * max_D;

    for (uint32_t run_id = start_run_id;
         run_id < sorted_linear_indices_run.size(0) && run_id < sorted_linear_indices_num_runs[0];
             run_id += gridDim.x * blockDim.y) {

        const int64_t linear_index = sorted_linear_indices_run[run_id];
        const int32_t segment_start =
            sorted_linear_indices_cumulative_run_lengths[run_id];
        const int32_t segment_end =
            sorted_linear_indices_cumulative_run_lengths[run_id + 1];
        const int32_t SL = segment_end - segment_start;


        if (SL >= max_segment_length_per_warp) {
            continue;
        }

        // now, each segment corresponds to exactly one table `t` and row in
        // that table (`idx`). Thus, we can hoist out some of the book-keeping.
        const auto info_0 = reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start];
        const auto t_0 = info_0 >> info_B_num_bits;

        int64_t hash_size = hash_size_cumsum[t_0];
        int32_t D = D_offsets[t_0 + 1] - D_offsets[t_0];
        int64_t idx = linear_index - hash_size;

        const int32_t SL_per_warp = div_round_up(SL, blockDim.y);
        const int32_t sl_start = 0;
        const int32_t sl_end = SL;
        Vec4T<at::acc_type<cache_t, true>> grad_sum[kMaxVecsPerThread];
        for (int32_t sl = sl_start; sl < sl_end; sl += kThreadGroupSize) {
            int32_t sl_j = sl + threadIdx.x;
            const auto b_t = sl_j < sl_end ? reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start + sl_j] : 0;
            const auto b = b_t & info_B_mask;
            const auto t = b_t >> info_B_num_bits; // if vbe

            int32_t D_start = sl_j < sl_end ? D_offsets[t] : 0; // if vbe // if not nobag

            for (int32_t j = 0; j < kThreadGroupSize && sl + j < sl_end; ++j) {
                int32_t b_j = SHFL_SYNC(b, j);
                int32_t D_start_j = SHFL_SYNC(D_start, j);

                // nvshmem_get
                int32_t target_gpu = b_j / local_batch_size; // target gpu _id
                int32_t b_local_offset = b_j % local_batch_size; // row offset in the nvshmem output buffer
                int32_t get_offset = b_local_offset * total_dim_output + dim_offset_per_rank[rank] + D_start_j;
                nvshmemx_float_get_warp(grad_get_buffer + shared_d_offset, nvshmem_buffer + get_offset, D, target_gpu); // copy from shared memory

                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                        i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                        ++i) {
                    int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                    Vec4T<at::acc_type<grad_t, true>> grad_out_vec(
                        // &grad_output[b_j][0] + D_start_j + d
                        grad_get_buffer + shared_d_offset + d
                    );
                    grad_sum[i].add_(grad_out_vec);
                }
            }
        }

        split_sgd_table_update_kernel
          <emb_t, cache_t, kMaxVecsPerThread, kThreadGroupSize, VEC_WIDTH>(
              dev_weights,
              uvm_weights,
              lxu_cache_weights,
              weights_placements,
              weights_offsets,
              sorted_lxu_cache_locations,
              grad_sum,
              stochastic_rounding,
              stochastic_rounding_philox_args,
              run_id,
              D,
              t_0,
              idx,
              segment_start,
              shfl_sync_mask,
              threadIdx.y * kMaxVecsPerThread * kThreadGroupSize, // shared_weight_offset
              learning_rate); // if not dense and optimizer != "none"

    }
}



template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize >
__global__ __launch_bounds__(kBackwardMaxThreads) void
nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1_cache(
    pta::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits> grad_output,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_lxu_cache_locations,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_num_runs,
    int32_t max_segment_length_per_warp,
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args, // if not dense and optimizer != "none"
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    // nvshmem parameters
    float* nvshmem_buffer,
    float* grad_get_buffer,
    int32_t* grad_get_signal,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> dim_offset_per_rank,
    int32_t local_batch_size,
    int32_t total_dim_output,
    int32_t rank,
    int32_t nranks,
    float learning_rate
    ) {
    int32_t T = D_offsets.size(0) - 1;
    int32_t buffer_width =  static_cast<int32_t>(grad_output.size(1));
    const int32_t start_run_id = blockIdx.x * blockDim.y + threadIdx.y;
    // if(rank==0&&threadIdx.x==0)
        // printf("T:%d, buffer_width:%d, start_run_id:%d\n", T, buffer_width, start_run_id);

#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
    const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
    const unsigned int shfl_sync_mask = 0xffffffffu;
#endif
    constexpr int VEC_WIDTH = 4;

    for (uint32_t run_id = start_run_id;
         run_id < sorted_linear_indices_run.size(0) && run_id < sorted_linear_indices_num_runs[0];
             run_id += gridDim.x * blockDim.y) {

        const int64_t linear_index = sorted_linear_indices_run[run_id];
        const int32_t segment_start =
            sorted_linear_indices_cumulative_run_lengths[run_id];
        const int32_t segment_end =
            sorted_linear_indices_cumulative_run_lengths[run_id + 1];
        const int32_t SL = segment_end - segment_start;
        if (SL >= max_segment_length_per_warp) {
            // printf("Segment Too Large!\n");
            continue;
        }
        // now, each segment corresponds to exactly one table `t` and row in
        // that table (`idx`). Thus, we can hoist out some of the book-keeping.
        const auto info_0 = reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start];
        const auto t_0 = info_0 >> info_B_num_bits;

        int64_t hash_size = hash_size_cumsum[t_0];
        int32_t D = D_offsets[t_0 + 1] - D_offsets[t_0];
        int64_t idx = linear_index - hash_size;

        const int32_t SL_per_warp = div_round_up(SL, blockDim.y);
        const int32_t sl_start = 0;
        const int32_t sl_end = SL;
        Vec4T<at::acc_type<cache_t, true>> grad_sum[kMaxVecsPerThread];
        for (int32_t sl = sl_start; sl < sl_end; sl += kThreadGroupSize) {
            int32_t sl_j = sl + threadIdx.x;
            const auto b_t = sl_j < sl_end ? reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start + sl_j] : 0;
            const auto b = b_t & info_B_mask;
            const auto t = b_t >> info_B_num_bits; // if vbe
            int32_t D_start = sl_j < sl_end ? D_offsets[t] : 0; // if vbe // if not nobag

            // ================================ warp-based ================================
            // for (int32_t j = 0; j < kThreadGroupSize && sl + j < sl_end; ++j) {
            //     int32_t b_j = SHFL_SYNC(b, j);
            //     int32_t D_start_j = SHFL_SYNC(D_start, j);
            //     int32_t b_local_offset = b_j % local_batch_size; // row offset in the nvshmem output buffer
            //     int32_t get_offset = b_local_offset * total_dim_output + dim_offset_per_rank[rank] + D_start_j;
            //     int32_t target_gpu = b_j / local_batch_size; // target gpu _id
            //     if(target_gpu != rank){ // gradients in remote GPU memory
            //         int32_t t_j = SHFL_SYNC(t, j);
            //         int32_t get_signal = 0;
            //         if(threadIdx.x == 0)
            //             get_signal = atomicCAS(grad_get_signal + b_j * T + t_j, 0, 1); // if equal to zero, set to 1
            //         get_signal = SHFL_SYNC(get_signal, 0);
            //         int32_t output_offset = b_j * buffer_width + D_start_j;

            //         if(get_signal == 0){ // initialize nvshmem get
            //             // nvshmemx_float_get_warp(grad_get_buffer + output_offset, nvshmem_buffer + get_offset, D, target_gpu); // copy from shared memory
            //             if(threadIdx.x == 0)
            //                 atomicAdd(grad_get_signal + b_j * T + t_j, 1);
            //         }
            //     }
            // }
            // ================================ thread-based ================================
            if(sl_j < sl_end){
                int32_t local_b = static_cast<int32_t>(b);
                int32_t local_t = static_cast<int32_t>(t);

                int32_t b_local_offset = local_b % local_batch_size; // row offset in the nvshmem output buffer
                int32_t get_offset = b_local_offset * total_dim_output + dim_offset_per_rank[rank] + D_start;
                int32_t target_gpu = local_b / local_batch_size; // target gpu _id

                if(target_gpu != rank){ // gradients in remote GPU memory
                    // int32_t get_signal = atomicCAS(const_cast<int32_t*>(grad_get_signal) + local_b * T + local_t, 0, 1); // if equal to zero, set to 1

                    int32_t get_signal = atomicCAS(grad_get_signal + local_b * T + local_t, 0, 1);
                    int32_t output_offset = local_b * buffer_width + D_start;

                    if(get_signal == 0){ // initialize nvshmem get
                        nvshmem_getmem(grad_get_buffer + output_offset, nvshmem_buffer + get_offset, D * sizeof(float), target_gpu);
                        __threadfence();
                        // atomicAdd(const_cast<int32_t*>(grad_get_signal) + local_b * T + local_t, 1);
                        atomicAdd(grad_get_signal + local_b * T + local_t, 1);
                    }
                }
            }
        }
    }
    __syncthreads();
    // // ===================================================================================================
    // // ===================================================================================================
    // // ===================================================================================================
    for (uint32_t run_id = start_run_id;
         run_id < sorted_linear_indices_run.size(0) && run_id < sorted_linear_indices_num_runs[0];
             run_id += gridDim.x * blockDim.y) {

        const int64_t linear_index = sorted_linear_indices_run[run_id];
        const int32_t segment_start =
            sorted_linear_indices_cumulative_run_lengths[run_id];
        const int32_t segment_end =
            sorted_linear_indices_cumulative_run_lengths[run_id + 1];
        const int32_t SL = segment_end - segment_start;


        if (SL >= max_segment_length_per_warp) {
            continue;
        }

        // now, each segment corresponds to exactly one table `t` and row in
        // that table (`idx`). Thus, we can hoist out some of the book-keeping.
        const auto info_0 = reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start];
        const auto t_0 = info_0 >> info_B_num_bits;

        int64_t hash_size = hash_size_cumsum[t_0];
        int32_t D = D_offsets[t_0 + 1] - D_offsets[t_0];
        int64_t idx = linear_index - hash_size;

        const int32_t SL_per_warp = div_round_up(SL, blockDim.y);
        const int32_t sl_start = 0;
        const int32_t sl_end = SL;
        Vec4T<at::acc_type<cache_t, true>> grad_sum[kMaxVecsPerThread];
        for (int32_t sl = sl_start; sl < sl_end; sl += kThreadGroupSize) {
            int32_t sl_j = sl + threadIdx.x;
            const auto b_t = sl_j < sl_end ? reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start + sl_j] : 0;
            const auto b = b_t & info_B_mask;
            const auto t = b_t >> info_B_num_bits; // if vbe

            int32_t D_start = sl_j < sl_end ? D_offsets[t] : 0; // if vbe // if not nobag

            for (int32_t j = 0; j < kThreadGroupSize && sl + j < sl_end; ++j) {
                int32_t b_j = SHFL_SYNC(b, j);
                int32_t D_start_j = SHFL_SYNC(D_start, j);

                int32_t b_local_offset = b_j % local_batch_size; // row offset in the nvshmem output buffer
                int32_t get_offset = b_local_offset * total_dim_output + dim_offset_per_rank[rank] + D_start_j;
                int32_t output_offset = b_j * buffer_width + D_start_j;
                int32_t target_gpu = b_j / local_batch_size; // target gpu _id

                // ================== Wait for Gradients Get ===================
                float* grad_addr;
                if(target_gpu != rank){ // gradients in remote GPU memory
                    int32_t t_j = SHFL_SYNC(t, j);
                    int32_t get_signal = 0;
                    if(threadIdx.x == 0)
                        get_signal = atomicCAS(const_cast<int32_t*>(grad_get_signal) + b_j * T + t_j, 0, 0);
                    get_signal = SHFL_SYNC(get_signal, 0);

                    while(get_signal==1){
                        if(threadIdx.x == 0)
                            get_signal = atomicCAS(const_cast<int32_t*>(grad_get_signal) + b_j * T + t_j, 0, 0);
                        get_signal = SHFL_SYNC(get_signal, 0);
                    }
                    // =========================================================
                    grad_addr = grad_get_buffer + output_offset;
                }
                else{ // gradients in local GPU memory
                    grad_addr = nvshmem_buffer + get_offset;
                }
                // =============================================================
                __syncwarp();
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                        i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                        ++i) {
                    int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                    Vec4T<at::acc_type<grad_t, true>> grad_out_vec(
                        grad_addr + d
                    );
                    grad_sum[i].add_(grad_out_vec);
                }
            }
        }

        split_sgd_table_update_kernel
          <emb_t, cache_t, kMaxVecsPerThread, kThreadGroupSize, VEC_WIDTH>(
              dev_weights,
              uvm_weights,
              lxu_cache_weights,
              weights_placements,
              weights_offsets,
              sorted_lxu_cache_locations,
              grad_sum,
              stochastic_rounding,
              stochastic_rounding_philox_args,
              run_id,
              D,
              t_0,
              idx,
              segment_start,
              shfl_sync_mask,
              threadIdx.y * kMaxVecsPerThread * kThreadGroupSize, // shared_weight_offset
              learning_rate); // if not dense and optimizer != "none"

    }
}


template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize >
__global__ __launch_bounds__(kMaxThreads) void
nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1_cache(
    const pta::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits> grad_output,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements, // if optimizer != "none"
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_ids,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> num_long_run_ids,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_lxu_cache_locations,
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args, // if not dense and optimizer != "none"
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_id_to_really_long_run_ids,
    pta::PackedTensorAccessor32<at::acc_type<cache_t, true>, 2, at::RestrictPtrTraits> temp_grad_accum,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> grad_accum_counter,
    const int32_t max_segment_length_per_cta,
    const bool use_deterministic_algorithms,
    // nvshmem parameters
    float* nvshmem_buffer,
    float* grad_get_buffer,
    int32_t* grad_get_signal,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> dim_offset_per_rank,
    int32_t local_batch_size,
    int32_t total_dim_output,
    int32_t rank,
    int32_t nranks,
    float learning_rate) {
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
  const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
  const unsigned int shfl_sync_mask = 0xffffffffu;
#endif
  constexpr int VEC_WIDTH = 4;

  int32_t buffer_width =  static_cast<int32_t>(grad_output.size(1));
  int32_t T = weights_offsets.size(0);
  const int32_t num_long_runs = num_long_run_ids[0];

//   int32_t max_D = 256;
//   __shared__ float grad_get_buffer[256 * 32]; //currently only support embedding_dim <= 256
//   int32_t shared_d_offset = threadIdx.y * max_D;

  // ===========================================================================================
    for (int32_t long_run_id = blockIdx.x; long_run_id < num_long_runs; long_run_id += gridDim.x) {
        // The first thread block in the really long run has run_id in long_run_ids
        // and the rest have the negative of its offset (see find_long_segments kernel).
        int32_t cta_rank_on_current_run = 0;
        int32_t current_run_id = long_run_ids[long_run_id];
        if (current_run_id < 0) {
            cta_rank_on_current_run = -long_run_ids[long_run_id];
            current_run_id = long_run_ids[long_run_id - cta_rank_on_current_run];
        }

        const int64_t linear_index = sorted_linear_indices_run[current_run_id];
        const int32_t segment_start =
            sorted_linear_indices_cumulative_run_lengths[current_run_id] +
            cta_rank_on_current_run * max_segment_length_per_cta;
        // const int32_t segment_end = std::min(
        const int32_t segment_end = min(
            use_deterministic_algorithms ? INT_MAX : segment_start + max_segment_length_per_cta,
            sorted_linear_indices_cumulative_run_lengths[current_run_id + 1]);
        const int32_t SL = segment_end - segment_start;
        const int32_t warp_id = threadIdx.y;
        const int32_t lane_id = threadIdx.x;

        const auto info_0 = reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start];
        const auto t_0 = info_0 >> info_B_num_bits;
        int32_t D = D_offsets[t_0 + 1] - D_offsets[t_0];

        const int32_t SL_per_warp = div_round_up(SL, blockDim.y);
        const int32_t sl_start = SL_per_warp * warp_id;
        const int32_t sl_end = min(SL_per_warp * (warp_id + 1), SL);
        Vec4T<at::acc_type<cache_t, true>> grad_sum[kMaxVecsPerThread];

        for (int32_t sl = sl_start; sl < sl_end; sl += kThreadGroupSize) {
            int32_t sl_j = sl + threadIdx.x;
            const auto b_t = sl_j < sl_end ? reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start + sl_j] : 0;
            const auto b = b_t & info_B_mask;
            const auto t = b_t >> info_B_num_bits; // if vbe
            int32_t D_start = sl_j < sl_end ? D_offsets[t] : 0; // if vbe // if not nobag

            // ================================ thread-based ================================
            if(sl_j < sl_end){
                int32_t local_b = static_cast<int32_t>(b);
                int32_t local_t = static_cast<int32_t>(t);

                int32_t b_local_offset = local_b % local_batch_size; // row offset in the nvshmem output buffer
                int32_t get_offset = b_local_offset * total_dim_output + dim_offset_per_rank[rank] + D_start;
                int32_t target_gpu = local_b / local_batch_size; // target gpu _id

                if(target_gpu != rank){ // gradients in remote GPU memory
                    // int32_t get_signal = atomicCAS(const_cast<int32_t*>(grad_get_signal) + local_b * T + local_t, 0, 1); // if equal to zero, set to 1

                    int32_t get_signal = atomicCAS(grad_get_signal + local_b * T + local_t, 0, 1);
                    int32_t output_offset = local_b * buffer_width + D_start;

                    if(get_signal == 0){ // initialize nvshmem get
                        nvshmem_getmem(grad_get_buffer + output_offset, nvshmem_buffer + get_offset, D * sizeof(float), target_gpu);
                        __threadfence();
                        // atomicAdd(const_cast<int32_t*>(grad_get_signal) + local_b * T + local_t, 1);
                        atomicAdd(grad_get_signal + local_b * T + local_t, 1);
                    }
                }
            }
        }
    }
    __syncthreads();


  // ===========================================================================================
  for (int32_t long_run_id = blockIdx.x; long_run_id < num_long_runs; long_run_id += gridDim.x) {
        // The first thread block in the really long run has run_id in long_run_ids
        // and the rest have the negative of its offset (see find_long_segments kernel).
        int32_t cta_rank_on_current_run = 0;
        int32_t current_run_id = long_run_ids[long_run_id];
        if (current_run_id < 0) {
            cta_rank_on_current_run = -long_run_ids[long_run_id];
            current_run_id = long_run_ids[long_run_id - cta_rank_on_current_run];
        }
        const int32_t run_length =
            sorted_linear_indices_cumulative_run_lengths[current_run_id + 1] -
            sorted_linear_indices_cumulative_run_lengths[current_run_id];
        // This computation must agree with how we compute num_ctas_for_run in
        // find_long_segments kernel!
        const int32_t num_ctas_on_current_run =
            use_deterministic_algorithms ? 1 : div_round_up(run_length, max_segment_length_per_cta);


        const int64_t linear_index = sorted_linear_indices_run[current_run_id];
        const int32_t segment_start =
            sorted_linear_indices_cumulative_run_lengths[current_run_id] +
            cta_rank_on_current_run * max_segment_length_per_cta;
        // const int32_t segment_end = std::min(
        const int32_t segment_end = min(
            use_deterministic_algorithms ? INT_MAX : segment_start + max_segment_length_per_cta,
            sorted_linear_indices_cumulative_run_lengths[current_run_id + 1]);
        const int32_t SL = segment_end - segment_start;
        const int32_t warp_id = threadIdx.y;
        const int32_t lane_id = threadIdx.x;

        // Note that with shared embedding tables we can have multiple tables
        // (i.e. different values of `t` sharing the same segment).
        //
        const auto info_0 = reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start];
        const auto t_0 = info_0 >> info_B_num_bits;

        int64_t hash_size = hash_size_cumsum[t_0];
        int32_t D = D_offsets[t_0 + 1] - D_offsets[t_0];
        int64_t idx = linear_index - hash_size;

        const int32_t SL_per_warp = div_round_up(SL, blockDim.y);
        const int32_t sl_start = SL_per_warp * warp_id;
        const int32_t sl_end = min(SL_per_warp * (warp_id + 1), SL);
        Vec4T<at::acc_type<cache_t, true>> grad_sum[kMaxVecsPerThread];
        for (int32_t sl = sl_start; sl < sl_end; sl += kThreadGroupSize) {
            int32_t sl_j = sl + threadIdx.x;
            const auto b_t = sl_j < sl_end ? reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start + sl_j] : 0;
            const auto b = b_t & info_B_mask;
            const auto t = b_t >> info_B_num_bits; // if vbe
            int32_t D_start = sl_j < sl_end ? D_offsets[t] : 0; // if vbe // if not nobag
            for (int32_t j = 0; j < kThreadGroupSize && sl + j < sl_end; ++j) {
                int32_t b_j = SHFL_SYNC(b, j);
                int32_t D_start_j = SHFL_SYNC(D_start, j);

                int32_t b_local_offset = b_j % local_batch_size; // row offset in the nvshmem output buffer
                int32_t get_offset = b_local_offset * total_dim_output + dim_offset_per_rank[rank] + D_start_j;
                int32_t output_offset = b_j * buffer_width + D_start_j;
                int32_t target_gpu = b_j / local_batch_size; // target gpu _id

                float* grad_addr;
                if(target_gpu != rank){ // gradients in remote GPU memory
                    int32_t t_j = SHFL_SYNC(t, j);
                    int32_t get_signal = 0;
                    if(threadIdx.x == 0)
                        get_signal = atomicCAS(const_cast<int32_t*>(grad_get_signal) + b_j * T + t_j, 0, 0);
                    get_signal = SHFL_SYNC(get_signal, 0);

                    while(get_signal==1){
                        if(threadIdx.x == 0)
                            get_signal = atomicCAS(const_cast<int32_t*>(grad_get_signal) + b_j * T + t_j, 0, 0);
                        get_signal = SHFL_SYNC(get_signal, 0);
                    }
                    // =========================================================
                    grad_addr = grad_get_buffer + output_offset;
                }
                else{ // gradients in local GPU memory
                    grad_addr = nvshmem_buffer + get_offset;
                }
                // =============================================================
                __syncwarp();
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                        i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                        ++i) {
                    int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                    Vec4T<at::acc_type<grad_t, true>> grad_out_vec(
                        grad_addr + d
                    );
                    grad_sum[i].add_(grad_out_vec);
                }
            }
        }

        // do shared memory reduction only if we used multiple warps.
        if (SL > SL_per_warp) {
            struct SharedMemory<Vec4T<at::acc_type<cache_t, true>>> smem;
            Vec4T<at::acc_type<cache_t, true>>* shared_grad_sums = smem.getPointer();

            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
            shared_grad_sums
                [lane_id + i * kThreadGroupSize +
                warp_id * kMaxVecsPerThread * kThreadGroupSize] = grad_sum[i];
            }
            __syncthreads();
            if (blockDim.y >= 32) {
            if (warp_id < 16) {
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kThreadGroupSize +
                    warp_id * kMaxVecsPerThread * kThreadGroupSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                warp_id * kMaxVecsPerThread * kThreadGroupSize],
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                (warp_id + 16) * kMaxVecsPerThread * kThreadGroupSize]);
                }
            }
            __syncthreads();
            }
            if (blockDim.y >= 16) {
            if (warp_id < 8) {
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kThreadGroupSize +
                    warp_id * kMaxVecsPerThread * kThreadGroupSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                warp_id * kMaxVecsPerThread * kThreadGroupSize],
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                (warp_id + 8) * kMaxVecsPerThread * kThreadGroupSize]);
                }
            }
            __syncthreads();
            }
            if (blockDim.y >= 8) {
            if (warp_id < 4) {
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kThreadGroupSize +
                    warp_id * kMaxVecsPerThread * kThreadGroupSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                warp_id * kMaxVecsPerThread * kThreadGroupSize],
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                (warp_id + 4) * kMaxVecsPerThread * kThreadGroupSize]);
                }
            }
            __syncthreads();
            }
            if (blockDim.y >= 4) {
            if (warp_id < 2) {
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kThreadGroupSize +
                    warp_id * kMaxVecsPerThread * kThreadGroupSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                warp_id * kMaxVecsPerThread * kThreadGroupSize],
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                (warp_id + 2) * kMaxVecsPerThread * kThreadGroupSize]);
                }
            }
            __syncthreads();
            }
            if (warp_id == 0) {
            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
                grad_sum[i] = vec4_acc(
                    shared_grad_sums
                        [lane_id + i * kThreadGroupSize +
                        warp_id * kMaxVecsPerThread * kThreadGroupSize],
                    shared_grad_sums
                        [lane_id + i * kThreadGroupSize +
                        (warp_id + 1) * kMaxVecsPerThread * kThreadGroupSize]);
            }
            }
        }

        if (warp_id != 0) {
            continue;
        }

        if (num_ctas_on_current_run > 1) {
            int really_long_run_id = long_run_id_to_really_long_run_ids[long_run_id];
            Vec4T<at::acc_type<cache_t, true>> *temp_grad_accum_ptr =
                reinterpret_cast<Vec4T<at::acc_type<cache_t, true>>*>(&temp_grad_accum[really_long_run_id][0]);
            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
                gpuAtomicAdd(&temp_grad_accum_ptr[lane_id + i * kThreadGroupSize].acc.x, grad_sum[i].acc.x);
                gpuAtomicAdd(&temp_grad_accum_ptr[lane_id + i * kThreadGroupSize].acc.y, grad_sum[i].acc.y);
                gpuAtomicAdd(&temp_grad_accum_ptr[lane_id + i * kThreadGroupSize].acc.z, grad_sum[i].acc.z);
                gpuAtomicAdd(&temp_grad_accum_ptr[lane_id + i * kThreadGroupSize].acc.w, grad_sum[i].acc.w);
            }
            int counter;
            if (threadIdx.x == 0) {
                __threadfence();
                counter = gpuAtomicAdd(&grad_accum_counter[really_long_run_id], -1);
            }
            counter = SHFL_SYNC(counter, 0);
            // Only the thread block accumulated the gradient last does the weight update.
            if (counter > 1) {
                continue;
            }
            CUDA_KERNEL_ASSERT(counter == 1 && "Invalid grad_accum_counter. Race condition?");
            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
                grad_sum[i] = temp_grad_accum_ptr[lane_id + i * kThreadGroupSize];
            }
        }
        split_sgd_table_update_kernel
          <emb_t, cache_t, kMaxVecsPerThread, kThreadGroupSize, VEC_WIDTH>(
              dev_weights,
              uvm_weights,
              lxu_cache_weights,
              weights_placements,
              weights_offsets,
              sorted_lxu_cache_locations,
              grad_sum,
              stochastic_rounding,
              stochastic_rounding_philox_args,
              current_run_id,
              D,
              t_0,
              idx,
              segment_start,
              shfl_sync_mask,
              0, // shared_weight_offset
              learning_rate);
    } // for each run
}



////////////////////////////////////////////////////////////////////////////////
// Operator Code
////////////////////////////////////////////////////////////////////////////////

Tensor nvshmem_split_embedding_backward_codegen_sgd_unweighted_exact_cuda(
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
    bool stochastic_rounding,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    // nvshmem_parameters:
    float* nvshmem_buffer,
    Tensor dim_offset_per_rank,
    int32_t local_batch_size,
    int32_t total_dim_output,
    int32_t rank,
    int32_t nranks,
    double learning_rate
    ) {

   TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
        dev_weights,
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        weights_offsets,
        D_offsets,
        hash_size_cumsum,
        indices,
        offsets,
        lxu_cache_locations,
        grad_output);

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(dev_weights.get_device());
    TORCH_CHECK(max_D <= 1024);
    // Set total_unique_indices to total num indices by default
    const auto total_unique_indices = indices.numel();

    // short-circuit if there are zero indices.
    if (indices.numel() == 0) {
        return Tensor();
    }
    int32_t T = D_offsets.numel() - 1;

    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]
    const auto total_B = offsets.size(0) - 1;
    TORCH_CHECK(total_B > 0);
    auto BT_block_size = kMaxThreads / kWarpSize;
    TORCH_CHECK(BT_block_size * kWarpSize <= kMaxThreads);

    // printf("BT_block_size:%d, %d, %d\n", BT_block_size, kMaxThreads, kWarpSize);

    // V100: 96 KB; A100: 160 KB; H100: 228 KB.
    int max_shared_bytes = 0;
#ifndef __HIP_PLATFORM_HCC__
    cudaDeviceGetAttribute(&max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_weights.get_device());
#else
    // MI100 has 64 KB local memory (shared memory) per workgroup
    max_shared_bytes = 64 << 10;
#endif
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int shared_kb = max_shared_bytes >> 10;
    // V100: 64 KB; A100: 96 KB; H100: 144 KB
#ifndef __HIP_PLATFORM_HCC__
    // Use 2/3 of the available GPU shared mem; leave rooms for L1$.
    int used_shared_kb = round_down(shared_kb * 2 / 3, 16);
    TORCH_CHECK(used_shared_kb > 0);
#else
    // MI100 has independent shared mem and L1
    int used_shared_kb = shared_kb;
#endif

    int nvshmem_shared_bytes = 256 * 32 * 4; // currently support max_D = 256;
    int used_shared_bytes = nvshmem_shared_bytes + (used_shared_kb << 10);

    Tensor linear_indices, linear_indices_sorted;
    Tensor infos_sorted;
    Tensor sorted_linear_indices_run, sorted_linear_indices_run_lengths,
        sorted_linear_indices_num_runs,
        sorted_linear_indices_cumulative_run_lengths;
    std::tie(
        linear_indices,
        linear_indices_sorted,
        infos_sorted,
        sorted_linear_indices_run,
        sorted_linear_indices_run_lengths,
        sorted_linear_indices_num_runs,
        sorted_linear_indices_cumulative_run_lengths) =
        transpose_embedding_input_local(
        // transpose_embedding_input(
            hash_size_cumsum,
            total_hash_size_bits,
            indices,
            offsets,
            false,
            c10::optional<Tensor>(),
            info_B_num_bits,
            info_B_mask,
            total_unique_indices);
    auto lxu_cache_locations_sorted = at::empty_like(lxu_cache_locations);
    if (lxu_cache_locations.size(0) > 0) {
        size_t temp_storage_bytes = 0;
        AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            // radix_sort_pairs(
            nullptr,
            temp_storage_bytes,
            linear_indices.data_ptr<int64_t>(),
            linear_indices_sorted.data_ptr<int64_t>(),
            lxu_cache_locations.data_ptr<int32_t>(),
            lxu_cache_locations_sorted.data_ptr<int32_t>(),
            linear_indices.numel(),
            0,
            total_hash_size_bits,
            at::cuda::getCurrentCUDAStream(),
            false));
        auto temp_storage = at::empty(
            {static_cast<int64_t>(temp_storage_bytes)},
            indices.options().dtype(at::kByte));
        AT_CUDA_CHECK( cub::DeviceRadixSort::SortPairs(
            // radix_sort_pairs(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            linear_indices.data_ptr<int64_t>(),
            linear_indices_sorted.data_ptr<int64_t>(),
            lxu_cache_locations.data_ptr<int32_t>(),
            lxu_cache_locations_sorted.data_ptr<int32_t>(),
            linear_indices.numel(),
            0,
            total_hash_size_bits,
            at::cuda::getCurrentCUDAStream(),
            false));
    }

    DISPATCH_EMB_GRAD_CACHE_TYPES(
        dev_weights.scalar_type(),
        grad_output.scalar_type(),
        lxu_cache_weights.scalar_type(),
            "split_embedding_backward_sgd_exact_kernel",
        [&] {

            // early memory release
            linear_indices.reset();
            linear_indices_sorted.reset();

            auto grad_output_accessor = MAKE_PTA_WITH_NAME("split_embedding_backward_codegen_sgd_unweighted_exact_cuda.1", grad_output, grad_t, 2, 64);
            Tensor grad_output_mean;
            if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN) {
              grad_output_mean = at::empty_like(grad_output);

#ifdef FBGEMM_GPU_MEMCHECK
              const auto func_name1 = "grad_mean_kernel";
#endif

              grad_mean_kernel<<<
                    div_round_up(total_B, kMaxThreads / kWarpSize),
                    dim3(kWarpSize, kMaxThreads / kWarpSize),
                    0,
                    at::cuda::getCurrentCUDAStream()>>>
                    (
                        MAKE_PTA_WITH_NAME(func_name1, grad_output_mean, grad_t, 2, 64),
                        MAKE_PTA_WITH_NAME(func_name1, grad_output, grad_t, 2, 64),
                        MAKE_PTA_WITH_NAME(func_name1, D_offsets, int32_t, 1, 32),
                        MAKE_PTA_WITH_NAME(func_name1, offsets, int64_t, 1, 32),
                        FixedDivisor(total_B / T)
                    );
              C10_CUDA_KERNEL_LAUNCH_CHECK(); // if not dense or not vbe


              grad_output_accessor = MAKE_PTA_WITH_NAME("split_embedding_backward_codegen_sgd_unweighted_exact_cuda.2", grad_output_mean, grad_t, 2, 64);
            }
            at::PhiloxCudaState rng_engine_inputs;
            if (stochastic_rounding && !std::is_same<emb_t, float>::value) {
                auto gen = at::cuda::detail::getDefaultCUDAGenerator();
                std::lock_guard<std::mutex> lock(gen.mutex());
                rng_engine_inputs =
                    at::check_generator<at::CUDAGeneratorImpl>(gen)
                        ->philox_cuda_state(4);
            }
            // kMaxElemPerThread is # of elements handled by thread if we use a full warp for a row
            // We consider kMaxElemPerThread 1 and 2, and then a multiple of 4.
//             if (max_D <= 32) {
//             // hipcc can't use max in constexpr
//             constexpr int kMaxVecsPerThread = 1 / 4 >= 1 ? 1 / 4 : 1;
//             // If max_D is small, use fewer number of threads than kWarpSize.
// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//             constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 1, 1);
// #else
//             constexpr int kThreadGroupSize = kWarpSize;
// #endif
//             // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
//             while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
//                 BT_block_size /= 2;
//             }
//             TORCH_CHECK(BT_block_size >= 1);
//             if (std::is_same<emb_t, double>::value) {
//                 // Otherwise we see CUDA kernel launch failures despite the above checks.
//                 BT_block_size = 1;
//             }

//             auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
//             auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

//             const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
//             const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
//             Tensor long_run_id_to_really_long_run_ids;
//             if (use_deterministic_algorithms) {
//                 long_run_id_to_really_long_run_ids =
//                     at::empty(0, sorted_linear_indices_run_lengths.options());
//             } else {
//                 long_run_id_to_really_long_run_ids =
//                     at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
//             }
//             auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
//             auto grad_accum_counter = at::empty(
//                 use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
//                 indices.options().dtype(at::kInt));

// #ifdef FBGEMM_GPU_MEMCHECK
//             const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
// #endif

//             split_embedding_backward_codegen_find_long_segments<<<
//                 div_round_up(total_unique_indices, kMaxThreads),
//                 kMaxThreads,
//                 0,
//                 at::cuda::getCurrentCUDAStream()
//             >>>(
//                 MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
//                 MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
//                 MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
//                 MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
//                 MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
//                 MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
//                 MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
//                 max_segment_length_per_warp,
//                 max_segment_length_per_cta,
//                 use_deterministic_algorithms);
//             C10_CUDA_KERNEL_LAUNCH_CHECK();

//             // A temp buffer to accumulate gradients with atomics.
//             auto temp_grad_accum = at::zeros(
//                 {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
//                 grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

//             int32_t grid_size = std::min(
//                 div_round_up(total_unique_indices, kMaxThreads),
//                 get_max_thread_blocks_());

//             // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
//             // "Compute capability 7.x devices allow a single thread block to
//             // address the full capacity of shared memory: 96 KB on Volta,
//             // 64 KB on Turing. Kernels relying on shared memory allocations
//             // over 48 KB per block are architecture-specific, as such they
//             // must use dynamic shared memory (rather than statically sized
//             // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

// #ifndef __HIP_PLATFORM_HCC__
//             cudaFuncSetAttribute(
//                 split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
//                 emb_t,
//                 grad_t,
//                 cache_t,
//                 kMaxVecsPerThread,
//                 kThreadGroupSize>,
//                 cudaFuncAttributeMaxDynamicSharedMemorySize,
//                 used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
// #endif
//             C10_CUDA_KERNEL_LAUNCH_CHECK();

// #ifdef FBGEMM_GPU_MEMCHECK
//             const auto func_name3 = "split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
// #endif

//             // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
//             split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
//                 emb_t,
//                 grad_t,
//                 cache_t,
//                 kMaxVecsPerThread,
//                 kThreadGroupSize>
//                 <<<grid_size,
//                     dim3(kThreadGroupSize, BT_block_size),
//                     BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
//                         kMaxVecsPerThread,
//                     at::cuda::getCurrentCUDAStream()>>>(
//                     grad_output_accessor,
//                     MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
//                     MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
//                     MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
//                     stochastic_rounding,
//                     rng_engine_inputs, // if not dense and optimizer != "none"
//                     info_B_num_bits,
//                     info_B_mask,
//                     MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
//                     MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
//                     MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
//                     max_segment_length_per_cta,
//                     use_deterministic_algorithms,
//                     learning_rate);

//             C10_CUDA_KERNEL_LAUNCH_CHECK();
//             grid_size = std::min(
//                 div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
//                 get_max_thread_blocks_());

//             // Shared memory is not needed for non uint8_t weights
//             size_t shmem_bytes = 0;
//             if (std::is_same<emb_t, uint8_t>::value) {
//                 shmem_bytes = BT_block_size * sizeof(
//                     at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
// #ifndef __HIP_PLATFORM_HCC__
//                 cudaFuncSetAttribute(
//                     split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
//                     emb_t,
//                     grad_t,
//                     cache_t,
//                     kMaxVecsPerThread,
//                     kThreadGroupSize>,
//                     cudaFuncAttributeMaxDynamicSharedMemorySize,
//                     used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
// #endif
//             }

// #ifdef FBGEMM_GPU_MEMCHECK
//             const auto func_name4 = "split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
// #endif

//             C10_CUDA_KERNEL_LAUNCH_CHECK();
//             split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
//                 emb_t,
//                 grad_t,
//                 cache_t,
//                 kMaxVecsPerThread,
//                 kThreadGroupSize>
//                 <<<grid_size,
//                     dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
//                     shmem_bytes,
//                     at::cuda::getCurrentCUDAStream()>>>(
//                     grad_output_accessor,
//                     MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
//                     MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
//                     max_segment_length_per_warp,
//                     stochastic_rounding,
//                     rng_engine_inputs, // if not dense and optimizer != "none"
//                     info_B_num_bits,
//                     info_B_mask,
//                     learning_rate);
//             C10_CUDA_KERNEL_LAUNCH_CHECK();
//             return;
//         }
//             if (max_D <= 64) {
//             // hipcc can't use max in constexpr
//             constexpr int kMaxVecsPerThread = 2 / 4 >= 1 ? 2 / 4 : 1;
//             // If max_D is small, use fewer number of threads than kWarpSize.
// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//             constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 2, 1);
// #else
//             constexpr int kThreadGroupSize = kWarpSize;
// #endif
//             // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
//             while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
//                 BT_block_size /= 2;
//             }
//             TORCH_CHECK(BT_block_size >= 1);
//             if (std::is_same<emb_t, double>::value) {
//                 // Otherwise we see CUDA kernel launch failures despite the above checks.
//                 BT_block_size = 1;
//             }

//             auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
//             auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

//             const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
//             const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
//             Tensor long_run_id_to_really_long_run_ids;
//             if (use_deterministic_algorithms) {
//                 long_run_id_to_really_long_run_ids =
//                     at::empty(0, sorted_linear_indices_run_lengths.options());
//             } else {
//                 long_run_id_to_really_long_run_ids =
//                     at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
//             }
//             auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
//             auto grad_accum_counter = at::empty(
//                 use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
//                 indices.options().dtype(at::kInt));

// #ifdef FBGEMM_GPU_MEMCHECK
//             const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
// #endif

//             split_embedding_backward_codegen_find_long_segments<<<
//                 div_round_up(total_unique_indices, kMaxThreads),
//                 kMaxThreads,
//                 0,
//                 at::cuda::getCurrentCUDAStream()
//             >>>(
//                 MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
//                 MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
//                 MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
//                 MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
//                 MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
//                 MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
//                 MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
//                 max_segment_length_per_warp,
//                 max_segment_length_per_cta,
//                 use_deterministic_algorithms);
//             C10_CUDA_KERNEL_LAUNCH_CHECK();

//             // A temp buffer to accumulate gradients with atomics.
//             auto temp_grad_accum = at::zeros(
//                 {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
//                 grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

//             int32_t grid_size = std::min(
//                 div_round_up(total_unique_indices, kMaxThreads),
//                 get_max_thread_blocks_());

//             // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
//             // "Compute capability 7.x devices allow a single thread block to
//             // address the full capacity of shared memory: 96 KB on Volta,
//             // 64 KB on Turing. Kernels relying on shared memory allocations
//             // over 48 KB per block are architecture-specific, as such they
//             // must use dynamic shared memory (rather than statically sized
//             // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

// #ifndef __HIP_PLATFORM_HCC__
//             cudaFuncSetAttribute(
//                 split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
//                 emb_t,
//                 grad_t,
//                 cache_t,
//                 kMaxVecsPerThread,
//                 kThreadGroupSize>,
//                 cudaFuncAttributeMaxDynamicSharedMemorySize,
//                 used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
// #endif
//             C10_CUDA_KERNEL_LAUNCH_CHECK();

// #ifdef FBGEMM_GPU_MEMCHECK
//             const auto func_name3 = "split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
// #endif

//             // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
//             split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
//                 emb_t,
//                 grad_t,
//                 cache_t,
//                 kMaxVecsPerThread,
//                 kThreadGroupSize>
//                 <<<grid_size,
//                     dim3(kThreadGroupSize, BT_block_size),
//                     BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
//                         kMaxVecsPerThread,
//                     at::cuda::getCurrentCUDAStream()>>>(
//                     grad_output_accessor,
//                     MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
//                     MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
//                     MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
//                     stochastic_rounding,
//                     rng_engine_inputs, // if not dense and optimizer != "none"
//                     info_B_num_bits,
//                     info_B_mask,
//                     MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
//                     MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
//                     MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
//                     max_segment_length_per_cta,
//                     use_deterministic_algorithms,
//                     learning_rate);

//             C10_CUDA_KERNEL_LAUNCH_CHECK();
//             grid_size = std::min(
//                 div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
//                 get_max_thread_blocks_());

//             // Shared memory is not needed for non uint8_t weights
//             size_t shmem_bytes = 0;
//             if (std::is_same<emb_t, uint8_t>::value) {
//                 shmem_bytes = BT_block_size * sizeof(
//                     at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
// #ifndef __HIP_PLATFORM_HCC__
//                 cudaFuncSetAttribute(
//                     split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
//                     emb_t,
//                     grad_t,
//                     cache_t,
//                     kMaxVecsPerThread,
//                     kThreadGroupSize>,
//                     cudaFuncAttributeMaxDynamicSharedMemorySize,
//                     used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
// #endif
//             }

// #ifdef FBGEMM_GPU_MEMCHECK
//             const auto func_name4 = "split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
// #endif

//             C10_CUDA_KERNEL_LAUNCH_CHECK();
//             split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
//                 emb_t,
//                 grad_t,
//                 cache_t,
//                 kMaxVecsPerThread,
//                 kThreadGroupSize>
//                 <<<grid_size,
//                     dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
//                     shmem_bytes,
//                     at::cuda::getCurrentCUDAStream()>>>(
//                     grad_output_accessor,
//                     MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
//                     MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
//                     max_segment_length_per_warp,
//                     stochastic_rounding,
//                     rng_engine_inputs, // if not dense and optimizer != "none"
//                     info_B_num_bits,
//                     info_B_mask,
//                     learning_rate);
//             C10_CUDA_KERNEL_LAUNCH_CHECK();
//             return;
//         }
            if (max_D <= 128) {
            // hipcc can't use max in constexpr
            constexpr int kMaxVecsPerThread = 4 / 4 >= 1 ? 4 / 4 : 1;
            // If max_D is small, use fewer number of threads than kWarpSize.
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
            constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 4, 1);
#else
            constexpr int kThreadGroupSize = kWarpSize;
#endif
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {

                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {

                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {

                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {

                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {

                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
        });
    return Tensor();
}



Tensor nvshmem_split_embedding_backward_codegen_sgd_unweighted_exact_cuda_cache(
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
    bool stochastic_rounding,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    // nvshmem_parameters:
    float* nvshmem_buffer,
    float* grad_get_buffer,
    int32_t* grad_get_signal,
    Tensor dim_offset_per_rank,
    int32_t local_batch_size,
    int32_t total_dim_output,
    int32_t rank,
    int32_t nranks,
    double learning_rate
)
{

   TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
        dev_weights,
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        weights_offsets,
        D_offsets,
        hash_size_cumsum,
        indices,
        offsets,
        lxu_cache_locations,
        grad_output);

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(dev_weights.get_device());
    TORCH_CHECK(max_D <= 1024);
    // Set total_unique_indices to total num indices by default
    const auto total_unique_indices = indices.numel();

    // short-circuit if there are zero indices.
    if (indices.numel() == 0) {
        return Tensor();
    }
    int32_t T = D_offsets.numel() - 1;

    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]
    const auto total_B = offsets.size(0) - 1;
    TORCH_CHECK(total_B > 0);
    auto BT_block_size = kMaxThreads / kWarpSize;
    TORCH_CHECK(BT_block_size * kWarpSize <= kMaxThreads);

    // V100: 96 KB; A100: 160 KB; H100: 228 KB.
    int max_shared_bytes = 0;
#ifndef __HIP_PLATFORM_HCC__
    cudaDeviceGetAttribute(&max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_weights.get_device());
#else
    // MI100 has 64 KB local memory (shared memory) per workgroup
    max_shared_bytes = 64 << 10;
#endif
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int shared_kb = max_shared_bytes >> 10;
    // V100: 64 KB; A100: 96 KB; H100: 144 KB
#ifndef __HIP_PLATFORM_HCC__
    // Use 2/3 of the available GPU shared mem; leave rooms for L1$.
    int used_shared_kb = round_down(shared_kb * 2 / 3, 16);
    TORCH_CHECK(used_shared_kb > 0);
#else
    // MI100 has independent shared mem and L1
    int used_shared_kb = shared_kb;
#endif
    int used_shared_bytes = used_shared_kb << 10;

    Tensor linear_indices, linear_indices_sorted;
    Tensor infos_sorted;
    Tensor sorted_linear_indices_run, sorted_linear_indices_run_lengths,
        sorted_linear_indices_num_runs,
        sorted_linear_indices_cumulative_run_lengths;
    std::tie(
        linear_indices,
        linear_indices_sorted,
        infos_sorted,
        sorted_linear_indices_run,
        sorted_linear_indices_run_lengths,
        sorted_linear_indices_num_runs,
        sorted_linear_indices_cumulative_run_lengths) =
        transpose_embedding_input_local(
        // transpose_embedding_input(
            hash_size_cumsum,
            total_hash_size_bits,
            indices,
            offsets,
            false,
            c10::optional<Tensor>(),
            info_B_num_bits,
            info_B_mask,
            total_unique_indices);
    auto lxu_cache_locations_sorted = at::empty_like(lxu_cache_locations);
    if (lxu_cache_locations.size(0) > 0) {
        size_t temp_storage_bytes = 0;
        AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            // radix_sort_pairs(
            nullptr,
            temp_storage_bytes,
            linear_indices.data_ptr<int64_t>(),
            linear_indices_sorted.data_ptr<int64_t>(),
            lxu_cache_locations.data_ptr<int32_t>(),
            lxu_cache_locations_sorted.data_ptr<int32_t>(),
            linear_indices.numel(),
            0,
            total_hash_size_bits,
            at::cuda::getCurrentCUDAStream(),
            false));
        auto temp_storage = at::empty(
            {static_cast<int64_t>(temp_storage_bytes)},
            indices.options().dtype(at::kByte));
        AT_CUDA_CHECK( cub::DeviceRadixSort::SortPairs(
            // radix_sort_pairs(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            linear_indices.data_ptr<int64_t>(),
            linear_indices_sorted.data_ptr<int64_t>(),
            lxu_cache_locations.data_ptr<int32_t>(),
            lxu_cache_locations_sorted.data_ptr<int32_t>(),
            linear_indices.numel(),
            0,
            total_hash_size_bits,
            at::cuda::getCurrentCUDAStream(),
            false));
    }

    DISPATCH_EMB_GRAD_CACHE_TYPES(
        dev_weights.scalar_type(),
        grad_output.scalar_type(),
        lxu_cache_weights.scalar_type(),
            "split_embedding_backward_sgd_exact_kernel",
        [&] {

            // early memory release
            linear_indices.reset();
            linear_indices_sorted.reset();

            auto grad_output_accessor = MAKE_PTA_WITH_NAME("split_embedding_backward_codegen_sgd_unweighted_exact_cuda.1", grad_output, grad_t, 2, 64);
            Tensor grad_output_mean;
            if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN) {
              grad_output_mean = at::empty_like(grad_output);

#ifdef FBGEMM_GPU_MEMCHECK
              const auto func_name1 = "grad_mean_kernel";
#endif

              grad_mean_kernel<<<
                    div_round_up(total_B, kMaxThreads / kWarpSize),
                    dim3(kWarpSize, kMaxThreads / kWarpSize),
                    0,
                    at::cuda::getCurrentCUDAStream()>>>
                    (
                        MAKE_PTA_WITH_NAME(func_name1, grad_output_mean, grad_t, 2, 64),
                        MAKE_PTA_WITH_NAME(func_name1, grad_output, grad_t, 2, 64),
                        MAKE_PTA_WITH_NAME(func_name1, D_offsets, int32_t, 1, 32),
                        MAKE_PTA_WITH_NAME(func_name1, offsets, int64_t, 1, 32),
                        FixedDivisor(total_B / T)
                    );
              C10_CUDA_KERNEL_LAUNCH_CHECK(); // if not dense or not vbe


              grad_output_accessor = MAKE_PTA_WITH_NAME("split_embedding_backward_codegen_sgd_unweighted_exact_cuda.2", grad_output_mean, grad_t, 2, 64);
            }
            at::PhiloxCudaState rng_engine_inputs;
            if (stochastic_rounding && !std::is_same<emb_t, float>::value) {
                auto gen = at::cuda::detail::getDefaultCUDAGenerator();
                std::lock_guard<std::mutex> lock(gen.mutex());
                rng_engine_inputs =
                    at::check_generator<at::CUDAGeneratorImpl>(gen)
                        ->philox_cuda_state(4);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1_cache";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1_cache<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    grad_get_buffer,
                    grad_get_signal,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1_cache";
#endif

            // printf("nBlock:%d, kThreadGroupSize:%d, kBackwardMaxThreads/kThreadGroupSize:%d\n", grid_size, kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1_cache<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    grad_get_buffer,
                    grad_get_signal,

                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1_cache";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1_cache<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    grad_get_buffer,
                    grad_get_signal,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1_cache";
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1_cache<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    grad_get_buffer,
                    grad_get_signal,

                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1_cache";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1_cache<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    grad_get_buffer,
                    grad_get_signal,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1_cache";
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1_cache<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    grad_get_buffer,
                    grad_get_signal,

                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1_cache";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1_cache<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    grad_get_buffer,
                    grad_get_signal,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1_cache";
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1_cache<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    grad_get_buffer,
                    grad_get_signal,

                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1_cache";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1_cache<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    grad_get_buffer,
                    grad_get_signal,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1_cache";
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1_cache<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    grad_get_buffer,
                    grad_get_signal,

                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1_cache";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1_cache<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    grad_get_buffer,
                    grad_get_signal,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1_cache";
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1_cache<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    grad_get_buffer,
                    grad_get_signal,

                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1_cache";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1_cache<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    grad_get_buffer,
                    grad_get_signal,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1_cache";
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1_cache<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    grad_get_buffer,
                    grad_get_signal,

                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1_cache";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1_cache<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    grad_get_buffer,
                    grad_get_signal,
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1_cache";
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        nvshmem_split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1_cache<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    // nvshmem_parameters:
                    nvshmem_buffer,
                    grad_get_buffer,
                    grad_get_signal,

                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank, int32_t, 1, 32),
                    local_batch_size,
                    total_dim_output,
                    rank,
                    nranks,
                    learning_rate);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
        });
    return Tensor();
}



// ==========================================================================================================
// ===================================== NVSHMEM-GET()-based All-to-All =====================================
// ==========================================================================================================
template <
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize >
__launch_bounds__(512) __global__
void nvshmem_bwd_alltoall(
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    FixedDivisor fd_B,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> indices,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,
    // all-to-all information:
    float* nvshmem_grad,
    float* grad_get_buffer,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> dim_sum_per_rank_data,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> dim_offset_per_rank_data,
    int32_t max_D,
    int32_t total_dim_output,
    int32_t nranks,
    int32_t rank,
    int32_t local_dim_output,
    int32_t local_batch_size
){
// shfl_sync_mask is implicitly used by SHFL_SYNC
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
    const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
    const unsigned int shfl_sync_mask = 0xffffffffu;
#endif

    // Determine the linearized warp ID, and exit early if needed
    int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    if (b_t >= offsets.size(0) - 1) {
        return;
    }

    // Determine the Table and Training Example IDs
    int32_t t;  // Table ID, which table
    int32_t b;  // Training Example ID, row offset in this table (max val of b == local_batch_size * nDev)
    fd_B.DivMod(b_t, &t, &b); // t = b_t / (local_batch_size * nDev); b = b_t % (local_batch_size * nDev)

    // Determine the number of indices (pooling factor) to look up within the bag
    int64_t indices_start = offsets[b_t];
    int64_t indices_end = offsets[b_t + 1];
    int32_t L = indices_end - indices_start;

    // Get the offsets of the embedding dimensions of the tables and determine D
    int32_t D_start = D_offsets[t];
    int32_t D_end = D_offsets[t + 1];
    int32_t D = D_end - D_start;

    // =============================== Get Gradients from the remote GPU ====================================
    int32_t target_gpu = b / local_batch_size; // target gpu _id
    int32_t b_local_offset = b % local_batch_size; // row offset in the nvshmem output buffer
    int32_t grad_offset = b_local_offset * total_dim_output + dim_offset_per_rank_data[rank] + D_start;
    int32_t output_offset = b * local_dim_output + D_start;

    nvshmemx_float_get_warp(grad_get_buffer + output_offset, nvshmem_grad + grad_offset, D, target_gpu); // copy from shared memory
}


template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize >
__global__ __launch_bounds__(kMaxThreads) void
nvshmem_alltoall_based_cta_per_row_kernel(
    const pta::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits> grad_output,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements, // if optimizer != "none"
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_ids,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> num_long_run_ids,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_lxu_cache_locations,
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args, // if not dense and optimizer != "none"
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_id_to_really_long_run_ids,
    pta::PackedTensorAccessor32<at::acc_type<cache_t, true>, 2, at::RestrictPtrTraits> temp_grad_accum,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> grad_accum_counter,
    const int32_t max_segment_length_per_cta,
    const bool use_deterministic_algorithms,
    float* nvshmem_get_grad,
    float learning_rate) {
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
  const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
  const unsigned int shfl_sync_mask = 0xffffffffu;
#endif
  constexpr int VEC_WIDTH = 4;
  int32_t T = weights_offsets.size(0);
  int32_t buffer_width =  static_cast<int32_t>(grad_output.size(1));
  const int32_t num_long_runs = num_long_run_ids[0];
  for (int32_t long_run_id = blockIdx.x; long_run_id < num_long_runs; long_run_id += gridDim.x) {
        // The first thread block in the really long run has run_id in long_run_ids
        // and the rest have the negative of its offset (see find_long_segments kernel).
        int32_t cta_rank_on_current_run = 0;
        int32_t current_run_id = long_run_ids[long_run_id];
        if (current_run_id < 0) {
            cta_rank_on_current_run = -long_run_ids[long_run_id];
            current_run_id = long_run_ids[long_run_id - cta_rank_on_current_run];
        }
        const int32_t run_length =
            sorted_linear_indices_cumulative_run_lengths[current_run_id + 1] -
            sorted_linear_indices_cumulative_run_lengths[current_run_id];
        // This computation must agree with how we compute num_ctas_for_run in
        // find_long_segments kernel!
        const int32_t num_ctas_on_current_run =
            use_deterministic_algorithms ? 1 : div_round_up(run_length, max_segment_length_per_cta);


        const int64_t linear_index = sorted_linear_indices_run[current_run_id];
        const int32_t segment_start =
            sorted_linear_indices_cumulative_run_lengths[current_run_id] +
            cta_rank_on_current_run * max_segment_length_per_cta;
        // const int32_t segment_end = std::min(
        const int32_t segment_end = min(
            use_deterministic_algorithms ? INT_MAX : segment_start + max_segment_length_per_cta,
            sorted_linear_indices_cumulative_run_lengths[current_run_id + 1]);
        const int32_t SL = segment_end - segment_start;
        const int32_t warp_id = threadIdx.y;
        const int32_t lane_id = threadIdx.x;

        // Note that with shared embedding tables we can have multiple tables
        // (i.e. different values of `t` sharing the same segment).
        //
        const auto info_0 = reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start];
        const auto t_0 = info_0 >> info_B_num_bits;

        int64_t hash_size = hash_size_cumsum[t_0];
        int32_t D = D_offsets[t_0 + 1] - D_offsets[t_0];
        int64_t idx = linear_index - hash_size;

        const int32_t SL_per_warp = div_round_up(SL, blockDim.y);
        const int32_t sl_start = SL_per_warp * warp_id;
        const int32_t sl_end = min(SL_per_warp * (warp_id + 1), SL);
        Vec4T<at::acc_type<cache_t, true>> grad_sum[kMaxVecsPerThread];
        for (int32_t sl = sl_start; sl < sl_end; sl += kThreadGroupSize) {
            int32_t sl_j = sl + threadIdx.x;
            const auto b_t = sl_j < sl_end ? reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start + sl_j] : 0;
            const auto b = b_t & info_B_mask;
            const auto t = b_t >> info_B_num_bits; // if vbe
            int32_t D_start = sl_j < sl_end ? D_offsets[t] : 0; // if vbe // if not nobag
            for (int32_t j = 0; j < kThreadGroupSize && sl + j < sl_end; ++j) {
                int32_t b_j = SHFL_SYNC(b, j);
                int32_t D_start_j = SHFL_SYNC(D_start, j);
                int32_t output_offset = b_j * buffer_width + D_start_j;

                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                    i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                    int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                    Vec4T<at::acc_type<grad_t, true>> grad_out_vec(
                        // &grad_output[b_j][0] + D_start_j + d
                        nvshmem_get_grad + output_offset + d
                    );
                    grad_sum[i].add_(grad_out_vec);
                }
            }
        }
        // do shared memory reduction only if we used multiple warps.
        if (SL > SL_per_warp) {
            struct SharedMemory<Vec4T<at::acc_type<cache_t, true>>> smem;
            Vec4T<at::acc_type<cache_t, true>>* shared_grad_sums = smem.getPointer();

            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
            shared_grad_sums
                [lane_id + i * kThreadGroupSize +
                warp_id * kMaxVecsPerThread * kThreadGroupSize] = grad_sum[i];
            }
            __syncthreads();
            if (blockDim.y >= 32) {
            if (warp_id < 16) {
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kThreadGroupSize +
                    warp_id * kMaxVecsPerThread * kThreadGroupSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                warp_id * kMaxVecsPerThread * kThreadGroupSize],
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                (warp_id + 16) * kMaxVecsPerThread * kThreadGroupSize]);
                }
            }
            __syncthreads();
            }
            if (blockDim.y >= 16) {
            if (warp_id < 8) {
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kThreadGroupSize +
                    warp_id * kMaxVecsPerThread * kThreadGroupSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                warp_id * kMaxVecsPerThread * kThreadGroupSize],
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                (warp_id + 8) * kMaxVecsPerThread * kThreadGroupSize]);
                }
            }
            __syncthreads();
            }
            if (blockDim.y >= 8) {
            if (warp_id < 4) {
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kThreadGroupSize +
                    warp_id * kMaxVecsPerThread * kThreadGroupSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                warp_id * kMaxVecsPerThread * kThreadGroupSize],
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                (warp_id + 4) * kMaxVecsPerThread * kThreadGroupSize]);
                }
            }
            __syncthreads();
            }
            if (blockDim.y >= 4) {
            if (warp_id < 2) {
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kThreadGroupSize +
                    warp_id * kMaxVecsPerThread * kThreadGroupSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                warp_id * kMaxVecsPerThread * kThreadGroupSize],
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                (warp_id + 2) * kMaxVecsPerThread * kThreadGroupSize]);
                }
            }
            __syncthreads();
            }
            if (warp_id == 0) {
            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
                grad_sum[i] = vec4_acc(
                    shared_grad_sums
                        [lane_id + i * kThreadGroupSize +
                        warp_id * kMaxVecsPerThread * kThreadGroupSize],
                    shared_grad_sums
                        [lane_id + i * kThreadGroupSize +
                        (warp_id + 1) * kMaxVecsPerThread * kThreadGroupSize]);
            }
            }
        }

        if (warp_id != 0) {
            continue;
        }

        if (num_ctas_on_current_run > 1) {
            int really_long_run_id = long_run_id_to_really_long_run_ids[long_run_id];
            Vec4T<at::acc_type<cache_t, true>> *temp_grad_accum_ptr =
                reinterpret_cast<Vec4T<at::acc_type<cache_t, true>>*>(&temp_grad_accum[really_long_run_id][0]);
            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
                gpuAtomicAdd(&temp_grad_accum_ptr[lane_id + i * kThreadGroupSize].acc.x, grad_sum[i].acc.x);
                gpuAtomicAdd(&temp_grad_accum_ptr[lane_id + i * kThreadGroupSize].acc.y, grad_sum[i].acc.y);
                gpuAtomicAdd(&temp_grad_accum_ptr[lane_id + i * kThreadGroupSize].acc.z, grad_sum[i].acc.z);
                gpuAtomicAdd(&temp_grad_accum_ptr[lane_id + i * kThreadGroupSize].acc.w, grad_sum[i].acc.w);
            }
            int counter;
            if (threadIdx.x == 0) {
                __threadfence();
                counter = gpuAtomicAdd(&grad_accum_counter[really_long_run_id], -1);
            }
            counter = SHFL_SYNC(counter, 0);
            // Only the thread block accumulated the gradient last does the weight update.
            if (counter > 1) {
                continue;
            }
            CUDA_KERNEL_ASSERT(counter == 1 && "Invalid grad_accum_counter. Race condition?");
            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
                grad_sum[i] = temp_grad_accum_ptr[lane_id + i * kThreadGroupSize];
            }
        }
        split_sgd_table_update_kernel
          <emb_t, cache_t, kMaxVecsPerThread, kThreadGroupSize, VEC_WIDTH>(
              dev_weights,
              uvm_weights,
              lxu_cache_weights,
              weights_placements,
              weights_offsets,
              sorted_lxu_cache_locations,
              grad_sum,
              stochastic_rounding,
              stochastic_rounding_philox_args,
              current_run_id,
              D,
              t_0,
              idx,
              segment_start,
              shfl_sync_mask,
              0, // shared_weight_offset
              learning_rate);
    } // for each run
}


template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize >
__global__ __launch_bounds__(kBackwardMaxThreads) void
nvshmem_alltoall_based_warp_per_row_kernel(
    const pta::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits> grad_output,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_lxu_cache_locations,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_num_runs,
    int32_t max_segment_length_per_warp,
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args, // if not dense and optimizer != "none"
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    float* nvshmem_get_grad,
    float learning_rate) {
    int32_t T = D_offsets.size(0) - 1;
    const int32_t start_run_id = blockIdx.x * blockDim.y + threadIdx.y;

#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
    const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
    const unsigned int shfl_sync_mask = 0xffffffffu;
#endif
    constexpr int VEC_WIDTH = 4;
    int32_t buffer_width =  static_cast<int32_t>(grad_output.size(1));

    for (uint32_t run_id = start_run_id;
         run_id < sorted_linear_indices_run.size(0) && run_id < sorted_linear_indices_num_runs[0];
             run_id += gridDim.x * blockDim.y) {

        const int64_t linear_index = sorted_linear_indices_run[run_id];
        const int32_t segment_start =
            sorted_linear_indices_cumulative_run_lengths[run_id];
        const int32_t segment_end =
            sorted_linear_indices_cumulative_run_lengths[run_id + 1];
        const int32_t SL = segment_end - segment_start;

        if (SL >= max_segment_length_per_warp) {
            continue;
        }

        // now, each segment corresponds to exactly one table `t` and row in
        // that table (`idx`). Thus, we can hoist out some of the book-keeping.
        const auto info_0 = reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start];
        const auto t_0 = info_0 >> info_B_num_bits;

        int64_t hash_size = hash_size_cumsum[t_0];
        int32_t D = D_offsets[t_0 + 1] - D_offsets[t_0];
        int64_t idx = linear_index - hash_size;

        const int32_t SL_per_warp = div_round_up(SL, blockDim.y);
        const int32_t sl_start = 0;
        const int32_t sl_end = SL;
        Vec4T<at::acc_type<cache_t, true>> grad_sum[kMaxVecsPerThread];
        for (int32_t sl = sl_start; sl < sl_end; sl += kThreadGroupSize) {
            int32_t sl_j = sl + threadIdx.x;
            const auto b_t = sl_j < sl_end ? reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start + sl_j] : 0;
            const auto b = b_t & info_B_mask;
            const auto t = b_t >> info_B_num_bits; // if vbe

            int32_t D_start = sl_j < sl_end ? D_offsets[t] : 0; // if vbe // if not nobag

            for (int32_t j = 0; j < kThreadGroupSize && sl + j < sl_end; ++j) {
                int32_t b_j = SHFL_SYNC(b, j);
                int32_t D_start_j = SHFL_SYNC(D_start, j);
                int32_t output_offset = b_j * buffer_width + D_start_j;

                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                        i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                        ++i) {
                    int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                    Vec4T<at::acc_type<grad_t, true>> grad_out_vec(
                        // &grad_output[b_j][0] + D_start_j + d
                        nvshmem_get_grad + output_offset + d
                    );
                    grad_sum[i].add_(grad_out_vec);

                }
            }
        }

        split_sgd_table_update_kernel
          <emb_t, cache_t, kMaxVecsPerThread, kThreadGroupSize, VEC_WIDTH>(
              dev_weights,
              uvm_weights,
              lxu_cache_weights,
              weights_placements,
              weights_offsets,
              sorted_lxu_cache_locations,
              grad_sum,
              stochastic_rounding,
              stochastic_rounding_philox_args,
              run_id,
              D,
              t_0,
              idx,
              segment_start,
              shfl_sync_mask,
              threadIdx.y * kMaxVecsPerThread * kThreadGroupSize, // shared_weight_offset
              learning_rate); // if not dense and optimizer != "none"

    }
}


Tensor nvshmem_alltoall_based_host_function(
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
    bool stochastic_rounding,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    // all-to-all information:
    float* nvshmem_grad,
    float* grad_get_buffer,
    Tensor dim_sum_per_rank_data,
    Tensor dim_offset_per_rank_data,
    int32_t total_dim_output,
    int32_t nranks,
    int32_t rank,
    int32_t local_batch_size,
    int64_t total_D,
    float learning_rate
    ) {

   TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
        dev_weights,
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        weights_offsets,
        D_offsets,
        hash_size_cumsum,
        indices,
        offsets,
        lxu_cache_locations,
        grad_output);

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(dev_weights.get_device());
    TORCH_CHECK(max_D <= 1024);
    // Set total_unique_indices to total num indices by default
    const auto total_unique_indices = indices.numel();

    // short-circuit if there are zero indices.
    if (indices.numel() == 0) {
        return Tensor();
    }
    int32_t T = D_offsets.numel() - 1;

    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]
    const auto total_B = offsets.size(0) - 1;
    const int32_t B = (total_B) / T; // local_batch_size_after_dist = local_batch_size * nDev
    TORCH_CHECK(total_B > 0);
    auto BT_block_size = kMaxThreads / kWarpSize;
    TORCH_CHECK(BT_block_size * kWarpSize <= kMaxThreads);

    // V100: 96 KB; A100: 160 KB; H100: 228 KB.
    int max_shared_bytes = 0;
#ifndef __HIP_PLATFORM_HCC__
    cudaDeviceGetAttribute(&max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_weights.get_device());
#else
    // MI100 has 64 KB local memory (shared memory) per workgroup
    max_shared_bytes = 64 << 10;
#endif
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int shared_kb = max_shared_bytes >> 10;
    // V100: 64 KB; A100: 96 KB; H100: 144 KB
#ifndef __HIP_PLATFORM_HCC__
    // Use 2/3 of the available GPU shared mem; leave rooms for L1$.
    int used_shared_kb = round_down(shared_kb * 2 / 3, 16);
    TORCH_CHECK(used_shared_kb > 0);
#else
    // MI100 has independent shared mem and L1
    int used_shared_kb = shared_kb;
#endif
    int used_shared_bytes = used_shared_kb << 10;

    Tensor linear_indices, linear_indices_sorted;
    Tensor infos_sorted;
    Tensor sorted_linear_indices_run, sorted_linear_indices_run_lengths,
        sorted_linear_indices_num_runs,
        sorted_linear_indices_cumulative_run_lengths;
    std::tie(
        linear_indices,
        linear_indices_sorted,
        infos_sorted,
        sorted_linear_indices_run,
        sorted_linear_indices_run_lengths,
        sorted_linear_indices_num_runs,
        sorted_linear_indices_cumulative_run_lengths) =
        transpose_embedding_input_local(
        // transpose_embedding_input(
            hash_size_cumsum,
            total_hash_size_bits,
            indices,
            offsets,
            false,
            c10::optional<Tensor>(),
            info_B_num_bits,
            info_B_mask,
            total_unique_indices);
    auto lxu_cache_locations_sorted = at::empty_like(lxu_cache_locations);
    if (lxu_cache_locations.size(0) > 0) {
        size_t temp_storage_bytes = 0;
        AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            // radix_sort_pairs(
            nullptr,
            temp_storage_bytes,
            linear_indices.data_ptr<int64_t>(),
            linear_indices_sorted.data_ptr<int64_t>(),
            lxu_cache_locations.data_ptr<int32_t>(),
            lxu_cache_locations_sorted.data_ptr<int32_t>(),
            linear_indices.numel(),
            0,
            total_hash_size_bits,
            at::cuda::getCurrentCUDAStream(),
            false));
        auto temp_storage = at::empty(
            {static_cast<int64_t>(temp_storage_bytes)},
            indices.options().dtype(at::kByte));
        AT_CUDA_CHECK( cub::DeviceRadixSort::SortPairs(
            // radix_sort_pairs(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            linear_indices.data_ptr<int64_t>(),
            linear_indices_sorted.data_ptr<int64_t>(),
            lxu_cache_locations.data_ptr<int32_t>(),
            lxu_cache_locations_sorted.data_ptr<int32_t>(),
            linear_indices.numel(),
            0,
            total_hash_size_bits,
            at::cuda::getCurrentCUDAStream(),
            false));
    }

    DISPATCH_EMB_GRAD_CACHE_TYPES(
        dev_weights.scalar_type(),
        grad_output.scalar_type(),
        lxu_cache_weights.scalar_type(),
            "split_embedding_backward_sgd_exact_kernel",
        [&] {

            // early memory release
            linear_indices.reset();
            linear_indices_sorted.reset();

            auto grad_output_accessor = MAKE_PTA_WITH_NAME("split_embedding_backward_codegen_sgd_unweighted_exact_cuda.1", grad_output, grad_t, 2, 64);
            Tensor grad_output_mean;
            if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN) {
              grad_output_mean = at::empty_like(grad_output);

#ifdef FBGEMM_GPU_MEMCHECK
              const auto func_name1 = "grad_mean_kernel";
#endif

              grad_mean_kernel<<<
                    div_round_up(total_B, kMaxThreads / kWarpSize),
                    dim3(kWarpSize, kMaxThreads / kWarpSize),
                    0,
                    at::cuda::getCurrentCUDAStream()>>>
                    (
                        MAKE_PTA_WITH_NAME(func_name1, grad_output_mean, grad_t, 2, 64),
                        MAKE_PTA_WITH_NAME(func_name1, grad_output, grad_t, 2, 64),
                        MAKE_PTA_WITH_NAME(func_name1, D_offsets, int32_t, 1, 32),
                        MAKE_PTA_WITH_NAME(func_name1, offsets, int64_t, 1, 32),
                        FixedDivisor(total_B / T)
                    );
              C10_CUDA_KERNEL_LAUNCH_CHECK(); // if not dense or not vbe


              grad_output_accessor = MAKE_PTA_WITH_NAME("split_embedding_backward_codegen_sgd_unweighted_exact_cuda.2", grad_output_mean, grad_t, 2, 64);
            }
            at::PhiloxCudaState rng_engine_inputs;
            if (stochastic_rounding && !std::is_same<emb_t, float>::value) {
                auto gen = at::cuda::detail::getDefaultCUDAGenerator();
                std::lock_guard<std::mutex> lock(gen.mutex());
                rng_engine_inputs =
                    at::check_generator<at::CUDAGeneratorImpl>(gen)
                        ->philox_cuda_state(4);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

// =====================================================================================================
#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name_0 = "nvshmem_bwd_alltoall";
#endif
                nvshmem_bwd_alltoall<kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kBackwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
                    FixedDivisor(B),
                    MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),

                    nvshmem_grad,
                    grad_get_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
                    max_D,
                    total_dim_output,
                    nranks,
                    rank,
                    total_D,
                    local_batch_size
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
// =====================================================================================================

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_alltoall_based_cta_per_row_kernel";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_alltoall_based_cta_per_row_kernel<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    grad_get_buffer,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_alltoall_based_warp_per_row_kernel";
#endif

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_alltoall_based_warp_per_row_kernel<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    grad_get_buffer,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));
// =====================================================================================================
#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name_0 = "nvshmem_bwd_alltoall";
#endif
                nvshmem_bwd_alltoall<kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kBackwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
                    FixedDivisor(B),
                    MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),

                    nvshmem_grad,
                    grad_get_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
                    max_D,
                    total_dim_output,
                    nranks,
                    rank,
                    total_D,
                    local_batch_size
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
// =====================================================================================================

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_alltoall_based_cta_per_row_kernel";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_alltoall_based_cta_per_row_kernel<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    grad_get_buffer,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_alltoall_based_warp_per_row_kernel";
#endif

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_alltoall_based_warp_per_row_kernel<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    grad_get_buffer,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

// =====================================================================================================
#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name_0 = "nvshmem_bwd_alltoall";
#endif
                nvshmem_bwd_alltoall<kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kBackwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
                    FixedDivisor(B),
                    MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),

                    nvshmem_grad,
                    grad_get_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
                    max_D,
                    total_dim_output,
                    nranks,
                    rank,
                    total_D,
                    local_batch_size
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
// =====================================================================================================

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_alltoall_based_cta_per_row_kernel";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_alltoall_based_cta_per_row_kernel<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    grad_get_buffer,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_alltoall_based_warp_per_row_kernel";
#endif

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_alltoall_based_warp_per_row_kernel<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    grad_get_buffer,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

// =====================================================================================================
#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name_0 = "nvshmem_bwd_alltoall";
#endif
                nvshmem_bwd_alltoall<kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kBackwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
                    FixedDivisor(B),
                    MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),

                    nvshmem_grad,
                    grad_get_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
                    max_D,
                    total_dim_output,
                    nranks,
                    rank,
                    total_D,
                    local_batch_size
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
// =====================================================================================================

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_alltoall_based_cta_per_row_kernel";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_alltoall_based_cta_per_row_kernel<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    grad_get_buffer,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_alltoall_based_warp_per_row_kernel";
#endif

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_alltoall_based_warp_per_row_kernel<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    grad_get_buffer,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));
// =====================================================================================================
#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name_0 = "nvshmem_bwd_alltoall";
#endif
                nvshmem_bwd_alltoall<kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kBackwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
                    FixedDivisor(B),
                    MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),

                    nvshmem_grad,
                    grad_get_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
                    max_D,
                    total_dim_output,
                    nranks,
                    rank,
                    total_D,
                    local_batch_size
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
// =====================================================================================================

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_alltoall_based_cta_per_row_kernel";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_alltoall_based_cta_per_row_kernel<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    grad_get_buffer,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_alltoall_based_warp_per_row_kernel";
#endif

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_alltoall_based_warp_per_row_kernel<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    grad_get_buffer,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));
// =====================================================================================================
#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name_0 = "nvshmem_bwd_alltoall";
#endif
                nvshmem_bwd_alltoall<kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kBackwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
                    FixedDivisor(B),
                    MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),

                    nvshmem_grad,
                    grad_get_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
                    max_D,
                    total_dim_output,
                    nranks,
                    rank,
                    total_D,
                    local_batch_size
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
// =====================================================================================================

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_alltoall_based_cta_per_row_kernel";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_alltoall_based_cta_per_row_kernel<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    grad_get_buffer,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_alltoall_based_warp_per_row_kernel";
#endif

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_alltoall_based_warp_per_row_kernel<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    grad_get_buffer,
                    learning_rate);
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
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

// =====================================================================================================
#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name_0 = "nvshmem_bwd_alltoall";
#endif
                nvshmem_bwd_alltoall<kMaxVecsPerThread, kThreadGroupSize><<<
                    div_round_up(total_B, kBackwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
                    FixedDivisor(B),
                    MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),

                    nvshmem_grad,
                    grad_get_buffer,
                    MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
                    max_D,
                    total_dim_output,
                    nranks,
                    rank,
                    total_D,
                    local_batch_size
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
// =====================================================================================================

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name3 = "nvshmem_alltoall_based_cta_per_row_kernel";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            nvshmem_alltoall_based_cta_per_row_kernel<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), // if optimizer != "none"
                    MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations_sorted, int32_t, 1, 32),
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                    MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    grad_get_buffer,
                    learning_rate);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name4 = "nvshmem_alltoall_based_warp_per_row_kernel";
#endif

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            nvshmem_alltoall_based_warp_per_row_kernel<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations_sorted, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    stochastic_rounding,
                    rng_engine_inputs, // if not dense and optimizer != "none"
                    info_B_num_bits,
                    info_B_mask,
                    grad_get_buffer,
                    learning_rate);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
        });
    return Tensor();
}
