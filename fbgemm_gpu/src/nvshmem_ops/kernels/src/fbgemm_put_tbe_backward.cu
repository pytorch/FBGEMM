#include "fbgemm_put_tbe_backward.cuh"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

////////////////////////////////////////////////////////////////////////////////
// Helper Functions
////////////////////////////////////////////////////////////////////////////////

std::tuple<int32_t, uint32_t> adjust_info_B_num_bits(
    int32_t B,
    int32_t T) {
  int32_t info_B_num_bits = DEFAULT_INFO_B_NUM_BITS;
  uint32_t info_B_mask = DEFAULT_INFO_B_MASK;
  uint32_t max_T = MAX_T;
  uint32_t max_B = MAX_B;
  bool invalid_T = T > max_T;
  bool invalid_B = B > max_B;

  TORCH_CHECK(
      !(invalid_T && invalid_B),
      "Not enough infos bits to accommodate T and B. Default num bits = ",
      DEFAULT_INFO_NUM_BITS);

  if (invalid_T) {
    // Reduce info_B_num_bits
    while (invalid_T && !invalid_B && info_B_num_bits > 0) {
      info_B_num_bits--;
      max_T = ((max_T + 1) << 1) - 1;
      max_B = ((max_B + 1) >> 1) - 1;
      invalid_T = T > max_T;
      invalid_B = B > max_B;
    }
  } else if (invalid_B) {
    // Increase info_B_num_bits
    while (!invalid_T && invalid_B && info_B_num_bits < DEFAULT_INFO_NUM_BITS) {
      info_B_num_bits++;
      max_T = ((max_T + 1) >> 1) - 1;
      max_B = ((max_B + 1) << 1) - 1;
      invalid_T = T > max_T;
      invalid_B = B > max_B;
    }
  }

  TORCH_CHECK(
      !invalid_T && !invalid_B,
      "Not enough infos bits to accommodate T and B. Default num bits = ",
      DEFAULT_INFO_NUM_BITS);

  // Recompute info_B_mask using new info_B_num_bits
  info_B_mask = (1u << info_B_num_bits) - 1;

  return {info_B_num_bits, info_B_mask};
}

inline at::Tensor asynchronous_complete_cumsum(at::Tensor t_in) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(t_in.get_device());
  size_t temp_storage_bytes = 0;
  TORCH_CHECK(t_in.is_contiguous());
  TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong);
  // CUB only handles up to INT_MAX elements.
  TORCH_CHECK_LT(t_in.numel(), std::numeric_limits<int32_t>::max());
  TORCH_CHECK_EQ(t_in.dim(), 1);
  auto t_out = at::empty({t_in.numel() + 1}, t_in.options());
  t_out[0].zero_();
  AT_DISPATCH_INDEX_TYPES(
      t_in.scalar_type(), "cub_inclusive_sum_wrapper1", [&] {
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
            nullptr,
            temp_storage_bytes,
            t_in.data_ptr<index_t>(),
            t_out.data_ptr<index_t>() + 1,
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      });
  auto temp_storage = at::empty(
      {static_cast<int64_t>(temp_storage_bytes)},
      t_in.options().dtype(at::kByte));
  AT_DISPATCH_INDEX_TYPES(
      t_in.scalar_type(), "cub_inclusive_sum_wrapper2", [&] {
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            t_in.data_ptr<index_t>(),
            t_out.data_ptr<index_t>() + 1,
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      });
  return t_out;
}

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


std::tuple<
    Tensor /*linear_indices*/,
    Tensor /*linear_indices_sorted*/,
    Tensor /*infos_sorted*/,
    Tensor /*sorted_linear_indices_run*/,
    Tensor /*sorted_linear_indices_run_lengths*/,
    Tensor /*sorted_linear_indices_num_runs*/,
    Tensor /*sorted_linear_indices_cumulative_run_lengths*/>
transpose_embedding_input_local(
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    bool nobag,
    const c10::optional<Tensor>& vbe_b_t_map,
    const int64_t info_B_num_bits,
    const int64_t info_B_mask,
    const int64_t total_unique_indices) {

  const bool vbe = vbe_b_t_map.has_value();
  TORCH_CHECK(nobag || !vbe || info_B_num_bits > 0);
  TORCH_CHECK(!vbe || info_B_mask > 0);

  const auto total_B = offsets.size(0) - 1;
  const auto T = hash_size_cumsum.size(0) - 1;

  auto infos = at::empty_like(
      indices, indices.options().dtype(nobag ? at::kLong : at::kInt));
  auto infos_sorted = at::empty_like(infos);
  auto linear_indices = at::empty_like(indices);
  auto linear_indices_sorted = at::empty_like(indices);

  Tensor sorted_linear_indices_run;
  Tensor sorted_linear_indices_run_lengths;
  Tensor sorted_linear_indices_num_runs;

  using at::RestrictPtrTraits;

#define INVOKE_LINEARIZE_INDEX_KERNEL(INFO_ACC_T, NOBAG)                   \
  const auto linearize_index_kernel_ =                                     \
      (vbe ? linearize_index_kernel<index_t, INFO_ACC_T, NOBAG, true>      \
           : linearize_index_kernel<index_t, INFO_ACC_T, NOBAG, false>);   \
  linearize_index_kernel_<<<                                               \
      div_round_up(total_B, kMaxThreads),                                  \
      kMaxThreads,                                                         \
      0,                                                                   \
      at::cuda::getCurrentCUDAStream()>>>(                                 \
      hash_size_cumsum.packed_accessor32<index_t, 1, RestrictPtrTraits>(), \
      indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),          \
      offsets.packed_accessor32<index_t, 1, RestrictPtrTraits>(),          \
      infos.packed_accessor32<INFO_ACC_T, 1, RestrictPtrTraits>(),         \
      linear_indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),   \
      info_B_num_bits,                                                     \
      info_B_mask,                                                         \
      (1u << (DEFAULT_INFO_NUM_BITS - info_B_num_bits)) - 1,               \
      (1u << info_B_num_bits) - 1,                                         \
      vbe ? reinterpret_cast<uint32_t*>(vbe_b_t_map.value().data_ptr())    \
          : nullptr,                                                       \
      FixedDivisor(total_B / T));                                          \
  C10_CUDA_KERNEL_LAUNCH_CHECK()

  AT_DISPATCH_INDEX_TYPES(
      infos.scalar_type(), "transpose_embedding_input1", [&] {
        using info_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "transpose_embedding_input2", [&] {
              if (!nobag) {
                INVOKE_LINEARIZE_INDEX_KERNEL(int32_t, false);
              } else {
                INVOKE_LINEARIZE_INDEX_KERNEL(int64_t, true);
              }
              {
                size_t temp_storage_bytes = 0;
                AT_CUDA_CHECK(
                    FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
                        nullptr,
                        temp_storage_bytes,
                        linear_indices.data_ptr<index_t>(),
                        linear_indices_sorted.data_ptr<index_t>(),
                        infos.data_ptr<info_t>(),
                        infos_sorted.data_ptr<info_t>(),
                        linear_indices.numel(),
                        0,
                        total_hash_size_bits,
                        at::cuda::getCurrentCUDAStream(),
                        false));
                auto temp_storage = at::empty(
                    {static_cast<int64_t>(temp_storage_bytes)},
                    indices.options().dtype(at::kByte));
                AT_CUDA_CHECK(
                    FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
                        temp_storage.data_ptr(),
                        temp_storage_bytes,
                        linear_indices.data_ptr<index_t>(),
                        linear_indices_sorted.data_ptr<index_t>(),
                        infos.data_ptr<info_t>(),
                        infos_sorted.data_ptr<info_t>(),
                        linear_indices.numel(),
                        0,
                        total_hash_size_bits,
                        at::cuda::getCurrentCUDAStream(),
                        false));
              }
            //   std::cout<< "linear_indices:" << linear_indices << std::endl;
            //   std::cout<< "linear_indices_sorted:" << linear_indices_sorted << std::endl;
            //   std::cout<< "infos:" << infos << std::endl;
            //   std::cout<< "infos_sorted:" << infos_sorted << std::endl;

              if (total_unique_indices != -1) {
                TORCH_CHECK(total_unique_indices >= 0);
                sorted_linear_indices_run =
                    at::empty({total_unique_indices}, indices.options());
                sorted_linear_indices_run_lengths = at::zeros(
                    {total_unique_indices}, indices.options().dtype(at::kInt));
              } else {
                sorted_linear_indices_run = at::empty_like(indices);
                sorted_linear_indices_run_lengths =
                    at::zeros_like(indices, indices.options().dtype(at::kInt));
              }
              sorted_linear_indices_num_runs =
                  at::zeros({1}, indices.options().dtype(at::kInt));

              {
                size_t temp_storage_bytes = 0;
                AT_CUDA_CHECK(
                    FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRunLengthEncode::Encode(
                        nullptr,
                        temp_storage_bytes,
                        linear_indices_sorted.data_ptr<index_t>(),
                        sorted_linear_indices_run.data_ptr<index_t>(),
                        sorted_linear_indices_run_lengths.data_ptr<int32_t>(),
                        sorted_linear_indices_num_runs.data_ptr<int32_t>(),
                        linear_indices_sorted.numel(),
                        at::cuda::getCurrentCUDAStream()));
                // Allocate temporary storage
                auto temp_storage = at::empty(
                    {static_cast<int64_t>(temp_storage_bytes)},
                    indices.options().dtype(at::kByte));
                // Run encoding
                AT_CUDA_CHECK(
                    FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRunLengthEncode::Encode(
                        temp_storage.data_ptr(),
                        temp_storage_bytes,
                        linear_indices_sorted.data_ptr<index_t>(), // d_input: 1,1,1,2,2,3,3,3,3,4,5,5,5,6
                        sorted_linear_indices_run.data_ptr<index_t>(), // d_unique_out: 1,2,3,4,5,6
                        sorted_linear_indices_run_lengths.data_ptr<int32_t>(), // d_counts_out: 3,2,4,1,3,1
                        sorted_linear_indices_num_runs.data_ptr<int32_t>(), // d_num_runs_out: 6
                        linear_indices_sorted.numel(),
                        at::cuda::getCurrentCUDAStream()));
              }
            });
      });

  auto sorted_linear_indices_cumulative_run_lengths =
      asynchronous_complete_cumsum(sorted_linear_indices_run_lengths);

#undef INVOKE_LINEARIZE_INDEX_KERNEL

  return {
      linear_indices,
      linear_indices_sorted,
      infos_sorted,
      sorted_linear_indices_run,
      sorted_linear_indices_run_lengths,
      sorted_linear_indices_num_runs,
      sorted_linear_indices_cumulative_run_lengths};
}

////////////////////////////////////////////////////////////////////////////////
// FBGEMM Kernels
////////////////////////////////////////////////////////////////////////////////

template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize >
__global__ __launch_bounds__(kMaxThreads) void
split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1(
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

                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                    i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                    int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                    Vec4T<at::acc_type<grad_t, true>> grad_out_vec(
                        &grad_output[b_j][0] + D_start_j + d
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

        // ========= Write grad test =========
        // const int64_t weights_offset = long_run_id * D;
        // idx = 0;
    	// #pragma unroll kMaxVecsPerThread
        // for (int32_t i = 0;
        //     i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
        //     ++i) {
        //     int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
        //     auto& grad = grad_sum[i];
        //     grad.store(&dev_weights[weights_offset + idx * D + d]);
        // } // if not dense and optimizer != "none"
        // ========= Write grad test =========

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
split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1(
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

    for (uint32_t run_id = start_run_id;
         run_id < sorted_linear_indices_run.size(0) && run_id < sorted_linear_indices_num_runs[0];
             run_id += gridDim.x * blockDim.y) {

        const int64_t linear_index = sorted_linear_indices_run[run_id];
        const int32_t segment_start =
            sorted_linear_indices_cumulative_run_lengths[run_id];
        const int32_t segment_end =
            sorted_linear_indices_cumulative_run_lengths[run_id + 1];
        const int32_t SL = segment_end - segment_start;

        // if( start_run_id == 0 && threadIdx.x == 0){
        //     printf("First %d,%d,%d,%d\n", sorted_linear_indices_cumulative_run_lengths[0], sorted_linear_indices_cumulative_run_lengths[1],
        //     sorted_linear_indices_cumulative_run_lengths[2],sorted_linear_indices_cumulative_run_lengths[3]);
        //     printf("First %d,%d, SL:%d, kThreadGroupSize:%d\n", sorted_linear_indices_cumulative_run_lengths[run_id], sorted_linear_indices_cumulative_run_lengths[run_id+1], SL, kThreadGroupSize);
        // }

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

            // if( start_run_id == 0 && threadIdx.x == 0){
            //     printf("info_B_mask:%d, info_B_num_bits:%d, b_t:%d\n", info_B_mask, info_B_num_bits, b_t);
            // }
            int32_t D_start = sl_j < sl_end ? D_offsets[t] : 0; // if vbe // if not nobag

            for (int32_t j = 0; j < kThreadGroupSize && sl + j < sl_end; ++j) {
                int32_t b_j = SHFL_SYNC(b, j);
                int32_t D_start_j = SHFL_SYNC(D_start, j);

                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                        i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                        ++i) {
                    int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                    Vec4T<at::acc_type<grad_t, true>> grad_out_vec(
                        &grad_output[b_j][0] + D_start_j + d
                    );
                    grad_sum[i].add_(grad_out_vec);

                }
            }
        }

        // ========= Write grad test =========
        // const int64_t weights_offset = run_id * D;
        // idx = 0;
    	// #pragma unroll kMaxVecsPerThread
        // for (int32_t i = 0;
        //     i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
        //     ++i) {
        //     int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
        //     auto& grad = grad_sum[i];
        //     grad.store(&dev_weights[weights_offset + idx * D + d]);
        // } // if not dense and optimizer != "none"
        // ========= Write grad test =========


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


__global__ __launch_bounds__(kMaxThreads) void
split_embedding_backward_codegen_find_long_segments(
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_num_runs,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_run_lengths,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        long_run_ids,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        num_long_run_ids,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        long_run_id_to_really_long_run_ids,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        num_really_long_run_ids,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        grad_accum_counter,
    const int32_t max_segment_length_per_warp,
    const int32_t max_segment_length_per_cta,
    const bool use_deterministic_algorithms) {
  const int32_t num_runs = sorted_linear_indices_num_runs[0];
  for (auto run_id = blockIdx.x * blockDim.x + threadIdx.x; run_id < num_runs; run_id += blockDim.x * gridDim.x) {
    if (sorted_linear_indices_run_lengths[run_id] >= max_segment_length_per_warp) {
        // A segment with length > max_segment_length_per_cta is handled by more than 1 thread block.
        const int num_ctas_for_run =
            use_deterministic_algorithms ? 1 : div_round_up(sorted_linear_indices_run_lengths[run_id], max_segment_length_per_cta);
        const auto long_run_idx = gpuAtomicAdd(&num_long_run_ids[0], num_ctas_for_run);
        // The first thread block in the really long run gets run_id in long_run_ids
        // and the rest get the negative of its offset.
        long_run_ids[long_run_idx] = run_id;
        for (int i = 1; i < num_ctas_for_run; ++i) {
            long_run_ids[long_run_idx + i] = -i;
        }
        if (num_ctas_for_run > 1) {
            const auto really_long_run_idx = gpuAtomicAdd(&num_really_long_run_ids[0], 1);
            grad_accum_counter[really_long_run_idx] = num_ctas_for_run;
            for (int i = 0; i < num_ctas_for_run; ++i) {
                long_run_id_to_really_long_run_ids[long_run_idx + i] = really_long_run_idx;
            }
        }
    }
  }
}


template <typename grad_t>
__global__ __launch_bounds__(kMaxThreads) void grad_mean_kernel(
    pta::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits>
        grad_output_mean,
    const pta::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits>
        grad_output,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,

    FixedDivisor fd_B

) {
  int32_t T = D_offsets.size(0) - 1;
  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t b;
  int32_t t;
  const auto total_B = offsets.size(0) - 1;

  if (b_t >= total_B) {
    return;
  }

  fd_B.DivMod(b_t, &t, &b);


  int32_t D_start = D_offsets[t];
  int32_t D_end = D_offsets[t + 1];
  int32_t D = D_end - D_start;
  int64_t indices_start = offsets[b_t];
  int64_t indices_end = offsets[b_t + 1];
  int32_t L = indices_end - indices_start;


  const auto grad_offset = D_start;
  const auto grad_outer_offset = b;


  const grad_t* shifted_grad_output = &grad_output[grad_outer_offset][grad_offset];
  grad_t* shifted_grad_output_mean = &grad_output_mean[grad_outer_offset][grad_offset];

  if (L != 0) {
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<grad_t> grad_out_vec(&shifted_grad_output[d * 4]);
      grad_out_vec.mul_(1.0 / L);
      grad_out_vec.store(&shifted_grad_output_mean[d * 4]);
    }
  } else {
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<grad_t> grad_out_vec(&shifted_grad_output[d * 4]);
      grad_out_vec.store(&shifted_grad_output_mean[d * 4]);
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
// Operator Code
////////////////////////////////////////////////////////////////////////////////

Tensor split_embedding_backward_codegen_sgd_unweighted_exact_cuda(
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
            const auto func_name3 = "split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
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
            const auto func_name4 = "split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
#endif

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
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
                    learning_rate);
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
            const auto func_name3 = "split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
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
            const auto func_name4 = "split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
#endif

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
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
                    learning_rate);
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
            const auto func_name3 = "split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
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
            const auto func_name4 = "split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
#endif

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
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
            const auto func_name3 = "split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
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
            const auto func_name4 = "split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
#endif

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
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
            const auto func_name3 = "split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
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
            const auto func_name4 = "split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
#endif

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
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
            const auto func_name3 = "split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
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
            const auto func_name4 = "split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
#endif

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
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
            const auto func_name3 = "split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
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
            const auto func_name4 = "split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
#endif

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
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
            const auto func_name3 = "split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
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
            const auto func_name4 = "split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
#endif

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
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
            const auto func_name3 = "split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
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
            const auto func_name4 = "split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
#endif

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
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
            const auto func_name3 = "split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1";
#endif

            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            split_embedding_backward_codegen_sgd_unweighted_kernel_cta_per_row_1<
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
            const auto func_name4 = "split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1";
#endif

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            split_embedding_backward_codegen_sgd_unweighted_kernel_warp_per_row_1<
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
                    learning_rate);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
        });
    return Tensor();
}
