#include "fbgemm_put_tbe_backward.cuh"
#include "fbgemm_gpu/split_embeddings_utils.cuh"
#include "nvshmem_decoupling_backward.cuh"


#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
#define DISPATCH_OPTIMAL_KERNEL(MAX_D, ...)                                    \
  [&] {                                                                        \
    if (MAX_D <= 32) {              \
      constexpr int kMaxVecsPerThread = 1 / 4 >= 1 ? 1 / 4 : 1;            \
      constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 1, 1);                           \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    if (MAX_D <= 64) {              \
      constexpr int kMaxVecsPerThread = 2 / 4 >= 1 ? 2 / 4 : 1;            \
      constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 2, 1);                           \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    if (MAX_D <= 128) {              \
      constexpr int kMaxVecsPerThread = 4 / 4 >= 1 ? 4 / 4 : 1;            \
      constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 4, 1);                           \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    if (MAX_D <= 256) {              \
      constexpr int kMaxVecsPerThread = 8 / 4 >= 1 ? 8 / 4 : 1;            \
      constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 8, 1);                           \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    if (MAX_D <= 384) {              \
      constexpr int kMaxVecsPerThread = 12 / 4 >= 1 ? 12 / 4 : 1;            \
      constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 12, 1);                           \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    if (MAX_D <= 512) {              \
      constexpr int kMaxVecsPerThread = 16 / 4 >= 1 ? 16 / 4 : 1;            \
      constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 16, 1);                           \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    if (MAX_D <= 640) {              \
      constexpr int kMaxVecsPerThread = 20 / 4 >= 1 ? 20 / 4 : 1;            \
      constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 20, 1);                           \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    if (MAX_D <= 768) {              \
      constexpr int kMaxVecsPerThread = 24 / 4 >= 1 ? 24 / 4 : 1;            \
      constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 24, 1);                           \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    if (MAX_D <= 896) {              \
      constexpr int kMaxVecsPerThread = 28 / 4 >= 1 ? 28 / 4 : 1;            \
      constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 28, 1);                           \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    if (MAX_D <= 1024) {              \
      constexpr int kMaxVecsPerThread = 32 / 4 >= 1 ? 32 / 4 : 1;            \
      constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 32, 1);                           \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    return;                                                                    \
  }()

#else
#define DISPATCH_OPTIMAL_KERNEL(MAX_D, ...)                                    \
  [&] {                                                                        \
    constexpr int kThreadGroupSize = kWarpSize;                                \
    if (MAX_D <= 32) {              \
      constexpr int kMaxVecsPerThread = 1 / 4 >= 1 ? 1 / 4 : 1;            \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    if (MAX_D <= 64) {              \
      constexpr int kMaxVecsPerThread = 2 / 4 >= 1 ? 2 / 4 : 1;            \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    if (MAX_D <= 128) {              \
      constexpr int kMaxVecsPerThread = 4 / 4 >= 1 ? 4 / 4 : 1;            \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    if (MAX_D <= 256) {              \
      constexpr int kMaxVecsPerThread = 8 / 4 >= 1 ? 8 / 4 : 1;            \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    if (MAX_D <= 384) {              \
      constexpr int kMaxVecsPerThread = 12 / 4 >= 1 ? 12 / 4 : 1;            \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    if (MAX_D <= 512) {              \
      constexpr int kMaxVecsPerThread = 16 / 4 >= 1 ? 16 / 4 : 1;            \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    if (MAX_D <= 640) {              \
      constexpr int kMaxVecsPerThread = 20 / 4 >= 1 ? 20 / 4 : 1;            \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    if (MAX_D <= 768) {              \
      constexpr int kMaxVecsPerThread = 24 / 4 >= 1 ? 24 / 4 : 1;            \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    if (MAX_D <= 896) {              \
      constexpr int kMaxVecsPerThread = 28 / 4 >= 1 ? 28 / 4 : 1;            \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    if (MAX_D <= 1024) {              \
      constexpr int kMaxVecsPerThread = 32 / 4 >= 1 ? 32 / 4 : 1;            \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    return;                                                                    \
  }()

#endif


// ////// =================================================================================================================================
// ////// =================================================================================================================================
// ////// =================================================================================================================================
// template <
//     typename emb_t,
//     typename cache_t,
//     size_t kMaxVecsPerThread,
//     int32_t kThreadGroupSize = kWarpSize,
//     int32_t VEC_WIDTH
// >
// DEVICE_INLINE void split_sgd_table_update_kernel(
//     pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits>& dev_weights,
//     pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits>& uvm_weights,
//     pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>& lxu_cache_weights,
//     const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& weights_placements,
//     const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>& weights_offsets,
//     const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& sorted_lxu_cache_locations,
//     Vec4T<at::acc_type<cache_t, true>>* grad_sum,
//     const bool stochastic_rounding,
//     const at::PhiloxCudaState& stochastic_rounding_philox_args,
//     const uint32_t run_id,
//     const int32_t D,
//     const int32_t t,
//     const int64_t idx,
//     const int32_t segment_start,
//     const uint32_t shfl_sync_mask,
//     const int32_t shared_weight_offset,
//     float learning_rate = 0
// ) {
//     constexpr auto is_int8 = std::is_same<emb_t, uint8_t>::value;
//     const int64_t weights_offset = weights_offsets[t];
//     emb_t* __restrict__ weights {nullptr};
//     cache_t* __restrict__ cache_weights {nullptr};
//     int32_t D_emb = D;
//     if (is_int8) {
//         D_emb += kINT8QparamsBytes;
//     }
//     const auto weights_placement = static_cast<PlacementType>(weights_placements[t]);
//     if (weights_placement == PlacementType::DEVICE) {
//         weights = &dev_weights[weights_offset + idx * D_emb];
//     } else {
//         weights = &uvm_weights[weights_offset + idx * D_emb];
//     }

//     struct SharedMemory<Vec4T<at::acc_type<cache_t, true>>> weight_update_buffer;
//     Vec4T<at::acc_type<cache_t, true>>* shared_weight_update_row =
//         is_int8 ? weight_update_buffer.getPointer() : nullptr;
//     auto weight_row_template =
//         WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(
//             weights, cache_weights, D, nullptr);

//     weight_row_template.set_stochastic_rounding(
//       stochastic_rounding,
//       stochastic_rounding_philox_args,
//       threadIdx.x + run_id * blockDim.x
//     );

//     float2 qparams_template;
//     if (is_int8 && !cache_weights) {
//         qparams_template = weight_row_template.load_qparams();
//     }



//     float2 qparams_new;
// #pragma unroll kMaxVecsPerThread
//     for (int32_t i = 0;
//         i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
//         ++i) {
//         int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
//         Vec4T<at::acc_type<cache_t, true>> weight_new = weight_row_template.load(d, qparams_template);
//         auto& grad = grad_sum[i];

//       weight_new.fma_(grad, -learning_rate);

//         if (is_int8 && !cache_weights) {
//             shared_weight_update_row[
//                 threadIdx.x + (i * kThreadGroupSize) + shared_weight_offset] = weight_new;
//         } else {
//             // qparams_new not used if type is not int8
//             weight_row_template.store(weight_new, d, qparams_new);
//         }
//     }

//     if (is_int8 && !cache_weights) {
//         // Calculate new qparams after row update
//         qparams_new = thrust_find_qparams<at::acc_type<cache_t, true>>(
//             &shared_weight_update_row[shared_weight_offset], D);
//         weight_row_template.store_qparams(qparams_new);

//         // Fetch cached updated row from shared mem and quantize on-the-fly
//         // when saving to lowp embedding
// #pragma unroll kMaxVecsPerThread
//         for (int32_t i = 0;
//             i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
//             ++i) {
//             const int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
//             weight_row_template.store(
//                 shared_weight_update_row[threadIdx.x + (i * kThreadGroupSize) + shared_weight_offset],
//                 d,
//                 qparams_new);
//         }
//     }
// }


// template <
//     typename emb_t,
//     typename cache_t,
//     size_t kMaxVecsPerThread,
//     int32_t kThreadGroupSize = kWarpSize,
//     int32_t VEC_WIDTH
// >
// __global__ __launch_bounds__(kMaxThreads)
// void sgd_decoupling_update_kernel(
//     at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
//     at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
//     at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
//     const at::PackedTensorAccessor32<emb_t, 1, at::RestrictPtrTraits> grad_dev_weights,
//     // grad_dev_indices is equivalent to sorted_linear_indices_run
//     const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> grad_dev_indices,
//     const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
//     const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
//     const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_lxu_cache_locations,
//     const int32_t max_D,
//     bool stochastic_rounding,
//     at::PhiloxCudaState stochastic_rounding_philox_args,
//     const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
//     const int64_t max_hash_size,
//     float learning_rate
// ){
//     const auto run_id = blockIdx.x * blockDim.y + threadIdx.y;
//     if (run_id >= grad_dev_indices.size(0)) {
//       return;
//     }

// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//     const unsigned int shfl_sync_mask =
//         ((1L << kThreadGroupSize) - 1) <<
//         (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
// #else
//     const unsigned int shfl_sync_mask = 0xffffffffu;
// #endif

//     Vec4T<at::acc_type<cache_t, true>> grad_sum[kMaxVecsPerThread];

//     int64_t idx = grad_dev_indices[run_id];
//     int64_t t = idx / max_hash_size;
//     int64_t t_offset = idx % max_hash_size;
//     int64_t D =  D_offsets[t + 1] - D_offsets[t];

//     // ============================== Atomic update ================================
//     // int64_t weights_offset = weights_offsets[t];
//     // int64_t weight_offset_emb =  weights_offset + t_offset * D;
//     // int32_t shared_d_offset = run_id * max_D;
//     // for(int32_t i = threadIdx.x; i < D; i+=kThreadGroupSize){
//     //     // gpuAtomicAdd(&dev_weights[weight_offset_emb + i], grad_dev_weights[shared_d_offset + i] * (-1) * learning_rate);
//     //     // gpuAtomicAdd(&dev_weights[weight_offset_emb + i], grad_dev_weights[shared_d_offset + i]);

//     //     float val = dev_weights[weight_offset_emb + i];
//     //     float val_2 = grad_dev_weights[shared_d_offset + i] * (-1) * learning_rate;
//     //     dev_weights[weight_offset_emb + i] = val + val_2;

//     //     // dev_weights[weight_offset_emb + i] = 0.01;

//     //     // dev_weights[weight_offset_emb + i] = val + grad_dev_weights[shared_d_offset + i] * (-1) * learning_rate;
//     // }
//     // ============================== Atomic update ================================

//     // ============================ FBGEMM SGD update ==============================
//     // Load grad_dev_weights into grad_sum

//     #pragma unroll kMaxVecsPerThread
//         for (int32_t i = 0;
//             i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
//             ++i) {
//             int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
//             grad_sum[i].load(&grad_dev_weights[run_id * max_D + d]);
//         }

//         split_sgd_table_update_kernel
//             <emb_t, cache_t, kMaxVecsPerThread, kThreadGroupSize, VEC_WIDTH>(
//                 dev_weights,
//                 uvm_weights,
//                 lxu_cache_weights,
//                 weights_placements,
//                 weights_offsets,
//                 sorted_lxu_cache_locations,
//                 grad_sum,
//                 stochastic_rounding,
//                 stochastic_rounding_philox_args,
//                 run_id,
//                 D,
//                 t, // t
//                 t_offset, // idx
//                 0, // segment_start (not used right now because lxu_cache is not
//                  // supported)
//                 shfl_sync_mask,
//                 0, // shared_weight_offset (not used because shared memory is not
//                  // needed as uint8_t is not supported)
//                 learning_rate); // if not dense and optimizer != "none"
//     // ============================ FBGEMM SGD update ==============================
// }

// void sgd_decoupling_update_host(
//     Tensor& dev_weights,
//     Tensor& uvm_weights,
//     Tensor& lxu_cache_weights,
//     const Tensor& grad_dev_weights,
//     const Tensor& grad_dev_indices,
//     const Tensor& weights_placements,
//     const Tensor& weights_offsets,
//     const int64_t max_D,
//     const bool stochastic_rounding,
//     const int64_t max_hash_size,
//     Tensor D_offsets,
//     const float learning_rate
// ){
//     TENSOR_ON_CUDA_GPU(dev_weights);
//     TENSOR_ON_CUDA_GPU(uvm_weights);
//     TENSOR_ON_CUDA_GPU(lxu_cache_weights);
//     TENSOR_ON_CUDA_GPU(grad_dev_weights);
//     TENSOR_ON_CUDA_GPU(grad_dev_indices);
//     TENSOR_ON_CUDA_GPU(weights_placements);
//     TENSOR_ON_CUDA_GPU(weights_offsets);


//     TENSORS_ON_SAME_DEVICE(dev_weights, uvm_weights);
//     TENSORS_ON_SAME_DEVICE(dev_weights, lxu_cache_weights);
//     TENSORS_ON_SAME_DEVICE(dev_weights, grad_dev_weights);
//     TENSORS_ON_SAME_DEVICE(dev_weights, grad_dev_indices);
//     TENSORS_ON_SAME_DEVICE(dev_weights, weights_placements);
//     TENSORS_ON_SAME_DEVICE(dev_weights, weights_offsets);

//     if (grad_dev_indices.numel() == 0) {
//         return;
//     }
//     at::cuda::OptionalCUDAGuard device_guard;
//     device_guard.set_index(dev_weights.get_device());

//     // Flatten dev_weights because it is currrently 2D
//     dev_weights = dev_weights.flatten();
//     const auto& flatten_grad_dev_weights = grad_dev_weights.flatten();
//     const auto& flatten_grad_dev_indices = grad_dev_indices.flatten();

//     DISPATCH_EMB_CACHE_TYPES(
//         dev_weights.scalar_type(),
//         lxu_cache_weights.scalar_type(),
//         "sgd_decoupling_update_kernel",
//         [&] {
//             TORCH_CHECK(!(std::is_same<emb_t, uint8_t>::value));

//             at::PhiloxCudaState rng_engine_inputs;
//             if (stochastic_rounding && !std::is_same<emb_t, float>::value) {
//                 auto gen = at::cuda::detail::getDefaultCUDAGenerator();
//                 std::lock_guard<std::mutex> lock(gen.mutex());
//                 rng_engine_inputs =
//                     at::check_generator<at::CUDAGeneratorImpl>(gen)
//                         ->philox_cuda_state(4);
//             }
//             if (max_D <= 128) {
//                 // hipcc can't use max in constexpr
//                 constexpr int kMaxVecsPerThread = 4 / 4 >= 1 ? 4 / 4 : 1;
//                 // If max_D is small, use fewer number of threads than kWarpSize.
// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//                 constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 4, 1);
// #else
//                 constexpr int kThreadGroupSize = kWarpSize;
// #endif
//                 sgd_decoupling_update_kernel<emb_t, cache_t, kMaxVecsPerThread, kThreadGroupSize, 4>
//                     <<<div_round_up(grad_dev_indices.numel(), kMaxThreads / kThreadGroupSize),
//                        dim3(kThreadGroupSize, kMaxThreads / kThreadGroupSize, 1),
//                        0, // Shared memory is not needed because uint8_t is not supported
//                        at::cuda::getCurrentCUDAStream()
//                     >>>
//                     (
//                         dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
//                         uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
//                         lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
//                         flatten_grad_dev_weights.packed_accessor32<emb_t, 1, at::RestrictPtrTraits>(),
//                         flatten_grad_dev_indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
//                         weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
//                         // Use weights_placements instead of
//                         // sorted_lxu_cache_locations because LXU cache is not
//                         // supported right now
//                         weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         max_D,
//                         stochastic_rounding,
//                         rng_engine_inputs,
//                         D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         max_hash_size,
//                         learning_rate
//                     );
//                 C10_CUDA_KERNEL_LAUNCH_CHECK();
//                 return;
//             }
//             if (max_D <= 256) {
//                 // hipcc can't use max in constexpr
//                 constexpr int kMaxVecsPerThread = 8 / 4 >= 1 ? 8 / 4 : 1;
//                 // If max_D is small, use fewer number of threads than kWarpSize.
// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//                 constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 8, 1);
// #else
//                 constexpr int kThreadGroupSize = kWarpSize;
// #endif
//                 sgd_decoupling_update_kernel<emb_t, cache_t, kMaxVecsPerThread, kThreadGroupSize, 4>
//                     <<<div_round_up(grad_dev_indices.numel(), kMaxThreads / kThreadGroupSize),
//                        dim3(kThreadGroupSize, kMaxThreads / kThreadGroupSize, 1),
//                        0, // Shared memory is not needed because uint8_t is not supported
//                        at::cuda::getCurrentCUDAStream()
//                     >>>
//                     (
//                         dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
//                         uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
//                         lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
//                         flatten_grad_dev_weights.packed_accessor32<emb_t, 1, at::RestrictPtrTraits>(),
//                         flatten_grad_dev_indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
//                         weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
//                         // Use weights_placements instead of
//                         // sorted_lxu_cache_locations because LXU cache is not
//                         // supported right now
//                         weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         max_D,
//                         stochastic_rounding,
//                         rng_engine_inputs,
//                         D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         max_hash_size,
//                         learning_rate
//                     );
//                 C10_CUDA_KERNEL_LAUNCH_CHECK();
//                 return;
//             }
//             if (max_D <= 384) {
//                 // hipcc can't use max in constexpr
//                 constexpr int kMaxVecsPerThread = 12 / 4 >= 1 ? 12 / 4 : 1;
//                 // If max_D is small, use fewer number of threads than kWarpSize.
// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//                 constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 12, 1);
// #else
//                 constexpr int kThreadGroupSize = kWarpSize;
// #endif
//                 sgd_decoupling_update_kernel<emb_t, cache_t, kMaxVecsPerThread, kThreadGroupSize, 4>
//                     <<<div_round_up(grad_dev_indices.numel(), kMaxThreads / kThreadGroupSize),
//                        dim3(kThreadGroupSize, kMaxThreads / kThreadGroupSize, 1),
//                        0, // Shared memory is not needed because uint8_t is not supported
//                        at::cuda::getCurrentCUDAStream()
//                     >>>
//                     (
//                         dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
//                         uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
//                         lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
//                         flatten_grad_dev_weights.packed_accessor32<emb_t, 1, at::RestrictPtrTraits>(),
//                         flatten_grad_dev_indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
//                         weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
//                         // Use weights_placements instead of
//                         // sorted_lxu_cache_locations because LXU cache is not
//                         // supported right now
//                         weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         max_D,
//                         stochastic_rounding,
//                         rng_engine_inputs,
//                         D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         max_hash_size,
//                         learning_rate
//                     );
//                 C10_CUDA_KERNEL_LAUNCH_CHECK();
//                 return;
//             }
//             if (max_D <= 512) {
//                 // hipcc can't use max in constexpr
//                 constexpr int kMaxVecsPerThread = 16 / 4 >= 1 ? 16 / 4 : 1;
//                 // If max_D is small, use fewer number of threads than kWarpSize.
// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//                 constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 16, 1);
// #else
//                 constexpr int kThreadGroupSize = kWarpSize;
// #endif
//                 sgd_decoupling_update_kernel<emb_t, cache_t, kMaxVecsPerThread, kThreadGroupSize, 4>
//                     <<<div_round_up(grad_dev_indices.numel(), kMaxThreads / kThreadGroupSize),
//                        dim3(kThreadGroupSize, kMaxThreads / kThreadGroupSize, 1),
//                        0, // Shared memory is not needed because uint8_t is not supported
//                        at::cuda::getCurrentCUDAStream()
//                     >>>
//                     (
//                         dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
//                         uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
//                         lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
//                         flatten_grad_dev_weights.packed_accessor32<emb_t, 1, at::RestrictPtrTraits>(),
//                         flatten_grad_dev_indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
//                         weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
//                         // Use weights_placements instead of
//                         // sorted_lxu_cache_locations because LXU cache is not
//                         // supported right now
//                         weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         max_D,
//                         stochastic_rounding,
//                         rng_engine_inputs,
//                         D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         max_hash_size,
//                         learning_rate
//                     );
//                 C10_CUDA_KERNEL_LAUNCH_CHECK();
//                 return;
//             }
//             if (max_D <= 640) {
//                 // hipcc can't use max in constexpr
//                 constexpr int kMaxVecsPerThread = 20 / 4 >= 1 ? 20 / 4 : 1;
//                 // If max_D is small, use fewer number of threads than kWarpSize.
// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//                 constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 20, 1);
// #else
//                 constexpr int kThreadGroupSize = kWarpSize;
// #endif
//                 sgd_decoupling_update_kernel<emb_t, cache_t, kMaxVecsPerThread, kThreadGroupSize, 4>
//                     <<<div_round_up(grad_dev_indices.numel(), kMaxThreads / kThreadGroupSize),
//                        dim3(kThreadGroupSize, kMaxThreads / kThreadGroupSize, 1),
//                        0, // Shared memory is not needed because uint8_t is not supported
//                        at::cuda::getCurrentCUDAStream()
//                     >>>
//                     (
//                         dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
//                         uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
//                         lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
//                         flatten_grad_dev_weights.packed_accessor32<emb_t, 1, at::RestrictPtrTraits>(),
//                         flatten_grad_dev_indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
//                         weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
//                         // Use weights_placements instead of
//                         // sorted_lxu_cache_locations because LXU cache is not
//                         // supported right now
//                         weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         max_D,
//                         stochastic_rounding,
//                         rng_engine_inputs,
//                         D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         max_hash_size,
//                         learning_rate
//                     );
//                 C10_CUDA_KERNEL_LAUNCH_CHECK();
//                 return;
//             }
//             if (max_D <= 768) {
//                 // hipcc can't use max in constexpr
//                 constexpr int kMaxVecsPerThread = 24 / 4 >= 1 ? 24 / 4 : 1;
//                 // If max_D is small, use fewer number of threads than kWarpSize.
// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//                 constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 24, 1);
// #else
//                 constexpr int kThreadGroupSize = kWarpSize;
// #endif
//                 sgd_decoupling_update_kernel<emb_t, cache_t, kMaxVecsPerThread, kThreadGroupSize, 4>
//                     <<<div_round_up(grad_dev_indices.numel(), kMaxThreads / kThreadGroupSize),
//                        dim3(kThreadGroupSize, kMaxThreads / kThreadGroupSize, 1),
//                        0, // Shared memory is not needed because uint8_t is not supported
//                        at::cuda::getCurrentCUDAStream()
//                     >>>
//                     (
//                         dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
//                         uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
//                         lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
//                         flatten_grad_dev_weights.packed_accessor32<emb_t, 1, at::RestrictPtrTraits>(),
//                         flatten_grad_dev_indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
//                         weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
//                         // Use weights_placements instead of
//                         // sorted_lxu_cache_locations because LXU cache is not
//                         // supported right now
//                         weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         max_D,
//                         stochastic_rounding,
//                         rng_engine_inputs,
//                         D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         max_hash_size,
//                         learning_rate
//                     );
//                 C10_CUDA_KERNEL_LAUNCH_CHECK();
//                 return;
//             }
//             if (max_D <= 896) {
//                 // hipcc can't use max in constexpr
//                 constexpr int kMaxVecsPerThread = 28 / 4 >= 1 ? 28 / 4 : 1;
//                 // If max_D is small, use fewer number of threads than kWarpSize.
// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//                 constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 28, 1);
// #else
//                 constexpr int kThreadGroupSize = kWarpSize;
// #endif
//                 sgd_decoupling_update_kernel<emb_t, cache_t, kMaxVecsPerThread, kThreadGroupSize, 4>
//                     <<<div_round_up(grad_dev_indices.numel(), kMaxThreads / kThreadGroupSize),
//                        dim3(kThreadGroupSize, kMaxThreads / kThreadGroupSize, 1),
//                        0, // Shared memory is not needed because uint8_t is not supported
//                        at::cuda::getCurrentCUDAStream()
//                     >>>
//                     (
//                         dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
//                         uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
//                         lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
//                         flatten_grad_dev_weights.packed_accessor32<emb_t, 1, at::RestrictPtrTraits>(),
//                         flatten_grad_dev_indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
//                         weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
//                         // Use weights_placements instead of
//                         // sorted_lxu_cache_locations because LXU cache is not
//                         // supported right now
//                         weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         max_D,
//                         stochastic_rounding,
//                         rng_engine_inputs,
//                         D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         max_hash_size,
//                         learning_rate
//                     );
//                 C10_CUDA_KERNEL_LAUNCH_CHECK();
//                 return;
//             }
//             if (max_D <= 1024) {
//                 // hipcc can't use max in constexpr
//                 constexpr int kMaxVecsPerThread = 32 / 4 >= 1 ? 32 / 4 : 1;
//                 // If max_D is small, use fewer number of threads than kWarpSize.
// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//                 constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 32, 1);
// #else
//                 constexpr int kThreadGroupSize = kWarpSize;
// #endif
//                 sgd_decoupling_update_kernel<emb_t, cache_t, kMaxVecsPerThread, kThreadGroupSize, 4>
//                     <<<div_round_up(grad_dev_indices.numel(), kMaxThreads / kThreadGroupSize),
//                        dim3(kThreadGroupSize, kMaxThreads / kThreadGroupSize, 1),
//                        0, // Shared memory is not needed because uint8_t is not supported
//                        at::cuda::getCurrentCUDAStream()
//                     >>>
//                     (
//                         dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
//                         uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
//                         lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
//                         flatten_grad_dev_weights.packed_accessor32<emb_t, 1, at::RestrictPtrTraits>(),
//                         flatten_grad_dev_indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
//                         weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
//                         // Use weights_placements instead of
//                         // sorted_lxu_cache_locations because LXU cache is not
//                         // supported right now
//                         weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         max_D,
//                         stochastic_rounding,
//                         rng_engine_inputs,
//                         D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
//                         max_hash_size,
//                         learning_rate
//                     );
//                 C10_CUDA_KERNEL_LAUNCH_CHECK();
//                 return;
//             }
//         }
//     );
// }
// ////// =================================================================================================================================
// ////// =================================================================================================================================
// ////// =================================================================================================================================



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
    int64_t shared_d_offset = warp_idx_in_block * max_D;

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
    int64_t indices_start = offsets[b_t];
    int64_t indices_end = offsets[b_t + 1];
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
    // for(int32_t i = threadIdx.x; i < D_emb; i+=kThreadGroupSize){
    //     shared_output_buffer[shared_d_offset + i] = shared_output_buffer[shared_d_offset + i] * 0.01 * (-1);
    // }

    // Iterate over each kThreadGroupSize-sized subset of L indices in the bag
    for (int32_t l_start = 0; l_start < L; l_start += kThreadGroupSize) {
        // Determine the L index that this thread will load data from in cooperative load
        int32_t l = l_start + threadIdx.x;
        // Cooperatively load the indices
        int64_t idx = l < L ? indices[indices_start + l] : 0;
        int64_t inv = l < L ? inverse[indices_start + l] : 0;

        // Cooperatively load the cache's indices
        // int32_t cache_idx = (use_lxu_cache && placement == PlacementType::MANAGED_CACHING && l < L) ? lxu_cache_locations[indices_start + l] : 0;

        // Iterate over kThreadGroupSize indices ()
        for (auto j = 0; j < kThreadGroupSize && l_start + j < L; ++j) {
            // Load index from thread j in the group
            int64_t idx_j = SHFL_SYNC(idx, j);
            int64_t inv_j = SHFL_SYNC(inv, j);

            // decouple
            int64_t grad_buffer_offset =  inv_j * max_D;
            for(int32_t i = threadIdx.x; i < D_emb; i+=kThreadGroupSize){
                gpuAtomicAdd(&grad_dev_weights[grad_buffer_offset + i], shared_output_buffer[shared_d_offset + i]);
            }

            // // fused
            // int64_t weight_offset_emb =  weights_offset + idx_j * D_emb;
            // for(int32_t i = threadIdx.x; i < D_emb; i+=kThreadGroupSize){
            //     gpuAtomicAdd(&dev_weights[weight_offset_emb + i], shared_output_buffer[shared_d_offset + i]);
            // }
        }
    }
}


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
    int64_t shared_d_offset = warp_idx_in_block * max_D;

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
    int64_t indices_start = offsets[b_t];
    int64_t indices_end = offsets[b_t + 1];
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
    // for(int32_t i = threadIdx.x; i < D_emb; i+=kThreadGroupSize){
    //     shared_output_buffer[shared_d_offset + i] = shared_output_buffer[shared_d_offset + i] * 0.01 * (-1);
    // }

    // Iterate over each kThreadGroupSize-sized subset of L indices in the bag
    for (int32_t l_start = 0; l_start < L; l_start += kThreadGroupSize) {
        // Determine the L index that this thread will load data from in cooperative load
        int32_t l = l_start + threadIdx.x;
        // Cooperatively load the indices
        int64_t idx = l < L ? indices[indices_start + l] : 0;
        int64_t inv = l < L ? inverse[indices_start + l] : 0;

        // Cooperatively load the cache's indices
        // int32_t cache_idx = (use_lxu_cache && placement == PlacementType::MANAGED_CACHING && l < L) ? lxu_cache_locations[indices_start + l] : 0;

        // Iterate over kThreadGroupSize indices ()
        for (auto j = 0; j < kThreadGroupSize && l_start + j < L; ++j) {
            // Load index from thread j in the group
            int64_t idx_j = SHFL_SYNC(idx, j);
            int64_t inv_j = SHFL_SYNC(inv, j);


            int32_t get_signal = 0;
            if(threadIdx.x == 0)
                get_signal = atomicCAS(&grad_dev_signal[inv_j], 0, 1);
            get_signal = SHFL_SYNC(get_signal, 0);

            int64_t grad_buffer_offset =  inv_j * max_D;
            if(get_signal == 0){ // Memory Set
                for(int32_t i = threadIdx.x; i < D_emb; i+=kThreadGroupSize){
                    atomicExch(&grad_dev_weights[grad_buffer_offset + i], shared_output_buffer[shared_d_offset + i]);
                }
                __threadfence();
                if(threadIdx.x == 0)
                    gpuAtomicAdd(&grad_dev_signal[inv_j], 1);
            }
            else{
                while(get_signal==1){
                    if(threadIdx.x == 0)
                        get_signal = atomicCAS(&grad_dev_signal[inv_j], 0, 0);
                    get_signal = SHFL_SYNC(get_signal, 0);
                }
                for(int32_t i = threadIdx.x; i < D_emb; i+=kThreadGroupSize){
                    gpuAtomicAdd(&grad_dev_weights[grad_buffer_offset + i], shared_output_buffer[shared_d_offset + i]);
                }
            }
        }
    }
}


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
) {

    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
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
    // grad_dev_weights has emb_t type
    auto grad_dev_weights = at::empty({total_unique_indices * max_D}, dev_weights.options()); // [TODO] move the at::zeros() to input distribution
    // auto grad_dev_weights = at::zeros({total_unique_indices * max_D}, dev_weights.options());

    // at::Device device(at::kCUDA);
    // auto int32_tensor_options = at::TensorOptions().device(device).dtype(at::kInt);
    // auto grad_dev_signal = at::zeros({total_unique_indices}, int32_tensor_options);



    // auto grad_dev_weights = at::zeros({total_unique_indices * max_D}, dev_weights.options());
    // grad_dev_weights.zero_();
    // cudaMemset(grad_dev_weights.data_ptr<float>(), 0, total_unique_indices * max_D * sizeof(float));

    // short-circuit if there are zero indices.
    if (indices.numel() == 0) {
        return at::sparse_coo_tensor(
            at::empty({1, 0}, indices.options()),
            grad_dev_weights.reshape({0, max_D}),
            {total_hash_size, max_D},
            dev_weights.options().layout(at::kSparse)
        );
    }
    int32_t T = D_offsets.numel() - 1;

    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]

    const auto total_B = offsets.size(0) - 1;
    const int32_t B = (total_B) / T; // local_batch_size_after_dist = local_batch_size * nDev
    int32_t local_batch_size = B / nranks;

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

    DISPATCH_EMB_GRAD_CACHE_TYPES(
        dev_weights.scalar_type(),
        grad_output.scalar_type(),
        lxu_cache_weights.scalar_type(),
        "nvshmem_unsorting_backward_kernel", [&] {

        //// ====================================================================================
        DISPATCH_OPTIMAL_KERNEL(max_D, [&] {
#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name = "nvshmem_unsorting_backward_kernel_decoupling";
#endif

        const auto backward_cta_per_row_kernel =
            nvshmem_unsorting_backward_kernel_decoupling
                <
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize
                >;

        backward_cta_per_row_kernel<<<
            div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
            dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
            max_D * (512/kThreadGroupSize) * sizeof(float),
            at::cuda::getCurrentCUDAStream()
        >>>
        (
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
            MAKE_PTA_WITH_NAME(func_name, grad_dev_weights, emb_t, 1, 64),
            MAKE_PTA_WITH_NAME(func_name, inverse, int64_t, 1, 32),
            max_D,
            total_dim_output,
            nranks,
            rank,
            total_D,
            local_batch_size
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        });
    //// ====================================================================================

//// ====================================================================================
//         DISPATCH_OPTIMAL_KERNEL(max_D, [&] {
// #ifdef FBGEMM_GPU_MEMCHECK
//                 const auto func_name = "nvshmem_unsorting_backward_kernel_decoupling_signal";
// #endif

//         const auto backward_cta_per_row_kernel =
//             nvshmem_unsorting_backward_kernel_decoupling_signal
//                 <
//                 emb_t,
//                 grad_t,
//                 cache_t,
//                 kMaxVecsPerThread,
//                 kThreadGroupSize
//                 >;

//         backward_cta_per_row_kernel<<<
//             div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
//             dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
//             max_D * (512/kThreadGroupSize) * sizeof(float),
//             at::cuda::getCurrentCUDAStream()
//         >>>
//         (
//             MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
//             MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
//             MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
//             MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
//             MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
//             MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
//             FixedDivisor(B),
//             MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
//             MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),
//             pooling_mode,
//             MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32), // if not dense

//             nvshmem_grad,
//             MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
//             MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
//             MAKE_PTA_WITH_NAME(func_name, grad_dev_weights, float, 1, 64),
//             MAKE_PTA_WITH_NAME(func_name, grad_dev_signal, int32_t, 1, 32),
//             MAKE_PTA_WITH_NAME(func_name, inverse, int64_t, 1, 32),
//             max_D,
//             total_dim_output,
//             nranks,
//             rank,
//             total_D,
//             local_batch_size
//             );
//             C10_CUDA_KERNEL_LAUNCH_CHECK();
//             return;
//         });
//// ====================================================================================

    });

    return at::sparse_coo_tensor(
        unique_linear_indices.unsqueeze(0),
        grad_dev_weights.reshape({total_unique_indices, max_D}),
        {total_hash_size, max_D},
        dev_weights.options().layout(at::kSparse));
}



// // Host function
// void nvshmem_unsorting_backward_host_function(
//     Tensor dev_weights,
//     Tensor uvm_weights,
//     Tensor lxu_cache_weights,
//     Tensor weights_placements,
//     Tensor weights_offsets,
//     Tensor D_offsets,
//     int64_t total_D,
//     int64_t max_D,
//     Tensor indices,
//     Tensor offsets,
//     int64_t pooling_mode,
//     Tensor lxu_cache_locations,
//     int64_t output_dtype,
//     bool is_experimental,
//     // all-to-all information:
//     float* nvshmem_grad,
//     Tensor dim_sum_per_rank_data,
//     Tensor dim_offset_per_rank_data,
//     int32_t total_dim_output,
//     int32_t nranks,
//     int32_t rank,
//     float learning_rate
// ){
//     TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
//         uvm_weights,
//         lxu_cache_weights,
//         weights_placements,
//         weights_offsets,
//         D_offsets,
//         indices,
//         offsets,
//         lxu_cache_locations,
//         dev_weights
//     );

//     at::cuda::OptionalCUDAGuard device_guard;
//     device_guard.set_index(dev_weights.get_device());
//     int32_t T = D_offsets.numel() - 1; // n_local_Table
//     TORCH_CHECK_GT(T, 0);
//     // offsets = [B x T  + 1]
//     const auto total_B = offsets.size(0) - 1; // global_batch_size = local_batch_size * nDev * n_local_Table
//     const int32_t B = (total_B) / T; // local_batch_size_after_dist = local_batch_size * nDev
//     int32_t local_batch_size = B / nranks;
//     TORCH_CHECK_GE(B, 0);
//     TORCH_CHECK_GT(total_D, 0);
//     TORCH_CHECK_EQ(total_D % 4, 0);
//     TORCH_CHECK_LE(max_D, 1024);

//     Tensor output;
//     SparseType o_dtype = static_cast<SparseType>(output_dtype);
//     TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
//                 o_dtype == SparseType::BF16 || o_dtype == SparseType::INT8);
//     int64_t total_adjusted_D = total_D;
//     if (o_dtype == SparseType::INT8) {
//         total_adjusted_D += T * kINT8QparamsBytes;
//     }

//     if (B == 0) {
//         return;
//     }

//     // =================================================================================================================================
//     DISPATCH_EMB_CACHE_OUTPUT_TYPES(
//         dev_weights.scalar_type(),
//         lxu_cache_weights.scalar_type(),
//         dev_weights.scalar_type(),
//         "nvshmem_unsorting_backward_kernel", [&] {
//         // Check if LXU cache is used
//         bool use_lxu_cache = lxu_cache_weights.numel() > 0;
//         if (is_experimental) {
//           if (std::is_same<emb_t, uint8_t>() || std::is_same<output_t, uint8_t>()) {
//             is_experimental = false;
//           }
//         }

//     if (!is_experimental) { // if has_experimental
//         // The dense case does not have cache so we have to generate code for
//         // only one case (value of use_cache/vbe does not matter)
//         if (use_lxu_cache == false) {
//             // kMaxElemPerThread is # of elements handled by thread if we use a full warp for a row
//             // We consider kMaxElemPerThread 1 and 2, and then a multiple of 4.
//             if (max_D <= 128) {
//                 // hipcc can't use max in constexpr
//                 constexpr int kMaxVecsPerThread = 4 / 4 >= 1 ? 4 / 4 : 1;
//                 // If max_D is small, use fewer number of threads than kWarpSize.

// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//                 constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 4, 1);
// #else
//                 constexpr int kThreadGroupSize = kWarpSize;
// #endif

// #ifdef FBGEMM_GPU_MEMCHECK
//                 const auto func_name = "nvshmem_unsorting_backward_kernel";
// #endif
//                 // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
//                 nvshmem_unsorting_backward_kernel<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
//                     div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
//                     dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
//                     max_D * (512/kThreadGroupSize) * sizeof(float),
//                     at::cuda::getCurrentCUDAStream()>>>(
//                     MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
//                     MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
//                     FixedDivisor(B),
//                     MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),
//                     pooling_mode,
//                     MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32), // if not dense

//                     nvshmem_grad,
//                     MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
//                     max_D,
//                     total_dim_output,
//                     nranks,
//                     rank,
//                     total_D,
//                     local_batch_size,
//                     learning_rate
//                     );
//                 C10_CUDA_KERNEL_LAUNCH_CHECK();
//                 return;
//             }

//             if (max_D <= 256) {
//                 // hipcc can't use max in constexpr
//                 constexpr int kMaxVecsPerThread = 8 / 4 >= 1 ? 8 / 4 : 1;
//                 // If max_D is small, use fewer number of threads than kWarpSize.

// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//                 constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 8, 1);
// #else
//                 constexpr int kThreadGroupSize = kWarpSize;
// #endif

// #ifdef FBGEMM_GPU_MEMCHECK
//                 const auto func_name = "nvshmem_unsorting_backward_kernel";
// #endif
//                 // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
//                 nvshmem_unsorting_backward_kernel<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
//                     div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
//                     dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
//                     max_D * (512/kThreadGroupSize) * sizeof(float),
//                     at::cuda::getCurrentCUDAStream()>>>(
//                     MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
//                     MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
//                     FixedDivisor(B),
//                     MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),
//                     pooling_mode,
//                     MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32), // if not dense

//                     nvshmem_grad,
//                     MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
//                     max_D,
//                     total_dim_output,
//                     nranks,
//                     rank,
//                     total_D,
//                     local_batch_size,
//                     learning_rate
//                     );
//                 C10_CUDA_KERNEL_LAUNCH_CHECK();
//                 return;
//             }
//             if (max_D <= 384) {
//                 // hipcc can't use max in constexpr
//                 constexpr int kMaxVecsPerThread = 12 / 4 >= 1 ? 12 / 4 : 1;
//                 // If max_D is small, use fewer number of threads than kWarpSize.

// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//                 constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 12, 1);
// #else
//                 constexpr int kThreadGroupSize = kWarpSize;
// #endif

// #ifdef FBGEMM_GPU_MEMCHECK
//                 const auto func_name = "nvshmem_unsorting_backward_kernel";
// #endif
//                 // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
//                 nvshmem_unsorting_backward_kernel<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
//                     div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
//                     dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
//                     max_D * (512/kThreadGroupSize) * sizeof(float),
//                     at::cuda::getCurrentCUDAStream()>>>(
//                     MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
//                     MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
//                     FixedDivisor(B),
//                     MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),
//                     pooling_mode,
//                     MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32), // if not dense

//                     nvshmem_grad,
//                     MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
//                     max_D,
//                     total_dim_output,
//                     nranks,
//                     rank,
//                     total_D,
//                     local_batch_size,
//                     learning_rate
//                     );
//                 C10_CUDA_KERNEL_LAUNCH_CHECK();
//                 return;
//             }
//             if (max_D <= 512) {
//                 // hipcc can't use max in constexpr
//                 constexpr int kMaxVecsPerThread = 16 / 4 >= 1 ? 16 / 4 : 1;
//                 // If max_D is small, use fewer number of threads than kWarpSize.

// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//                 constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 16, 1);
// #else
//                 constexpr int kThreadGroupSize = kWarpSize;
// #endif

// #ifdef FBGEMM_GPU_MEMCHECK
//                 const auto func_name = "nvshmem_unsorting_backward_kernel";
// #endif
//                 // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
//                 nvshmem_unsorting_backward_kernel<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
//                     div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
//                     dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
//                     max_D * (512/kThreadGroupSize) * sizeof(float),
//                     at::cuda::getCurrentCUDAStream()>>>(
//                     MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
//                     MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
//                     FixedDivisor(B),
//                     MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),
//                     pooling_mode,
//                     MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32), // if not dense

//                     nvshmem_grad,
//                     MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
//                     max_D,
//                     total_dim_output,
//                     nranks,
//                     rank,
//                     total_D,
//                     local_batch_size,
//                     learning_rate
//                     );
//                 C10_CUDA_KERNEL_LAUNCH_CHECK();
//                 return;
//             }
//             if (max_D <= 640) {
//                 // hipcc can't use max in constexpr
//                 constexpr int kMaxVecsPerThread = 20 / 4 >= 1 ? 20 / 4 : 1;
//                 // If max_D is small, use fewer number of threads than kWarpSize.

// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//                 constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 20, 1);
// #else
//                 constexpr int kThreadGroupSize = kWarpSize;
// #endif

// #ifdef FBGEMM_GPU_MEMCHECK
//                 const auto func_name = "nvshmem_unsorting_backward_kernel";
// #endif
//                 // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
//                 nvshmem_unsorting_backward_kernel<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
//                     div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
//                     dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
//                     max_D * (512/kThreadGroupSize) * sizeof(float),
//                     at::cuda::getCurrentCUDAStream()>>>(
//                     MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
//                     MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
//                     FixedDivisor(B),
//                     MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),
//                     pooling_mode,
//                     MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32), // if not dense

//                     nvshmem_grad,
//                     MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
//                     max_D,
//                     total_dim_output,
//                     nranks,
//                     rank,
//                     total_D,
//                     local_batch_size,
//                     learning_rate
//                     );
//                 C10_CUDA_KERNEL_LAUNCH_CHECK();
//                 return;
//             }
//             if (max_D <= 768) {
//                 // hipcc can't use max in constexpr
//                 constexpr int kMaxVecsPerThread = 24 / 4 >= 1 ? 24 / 4 : 1;
//                 // If max_D is small, use fewer number of threads than kWarpSize.

// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//                 constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 24, 1);
// #else
//                 constexpr int kThreadGroupSize = kWarpSize;
// #endif

// #ifdef FBGEMM_GPU_MEMCHECK
//                 const auto func_name = "nvshmem_unsorting_backward_kernel";
// #endif
//                 // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
//                 nvshmem_unsorting_backward_kernel<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
//                     div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
//                     dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
//                     max_D * (512/kThreadGroupSize) * sizeof(float),
//                     at::cuda::getCurrentCUDAStream()>>>(
//                     MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
//                     MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
//                     FixedDivisor(B),
//                     MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),
//                     pooling_mode,
//                     MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32), // if not dense

//                     nvshmem_grad,
//                     MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
//                     max_D,
//                     total_dim_output,
//                     nranks,
//                     rank,
//                     total_D,
//                     local_batch_size,
//                     learning_rate
//                     );
//                 C10_CUDA_KERNEL_LAUNCH_CHECK();
//                 return;
//             }
//             if (max_D <= 896) {
//                 // hipcc can't use max in constexpr
//                 constexpr int kMaxVecsPerThread = 28 / 4 >= 1 ? 28 / 4 : 1;
//                 // If max_D is small, use fewer number of threads than kWarpSize.

// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//                 constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 28, 1);
// #else
//                 constexpr int kThreadGroupSize = kWarpSize;
// #endif

// #ifdef FBGEMM_GPU_MEMCHECK
//                 const auto func_name = "nvshmem_unsorting_backward_kernel";
// #endif
//                 // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
//                 nvshmem_unsorting_backward_kernel<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
//                     div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
//                     dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
//                     max_D * (512/kThreadGroupSize) * sizeof(float),
//                     at::cuda::getCurrentCUDAStream()>>>(
//                     MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
//                     MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
//                     FixedDivisor(B),
//                     MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),
//                     pooling_mode,
//                     MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32), // if not dense

//                     nvshmem_grad,
//                     MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
//                     max_D,
//                     total_dim_output,
//                     nranks,
//                     rank,
//                     total_D,
//                     local_batch_size,
//                     learning_rate
//                     );
//                 C10_CUDA_KERNEL_LAUNCH_CHECK();
//                 return;
//             }
//             if (max_D <= 1024) {
//                 // hipcc can't use max in constexpr
//                 constexpr int kMaxVecsPerThread = 32 / 4 >= 1 ? 32 / 4 : 1;
//                 // If max_D is small, use fewer number of threads than kWarpSize.

// #ifdef FBGEMM_USE_SUBWARP_SHUFFLE
//                 constexpr int kThreadGroupSize = kWarpSize / std::max(4 / 32, 1);
// #else
//                 constexpr int kThreadGroupSize = kWarpSize;
// #endif

// #ifdef FBGEMM_GPU_MEMCHECK
//                 const auto func_name = "nvshmem_unsorting_backward_kernel";
// #endif
//                 // std::cout<< "kMaxVecsPerThread:" << kMaxVecsPerThread << ", kThreadGroupSize:" << kThreadGroupSize << ", nBlock:" << div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize) << ", nThread:" << kForwardMaxThreads << "\n";
//                 nvshmem_unsorting_backward_kernel<emb_t, cache_t, output_t, false, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
//                     div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
//                     dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
//                     max_D * (512/kThreadGroupSize) * sizeof(float),
//                     at::cuda::getCurrentCUDAStream()>>>(
//                     MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
//                     MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
//                     MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
//                     FixedDivisor(B),
//                     MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),
//                     pooling_mode,
//                     MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32), // if not dense

//                     nvshmem_grad,
//                     MAKE_PTA_WITH_NAME(func_name, dim_sum_per_rank_data, int32_t, 1, 32),
//                     MAKE_PTA_WITH_NAME(func_name, dim_offset_per_rank_data, int32_t, 1, 32),
//                     max_D,
//                     total_dim_output,
//                     nranks,
//                     rank,
//                     total_D,
//                     local_batch_size,
//                     learning_rate
//                     );
//                 C10_CUDA_KERNEL_LAUNCH_CHECK();
//                 return;
//             }

//         }
//         } // if (!is_experimental)
//     }
// );

//   cudaDeviceSynchronize();
//   return;
// }


// using namespace fbgemm_gpu;

// __global__ __launch_bounds__(kMaxThreads) void linearize_index_kernel_for_unique(
//     const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
//     const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> indices,
//     const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,
//     at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> infos,
//     at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> linear_indices,
//     const int32_t info_B_num_bits,
//     const uint32_t info_B_mask,
//     const uint32_t max_T,
//     const uint32_t max_B,
//     FixedDivisor fd) {
//   const int32_t T = hash_size_cumsum.size(0) - 1;
//   auto b_t = blockIdx.x * blockDim.x + threadIdx.x;
//   int32_t b;
//   int32_t t;
//   const auto total_B = offsets.size(0) - 1;
//   bool valid = b_t < total_B;
//   // info must be uint32_t (using auto will assign int32_t to info)
//   uint32_t info = 0;

//   fd.DivMod(b_t, &t, &b);

//   const int64_t hash_offset = valid ? hash_size_cumsum[t] : -1;
//   const int64_t indices_start = valid ? offsets[b_t] : -1;
//   const int32_t L = valid ? offsets[b_t + 1] - indices_start : 0;
//   const int32_t lane_id = threadIdx.x % kWarpSize;
//     if (valid) {
//         info = (reinterpret_cast<uint32_t*>(&t)[0] << info_B_num_bits) |
//             reinterpret_cast<uint32_t*>(&b)[0];
//     }
//     for (int32_t j = 0; j < kWarpSize; ++j) {
//         const int64_t indices_start_warp =
//             fbgemm_gpu::shfl_sync(indices_start, j);
//         const uint32_t info_warp = fbgemm_gpu::shfl_sync(info, j);
//         const int32_t L_warp = fbgemm_gpu::shfl_sync(L, j);
//         const int64_t hash_offset_warp = fbgemm_gpu::shfl_sync(hash_offset, j);
//         for (int32_t i = lane_id; i < L_warp; i += kWarpSize) {
//         const int64_t idx = __ldg(&indices[indices_start_warp + i]);
//         reinterpret_cast<uint32_t*>(&infos[0])[indices_start_warp + i] =
//             info_warp;
//         linear_indices[indices_start_warp + i] = hash_offset_warp + idx;
//         }
//     }
// }
