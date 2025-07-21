/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <optional>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAStream.h>
#ifdef USE_ROCM
#include <hip/hip_fp16.h>
#else
#include <cuda_bf16.h>
#endif
#include <torch/torch.h>

#define DISPATCH_CASE_FLOATING_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

namespace fbgemm_gpu {

namespace {

#ifdef USE_ROCM
constexpr int kNumThreadsPerWarp = 64;
#else
constexpr int kNumThreadsPerWarp = 32;
#endif
constexpr int kNumWarps = 4;
constexpr int kNumThreads = kNumThreadsPerWarp * kNumWarps;

__inline__ constexpr int ceil_of_ratio(int a, int b) {
  return (a + b - 1) / b;
};

#ifdef USE_ROCM
__device__ __forceinline__ int atomic_add_relaxed(int* addr, int inc) {
  return __hip_atomic_fetch_add(
      addr, inc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
};

__device__ __forceinline__ int atomic_add_release(int* addr, int inc) {
  return __hip_atomic_fetch_add(
      addr, inc, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
};

__device__ __forceinline__ int load_aquire(int* addr) {
  return __hip_atomic_load(addr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
};
#else
__device__ __forceinline__ int atomic_add_relaxed(int* addr, int inc) {
  int val;
  asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], %2;\n"
               : "=r"(val)
               : "l"(addr), "r"(inc));
  return val;
};

__device__ __forceinline__ int atomic_add_release(int* addr, int inc) {
  int val;
  asm volatile("atom.release.gpu.global.add.s32 %0, [%1], %2;\n"
               : "=r"(val)
               : "l"(addr), "r"(inc));
  return val;
};

__device__ __forceinline__ int load_aquire(int* addr) {
  int val;
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(val) : "l"(addr));
  return val;
};
#endif

template <class DataType, class IndexType, int NumExperts, int NumTokensPerTile>
struct SharedStorage {
  DataType routing_scores[NumTokensPerTile * NumExperts];
  IndexType expert_indices[NumTokensPerTile * NumExperts];
  IndexType token_count_cumsums[NumExperts];
};

template <class DataType, class IndexType>
struct Params {
  // 1. Inputs
  // 1.1. Routing scores.
  const DataType* routing_scores;
  const int stride_t;
  const int stride_e;
  // 1.2. Expert ranges.
  const int expert_index_start;
  const int expert_index_end;
  // 1.3. Token counts.
  const IndexType* valid_token_count;
  const int num_tokens;
  const int num_tokens_per_cta;

  // 2. Buffers
  IndexType* buffered_expert_indices;
  IndexType* buffered_token_indices;

  // 3. Outputs
  IndexType* token_count_per_expert;
  IndexType* shuffled_expert_indices;
  IndexType* shuffled_token_indices;
};

template <class DataType, class IndexType, int NumExperts, int NumTokensPerTile>
__global__ void index_shuffling_kernel(Params<DataType, IndexType> params) {
  // scores: [num_tokens, num_experts]
  // counts: [num_experts]
  // expert_indices/shuffled_expert_indices: [num_tokens]
  // token_indices/shuffled_token_indices: [num_tokens]

  __shared__ SharedStorage<DataType, IndexType, NumExperts, NumTokensPerTile>
      smem;

  const auto tidx = threadIdx.x;
  const auto bidx = blockIdx.x;

  const int num_total_tokens = params.num_tokens;
  const int token_index_start = bidx * params.num_tokens_per_cta;
  if (token_index_start >= num_total_tokens) {
    return;
  }

  const int token_index_end =
      std::min(token_index_start + params.num_tokens_per_cta, num_total_tokens);
  const int num_valid_tokens =
      params.valid_token_count ? *params.valid_token_count : num_total_tokens;

  const int expert_index_start = params.expert_index_start;
  const int expert_index_end = params.expert_index_end;

  const int stride_t = params.stride_t;
  const int stride_e = params.stride_e;

  const DataType zero = static_cast<DataType>(0.0f);
  for (int token_index_offset = token_index_start;
       token_index_offset < token_index_end;
       token_index_offset += NumTokensPerTile) {
    // 1. Read scores
    // TODO(shikaili): vectorized. asynchronous.
#pragma unroll
    for (int i = tidx; i < NumTokensPerTile * NumExperts; i += kNumThreads) {
      const int token_index = token_index_offset + i / NumExperts;
      const int expert_index = i % NumExperts;

      smem.routing_scores[i] = (token_index < num_valid_tokens)
          ? params.routing_scores
                [token_index * stride_t + expert_index * stride_e]
          : zero;
      smem.expert_indices[i] = expert_index;
    }
    __syncthreads();

    // 2. Top-1 Reduction
    static_assert(NumExperts % 2 == 0, "");
    constexpr int kNumParallelReductionThreads = NumExperts / 2;

    static_assert(kNumThreads % kNumParallelReductionThreads == 0, "");
    constexpr int kNumParallelReductionGroups =
        kNumThreads / kNumParallelReductionThreads;

    static_assert(NumTokensPerTile % kNumParallelReductionGroups == 0, "");

    // 2D parallel reduction. No bank conflicts.
    for (int num_reduced_threads = 1;
         num_reduced_threads <= kNumParallelReductionThreads;
         num_reduced_threads <<= 1) {
#pragma unroll
      for (int local_token_offset = 0; local_token_offset < NumTokensPerTile;
           local_token_offset += kNumParallelReductionGroups) {
        if (!(tidx & (num_reduced_threads - 1))) {
          int local_token_index =
              local_token_offset + tidx / kNumParallelReductionThreads;
          int lhs_smem_index = local_token_index * NumExperts +
              (tidx % kNumParallelReductionThreads) * 2;
          int rhs_smem_index = lhs_smem_index + num_reduced_threads;

          auto lhs_score = smem.routing_scores[lhs_smem_index];
          auto rhs_score = smem.routing_scores[rhs_smem_index];
          auto lhs_expert_index = smem.expert_indices[lhs_smem_index];
          auto rhs_expert_index = smem.expert_indices[rhs_smem_index];

          bool lhs_larger = lhs_score >= rhs_score;
          smem.routing_scores[lhs_smem_index] =
              lhs_larger ? lhs_score : rhs_score;
          smem.expert_indices[lhs_smem_index] =
              lhs_larger ? lhs_expert_index : rhs_expert_index;
        }
      }
#ifdef USE_ROCM
      __syncthreads();
#else
      if constexpr (kNumParallelReductionThreads <= kNumThreadsPerWarp) {
        __syncwarp();
      } else {
        __syncthreads();
      }
#endif
    }
    if constexpr (kNumParallelReductionThreads > kNumThreadsPerWarp) {
      __syncthreads();
    }

    // 3. Counting
#pragma unroll
    for (int i = tidx; i < NumTokensPerTile; i += kNumThreads) {
      int local_token_index = i;
      int token_index = token_index_offset + i;
      if (token_index < num_valid_tokens) {
        auto expert_index = smem.expert_indices[local_token_index * NumExperts];
        params.buffered_expert_indices[token_index] = expert_index;
        if (expert_index >= expert_index_start &&
            expert_index < expert_index_end) {
          auto token_index_in_expert = atomic_add_relaxed(
              &params.token_count_per_expert[expert_index], 1);
          params.buffered_token_indices[token_index] = token_index_in_expert;
        }
      }
    }
    __syncthreads();
  }

  if (tidx == 0) {
    int processed_tokens = 0;
    int* processed_tokens_addr = &params.token_count_per_expert[NumExperts];

    int inc = token_index_end - token_index_start;
    atomic_add_release(processed_tokens_addr, inc);

    do {
      processed_tokens = load_aquire(processed_tokens_addr);
    } while (processed_tokens != num_total_tokens);
  }
  __syncthreads();

  // 4. Scan
  static_assert(kNumThreads >= NumExperts, "");
  if (tidx < NumExperts) {
    smem.token_count_cumsums[tidx] = params.token_count_per_expert[tidx];
  }
  __syncthreads();

  if (tidx == 0) {
    // TODO(shikaili): parallel.
#pragma unroll
    for (int i = 1; i < NumExperts; ++i) {
      smem.token_count_cumsums[i] += smem.token_count_cumsums[i - 1];
    }
  }
  __syncthreads();

  // 5. Store
  auto get_token_count_cumsum = [](int index) {
    return index == 0 ? 0 : smem.token_count_cumsums[index - 1];
  };

  const int token_count_cumsum_start =
      get_token_count_cumsum(expert_index_start);
  const int token_count_cumsum_end = get_token_count_cumsum(expert_index_end);
  const int num_selected_tokens =
      token_count_cumsum_end - token_count_cumsum_start;

  for (int global_token_offset = bidx * params.num_tokens_per_cta;
       global_token_offset < (bidx + 1) * params.num_tokens_per_cta;
       global_token_offset += kNumThreads) {
    int token_index = global_token_offset + tidx;
    if (token_index < num_valid_tokens) {
      int expert_index = params.buffered_expert_indices[token_index];
      if (expert_index >= expert_index_start &&
          expert_index < expert_index_end) {
        int new_token_index_in_expert =
            params.buffered_token_indices[token_index];
        int new_token_index = get_token_count_cumsum(expert_index) -
            token_count_cumsum_start + new_token_index_in_expert;
        params.shuffled_expert_indices[new_token_index] =
            expert_index - expert_index_start;
        params.shuffled_token_indices[new_token_index] = token_index;
      }
    }
  }

  if (tidx == 0 && bidx == 0) {
    params.token_count_per_expert[NumExperts + 1] = num_selected_tokens;
  }
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> index_shuffling_torch(
    const at::Tensor& routing_scores,
    const std::optional<int64_t>& expert_index_start,
    const std::optional<int64_t>& expert_index_end,
    const std::optional<at::Tensor>& valid_token_count) {
  TORCH_CHECK(
      routing_scores.dtype() == torch::kBFloat16 ||
          routing_scores.dtype() == torch::kFloat,
      "routing_scores must be either BFloat16 or Float");

  using IndexType = int32_t;

  // Declare tensors outside the dispatch to ensure they're accessible for the
  // return statement
  at::Tensor token_count_per_expert;
  at::Tensor shuffled_expert_indices;
  at::Tensor shuffled_token_indices;

  AT_DISPATCH_SWITCH(
      routing_scores.scalar_type(),
      "index_shuffling_params",
      DISPATCH_CASE_FLOATING_TYPES([&] {
        using DataType = scalar_t;

        TORCH_CHECK(routing_scores.dim() == 2);
        const int num_tokens = routing_scores.size(0);
        const int num_experts = routing_scores.size(1);
        TORCH_CHECK(num_experts == 16 || num_experts == 128);

        auto allocate_index_tensor = [&](int size) {
          return at::empty(
              {size},
              at::TensorOptions().dtype(at::kInt).device(
                  routing_scores.device()));
        };
        token_count_per_expert = allocate_index_tensor(num_experts + 2);
        shuffled_expert_indices = allocate_index_tensor(num_tokens);
        shuffled_token_indices = allocate_index_tensor(num_tokens);
        at::Tensor buffered_expert_indices = allocate_index_tensor(num_tokens);
        at::Tensor buffered_token_indices = allocate_index_tensor(num_tokens);

#ifdef USE_ROCM
        // TODO(shikaili): hipMetsetAsync is more expensive than ATen set zero.
        token_count_per_expert.zero_();
#else
        cudaMemsetAsync(
            token_count_per_expert.data_ptr(),
            0,
            token_count_per_expert.numel() *
                token_count_per_expert.dtype().itemsize(),
            at::cuda::getCurrentCUDAStream());
#endif

        // Avoid expensive `cudaGetDeviceProperties` call.
        static int num_sms = -1;
        if (num_sms < 0) {
          cudaDeviceProp deviceProp;
          cudaGetDeviceProperties(&deviceProp, 0);
          num_sms = deviceProp.multiProcessorCount;
        }

#ifdef USE_ROCM
        constexpr int kNumTokensPerTile = 32;
#else
        constexpr int kNumTokensPerTile = 16;
#endif

        void* kernel;
        int smem_size;

#define DISPATCH(E, B)                                               \
  kernel = (void*)index_shuffling_kernel<DataType, IndexType, E, B>; \
  smem_size = sizeof(SharedStorage<DataType, IndexType, E, B>);

        if (num_experts == 16) {
          DISPATCH(16, kNumTokensPerTile);
        } else {
          TORCH_CHECK(num_experts == 128);
          DISPATCH(128, kNumTokensPerTile);
        }

        const int num_tiles = ceil_of_ratio(num_tokens, kNumTokensPerTile);
        const int num_ctas = std::min(num_tiles, num_sms);
        const int num_tiles_per_cta = ceil_of_ratio(
            ceil_of_ratio(num_tokens, num_ctas), kNumTokensPerTile);
        const int num_tokens_per_cta = num_tiles_per_cta * kNumTokensPerTile;

        Params<DataType, IndexType> params = {
            // Inputs
            .routing_scores =
                reinterpret_cast<DataType*>(routing_scores.data_ptr()),
            .stride_t = static_cast<int>(routing_scores.stride(0)),
            .stride_e = static_cast<int>(routing_scores.stride(1)),
            .expert_index_start =
                expert_index_start.has_value() ? int(*expert_index_start) : 0,
            .expert_index_end = expert_index_end.has_value()
                ? int(*expert_index_end)
                : num_experts,
            .valid_token_count = reinterpret_cast<IndexType*>(
                valid_token_count.has_value() ? valid_token_count->data_ptr()
                                              : nullptr),
            .num_tokens = num_tokens,
            .num_tokens_per_cta = num_tokens_per_cta,
            // Buffer
            .buffered_expert_indices = reinterpret_cast<IndexType*>(
                buffered_expert_indices.data_ptr()),
            .buffered_token_indices =
                reinterpret_cast<IndexType*>(buffered_token_indices.data_ptr()),
            // Outputs
            .token_count_per_expert =
                reinterpret_cast<IndexType*>(token_count_per_expert.data_ptr()),
            .shuffled_expert_indices = reinterpret_cast<IndexType*>(
                shuffled_expert_indices.data_ptr()),
            .shuffled_token_indices = reinterpret_cast<IndexType*>(
                shuffled_token_indices.data_ptr())};

        dim3 grids(num_ctas);
        dim3 blocks(kNumThreads);
        void* args[] = {(void*)&params};
        auto stream = at::cuda::getCurrentCUDAStream();

#ifdef USE_ROCM
        // hipLaunchCooperativeKernel seems to cause incorrect memory order
        // across kernel launches.
        C10_CUDA_CHECK(hipLaunchKernel(
            (void*)kernel, grids, blocks, args, smem_size, stream));
#else
        C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void*)kernel, grids, blocks, args, smem_size, stream));
#endif
      }));

  return std::make_tuple(
      token_count_per_expert, shuffled_expert_indices, shuffled_token_indices);
}

} // namespace fbgemm_gpu
