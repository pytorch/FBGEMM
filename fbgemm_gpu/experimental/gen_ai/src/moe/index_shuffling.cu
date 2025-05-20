/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <optional>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/torch.h>

namespace fbgemm_gpu {

namespace {

#ifdef USE_ROCM
constexpr int kNumThreadsPerWarp = 64;
#else
constexpr int kNumThreadsPerWarp = 32;
#endif
constexpr int kNumWarps = 4;
constexpr int kNumThreads = kNumThreadsPerWarp * kNumWarps;

static int num_sms = -1;

__inline__ constexpr int ceil_of_ratio(int a, int b) {
  return (a + b - 1) / b;
};

template <typename T>
__inline__ T* get_ptr(std::optional<at::Tensor> tensor) {
  return reinterpret_cast<T*>(
      tensor.has_value() ? tensor->data_ptr() : nullptr);
};

template <typename T>
__inline__ __device__ T get_item(const T* ptr, const T& default_value) {
  return ptr != nullptr ? *ptr : default_value;
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
  // Inputs
  const DataType* routing_scores;
  const int stride_t;
  const int stride_e;
  const IndexType* valid_token_count;
  const int num_tokens;
  const int num_tokens_per_cta;

  // Buffer
  IndexType* buffered_expert_indices;
  IndexType* buffered_token_indices;

  // Outputs
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
  const int num_valid_tokens =
      get_item(params.valid_token_count, num_total_tokens);

  const int token_index_offset_start = bidx * params.num_tokens_per_cta;
  const int token_index_offset_end = std::min(
      token_index_offset_start + params.num_tokens_per_cta, num_total_tokens);

  if (token_index_offset_start >= num_total_tokens) {
    return;
  }

  const int stride_t = params.stride_t;
  const int stride_e = params.stride_e;

  for (int token_index_offset = token_index_offset_start;
       token_index_offset < token_index_offset_end;
       token_index_offset += NumTokensPerTile) {
    // 1. Read scores
    // TODO(shikaili): vectorized. asynchronous.
#pragma unroll
    for (int i = tidx; i < NumTokensPerTile * NumExperts; i += kNumThreads) {
      int token_index = token_index_offset + i / NumExperts;
      int expert_index = i % NumExperts;

      smem.routing_scores[i] = token_index < num_valid_tokens
          ? params.routing_scores
                [token_index * stride_t + expert_index * stride_e]
          : static_cast<DataType>(0.0f);
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
        auto token_index_in_expert =
            atomic_add_relaxed(&params.token_count_per_expert[expert_index], 1);
        params.buffered_expert_indices[token_index] = expert_index;
        params.buffered_token_indices[token_index] = token_index_in_expert;
      }
    }
    __syncthreads();
  }

  if (tidx == 0) {
    int processed_tokens = 0;
    int* processed_tokens_addr = &params.token_count_per_expert[NumExperts];

    int inc = token_index_offset_end - token_index_offset_start;
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
  for (int global_token_offset = bidx * params.num_tokens_per_cta;
       global_token_offset < (bidx + 1) * params.num_tokens_per_cta;
       global_token_offset += kNumThreads) {
    int token_index = global_token_offset + tidx;
    if (token_index < num_valid_tokens) {
      int expert_index = params.buffered_expert_indices[token_index];
      int token_index_in_expert = params.buffered_token_indices[token_index];
      int new_token_index =
          (expert_index == 0 ? 0 : smem.token_count_cumsums[expert_index - 1]) +
          token_index_in_expert;
      params.shuffled_expert_indices[new_token_index] = expert_index;
      params.shuffled_token_indices[new_token_index] = token_index;
    } else if (token_index < num_total_tokens) {
      // Overwrites to have the padded indices to use the original indices to
      // avoid illegal memory access.
      params.shuffled_expert_indices[token_index] = NumExperts - 1;
      params.shuffled_token_indices[token_index] = token_index;
    }
  }
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> index_shuffling_torch(
    const at::Tensor& routing_scores,
    std::optional<at::Tensor> valid_token_count) {
  TORCH_CHECK(routing_scores.dtype() == torch::kBFloat16);
  using DataType = __nv_bfloat16;
  using IndexType = int32_t;

  TORCH_CHECK(routing_scores.dim() == 2);
  const int num_tokens = routing_scores.size(0);
  const int num_experts = routing_scores.size(1);
  TORCH_CHECK(num_experts == 16 || num_experts == 128);

  auto allocate_index_tensor = [&](int size) {
    return at::empty(
        {size},
        at::TensorOptions().dtype(at::kInt).device(routing_scores.device()));
  };
  at::Tensor token_count_per_expert = allocate_index_tensor(num_experts + 1);
  at::Tensor shuffled_expert_indices = allocate_index_tensor(num_tokens);
  at::Tensor shuffled_token_indices = allocate_index_tensor(num_tokens);
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
  const int num_tiles_per_cta =
      ceil_of_ratio(ceil_of_ratio(num_tokens, num_ctas), kNumTokensPerTile);
  const int num_tokens_per_cta = num_tiles_per_cta * kNumTokensPerTile;

  Params<DataType, IndexType> params = {
      // Inputs
      .routing_scores = reinterpret_cast<DataType*>(routing_scores.data_ptr()),
      .stride_t = static_cast<int>(routing_scores.stride(0)),
      .stride_e = static_cast<int>(routing_scores.stride(1)),
      .valid_token_count = get_ptr<IndexType>(valid_token_count),
      .num_tokens = num_tokens,
      .num_tokens_per_cta = num_tokens_per_cta,
      // Buffer
      .buffered_expert_indices =
          reinterpret_cast<IndexType*>(buffered_expert_indices.data_ptr()),
      .buffered_token_indices =
          reinterpret_cast<IndexType*>(buffered_token_indices.data_ptr()),
      // Outputs
      .token_count_per_expert =
          reinterpret_cast<IndexType*>(token_count_per_expert.data_ptr()),
      .shuffled_expert_indices =
          reinterpret_cast<IndexType*>(shuffled_expert_indices.data_ptr()),
      .shuffled_token_indices =
          reinterpret_cast<IndexType*>(shuffled_token_indices.data_ptr())};

  dim3 grids(num_ctas);
  dim3 blocks(kNumThreads);
  void* args[] = {(void*)&params};
  auto stream = at::cuda::getCurrentCUDAStream();

#ifdef USE_ROCM
  // hipLaunchCooperativeKernel seems to cause incorrect memory order across
  // kernel launches.
  C10_CUDA_CHECK(
      hipLaunchKernel((void*)kernel, grids, blocks, args, smem_size, stream));
#else
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      (void*)kernel, grids, blocks, args, smem_size, stream));
#endif

  return std::make_tuple(
      token_count_per_expert, shuffled_expert_indices, shuffled_token_indices);
}

} // namespace fbgemm_gpu
