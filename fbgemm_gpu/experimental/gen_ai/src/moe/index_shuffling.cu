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
  DataType scores[NumTokensPerTile * NumExperts];
  IndexType expert_indices[NumTokensPerTile * NumExperts];
  IndexType expert_count_cumsums[NumExperts];
};

template <class DataType, class IndexType>
struct Params {
  const DataType* scores;
  int stride_t_;
  int stride_e_;
  int num_tokens;
  int num_tokens_per_cta;
  IndexType* counts;
  IndexType* expert_indices;
  IndexType* token_indices;
  IndexType* shuffled_expert_indices;
  IndexType* shuffled_token_indices;
  IndexType* num_valid_tokens;
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
      params.num_valid_tokens ? *params.num_valid_tokens : num_total_tokens;

  const int token_index_offset_start = bidx * params.num_tokens_per_cta;
  const int token_index_offset_end = std::min(
      token_index_offset_start + params.num_tokens_per_cta, num_total_tokens);

  if (token_index_offset_start >= num_total_tokens) {
    return;
  }

  const int stride_t_ = params.stride_t_;
  const int stride_e_ = params.stride_e_;

  for (int token_index_offset = token_index_offset_start;
       token_index_offset < token_index_offset_end;
       token_index_offset += NumTokensPerTile) {
    // 1. Read scores
    // TODO(shikaili): vectorized. asynchronous.
#pragma unroll
    for (int i = tidx; i < NumTokensPerTile * NumExperts; i += kNumThreads) {
      int token_index = token_index_offset + i / NumExperts;
      int expert_index = i % NumExperts;

      smem.scores[i] = token_index < num_valid_tokens
          ? params.scores[token_index * stride_t_ + expert_index * stride_e_]
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

          auto lhs_score = smem.scores[lhs_smem_index];
          auto rhs_score = smem.scores[rhs_smem_index];
          auto lhs_expert_index = smem.expert_indices[lhs_smem_index];
          auto rhs_expert_index = smem.expert_indices[rhs_smem_index];

          bool lhs_larger = lhs_score >= rhs_score;
          smem.scores[lhs_smem_index] = lhs_larger ? lhs_score : rhs_score;
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
            atomic_add_relaxed(&params.counts[expert_index], 1);
        params.expert_indices[token_index] = expert_index;
        params.token_indices[token_index] = token_index_in_expert;
      }
    }
    __syncthreads();
  }

  if (tidx == 0) {
    int processed_tokens = 0;
    int* processed_tokens_addr = &params.counts[NumExperts];

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
    smem.expert_count_cumsums[tidx] = params.counts[tidx];
  }
  __syncthreads();

  if (tidx == 0) {
    // TODO(shikaili): parallel.
#pragma unroll
    for (int i = 1; i < NumExperts; ++i) {
      smem.expert_count_cumsums[i] += smem.expert_count_cumsums[i - 1];
    }
  }
  __syncthreads();

  // 5. Store
  for (int global_token_offset = bidx * params.num_tokens_per_cta;
       global_token_offset < (bidx + 1) * params.num_tokens_per_cta;
       global_token_offset += kNumThreads) {
    int token_index = global_token_offset + tidx;
    if (token_index < num_valid_tokens) {
      int expert_index = params.expert_indices[token_index];
      int token_index_in_expert = params.token_indices[token_index];
      int new_token_index =
          (expert_index == 0 ? 0
                             : smem.expert_count_cumsums[expert_index - 1]) +
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
    const at::Tensor& scores,
    std::optional<at::Tensor> num_valid_tokens) {
  TORCH_CHECK(scores.dtype() == torch::kBFloat16);
  using DataType = __nv_bfloat16;
  using IndexType = int32_t;

  TORCH_CHECK(scores.dim() == 2);
  const int num_tokens = scores.size(0);
  const int num_experts = scores.size(1);
  TORCH_CHECK(num_experts == 16 || num_experts == 128);

  auto allocate_index_tensor = [&](int size) {
    return at::empty(
        {size}, at::TensorOptions().dtype(at::kInt).device(scores.device()));
  };
  at::Tensor counts = allocate_index_tensor(num_experts + 1);
  at::Tensor expert_indices = allocate_index_tensor(num_tokens);
  at::Tensor token_indices = allocate_index_tensor(num_tokens);
  at::Tensor shuffled_expert_indices = allocate_index_tensor(num_tokens);
  at::Tensor shuffled_token_indices = allocate_index_tensor(num_tokens);

#ifdef USE_ROCM
  counts.zero_();
  // TODO(shikaili): hipMetsetAsync is more expensive than ATen set zero.
  /*
  hipMemsetAsync(
      counts.data_ptr(),
      0,
      counts.numel() * counts.dtype().itemsize(),
      at::cuda::getCurrentCUDAStream());
  */
#else
  cudaMemsetAsync(
      counts.data_ptr(),
      0,
      counts.numel() * counts.dtype().itemsize(),
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

  int num_tokens_per_cta = ceil_of_ratio(num_tokens, num_ctas);
  const int num_tiles_per_cta =
      ceil_of_ratio(num_tokens_per_cta, kNumTokensPerTile);
  num_tokens_per_cta = num_tiles_per_cta * kNumTokensPerTile;

  Params<DataType, IndexType> params = {
      reinterpret_cast<DataType*>(scores.data_ptr()),
      static_cast<int>(scores.stride(0)),
      static_cast<int>(scores.stride(1)),
      num_tokens,
      num_tokens_per_cta,
      reinterpret_cast<IndexType*>(counts.data_ptr()),
      reinterpret_cast<IndexType*>(expert_indices.data_ptr()),
      reinterpret_cast<IndexType*>(token_indices.data_ptr()),
      reinterpret_cast<IndexType*>(shuffled_expert_indices.data_ptr()),
      reinterpret_cast<IndexType*>(shuffled_token_indices.data_ptr()),
      reinterpret_cast<IndexType*>(
          num_valid_tokens.has_value() ? num_valid_tokens->data_ptr()
                                       : nullptr)};

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
      counts, shuffled_expert_indices, shuffled_token_indices);
}

} // namespace fbgemm_gpu
