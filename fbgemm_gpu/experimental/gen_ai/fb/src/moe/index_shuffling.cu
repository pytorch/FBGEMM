// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/torch.h>

namespace fbgemm_gpu {

#ifndef USE_ROCM
namespace {

constexpr int kNumThreadsPerWarp = 32;
constexpr int kNumWarps = 4;
constexpr int kNumThreads = kNumThreadsPerWarp * kNumWarps;

static int num_sms = -1;

__inline__ constexpr int ceil_of_ratio(int a, int b) {
  return (a + b - 1) / b;
};

template <class DataType, class IndexType, int NumExperts, int NumTokensPerTile>
struct SharedStorage {
  DataType scores[NumTokensPerTile * NumExperts];
  IndexType expert_indices[NumTokensPerTile * NumExperts];
  IndexType expert_count_cumsums[NumExperts];
};

template <class DataType, class IndexType>
struct Params {
  const DataType* scores;
  int num_tokens;
  int num_tokens_per_cta;
  IndexType* counts;
  IndexType* expert_indices;
  IndexType* token_indices;
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

  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;

  const int token_index_offset_start = bidx * params.num_tokens_per_cta;
  const int token_index_offset_end = std::min(
      token_index_offset_start + params.num_tokens_per_cta, params.num_tokens);

  if (token_index_offset_start >= params.num_tokens) {
    return;
  }

  for (int token_index_offset = token_index_offset_start;
       token_index_offset < token_index_offset_end;
       token_index_offset += NumTokensPerTile) {
    // 1. Read scores
    // TODO(shikaili): vectorized. asynchronous.
#pragma unroll
    for (int i = tidx; i < NumTokensPerTile * NumExperts; i += kNumThreads) {
      int token_index = token_index_offset + i / NumExperts;
      int expert_index = i % NumExperts;

      smem.scores[i] = token_index < params.num_tokens
          ? params.scores[token_index * NumExperts + expert_index]
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
      if constexpr (kNumParallelReductionThreads <= kNumThreadsPerWarp) {
        __syncwarp();
      } else {
        __syncthreads();
      }
    }
    if constexpr (kNumParallelReductionThreads > kNumThreadsPerWarp) {
      __syncthreads();
    }

    // 3. Counting
#pragma unroll
    for (int i = tidx; i < NumTokensPerTile; i += kNumThreads) {
      int local_token_index = i;
      int token_index = token_index_offset + i;
      if (token_index < params.num_tokens) {
        auto expert_index = smem.expert_indices[local_token_index * NumExperts];
        auto token_index_in_expert = atomicAdd(&params.counts[expert_index], 1);
        params.expert_indices[token_index] = expert_index;
        params.token_indices[token_index] = token_index_in_expert;
      }
    }
    __syncthreads();

    if (tidx == 0) {
      atomicAdd(
          &params.counts[NumExperts],
          std::min(
              NumTokensPerTile, token_index_offset_end - token_index_offset));
    }
  }

  int processed_tokens = 0;
  int* processed_tokens_counter = &params.counts[NumExperts];

  if (tidx == 0) {
    do {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(processed_tokens)
                   : "l"(processed_tokens_counter));
    } while (processed_tokens != params.num_tokens);
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
    if (token_index < params.num_tokens) {
      int expert_index = params.expert_indices[token_index];
      int token_index_in_expert = params.token_indices[token_index];
      int new_token_index =
          (expert_index == 0 ? 0
                             : smem.expert_count_cumsums[expert_index - 1]) +
          token_index_in_expert;
      params.shuffled_expert_indices[new_token_index] = expert_index;
      params.shuffled_token_indices[new_token_index] = token_index;
    }
  }
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> index_shuffling_torch(
    const at::Tensor& scores) {
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

  counts.zero_();

  // Avoid expensive `cudaGetDeviceProperties` call.
  if (num_sms < 0) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    num_sms = deviceProp.multiProcessorCount;
  }

  constexpr int kNumTokensPerTile = 16;
  int num_tokens_per_tile = kNumTokensPerTile;

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

  const int num_tiles = ceil_of_ratio(num_tokens, num_tokens_per_tile);
  const int num_ctas = std::min(num_tiles, num_sms);

  int num_tokens_per_cta = ceil_of_ratio(num_tokens, num_ctas);
  const int num_tiles_per_cta =
      ceil_of_ratio(num_tokens_per_cta, num_tokens_per_tile);

  num_tokens_per_cta = num_tiles_per_cta * num_tokens_per_tile;

  Params<DataType, IndexType> params = {
      reinterpret_cast<DataType*>(scores.data_ptr()),
      num_tokens,
      num_tokens_per_cta,
      reinterpret_cast<IndexType*>(counts.data_ptr()),
      reinterpret_cast<IndexType*>(expert_indices.data_ptr()),
      reinterpret_cast<IndexType*>(token_indices.data_ptr()),
      reinterpret_cast<IndexType*>(shuffled_expert_indices.data_ptr()),
      reinterpret_cast<IndexType*>(shuffled_token_indices.data_ptr())};

  dim3 grids(num_ctas);
  dim3 blocks(kNumThreads);
  void* args[] = {(void*)&params};
  auto stream = at::cuda::getCurrentCUDAStream();

  C10_CUDA_CHECK(
      cudaLaunchKernel((void*)kernel, grids, blocks, args, smem_size, stream));

  return std::make_tuple(
      counts, shuffled_expert_indices, shuffled_token_indices);
}
#endif

} // namespace fbgemm_gpu
