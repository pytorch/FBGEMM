/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>
#include <algorithm>
#include "c10/core/ScalarType.h"
#include "c10/util/BFloat16.h"

#if !(                                                  \
    defined(USE_ROCM) ||                                \
    ((defined(CUDA_VERSION) && CUDA_VERSION < 11000) || \
     (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
#include <cuda_bf16.h>
#elif (defined(USE_ROCM))
#include <hip/hip_bfloat16.h>
#endif

#ifndef USE_ROCM
#include <mma.h>
#endif
#include <cub/cub.cuh>

#include <torch/torch.h>
#include "fbgemm_gpu/utils/cuda_block_count.h"
#include "fbgemm_gpu/utils/vec_quant.cuh"

namespace fbgemm_gpu {

DEVICE_INLINE __nv_bfloat162
bf16hadd2(const __nv_bfloat162 x, const __nv_bfloat162 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fxl, fxh, fyl, fyh;
  fxl = __low2float(x);
  fxh = __high2float(x);
  fyl = __low2float(y);
  fyh = __high2float(y);
  return __floats2bfloat162_rn(fxl + fyl, fxh + fyh);
#elif defined(USE_ROCM)
  float fxl, fxh, fyl, fyh;
  fxl = __bfloat162float(x.x);
  fxh = __bfloat162float(x.y);
  fyl = __bfloat162float(y.x);
  fyh = __bfloat162float(y.y);
  return __floats2bfloat162_rn(fxl + fyl, fxh + fyh);
#else
  return __hadd2(x, y);
#endif
}

DEVICE_INLINE bf16x8 add_bf16x8(bf16x8 a, bf16x8 b) {
  bf16x8 c;
  c.vals[0] = bf16hadd2(a.vals[0], b.vals[0]);
  c.vals[1] = bf16hadd2(a.vals[1], b.vals[1]);
  c.vals[2] = bf16hadd2(a.vals[2], b.vals[2]);
  c.vals[3] = bf16hadd2(a.vals[3], b.vals[3]);
  return c;
}

template <int32_t kWorldSize, bool has_acc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
    __global__ void one_shot_all_reduce(
        int32_t rank,
        int32_t world_size,
        int32_t flag,
        std::array<int32_t*, 8> barriers,
        std::array<at::BFloat16*, 8> inputs,
#if defined(USE_ROCM)
        at::BFloat16* __restrict__ ar_input,
        at::BFloat16* __restrict__ acc,
        at::BFloat16* __restrict__ output,
#else
    at::BFloat16* ar_input,
    at::BFloat16* acc,
    at::BFloat16* output,
#endif
        int32_t N) {
  // It is expensive to launch hipMemcpyAsync on ROCm
  // Move data copy here. Each block copies part of input data
  at::BFloat16* input = inputs[rank];
  for (size_t i = blockDim.x * blockIdx.x * 8 + threadIdx.x * 8; i < N;
       i += (size_t)blockDim.x * gridDim.x * 8) {
#if defined(USE_ROCM)
    __builtin_nontemporal_store(
        reinterpret_cast<uint64_t*>(&ar_input[i])[0], (uint64_t*)(&input[i]));
    __builtin_nontemporal_store(
        reinterpret_cast<uint64_t*>(&ar_input[i])[1],
        (uint64_t*)(&input[i]) + 1);
#else
    *reinterpret_cast<uint64_t*>(&input[i]) =
        reinterpret_cast<uint64_t*>(&ar_input[i])[0];
    *(reinterpret_cast<uint64_t*>(&input[i]) + 1) =
        reinterpret_cast<uint64_t*>(&ar_input[i])[1];
#endif
  }
  // Synchronize the ranks.
  volatile int32_t* barrier_d = barriers[rank];
  if (threadIdx.x < kWorldSize) {
    // The 1st block notifies the other ranks.
    if (blockIdx.x == 0) {
#if defined(USE_ROCM)
      __atomic_store_n(barriers[threadIdx.x] + rank, flag, __ATOMIC_RELEASE);
#else
      barriers[threadIdx.x][rank] = flag;
#endif
    }

    // Busy-wait until all ranks are ready.
#if defined(USE_ROCM)
    while (__atomic_load_n(barrier_d + threadIdx.x, __ATOMIC_ACQUIRE) != flag) {
    }
#else
    while (barrier_d[threadIdx.x] != flag) {
    }
#endif
  }

  // Make sure we can move on...
  __syncthreads();
  // The source pointers. Distributed round-robin for the different warps.
  const at::BFloat16* src_d[kWorldSize];
#pragma unroll kWorldSize
  for (int ii = 0; ii < kWorldSize; ++ii) {
    int src_rank = (rank + ii) % kWorldSize;
    src_d[ii] = inputs[src_rank];
  }

  // Each block accumulates the values from the different GPUs on the same
  // node.
  for (size_t i = blockDim.x * blockIdx.x * 8 + threadIdx.x * 8; i < N;
       i += blockDim.x * gridDim.x * 8) {
    // Iterate over the different ranks/devices on the node to load the
    // values.
    bf16x8 vals[kWorldSize];
#pragma unroll kWorldSize
    for (int ii = 0; ii < kWorldSize; ++ii) {
      *reinterpret_cast<uint4*>(&vals[ii]) =
          reinterpret_cast<const uint4*>(&src_d[ii][i])[0];
    }

    // Sum the values from the different ranks.
    bf16x8 sums;
    if constexpr (has_acc) {
      *reinterpret_cast<uint4*>(&sums) =
          *reinterpret_cast<const uint4*>(&acc[i]);
    } else {
      *reinterpret_cast<uint4*>(&sums) = uint4{0};
    }

#pragma unroll kWorldSize
    for (int ii = 0; ii < kWorldSize; ++ii) {
      sums = add_bf16x8(sums, vals[ii]);
    }

    // Store to the destination buffer.
    *reinterpret_cast<uint4*>(&output[i]) =
        *reinterpret_cast<const uint4*>(&sums);
  }

  // barrier to sync with all other ranks on the same blockIdx
  // this is needed to ensure this-rank won't override its inputs buffer
  // (as we always do memcpy from srcbuff to inputs buffer first)
  // while other ranks are still reading them.
  __syncthreads();

  if (threadIdx.x < kWorldSize) {
    // notify all other blocks this blockIdx is ready
    const int32_t flag_block_offset = kWorldSize + blockIdx.x * kWorldSize;

#if defined(USE_ROCM)
    __atomic_store_n(
        barriers[threadIdx.x] + flag_block_offset + rank,
        flag,
        __ATOMIC_RELEASE);
#else
    barriers[threadIdx.x][flag_block_offset + rank] = flag;
#endif

    // busy-wait until all ranks are ready
#if defined(USE_ROCM)
    while (__atomic_load_n(
               barrier_d + flag_block_offset + threadIdx.x, __ATOMIC_ACQUIRE) !=
           flag) {
    }
#else
    while (barrier_d[flag_block_offset + threadIdx.x] != flag) {
    }
#endif
  }
}

struct CustomAllReduceState {
  std::vector<at::Tensor> barriers_;
  std::vector<at::Tensor> buffers_;

  int32_t rank_;
  int32_t world_size_;
  int32_t flag_{0};
};

CustomAllReduceState* get_car_state() {
  static auto* r = new CustomAllReduceState();
  return r;
}
constexpr int64_t kMaxCAR = 50 * 1024 * 1024;

void car_init(
    int64_t rank,
    int64_t world_size,
    at::Tensor local_barrier,
    std::vector<at::Tensor> all_barrier_handles,
    at::Tensor local_buffer,
    std::vector<at::Tensor> all_buffer_handles) {
  at::OptionalDeviceGuard guard(local_buffer.device());
  auto to_handle = [](at::Tensor r) {
    cudaIpcMemHandle_t handle;
    std::memcpy(&handle, r.data_ptr(), sizeof(handle));
    return handle;
  };

  auto state = get_car_state();
  state->rank_ = rank;
  state->world_size_ = world_size;
  state->flag_ = 0;
  state->buffers_.resize(world_size);
  state->barriers_.resize(world_size);
  TORCH_CHECK(world_size == all_buffer_handles.size());
  TORCH_CHECK(world_size == all_barrier_handles.size());

  for (auto ii = 0; ii < world_size; ++ii) {
    void* ptr = nullptr;
    if (ii != rank) {
      AT_CUDA_CHECK(cudaIpcOpenMemHandle(
          &ptr,
          to_handle(all_buffer_handles[ii]),
          cudaIpcMemLazyEnablePeerAccess));
    } else {
      ptr = local_buffer.data_ptr();
    }
#if (defined(USE_ROCM) && ROCM_VERSION < 60200)
    /*
     * This is to mitigate an issue for ROCm where the
     * device for the data ptr from hipIpcOpenMemHandle
     * is always 0, tracked in FBA-288
     * This issue is fixed after RoCM 6.2.0
     */
    auto target_rank = (ii == rank ? rank : 0);
#else
    auto target_rank = at::cuda::current_device();
#endif
    state->buffers_[ii] = at::from_blob(
        ptr,
        {kMaxCAR},
        at::TensorOptions()
            .dtype(at::kBFloat16)
            .device(at::Device(at::kCUDA, target_rank)));
  }
  for (auto ii = 0; ii < world_size; ++ii) {
    void* ptr = nullptr;
#if (defined(USE_ROCM) && ROCM_VERSION < 60200)
    auto target_rank = (ii == rank ? rank : 0);
#else
    auto target_rank = at::cuda::current_device();
#endif
    if (ii != rank) {
      AT_CUDA_CHECK(cudaIpcOpenMemHandle(
          &ptr,
          to_handle(all_barrier_handles[ii]),
          cudaIpcMemLazyEnablePeerAccess));
    } else {
      ptr = local_barrier.data_ptr();
    }
    state->barriers_[ii] = at::from_blob(
        ptr,
        {kMaxCAR},
        at::TensorOptions().dtype(at::kInt).device(
            at::Device(at::kCUDA, target_rank)));
  }
}

at::Tensor car_ipc_handle(at::Tensor x) {
  cudaIpcMemHandle_t handle;
  AT_CUDA_CHECK(cudaIpcGetMemHandle(&handle, x.data_ptr()));
  auto r = at::empty(
      sizeof(cudaIpcMemHandle_t), at::TensorOptions().dtype(at::kChar));
  std::memcpy(r.data_ptr(), &handle, sizeof(handle));
  return r;
}

// need to cudaMalloc ourselves to avoid caching allocator handing out wrong
// base pointer.
at::Tensor car_tensor() {
  void* ptr = nullptr;
  // 1M N
#if defined(USE_ROCM)
  // for MI300, we need to allocate uncached (fine-grained) memory so that the
  // barrier value will be visible within the kernel instead of at the kernel
  // boundary
  int flag = hipDeviceMallocUncached;
  C10_CUDA_CHECK(hipExtMallocWithFlags(&ptr, kMaxCAR * 2, flag));
#else
  C10_CUDA_CHECK(cudaMalloc(&ptr, kMaxCAR * 2));
#endif
  return at::from_blob(
      ptr,
      {kMaxCAR},
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA));
}

static DEVICE_INLINE void st_flag_release(int32_t& flag, int32_t* flag_addr) {
#if defined(USE_ROCM)
  __atomic_store_n(flag_addr, flag, __ATOMIC_RELEASE);
#elif __CUDA_ARCH__ >= 700
  asm volatile(
      "st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#else
  __threadfence_system();
  asm volatile("st.global.volatile.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#endif
}

static DEVICE_INLINE void ld_flag_acquire(int32_t& flag, int32_t* flag_addr) {
#if defined(USE_ROCM)
  flag = __atomic_load_n(flag_addr, __ATOMIC_ACQUIRE);
#elif __CUDA_ARCH__ >= 700
  asm volatile("ld.global.acquire.sys.b32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
#else
  asm volatile("ld.global.volatile.b32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
#endif
}

template <int32_t kWorldSize, bool split_last_dim>
#if defined(USE_ROCM)
__launch_bounds__(512) __global__ void reduce_scatter(
#else
__launch_bounds__(1024) __global__ void reduce_scatter(
#endif
    int32_t rank,
    int32_t world_size,
    int32_t flag,
    std::array<int32_t*, 8> barriers,
    std::array<at::BFloat16*, 8> inputs,
#if defined(USE_ROCM)
    at::BFloat16* __restrict__ output,
#else
    at::BFloat16* output,
#endif
    int32_t last_dim,
    int32_t N) {
  int32_t N_per_rank = N / kWorldSize;
  int32_t N_start = N_per_rank * rank;
  int32_t N_last_dim = last_dim / kWorldSize;

  if constexpr (split_last_dim) {
    N_start = last_dim / kWorldSize * rank;
  }

  // Synchronize the ranks.
  volatile int32_t* barrier_d = barriers[rank];
  if (threadIdx.x < kWorldSize) {
    // The 1st block notifies the other ranks.
    if (blockIdx.x == 0) {
#if defined(USE_ROCM)
      __atomic_store_n(barriers[threadIdx.x] + rank, flag, __ATOMIC_RELEASE);
#else
      barriers[threadIdx.x][rank] = flag;
#endif
    }

    // Busy-wait until all ranks are ready.
#if defined(USE_ROCM)
    while (__atomic_load_n(barrier_d + threadIdx.x, __ATOMIC_ACQUIRE) != flag) {
    }
#else
    while (barrier_d[threadIdx.x] != flag) {
    }
#endif
  }

  __syncthreads();

  at::BFloat16* src_d[kWorldSize];

#pragma unroll kWorldSize
  for (int ii = 0; ii < kWorldSize; ++ii) {
    int d_rank = (rank + ii) % kWorldSize;
    src_d[ii] = inputs[d_rank];
  }

  // Each block accumulates the values from the different GPUs on the same
  // node.
  for (size_t i = threadIdx.x * 8 + blockIdx.x * blockDim.x * 8; i < N_per_rank;
       i += gridDim.x * blockDim.x * 8) {
    bf16x8 vals[kWorldSize];
    size_t idx = i;
    if constexpr (split_last_dim) {
      idx = i / N_last_dim * last_dim + i % N_last_dim;
    }
#pragma unroll kWorldSize
    for (int ii = 0; ii < kWorldSize; ++ii) {
      *reinterpret_cast<uint4*>(
          &vals[(ii + kWorldSize - rank) & (kWorldSize - 1)]) =
          reinterpret_cast<const uint4*>(&src_d[ii][idx + N_start])[0];
    }

    bf16x8 sums;
    *reinterpret_cast<uint4*>(&sums) = uint4{0};

#pragma unroll kWorldSize
    for (int ii = 0; ii < kWorldSize; ++ii) {
      sums = add_bf16x8(sums, vals[ii]);
    }

    // Store to the local buffer.
    *reinterpret_cast<uint4*>(&output[i]) =
        *reinterpret_cast<const uint4*>(&sums);
  }

  __syncthreads();

  // barriers among the blocks with the same idx (release-acuqire semantics)
  if (threadIdx.x < kWorldSize) {
    // The all blocks notifies the other ranks.
    int32_t flag_block_offset = kWorldSize + blockIdx.x * kWorldSize;
    st_flag_release(flag, barriers[threadIdx.x] + flag_block_offset + rank);

    // Busy-wait until all ranks are ready.
    int32_t rank_barrier = 0;
    int32_t* peer_barrier_d = barriers[rank] + flag_block_offset + threadIdx.x;
    do {
      ld_flag_acquire(rank_barrier, peer_barrier_d);
    } while (rank_barrier != flag);
  }
}

template <int32_t kWorldSize, bool has_acc>
#if defined(USE_ROCM)
__launch_bounds__(512) __global__ void two_shot_all_reduce(
#else
__launch_bounds__(1024) __global__ void two_shot_all_reduce(
#endif
    int32_t rank,
    int32_t world_size,
    int32_t flag,
    std::array<int32_t*, 8> barriers,
    std::array<at::BFloat16*, 8> inputs,
#if defined(USE_ROCM)
    at::BFloat16* __restrict__ acc,
    at::BFloat16* __restrict__ output,
#else
    at::BFloat16* acc,
    at::BFloat16* output,
#endif
    int32_t N) {
  int32_t N_per_rank = N / kWorldSize;
  int32_t N_start = N_per_rank * rank;

  // Synchronize the ranks.
  volatile int32_t* barrier_d = barriers[rank];
  if (threadIdx.x < kWorldSize) {
    // The 1st block notifies the other ranks.
    if (blockIdx.x == 0) {
#if defined(USE_ROCM)
      __atomic_store_n(barriers[threadIdx.x] + rank, flag, __ATOMIC_RELEASE);
#else
      barriers[threadIdx.x][rank] = flag;
#endif
    }

    // Busy-wait until all ranks are ready.
#if defined(USE_ROCM)
    while (__atomic_load_n(barrier_d + threadIdx.x, __ATOMIC_ACQUIRE) != flag) {
    }
#else
    while (barrier_d[threadIdx.x] != flag) {
    }
#endif
  }

  __syncthreads();

  at::BFloat16* src_d[kWorldSize];

#pragma unroll kWorldSize
  for (int ii = 0; ii < kWorldSize; ++ii) {
    int d_rank = (rank + ii) % kWorldSize;
    src_d[ii] = inputs[d_rank];
  }

  // Each block accumulates the values from the different GPUs on the same
  // node.
  for (size_t i = threadIdx.x * 8 + blockIdx.x * blockDim.x * 8; i < N_per_rank;
       i += gridDim.x * blockDim.x * 8) {
    bf16x8 vals[kWorldSize];
#pragma unroll kWorldSize
    for (int ii = 0; ii < kWorldSize; ++ii) {
      *reinterpret_cast<uint4*>(&vals[ii]) =
          reinterpret_cast<const uint4*>(&src_d[ii][i + N_start])[0];
    }

    bf16x8 sums;

    if constexpr (has_acc) {
      *reinterpret_cast<uint4*>(&sums) =
          *reinterpret_cast<const uint4*>(&acc[i + N_start]);
    } else {
      *reinterpret_cast<uint4*>(&sums) = uint4{0};
    }

#pragma unroll kWorldSize
    for (int ii = 0; ii < kWorldSize; ++ii) {
      sums = add_bf16x8(sums, vals[ii]);
    }

    // Store to the local buffer.
    *reinterpret_cast<uint4*>(&src_d[0][i + N_start]) =
        *reinterpret_cast<const uint4*>(&sums);
  }

  __syncthreads();

  // barriers among the blocks with the same idx (release-acuqire semantics)
  if (threadIdx.x < kWorldSize) {
    // The all blocks notifies the other ranks.
    int32_t flag_block_offset = kWorldSize + blockIdx.x * kWorldSize;
    st_flag_release(flag, barriers[threadIdx.x] + flag_block_offset + rank);

    // Busy-wait until all ranks are ready.
    int32_t rank_barrier = 0;
    int32_t* peer_barrier_d = barriers[rank] + flag_block_offset + threadIdx.x;
    do {
      ld_flag_acquire(rank_barrier, peer_barrier_d);
    } while (rank_barrier != flag);
  }

  __syncthreads();

  // Gather all needed elts from other intra-node ranks
  for (size_t i = threadIdx.x * 8 + blockIdx.x * blockDim.x * 8; i < N_per_rank;
       i += gridDim.x * blockDim.x * 8) {
    uint4 temp[kWorldSize];
#pragma unroll kWorldSize
    for (int ii = 0; ii < kWorldSize; ++ii) {
      int d_rank = (rank + ii) % kWorldSize;
      int i_r = N_start + i + (d_rank - rank) * N_per_rank;
      temp[ii] = reinterpret_cast<const uint4*>(&src_d[ii][i_r])[0];
    }

#pragma unroll kWorldSize
    for (int ii = 0; ii < kWorldSize; ++ii) {
      int d_rank = (rank + ii) % kWorldSize;
      int i_r = N_start + i + (d_rank - rank) * N_per_rank;
      *reinterpret_cast<uint4*>(&output[i_r]) = temp[ii];
    }
  }
}

void one_shot_car_allreduce(
    at::Tensor y_allreduce,
    at::Tensor y,
    std::optional<at::Tensor> z,
    int64_t comm_idx) { // match the API with nccl_allreduce in
                        // https://fburl.com/code/v538vig9
  c10::cuda::CUDAGuard gg(y_allreduce.device());
  TORCH_CHECK(y_allreduce.is_contiguous());
  TORCH_CHECK(y.is_contiguous());
  TORCH_CHECK(y.numel() == y_allreduce.numel());
  TORCH_CHECK(y.numel() % 8 == 0);
  TORCH_CHECK(y.numel() < kMaxCAR);
  const auto N = y.numel();
  if (N == 0) {
    // no data to allreduce, return
    return;
  }
  if (z) {
    TORCH_CHECK(z->numel() == y.numel());
  }
  auto state = get_car_state();
  ++state->flag_;

  std::array<at::BFloat16*, 8> inputs;
  for (auto ii = 0; ii < state->world_size_; ++ii) {
    inputs[ii] = state->buffers_[ii].data_ptr<at::BFloat16>();
  }

  std::array<int32_t*, 8> barriers;
  for (auto ii = 0; ii < state->world_size_; ++ii) {
    barriers[ii] = state->barriers_[ii].data_ptr<int32_t>();
  }

  constexpr int32_t N_per_thread = 8;
  constexpr int32_t N_per_warp = N_per_thread * kThreadsPerWarp;
  TORCH_CHECK(N % N_per_warp == 0);
#if defined(USE_ROCM)
  constexpr int32_t kThreadsPerBlock = 512;
#else
  constexpr int32_t kThreadsPerBlock = 1024;
#endif
  constexpr int32_t kMaxBlocks = 24;

  dim3 threads(0, 1, 1);
  dim3 blocks(0, 1, 1);
  if (N < N_per_thread * kThreadsPerBlock) {
    threads.x = div_round_up(N, N_per_warp) * kThreadsPerWarp;
    blocks.x = 1;
  } else {
    auto warps_required = div_round_up(N, N_per_warp);
    blocks.x = std::min<int32_t>(
        cuda_calc_block_count(div_round_up(N, N_per_thread), kThreadsPerBlock),
        kMaxBlocks);
    auto warps_per_block = div_round_up(warps_required, blocks.x);
    auto threads_per_block =
        std::min<int32_t>(kThreadsPerBlock, warps_per_block * kThreadsPerWarp);

    threads.x = threads_per_block;
  }

#define X(kWorldSize)                                                 \
  if (state->world_size_ == kWorldSize) {                             \
    if (z) {                                                          \
      one_shot_all_reduce<kWorldSize, true>                           \
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>( \
              state->rank_,                                           \
              state->world_size_,                                     \
              state->flag_ * state->world_size_,                      \
              barriers,                                               \
              inputs,                                                 \
              y.data_ptr<at::BFloat16>(),                             \
              z->data_ptr<at::BFloat16>(),                            \
              y_allreduce.data_ptr<at::BFloat16>(),                   \
              N);                                                     \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                 \
      return;                                                         \
    } else {                                                          \
      one_shot_all_reduce<kWorldSize, false>                          \
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>( \
              state->rank_,                                           \
              state->world_size_,                                     \
              state->flag_ * state->world_size_,                      \
              barriers,                                               \
              inputs,                                                 \
              y.data_ptr<at::BFloat16>(),                             \
              nullptr,                                                \
              y_allreduce.data_ptr<at::BFloat16>(),                   \
              N);                                                     \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                 \
      return;                                                         \
    }                                                                 \
  }

  TORCH_CHECK(
      state->world_size_ == 2 || state->world_size_ == 4 ||
      state->world_size_ == 8);
  X(2);
  X(4);
  X(8);

#undef X
  return;
} // namespace fbgemm_gpu

void two_shot_car_allreduce(
    at::Tensor y_allreduce,
    at::Tensor y,
    std::optional<at::Tensor> z,
    int64_t comm_idx) { // match the API with nccl_allreduce in
                        // https://fburl.com/code/v538vig9
  c10::cuda::CUDAGuard gg(y_allreduce.device());
  TORCH_CHECK(y_allreduce.is_contiguous());
  TORCH_CHECK(y.is_contiguous());
  TORCH_CHECK(y.numel() == y_allreduce.numel());
  TORCH_CHECK(y.numel() % 8 == 0);
  TORCH_CHECK(y.numel() < kMaxCAR);
  const auto N = y.numel();
  if (z) {
    TORCH_CHECK(z->numel() == y.numel());
  }
  auto state = get_car_state();
  ++state->flag_;

  std::array<at::BFloat16*, 8> inputs;
  for (auto ii = 0; ii < state->world_size_; ++ii) {
    inputs[ii] = state->buffers_[ii].data_ptr<at::BFloat16>();
  }

  std::array<int32_t*, 8> barriers;
  for (auto ii = 0; ii < state->world_size_; ++ii) {
    barriers[ii] = state->barriers_[ii].data_ptr<int32_t>();
  }

  AT_CUDA_CHECK(cudaMemcpyAsync(
      inputs[state->rank_],
      y.data_ptr<at::BFloat16>(),
      y.numel() * y.element_size(),
      cudaMemcpyDeviceToDevice,
      at::cuda::getCurrentCUDAStream()));

  constexpr int32_t N_per_thread = 8;
  TORCH_CHECK(N % state->world_size_ == 0);
  const auto N_per_rank = N / state->world_size_;

  TORCH_CHECK(N_per_rank % N_per_thread == 0);
  auto threads_per_rank = N_per_rank / N_per_thread;

#if defined(USE_ROCM)
  constexpr int32_t kThreadsPerBlock = 512;
#else
  constexpr int32_t kThreadsPerBlock = 1024;
#endif

  constexpr int32_t kMaxBlocks = 24;

  auto blocks = std::min<int32_t>(
      cuda_calc_block_count(threads_per_rank, kThreadsPerBlock), kMaxBlocks);

#define X(kWorldSize)                                                          \
  if (state->world_size_ == kWorldSize) {                                      \
    if (z) {                                                                   \
      two_shot_all_reduce<kWorldSize, true>                                    \
          <<<blocks, kThreadsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>( \
              state->rank_,                                                    \
              state->world_size_,                                              \
              state->flag_ * state->world_size_,                               \
              barriers,                                                        \
              inputs,                                                          \
              z->data_ptr<at::BFloat16>(),                                     \
              y_allreduce.data_ptr<at::BFloat16>(),                            \
              N);                                                              \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
      return;                                                                  \
    } else {                                                                   \
      two_shot_all_reduce<kWorldSize, false>                                   \
          <<<blocks, kThreadsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>( \
              state->rank_,                                                    \
              state->world_size_,                                              \
              state->flag_ * state->world_size_,                               \
              barriers,                                                        \
              inputs,                                                          \
              nullptr,                                                         \
              y_allreduce.data_ptr<at::BFloat16>(),                            \
              N);                                                              \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
      return;                                                                  \
    }                                                                          \
  }

  TORCH_CHECK(
      state->world_size_ == 2 || state->world_size_ == 4 ||
      state->world_size_ == 8);
  X(2);
  X(4);
  X(8);

#undef X
  return;
}

void car_reducescatter(
    at::Tensor dst,
    at::Tensor src,
    bool split_last_dim,
    int64_t comm_idx) { // match the API with nccl_allreduce in
                        // https://fburl.com/code/v538vig9
  auto state = get_car_state();
  c10::cuda::CUDAGuard gg(dst.device());
  TORCH_CHECK(dst.is_contiguous());
  TORCH_CHECK(src.is_contiguous());
  TORCH_CHECK((state->world_size_ * dst.numel()) == src.numel());
  TORCH_CHECK(src.numel() % 8 == 0);
  TORCH_CHECK(src.numel() < kMaxCAR);
  TORCH_CHECK(
      state->world_size_ == 2 || state->world_size_ == 4 ||
      state->world_size_ == 8);

  const auto N = src.numel();

  if (N == 0) {
    return;
  }
  ++state->flag_;

  std::array<at::BFloat16*, 8> inputs;
  for (auto ii = 0; ii < state->world_size_; ++ii) {
    inputs[ii] = state->buffers_[ii].data_ptr<at::BFloat16>();
  }

  AT_CUDA_CHECK(cudaMemcpyAsync(
      inputs[state->rank_],
      src.data_ptr<at::BFloat16>(),
      src.numel() * src.element_size(),
      cudaMemcpyDeviceToDevice,
      at::cuda::getCurrentCUDAStream()));

  std::array<int32_t*, 8> barriers;
  for (auto ii = 0; ii < state->world_size_; ++ii) {
    barriers[ii] = state->barriers_[ii].data_ptr<int32_t>();
  }

  constexpr int32_t N_per_thread = 8;
  TORCH_CHECK(N % state->world_size_ == 0);
  const auto N_per_rank = N / state->world_size_;

  TORCH_CHECK(N_per_rank % N_per_thread == 0);
  auto threads_per_rank = div_round_up(N_per_rank, N_per_thread);

#if defined(USE_ROCM)
  constexpr int32_t kThreadsPerBlock = 512;
#else
  constexpr int32_t kThreadsPerBlock = 1024;
#endif

  constexpr int32_t kMaxBlocks = 24;

  auto blocks = std::min<int32_t>(
      cuda_calc_block_count(threads_per_rank, kThreadsPerBlock), kMaxBlocks);

#define X(kWorldSize)                                                          \
  if (state->world_size_ == kWorldSize) {                                      \
    if (split_last_dim) {                                                      \
      reduce_scatter<kWorldSize, true>                                         \
          <<<blocks, kThreadsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>( \
              state->rank_,                                                    \
              state->world_size_,                                              \
              state->flag_ * state->world_size_,                               \
              barriers,                                                        \
              inputs,                                                          \
              dst.data_ptr<at::BFloat16>(),                                    \
              src.size(-1),                                                    \
              N);                                                              \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
      return;                                                                  \
    } else {                                                                   \
      reduce_scatter<kWorldSize, false>                                        \
          <<<blocks, kThreadsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>( \
              state->rank_,                                                    \
              state->world_size_,                                              \
              state->flag_ * state->world_size_,                               \
              barriers,                                                        \
              inputs,                                                          \
              dst.data_ptr<at::BFloat16>(),                                    \
              src.size(-1),                                                    \
              N);                                                              \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
      return;                                                                  \
    }                                                                          \
  }

  X(2);
  X(4);
  X(8);

#undef X
  return;
} // namespace fbgemm_gpu

} // namespace fbgemm_gpu
