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
// #include "cuda_dispatch_utils.h"

#include <fbgemm_gpu/sparse_ops_utils.h>

#include <torch/torch.h>

#if (                         \
    defined(__CUDA_ARCH__) && \
    ((__CUDA_ARCH__ == 800) || (__CUDA_ARCH__ == 900)))
#define USE_WMMA_FRAG
#endif

namespace fbgemm_gpu {

#define DEVICE_INLINE __device__ inline __attribute__((always_inline))

static __host__ DEVICE_INLINE int32_t div_up(int32_t a, int32_t b) {
  return (a + b - 1) / b;
};

#ifdef __HIP_PLATFORM_AMD__
constexpr int32_t kThreadsPerWarp = 64;
#else
constexpr int32_t kThreadsPerWarp = 32;
#endif

#ifdef __HIP_PLATFORM_AMD__
using __nv_bfloat16 = hip_bfloat16;

typedef struct __align__(4) {
  uint16_t x;
  uint16_t y;
}
__nv_bfloat162_raw;

struct __align__(4) __nv_bfloat162 {
  __nv_bfloat16 x;
  __nv_bfloat16 y;
};

// the descriptions of __float2bfloat16 and __float2bfloat16_rn are identical
// https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____BFLOAT16__MISC.html#group__CUDA__MATH____BFLOAT16__MISC
static __host__ __device__ __nv_bfloat16 __float2bfloat16(float f) {
  __nv_bfloat16 output;
  return output.round_to_bfloat16(f);
}

static __host__ __device__ __nv_bfloat16 __float2bfloat16_rn(float f) {
  __nv_bfloat16 output;
  return output.round_to_bfloat16(f);
}

static __host__ __device__ float __bfloat162float(__nv_bfloat16 f) {
  // float output;
  // https://docs.amd.com/projects/HIP/en/docs-5.0.0/doxygen/html/hip__bfloat16_8h_source.html
  return float(f);
}

static __host__ __device__ __nv_bfloat162
__floats2bfloat162_rn(float x, float y) {
  __nv_bfloat162 output;
  output.x = __float2bfloat16_rn(x);
  output.y = __float2bfloat16_rn(y);
  return output;
}

#endif

struct __align__(16) bf16x8 {
  __nv_bfloat162 vals[4];
};

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

template <int32_t kWorldSize>
__global__ void one_shot_all_reduce(
    int32_t rank,
    int32_t world_size,
    int32_t flag,
    std::array<int32_t*, 8> barriers,
    std::array<at::BFloat16*, 8> inputs,
    at::BFloat16* ar_input,
    at::BFloat16* acc,
    at::BFloat16* output,
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
    if (acc) {
      *reinterpret_cast<uint4*>(&sums) =
          *reinterpret_cast<const uint4*>(&acc[i]);
    } else {
      memset(reinterpret_cast<void*>(&sums), 0, sizeof(sums));
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
#ifndef __HIP_PLATFORM_AMD__
    auto target_rank = at::cuda::current_device();
#else
    /*
     * This is to mitigate an issue for ROCm where the
     * device for the data ptr from hipIpcOpenMemHandle
     * is always 0, tracked in FBA-288
     */
    auto target_rank = (ii == rank ? rank : 0);
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
#ifndef __HIP_PLATFORM_AMD__
    auto target_rank = at::cuda::current_device();
#else
    auto target_rank = (ii == rank ? rank : 0);
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

template <int32_t kWorldSize>
__launch_bounds__(1024) __global__ void two_shot_all_reduce(
    int32_t rank,
    int32_t world_size,
    int32_t flag,
    std::array<int32_t*, 8> barriers,
    std::array<at::BFloat16*, 8> inputs,
    at::BFloat16* acc,
    at::BFloat16* output,
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
  int dst_rank[kWorldSize];

#pragma unroll kWorldSize
  for (int ii = 0; ii < kWorldSize; ++ii) {
    int d_rank = (rank + ii) % kWorldSize;
    src_d[ii] = inputs[d_rank];
    dst_rank[ii] = d_rank;
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
    if (acc) {
      *reinterpret_cast<uint4*>(&sums) =
          *reinterpret_cast<const uint4*>(&acc[i + N_start]);
    } else {
      memset(reinterpret_cast<void*>(&sums), 0, sizeof(sums));
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

  // barreris among the blocks with the same idx (release-acuqire semantics)
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
#pragma unroll kWorldSize
    for (int ii = 0; ii < kWorldSize; ++ii) {
      int i_r = N_start + i + (dst_rank[ii] - rank) * N_per_rank;
      *reinterpret_cast<uint4*>(&output[i_r]) =
          reinterpret_cast<const uint4*>(&src_d[ii][i_r])[0];
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
  constexpr int32_t kThreadsPerBlock = 1024;
  constexpr int32_t kMaxBlocks = 24;

  dim3 threads(0, 1, 1);
  dim3 blocks(0, 1, 1);
  if (N < N_per_thread * kThreadsPerBlock) {
    threads.x = div_up(N, N_per_warp) * kThreadsPerWarp;
    blocks.x = 1;
  } else {
    auto warps_required = div_up(N, N_per_warp);
    blocks.x = std::min<int32_t>(
        cuda_calc_block_count(div_up(N, N_per_thread), kThreadsPerBlock),
        kMaxBlocks);
    auto warps_per_block = div_up(warps_required, blocks.x);
    auto threads_per_block =
        std::min<int32_t>(kThreadsPerBlock, warps_per_block * kThreadsPerWarp);

    threads.x = threads_per_block;
  }

#define X(kWorldSize)                                               \
  if (state->world_size_ == kWorldSize) {                           \
    one_shot_all_reduce<kWorldSize>                                 \
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>( \
            state->rank_,                                           \
            state->world_size_,                                     \
            state->flag_ * state->world_size_,                      \
            barriers,                                               \
            inputs,                                                 \
            y.data_ptr<at::BFloat16>(),                             \
            z ? z->data_ptr<at::BFloat16>() : nullptr,              \
            y_allreduce.data_ptr<at::BFloat16>(),                   \
            N);                                                     \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                 \
    return;                                                         \
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

  constexpr int32_t kThreadsPerBlock = 1024;
  constexpr int32_t kMaxBlocks = 24;

  auto blocks = std::min<int32_t>(
      cuda_calc_block_count(threads_per_rank, kThreadsPerBlock), kMaxBlocks);

#define X(kWorldSize)                                                        \
  if (state->world_size_ == kWorldSize) {                                    \
    two_shot_all_reduce<kWorldSize>                                          \
        <<<blocks, kThreadsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>( \
            state->rank_,                                                    \
            state->world_size_,                                              \
            state->flag_ * state->world_size_,                               \
            barriers,                                                        \
            inputs,                                                          \
            z ? z->data_ptr<at::BFloat16>() : nullptr,                       \
            y_allreduce.data_ptr<at::BFloat16>(),                            \
            N);                                                              \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    return;                                                                  \
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

} // namespace fbgemm_gpu
