// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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

#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
#include <cuda_fp8.h>
#elif (defined(USE_ROCM) && ROCM_VERSION >= 60200)
#include <hip/hip_fp8.h>
#endif

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200)
using __nv_fp8_e4m3 = __hip_fp8_e4m3_fnuz;
#endif

#if (                         \
    defined(__CUDA_ARCH__) && \
    ((__CUDA_ARCH__ == 800) || (__CUDA_ARCH__ == 900)))
#define USE_WMMA_FRAG
#endif

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200)
using __nv_fp8_e4m3 = __hip_fp8_e4m3_fnuz;
#endif

namespace fbgemm_gpu {

#ifndef __HIP_PLATFORM_AMD__
struct __align__(16) bfx8 {
  __nv_bfloat162 vals[4];
};
struct __align__(8) bfx4 {
  __nv_bfloat162 vals[2];
};

#define DEVICE_INLINE __device__ inline __attribute__((always_inline))

#ifdef __HIP_PLATFORM_AMD__
constexpr int32_t kThreadsPerWarp = 64;
constexpr int32_t kWarpsPerBlock = 16;
#else
constexpr int32_t kThreadsPerWarp = 32;
constexpr int32_t kWarpsPerBlock = 32;
#endif

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200) || \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
DEVICE_INLINE bfx4 dequantize_packed_fp8(uint32_t vs, __half2 shift_scale_0);
#endif

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace {
__device__ __forceinline__ int32_t
index(int32_t total_col, int32_t row, int32_t col) {
  // don't worry about overflow because num_experts is relatively small
  return row * total_col + col;
}
} // namespace

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200) || \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
DEVICE_INLINE bfx4 dequantize_packed_fp8(uint32_t vs, __half2 shift_scale_0) {
  uint32_t v = vs;
  __nv_fp8_e4m3* fp8_k = reinterpret_cast<__nv_fp8_e4m3*>(&v); // 4 element

  auto shift_0 = __half2float(__high2half(shift_scale_0));
  auto scale_0 = __half2float(__low2half(shift_scale_0));

  // now, dequantize
  auto r0 = make_float2(
      float(fp8_k[0]) * scale_0 + shift_0, float(fp8_k[1]) * scale_0 + shift_0);
  auto r1 = make_float2(
      float(fp8_k[2]) * scale_0 + shift_0, float(fp8_k[3]) * scale_0 + shift_0);

  bfx4 result;
  result.vals[0] = __floats2bfloat162_rn(r0.x, r0.y);
  result.vals[1] = __floats2bfloat162_rn(r1.x, r1.y);
  return result;
}

DEVICE_INLINE bfx8 dequantize_packed_fp8(
    uint64_t v, // Vq1 Vq0 Kq1 Kq0
    __half2 shift_scale_k,
    __half2 shift_scale_v) {
  uint32_t k_ = v & 0xFFFFFFFF; // 32 LSB
  __nv_fp8_e4m3* fp8_k = reinterpret_cast<__nv_fp8_e4m3*>(&k_);
  v >>= 32;
  uint32_t v_ = v & 0xFFFFFFFF;
  __nv_fp8_e4m3* fp8_v = reinterpret_cast<__nv_fp8_e4m3*>(&v_);

  auto shift_0 = __half2float(__high2half(shift_scale_k));
  auto scale_0 = __half2float(__low2half(shift_scale_k));
  auto shift_1 = __half2float(__high2half(shift_scale_v));
  auto scale_1 = __half2float(__low2half(shift_scale_v));

  // now, dequantize
  auto r0 = make_float2(
      float(fp8_k[0]) * scale_0 + shift_0, float(fp8_k[1]) * scale_0 + shift_0);
  auto r1 = make_float2(
      float(fp8_k[2]) * scale_0 + shift_0, float(fp8_k[3]) * scale_0 + shift_0);
  auto r2 = make_float2(
      float(fp8_v[0]) * scale_1 + shift_1, float(fp8_v[1]) * scale_1 + shift_1);
  auto r3 = make_float2(
      float(fp8_v[2]) * scale_1 + shift_1, float(fp8_v[3]) * scale_1 + shift_1);

  bfx8 result;
  result.vals[0] = __floats2bfloat162_rn(r0.x, r0.y); // (k0_dq, k1_dq)
  result.vals[1] = __floats2bfloat162_rn(r1.x, r1.y);
  result.vals[2] = __floats2bfloat162_rn(r2.x, r2.y); // (v0_dq, v1_dq)
  result.vals[3] = __floats2bfloat162_rn(r3.x, r3.y);
  return result;
}
#else
DEVICE_INLINE void quantize_fp8_kv(fx4 dst, uint8_t* dst_row_q) {}
std::vector<at::Tensor> quantize_fp8_per_tensor(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub) { // scale upperbound
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

std::tuple<at::Tensor, at::Tensor> dequantize_fp8_cache(
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor kv_seqlen) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}
#endif

__global__ void dequantize_fp8_cache_kernel(
    // This code currently represents FP8 version not int4
    at::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_K, // [B][MAX_T][N_KVH][D_H]
    at::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_V, // [B][MAX_T][N_KVH][D_H // G]
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> kv_seqlen,
    at::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_K_dq, // [B][MAX_T][N_KVH][D_H]
    at::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_V_dq // [B][MAX_T][N_KVH][D_H]
) {
  auto N_KVH = cache_K.size(2);
  auto MAX_T = cache_K.size(1);
  auto D_H = cache_K_dq.size(3);
  auto D_H_q = cache_K.size(3);
  CUDA_KERNEL_ASSERT(D_H_q - D_H == 4);

  auto b = blockIdx.x;
  // only need to dequantize this far.
  auto max_t = kv_seqlen[b];

  // one warp per T/H
  for (auto t_h = threadIdx.y + blockIdx.y * blockDim.y; t_h < max_t * N_KVH;
       t_h += blockDim.y * gridDim.y) {
    auto h = t_h % N_KVH;
    auto t = t_h / N_KVH;

    auto* row_k = &cache_K[b][t][h][0]; // uint8_t*
    auto* row_v = &cache_V[b][t][h][0];
    bfx8 kv_dq;
    __half2 k_shift_scale;
    __half2 v_shift_scale;
    *reinterpret_cast<uint32_t*>(&k_shift_scale) =
        *reinterpret_cast<uint32_t*>(&row_k[0]); // reads 32 bits
    *reinterpret_cast<uint32_t*>(&v_shift_scale) =
        *reinterpret_cast<uint32_t*>(&row_v[0]);
    if (4 * threadIdx.x >= D_H) {
      continue;
    }
    // each thread reads 4 x 8 bits

    uint64_t kq = *reinterpret_cast<uint32_t*>(&row_k[threadIdx.x * 4 + 4]);
    uint64_t vq = *reinterpret_cast<uint32_t*>(&row_v[threadIdx.x * 4 + 4]);

    uint64_t packed = kq | (vq << 32);

    kv_dq = dequantize_packed_fp8(packed, k_shift_scale, v_shift_scale);

    // now, write our outputs
    auto* row_k_dq = &cache_K_dq[b][t][h][0];
    auto* row_v_dq = &cache_V_dq[b][t][h][0];
    // each thread writes 4 elements of type bf16
    *reinterpret_cast<uint2*>(&row_k_dq[4 * threadIdx.x]) =
        *reinterpret_cast<uint2*>(&kv_dq.vals[0]);
    *reinterpret_cast<uint2*>(&row_v_dq[4 * threadIdx.x]) =
        *reinterpret_cast<uint2*>(&kv_dq.vals[2]);
  }
}
std::tuple<at::Tensor, at::Tensor> dequantize_fp8_cache(
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor kv_seqlen) {
  TORCH_CHECK(cache_K.is_cuda());
  TORCH_CHECK(cache_V.is_cuda());
  TORCH_CHECK(kv_seqlen.is_cuda());
  auto B = cache_K.size(0);
  auto MAX_T = cache_K.size(1);
  auto N_KVH = cache_K.size(2);
  auto D_HQ = cache_K.size(3);
  auto num_groups = 1;
  auto fp8_qparam_offset = num_groups * 4;
  auto D_H = (D_HQ - fp8_qparam_offset);

  auto cache_K_dq =
      at::empty({B, MAX_T, N_KVH, D_H}, cache_K.options().dtype(at::kBFloat16));
  auto cache_V_dq =
      at::empty({B, MAX_T, N_KVH, D_H}, cache_K.options().dtype(at::kBFloat16));

  if (B == 0) {
    return {cache_K_dq, cache_V_dq};
  }

  constexpr int32_t kMaxBlocks = 256;
  dim3 blocks(B, std::max<int32_t>(1, kMaxBlocks / B));
  dim3 threads(kThreadsPerWarp, kWarpsPerBlock);
  dequantize_fp8_cache_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      cache_K.packed_accessor64<uint8_t, 4, at::RestrictPtrTraits>(),
      cache_V.packed_accessor64<uint8_t, 4, at::RestrictPtrTraits>(),
      kv_seqlen.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      cache_K_dq.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>(),
      cache_V_dq.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {cache_K_dq, cache_V_dq};
}

/**
 * moe_align_block_size
 **/
#define DISPATCH_CASE_FLOATING_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define DISPATCH_CASE_INTEGRAL_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    scalar_t* __restrict__ topk_ids,
    int32_t* sorted_token_ids,
    int32_t* expert_ids,
    int32_t* total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel) {
  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  extern __shared__ int32_t shared_mem[];

  int32_t* tokens_cnts =
      shared_mem; // 2d tensor with shape (num_experts + 1, num_experts)
  int32_t* cumsum = shared_mem +
      (num_experts + 1) * num_experts; // 1d tensor with shape (num_experts + 1)

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[index(num_experts, threadIdx.x + 1, i)] = 0;
  }

  /**
   * In the first step we compute token_cnts[thread_index + 1][expert_index],
   * which counts how many tokens in the token shard of thread_index are
   * assigned to expert expert_index.
   */
  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    ++tokens_cnts[index(num_experts, threadIdx.x + 1, topk_ids[i])];
  }

  __syncthreads();

  // For each expert we accumulate the token counts from the different threads.
  tokens_cnts[index(num_experts, 0, threadIdx.x)] = 0;
  for (int i = 1; i <= blockDim.x; ++i) {
    tokens_cnts[index(num_experts, i, threadIdx.x)] +=
        tokens_cnts[index(num_experts, i - 1, threadIdx.x)];
  }

  __syncthreads();

  // We accumulate the token counts of all experts in thread 0.
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] +
          CEILDIV(
              tokens_cnts[index(num_experts, blockDim.x, i - 1)], block_size) *
              block_size;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }

  __syncthreads();

  /**
   * For each expert, each thread processes the tokens of the corresponding
   * blocks and stores the corresponding expert_id for each block.
   */
  for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
       i += block_size) {
    expert_ids[i / block_size] = threadIdx.x;
  }

  /**
   * Each thread processes a token shard, calculating the index of each token
   * after sorting by expert number. Given the example topk_ids =
   * [0,1,2,1,2,3,0,3,4] and block_size = 4, then the output would be [0, 6, *,
   * *, 1, 3, *, *, 2, 4, *, *, 5, 7, *, *, 8, *, *, *], where * represents a
   * padding value(preset in python).
   */
  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int32_t expert_id = topk_ids[i];
    /** The cumsum[expert_id] stores the starting index of the tokens that the
     * expert with expert_id needs to process, and
     * tokens_cnts[threadIdx.x][expert_id] stores the indices of the tokens
     * processed by the expert with expert_id within the current thread's token
     * shard.
     */
    int32_t rank_post_pad =
        tokens_cnts[index(num_experts, threadIdx.x, expert_id)] +
        cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[index(num_experts, threadIdx.x, expert_id)];
  }
}

void moe_align_block_size(
    torch::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
        // calc needed amount of shared mem for `tokens_cnts` and `cumsum`
        // tensors
        const int32_t shared_mem =
            ((num_experts + 1) * num_experts + (num_experts + 1)) *
            sizeof(int32_t);

        // set dynamic shared mem
        auto kernel = moe_align_block_size_kernel<scalar_t>;
        kernel<<<1, num_experts, shared_mem, stream>>>(
            topk_ids.data_ptr<scalar_t>(),
            sorted_token_ids.data_ptr<int32_t>(),
            experts_ids.data_ptr<int32_t>(),
            num_tokens_post_pad.data_ptr<int32_t>(),
            num_experts,
            block_size,
            topk_ids.numel());
      });
}

/**
 * scaled_fp8_quant
 **/
#if (defined(USE_ROCM) && ROCM_VERSION >= 60200) || \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
class FP8_E4M3_MAX {
 public:
#ifndef USE_ROCM
  static constexpr float value = 448.0;
#else
  static constexpr float value = 240.0;
#endif
};
class FP8_E5M2_MAX {
 public:
  static constexpr float value = 57344.0;
};
#endif

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  float old;
  old = (value >= 0)
      ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
      : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

  return old;
}

template <bool is_scale_inverted>
__device__ __forceinline__ c10::Float8_e4m3fn scaled_fp8_conversion(
    float const val,
    float const scale) {
  float x = 0.0f;
  if constexpr (is_scale_inverted) {
    x = val * scale;
  } else {
    x = val / scale;
  }

  float r = fmax(-FP8_E4M3_MAX::value, fmin(x, FP8_E4M3_MAX::value));
  return static_cast<c10::Float8_e4m3fn>(r);
}

// Compute the absolute maximum m of the input tensor and store
// m / float8_e4m3::max() in *scale. Each thread block performs a
// reduction tree and the memory in scale is atomically updated.
// So to get the right answer, *scale needs to be initialized to
// a value <= 0.0 and we need to wait for all thread blocks to
// finish before consuming *scale.
template <typename scalar_t>
__global__ void segmented_max_reduction(
    float* __restrict__ scale,
    const scalar_t* __restrict__ input,
    int64_t num_elems) {
  __shared__ float cache[1024];
  int64_t i = blockDim.x * blockIdx.x + threadIdx.x;

  // First store maximum for all values processes by
  // the current thread in cache[threadIdx.x]
  scalar_t tmp = 0.0;
  while (i < num_elems) {
    float x = static_cast<float>(input[i]);
    tmp = max(tmp, fabs(x));
    i += (size_t)blockDim.x * gridDim.x;
  }
  cache[threadIdx.x] = tmp;

  __syncthreads();

  // Now perform parallel reduction within the thread block
  int ib = blockDim.x / 2;
  while (ib != 0) {
    if (threadIdx.x < ib && cache[threadIdx.x + ib] > cache[threadIdx.x]) {
      cache[threadIdx.x] = cache[threadIdx.x + ib];
    }
    __syncthreads();
    ib /= 2;
  }
  // Finally, since cache[0] contains the maximum for this thread block,
  // atomically write the max to the target location
  if (threadIdx.x == 0) {
    atomicMaxFloat(scale, cache[0] / FP8_E4M3_MAX::value);
  }
}

template <typename scalar_t>
struct __align__(8) vec4_t {
  scalar_t x;
  scalar_t y;
  scalar_t z;
  scalar_t w;
};

typedef struct __align__(4) {
  c10::Float8_e4m3fn x;
  c10::Float8_e4m3fn y;
  c10::Float8_e4m3fn z;
  c10::Float8_e4m3fn w;
}
float8x4_t;

template <typename scalar_t>
__device__ float thread_max_vec(
    scalar_t const* __restrict__ input,
    int64_t const num_elems,
    int const tid,
    int const step) {
  // Vectorized input/output to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vectorized_in =
      reinterpret_cast<vec4_t<scalar_t> const*>(input);

  int64_t const num_vec_elems = num_elems >> 2;
  float absmax_val = 0.0f;

#pragma unroll 4
  for (int64_t i = tid; i < num_vec_elems; i += step) {
    vec4_t<scalar_t> in_vec = vectorized_in[i];
    absmax_val = max(absmax_val, fabs(in_vec.x));
    absmax_val = max(absmax_val, fabs(in_vec.y));
    absmax_val = max(absmax_val, fabs(in_vec.z));
    absmax_val = max(absmax_val, fabs(in_vec.w));
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int64_t i = num_vec_elems * 4 + tid; i < num_elems; i += step) {
    absmax_val = max(absmax_val, fabs(input[i]));
  }

  return absmax_val;
}

template <typename scalar_t, bool is_scale_inverted>
__device__ void scaled_fp8_conversion_vec(
    c10::Float8_e4m3fn* __restrict__ out,
    scalar_t const* __restrict__ input,
    float const scale,
    int64_t const num_elems,
    int const tid,
    int const step) {
  // Vectorized input/output to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vectorized_in =
      reinterpret_cast<vec4_t<scalar_t> const*>(input);
  float8x4_t* vectorized_out = reinterpret_cast<float8x4_t*>(out);

  int64_t const num_vec_elems = num_elems >> 2;

#pragma unroll 4
  for (int64_t i = tid; i < num_vec_elems; i += step) {
    vec4_t<scalar_t> in_vec = vectorized_in[i];
    float8x4_t out_vec;

    out_vec.x = scaled_fp8_conversion<is_scale_inverted>(
        static_cast<float>(in_vec.x), scale);
    out_vec.y = scaled_fp8_conversion<is_scale_inverted>(
        static_cast<float>(in_vec.y), scale);
    out_vec.z = scaled_fp8_conversion<is_scale_inverted>(
        static_cast<float>(in_vec.z), scale);
    out_vec.w = scaled_fp8_conversion<is_scale_inverted>(
        static_cast<float>(in_vec.w), scale);
    vectorized_out[i] = out_vec;
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int64_t i = num_vec_elems * 4 + tid; i < num_elems; i += step) {
    out[i] = scaled_fp8_conversion<is_scale_inverted>(
        static_cast<float>(input[i]), scale);
  }
}

template <typename scalar_t>
__global__ void scaled_fp8_quant_kernel(
    c10::Float8_e4m3fn* __restrict__ out,
    const scalar_t* __restrict__ input,
    const float* __restrict__ scale,
    int64_t num_elems) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  // Invert the scale so that we can use multiplications to avoid expensive
  // division.
  const float inverted_scale = 1.0f / (*scale);
  scaled_fp8_conversion_vec<scalar_t, true>(
      out, input, inverted_scale, num_elems, tid, blockDim.x * gridDim.x);
}

template <typename T>
__inline__ __device__ T _max(T a, T b) {
  return max(a, b);
}

template <typename T>
using ReduceFnType = T (*)(T, T);

#define WARP_SIZE 32

// Helper function to return the next largest power of 2
static constexpr int _nextPow2(unsigned int num) {
  if (num <= 1)
    return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

#ifndef USE_ROCM
#define SHFL_XOR_SYNC(var, lane_mask) \
  __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#define SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
  __shfl_xor_sync(uint32_t(-1), var, lane_mask, width)
#else
#define SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
#define SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
  __shfl_xor(var, lane_mask, width)
#endif

template <typename T, int numLanes = WARP_SIZE>
__inline__ __device__ T warpReduce(T val, ReduceFnType<T> fn) {
  static_assert(
      numLanes > 0 && (numLanes & (numLanes - 1)) == 0,
      "numLanes is not a positive power of 2!");
  static_assert(numLanes <= WARP_SIZE);
#pragma unroll
  for (int mask = numLanes >> 1; mask > 0; mask >>= 1)
    val = fn(val, SHFL_XOR_SYNC(val, mask));

  return val;
}

template <typename T, int maxBlockSize = 1024>
__inline__ __device__ T blockReduce(T val, ReduceFnType<T> fn) {
  static_assert(maxBlockSize <= 1024);
  if constexpr (maxBlockSize > WARP_SIZE) {
    val = warpReduce<T>(val, fn);
    // Calculates max number of lanes that need to participate in the last
    // warpReduce
    constexpr int maxActiveLanes = (maxBlockSize + WARP_SIZE - 1) / WARP_SIZE;
    static __shared__ T shared[maxActiveLanes];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    if (lane == 0)
      shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < blockDim.x / float(WARP_SIZE)) ? shared[lane]
                                                        : (T)(0.0f);
    val = warpReduce<T, _nextPow2(maxActiveLanes)>(val, fn);
  } else {
    // A single warpReduce is equal to blockReduce
    val = warpReduce<T, _nextPow2(maxBlockSize)>(val, fn);
  }
  return val;
}

template <typename T, int maxBlockSize = 1024>
__inline__ __device__ T blockReduceMax(T val) {
  return blockReduce<T, maxBlockSize>(val, _max<T>);
}

template <typename scalar_t>
__global__ void dynamic_per_token_scaled_fp8_quant_kernel(
    c10::Float8_e4m3fn* __restrict__ out,
    float* __restrict__ scale,
    scalar_t const* __restrict__ input,
    float const* __restrict__ scale_ub,
    const int hidden_size) {
  float const min_scaling_factor =
      1.0f / (std::numeric_limits<c10::Float8_e4m3fn>::max() * 512.f);

  int const tid = threadIdx.x;
  int const token_idx = blockIdx.x;

  scalar_t const* __restrict__ token_input = &input[token_idx * hidden_size];
  c10::Float8_e4m3fn* __restrict__ token_output = &out[token_idx * hidden_size];

  // For vectorization, token_input and token_output pointers need to be
  // aligned at 8-byte and 4-byte addresses respectively.
  bool const can_vectorize = hidden_size % 4 == 0;

  float absmax_val = 0.0f;
  if (can_vectorize) {
    absmax_val = thread_max_vec(token_input, hidden_size, tid, blockDim.x);
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      float const x = static_cast<float>(token_input[i]);
      absmax_val = max(absmax_val, fabs(x));
    }
  }

  float const block_absmax_val_maybe = blockReduceMax(absmax_val);
  __shared__ float token_scale;
  if (tid == 0) {
    if (scale_ub) {
      token_scale = min(block_absmax_val_maybe, *scale_ub);
    } else {
      token_scale = block_absmax_val_maybe;
    }
    // token scale computation
    token_scale = max(token_scale / FP8_E4M3_MAX::value, min_scaling_factor);
    scale[token_idx] = token_scale;
  }
  __syncthreads();

  // Note that we don't use inverted scales so we can match FBGemm impl.
  if (can_vectorize) {
    scaled_fp8_conversion_vec<scalar_t, false>(
        token_output, token_input, token_scale, hidden_size, tid, blockDim.x);
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      token_output[i] = scaled_fp8_conversion<false>(
          static_cast<float>(token_input[i]), token_scale);
    }
  }
}

void static_scaled_fp8_quant(
    torch::Tensor& out, // [..., d]
    torch::Tensor const& input, // [..., d]
    torch::Tensor const& scale) // [1]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int64_t num_elems = input.numel();
  dim3 grid(num_tokens);
  dim3 block(1024);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_FLOATING_TYPES(input.scalar_type(), "scaled_fp8_quant_kernel", [&] {
    scaled_fp8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<c10::Float8_e4m3fn>(),
        input.data_ptr<scalar_t>(),
        scale.data_ptr<float>(),
        num_elems);
  });
}

void dynamic_scaled_fp8_quant(
    torch::Tensor& out, // [..., d]
    torch::Tensor const& input, // [..., d]
    torch::Tensor& scale) // [1]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int64_t num_elems = input.numel();
  dim3 grid(num_tokens);
  dim3 block(1024);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_FLOATING_TYPES(input.scalar_type(), "scaled_fp8_quant_kernel", [&] {
    segmented_max_reduction<scalar_t><<<grid, block, 0, stream>>>(
        scale.data_ptr<float>(), input.data_ptr<scalar_t>(), num_elems);
    scaled_fp8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<c10::Float8_e4m3fn>(),
        input.data_ptr<scalar_t>(),
        scale.data_ptr<float>(),
        num_elems);
  });
}

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out, // [..., d]
    torch::Tensor const& input, // [..., d]
    torch::Tensor& scales,
    std::optional<at::Tensor> const& scale_ub) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 1024));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "dynamic_per_token_scaled_fp8_quant_kernel", [&] {
        dynamic_per_token_scaled_fp8_quant_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                out.data_ptr<c10::Float8_e4m3fn>(),
                scales.data_ptr<float>(),
                input.data_ptr<scalar_t>(),
                scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                hidden_size);
      });
}

/**
 * topk_softmax
 **/
/// Aligned array type
template <
    typename T,
    /// Number of elements in the array
    int N,
    /// Alignment requirement in bytes
    int Alignment = sizeof(T) * N>
class alignas(Alignment) AlignedArray {
  float data[N];
};

// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing
// the output in the softmax kernel when we extend this module to support
// expert-choice routing.
template <int TPB>
__launch_bounds__(TPB) __global__ void moeSoftmax(
    const float* input,
    const bool* finished,
    float* output,
    const int num_cols) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  __shared__ float normalizing_factor;
  __shared__ float float_max;

  const int thread_row_offset = blockIdx.x * num_cols;

  cub::Sum sum;
  float threadData(-FLT_MAX);

  // Don't touch finished rows.
  if ((finished != nullptr) && finished[blockIdx.x]) {
    return;
  }

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    threadData = max(static_cast<float>(input[idx]), threadData);
  }

  const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
  if (threadIdx.x == 0) {
    float_max = maxElem;
  }
  __syncthreads();

  threadData = 0;

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    threadData += exp((static_cast<float>(input[idx]) - float_max));
  }

  const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

  if (threadIdx.x == 0) {
    normalizing_factor = 1.f / Z;
  }
  __syncthreads();

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    const float val =
        exp((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
    output[idx] = val;
  }
}

template <int TPB>
__launch_bounds__(TPB) __global__ void moeTopK(
    const float* inputs_after_softmax,
    const bool* finished,
    float* output,
    int* indices,
    int* source_rows,
    const int num_experts,
    const int k,
    const int start_expert,
    const int end_expert) {
  using cub_kvp = cub::KeyValuePair<int, float>;
  using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  cub_kvp thread_kvp;
  cub::ArgMax arg_max;

  const int num_rows = gridDim.x;
  const int block_row = blockIdx.x;

  const bool row_is_active = finished ? !finished[block_row] : true;
  const int thread_read_offset = blockIdx.x * num_experts;
  for (int k_idx = 0; k_idx < k; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = -1.f; // This is OK because inputs are probabilities

    cub_kvp inp_kvp;
    for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
      const int idx = thread_read_offset + expert;
      inp_kvp.key = expert;
      inp_kvp.value = inputs_after_softmax[idx];

      for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
        const int prior_winning_expert = indices[k * block_row + prior_k];

        if (prior_winning_expert == expert) {
          inp_kvp = thread_kvp;
        }
      }

      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    const cub_kvp result_kvp =
        BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      // Ignore experts the node isn't responsible for with expert parallelism
      const int expert = result_kvp.key;
      const bool node_uses_expert =
          expert >= start_expert && expert < end_expert;
      const bool should_process_row = row_is_active && node_uses_expert;

      const int idx = k * block_row + k_idx;
      output[idx] = result_kvp.value;
      indices[idx] = should_process_row ? (expert - start_expert) : num_experts;
      assert(indices[idx] >= 0);
      source_rows[idx] = k_idx * num_rows + block_row;
    }
    __syncthreads();
  }
}

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the
  MoE layers are a small power of 2. This allows us to cleanly share the rows
  among the threads in a single warp and eliminate communication between warps
  (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is intended for when the number of experts is a small
  power of 2. 2) This implementation assumes k is small, but will work for any
  k.
*/

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__ void topkGatingSoftmax(
    const float* input,
    const bool* finished,
    float* output,
    const int num_rows,
    int* indices,
    int* source_rows,
    const int k,
    const int start_expert,
    const int end_expert) {
  // We begin by enforcing compile time assertions and setting up compile time
  // constants.
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(
      NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS),
      "NUM_EXPERTS must be power of 2");
  static_assert(
      BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
      "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  // Number of bytes each thread pulls in per load
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  // Restrictions based on previous section.
  static_assert(
      VPT % ELTS_PER_LDG == 0,
      "The elements per thread must be a multiple of the elements per ldg");
  static_assert(
      WARP_SIZE % THREADS_PER_ROW == 0,
      "The threads per row must cleanly divide the threads per warp");
  static_assert(
      THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW),
      "THREADS_PER_ROW must be power of 2");
  static_assert(
      THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

  // We have NUM_EXPERTS elements per row. We specialize for small #experts
  static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

  // Restrictions for previous section.
  static_assert(
      ELTS_PER_WARP % ELTS_PER_ROW == 0,
      "The elts per row must cleanly divide the total elt per warp");

  // Compute CTA and warp rows. We pack multiple rows into a single warp, and a
  // block contains WARPS_PER_CTA warps. This, each block processes a chunk of
  // rows. We start by computing the start row for each block.
  const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

  // Now, using the base row per thread block, we compute the base row per warp.
  const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

  // The threads in a warp are split into sub-groups that will work on a row.
  // We compute row offset for each thread sub-group
  const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
  const int thread_row = warp_base_row + thread_row_in_warp;

  // Threads with indices out of bounds should early exit here.
  if (thread_row >= num_rows) {
    return;
  }
  const bool row_is_active = finished ? !finished[thread_row] : true;

  // We finally start setting up the read pointers for each thread. First, each
  // thread jumps to the start of the row it will read.
  const float* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

  // Now, we compute the group each thread belong to in order to determine the
  // first column to start loads.
  const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
  const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
  const float* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

  // Determine the pointer type to use to read in the data depending on the
  // BYTES_PER_LDG template param. In theory, this can support all powers of 2
  // up to 16. NOTE(woosuk): The original implementation uses CUTLASS aligned
  // array here. We defined our own aligned array and use it here to avoid the
  // dependency on CUTLASS.
  using AccessType = AlignedArray<float, ELTS_PER_LDG>;

  // Finally, we pull in the data from global mem
  float row_chunk[VPT];
  AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk);
  const AccessType* vec_thread_read_ptr =
      reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
  for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
    row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
  }

  // First, we perform a max reduce within the thread. We can do the max in fp16
  // safely (I think) and just convert to float afterwards for the exp + sum
  // reduction.
  float thread_max = row_chunk[0];
#pragma unroll
  for (int ii = 1; ii < VPT; ++ii) {
    thread_max = max(thread_max, row_chunk[ii]);
  }

// Now, we find the max within the thread group and distribute among the
// threads. We use a butterfly reduce.
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    thread_max =
        max(thread_max, SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
  }

  // From this point, thread max in all the threads have the max within the row.
  // Now, we subtract the max from each element in the thread and take the exp.
  // We also compute the thread local sum.
  float row_sum = 0;
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = expf(row_chunk[ii] - thread_max);
    row_sum += row_chunk[ii];
  }

// Now, we perform the sum reduce within each thread group. Similar to the max
// reduce, we use a bufferfly pattern.
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    row_sum += SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
  }

  // From this point, all threads have the max and the sum for their rows in the
  // thread_max and thread_sum variables respectively. Finally, we can scale the
  // rows for the softmax. Technically, for top-k gating we don't need to
  // compute the entire softmax row. We can likely look at the maxes and only
  // compute for the top-k values in the row. However, this kernel will likely
  // not be a bottle neck and it seems better to closer match torch and find the
  // argmax after computing the softmax.
  const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
  }

  // Now, softmax_res contains the softmax of the row chunk. Now, I want to find
  // the topk elements in each row, along with the max index.
  int start_col = first_elt_read_by_thread;
  static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

  for (int k_idx = 0; k_idx < k; ++k_idx) {
    // First, each thread does the local argmax
    float max_val = row_chunk[0];
    int expert = start_col;
#pragma unroll
    for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
         ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        float val = row_chunk[ldg * ELTS_PER_LDG + ii];

        // No check on the experts here since columns with the smallest index
        // are processed first and only updated if > (not >=)
        if (val > max_val) {
          max_val = val;
          expert = col + ii;
        }
      }
    }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads
// reach consensus about the max. This will be useful for K > 1 so that the
// threads can agree on "who" had the max value. That thread can then blank out
// their max with -inf and the warp can run more iterations...
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other_max = SHFL_XOR_SYNC_WIDTH(max_val, mask, THREADS_PER_ROW);
      int other_expert = SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);

      // We want lower indices to "win" in every thread so we break ties this
      // way
      if (other_max > max_val ||
          (other_max == max_val && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    // Write the max for this k iteration to global memory.
    if (thread_group_idx == 0) {
      // Add a guard to ignore experts not included by this node
      const bool node_uses_expert =
          expert >= start_expert && expert < end_expert;
      const bool should_process_row = row_is_active && node_uses_expert;

      // The lead thread from each sub-group will write out the final results to
      // global memory. (This will be a single) thread per row of the
      // input/output matrices.
      const int idx = k * thread_row + k_idx;
      output[idx] = max_val;
      indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;
      source_rows[idx] = k_idx * num_rows + thread_row;
    }

    // Finally, we clear the value in the thread with the current max if there
    // is another iteration to run.
    if (k_idx + 1 < k) {
      const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
      const int thread_to_clear_in_group =
          (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

      // Only the thread in the group which produced the max will reset the
      // "winning" value to -inf.
      if (thread_group_idx == thread_to_clear_in_group) {
        const int offset_for_expert = expert % ELTS_PER_LDG;
        // Safe to set to any negative value since row_chunk values must be
        // between 0 and 1.
        row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] =
            -10000.f;
      }
    }
  }
}

template <int EXPERTS, int WARPS_PER_TB>
void topkGatingSoftmaxLauncherHelper(
    const float* input,
    const bool* finished,
    float* output,
    int* indices,
    int* source_row,
    const int num_rows,
    const int k,
    const int start_expert,
    const int end_expert,
    cudaStream_t stream) {
  static constexpr std::size_t MAX_BYTES_PER_LDG = 16;

  static constexpr int BYTES_PER_LDG =
      MIN(MAX_BYTES_PER_LDG, sizeof(float) * EXPERTS);
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
  static constexpr int VECs_PER_THREAD =
      MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
  const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
  topkGatingSoftmax<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG>
      <<<num_blocks, block_dim, 0, stream>>>(
          input,
          finished,
          output,
          num_rows,
          indices,
          source_row,
          k,
          start_expert,
          end_expert);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define LAUNCH_SOFTMAX(NUM_EXPERTS, WARPS_PER_TB)             \
  topkGatingSoftmaxLauncherHelper<NUM_EXPERTS, WARPS_PER_TB>( \
      gating_output,                                          \
      nullptr,                                                \
      topk_weights,                                           \
      topk_indicies,                                          \
      token_expert_indices,                                   \
      num_tokens,                                             \
      topk,                                                   \
      0,                                                      \
      num_experts,                                            \
      stream);

void topkGatingSoftmaxKernelLauncher(
    const float* gating_output,
    float* topk_weights,
    int* topk_indicies,
    int* token_expert_indices,
    float* softmax_workspace,
    const int num_tokens,
    const int num_experts,
    const int topk,
    cudaStream_t stream) {
  static constexpr int WARPS_PER_TB = 4;
  switch (num_experts) {
    case 1:
      LAUNCH_SOFTMAX(1, WARPS_PER_TB);
      break;
    case 2:
      LAUNCH_SOFTMAX(2, WARPS_PER_TB);
      break;
    case 4:
      LAUNCH_SOFTMAX(4, WARPS_PER_TB);
      break;
    case 8:
      LAUNCH_SOFTMAX(8, WARPS_PER_TB);
      break;
    case 16:
      LAUNCH_SOFTMAX(16, WARPS_PER_TB);
      break;
    case 32:
      LAUNCH_SOFTMAX(32, WARPS_PER_TB);
      break;
    case 64:
      LAUNCH_SOFTMAX(64, WARPS_PER_TB);
      break;
    case 128:
      LAUNCH_SOFTMAX(128, WARPS_PER_TB);
      break;
    case 256:
      LAUNCH_SOFTMAX(256, WARPS_PER_TB);
      break;
    default: {
      TORCH_CHECK(
          softmax_workspace != nullptr,
          "softmax_workspace must be provided for num_experts that are not a power of 2.");
      static constexpr int TPB = 256;
      moeSoftmax<TPB><<<num_tokens, TPB, 0, stream>>>(
          gating_output, nullptr, softmax_workspace, num_experts);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      moeTopK<TPB><<<num_tokens, TPB, 0, stream>>>(
          softmax_workspace,
          nullptr,
          topk_weights,
          topk_indicies,
          token_expert_indices,
          num_experts,
          topk,
          0,
          num_experts);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
}

void topk_softmax(
    torch::Tensor& topk_weights, // [num_tokens, topk]
    torch::Tensor& topk_indices, // [num_tokens, topk]
    torch::Tensor& token_expert_indices, // [num_tokens, topk]
    torch::Tensor& gating_output) // [num_tokens, num_experts]
{
  const int num_experts = gating_output.size(-1);
  const int num_tokens = gating_output.numel() / num_experts;
  const int topk = topk_weights.size(-1);

  const bool is_pow_2 =
      (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
  const bool needs_workspace = !is_pow_2 || num_experts > 256;
  const int64_t workspace_size = needs_workspace ? num_tokens * num_experts : 0;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  torch::Tensor softmax_workspace =
      torch::empty({workspace_size}, gating_output.options());
  topkGatingSoftmaxKernelLauncher(
      gating_output.data_ptr<float>(),
      topk_weights.data_ptr<float>(),
      topk_indices.data_ptr<int>(),
      token_expert_indices.data_ptr<int>(),
      softmax_workspace.data_ptr<float>(),
      num_tokens,
      num_experts,
      topk,
      stream);
}

/**
 * silu_and_mul
 **/
#ifndef USE_ROCM
#define LDG(arg) __ldg(arg)
#else
#define LDG(arg) *(arg)
#endif

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

// Activation and gating kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out, // [..., d]
    const scalar_t* __restrict__ input, // [..., 2, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = ACT_FN(x) * y;
  }
}

// Launch activation and gating kernel.
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL)                              \
  int d = input.size(-1) / 2;                                              \
  int64_t num_tokens = input.numel() / input.size(-1);                     \
  dim3 grid(num_tokens);                                                   \
  dim3 block(std::min(d, 1024));                                           \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));        \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();            \
  DISPATCH_FLOATING_TYPES(input.scalar_type(), "act_and_mul_kernel", [&] { \
    act_and_mul_kernel<scalar_t, KERNEL<scalar_t>>                         \
        <<<grid, block, 0, stream>>>(                                      \
            out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);      \
  });

void silu_and_mul(
    torch::Tensor& out, // [..., d]
    torch::Tensor& input) // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(silu_kernel);
}
#endif
} // namespace fbgemm_gpu
