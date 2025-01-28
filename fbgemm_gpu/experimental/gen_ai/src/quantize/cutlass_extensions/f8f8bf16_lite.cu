/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/numeric_conversion.h>
#include <cub/cub.cuh>

namespace fbgemm_gpu {

#if CUDART_VERSION >= 12000

using SizeType32 = std::size_t;

struct Params {
  void const* act;
  void const* weight;
  void const* alpha;
  void* output;
  SizeType32 m, n, k;

  Params(
      void const* _act,
      void const* _weight,
      void const* _alpha,
      void* _output,
      SizeType32 _m,
      SizeType32 _n,
      SizeType32 _k)
      : act(_act),
        weight(_weight),
        alpha(_alpha),
        output(_output),
        m(_m),
        n(_n),
        k(_k) {}
};

template <
    typename InputType,
    typename OutputType,
    SizeType32 TILE_M,
    SizeType32 TILE_N,
    SizeType32 BLOCK_SIZE>
__global__ void cudaCoreGemm(
    InputType const* __restrict__ act,
    InputType const* __restrict__ weight,
    float const* alpha,
    OutputType* __restrict__ output,
    SizeType32 m,
    SizeType32 n,
    SizeType32 k) {
  using VecType = int4;
  static constexpr SizeType32 kStepK =
      static_cast<SizeType32>(128 / (8 * sizeof(InputType)));
  static constexpr SizeType32 kTileK = kStepK * BLOCK_SIZE;
  auto tileIdM = static_cast<SizeType32>(blockIdx.x * TILE_M);
  auto tileIdN = static_cast<SizeType32>(blockIdx.y * TILE_N);
  auto tid = static_cast<SizeType32>(threadIdx.x);
  float tile_a[kStepK], tile_w[TILE_N * kStepK];
  float acc[TILE_M * TILE_N];

  static_assert(kStepK % 4 == 0);
  using CvtInputType = cutlass::float_e4m3_t;
  using Converter = cutlass::NumericArrayConverter<float, CvtInputType, 4>;
  using CvtSrcType = typename Converter::source_type;
  using CvtResType = typename Converter::result_type;
  static constexpr SizeType32 kCvtCount =
      static_cast<SizeType32>(sizeof(VecType) / sizeof(CvtSrcType));

#pragma unroll
  for (SizeType32 i = 0; i < TILE_M * TILE_N; ++i) {
    acc[i] = 0;
  }
  act += tileIdM * k;
  weight += tileIdN * k;
  output += tileIdM * n + tileIdN;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  for (SizeType32 idxK = tid * kStepK; idxK < k; idxK += kTileK) {
    for (SizeType32 i = 0; i < TILE_N; ++i) {
      auto tile_w_quantized =
          reinterpret_cast<VecType const*>(weight + i * k + idxK)[0];
#pragma unroll
      for (SizeType32 cvtIdx = 0; cvtIdx < kCvtCount; ++cvtIdx) {
        reinterpret_cast<CvtResType*>(tile_w)[i * kCvtCount + cvtIdx] =
            Converter::convert(
                reinterpret_cast<CvtSrcType*>(&tile_w_quantized)[cvtIdx]);
      }
    }
#pragma unroll
    for (SizeType32 i = 0; i < TILE_M; ++i) {
      auto tile_a_quantized =
          reinterpret_cast<VecType const*>(act + i * k + idxK)[0];
#pragma unroll
      for (SizeType32 cvtIdx = 0; cvtIdx < kCvtCount; ++cvtIdx) {
        reinterpret_cast<CvtResType*>(tile_a)[cvtIdx] = Converter::convert(
            reinterpret_cast<CvtSrcType*>(&tile_a_quantized)[cvtIdx]);
      }
#pragma unroll
      for (SizeType32 j = 0; j < TILE_N; ++j) {
#pragma unroll
        for (SizeType32 l = 0; l < kStepK; ++l) {
          acc[i * TILE_N + j] =
              fma(tile_a[l], tile_w[j * kStepK + l], acc[i * TILE_N + j]);
        }
      }
    }
  }

  typedef cub::WarpReduce<float> WarpReduce;

  static constexpr SizeType32 kWarpSize = 32;
  static constexpr SizeType32 kWarpNum = BLOCK_SIZE / kWarpSize;
  SizeType32 warpId = tid / kWarpSize, laneId = tid % kWarpSize;
  __shared__ float shmem[TILE_M * TILE_N * kWarpNum];
  __shared__ typename WarpReduce::TempStorage tempStorage[kWarpNum];
#pragma unroll
  for (SizeType32 mi = 0; mi < TILE_M; ++mi) {
#pragma unroll
    for (SizeType32 ni = 0; ni < TILE_N; ++ni) {
      float val = WarpReduce(tempStorage[warpId]).Sum(acc[mi * TILE_N + ni]);
      if (laneId == 0) {
        shmem[mi * TILE_N + ni + warpId * TILE_M * TILE_N] = val;
      }
    }
  }
  __syncthreads();
  for (SizeType32 ii = tid; ii < TILE_M * TILE_N; ii += BLOCK_SIZE) {
    SizeType32 mid = ii / TILE_N, nid = ii % TILE_N;
    float val = 0;
#pragma unroll
    for (SizeType32 jj = 0; jj < kWarpNum; ++jj) {
      val += shmem[jj * TILE_M * TILE_N + ii];
    }
    output[mid * n + nid] = static_cast<OutputType>(val * *alpha);
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <
    typename InputType,
    typename OutputType,
    SizeType32 TILE_M,
    SizeType32 TILE_N,
    SizeType32 BLOCK_SIZE>
void cudaCoreGemmKernel(Params const& params, cudaStream_t stream) {
  dim3 block(BLOCK_SIZE);
  dim3 grid(params.m / TILE_M, params.n / TILE_N);

  cudaCoreGemm<InputType, OutputType, TILE_M, TILE_N, BLOCK_SIZE>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<InputType const*>(params.act),
          reinterpret_cast<InputType const*>(params.weight),
          reinterpret_cast<float const*>(params.alpha),
          reinterpret_cast<OutputType*>(params.output),
          params.m,
          params.n,
          params.k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <
    typename InputType,
    typename OutputType,
    int TILE_M,
    int TILE_N,
    int BLOCK_SIZE>
bool cudaCoreGemmTemplateCaller(Params const& params, cudaStream_t stream) {
  constexpr int cudaCoreGemmTemplateMaxM = 4;
  if (params.m == TILE_M) {
    cudaCoreGemmKernel<InputType, OutputType, TILE_M, TILE_N, BLOCK_SIZE>(
        params, stream);
    return true;
  }
  if constexpr (TILE_M < cudaCoreGemmTemplateMaxM) {
    return cudaCoreGemmTemplateCaller<
        InputType,
        OutputType,
        TILE_M + 1,
        TILE_N,
        BLOCK_SIZE>(params, stream);
  }
  return false;
}

template <typename InputType, typename OutputType>
bool cudaCoreGemmLauncher(Params const& params, cudaStream_t stream) {
  return cudaCoreGemmTemplateCaller<InputType, OutputType, 1, 2, 128>(
      params, stream);
}

at::Tensor f8f8bf16_lite(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor scale) {
  bool dispatched = true;
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = WQ.size(0);
  int K = WQ.size(1);
  TORCH_CHECK(XQ.size(-1) == K);

  if (M > 4) {
    throw std::runtime_error("f8f8bf16_lite cannot run when M > 4");
  } else if (N % 2 != 0) {
    throw std::runtime_error("f8f8bf16_lite cannot run when N % 2 != 0");
  } else if (K % 16 != 0) {
    throw std::runtime_error("f8f8bf16_lite cannot run when K % 16 != 0");
  }

  auto out_sizes = XQ.sizes().vec();
  out_sizes.back() = N;
  at::Tensor Y = at::empty(out_sizes, XQ.options().dtype(at::kBFloat16));

  Params params{
      XQ.data_ptr(),
      WQ.data_ptr(),
      scale.data_ptr(),
      Y.data_ptr(),
      (SizeType32)M,
      (SizeType32)N,
      (SizeType32)K};
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dispatched = cudaCoreGemmLauncher<cutlass::float_e4m3_t, __nv_bfloat16>(
      params, stream);
  if (!dispatched) {
    throw std::runtime_error("f8f8bf16_lite cannot run");
  }
  return Y;
}

#else

at::Tensor f8f8bf16_lite(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor scale) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
