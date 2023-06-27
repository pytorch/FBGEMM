/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

// Copied from cupy/random/_kernels.py v11
// (commit id 420e41fd41157d4cf526b0e94eb86a3f8eb5a231)

typedef struct {
  unsigned int xor128[4];
  double gauss;
  int has_gauss; // !=0: gauss contains a gaussian deviate

#ifdef CUPY_USE_BINOMIAL
  int has_binomial; // !=0: following parameters initialized for binomial
  /* The rk_state structure has been extended to store the following
   * information for the binomial generator. If the input values of n or p
   * are different than nsave and psave, then the other parameters will be
   * recomputed. RTK 2005-09-02 */
  int nsave, m;
  double psave, r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
#endif
} rk_state;

__device__ void rk_seed(unsigned long long s, rk_state* state) {
  for (int i = 1; i <= 4; i++) {
    s = 1812433253U * (s ^ (s >> 30)) + i;
    state->xor128[i - 1] = s;
  }
  state->has_gauss = 0;
#ifdef CUPY_USE_BINOMIAL
  state->has_binomial = 0;
#endif
}

__device__ unsigned long rk_random(rk_state* state) {
  unsigned int* xor128 = state->xor128;
  unsigned int t = xor128[0] ^ (xor128[0] << 11);
  xor128[0] = xor128[1];
  xor128[1] = xor128[2];
  xor128[2] = xor128[3];
  return xor128[3] ^= (xor128[3] >> 19) ^ t ^ (t >> 8);
}

__device__ double rk_double(rk_state* state) {
  /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
  int a = rk_random(state) >> 5, b = rk_random(state) >> 6;
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

__device__ long rk_zipf(rk_state* state, double a) {
  double am1, b;

  am1 = a - 1.0;
  b = pow(2.0, am1);
  while (1) {
    double T, U, V, X;

    U = 1.0 - rk_double(state);
    V = rk_double(state);
    X = floor(pow(U, -1.0 / am1));

    if (X < 1.0) {
      continue;
    }

    T = pow(1.0 + 1.0 / X, am1);
    if (V * X * (T - 1.0) / (b - 1.0) <= T / b) {
      return (long)X;
    }
  }
}

__global__ void zipf_kernel(
    const double a,
    const int64_t seed,
    at::PackedTensorAccessor64<long, 1, at::RestrictPtrTraits> y) {
  rk_state internal_state;
  auto N = y.size(0);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    rk_seed(seed + i, &internal_state);
    y[i] = rk_zipf(&internal_state, a);
  }
}

DLL_PUBLIC Tensor
zipf_cuda(const double a, const int64_t n, const int64_t seed) {
  Tensor y = at::empty(
      {n},
      at::TensorOptions().dtype(at::kLong).device(
          at::kCUDA, at::cuda::current_device()));
  zipf_kernel<<<
      cuda_calc_xblock_count(n, kMaxThreads),
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      a, seed, y.packed_accessor64<long, 1, at::RestrictPtrTraits>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def("zipf_cuda(float a, int n, int seed) -> Tensor");
  DISPATCH_TO_ALL("zipf_cuda", fbgemm_gpu::zipf_cuda);
}
