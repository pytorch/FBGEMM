/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <tuple>
#include <unordered_map>

#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <torch/torch.h>

#include "kernels/fp8_rowwise_batched_kernel_manifest.h"

namespace fbgemm_gpu {

using RowwiseBatchedKernel = std::function<
    at::Tensor(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor)>;

using BNKLookupTableType = std::map<int, RowwiseBatchedKernel>;

// Define a custom hash function for std::tuple<int, int, int>
struct IntTupleHash {
  size_t operator()(const std::tuple<int, int, int>& t) const {
    auto hash1 = std::hash<int>{}(std::get<0>(t));
    auto hash2 = std::hash<int>{}(std::get<1>(t));
    auto hash3 = std::hash<int>{}(std::get<2>(t));
    return hash1 ^ hash2 ^ hash3;
  }
};

static const std::map<int, RowwiseBatchedKernel> B_2_N_5120_K_8192_dispatch_table = {
  { 8, fp8_rowwise_batched_64x16x16x512_16x16_1x1_32x2x1_32x2x1_1x16x1x4_4x4x1_1x1_interwave_v2},
  { 16, fp8_rowwise_batched_64x16x16x512_16x16_1x1_32x2x1_32x2x1_1x16x1x4_4x4x1_1x1_intrawave_v2},
  { 32, fp8_rowwise_batched_256x32x64x512_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 64, fp8_rowwise_batched_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 72, fp8_rowwise_batched_256x32x64x512_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 96, fp8_rowwise_batched_256x32x128x256_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 192, fp8_rowwise_batched_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 248, fp8_rowwise_batched_256x128x96x256_32x32_1x3_16x16x1_16x16x1_1x64x1x4_8x8x1_1x1_intrawave_v3},
  { 384, fp8_rowwise_batched_256x128x128x256_32x32_2x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 512, fp8_rowwise_batched_256x128x160x128_32x32_1x5_8x32x1_8x32x1_1x64x1x4_8x8x1_1x1_intrawave_v3},
  { 640, fp8_rowwise_batched_256x128x192x128_32x32_2x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 768, fp8_rowwise_batched_256x256x128x128_16x16_8x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 896, fp8_rowwise_batched_256x128x256x128_32x32_2x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 1024, fp8_rowwise_batched_256x256x160x128_16x16_8x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 1568, fp8_rowwise_batched_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 1792, fp8_rowwise_batched_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 2304, fp8_rowwise_batched_256x256x160x128_16x16_8x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 2816, fp8_rowwise_batched_256x256x192x128_16x16_8x6_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 3360, fp8_rowwise_batched_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 8992, fp8_rowwise_batched_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
};

static const std::map<int, RowwiseBatchedKernel> B_2_N_8192_K_5120_dispatch_table = {
  { 4, fp8_rowwise_batched_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2},
  { 8, fp8_rowwise_batched_128x16x32x256_16x16_1x1_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_interwave_v1},
  { 16, fp8_rowwise_batched_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v3},
  { 32, fp8_rowwise_batched_256x32x64x512_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 64, fp8_rowwise_batched_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 128, fp8_rowwise_batched_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 192, fp8_rowwise_batched_256x64x192x256_32x32_1x3_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 208, fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5},
  { 232, fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 256, fp8_rowwise_batched_256x128x128x256_32x32_2x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 384, fp8_rowwise_batched_256x128x192x128_32x32_2x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 512, fp8_rowwise_batched_256x256x128x128_16x16_8x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 768, fp8_rowwise_batched_256x256x192x128_16x16_8x6_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 896, fp8_rowwise_batched_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 1024, fp8_rowwise_batched_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 1280, fp8_rowwise_batched_256x256x160x128_16x16_8x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 1792, fp8_rowwise_batched_256x256x192x128_16x16_8x6_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 2048, fp8_rowwise_batched_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 2304, fp8_rowwise_batched_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 2560, fp8_rowwise_batched_256x256x192x128_16x16_8x6_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 3136, fp8_rowwise_batched_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 8992, fp8_rowwise_batched_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
};

static const std::unordered_map<std::tuple<int, int, int>, BNKLookupTableType, IntTupleHash> BNK_lookup_table = {
  {{2, 5120, 8192}, B_2_N_5120_K_8192_dispatch_table},
  {{2, 8192, 5120}, B_2_N_8192_K_5120_dispatch_table}
};

RowwiseBatchedKernel rowwise_batched_bnk_lookup(int M, const BNKLookupTableType& table) {
  auto it = table.lower_bound(M);
  if (it != table.end()) {
    return it->second;
  } else {
    --it;
    return it->second;
  }
}

RowwiseBatchedKernel
rowwise_batched_heuristic_dispatch(int B, int M, int N, int K) {
  // Use shape heuristics to guess what the best kernel might be for the given
  // We divide cases based on the size of M to simplify sub-shape reasoning.
  // Start by handling cases with small M.
  if (M < 16) {
    // workload. For super small shapes, use specialized thin kernel.
    if (B <= 16 && (N <= 2048 && K >= 2048)) {
      return fp8_rowwise_batched_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4_1x1_interwave_v2;
    }
    // Kernel for small m but larger other dims.
    if (B < 32 && K < 8192) {
      return fp8_rowwise_batched_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2;
    }
    if (B < 32 && N < 8192) {
      return fp8_rowwise_batched_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4_1x1_interwave_v1;
    }
    // Catch other small m cases.
    return fp8_rowwise_batched_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1;
  }
  if (M < 32) {
    if (B < 8 && K < 8192) {
      return fp8_rowwise_batched_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2;
    }
    if (B < 8 && N < 8192) {
      return fp8_rowwise_batched_128x16x32x512_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1;
    }
    // handle large shapes
    if (B < 8 && K > 8192 && N > 8192) {
      return fp8_rowwise_batched_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1;
    }
    // Fallback kernel that works well for most other gemms.
    return fp8_rowwise_batched_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2;
  }
  if (M < 64) {
    // Some small batch size but large N and K require special handling.
    if (B < 8) {
      if (K >= 20480) {
        return fp8_rowwise_batched_128x16x32x512_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2;
      }
      if (N >= 8192 && K >= 8192) {
        return fp8_rowwise_batched_128x32x128x128_32x32_1x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2;
      }
    }
    return fp8_rowwise_batched_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2;
  }
  if (M < 128) {
    if (K >= 20480) {
      return fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5;
    }
    return fp8_rowwise_batched_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
  }
  if (M < 256) {
    if (N <= 8192 && K <= 8192) {
      return fp8_rowwise_batched_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
    }
    if (N >= 8192 && K >= 8192) {
      return fp8_rowwise_batched_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3;
    }
    return fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v4;
  }
  if (M < 512) {
    if (N <= 8192 && K <= 5120) {
      return fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5;
    }
    if (N <= 8192 && K <= 8192) {
      return fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v4;
    }
    if (N < 20480 && K >= 20480) {
      return fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5;
    }
    return fp8_rowwise_batched_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3;
  }
  if (M < 1024) {
    if (B < 8 && N <= 8192 && K <= 5120) {
      return fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
    }
    if (B < 8 && N <= 8192 && K <= 8192) {
      return fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5;
    }
    if (N >= 20480 && K < 20480) {
      return fp8_rowwise_batched_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3;
    }
    return fp8_rowwise_batched_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3;
  }
  // Otherwise use super large kernel.
  return fp8_rowwise_batched_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3;
}

RowwiseBatchedKernel
rowwise_batched_dispatch(int B, int M, int N, int K) {
  auto it = BNK_lookup_table.find({B, N, K});
  if (it != BNK_lookup_table.end()) {
    return rowwise_batched_bnk_lookup(M, it->second);
  }
  return rowwise_batched_heuristic_dispatch(B, M, N, K);
}

at::Tensor f8f8bf16_rowwise_batched(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::Tensor> output = std::nullopt) {
  // Check that input datatypes are valid.
  TORCH_CHECK(
      XQ.dim() == 3 && WQ.dim() == 3,
      "Inputs must have 3 dimensions, with the first being batch.");
  TORCH_CHECK(
      (XQ.dtype() == at::kFloat8_e4m3fnuz) &&
          (WQ.dtype() == at::kFloat8_e4m3fnuz),
      "Inputs must be type float8_e4m3fnuz.");
  TORCH_CHECK(
      (x_scale.dtype() == at::kFloat) && (w_scale.dtype() == at::kFloat),
      "Scales must be float32.");
  TORCH_CHECK(use_fast_accum, "AMD does not support disabling use_fast_accum.");

  // Check inputs are in expected format.
  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());

  // XQ: B x M x K
  // WQ: B x N x K
  // output: B x M x N
  int B = XQ.size(0);
  int M = XQ.size(1);
  int N = WQ.size(1);
  int K = WQ.size(2);

  // Prepare output tensor if needed.
  at::Tensor Y;
  if (output.has_value()) {
    Y = output.value();
    // Make sure the provided output has the proper shape and dtype.
    TORCH_CHECK(Y.dim() == 3, "Output tensor must have three dimensions.");
    int Y_B = Y.size(0);
    int Y_M = Y.size(1);
    int Y_N = Y.size(2);
    TORCH_CHECK(Y_B == B && Y_M == M && Y_N == N);
    TORCH_CHECK(Y.dtype() == at::kBFloat16);
  } else {
    auto out_sizes = XQ.sizes().vec();
    out_sizes.back() = N;
    Y = at::empty(out_sizes, XQ.options().dtype(at::kBFloat16));
  }

  RowwiseBatchedKernel selected_kernel =
      rowwise_batched_dispatch(B, M, N, K);
  return selected_kernel(XQ, WQ, x_scale, w_scale, Y);
}

} // namespace fbgemm_gpu
