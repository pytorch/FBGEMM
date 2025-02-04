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

#include "kernels/fp8_rowwise_kernel_manifest.h"

namespace fbgemm_gpu {

using RowwiseKernel = std::function<
    at::Tensor(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor)>;

using NKLookupTableType = std::map<int, RowwiseKernel>;

// Define a custom hash function for std::tuple<int, int, int>
struct IntTupleHash {
  size_t operator()(const std::tuple<int, int>& t) const {
    auto hash1 = std::hash<int>{}(std::get<0>(t));
    auto hash2 = std::hash<int>{}(std::get<1>(t));
    return hash1 ^ hash2;
  }
  size_t operator()(const std::tuple<int, int, int>& t) const {
    auto hash1 = std::hash<int>{}(std::get<0>(t));
    auto hash2 = std::hash<int>{}(std::get<1>(t));
    auto hash3 = std::hash<int>{}(std::get<2>(t));
    return hash1 ^ hash2 ^ hash3;
  }
};

// For certain high priority shapes, we directly map to the best kernel rather
// than use heuristics.
static const std::unordered_map<std::tuple<int, int, int>, RowwiseKernel, IntTupleHash> rowwise_lookup_dispatch = {
    // LLama 70B Decode shapes.
    // Support for decode across batch sizes for [1280, 8192]
    {{16, 1280, 8192},
     fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2},
    {{32, 1280, 8192},
     fp8_rowwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2},
    {{64, 1280, 8192},
     fp8_rowwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2},
    {{128, 1280, 8192},
     fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2},
    // Support for decode across batch sizes for [8192, 1024]
    {{16, 8192, 1024},
     fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2},
    {{32, 8192, 1024},
     fp8_rowwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2},
    {{64, 8192, 1024},
     fp8_rowwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2},
    {{128, 8192, 1024},
     fp8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
    // Llama 405B Decode Shapes.
    // Support for decode across batch sizes for [13312, 6656].
    {{16, 13312, 6656},
     fp8_rowwise_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1},
    {{32, 13312, 6656},
     fp8_rowwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2},
    {{64, 13312, 6656},
     fp8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
    {{128, 13312, 6656},
     fp8_rowwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
    // Support for decode across batch sizes for [13312, 16384].
    {{16, 13312, 16384},
     fp8_rowwise_64x16x16x512_16x16_1x1_32x2x1_32x2x1_1x16x1x4_4x4x1_1x1_interwave_v2},
    {{32, 13312, 16384},
     fp8_rowwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2},
    {{64, 13312, 16384},
     fp8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
    {{128, 13312, 16384},
     fp8_rowwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
    {{1024, 13312, 16384},
     fp8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
    {{2048, 13312, 16384},
     fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
    {{4096, 13312, 16384},
     fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
    {{8192, 13312, 16384},
     fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
    // Support for decode across batch sizes for [16384, 6656].
    {{16, 16384, 6656},
     fp8_rowwise_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1},
    {{32, 16384, 6656},
     fp8_rowwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2},
    {{64, 16384, 6656},
     fp8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
    {{128, 16384, 6656},
     fp8_rowwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
    {{1024, 16384, 6656},
     fp8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
    {{2048, 16384, 6656},
     fp8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
    {{4096, 16384, 6656},
     fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
    {{8192, 16384, 6656},
     fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
    // Support for decode across batch sizes for [16384, 16384].
    {{16, 16384, 16384},
     fp8_rowwise_64x16x16x512_16x16_1x1_32x2x1_32x2x1_1x16x1x4_4x4x1_1x1_interwave_v2},
    {{32, 16384, 16384},
     fp8_rowwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2},
    {{64, 16384, 16384},
     fp8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
    {{128, 16384, 16384},
     fp8_rowwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
    // EMU 1.6 Shapes.
    {{1536, 3584, 3584},
     fp8_rowwise_256x128x160x128_32x32_1x5_8x32x1_8x32x1_1x64x1x4_8x8x1_1x1_intrawave_v3},
    {{8192, 9728, 3584},
     fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
    {{8192, 3584, 9728},
     fp8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
    {{8192, 3584, 3584},
     fp8_rowwise_256x256x192x128_32x32_4x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
    {{4096, 3584, 3584},
     fp8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
    {{768, 3584, 3584},
     fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
    {{4096, 9728, 3584},
     fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
    {{4096, 3584, 9728},
     fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
    {{7200, 3584, 3584},
     fp8_rowwise_256x256x192x128_32x32_4x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
    {{7200, 9728, 3584},
     fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
    {{7200, 3584, 9728},
     fp8_rowwise_256x256x192x128_32x32_4x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
    {{3600, 3584, 3584},
     fp8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
    {{3600, 9728, 3584},
     fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
    {{3600, 3584, 9728},
     fp8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
    // EMU 1.7 shapes.
    {{1536, 4096, 4096},
     fp8_rowwise_256x192x128x128_16x16_6x4_8x32x1_8x32x1_1x32x1x8_8x8x1_2x2_intrawave_v3},
    {{3600, 4096, 4096},
     fp8_rowwise_256x192x256x128_16x16_6x8_8x32x1_8x32x1_1x32x1x8_8x8x1_2x2_intrawave_v3},
    {{3600, 11008, 4096},
     fp8_rowwise_256x192x256x128_16x16_6x8_8x32x1_8x32x1_1x32x1x8_8x8x1_2x2_intrawave_v3},
    {{3600, 4096, 11008},
     fp8_rowwise_256x192x256x128_16x16_6x8_8x32x1_8x32x1_1x32x1x8_8x8x1_2x2_intrawave_v3},
    {{4096, 4096, 4096},
     fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
    {{4096, 11008, 4096},
     fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
    {{4096, 4096, 11008},
     fp8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
    // Pro Shapes.
    {{32768, 128, 8192},
     fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
    {{32768, 8192, 1024},
     fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
    {{32768, 8192, 3072},
     fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
    {{32768, 3072, 8192},
     fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
    {{32768, 1024, 8192},
     fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3}};

static const std::map<int, RowwiseKernel> N_7168_K_8192_dispatch_table = {
  { 8, fp8_rowwise_128x32x16x512_16x16_1x1_32x4x1_32x4x1_1x32x1x4_4x4x1_1x1_interwave_v2},
  { 32, fp8_rowwise_128x32x16x512_16x16_1x1_32x4x1_32x4x1_1x32x1x4_4x4x1_1x1_intrawave_v2},
  { 64, fp8_rowwise_256x32x64x512_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 128, fp8_rowwise_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 320, fp8_rowwise_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 512, fp8_rowwise_256x128x96x256_32x32_1x3_16x16x1_16x16x1_1x64x1x4_8x8x1_1x1_intrawave_v3},
  { 576, fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 640, fp8_rowwise_256x128x128x256_32x32_2x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 768, fp8_rowwise_256x128x160x128_32x32_1x5_8x32x1_8x32x1_1x64x1x4_8x8x1_1x1_intrawave_v3},
  { 1024, fp8_rowwise_256x256x96x128_16x16_8x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 1280, fp8_rowwise_256x256x128x128_16x16_8x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 1536, fp8_rowwise_256x256x160x128_16x16_8x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 2048, fp8_rowwise_256x256x192x128_16x16_8x6_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 2304, fp8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 2560, fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 3328, fp8_rowwise_256x256x160x128_16x16_8x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 4096, fp8_rowwise_256x256x192x128_16x16_8x6_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 4672, fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 4864, fp8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 5376, fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 6144, fp8_rowwise_256x256x192x128_16x16_8x6_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 7168, fp8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 8192, fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3}
};

static const std::map<int, RowwiseKernel> N_8192_K_3584_dispatch_table = {
  { 8, fp8_rowwise_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_interwave_v2},
  { 16, fp8_rowwise_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2},
  { 32, fp8_rowwise_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v3},
  { 64, fp8_rowwise_256x32x64x512_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 128, fp8_rowwise_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 192, fp8_rowwise_256x64x96x256_16x16_2x3_16x16x1_16x16x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 256, fp8_rowwise_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 384, fp8_rowwise_256x128x96x256_32x32_1x3_16x16x1_16x16x1_1x64x1x4_8x8x1_1x1_intrawave_v3},
  { 512, fp8_rowwise_256x128x128x256_32x32_2x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 640, fp8_rowwise_256x128x160x128_32x32_1x5_8x32x1_8x32x1_1x64x1x4_8x8x1_1x1_intrawave_v3},
  { 896, fp8_rowwise_256x128x192x128_32x32_2x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 1024, fp8_rowwise_256x256x128x128_16x16_8x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 1280, fp8_rowwise_256x256x160x128_16x16_8x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 1792, fp8_rowwise_256x256x192x128_16x16_8x6_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 2048, fp8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 2304, fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 2368, fp8_rowwise_256x128x256x128_32x32_2x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 2816, fp8_rowwise_256x256x160x128_16x16_8x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 3584, fp8_rowwise_256x256x192x128_16x16_8x6_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 4256, fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 4864, fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 5376, fp8_rowwise_256x256x192x128_16x16_8x6_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 6272, fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 7168, fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 7424, fp8_rowwise_256x256x160x128_16x16_8x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 8192, fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3}
};

static const std::map<int, RowwiseKernel> N_1024_K_5120_dispatch_table = {
  { 32, fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2_8},
  { 64, fp8_rowwise_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2_2},
  { 128, fp8_rowwise_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2},
  { 192, fp8_rowwise_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v3},
  { 608, fp8_rowwise_256x32x64x512_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 1216, fp8_rowwise_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 2432, fp8_rowwise_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 3456, fp8_rowwise_256x128x96x256_32x32_1x3_16x16x1_16x16x1_1x64x1x4_8x8x1_1x1_intrawave_v3},
  { 4864, fp8_rowwise_256x128x128x256_32x32_2x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 5472, fp8_rowwise_256x128x160x128_32x32_1x5_8x32x1_8x32x1_1x64x1x4_8x8x1_1x1_intrawave_v3},
  { 6368, fp8_rowwise_256x128x192x128_32x32_2x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 6912, fp8_rowwise_256x256x96x128_16x16_8x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 8192, fp8_rowwise_256x256x128x128_16x16_8x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3}
};

static const std::map<int, RowwiseKernel> N_5120_K_1024_dispatch_table = {
  { 16, fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2},
  { 32, fp8_rowwise_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1},
  { 64, fp8_rowwise_128x32x16x256_16x16_1x1_16x8x1_16x8x1_1x32x1x4_4x4x1_1x1_interwave_v1},
  { 96, fp8_rowwise_128x16x32x256_16x16_1x1_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_intrawave_v1},
  { 192, fp8_rowwise_256x32x128x256_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 256, fp8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 320, fp8_rowwise_256x64x96x256_16x16_2x3_16x16x1_16x16x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 448, fp8_rowwise_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 640, fp8_rowwise_256x128x96x256_32x32_1x3_16x16x1_16x16x1_1x64x1x4_8x8x1_1x1_intrawave_v3},
  { 896, fp8_rowwise_256x128x128x256_32x32_2x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 1152, fp8_rowwise_256x128x160x128_32x32_1x5_8x32x1_8x32x1_1x64x1x4_8x8x1_1x1_intrawave_v3},
  { 1408, fp8_rowwise_256x128x192x128_32x32_2x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 1920, fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 2304, fp8_rowwise_256x256x160x128_16x16_8x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 2816, fp8_rowwise_256x256x192x128_16x16_8x6_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 3360, fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 3840, fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 4864, fp8_rowwise_256x256x160x128_16x16_8x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 5632, fp8_rowwise_256x256x192x128_16x16_8x6_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 6720, fp8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 7680, fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 8192, fp8_rowwise_256x256x192x128_16x16_8x6_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3}
};

static const std::map<int, RowwiseKernel> N_2048_K_5120_dispatch_table = {
  { 4, fp8_rowwise_256x16x64x128_16x16_1x1_16x16x1_8x32x1_1x16x1x16_4x4x1_1x1_intrawave_v2_8},
  { 8, fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2_4},
  { 64, fp8_rowwise_128x32x16x512_16x16_1x1_32x4x1_32x4x1_1x32x1x4_4x4x1_1x1_intrawave_v2},
  { 288, fp8_rowwise_256x32x64x512_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 576, fp8_rowwise_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 1216, fp8_rowwise_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 1664, fp8_rowwise_256x128x96x256_32x32_1x3_16x16x1_16x16x1_1x64x1x4_8x8x1_1x1_intrawave_v3},
  { 2432, fp8_rowwise_256x128x128x256_32x32_2x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 2944, fp8_rowwise_256x128x160x128_32x32_1x5_8x32x1_8x32x1_1x64x1x4_8x8x1_1x1_intrawave_v3},
  { 3456, fp8_rowwise_256x128x192x128_32x32_2x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 4864, fp8_rowwise_256x256x128x128_16x16_8x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 5888, fp8_rowwise_256x256x160x128_16x16_8x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 5984, fp8_rowwise_256x256x192x128_16x16_8x6_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
};

static const std::map<int, RowwiseKernel> N_896_K_5120_dispatch_table = {
  { 64, fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2_8},
  { 72, fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_8},
  { 80, fp8_rowwise_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_interwave_v2_2},
  { 160, fp8_rowwise_128x32x16x512_16x16_1x1_32x4x1_32x4x1_1x32x1x4_4x4x1_1x1_intrawave_v2},
  { 200, fp8_rowwise_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v3},
  { 256, fp8_rowwise_256x64x16x512_16x16_1x1_32x8x1_32x8x1_1x64x1x4_4x4x1_1x1_intrawave_v2},
  { 672, fp8_rowwise_256x32x64x512_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 1344, fp8_rowwise_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 2752, fp8_rowwise_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 3840, fp8_rowwise_256x128x96x256_32x32_1x3_16x16x1_16x16x1_1x64x1x4_8x8x1_1x1_intrawave_v3},
  { 5504, fp8_rowwise_256x128x128x256_32x32_2x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 5984, fp8_rowwise_256x128x160x128_32x32_1x5_8x32x1_8x32x1_1x64x1x4_8x8x1_1x1_intrawave_v3},
};

static const std::map<int, RowwiseKernel> N_5120_K_640_dispatch_table = {
  { 64, fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2},
  { 80, fp8_rowwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2},
  { 112, fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1},
  { 192, fp8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 224, fp8_rowwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2},
  { 256, fp8_rowwise_128x64x32x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2},
  { 384, fp8_rowwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 448, fp8_rowwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 512, fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 704, fp8_rowwise_256x64x192x128_32x32_1x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 896, fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 960, fp8_rowwise_256x64x256x128_32x32_1x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 1152, fp8_rowwise_256x128x160x128_32x32_1x5_8x32x1_8x32x1_1x64x1x4_8x8x1_1x1_intrawave_v3},
  { 1280, fp8_rowwise_256x256x96x128_32x32_2x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 1408, fp8_rowwise_256x128x192x128_32x32_2x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 1920, fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 2304, fp8_rowwise_256x256x160x128_16x16_8x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 2816, fp8_rowwise_256x256x192x128_16x16_8x6_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 3360, fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 3840, fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 4864, fp8_rowwise_256x256x160x128_16x16_8x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3},
  { 5520, fp8_rowwise_256x256x192x128_16x16_8x6_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
  { 5760, fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
  { 5984, fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
};

static const std::unordered_map<std::tuple<int, int>, NKLookupTableType, IntTupleHash> NK_lookup_table = {
  {{7168, 8192}, N_7168_K_8192_dispatch_table},
  {{8192, 3584}, N_8192_K_3584_dispatch_table},
  {{1024, 5120}, N_1024_K_5120_dispatch_table},
  {{5120, 1024}, N_5120_K_1024_dispatch_table},
  {{2048, 5120}, N_2048_K_5120_dispatch_table},
  {{896, 5120}, N_896_K_5120_dispatch_table},
  {{5120, 640}, N_5120_K_640_dispatch_table}
};

RowwiseKernel rowwise_nk_lookup(int M, const NKLookupTableType& table) {
  auto it = table.lower_bound(M);
  if (it != table.end()) {
    return it->second;
  } else {
    --it;
    return it->second;
  }
}

RowwiseKernel rowwise_heuristic_dispatch(int M, int N, int K) {
  // Apply shape heuristics to find a suitable kernel implementation.

  //Fallback of irregular data types
  if(!((N % 8 == 0) && (K % 16 == 0)))
      return fp8_rowwise_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1;

  if (M < 64 && N < 2048 && K < 2048) {
    // Kernel that generally works well on small shapes.
    return fp8_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2;
  } else if (M < 64 && K < 2048) {
    // Kernel that works well for small batch size and small K.
    return fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2;
  } else if (M < 64 && N < 2048) {
    // Kernel that works well for small batch size and small N.
    return fp8_rowwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2;
  } else if (M < 64 && N > 2048 && K > 2048) {
    // Kernel that works well for small M but larger N and K.
    return fp8_rowwise_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1;
  } else if (M < 64) {
    // Fallback to generic small batch kernel if we cant find a good match.
    return fp8_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2;
  } else if (
      ((M < 512 && K < 8192) || (N <= 2048 && K <= 8192) ||
       (K <= 2048 && N <= 8192)) &&
      K >= 1024) {
    // Kernel that is optimized for larger batch sizes but otherwise small
    // tensors.
    return fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5;
  } else if (K < 1024) {
    // Special case for small K.
    return fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1;
  } else if (M < 1024) {
    // Kernel for generic medium batch sizes.
    return fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
  } else if (M >= 1024 && N >= 1024 && K >= 1024) {
    // Kernel for very large gemm
    return fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3;
  } else {
    // Fallback large kernel.
    return fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3;
  }
}

RowwiseKernel rowwise_dispatch(int M, int N, int K) {
  // For a given shape, either find the best kernel via lookup or heuristic.
  // For many small M shapes, we bucket them to the next largest kernel.
  // This is fine since kernels are padded anyway.
  int padded_m = M;
  if (M <= 16) {
    padded_m = 16;
  } else if (M <= 32) {
    padded_m = 32;
  } else if (M <= 64) {
    padded_m = 64;
  } else if (M <= 128) {
    padded_m = 128;
  }
  // First check if this shape is available in the direct lookup.
  auto it = rowwise_lookup_dispatch.find({padded_m, N, K});
  // If we found an optimal kernel, use it.
  if (it != rowwise_lookup_dispatch.end()) {
    return it->second;
  } else {
    auto nk_lookup_it = NK_lookup_table.find({N,K});
    if (nk_lookup_it != NK_lookup_table.end()){
      return rowwise_nk_lookup(M, nk_lookup_it->second);
    }
  }
  // Otherwise, use heuristics.
  return rowwise_heuristic_dispatch(M, N, K);
}

at::Tensor f8f8bf16_rowwise_wrapper(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::Tensor> output = std::nullopt) {
  // Check that input datatypes are valid.
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

  // XQ: M x K
  // WQ: N x K
  // output: M x N
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = WQ.size(0);
  int K = WQ.size(1);
  // Compute target output sizes.
  auto out_sizes = XQ.sizes().vec();
  out_sizes.back() = N;
  // Handle case where an input dimension is zero.
  if (M == 0 || N == 0 || K == 0) {
    // Return a tensor of zeros to handle case where K is 0.
    return at::zeros(out_sizes, XQ.options().dtype(at::kBFloat16));
  }

  // Prepare output tensor if needed.
  at::Tensor Y;
  if (output.has_value()) {
    Y = output.value();
    // Make sure the provided output has the proper shape and dtype.
    int Y_M = size_to_dim_(Y.dim() - 1, Y.sizes());
    TORCH_CHECK(Y_M == M && Y.sizes().vec().back() == N);
    TORCH_CHECK(Y.dtype() == at::kBFloat16);
  } else {
    Y = at::empty(out_sizes, XQ.options().dtype(at::kBFloat16));
  }

  RowwiseKernel rowwise_impl = rowwise_dispatch(M, N, K);
  return rowwise_impl(XQ, WQ, x_scale, w_scale, Y);
}

at::Tensor f8f8bf16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum) {
  // Invoke f8f8bf16 rowwise without preallocated output.
  return f8f8bf16_rowwise_wrapper(
      XQ, WQ, x_scale, w_scale, bias, use_fast_accum);
}

void f8f8bf16_rowwise_out(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> bias,
    bool use_fast_accum) {
  // Invoke f8f8bf16 rowwise with preallocated output.
  f8f8bf16_rowwise_wrapper(
      XQ, WQ, x_scale, w_scale, bias, use_fast_accum, output);
}

} // namespace fbgemm_gpu
