/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/quantize/utils.h"
#include "fp8_rowwise_grouped_kernel_manifest.h"

namespace fbgemm_gpu {
namespace {
#if !defined(FBGEMM_GENAI_NO_EXTENDED_SHAPES)

// Define a custom hash function for std::tuple<int, int, int>
struct IntTupleHash {
  size_t operator()(const std::tuple<int64_t, int64_t>& t) const {
    auto hash1 = std::hash<int64_t>{}(std::get<0>(t));
    auto hash2 = std::hash<int64_t>{}(std::get<1>(t));
    return hash1 ^ hash2;
  }
  size_t operator()(const std::tuple<int64_t, int64_t, int64_t>& t) const {
    auto hash1 = std::hash<int64_t>{}(std::get<0>(t));
    auto hash2 = std::hash<int64_t>{}(std::get<1>(t));
    auto hash3 = std::hash<int64_t>{}(std::get<2>(t));
    return hash1 ^ hash2 ^ hash3;
  }
  size_t operator()(
      const std::tuple<int64_t, int64_t, int64_t, int64_t>& t) const {
    auto hash1 = std::hash<int64_t>{}(std::get<0>(t));
    auto hash2 = std::hash<int64_t>{}(std::get<1>(t));
    auto hash3 = std::hash<int64_t>{}(std::get<2>(t));
    auto hash4 = std::hash<int64_t>{}(std::get<3>(t));
    return hash1 ^ hash2 ^ hash3 ^ hash4;
  }
};
// For certain high priority shapes, we directly map to the best kernel rather
// than use heuristics.
// clang-format off
template <typename InputType, typename OutputType>
static const std::unordered_map<std::tuple<int64_t, int64_t, int64_t, int64_t>, RowwiseGroupedKernel<InputType, OutputType>, IntTupleHash> rowwise_grouped_lookup_dispatch = {
{{16,16,2048,5120},fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v2<InputType, OutputType>},
{{16,16,5120,1024},fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_interwave_v2<InputType, OutputType>},
{{16,16,16384,5120},fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v1<InputType, OutputType>},
{{16,16,5120,8192},fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_interwave_v2<InputType, OutputType>},
{{16,32,2048,5120},fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v2<InputType, OutputType>},
{{16,32,5120,1024},fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_interwave_v2<InputType, OutputType>},
{{16,32,16384,5120},fp8_rowwise_grouped_128x16x96x256_16x16_1x3_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_intrawave_v1<InputType, OutputType>},
{{16,32,5120,8192},fp8_rowwise_grouped_128x16x32x256_16x16_1x1_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_interwave_v2<InputType, OutputType>},
{{16,64,2048,5120},fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v2<InputType, OutputType>},
{{16,64,5120,1024},fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v1<InputType, OutputType>},
{{16,64,16384,5120},fp8_rowwise_grouped_64x16x64x256_16x16_1x4_16x4x1_16x4x1_1x16x1x4_8x8x1_1x2_intrawave_v1<InputType, OutputType>},
{{16,64,5120,8192},fp8_rowwise_grouped_256x32x32x512_16x16_1x1_32x8x1_32x8x1_1x32x1x8_4x4x1_1x1_interwave_v2<InputType, OutputType>},
{{16,128,2048,5120},fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_intrawave_v1<InputType, OutputType>},
{{16,128,5120,1024},fp8_rowwise_grouped_64x16x32x256_16x16_1x2_16x4x1_16x4x1_1x16x1x4_8x8x1_1x2_intrawave_v1<InputType, OutputType>},
{{16,128,16384,5120},fp8_rowwise_grouped_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<InputType, OutputType>},
{{16,128,5120,8192},fp8_rowwise_grouped_128x16x32x256_16x16_1x1_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_interwave_v2<InputType, OutputType>},
{{16,256,2048,5120},fp8_rowwise_grouped_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<InputType, OutputType>},
{{16,256,5120,1024},fp8_rowwise_grouped_128x16x32x256_16x16_1x1_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_interwave_v2<InputType, OutputType>},
{{16,256,16384,5120},fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<InputType, OutputType>},
{{16,256,5120,8192},fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_interwave_v2<InputType, OutputType>},
{{16,512,2048,5120},fp8_rowwise_grouped_128x32x64x256_32x32_1x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_interwave_v1<InputType, OutputType>},
{{16,512,5120,1024},fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v1<InputType, OutputType>},
{{16,512,16384,5120},fp8_rowwise_grouped_128x32x64x256_16x16_1x4_16x8x1_16x8x1_1x32x1x4_8x8x1_1x2_intrawave_v1<InputType, OutputType>},
{{16,512,5120,8192},fp8_rowwise_grouped_256x32x32x512_16x16_1x1_32x8x1_32x8x1_1x32x1x8_4x4x1_1x1_intrawave_v2<InputType, OutputType>},
{{16,1024,2048,5120},fp8_rowwise_grouped_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<InputType, OutputType>},
{{16,1024,5120,1024},fp8_rowwise_grouped_256x64x160x128_16x16_2x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<InputType, OutputType>},
{{16,1024,16384,5120},fp8_rowwise_grouped_256x64x128x256_32x32_2x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3<InputType, OutputType>},
{{16,1024,5120,8192},fp8_rowwise_grouped_128x64x64x256_32x32_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_intrawave_v3<InputType, OutputType>},
{{16,2048,2048,5120},fp8_rowwise_grouped_256x128x128x256_32x32_2x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<InputType, OutputType>},
{{16,2048,5120,1024},fp8_rowwise_grouped_256x128x96x128_16x16_4x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<InputType, OutputType>},
{{16,2048,16384,5120},fp8_rowwise_grouped_256x128x224x128_16x16_4x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<InputType, OutputType>},
{{16,2048,5120,8192},fp8_rowwise_grouped_256x128x96x128_16x16_4x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<InputType, OutputType>},
{{16,4096,2048,5120},fp8_rowwise_grouped_256x128x256x128_32x32_4x2_8x32x1_8x32x1_1x16x1x16_8x8x1_1x1_intrawave_v3<InputType, OutputType>},
{{16,4096,5120,1024},fp8_rowwise_grouped_256x128x96x128_16x16_4x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<InputType, OutputType>},
{{16,4096,16384,5120},fp8_rowwise_grouped_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<InputType, OutputType>},
{{16,4096,5120,8192},fp8_rowwise_grouped_256x256x160x128_32x32_2x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<InputType, OutputType>},
{{16,8192,2048,5120},fp8_rowwise_grouped_256x256x256x128_32x32_8x2_8x32x1_8x32x1_1x16x1x16_8x8x1_1x1_intrawave_v3<InputType, OutputType>},
{{16,8192,5120,1024},fp8_rowwise_grouped_256x256x192x128_32x32_4x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<InputType, OutputType>},
{{16,8192,16384,5120},fp8_rowwise_grouped_256x256x256x128_32x32_4x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<InputType, OutputType>},
{{16,8192,5120,8192},fp8_rowwise_grouped_256x256x192x128_32x32_4x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<InputType, OutputType>},
{{128,128,2048,5120},fp8_rowwise_grouped_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_interwave_v2<InputType, OutputType>},
{{128,128,5120,1024},fp8_rowwise_grouped_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<InputType, OutputType>},
{{128,128,16384,5120},fp8_rowwise_grouped_64x16x64x256_16x16_1x4_16x4x1_16x4x1_1x16x1x4_8x8x1_1x2_interwave_v1<InputType, OutputType>},
{{128,128,5120,8192},fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_interwave_v2<InputType, OutputType>},
{{128,256,2048,5120},fp8_rowwise_grouped_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v2<InputType, OutputType>},
{{128,256,5120,1024},fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v2<InputType, OutputType>},
{{128,256,16384,5120},fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v2<InputType, OutputType>},
{{128,256,5120,8192},fp8_rowwise_grouped_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<InputType, OutputType>},
{{128,512,2048,5120},fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_intrawave_v1<InputType, OutputType>},
{{128,512,5120,1024},fp8_rowwise_grouped_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<InputType, OutputType>},
{{128,512,16384,5120},fp8_rowwise_grouped_64x16x64x256_16x16_1x4_16x4x1_16x4x1_1x16x1x4_8x8x1_1x2_interwave_v1<InputType, OutputType>},
{{128,512,5120,8192},fp8_rowwise_grouped_128x16x32x256_16x16_1x1_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_interwave_v2<InputType, OutputType>},
{{128,1024,2048,5120},fp8_rowwise_grouped_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<InputType, OutputType>},
{{128,1024,5120,1024},fp8_rowwise_grouped_256x192x96x128_16x16_6x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<InputType, OutputType>},
{{128,1024,16384,5120},fp8_rowwise_grouped_64x16x64x256_16x16_1x4_16x4x1_16x4x1_1x16x1x4_8x8x1_1x2_intrawave_v1<InputType, OutputType>},
{{128,1024,5120,8192},fp8_rowwise_grouped_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<InputType, OutputType>},
{{128,2048,2048,5120},fp8_rowwise_grouped_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v2<InputType, OutputType>},
{{128,2048,5120,1024},fp8_rowwise_grouped_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v2<InputType, OutputType>},
{{128,2048,16384,5120},fp8_rowwise_grouped_64x16x64x256_16x16_1x4_16x4x1_16x4x1_1x16x1x4_8x8x1_1x2_interwave_v1<InputType, OutputType>},
{{128,2048,5120,8192},fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v2<InputType, OutputType>},
{{128,4096,2048,5120},fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<InputType, OutputType>},
{{128,4096,5120,1024},fp8_rowwise_grouped_256x32x128x128_16x16_1x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_interwave_v2<InputType, OutputType>},
{{128,4096,16384,5120},fp8_rowwise_grouped_256x32x256x128_16x16_1x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v1<InputType, OutputType>},
{{128,4096,5120,8192},fp8_rowwise_grouped_256x32x64x512_16x16_2x1_32x8x1_32x8x1_1x32x1x8_8x8x1_2x1_intrawave_v2<InputType, OutputType>},
{{128,8192,2048,5120},fp8_rowwise_grouped_256x64x128x256_32x32_2x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3<InputType, OutputType>},
{{128,8192,5120,1024},fp8_rowwise_grouped_256x64x192x128_16x16_4x3_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<InputType, OutputType>},
{{128,8192,16384,5120},fp8_rowwise_grouped_256x64x128x256_32x32_2x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3<InputType, OutputType>},
{{128,8192,5120,8192},fp8_rowwise_grouped_128x64x64x256_32x32_2x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_intrawave_v3<InputType, OutputType>},
};
// clang-format on
#endif
} // namespace

template <typename InputType, typename OutputType>
RowwiseGroupedKernel<InputType, OutputType>
get_kernel_via_heuristic(int64_t M, int64_t N, int64_t K) {
  if (M <= 1) {
    if (N <= 128) {
      if (K <= 1024) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      }
    } else if (N <= 256) {
      if (K <= 256) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      }
    } else if (N <= 512) {
      if (K <= 512) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_interwave_v2<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      }
    } else if (N <= 1024) {
      if (K <= 128) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 256) {
        return fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 2048) {
      if (K <= 128) {
        return fp8_rowwise_grouped_64x16x32x256_16x16_1x2_16x4x1_16x4x1_1x16x1x4_8x8x1_1x2_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 256) {
        return fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_64x16x32x256_16x16_1x2_16x4x1_16x4x1_1x16x1x4_8x8x1_1x2_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 4096) {
        return fp8_rowwise_grouped_256x32x128x128_16x16_1x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_interwave_v2<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<
            InputType,
            OutputType>;
      }
    } else if (N <= 4096) {
      if (K <= 512) {
        return fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 4096) {
        return fp8_rowwise_grouped_256x32x128x128_16x16_1x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_interwave_v2<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_interwave_v2<
            InputType,
            OutputType>;
      }
    } else {
      if (K <= 256) {
        return fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_256x32x128x128_16x16_1x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 4096) {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_interwave_v2<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x32x32x512_16x16_1x1_32x8x1_32x8x1_1x32x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      }
    }
  } else if (M <= 16) {
    if (N <= 128) {
      if (K <= 1024) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      }
    } else if (N <= 256) {
      if (K <= 256) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      }
    } else if (N <= 512) {
      if (K <= 512) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 4096) {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      }
    } else if (N <= 1024) {
      if (K <= 256) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 4096) {
        return fp8_rowwise_grouped_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 2048) {
      if (K <= 512) {
        return fp8_rowwise_grouped_64x16x32x256_16x16_1x2_16x4x1_16x4x1_1x16x1x4_8x8x1_1x2_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 4096) {
        return fp8_rowwise_grouped_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_interwave_v2<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v2<
            InputType,
            OutputType>;
      }
    } else if (N <= 4096) {
      if (K <= 1024) {
        return fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_256x32x128x128_16x16_1x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 4096) {
        return fp8_rowwise_grouped_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      }
    } else {
      if (K <= 256) {
        return fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_64x16x32x256_16x16_1x2_16x4x1_16x4x1_1x16x1x4_8x8x1_1x2_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 4096) {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      }
    }
  } else if (M <= 32) {
    if (N <= 128) {
      if (K <= 128) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 256) {
        return fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_256x32x32x512_16x16_1x1_32x8x1_32x8x1_1x32x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      }
    } else if (N <= 256) {
      if (K <= 128) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 256) {
        return fp8_rowwise_grouped_128x32x64x256_32x32_1x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      }
    } else if (N <= 512) {
      if (K <= 512) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x32x64x512_16x16_2x1_32x8x1_32x8x1_1x32x1x8_8x8x1_2x1_intrawave_v2<
            InputType,
            OutputType>;
      }
    } else if (N <= 1024) {
      if (K <= 256) {
        return fp8_rowwise_grouped_128x32x64x256_32x32_1x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_256x32x64x512_16x16_2x1_32x8x1_32x8x1_1x32x1x8_8x8x1_2x1_intrawave_v2<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x32x64x256_32x32_1x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      }
    } else if (N <= 2048) {
      return fp8_rowwise_grouped_128x32x64x256_32x32_1x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_interwave_v1<
          InputType,
          OutputType>;
    } else if (N <= 4096) {
      if (K <= 128) {
        return fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_128x32x64x256_32x32_1x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_256x32x128x128_16x16_1x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 4096) {
        return fp8_rowwise_grouped_128x32x64x256_32x32_1x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x32x32x512_16x16_1x1_32x8x1_32x8x1_1x32x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      }
    } else {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_256x32x128x128_16x16_1x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_128x32x64x256_32x32_1x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 4096) {
        return fp8_rowwise_grouped_256x32x64x512_16x16_2x1_32x8x1_32x8x1_1x32x1x8_8x8x1_2x1_intrawave_v2<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      }
    }
  } else if (M <= 64) {
    if (N <= 128) {
      if (K <= 512) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      }
    } else if (N <= 256) {
      if (K <= 512) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      }
    } else if (N <= 512) {
      if (K <= 512) {
        return fp8_rowwise_grouped_64x16x32x256_16x16_1x2_16x4x1_16x4x1_1x16x1x4_8x8x1_1x2_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_256x32x32x512_16x16_1x1_32x8x1_32x8x1_1x32x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x32x64x512_16x16_2x1_32x8x1_32x8x1_1x32x1x8_8x8x1_2x1_intrawave_v2<
            InputType,
            OutputType>;
      }
    } else if (N <= 1024) {
      if (K <= 128) {
        return fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_128x32x64x256_32x32_1x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x64x64x256_32x32_2x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 2048) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_256x32x128x128_16x16_1x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_128x64x64x256_32x32_2x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x64x128x256_32x32_2x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 4096) {
      if (K <= 128) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 256) {
        return fp8_rowwise_grouped_128x32x64x256_32x32_1x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_256x64x160x128_16x16_2x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<
            InputType,
            OutputType>;
      } else if (K <= 4096) {
        return fp8_rowwise_grouped_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x64x64x256_32x32_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_256x64x160x128_16x16_2x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x64x64x256_32x32_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_intrawave_v3<
            InputType,
            OutputType>;
      }
    }
  } else if (M <= 128) {
    if (N <= 128) {
      if (K <= 512) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      } else if (K <= 4096) {
        return fp8_rowwise_grouped_256x32x32x512_16x16_1x1_32x8x1_32x8x1_1x32x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x32x64x512_16x16_2x1_32x8x1_32x8x1_1x32x1x8_8x8x1_2x1_intrawave_v2<
            InputType,
            OutputType>;
      }
    } else if (N <= 256) {
      if (K <= 512) {
        return fp8_rowwise_grouped_64x16x32x256_16x16_1x2_16x4x1_16x4x1_1x16x1x4_8x8x1_1x2_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_256x32x32x512_16x16_1x1_32x8x1_32x8x1_1x32x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      } else if (K <= 4096) {
        return fp8_rowwise_grouped_256x32x64x512_16x16_2x1_32x8x1_32x8x1_1x32x1x8_8x8x1_2x1_intrawave_v2<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x32x64x256_32x32_1x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      }
    } else if (N <= 512) {
      if (K <= 2048) {
        return fp8_rowwise_grouped_128x32x64x256_32x32_1x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x64x64x256_32x32_2x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 1024) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_256x128x96x128_16x16_4x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x64x128x256_32x32_2x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 2048) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_256x128x96x128_16x16_4x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 4096) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    }
  } else if (M <= 256) {
    if (N <= 128) {
      if (K <= 512) {
        return fp8_rowwise_grouped_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_256x32x32x512_16x16_1x1_32x8x1_32x8x1_1x32x1x8_4x4x1_1x1_intrawave_v2<
            InputType,
            OutputType>;
      } else if (K <= 4096) {
        return fp8_rowwise_grouped_256x32x64x512_16x16_2x1_32x8x1_32x8x1_1x32x1x8_8x8x1_2x1_intrawave_v2<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x64x64x256_32x32_2x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 256) {
      if (K <= 4096) {
        return fp8_rowwise_grouped_128x32x64x256_32x32_1x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x64x64x256_32x32_2x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 512) {
      if (K <= 128) {
        return fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 256) {
        return fp8_rowwise_grouped_128x32x64x256_32x32_1x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_256x32x128x128_16x16_1x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_128x64x64x256_32x32_2x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x64x128x256_32x32_2x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 1024) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 2048) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 4096) {
      if (K <= 128) {
        return fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x96x128_16x16_4x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    }
  } else if (M <= 512) {
    if (N <= 128) {
      if (K <= 128) {
        return fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 256) {
        return fp8_rowwise_grouped_128x16x64x256_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_256x32x128x128_16x16_1x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 4096) {
        return fp8_rowwise_grouped_128x32x64x256_32x32_1x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_128x64x64x256_32x32_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 256) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_256x32x128x128_16x16_1x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_interwave_v2<
            InputType,
            OutputType>;
      } else if (K <= 1024) {
        return fp8_rowwise_grouped_128x64x64x256_32x32_2x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_256x64x128x256_32x32_2x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      } else if (K <= 4096) {
        return fp8_rowwise_grouped_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x64x128x256_32x32_2x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 512) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 1024) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 2048) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x96x128_16x16_4x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 4096) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    }
  } else if (M <= 1024) {
    if (N <= 128) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_256x32x128x128_16x16_1x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_interwave_v2<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x64x128x256_32x32_2x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 256) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 512) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 1024) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x96x128_16x16_4x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 2048) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 4096) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    }
  } else if (M <= 2048) {
    if (N <= 128) {
      if (K <= 128) {
        return fp8_rowwise_grouped_256x32x160x128_16x16_1x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v1<
            InputType,
            OutputType>;
      } else if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_256x64x160x128_16x16_2x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 256) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 512) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 2048) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      } else if (K <= 4096) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 1024) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 2048) {
      if (K <= 512) {
        return fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 4096) {
      if (K <= 512) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else {
      if (K <= 512) {
        return fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    }
  } else if (M <= 4096) {
    if (N <= 128) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 256) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 512) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 1024) {
      if (K <= 512) {
        return fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 2048) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 4096) {
      if (K <= 512) {
        return fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else {
      if (K <= 512) {
        return fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    }
  } else {
    if (N <= 128) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 256) {
      if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 512) {
      if (K <= 128) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 256) {
        return fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 1024) {
      if (K <= 128) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 256) {
        return fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 2048) {
      if (K <= 128) {
        return fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 512) {
        return fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else if (N <= 4096) {
      if (K <= 128) {
        return fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else if (K <= 256) {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    } else {
      if (K <= 512) {
        return fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1<
            InputType,
            OutputType>;
      } else {
        return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
            InputType,
            OutputType>;
      }
    }
  }
}

template <typename InputType, typename OutputType>
RowwiseGroupedKernel<InputType, OutputType> rowwise_grouped_heuristic_dispatch(
    int64_t G,
    int64_t total_M,
    int64_t N,
    int64_t K) {
// Avoid compiling full range of kernels when building with PyTorch.
#if !defined(FBGEMM_GENAI_NO_EXTENDED_SHAPES)
  // First check if this shape is available in the direct lookup.
  int64_t padded_m = nextPowerOf2(total_M);
  padded_m = padded_m < G ? G : padded_m;
  padded_m = padded_m > 8192 ? 8192 : padded_m;
  auto it = rowwise_grouped_lookup_dispatch<InputType, OutputType>.find(
      {G, padded_m, N, K});
  // If we found an optimal kernel, use it.
  if (it != rowwise_grouped_lookup_dispatch<InputType, OutputType>.end()) {
    return it->second;
  }
#endif

  // Fallback to general heuristic.
  return get_kernel_via_heuristic<InputType, OutputType>(total_M / G, N, K);
}

} // namespace fbgemm_gpu
