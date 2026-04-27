/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Benchmark for NBit embedding with fp16 output.
// Compares SVE FP16 register path (via GenerateEmbeddingSpMDMNBitWithStrides)
// against autovec baseline.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "./BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "fbgemm/FbgemmConvert.h"
#include "src/EmbeddingSpMDMAutovec.h"
#include "src/RefImplementations.h" // @manual

using namespace std;
using namespace fbgemm;

static vector<vector<int>> GetInputs_() {
  vector<vector<int>> input_dims = {
      // batch size, number of rows of table, emb dim, avg length
      {10, 4000000, 32, 100},
      {10, 4000000, 64, 100},
      {10, 4000000, 128, 100},
      {10, 4000000, 256, 100},
      {10, 4000000, 512, 100},
      {10, 4000000, 1024, 100}};
  return input_dims;
}

static int run_benchmark(
    int bit_rate,
    int batch_size,
    int num_rows,
    int embedding_dim,
    int average_len,
    bool normalize_by_lengths,
    bool use_32_bit_indices = false,
    bool prefetch = false) {
  int num_elem_per_byte = 8 / bit_rate;
  int fused_embedding_dim =
      (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte +
      2 * sizeof(float16);
  default_random_engine generator;
  normal_distribution<float> embedding_distribution;

  vector<uint8_t> fused_embedding_table(
      static_cast<size_t>(num_rows) * fused_embedding_dim);
  for (int i = 0; i < num_rows; i++) {
    for (int ii = 0;
         ii < (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte;
         ii++) {
      fused_embedding_table[static_cast<size_t>(i) * fused_embedding_dim + ii] =
          2;
    }
    float16* scale_bias = reinterpret_cast<float16*>(
        &fused_embedding_table[static_cast<size_t>(i) * fused_embedding_dim] +
        (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte);
    float scale = 2.0f;
    float bias = 1.0f;
    FloatToFloat16_ref(&scale, scale_bias, 1, true);
    FloatToFloat16_ref(&bias, scale_bias + 1, 1, true);
  }

  uniform_int_distribution<int> length_distribution(
      1, std::min(2 * average_len + 1, num_rows));
  vector<int> offsets(batch_size + 1);
  offsets[0] = 0;
  for (int i = 0; i < batch_size; ++i) {
    offsets[i + 1] = offsets[i] + length_distribution(generator);
  }
  int lengths_sum = offsets[batch_size];
  cout << "lengths_sum " << lengths_sum << '\n';

  vector<int64_t> indices;
  vector<int32_t> indices_32;
  vector<int> container(num_rows);
  for (int i = 0; i < batch_size; ++i) {
    iota(container.begin(), container.end(), 0);
    shuffle(container.begin(), container.end(), generator);
    copy(
        container.begin(),
        container.begin() + (offsets[i + 1] - offsets[i]),
        back_inserter(indices));
  }
  copy(begin(indices), end(indices), back_inserter(indices_32));

  vector<float> weights(lengths_sum);
  for (int i = 0; i < lengths_sum; ++i) {
    weights[i] = embedding_distribution(generator);
  }

  using OutType = float16;
  vector<OutType> output_ref(static_cast<size_t>(batch_size) * embedding_dim);
  vector<OutType> output(output_ref.size());
  vector<OutType> output_autovec(output_ref.size());

  constexpr int NUM_WARMUP = 10;
  constexpr int NUM_ITER = 100;
  double bytes = static_cast<double>(lengths_sum) * fused_embedding_dim;
  constexpr int CACHE_LINE_LEN = 64;
  double bytes_padded = static_cast<double>(lengths_sum) * CACHE_LINE_LEN *
      static_cast<int>((fused_embedding_dim + CACHE_LINE_LEN - 1) /
                       CACHE_LINE_LEN);

  for (bool has_weight : {false, true}) {
    // Main kernel via GenerateEmbeddingSpMDMNBitWithStrides (includes SVE FP16
    // dispatch when FBGEMM_SVE_FP16=1)
    auto kernel_32 = GenerateEmbeddingSpMDMNBitWithStrides<
        /*IndexType=*/int32_t,
        /*OffsetType=*/int32_t,
        /*OutType=*/OutType>(
        bit_rate,
        embedding_dim,
        has_weight,
        normalize_by_lengths,
        prefetch ? 16 : 0,
        /*is_weight_positional=*/false,
        /*use_offsets=*/true,
        /*output_stride=*/-1,
        /*input_stride=*/-1,
        /*scale_bias_last=*/true,
        /*is_bf16_out=*/false);
    auto kernel_64 = GenerateEmbeddingSpMDMNBitWithStrides<
        /*IndexType=*/int64_t,
        /*OffsetType=*/int32_t,
        /*OutType=*/OutType>(
        bit_rate,
        embedding_dim,
        has_weight,
        normalize_by_lengths,
        prefetch ? 16 : 0,
        /*is_weight_positional=*/false,
        /*use_offsets=*/true,
        /*output_stride=*/-1,
        /*input_stride=*/-1,
        /*scale_bias_last=*/true,
        /*is_bf16_out=*/false);

#ifdef FBGEMM_AUTOVEC_AVAILABLE
    auto kernel_32_autovec = GenerateEmbeddingSpMDMNBitWithStrides_autovec<
        /*IndexType=*/int32_t,
        /*OffsetType=*/int32_t,
        /*OutType=*/OutType>(
        bit_rate,
        embedding_dim,
        has_weight,
        normalize_by_lengths,
        prefetch ? 16 : 0,
        /*is_weight_positional=*/false,
        /*use_offsets=*/true,
        /*output_stride=*/-1,
        /*input_stride=*/-1,
        /*scale_bias_last=*/true,
        /*is_bf16_out=*/false,
        /*no_bag=*/false,
        /*output_bit_rate=*/-1);
    auto kernel_64_autovec = GenerateEmbeddingSpMDMNBitWithStrides_autovec<
        /*IndexType=*/int64_t,
        /*OffsetType=*/int32_t,
        /*OutType=*/OutType>(
        bit_rate,
        embedding_dim,
        has_weight,
        normalize_by_lengths,
        prefetch ? 16 : 0,
        /*is_weight_positional=*/false,
        /*use_offsets=*/true,
        /*output_stride=*/-1,
        /*input_stride=*/-1,
        /*scale_bias_last=*/true,
        /*is_bf16_out=*/false,
        /*no_bag=*/false,
        /*output_bit_rate=*/-1);
#endif

    for (bool flush_cache : {false, true}) {
      // Main kernel
      double t = measureWithWarmup(
          [&]() {
            if (use_32_bit_indices) {
              kernel_32(
                  batch_size,
                  lengths_sum,
                  num_rows,
                  fused_embedding_table.data(),
                  indices_32.data(),
                  offsets.data(),
                  has_weight ? weights.data() : nullptr,
                  output.data());
            } else {
              kernel_64(
                  batch_size,
                  lengths_sum,
                  num_rows,
                  fused_embedding_table.data(),
                  indices.data(),
                  offsets.data(),
                  has_weight ? weights.data() : nullptr,
                  output.data());
            }
          },
          NUM_WARMUP,
          NUM_ITER,
          [&]() {
            if (flush_cache) {
              cache_evict(fused_embedding_table);
              cache_evict(indices);
              cache_evict(indices_32);
              cache_evict(offsets);
              cache_evict(weights);
              cache_evict(output);
            }
          });

#ifdef FBGEMM_AUTOVEC_AVAILABLE
      // Autovec kernel
      double t_autovec = measureWithWarmup(
          [&]() {
            if (use_32_bit_indices) {
              kernel_32_autovec(
                  batch_size,
                  lengths_sum,
                  num_rows,
                  fused_embedding_table.data(),
                  indices_32.data(),
                  offsets.data(),
                  has_weight ? weights.data() : nullptr,
                  output_autovec.data());
            } else {
              kernel_64_autovec(
                  batch_size,
                  lengths_sum,
                  num_rows,
                  fused_embedding_table.data(),
                  indices.data(),
                  offsets.data(),
                  has_weight ? weights.data() : nullptr,
                  output_autovec.data());
            }
          },
          NUM_WARMUP,
          NUM_ITER,
          [&]() {
            if (flush_cache) {
              cache_evict(fused_embedding_table);
              cache_evict(indices);
              cache_evict(indices_32);
              cache_evict(offsets);
              cache_evict(weights);
              cache_evict(output_autovec);
            }
          });
#endif

      // Output
      cout << "out type fp16, ";
      if (has_weight) {
        cout << "SLW(WEIGHTED), ";
      } else {
        cout << "SLS, ";
      }
      if (flush_cache) {
        cout << "cache flushed, ";
      } else {
        cout << "cache not flushed, ";
      }
      if (prefetch) {
        cout << "prefetch on, ";
      } else {
        cout << "prefetch off, ";
      }

      cout << "b/w, " << bytes / 1e9 / t << ", GB/s, "
           << "effective b/w, " << bytes_padded / 1e9 / t << ", GB/s, "
           << "time, " << t;
#ifdef FBGEMM_AUTOVEC_AVAILABLE
      cout << ", autovec b/w, " << bytes / 1e9 / t_autovec << ", GB/s, "
           << "autovec eff. b/w, " << bytes_padded / 1e9 / t_autovec
           << ", GB/s, "
           << "autovec time, " << t_autovec << ", speedup vs autovec, "
           << t_autovec / t;
#endif
      cout << '\n';
      cout.flush();
    } // flush_cache
  } // has_weight
  return 0;
}

int main() {
  vector<vector<int>> inputs(GetInputs_());

  for (int bit_rate : {4, 2}) {
    for (auto& input : inputs) {
      assert(input.size() > 3);
      int batch_size = input[0];
      int num_rows = input[1];
      int embedding_dim = input[2];
      int average_len = input[3];

      cout << "bit_rate, " << bit_rate << ", batch size, " << batch_size
           << ", num rows, " << num_rows << ", emb dim, " << embedding_dim
           << ", avg length, " << average_len << '\n';

      // 64-bit indices, no prefetch
      cout << "64 bit indices, ";
      run_benchmark(
          bit_rate, batch_size, num_rows, embedding_dim, average_len, false);

      // 64-bit indices, with prefetch
      cout << "64 bit indices with prefetching, ";
      run_benchmark(
          bit_rate,
          batch_size,
          num_rows,
          embedding_dim,
          average_len,
          false,
          /*use_32_bit_indices=*/false,
          /*prefetch=*/true);

      // 32-bit indices, no prefetch
      cout << "32 bit indices, ";
      run_benchmark(
          bit_rate,
          batch_size,
          num_rows,
          embedding_dim,
          average_len,
          false,
          /*use_32_bit_indices=*/true,
          /*prefetch=*/false);

      // 32-bit indices, with prefetch
      cout << "32 bit indices with prefetching, ";
      run_benchmark(
          bit_rate,
          batch_size,
          num_rows,
          embedding_dim,
          average_len,
          false,
          /*use_32_bit_indices=*/true,
          /*prefetch=*/true);
    }
  }
  return 0;
}
