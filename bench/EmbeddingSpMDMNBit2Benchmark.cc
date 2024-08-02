/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(__x86_64__) || defined(__i386__) || \
    (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#include <immintrin.h>
#endif
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
#include <utility>
#include <vector>

#include "./BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "fbgemm/FbgemmConvert.h"
#include "src/EmbeddingSpMDMAutovec.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

namespace {

enum KernelType {
  REF = 1,
  AUTOVEC = 2,
  ASMJIT = 3,
};

struct BenchmarkSpec {
  int bit_rate;
  int batch_size;
  int num_rows;
  int emb_dims;
  int avg_length;
  int indices_bits;
  int lengths_sum;
  bool has_weight;
  bool cache_flushed;
  bool prefetch;

  // Constructor that takes parameters and fills in all the fields in the struct
  BenchmarkSpec(
      int bit_rate,
      int batch_size,
      int num_rows,
      int emb_dims,
      int avg_length,
      int indices_bits,
      int lengths_sum,
      bool has_weight,
      bool cache_flushed,
      bool prefetch)
      : bit_rate(bit_rate),
        batch_size(batch_size),
        num_rows(num_rows),
        emb_dims(emb_dims),
        avg_length(avg_length),
        indices_bits(indices_bits),
        lengths_sum(lengths_sum),
        has_weight(has_weight),
        cache_flushed(cache_flushed),
        prefetch(prefetch) {}

  // Overload the equal operator (==) to compare equality of two BenchmarkSpec
  // objects by comparing equality of each member
  bool operator==(const BenchmarkSpec& that) const {
    return bit_rate == that.bit_rate && batch_size == that.batch_size &&
        num_rows == that.num_rows && emb_dims == that.emb_dims &&
        avg_length == that.avg_length && indices_bits == that.indices_bits &&
        lengths_sum == that.lengths_sum && has_weight == that.has_weight &&
        cache_flushed == that.cache_flushed && prefetch == that.prefetch;
  }
};

struct BenchmarkResult {
  float ref_bw;
  float ref_eff_bw;
  float ref_time;
  float asmjit_bw;
  float asmjit_eff_bw;
  float asmjit_time;
  float autovec_bw;
  float autovec_eff_bw;
  float autovec_time;

  BenchmarkResult()
      : ref_bw(0.0),
        ref_eff_bw(0.0),
        ref_time(0.0),
        asmjit_bw(0.0),
        asmjit_eff_bw(0.0),
        asmjit_time(0.0),
        autovec_bw(0.0),
        autovec_eff_bw(0.0),
        autovec_time(0.0) {}

  void set_ref_result(float bw, float eff_bw, float time) {
    ref_bw = bw;
    ref_eff_bw = eff_bw;
    ref_time = time;
  }
  void set_asmjit_result(float bw, float eff_bw, float time) {
    asmjit_bw = bw;
    asmjit_eff_bw = eff_bw;
    asmjit_time = time;
  }
  void set_autovec_result(float bw, float eff_bw, float time) {
    autovec_bw = bw;
    autovec_eff_bw = eff_bw;
    autovec_time = time;
  }
};

} // namespace

static std::vector<std::pair<BenchmarkSpec, BenchmarkResult>> benchmarks;

// Return the reference to the BenchmarkResult associated with the
// BenchmarkSpec being queried. If the benchmark spec is recorded,
// return reference to the benchmark result object on the record;
// if the spec is not found, create a new record of the spec and a
// blank benchmark result object.
static BenchmarkResult& find_benchmark_record(const BenchmarkSpec& spec) {
  for (int i = benchmarks.size() - 1; i >= 0; --i) {
    if (benchmarks[i].first == spec) {
      return benchmarks[i].second;
    }
  }
  benchmarks.push_back(std::make_pair(spec, BenchmarkResult()));
  return benchmarks.back().second;
}

static void print_benchmark_results() {
  std::cout
      << "bit_rate, batch_size, num_rows, emb_dim, avg_length, "
      << "indices_bits, lengths_sum, has_weight, cache_flushed, prefetch, "
      << "asmjit b/w (GB/s), asmjit effective b/w (GB/s), asmjit time, "
      << "autovec b/w (GB/s), autovec effective b/w (GB/s), autovec time, "
      << "ref b/w (GB/s), ref effective b/w (GB/s), ref time, "
      << "asmjit speedup ratio, autovec speedup ratio" << std::endl;
  for (size_t i = 0; i < benchmarks.size(); ++i) {
    BenchmarkSpec& spec = benchmarks[i].first;
    BenchmarkResult& res = benchmarks[i].second;
    float asmjit_speedup = res.ref_bw > 0.0 ? res.asmjit_bw / res.ref_bw : 0;
    float autovec_speedup = res.ref_bw > 0.0 ? res.autovec_bw / res.ref_bw : 0;
    std::cout << spec.bit_rate << ", " << spec.batch_size << ", "
              << spec.num_rows << ", " << spec.emb_dims << ", "
              << spec.avg_length << ", " << spec.indices_bits << ", "
              << spec.lengths_sum << ", " << spec.has_weight << ", "
              << spec.cache_flushed << ", " << spec.prefetch << ", "
              << res.asmjit_bw << ", " << res.asmjit_eff_bw << ", "
              << res.asmjit_time << ", " << res.autovec_bw << ", "
              << res.autovec_eff_bw << ", " << res.autovec_time << ", "
              << res.ref_bw << ", " << res.ref_eff_bw << ", " << res.ref_time
              << ", " << asmjit_speedup << ", " << autovec_speedup << std::endl;
  }
}

void print_fused_table(int rows, int embedding_dim, const uint8_t* table) {
  for (int i = 0; i < rows; i++) {
    std::cout << "row: " << i << " : " << std::endl;
    for (int ii = 0; ii < embedding_dim; ii++) {
      std::cout << (int)table[i * (embedding_dim + 2 * sizeof(float)) + ii]
                << ",";
    }
    std::cout << std::endl;
  }
}

static vector<vector<int>> GetInputs_() {
  vector<vector<int>> input_dims = {
      // batch size, number of rows of table, emb dim , avg lengthl
      // TODO: Add more inputs
      // Use these -- but they are slow.
      {10, 4000000, 32, 100},
      {10, 4000000, 64, 100},
      {10, 4000000, 128, 100},
      {10, 4000000, 256, 100},
      // Use these for debugging
      // {2, 16, 128, 10},
      // {10, 4000, 128, 100},
      // {10, 4000, 128, 100},
      // {10, 4000, 128, 100},
  };
  return input_dims;
}

int run_benchmark(
    int bit_rate,
    int batch_size,
    int num_rows,
    int embedding_dim,
    int average_len,
    bool normalize_by_lengths,
    bool use_32_bit_indices = false,
    bool prefetch = false,
    enum KernelType kern_type = REF) {
  // Create embedding table
  int num_elem_per_byte = 8 / bit_rate;
  int fused_embedding_dim =
      (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte +
      2 * sizeof(float16);
  default_random_engine generator;
  normal_distribution<float> embedding_distribution;

  vector<uint8_t> fused_embedding_table(num_rows * fused_embedding_dim);
  for (int i = 0; i < num_rows; i++) {
    for (int ii = 0;
         ii < (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte;
         ii++) {
      fused_embedding_table[i * fused_embedding_dim + ii] = 2;
    }
    float16* scale_bias = reinterpret_cast<float16*>(
        &fused_embedding_table[i * fused_embedding_dim] +
        (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte);
    float scale = 2.0f;
    float bias = 1.0f;
    FloatToFloat16_ref(&scale, scale_bias, 1, true /* clip */);
    FloatToFloat16_ref(&bias, scale_bias + 1, 1, true /* clip */);
  }

  // Generate lengths
  uniform_int_distribution<int> length_distribution(
      1, std::min(2 * average_len + 1, num_rows));
  vector<int> offsets(batch_size + 1);
  offsets[0] = 0;
  for (int i = 0; i < batch_size; ++i) {
    offsets[i + 1] = offsets[i] + length_distribution(generator);
  }

  // Compute the number of indices
  int lengths_sum = offsets[batch_size];

  // Generate indices
  vector<int64_t> indices;
  vector<int32_t> indices_32;

  vector<int> container(num_rows);
  map<int64_t, set<int>> dedup_map; // index -> set(output index)

  // please note we generate unique indices
  for (int i = 0; i < batch_size; ++i) {
    iota(container.begin(), container.end(), 0);
    shuffle(container.begin(), container.end(), generator);
    copy(
        container.begin(),
        container.begin() + (offsets[i + 1] - offsets[i]),
        back_inserter(indices));
  }
  copy(begin(indices), end(indices), back_inserter(indices_32));

  // Generate weights
  vector<float> weights(lengths_sum);
  for (int i = 0; i < lengths_sum; ++i) {
    weights[i] = embedding_distribution(generator);
  }

  vector<float> output_sls(batch_size * embedding_dim);
  vector<float> output_slws(output_sls.size());

  constexpr int NUM_WARMUP = 10;
  constexpr int NUM_ITER = 100;
  // Only counts the number of bytes for reading embedding table and ignore
  // others. Should be good enough as long as embdding_dim is big enough.
  double bytes = lengths_sum * fused_embedding_dim;
  constexpr int CACHE_LINE_LEN = 64;
  double bytes_padded = lengths_sum * CACHE_LINE_LEN *
      static_cast<int>((fused_embedding_dim + CACHE_LINE_LEN - 1) /
                       CACHE_LINE_LEN);

  for (bool has_weight : {false, true}) {
    bool success = false;

    auto kernel_32 = GenerateEmbeddingSpMDMNBit<int32_t>(
        bit_rate,
        embedding_dim,
        has_weight,
        normalize_by_lengths,
        prefetch ? 16 : 0);
    auto kernel_64 = GenerateEmbeddingSpMDMNBit<int64_t>(
        bit_rate,
        embedding_dim,
        has_weight,
        normalize_by_lengths,
        prefetch ? 16 : 0);

    vector<float>& output = has_weight ? output_slws : output_sls;
    for (bool flush_cache : {false, true}) {
      BenchmarkSpec spec(
          bit_rate,
          batch_size,
          num_rows,
          embedding_dim,
          average_len,
          use_32_bit_indices ? 32 : 64,
          lengths_sum,
          has_weight,
          flush_cache,
          prefetch);
      if (kern_type == REF) {
        // Reference implementation
        double t_ref = measureWithWarmup(
            [&]() {
              if (use_32_bit_indices) {
                success = EmbeddingSpMDMNBit_ref(
                    bit_rate,
                    embedding_dim,
                    batch_size,
                    lengths_sum,
                    num_rows,
                    fused_embedding_table.data(),
                    indices_32.data(),
                    offsets.data(),
                    has_weight ? weights.data() : nullptr,
                    normalize_by_lengths,
                    output.data());
              } else {
                success = EmbeddingSpMDMNBit_ref(
                    bit_rate,
                    embedding_dim,
                    batch_size,
                    lengths_sum,
                    num_rows,
                    fused_embedding_table.data(),
                    indices.data(),
                    offsets.data(),
                    has_weight ? weights.data() : nullptr,
                    normalize_by_lengths,
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
        find_benchmark_record(spec).set_ref_result(
            bytes / 1e9 / t_ref, bytes_padded / 1e9 / t_ref, t_ref);
      } else if (kern_type == AUTOVEC) {
        // Auto-vectorization implementation
        double t_autovec = measureWithWarmup(
            [&]() {
              if (use_32_bit_indices) {
                success = EmbeddingSpMDMNBit_autovec(
                    bit_rate,
                    embedding_dim,
                    batch_size,
                    lengths_sum,
                    num_rows,
                    fused_embedding_table.data(),
                    indices_32.data(),
                    offsets.data(),
                    has_weight ? weights.data() : nullptr,
                    normalize_by_lengths,
                    output.data());
              } else {
                success = EmbeddingSpMDMNBit_autovec(
                    bit_rate,
                    embedding_dim,
                    batch_size,
                    lengths_sum,
                    num_rows,
                    fused_embedding_table.data(),
                    indices.data(),
                    offsets.data(),
                    has_weight ? weights.data() : nullptr,
                    normalize_by_lengths,
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
        find_benchmark_record(spec).set_autovec_result(
            bytes / 1e9 / t_autovec, bytes_padded / 1e9 / t_autovec, t_autovec);
      } else if (kern_type == ASMJIT) {
        // Hand-written AVX2/AVX512 implementation
        double t = measureWithWarmup(
            [&]() {
              if (use_32_bit_indices) {
                success = kernel_32(
                    batch_size,
                    lengths_sum,
                    num_rows,
                    fused_embedding_table.data(),
                    indices_32.data(),
                    offsets.data(),
                    has_weight ? weights.data() : nullptr,
                    output.data());
              } else {
                success = kernel_64(
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
        find_benchmark_record(spec).set_asmjit_result(
            bytes / 1e9 / t, bytes_padded / 1e9 / t, t);
      } else {
        std::cerr << "Bad kern_type parameter: " << kern_type << std::endl;
        assert(false);
      }
      if (!success) {
        assert(false && "ERROR: benchmark did not succeed");
      }
    } // flush_cache
  } // has_weight
  return 0;
}

void sweep_benchmark(KernelType kern_type) {
  int batch_size;
  int num_rows;
  int embedding_dim;
  int average_len;

  vector<vector<int>> inputs(GetInputs_());

  for (int bit_rate : {4, 2}) {
    for (auto& input : inputs) {
      assert(input.size() > 3);
      batch_size = input[0];
      num_rows = input[1];
      embedding_dim = input[2];
      average_len = input[3];

      auto run_benchmark_with_above_shape = [&](bool use_32_bit_indices,
                                                bool prefetch) {
        run_benchmark(
            bit_rate,
            batch_size,
            num_rows,
            embedding_dim,
            average_len,
            false, // normalize_by_lengths
            use_32_bit_indices,
            prefetch,
            kern_type);
      };

      // 64 bit indices
      run_benchmark_with_above_shape(false, false);

      // 64 bit indices with prefetching
      run_benchmark_with_above_shape(false, true);

      // 32 bit indices
      run_benchmark_with_above_shape(true, false);

      // 32 bit indices with prefetching
      run_benchmark_with_above_shape(true, true);
    }
  }
}

int main() {
  sweep_benchmark(REF);
  sweep_benchmark(AUTOVEC);
  sweep_benchmark(ASMJIT);
  print_benchmark_results();
  return 0;
}
