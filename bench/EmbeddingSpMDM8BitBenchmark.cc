/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <immintrin.h>
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
#include "fbgemm/Utils.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

void print_outupt(int rows, int embedding_dim, const float* output) {
  for (int i = 0; i < rows; i++) {
    std::cout << "output row: " << i << " : " << std::endl;
    for (int ii = 0; ii < embedding_dim; ii++) {
      std::cout << output[i * embedding_dim + ii] << ",";
    }
    std::cout << std::endl;
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

vector<double> times;

int run_benchmark(
    int batch_size,
    int num_unique_ids,
    int embedding_dim,
    int average_len,
    bool normalize_by_lengths,
    bool use_32_bit_indices = false,
    bool prefetch = false,
    bool stress_multi_threading = false) {
  // Create embedding table
  vector<uint8_t> embedding_table(
      num_unique_ids * (embedding_dim + 2 * sizeof(float)));
  default_random_engine generator;
  normal_distribution<float> embedding_distribution;

  vector<uint8_t> fused_embedding_table(
      num_unique_ids * (embedding_dim + 2 * sizeof(float)));
  for (int i = 0; i < num_unique_ids; i++) {
    for (int ii = 0; ii < embedding_dim; ii++) {
      fused_embedding_table[i * (embedding_dim + 2 * sizeof(float)) + ii] = 2;
    }
    float* scale_bias = reinterpret_cast<float*>(
        &fused_embedding_table[i * (embedding_dim + 2 * sizeof(float))] +
        embedding_dim);
    scale_bias[0] = 2.0;
    scale_bias[1] = 1.0;
  }

  // print_fused_table(num_unique_ids, embedding_dim, fused_embedding_table);

  // Generate lengths
  uniform_int_distribution<int> length_distribution(
      1, std::min(2 * average_len + 1, num_unique_ids));
  vector<int> lengths(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    lengths[i] = length_distribution(generator);
  }

  // Compute the number of indices
  int lengths_sum = accumulate(lengths.begin(), lengths.end(), 0);
  if (fbgemm_get_thread_num() == 0) {
    cout << "lengths_sum " << lengths_sum << endl;
  }

  // Generate indices
  vector<int64_t> indices;
  vector<int32_t> indices_32;

  vector<int> container(num_unique_ids);
  map<int64_t, set<int>> dedup_map; // index -> set(output index)

  // please note we generate unique indices
  for (int i = 0; i < batch_size; ++i) {
    iota(container.begin(), container.end(), 0);
    random_shuffle(container.begin(), container.end());
    copy(
        container.begin(),
        container.begin() + lengths[i],
        back_inserter(indices));
  }
  copy(begin(indices), end(indices), back_inserter(indices_32));

  // Generate weights
  vector<float> weights(lengths_sum);
  for (int i = 0; i < lengths_sum; ++i) {
    weights[i] = embedding_distribution(generator);
  }

  vector<float> output_sls_ref(batch_size * embedding_dim);
  vector<float> output_slws_ref(output_sls_ref.size()),
      output_sls(output_sls_ref.size()), output_slws(output_sls_ref.size());

  constexpr int NUM_WARMUP = 4;
  int NUM_ITER = stress_multi_threading ? 1 << 20 : 10;
  // Only counts the number of bytes for reading embedding table and ignore
  // others. Should be good enough as long as embdding_dim is big enough.
  double bytes =
      lengths_sum * (embedding_dim * sizeof(uint8_t) + 2 * sizeof(float));
  double bytes_padded =
      lengths_sum * 64 *
      static_cast<int>(
          (embedding_dim * sizeof(uint8_t) + 2 * sizeof(float) + 63) / 64);

  vector<bool> has_weight_options;
  has_weight_options.push_back(false);
  if (!stress_multi_threading) {
    has_weight_options.push_back(true);
  }
  for (bool has_weight : has_weight_options) {
    vector<float>& output_ref = has_weight ? output_slws_ref : output_sls_ref;

    if (use_32_bit_indices) {
      fbgemm::EmbeddingSpMDM_ref(
          embedding_dim,
          batch_size,
          lengths_sum,
          num_unique_ids,
          fused_embedding_table.data(),
          indices_32.data(),
          lengths.data(),
          has_weight ? weights.data() : nullptr,
          normalize_by_lengths,
          output_ref.data());

    } else {
      fbgemm::EmbeddingSpMDM_ref(
          embedding_dim,
          batch_size,
          lengths_sum,
          num_unique_ids,
          fused_embedding_table.data(),
          indices.data(),
          lengths.data(),
          has_weight ? weights.data() : nullptr,
          normalize_by_lengths,
          output_ref.data());
    }

    vector<float>& output = has_weight ? output_slws : output_sls;
    vector<bool> flush_cache_options;
    flush_cache_options.push_back(false);
    if (!stress_multi_threading) {
      flush_cache_options.push_back(true);
    }

#define PRE_GENERATE
#ifdef PRE_GENERATE
    auto fn_32 = GenerateEmbeddingSpMDM<uint8_t, int32_t>(
        embedding_dim, has_weight, normalize_by_lengths, prefetch ? 16 : 0);
    auto fn_64 = GenerateEmbeddingSpMDM<uint8_t, int64_t>(
        embedding_dim, has_weight, normalize_by_lengths, prefetch ? 16 : 0);
#endif

#ifdef _OPENMP
#pragma omp barrier
#endif
    for (bool flush_cache : flush_cache_options) {
      times[fbgemm_get_thread_num()] = measureWithWarmup(
          [&]() {
            if (use_32_bit_indices) {
#ifdef PRE_GENERATE
              fn_32(
                  batch_size,
                  lengths_sum,
                  num_unique_ids,
                  fused_embedding_table.data(),
                  indices_32.data(),
                  lengths.data(),
                  has_weight ? weights.data() : nullptr,
                  output.data());
#else
              fbgemm::EmbeddingSpMDM<uint8_t, int32_t>(
                  embedding_dim,
                  batch_size,
                  lengths_sum,
                  num_unique_ids,
                  fused_embedding_table.data(),
                  indices_32.data(),
                  lengths.data(),
                  has_weight ? weights.data() : nullptr,
                  normalize_by_lengths,
                  output.data(),
                  prefetch ? 16 : 0);
#endif
            } else {
#ifdef PRE_GENERATE
              fn_64(
                  batch_size,
                  lengths_sum,
                  num_unique_ids,
                  fused_embedding_table.data(),
                  indices.data(),
                  lengths.data(),
                  has_weight ? weights.data() : nullptr,
                  output.data());
#else
              fbgemm::EmbeddingSpMDM<uint8_t, int64_t>(
                  embedding_dim,
                  batch_size,
                  lengths_sum,
                  num_unique_ids,
                  fused_embedding_table.data(),
                  indices.data(),
                  lengths.data(),
                  has_weight ? weights.data() : nullptr,
                  normalize_by_lengths,
                  output.data(),
                  prefetch ? 16 : 0);
#endif
            }
          },
          NUM_WARMUP,
          NUM_ITER,
          [&]() {
            if (flush_cache) {
              cache_evict(fused_embedding_table);
              cache_evict(indices);
              cache_evict(indices_32);
              cache_evict(lengths);
              cache_evict(weights);
              cache_evict(output);
            }
          });

      // print_outupt(batch_size, embedding_dim, output.data());
      // print_outupt(batch_size, embedding_dim, output_ref.data());
      // Check correctness
      if (!flush_cache) {
        // vector<float>& output_ref =
        //     has_weight ? output_slws_ref : output_sls_ref;
        for (int i = 0; i < output.size(); ++i) {
          assert(fabs(output[i] - output_ref[i]) < 1e-3);
          if (fabs(output[i] - output_ref[i]) >= 1e-3) {
            cout << i << " " << output[i] << " " << output_ref[i] << endl;
          }
        }
      }

#ifdef _OPENMP
#pragma omp barrier
#endif
      if (fbgemm_get_thread_num() == 0) {
        if (has_weight) {
          cout << setw(16) << "SLW(WEIGHTED) ";
        } else {
          cout << setw(16) << "SLS ";
        }
        if (flush_cache) {
          cout << setw(20) << "cache flushed";
        } else {
          cout << setw(20) << "cache not flushed";
        }
        if (prefetch) {
          cout << setw(16) << "prefetch on";
        } else {
          cout << setw(16) << "prefetch off";
        }

        double max_time = *std::max_element(
            times.begin(), times.begin() + fbgemm_get_num_threads());
        double avg_time =
            std::accumulate(
                times.begin(), times.begin() + fbgemm_get_num_threads(), 0.0) /
            fbgemm_get_num_threads();
        double load_imbalance = (max_time - avg_time) / avg_time;

        cout << setw(8) << "b/w" << setw(10) << bytes / 1e9 / max_time
             << " GB/s" << setw(20) << "effective b/w: " << setw(16)
             << bytes_padded / 1e9 / max_time << "GB/s" << setw(8) << " time "
             << setw(16) << max_time << " load_imbalance " << load_imbalance
             << endl;
      }
    } // flush_cache
  } // has_weight
  return 0;
}

int main() {
  int batch_size;
  int num_unique_ids;
  int embedding_dim;
  int average_len;

  bool stress_multi_threading = false;

  vector<vector<int>> inputs(GetInputs_());
  times.resize(fbgemm_get_max_threads());

  for (auto& input : inputs) {
    assert(input.size() > 3);
    batch_size = input[0];
    num_unique_ids = input[1];
    embedding_dim = input[2];
    average_len = input[3];

    cout << "batch size" << setw(6) << batch_size << setw(10) << "num rows"
         << setw(16) << num_unique_ids << setw(10) << "emb dim" << setw(6)
         << embedding_dim << setw(16) << "avg length" << setw(6) << average_len
         << endl;
    // args: batch sz, num rows, emb dim, avg len, normalize, use 32b, prefetch
    cout << "64 bit indices, ";
#ifdef _OPENMP
#pragma omp parallel if (stress_multi_threading)
#endif
    run_benchmark(
        batch_size,
        num_unique_ids,
        embedding_dim,
        average_len,
        false,
        false,
        false,
        stress_multi_threading);

    if (stress_multi_threading) {
      return 0;
    }

    cout << "64 bit indices with prefetching, ";
    run_benchmark(
        batch_size,
        num_unique_ids,
        embedding_dim,
        average_len,
        false,
        false,
        true);

    cout << "32 bit indices, ";
    run_benchmark(
        batch_size, num_unique_ids, embedding_dim, average_len, false, true);

    cout << "32 bit indices with prefetching, ";
    run_benchmark(
        batch_size,
        num_unique_ids,
        embedding_dim,
        average_len,
        false,
        true,
        true);

    // running with normalize by lengths
    // run_benchmark(batch_size, num_unique_ids, embedding_dim, average_len,
    // true); run_benchmark(
    //     batch_size, num_unique_ids, embedding_dim, average_len, true, true);
    // run_benchmark(
    //     batch_size,
    //     num_unique_ids,
    //     embedding_dim,
    //     average_len,
    //     false,
    //     true,
    //     true);
  }
  return 0;
}
