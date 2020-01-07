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

namespace {
void print_table(int rows, int embedding_dim, const float* output) {
  for (int i = 0; i < rows; i++) {
    std::cout << "row: " << i << " : " << std::endl;
    for (int ii = 0; ii < embedding_dim; ii++) {
      std::cout << output[i * embedding_dim + ii] << ",";
    }
    std::cout << std::endl;
  }
}
} // anonymous namespace

static vector<vector<int>> GetInputs_() {
  vector<vector<int>> input_dims = {
      // batch size, number of rows of table, emb dim , avg lengthl
      // TODO: Add more inputs
      // Use these -- but they are slow.
      //{100, 4000000, 32, 100},
      // {10, 4000000, 64, 100},
      // {10, 4000000, 128, 100},
      // {10, 4000000, 256, 100},
      // Use these for debugging
      {2, 16, 128, 10},
      {10, 4000, 128, 100},
      {10, 4000, 128, 100},
      {10, 4000, 128, 100},
  };
  return input_dims;
}

int run_benchmark(
    int batch_size,
    int num_unique_ids,
    int embedding_dim,
    int average_len,
    bool normalize_by_lengths,
    bool use_32_bit_indices = false,
    bool prefetch = false) {
  // Create embedding table
  default_random_engine generator;

  vector<float> embedding_table(num_unique_ids * embedding_dim);
  normal_distribution<float> embedding_distribution;
  for (int i = 0; i < embedding_table.size(); ++i) {
    embedding_table[i] = embedding_distribution(generator);
  }

  // Generate lengths
  uniform_int_distribution<int> length_distribution(
      1, std::min(2 * average_len + 1, num_unique_ids));
  vector<int> lengths(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    lengths[i] = length_distribution(generator);
  }

  // Compute the number of indices
  int lengths_sum = accumulate(lengths.begin(), lengths.end(), 0);
  cout << "lengths_sum " << lengths_sum << endl;

  // Generate indices
  vector<int64_t> indices;
  vector<int32_t> indices_32;

  vector<int> container(num_unique_ids);

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
  constexpr int NUM_ITER = 10;
  // Only counts the number of bytes for reading embedding table and ignore
  // others. Should be good enough as long as embdding_dim is big enough.
  double bytes =
      lengths_sum * (embedding_dim * sizeof(uint8_t) + 2 * sizeof(float));
  double bytes_padded =
      lengths_sum * 64 *
      static_cast<int>(
          (embedding_dim * sizeof(uint8_t) + 2 * sizeof(float) + 63) / 64);

  for (bool has_weight : {false, true}) {
    vector<float>& output_ref = has_weight ? output_slws_ref : output_sls_ref;

    bool success, success_ref;

    for (int i = 0; i < NUM_WARMUP + NUM_ITER; ++i) {
      if (use_32_bit_indices) {
        success_ref = fbgemm::EmbeddingSpMDM_ref(
            embedding_dim,
            batch_size,
            lengths_sum,
            num_unique_ids,
            embedding_table.data(),
            indices_32.data(),
            lengths.data(),
            has_weight ? weights.data() : nullptr,
            normalize_by_lengths,
            output_ref.data());
      } else {
        success_ref = fbgemm::EmbeddingSpMDM_ref(
            embedding_dim,
            batch_size,
            lengths_sum,
            num_unique_ids,
            embedding_table.data(),
            indices.data(),
            lengths.data(),
            has_weight ? weights.data() : nullptr,
            normalize_by_lengths,
            output_ref.data());
      }
    }

    vector<float>& output = has_weight ? output_slws : output_sls;
    for (bool flush_cache : {false, true}) {
      double t = measureWithWarmup(
          [&]() {
            if (use_32_bit_indices) {
              success = fbgemm::EmbeddingSpMDM<float, int32_t>(
                  embedding_dim,
                  batch_size,
                  lengths_sum,
                  num_unique_ids,
                  embedding_table.data(),
                  indices_32.data(),
                  lengths.data(),
                  has_weight ? weights.data() : nullptr,
                  normalize_by_lengths,
                  output.data(),
                  prefetch ? 16 : 0);
            } else {
              success = fbgemm::EmbeddingSpMDM<float, int64_t>(
                  embedding_dim,
                  batch_size,
                  lengths_sum,
                  num_unique_ids,
                  embedding_table.data(),
                  indices.data(),
                  lengths.data(),
                  has_weight ? weights.data() : nullptr,
                  normalize_by_lengths,
                  output.data(),
                  prefetch ? 16 : 0);
            }
          },
          NUM_WARMUP,
          NUM_ITER,
          [&]() {
            if (flush_cache) {
              cache_evict(embedding_table);
              cache_evict(indices);
              cache_evict(indices_32);
              cache_evict(lengths);
              cache_evict(weights);
              cache_evict(output);
            }
          });

      // print_table(batch_size, embedding_dim, output.data());
      // cout << "reference data\n";
      // print_table(batch_size, embedding_dim, output_ref.data());
      // Check correctness
      if (!flush_cache) {
        if (success != success_ref) {
          assert(0 && "ERROR: refernce impl and JIt imp did not both succeed");
        } else if (success == true) {
          for (int i = 0; i < output.size(); ++i) {
            assert(output[i] == output_ref[i]);
            if (output[i] != output_ref[i]) {
              cout << i << " " << output[i] << " " << output_ref[i] << endl;
            }
          }
        }
      }

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

      cout << setw(8) << "b/w" << setw(10) << bytes / 1e9 / t << " GB/s"
           << setw(20) << "effective b/w: " << setw(16)
           << bytes_padded / 1e9 / t << "GB/s" << setw(8) << " time "
           << setw(16) << t << endl;
    } // flush_cache
  } // has_weight
  return 0;
}

int main() {
  int batch_size;
  int num_unique_ids;
  int embedding_dim;
  int average_len;

  vector<vector<int>> inputs(GetInputs_());

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
    run_benchmark(
        batch_size, num_unique_ids, embedding_dim, average_len, false);

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
    run_benchmark(batch_size, num_unique_ids, embedding_dim, average_len, true);
    run_benchmark(
        batch_size, num_unique_ids, embedding_dim, average_len, true, true);
    run_benchmark(
        batch_size,
        num_unique_ids,
        embedding_dim,
        average_len,
        false,
        true,
        true);
  }
  return 0;
}
