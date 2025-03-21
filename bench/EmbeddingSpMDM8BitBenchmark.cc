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
#include <vector>

#include "./BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

void print_fused_table(int rows, int embedding_dim, const uint8_t* table) {
  for (int i = 0; i < rows; i++) {
    cout << "row: " << i << " : " << endl;
    for (int ii = 0; ii < embedding_dim; ii++) {
      cout << (int)table[i * (embedding_dim + 2 * sizeof(float)) + ii] << ",";
    }
    cout << endl;
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

vector<double> benchmarkTimes;

template <typename OutType>
int run_benchmark(
    int batch_size,
    int num_rows,
    int embedding_dim,
    int average_len,
    bool normalize_by_lengths,
    bool use_32_bit_indices = false,
    bool prefetch = false,
    bool stress_multi_threading = false,
    bool is_bf16_out = false) {
  // Create embedding table
  default_random_engine generator;
  normal_distribution<float> embedding_distribution;

  vector<uint8_t> fused_embedding_table(
      num_rows * (embedding_dim + 2 * sizeof(float)));
  for (int i = 0; i < num_rows; i++) {
    for (int ii = 0; ii < embedding_dim; ii++) {
      fused_embedding_table[i * (embedding_dim + 2 * sizeof(float)) + ii] = 2;
    }
    float* scale_bias = reinterpret_cast<float*>(
        &fused_embedding_table[i * (embedding_dim + 2 * sizeof(float))] +
        embedding_dim);
    scale_bias[0] = 2.0;
    scale_bias[1] = 1.0;
  }

  // print_fused_table(num_rows, embedding_dim, fused_embedding_table);

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
  if (fbgemm_get_thread_num() == 0) {
    cout << "lengths_sum " << lengths_sum << endl;
  }

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

  vector<OutType> output_sls_ref(batch_size * embedding_dim);
  vector<OutType> output_slws_ref(output_sls_ref.size()),
      output_sls(output_sls_ref.size()), output_slws(output_sls_ref.size());

  constexpr int NUM_WARMUP = 10;
  int NUM_ITER = stress_multi_threading ? 1 << 20 : 100;
  double bytes = lengths_sum *
          (embedding_dim * sizeof(uint8_t) + 2 * sizeof(float) +
           (use_32_bit_indices ? 4 : 8)) +
      batch_size * sizeof(int);
  double bytes_padded = lengths_sum *
          ((embedding_dim * sizeof(uint8_t) + 2 * sizeof(float) + 63) / 64 *
               64 +
           (use_32_bit_indices ? 4 : 8)) +
      batch_size * sizeof(int);

  vector<bool> has_weight_options;
  has_weight_options.push_back(false);
  if (!stress_multi_threading) {
    has_weight_options.push_back(true);
  }
  for (bool has_weight : has_weight_options) {
    vector<OutType>& output_ref = has_weight ? output_slws_ref : output_sls_ref;

    bool success = false, success_ref = false;

    if (use_32_bit_indices) {
      success_ref = EmbeddingSpMDM_ref(
          embedding_dim,
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table.data(),
          indices_32.data(),
          offsets.data(),
          has_weight ? weights.data() : nullptr,
          normalize_by_lengths,
          output_ref.data());
    } else {
      success_ref = EmbeddingSpMDM_ref(
          embedding_dim,
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table.data(),
          indices.data(),
          offsets.data(),
          has_weight ? weights.data() : nullptr,
          normalize_by_lengths,
          output_ref.data());
    }

    vector<OutType>& output = has_weight ? output_slws : output_sls;
    vector<bool> flush_cache_options;
    flush_cache_options.push_back(false);
    if (!stress_multi_threading) {
      flush_cache_options.push_back(true);
    }

    auto kernel_32 =
        GenerateEmbeddingSpMDM<uint8_t, int32_t, std::int32_t, OutType>(
            embedding_dim, has_weight, normalize_by_lengths, prefetch ? 16 : 0);
    auto kernel_64 =
        GenerateEmbeddingSpMDM<uint8_t, int64_t, std::int32_t, OutType>(
            embedding_dim, has_weight, normalize_by_lengths, prefetch ? 16 : 0);

#ifdef _OPENMP
#pragma omp barrier
#endif
    for (bool flush_cache : flush_cache_options) {
      benchmarkTimes[fbgemm_get_thread_num()] = measureWithWarmup(
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

      // printMatrix(
      //     matrix_op_t::NoTranspose,
      //     output.data(),
      //     batch_size,
      //     embedding_dim,
      //     embedding_dim,
      //     "");
      // printMatrix(
      //     matrix_op_t::NoTranspose,
      //     output_ref.data(),
      //     batch_size,
      //     embedding_dim,
      //     embedding_dim,
      //     "");
      // Check correctness
      if (!flush_cache) {
        // vector<float>& output_ref =
        //     has_weight ? output_slws_ref : output_sls_ref;
        if (success != success_ref) {
          assert(
              false && "ERROR: refernce impl and JIT imp did not both succeed");
        } else if (success) {
          for (size_t i = 0; i < output.size(); ++i) {
            float tmp1 = 0;
            float tmp2 = 0;
            if (std::is_same<OutType, float>::value) {
              tmp1 = output[i];
              tmp2 = output_ref[i];
            } else if (std::is_same<OutType, uint16_t>::value) {
              if (is_bf16_out) {
                tmp1 = cpu_bf162float(output[i]);
                tmp2 = cpu_bf162float(output_ref[i]);
              } else {
                tmp1 = cpu_half2float(output[i]);
                tmp2 = cpu_half2float(output_ref[i]);
              }
            } else {
              assert(false && "ERROR: unsupported output type");
              cout << "ERROR: unsupported output type" << endl;
            }
            assert(fabs(tmp1 - tmp2) < 1e-3);
            if (fabs(tmp1 - tmp2) >= 1e-3) {
              cout << i << " " << tmp1 << " " << tmp2 << endl;
            }
          }
        }
      }

#ifdef _OPENMP
#pragma omp barrier
#endif
      if (fbgemm_get_thread_num() == 0) {
        if (std::is_same<OutType, float>::value) {
          cout << "out type fp32";
        } else if (std::is_same<OutType, uint16_t>::value) {
          if (is_bf16_out) {
            cout << "out type bf16";
          } else {
            cout << "out type fp16";
          }
        } else {
          assert(false && "ERROR: unsupported output type");
          cout << "ERROR: unsupported output type" << endl;
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

        double max_time = *std::max_element(
            benchmarkTimes.begin(),
            benchmarkTimes.begin() + fbgemm_get_num_threads());
        double avg_time = std::accumulate(
                              benchmarkTimes.begin(),
                              benchmarkTimes.begin() + fbgemm_get_num_threads(),
                              0.0) /
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
  int num_rows;
  int embedding_dim;
  int average_len;

  bool stress_multi_threading = false;

  vector<vector<int>> inputs(GetInputs_());
  benchmarkTimes.resize(fbgemm_get_max_threads());

  for (auto& input : inputs) {
    assert(input.size() > 3);
    batch_size = input[0];
    num_rows = input[1];
    embedding_dim = input[2];
    average_len = input[3];

    cout << "batch size" << setw(6) << batch_size << setw(10) << "num rows"
         << setw(16) << num_rows << setw(10) << "emb dim" << setw(6)
         << embedding_dim << setw(16) << "avg length" << setw(6) << average_len
         << endl;
    // args: batch sz, num rows, emb dim, avg len, normalize, use 32b,
    // prefetch
    cout << "64 bit indices, ";
#ifdef _OPENMP
#pragma omp parallel if (stress_multi_threading)
#endif
#if defined(OUT_TYPE_FLOAT16)
    run_benchmark<float16>(
        batch_size,
        num_rows,
        embedding_dim,
        average_len,
        false,
        false,
        false,
        stress_multi_threading);
#else
    run_benchmark<float>(
        batch_size,
        num_rows,
        embedding_dim,
        average_len,
        false,
        false,
        false,
        stress_multi_threading);
#endif
    if (stress_multi_threading) {
      return 0;
    }

    cout << "64 bit indices with prefetching, ";
#if defined(OUT_TYPE_FLOAT16)
    run_benchmark<float16>(
        batch_size, num_rows, embedding_dim, average_len, false, false, true);
#else
    run_benchmark<float>(
        batch_size, num_rows, embedding_dim, average_len, false, false, true);
#endif
    cout << "32 bit indices, ";
#if defined(OUT_TYPE_FLOAT16)
    run_benchmark<float16>(
        batch_size, num_rows, embedding_dim, average_len, false, true);
#else
    run_benchmark<float>(
        batch_size, num_rows, embedding_dim, average_len, false, true);
#endif

    cout << "32 bit indices with prefetching, ";
#if defined(OUT_TYPE_FLOAT16)
    run_benchmark<float16>(
        batch_size, num_rows, embedding_dim, average_len, false, true, true);
#else
    run_benchmark<float>(
        batch_size, num_rows, embedding_dim, average_len, false, true, true);
#endif

    // running with normalize by lengths
    // run_benchmark(batch_size, num_rows, embedding_dim, average_len,
    // true); run_benchmark(
    //     batch_size, num_rows, embedding_dim, average_len, true,
    //     true);
    // run_benchmark(
    //     batch_size,
    //     num_rows,
    //     embedding_dim,
    //     average_len,
    //     false,
    //     true,
    //     true);
  }
  return 0;
}
