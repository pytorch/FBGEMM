/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Copyright (c) Intel Corporation.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <omp.h>
#include <cstdint>
#include <utility>

// histogram size per thread
constexpr int RDX_HIST_SIZE = 256;
template <typename T>
std::pair<T*, T*> radix_sort_parallel(
    T* inp_key_buf,
    T* inp_value_buf,
    T* tmp_key_buf,
    T* tmp_value_buf,
    int64_t elements_count,
    int64_t max_value) {
  int maxthreads = omp_get_max_threads();
  alignas(64) int histogram[RDX_HIST_SIZE * maxthreads],
      histogram_ps[RDX_HIST_SIZE * maxthreads + 1];
  if (max_value == 0) {
    return std::make_pair(inp_key_buf, inp_value_buf);
  }
  int num_bits = sizeof(T) * 8 - __builtin_clz(max_value);
  unsigned int num_passes = (num_bits + 7) / 8;

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int* local_histogram = &histogram[RDX_HIST_SIZE * tid];
    int* local_histogram_ps = &histogram_ps[RDX_HIST_SIZE * tid];
    int elements_count_4 = elements_count / 4 * 4;
    T* input_keys = inp_key_buf;
    T* input_values = inp_value_buf;
    T* output_keys = tmp_key_buf;
    T* output_values = tmp_value_buf;

    for (unsigned int pass = 0; pass < num_passes; pass++) {
      // Step 1: compute histogram
      for (int i = 0; i < RDX_HIST_SIZE; i++)
        local_histogram[i] = 0;

#pragma omp for schedule(static)
      for (int64_t i = 0; i < elements_count_4; i += 4) {
        T key_1 = input_keys[i];
        T key_2 = input_keys[i + 1];
        T key_3 = input_keys[i + 2];
        T key_4 = input_keys[i + 3];

        local_histogram[(key_1 >> (pass * 8)) & 0xFF]++;
        local_histogram[(key_2 >> (pass * 8)) & 0xFF]++;
        local_histogram[(key_3 >> (pass * 8)) & 0xFF]++;
        local_histogram[(key_4 >> (pass * 8)) & 0xFF]++;
      }
      if (tid == (nthreads - 1)) {
        for (int64_t i = elements_count_4; i < elements_count; i++) {
          T key = input_keys[i];
          local_histogram[(key >> (pass * 8)) & 0xFF]++;
        }
      }
#pragma omp barrier
      // Step 2: prefix sum
      if (tid == 0) {
        int sum = 0, prev_sum = 0;
        for (int bins = 0; bins < RDX_HIST_SIZE; bins++)
          for (int t = 0; t < nthreads; t++) {
            sum += histogram[t * RDX_HIST_SIZE + bins];
            histogram_ps[t * RDX_HIST_SIZE + bins] = prev_sum;
            prev_sum = sum;
          }
        histogram_ps[RDX_HIST_SIZE * nthreads] = prev_sum;
        if (prev_sum != elements_count) {
        }
      }
#pragma omp barrier

      // Step 3: scatter
#pragma omp for schedule(static)
      for (int64_t i = 0; i < elements_count_4; i += 4) {
        T key_1 = input_keys[i];
        T key_2 = input_keys[i + 1];
        T key_3 = input_keys[i + 2];
        T key_4 = input_keys[i + 3];
        T bin_1 = (key_1 >> (pass * 8)) & 0xFF;
        T bin_2 = (key_2 >> (pass * 8)) & 0xFF;
        T bin_3 = (key_3 >> (pass * 8)) & 0xFF;
        T bin_4 = (key_4 >> (pass * 8)) & 0xFF;
        int pos;
        pos = local_histogram_ps[bin_1]++;
        output_keys[pos] = key_1;
        output_values[pos] = input_values[i];
        pos = local_histogram_ps[bin_2]++;
        output_keys[pos] = key_2;
        output_values[pos] = input_values[i + 1];
        pos = local_histogram_ps[bin_3]++;
        output_keys[pos] = key_3;
        output_values[pos] = input_values[i + 2];
        pos = local_histogram_ps[bin_4]++;
        output_keys[pos] = key_4;
        output_values[pos] = input_values[i + 3];
      }
      if (tid == (nthreads - 1)) {
        for (int64_t i = elements_count_4; i < elements_count; i++) {
          T key = input_keys[i];
          int pos = local_histogram_ps[(key >> (pass * 8)) & 0xFF]++;
          output_keys[pos] = key;
          output_values[pos] = input_values[i];
        }
      }

      T* temp = input_keys;
      input_keys = output_keys;
      output_keys = temp;

      temp = input_values;
      input_values = output_values;
      output_values = temp;
#pragma omp barrier
    }
  }
  return (
      num_passes % 2 == 0 ? std::make_pair(inp_key_buf, inp_value_buf)
                          : std::make_pair(tmp_key_buf, tmp_value_buf));
}
