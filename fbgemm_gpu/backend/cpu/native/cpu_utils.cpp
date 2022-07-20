/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Intel Corporation.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/cpu_utils.h"

#include <omp.h>
#include "c10/util/Exception.h"

namespace fbgemm_gpu {

namespace {

// histogram size per thread
constexpr int RDX_HIST_SIZE = 256;

template <typename K, typename V>
void radix_sort_kernel(
    K* input_keys,
    V* input_values,
    K* output_keys,
    V* output_values,
    int elements_count,
    int* histogram,
    int* histogram_ps,
    int pass) {
  int tid = omp_get_thread_num();
  int nthreads = omp_get_num_threads();
  int elements_count_4 = elements_count / 4 * 4;

  int* local_histogram = &histogram[RDX_HIST_SIZE * tid];
  int* local_histogram_ps = &histogram_ps[RDX_HIST_SIZE * tid];

  // Step 1: compute histogram
  for (int i = 0; i < RDX_HIST_SIZE; i++) {
    local_histogram[i] = 0;
  }

#pragma omp for schedule(static)
  for (int64_t i = 0; i < elements_count_4; i += 4) {
    K key_1 = input_keys[i];
    K key_2 = input_keys[i + 1];
    K key_3 = input_keys[i + 2];
    K key_4 = input_keys[i + 3];

    local_histogram[(key_1 >> (pass * 8)) & 0xFF]++;
    local_histogram[(key_2 >> (pass * 8)) & 0xFF]++;
    local_histogram[(key_3 >> (pass * 8)) & 0xFF]++;
    local_histogram[(key_4 >> (pass * 8)) & 0xFF]++;
  }
  if (tid == (nthreads - 1)) {
    for (int64_t i = elements_count_4; i < elements_count; ++i) {
      K key = input_keys[i];
      local_histogram[(key >> (pass * 8)) & 0xFF]++;
    }
  }
#pragma omp barrier
  // Step 2: prefix sum
  if (tid == 0) {
    int sum = 0, prev_sum = 0;
    for (int bins = 0; bins < RDX_HIST_SIZE; bins++) {
      for (int t = 0; t < nthreads; t++) {
        sum += histogram[t * RDX_HIST_SIZE + bins];
        histogram_ps[t * RDX_HIST_SIZE + bins] = prev_sum;
        prev_sum = sum;
      }
    }
    histogram_ps[RDX_HIST_SIZE * nthreads] = prev_sum;
    TORCH_CHECK(prev_sum == elements_count);
  }
#pragma omp barrier

  // Step 3: scatter
#pragma omp for schedule(static)
  for (int64_t i = 0; i < elements_count_4; i += 4) {
    K key_1 = input_keys[i];
    K key_2 = input_keys[i + 1];
    K key_3 = input_keys[i + 2];
    K key_4 = input_keys[i + 3];

    int bin_1 = (key_1 >> (pass * 8)) & 0xFF;
    int bin_2 = (key_2 >> (pass * 8)) & 0xFF;
    int bin_3 = (key_3 >> (pass * 8)) & 0xFF;
    int bin_4 = (key_4 >> (pass * 8)) & 0xFF;

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
    for (int64_t i = elements_count_4; i < elements_count; ++i) {
      K key = input_keys[i];
      int pos = local_histogram_ps[(key >> (pass * 8)) & 0xFF]++;
      output_keys[pos] = key;
      output_values[pos] = input_values[i];
    }
  }
}
} // namespace

template <typename K, typename V>
std::pair<K*, V*> radix_sort_parallel(
    K* inp_key_buf,
    V* inp_value_buf,
    K* tmp_key_buf,
    V* tmp_value_buf,
    int64_t elements_count,
    int64_t max_value) {
  int maxthreads = omp_get_max_threads();
  alignas(64) int histogram[RDX_HIST_SIZE * maxthreads];
  alignas(64) int histogram_ps[RDX_HIST_SIZE * maxthreads + 1];
  if (max_value == 0) {
    return std::make_pair(inp_key_buf, inp_value_buf);
  }
  int num_bits = sizeof(K) * 8 - __builtin_clz(max_value);
  unsigned int num_passes = (num_bits + 7) / 8;

#pragma omp parallel
  {
    K* input_keys = inp_key_buf;
    V* input_values = inp_value_buf;
    K* output_keys = tmp_key_buf;
    V* output_values = tmp_value_buf;

    for (unsigned int pass = 0; pass < num_passes; pass++) {
      radix_sort_kernel(
          input_keys,
          input_values,
          output_keys,
          output_values,
          elements_count,
          histogram,
          histogram_ps,
          pass);

      std::swap(input_keys, output_keys);
      std::swap(input_values, output_values);
#pragma omp barrier
    }
  }
  return (
      num_passes % 2 == 0 ? std::make_pair(inp_key_buf, inp_value_buf)
                          : std::make_pair(tmp_key_buf, tmp_value_buf));
}

template std::pair<int*, int*> radix_sort_parallel(
    int* inp_key_buf,
    int* inp_value_buf,
    int* tmp_key_buf,
    int* tmp_value_buf,
    int64_t elements_count,
    int64_t max_value);

template std::pair<int64_t*, int*> radix_sort_parallel(
    int64_t* inp_key_buf,
    int* inp_value_buf,
    int64_t* tmp_key_buf,
    int* tmp_value_buf,
    int64_t elements_count,
    int64_t max_value);

template std::pair<int*, std::pair<int, double>*> radix_sort_parallel(
    int* inp_key_buf,
    std::pair<int, double>* inp_value_buf,
    int* tmp_key_buf,
    std::pair<int, double>* tmp_value_buf,
    int64_t elements_count,
    int64_t max_value);

template std::pair<int*, std::pair<int, float>*> radix_sort_parallel(
    int* inp_key_buf,
    std::pair<int, float>* inp_value_buf,
    int* tmp_key_buf,
    std::pair<int, float>* tmp_value_buf,
    int64_t elements_count,
    int64_t max_value);
} // namespace fbgemm_gpu
