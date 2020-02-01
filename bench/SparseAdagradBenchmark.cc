/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <vector>
#include <iomanip>

#include "./BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

static vector<vector<int>> GetInputs_() {
  vector<vector<int>> input_dims = {
      // num_rows, emb dim
      {1500000, 24},
      {1500000, 32},
      {1500000, 40},
      {1500000, 64},
      {1500000, 80},
      {1500000, 128},
      {1500000, 144},
      {1500000, 192},
      {1500000, 384},
  };
  return input_dims;
}

void run_benchmark(
    const int num_rows, // number of rows reading
    const int block_size, // number of parameters per row
    const std::uint64_t param_size, // total number of parameters
    const bool isIndex64b) {
  vector<char> llc(64L * 1024L * 1024L, 1.0);
  vector<float> g(param_size); // gradients
  vector<float> h(param_size); // input momentums
  vector<float> w(param_size); // input params
  vector<float> h_ref(param_size);
  vector<float> w_ref(param_size);

  default_random_engine generator;
  normal_distribution<float> h_w_distribution;

  // TODO: check appropriate vals for g,h,w
  for (int i = 0; i < g.size(); ++i) {
    g[i] = 4 + i; // h_w_distribution(generator);
  }
  for (int i = 0; i < h.size(); ++i) {
    h_ref[i] = h[i] = 2 + i; // h_w_distribution(generator);
  }
  for (int i = 0; i < w.size(); ++i) {
    w_ref[i] = w[i] = 3 + i; // h_w_distribution(generator);
  }

  vector<std::int64_t> indices(num_rows);
  vector<std::int32_t> indices_32(num_rows);
  float epsilon = 1e-5;
  float lr = 0.5;

  uniform_int_distribution<std::int64_t> length_distribution(0, num_rows - 1);
  for (int i = 0; i < num_rows; ++i) {
    indices_32[i] = indices[i] = length_distribution(generator);
  }

  constexpr int NUM_WARMUP = 4;
  constexpr int NUM_ITER = 10;
  double data_moved = 5 * sizeof(float) * num_rows * block_size;

#define PRE_GENERATE
  double t = 0.0;
  if (isIndex64b) {
#ifdef PRE_GENERATE
    auto fn_indices_64 = GenerateSparseAdaGrad<std::int64_t>(
        block_size);
#endif

    t = measureWithWarmup(
        [&]() {
#ifdef PRE_GENERATE
          fn_indices_64(
              num_rows, // number of rows reading
              param_size, // total number of parameters
              w.data(), // input parameters
              g.data(), // input gradients
              h.data(), // input momentums
              indices.data(), // indices of each row
              epsilon,
              lr);
#else
          fbgemm::SparseAdaGrad(
              num_rows, // number of rows reading
              block_size, // number of parameters per row
              param_size, // total number of parameters
              w.data(), // input parameters
              g.data(), // input gradients
              h.data(), // input momentums
              indices.data(), // indices of each row
              epsilon,
              lr);
#endif
        },
        NUM_WARMUP,
        NUM_ITER,
        [&]() { llc_flush(llc); });

    for (int i = 0; i < NUM_WARMUP + NUM_ITER; ++i) {
      fbgemm::sparse_adagrad_ref(
          num_rows, // number of rows reading
          block_size, // number of parameters per row
          param_size, // total number of parameters
          w_ref.data(), // input parameters
          g.data(), // input gradients
          h_ref.data(), // input momentums
          indices.data(), // indices of each row
          epsilon,
          lr);
    }
  }
  else {
#ifdef PRE_GENERATE
    auto fn_indices_32 = GenerateSparseAdaGrad<std::int32_t>(
        block_size);
#endif

    t = measureWithWarmup(
        [&]() {
#ifdef PRE_GENERATE
          fn_indices_32(
              num_rows, // number of rows reading
              param_size, // total number of parameters
              w.data(), // input parameters
              g.data(), // input gradients
              h.data(), // input momentums
              indices_32.data(), // indices of each row
              epsilon,
              lr);
#else
          fbgemm::SparseAdaGrad(
              num_rows, // number of rows reading
              block_size, // number of parameters per row
              param_size, // total number of parameters
              w.data(), // input parameters
              g.data(), // input gradients
              h.data(), // input momentums
              indices_32.data(), // indices of each row
              epsilon,
              lr);
#endif
        },
        NUM_WARMUP,
        NUM_ITER,
        [&]() { llc_flush(llc); });

    for (int i = 0; i < NUM_WARMUP + NUM_ITER; ++i) {
      fbgemm::sparse_adagrad_ref(
          num_rows, // number of rows reading
          block_size, // number of parameters per row
          param_size, // total number of parameters
          w_ref.data(), // input parameters
          g.data(), // input gradients
          h_ref.data(), // input momentums
          indices_32.data(), // indices of each row
          epsilon,
          lr);
    }
  }

  for (int i = 0; i < w.size(); ++i) {
    assert(fabs(w[i] - w_ref[i]) < 1e-5);
    if (fabs(w[i] - w_ref[i]) >= 1e-5) {
      fprintf(stderr, "%d %f %f\n", i, w[i], w_ref[i]);
    }
  }

  for (int i = 0; i < h.size(); ++i) {
    assert(fabs(h[i] - h_ref[i]) < 1e-5);
    if (fabs(h[i] - h_ref[i]) >= 1e-5) {
      fprintf(stderr, "%d %f %f\n", i, h[i], h_ref[i]);
    }
  }

  std::cout << "indices: " << (isIndex64b? " 64bits ": " 32bits ")
    << " | ";

  std::cout << "num_rows: " << std::setw(8) << num_rows
    << " block_size: " << std::setw(4) << block_size
    << " | ";
  std::cout << "time taken by jit code(secs): " << std::setw(10) << std::fixed
    << std::setprecision(6) << t << " | ";
  std::cout << "bandwidth fbgemm (GB/s) "<< std::setw(10) << std::fixed
    << std::setprecision(6) << data_moved / t / 1e9 << std::endl;
}

int main() {
  int num_rows;
  int block_size;
  std::uint64_t param_size;
  vector<vector<int>> inputs(GetInputs_());

  for (auto isIndex64b: vector<bool> {true, false}) {
    for (auto& input : inputs) {
      assert(input.size() > 2);
      num_rows = input[0];
      block_size = input[1];
      param_size = num_rows * block_size;
      run_benchmark(num_rows, block_size, param_size, isIndex64b);
    }
  }
  return 0;
}
