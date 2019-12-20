#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

namespace {
void llc_flush(std::vector<char>& llc) {
  volatile char* data = llc.data();
  for (int i = 0; i < llc.size(); i++) {
    data[i]++;
  }
}
} // anonymous namespace

static vector<vector<int>> GetInputs_() {
  vector<vector<int>> input_dims = {
      // num_rows, emb dim
      {1500000, 16},
      {1500000, 24},
      {1500000, 32},
      {1500000, 72},
      {1500000, 128},
  };
  return input_dims;
}

void run_benchmark(
    int num_rows, // number of rows reading
    int block_size, // number of parameters per rows
    std::uint64_t param_size) { // total number of parameters
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
    indices[i] = length_distribution(generator);
  }
  copy(begin(indices), end(indices), back_inserter(indices_32));

  chrono::time_point<chrono::system_clock> t_begin, t_end;
  double t = 0.0;
  constexpr int NUM_WARMUP = 4;
  constexpr int NUM_ITER = 10;
  double data_moved = static_cast<double>(NUM_ITER) * num_rows *
      (3 * sizeof(float) * block_size + 2 * 64);

  for (int i = 0; i < NUM_WARMUP + NUM_ITER; ++i) {
    llc_flush(llc);

    t_begin = chrono::system_clock::now();

    fbgemm::SparseAdaGrad(
        num_rows, // number of rows reading
        block_size, // number of parameters per rows
        param_size, // total number of parameters
        w.data(), // input parameters
        g.data(), // input gradients
        h.data(), // input momentums
        indices.data(), // indices of each row
        epsilon,
        lr,
        true);

    t_end = chrono::system_clock::now();

    if (i >= NUM_WARMUP) {
      t += chrono::duration<double>(t_end - t_begin).count();
    }
  }

  std::cout << "num_rows: " << num_rows << " block_size: " << block_size
            << std::endl;
  std::cout << "time taken by jit code(secs): " << t << std::endl;
  std::cout << "bandwidth fbgemm (GB/s) " << data_moved / t / 1e9 << std::endl;
}

int main() {
  int num_rows;
  int block_size;
  std::uint64_t param_size;
  vector<vector<int>> inputs(GetInputs_());

  for (auto& input : inputs) {
    assert(input.size() > 2);
    num_rows = input[0];
    block_size = input[1];
    param_size = num_rows * block_size;
    run_benchmark(num_rows, block_size, param_size);
  }
  return 0;
}
