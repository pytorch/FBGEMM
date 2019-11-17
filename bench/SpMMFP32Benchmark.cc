#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmSpMM.h"

#include <iostream>

using namespace std;
using namespace fbgemm;

int main(int, char**) {
  vector<char> llc(128 * 1024 * 1024);

  vector<vector<int>> shapes = {{1024, 128, 1024}};

  // C is MxN -> CT is NxM
  // A is MxK -> BT is KxM
  // B is KxN -> AT is NxK

  // for (int s = 64; s <= 128; s *= 2)
  for (auto const& s : shapes) {
    int m = s[0];
    int n = s[1];
    int k = s[2];

    int lda = k;
    int ldb = n;
    int ldc = n;

    for (float fnz = 0.99; fnz >= 0.009999; fnz -= 0.01) {
      auto aData = getRandomSparseVector(m * k, fnz);
      auto bData = getRandomSparseVector(k * n);
      auto cData = getRandomSparseVector(m * n);

      auto fn = generateSpMM<float>(m, n, k, aData.data(), lda, ldb, ldc);
      auto fn_varying_n = generateSpMM<float>(m, k, aData.data(), lda);

      double effective_flop = m * n * k * 2;

      constexpr int NWARMUP = 5;
      constexpr int NITER = 32;
      auto secs = measureWithWarmup(
          [&]() { fn(bData.data(), cData.data(), 0); }, NWARMUP, NITER, &llc);

      auto secs_varying_n = measureWithWarmup(
          [&]() {
            fn_varying_n(
                bData.data(),
                cData.data(),
                n,
                n, /* ldb */
                n, /* ldc */
                0 /* accum_flag */);
          },
          NWARMUP,
          NITER,
          &llc);

      double effective_gflops = effective_flop / secs / 1e9;
      double effective_gflops_varying_n = effective_flop / secs_varying_n / 1e9;
      cout << fnz << "," << effective_gflops << "," << fnz * effective_gflops
           << "," << effective_gflops_varying_n << ","
           << fnz * effective_gflops_varying_n << endl;
    }
  }
}
