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

      double effective_flop = m * n * k * 2;

      auto secs = measureWithWarmup(
          [&]() { fn(bData.data(), cData.data(), 0); }, 5, 10, &llc);

      double effective_gflops = effective_flop / secs / 1e9;
      cout << fnz << "," << effective_gflops << "," << fnz * effective_gflops
           << endl;
    }
  }
}
