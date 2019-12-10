#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmSpMM.h"
#include "src/RefImplementations.h"

#include <iostream>

using namespace std;
using namespace fbgemm;

int main(int, char**) {
  vector<char> llc(128 * 1024 * 1024);

  // vector<vector<unsigned>> shapes = {{64, 64, 64}};
  // vector<vector<unsigned>> shapes = {{1, 16, 4}};

  vector<vector<unsigned>> shapes = {{1024, 128, 1024}};

  // C is MxN -> CT is NxM
  // A is MxK -> BT is KxM
  // B is KxN -> AT is NxK

  // for (unsigned s = 64; s <= 128; s *= 2)
  for (auto const& s : shapes) {
    int m = s[0];
    int n = s[1];
    int k = s[2];

    int lda = k;
    int ldb = n;
    int ldc = n;

    for (float fnz = 0.99; fnz >= 0.009999; fnz -= 0.01) {
      auto aData = getRandomSparseVector(m * k / 4, fnz);
      auto bData = getRandomSparseVector(k * n / 4);
      auto cData = getRandomSparseVector(m * n);

      auto aptr = reinterpret_cast<const int8_t*>(aData.data());
      auto bptr = reinterpret_cast<uint8_t*>(bData.data());

      for (int i = 0; i < k * n; ++i) {
        bptr[i] &= 0x7F;
      }

      auto cptr = reinterpret_cast<int32_t*>(cData.data());

      auto fn = generateSpMM<int32_t>(m, n, k, aptr, lda, ldb, ldc);
      auto fn_varying_n = generateSpMM<int32_t>(m, k, aptr, lda);

      double FLOPs = m * n * k * 2;

      constexpr int NWARMUP = 5;
      constexpr int NITER = 32;
      auto secs = measureWithWarmup(
          [&]() {
            fn(bptr, cptr, 0);
          },
          NWARMUP,
          NITER,
          [&]() {
            llc_flush(llc);
          });

      auto secs_varying_n = measureWithWarmup(
          [&]() {
            fn_varying_n(
                bptr, cptr, n, n /* ldb */, n /* ldc */, 0 /* accum_flag */);
          },
          NWARMUP,
          NITER,
          [&]() {
            llc_flush(llc);
          });

      cout << fnz << "," << (FLOPs / secs / 1e9) << ","
           << (fnz * FLOPs / secs / 1e9) << ","
           << (FLOPs / secs_varying_n / 1e9) << ","
           << (fnz * FLOPs / secs_varying_n / 1e9) << endl;
    }
  }
}
