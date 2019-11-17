#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmSpConv.h"

#include <iostream>

using namespace std;
using namespace fbgemm;

int main(int, char**) {
  vector<char> llc(128 * 1024 * 1024);

  vector<vector<unsigned>> shapes = {{128, 128, 28, 28}};

  // C is MxN -> CT is NxM
  // A is MxK -> BT is KxM
  // B is KxN -> AT is NxK

  // for (unsigned s = 64; s <= 128; s *= 2)
  for (auto const& s : shapes) {
    int Cout = s[0];
    int Cin = s[1];
    int IY = s[2];
    int IX = s[3];

    for (float fnz = 0.99; fnz >= 0.009999; fnz -= 0.01) {
      auto kData = getRandomSparseVector(3 * 3 * Cout * Cin, fnz);
      auto bData = getRandomSparseVector(Cin * IY * IX);
      auto cData = getRandomSparseVector(Cout * IY * IX);

      auto fn = generateSpConv<float>(Cin, Cout, IY, IX, kData.data());

      double effective_flop = IY * IX * Cin * Cout * 9 * 2;

      auto secs = fbgemm::measureWithWarmup(
          [&]() { fn(bData.data(), cData.data()); }, 5, 10, &llc);

      double effective_gflops = effective_flop / secs / 1e9;
      cout << fnz << "," << effective_gflops << "," << fnz * effective_gflops
           << endl;
    }
  }
}
