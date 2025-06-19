/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "eeg_utils.h"

#include <array>
#include <cmath>
#include <fstream>
#include <vector>

namespace {

// This file contains code that was formerly a part of Julia. License is MIT:
// http://julialang.org/license Code from
// https://github.com/JuliaMath/SpecialFunctions.jl/blob/master/src/gamma.jl
// adapted to c++
/* Helper fn for polygamma(m, z):
Evaluate p[1]*c[1] + x*p[2]*c[2] + x^2*p[3]*c[3] + ...
   where c[1] = m + 1
         c[k] = c[k-1] * (2k+m-1)*(2k+m-2) / ((2k-1)*(2k-2)) = c[k-1] * d[k]
         i.e. d[k] = c[k]/c[k-1] = (2k+m-1)*(2k+m-2) / ((2k-1)*(2k-2))
   by a modified version of Horner's rule:
      c[1] * (p[1] + d[2]*x * (p[2] + d[3]*x * (p[3] + ...))).
The entries of p must be literal constants and there must be > 1 of them. */
template <size_t N>
inline double pgHorner(double x, double m, const std::array<double, N>& p) {
  int k = p.size();
  double ex = ((m + 2 * k - 1)) * (m + 2 * k - 2) *
      (p.back() / ((2 * k - 1) * (2 * k - 2)));
  for (k = p.size() - 1; k >= 2; k--) {
    double cdiv = 1.0 / ((2 * k - 1) * (2 * k - 2));
    ex = (cdiv * (m + 2 * k - 1) * (m + 2 * k - 2)) * (p[k - 1] + x * ex);
  }
  return (m + 1) * (p[0] + x * ex);
}

// Helper: Hurwitz zeta function \zeta(s, q) = \sum_{k=0}^\infty (k+q)^{-s}
double hurwitzZeta(double s, double q) {
  // Special case: q = 1 may be handled by zeta.
  if (q == 1) {
    return std::riemann_zeta(s);
  }

  // Otherwise we follow https://github.com/JuliaLang/julia/issues/7228 for the
  // analytic continuation.
  double z = q;
  double x = q;
  double m = s - 1;
  double zeta = 0;

  double cutoff = 7 + m;
  if (x < cutoff) {
    double xf = std::floor(x);
    int64_t nx = std::llrint(xf);
    int64_t n = std::ceil(cutoff - nx);
    double minus_s = -s;

    if (nx < 0) {
      double minus_z = -z;
      zeta += std::pow(minus_z, minus_s);

      if (xf != z) {
        zeta += std::pow(z - nx, minus_s);
      }

      if (s > 0) {
        for (int64_t nu = -nx - 1; nu >= 1; nu--) {
          double zeta_0 = zeta;
          zeta += std::pow(minus_z - nu, minus_s);
          if (zeta == zeta_0) {
            break;
          }
        }
      } else {
        // Same loop, different order
        for (int64_t nu = 1; nu <= -nx - 1; nu++) {
          double zeta_0 = zeta;
          zeta += std::pow(minus_z - nu, minus_s);
          if (zeta == zeta_0) {
            break;
          }
        }
      }
    } else {
      zeta += std::pow(z, minus_s);
    }

    if (s > 0) {
      for (int64_t nu = std::max(static_cast<int64_t>(1), 1 - nx); nu <= n - 1;
           nu++) {
        double zeta_0 = zeta;
        zeta += std::pow(z + nu, minus_s);
        if (zeta == zeta_0) {
          break;
        }
      }
    } else {
      // Same loop, different order
      for (int64_t nu = n - 1; nu >= std::max(static_cast<int64_t>(1), 1 - nx);
           nu--) {
        double zeta_0 = zeta;
        zeta += std::pow(z + nu, minus_s);
        if (zeta == zeta_0) {
          break;
        }
      }
    }
    z += n;
  }

  double t = 1.0 / z;
  double w = std::pow(t, m);
  zeta += w * (1.0 / m + 0.5 * t);

  t *= t;
  static constexpr std::array<double, 9> polyCoeffs{
      {0.08333333333333333,
       -0.008333333333333333,
       0.003968253968253968,
       -0.004166666666666667,
       0.007575757575757576,
       -0.021092796092796094,
       0.08333333333333333,
       -0.4432598039215686,
       3.0539543302701198}};
  zeta += w * t * pgHorner(t, m, polyCoeffs);
  return zeta;
}

// Helper: direct sum(i=0)^{n-1} 1/(i+q)^s
// This is better than using hurwitzZeta near the pole s = 1.0
// NOTE: functionally correct but not meant to be super accurate, just provided
// for completeness and as fallback path when we lack special function support.
// Use Kahan summation for some accuracy.
double directSumOfHarmonicSeries(double s, double q, int64_t n) {
  double ans = 0.0;
  double compensator = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    double term = std::pow(n - 1 - i + q, -s);
    double y = term - compensator;
    double t = ans + y;
    compensator = (t - ans) - y;
    ans = t;
  }
  return ans;
}

} // namespace

namespace fbgemm_gpu::tbe {

torch::Tensor loadTensorFromFile(const std::filesystem::path& tensorsPath) {
  std::cout << "Loading tensor from " << tensorsPath << std::endl;
  // PyTorch API requires us to use a torch::pickle_load on a vector<char>
  // (torch::load doesn't work here)
  // https://fb.workplace.com/groups/1405155842844877/posts/4947064988653927/?comment_id=4947149218645504
  std::ifstream input(tensorsPath, std::ios::binary);

  std::vector<char> bytes(
      (std::istreambuf_iterator<char>(input)),
      (std::istreambuf_iterator<char>()));

  input.close();
  auto ival = torch::pickle_load(bytes);
  assert((ival.isTensor()) && "Loaded file is not a tensor!");
  return ival.toTensor();
}

void saveTensorToFile(
    const torch::Tensor& t,
    const std::filesystem::path& path) {
  auto pickled = torch::pickle_save(t);
  std::ofstream fout(path, std::ios::out | std::ios::binary);
  fout.write(pickled.data(), pickled.size());
  fout.close();
}

double getZipfianConstant(double s, double q, int64_t n) {
  assert(((n >= 1) || (s > 1.0)) && "For infinite zipfian, s must be > 1.0!");
  if (n < 0) {
    return hurwitzZeta(s, q);
  }
  if (s == 1.0) {
    return directSumOfHarmonicSeries(s, q, n);
    // return gsl_sf_psi_n(0, n + q) - gsl_sf_psi_n(0, q); (keeping for
    // reference)
  }
  static constexpr double kEpsilon = 1e-8;
  if (std::abs(s - 1.0) > kEpsilon) {
    return hurwitzZeta(s, q) - hurwitzZeta(s, q + n);
  }
  // TODO: there is very likely a better way to handle |s - 1.0| < epsilon
  return directSumOfHarmonicSeries(s, q, n);
}

} // namespace fbgemm_gpu::tbe
