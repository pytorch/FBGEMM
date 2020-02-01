/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <chrono>
#include <cmath>
#include <memory>
#include <random>

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_BLAS
#if __APPLE__
// not sure whether need to differentiate TARGET_OS_MAC or TARGET_OS_IPHONE,
// etc.
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "./AlignedVec.h"
#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmFP16.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

#if defined(USE_MKL)
void test_xerbla(char* srname, const int* info, int) {
  // srname - name of the function that called xerbla
  // info - position of the invalid parameter in the parameter list
  // len - length of the name in bytes
  printf("\nXERBLA(MKL Error) is called :%s: %d\n", srname, *info);
}
#endif

void performance_test(
    int num_instances,
    bool flush,
    int repetitions,
    bool is_mkl) {
#if defined(USE_MKL)
  mkl_set_xerbla((XerblaEntry)test_xerbla);
#endif

  float alpha = 1.f, beta = 1.f;
  matrix_op_t btran = matrix_op_t::Transpose;

  using btype = float16;

#define dataset 1

#if dataset == 1
  const int NITER = (flush) ? 10 : 100;
  std::vector<std::vector<int>> shapes;
  for (auto m = 1; m < 120; m++) {
    // shapes.push_back({m, 128, 512});
    shapes.push_back({m, 512, 512});
  }

#elif dataset == 2
  const int NITER = (flush) ? 10 : 100;
#include "shapes_dataset.h"

#else
  flush = false;
  constexpr int NITER = 1;
  std::vector<std::vector<int>> shapes;
  std::random_device r;
  std::default_random_engine generator(r());
  std::uniform_int_distribution<int> dm(1, 100);
  std::uniform_int_distribution<int> dnk(1, 1024);
  for (int i = 0; i < 1000; i++) {
    int m = dm(generator);
    int n = dnk(generator);
    int k = dnk(generator);
    shapes.push_back({m, n, k});
  }
#endif

  std::string type;
  double gflops, gbs, ttot;
  for (auto s : shapes) {
    int m = s[0];
    int n = s[1];
    int k = s[2];

    // initialize with small numbers
    aligned_vector<int> Aint(m * k);
    randFill(Aint, 0, 4);
    vector<aligned_vector<float>> A;
    for (int i = 0; i < num_instances; ++i) {
      A.push_back(aligned_vector<float>(Aint.begin(), Aint.end()));
    }

    aligned_vector<int> Bint(k * n);
    randFill(Bint, 0, 4);
    aligned_vector<float> B(Bint.begin(), Bint.end());

    vector<unique_ptr<PackedGemmMatrixFP16>> Bp;
    for (int i = 0; i < num_instances; ++i) {
      Bp.push_back(unique_ptr<PackedGemmMatrixFP16>(
          new PackedGemmMatrixFP16(btran, k, n, alpha, B.data())));
    }

    auto kAligned = ((k * sizeof(float) + 64) & ~63) / sizeof(float);
    auto nAligned = ((n * sizeof(float) + 64) & ~63) / sizeof(float);
    vector<aligned_vector<float>> Bt(num_instances);
    auto& Bt_ref = Bt[0];

    if (btran == matrix_op_t::Transpose) {
      Bt_ref.resize(k * nAligned);
      for (auto row = 0; row < k; ++row) {
        for (auto col = 0; col < n; ++col) {
          Bt_ref[row * nAligned + col] = alpha * B[col * k + row];
        }
      }
    } else {
      Bt_ref.resize(kAligned * n);
      for (auto row = 0; row < k; ++row) {
        for (auto col = 0; col < n; ++col) {
          Bt_ref[col * kAligned + row] = alpha * B[col * k + row];
        }
      }
    }

    for (auto i = 1; i < num_instances; ++i) {
      Bt[i] = Bt_ref;
    }

    vector<aligned_vector<float>> C_ref;
    vector<aligned_vector<float>> C_fb;
    if (beta != 0.0f) {
      aligned_vector<int> Cint(m * n);
      randFill(Cint, 0, 4);
      for (int i = 0; i < num_instances; ++i) {
        C_ref.push_back(aligned_vector<float>(Cint.begin(), Cint.end()));
        C_fb.push_back(aligned_vector<float>(Cint.begin(), Cint.end()));
      }
    } else {
      for (int i = 0; i < num_instances; ++i) {
        C_ref.push_back(aligned_vector<float>(m * n, 1.f));
        C_fb.push_back(aligned_vector<float>(m * n, NAN));
      }
    }

    double nflops = 2.0 * m * n * k;
    double nbytes = 4.0 * m * k + 2.0 * k * n + 4.0 * m * n;

    // warm up MKL and fbgemm
    // check correctness at the same time
    for (auto w = 0; w < 3; w++) {
#if defined(USE_MKL) || defined(USE_BLAS)
      cblas_sgemm(
          CblasRowMajor,
          CblasNoTrans,
          CblasNoTrans, // B is pretransposed, if required by operation
          m,
          n,
          k,
          1.0, // Mutliplication by Alpha is done during transpose of B
          A[0].data(),
          k,
          Bt[0].data(),
          btran == matrix_op_t::NoTranspose ? kAligned : nAligned,
          beta,
          C_ref[0].data(),
          n);
#else
      cblas_sgemm_ref(
          matrix_op_t::NoTranspose,
          matrix_op_t::NoTranspose,
          m,
          n,
          k,
          1.0,
          A[0].data(),
          k,
          Bt[0].data(),
          (btran == matrix_op_t::NoTranspose) ? kAligned : nAligned,
          beta,
          C_ref[0].data(),
          n);
#endif
#ifdef _OPENMP
#pragma omp parallel if (num_instances == 1)
#endif
      {
        int num_threads = num_instances == 1 ? fbgemm_get_num_threads() : 1;
        int tid = num_instances == 1 ? fbgemm_get_thread_num() : 0;
        cblas_gemm_compute(
            matrix_op_t::NoTranspose,
            m,
            A[0].data(),
            *Bp[0],
            beta,
            C_fb[0].data(),
            tid,
            num_threads);
      }

#if defined(USE_MKL) || defined(USE_BLAS)
      // Compare results
      for (auto i = 0; i < C_ref[0].size(); i++) {
        if (std::abs(C_ref[0][i] - C_fb[0][i]) > 1e-3) {
          fprintf(
              stderr,
              "Error: too high diff between fp32 ref %f and fp16 %f at %d\n",
              C_ref[0][i],
              C_fb[0][i],
              i);
          return;
        }
      }
#endif
    }

#if defined(USE_MKL)
    if (is_mkl) {
      // Gold via MKL sgemm
      type = "MKL_FP32";
#elif defined(USE_BLAS)
    type = "BLAS_FP32";
#else
    type = "REF_FP32";
#endif

      ttot = measureWithWarmup(
          [&]() {
            int copy = num_instances == 1 ? 0 : fbgemm_get_thread_num();
            for (int i = 0; i < repetitions; ++i) {
#if defined(USE_MKL) || defined(USE_BLAS)
              cblas_sgemm(
                  CblasRowMajor,
                  CblasNoTrans,
                  CblasNoTrans,
                  m,
                  n,
                  k,
                  1.0,
                  A[copy].data(),
                  k,
                  Bt[copy].data(),
                  btran == matrix_op_t::NoTranspose ? kAligned : nAligned,
                  beta,
                  C_ref[copy].data(),
                  n);
#else
            cblas_sgemm_ref(
                matrix_op_t::NoTranspose,
                matrix_op_t::NoTranspose,
                m,
                n,
                k,
                1.0,
                A[copy].data(),
                k,
                Bt[copy].data(),
                (btran == matrix_op_t::NoTranspose) ? kAligned : nAligned,
                beta,
                C_ref[copy].data(),
                n);
#endif
            }
          },
          3,
          NITER,
          [&]() {
            if (flush) {
              int copy = num_instances == 1 ? 0 : fbgemm_get_thread_num();
              cache_evict(A[copy]);
              cache_evict(Bt[copy]);
              cache_evict(C_ref[copy]);
            }
          },
          // Use OpenMP if num instances > 1
          num_instances > 1);

      gflops = nflops / ttot / 1e9;
      gbs = nbytes / ttot / 1e9;
      printf(
          "\n%30s m = %5d n = %5d k = %5d Gflops = %8.4lf GBytes = %8.4lf\n",
          type.c_str(),
          m,
          n,
          k,
          gflops * repetitions,
          gbs * repetitions);
#ifdef USE_MKL
    }
#endif
    type = "FBP_" + std::string(typeid(btype).name());

    ttot = measureWithWarmup(
        [&]() {
          // When executing in data decomposition (single-instance) mode
          // Different threads will access different regions of the same
          // matrices. Thus, copy to be used is always 0. The numbers of
          // threads would be the as number of threads in the parallel
          // region.
          // When running in functional decomposition (multi-instance) mode
          // different matrices are used. The copy to be used selected by
          // thread_id (thread_num), and the number of threads performance
          // the compute of the same instance is 1.
          int copy = num_instances == 1 ? 0 : fbgemm_get_thread_num();
          int num_threads = num_instances == 1 ? fbgemm_get_num_threads() : 1;
          int tid = num_instances == 1 ? fbgemm_get_thread_num() : 0;

          for (int i = 0; i < repetitions; ++i) {
            cblas_gemm_compute(
                matrix_op_t::NoTranspose,
                m,
                A[copy].data(),
                *Bp[copy],
                beta,
                C_fb[copy].data(),
                tid,
                num_threads);
          }
        },
        3,
        NITER,
        [&]() {
          if (flush) {
            int copy = num_instances == 1 ? 0 : fbgemm_get_thread_num();
            cache_evict(A[copy]);
            cache_evict(*Bp[copy]);
            cache_evict(C_fb[copy]);
          }
        },
        true /*useOpenMP*/);

    gflops = nflops / ttot / 1e9;
    gbs = nbytes / ttot / 1e9;
    printf(
        "%30s m = %5d n = %5d k = %5d Gflops = %8.4lf GBytes = %8.4lf\n",
        type.c_str(),
        m,
        n,
        k,
        gflops * repetitions,
        gbs * repetitions);
  }
}

int main(int argc, const char* argv[]) {
  int num_instances = 1;
#ifdef _OPENMP
  const char* inst = getenv("GEMMBENCH_NUM_INSTANCES");
  if (inst != nullptr && *inst) {
    num_instances = std::max(atoi(inst), num_instances);
  }
  num_instances =
      parseArgumentInt(argc, argv, "--inst=", num_instances, num_instances);
  printf("Running %d instances\n", num_instances);
  if (num_instances > 1) {
    // Set-up execution for multi-instance mode
    // Number of threads in OpenMP parallel region is explicitly
    // set to the number of instances to be executed.
    omp_set_num_threads(num_instances);
#ifdef USE_MKL
    // each instance should be run with a single thread
    mkl_set_num_threads(1);
#endif
  } else {
    // When running single instance use OMP_NUM_THREADS to determine
    // parallelism. Default behaviour is using a single thread.
    int num_threads = parseArgumentInt(argc, argv, "--num_threads=", 1, 1);
    const char* val = getenv("OMP_NUM_THREADS");
    if (val == nullptr || !*val) {
      omp_set_num_threads(num_threads);
    }
  }

#endif

  int repetitions = parseArgumentInt(argc, argv, "--repit=", 1, 1);
  bool no_flush = parseArgumentBool(argc, argv, "--no-flush", false);
  bool no_mkl = parseArgumentBool(argc, argv, "--no-mkl", false);
  bool enableAvx512_ymm = parseArgumentBool(argc, argv, "--avx512-256", false);
  fbgemmEnableAvx512Ymm(enableAvx512_ymm);

  performance_test(num_instances, !no_flush, repetitions, !no_mkl);
}
