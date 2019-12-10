/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <chrono>
#include <cmath>
#include <random>

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_BLAS
#include <cblas.h>
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

void performance_test(int num_instances, bool flush) {
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
    for(int i = 0; i < num_instances; ++i) {
      A.push_back(aligned_vector<float>(Aint.begin(), Aint.end()));
    }

    aligned_vector<int> Bint(k * n);
    randFill(Bint, 0, 4);
    vector<aligned_vector<float>> B;
    for(int i = 0; i < num_instances; ++i) {
      B.push_back(aligned_vector<float>(Bint.begin(), Bint.end()));
    }


    vector<unique_ptr<PackedGemmMatrixFP16>> Bp;
    for(int i = 0; i < num_instances; ++i) {
      Bp.push_back(
        make_unique<PackedGemmMatrixFP16>(btran, k, n, alpha, B[i].data()));
    }


    vector<aligned_vector<float>> C_ref;
    vector<aligned_vector<float>> C_fb;
    if (beta != 0.0f) {
      aligned_vector<int> Cint(m * n);
      randFill(Cint, 0, 4);
      for(int i = 0; i < num_instances; ++i) {
        C_ref.push_back(aligned_vector<float>(Cint.begin(), Cint.end()));
        C_fb.push_back(aligned_vector<float>(Cint.begin(), Cint.end()));
      }
    } else {
      for(int i = 0; i < num_instances; ++i) {
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
          btran == matrix_op_t::Transpose ? CblasTrans : CblasNoTrans,
          m,
          n,
          k,
          alpha,
          A[0].data(),
          k,
          B[0].data(),
          (btran == matrix_op_t::NoTranspose) ? n : k,
          beta,
          C_ref[0].data(),
          n);
#else
      cblas_sgemm_ref(
          matrix_op_t::NoTranspose,
          btran,
          m,
          n,
          k,
          alpha,
          A[0].data(),
          k,
          B[0].data(),
          (btran == matrix_op_t::NoTranspose) ? n : k,
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
#if defined(USE_MKL) || defined(USE_BLAS)
          cblas_sgemm(
              CblasRowMajor,
              CblasNoTrans,
              btran == matrix_op_t::Transpose ? CblasTrans : CblasNoTrans,
              m,
              n,
              k,
              alpha,
              A[copy].data(),
              k,
              B[copy].data(),
              (btran == matrix_op_t::NoTranspose) ? n : k,
              beta,
              C_ref[copy].data(),
              n);
#else
          cblas_sgemm_ref(
              matrix_op_t::NoTranspose,
              btran,
              m,
              n,
              k,
              alpha,
              A[copy].data(),
              k,
              B[copy].data(),
              (btran == matrix_op_t::NoTranspose) ? n : k,
              beta,
              C_ref[copy].data(),
              n);
#endif
        },
        3,
        NITER,
        [&]() {
          if (flush) {
            int copy = num_instances == 1 ? 0 : fbgemm_get_thread_num();
            cache_evict(A[copy]);
            cache_evict(B[copy]);
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
        gflops,
        gbs);

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

          cblas_gemm_compute(
              matrix_op_t::NoTranspose,
              m,
              A[copy].data(),
              *Bp[copy],
              beta,
              C_fb[copy].data(),
              tid,
              num_threads);
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
        gflops,
        gbs);
  }
}

int main(int argc, char** argv) {
#ifdef _OPENMP
  const char* inst = getenv("GEMMBENCH_NUM_INSTANCES");
  int num_instances = 1;
  if (inst != nullptr && *inst) {
    num_instances = std::max(atoi(inst), num_instances);
  }

  for (auto i = 1; i < argc; ++i) {
    static const char param[] = "--inst=";
    const char* ptr = strstr(argv[i], param);
    if (ptr) {
      ptr += sizeof(param) - 1; // null terminated
      num_instances = std::max(atoi(ptr), num_instances);
    }
  }
  printf("Running %d instances\n", num_instances);
  if (num_instances > 1) {
      // Set-up execution for multi-instance mode
      // Number of threads in OpenMP parallel region is explicitly
      // set to the number of instances to be executed
      // If not previosly set by KMP_AFFINITY env. variable
      // threads are affinitized sequentially to logical processors
      char env_var[1024];
      sprintf(
          env_var, "granularity=fine,explicit,proclist=[1-%d]", num_instances);
      setenv("KMP_AFFINITY", env_var, 0); // Don't overide if already set
      omp_set_num_threads(num_instances);
  } else {
    // When running single instance use OMP_NUM_THREADS to determine
    // parallelism. Default behaviour is using a single thread.
    // Use 1 thread unless OMP_NUM_THREADS is explicit set.
    const char* val = getenv("OMP_NUM_THREADS");
    if (val == nullptr || !*val) {
      omp_set_num_threads(1);
    }
  }

#endif

  bool flush = true;
  for (auto i = 1; i < argc; ++i) {
    static const char param[] = "--no-flush";
    const char* ptr = strstr(argv[i], param);
    if (ptr) {
      flush = false;
    }
  }

  performance_test(num_instances, flush);
}
