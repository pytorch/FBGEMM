/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <emmintrin.h>
#include <immintrin.h>
#include <algorithm>
#include <bitset>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif

#include "./BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"
#include "test/QuantizationHelpers.h"

using namespace std;
using namespace fbgemm;

// ref implementation of encode checksum of A's column
// sum to an array
void encodeMatColCksm_ref(
    int rowLength,
    int colLength,
    uint8_t* mat,
    uint8_t* cksmArr,
    uint32_t mod) {
  std::vector<uint32_t> cksmBeforeMod(colLength);

  for (int i = 0; i < rowLength; i++) {
    for (int j = 0; j < colLength; j++) {
      cksmBeforeMod[j] += static_cast<uint32_t>(mat[i * colLength + j]);
    }
  }
  for (int j = 0; j < colLength; j++) {
    cksmArr[j] = static_cast<uint8_t>(cksmBeforeMod[j] % mod);
  }
}

// vectorized version of the above function
void encodeMatColCksm_vec(
    int rowLength,
    int colLength,
    uint8_t* mat,
    uint8_t* cksmArr,
    uint32_t mod) {
  std::vector<uint32_t> cksmBeforeMod(colLength);

  for (int i = 0; i < rowLength; i++) {
    uint8_t* matTmp = mat + i * colLength;
    for (int j = 0; j < colLength; j += 16) {
      __m128i mat8 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(matTmp + j));
      __m512i mat32 = _mm512_cvtepu8_epi32(mat8);
      __m512i sum32 = _mm512_load_epi32(&cksmBeforeMod[j]);
      __m512i res32 = _mm512_add_epi32(mat32, sum32);
      _mm512_store_epi32(&cksmBeforeMod[j], res32);
    }
  }

  for (int j = 0; j < colLength; j++) {
    cksmArr[j] = static_cast<uint8_t>(cksmBeforeMod[j] % mod);
  }
}

// verify C's column sum with an array, cksmArr,
// to detect silent error in that column
// return the number of wrong columns
int32_t verifyMatColCksm_ref(
    int rowLength,
    int colLength,
    int32_t* mat,
    int32_t* cksmArr,
    uint32_t mod1) {
  std::vector<int64_t> sums(colLength);
  // must convert the mod into signed integer
  // otherwise, there is no negative remainder
  int32_t mod = static_cast<int32_t>(mod1);

  for (int i = 0; i < rowLength; i++) {
    for (int j = 0; j < colLength; j++) {
      sums[j] += mat[i * colLength + j];
    }
  }
  int32_t errCnt = 0;
  for (int i = 0; i < colLength; i++) {
    // analysis shows it's safe to do this
    // since sums[i] is int64_t, no overflow
    if ((sums[i] - cksmArr[i]) % mod != 0) {
    //  if (sums[i] - cksmArr[i] == i){
      errCnt++;
    }
  }
  return errCnt;
}

// vectorized version of the above function
int32_t verifyMatColCksm_vec(
    int rowLength,
    int colLength,
    int32_t* mat,
    int32_t* cksmArr,
    uint32_t mod1) {
  std::vector<int64_t> sums(colLength);
  int32_t mod = static_cast<int32_t>(mod1);

  for (int i = 0; i < rowLength; i++) {
    int32_t* matTmp = mat + i * colLength;
    for (int j = 0; j < colLength; j += 8) {
      __m256i mat32 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(matTmp + j));
      __m512i mat64 = _mm512_cvtepi32_epi64(mat32);
      __m512i sum64 = _mm512_load_epi64(&sums[j]);
      __m512i res64 = _mm512_add_epi64(mat64, sum64);
      _mm512_store_epi64(&sums[j], res64);
    }
  }

  int32_t errCnt = 0;
  for (int i = 0; i < colLength; i++) {
    //if ((sums[i] - cksmArr[i]) % mod != 0) {
    //  if (sums[i] - cksmArr[i] == i){
      errCnt++;
    }
  }
  return errCnt;
}

// randomly flip a bit in matrix C
// argument "high" is number of bits
// we could flip, defualt is 32
void randInjectC(int rowLength, int colLength, int32_t* mat, int high) {
  uniform_int_distribution<uint32_t> unif;
  random_device rd;
  mt19937 engine(rd());
  function<uint32_t()> rnd = bind(unif, engine);
  int randIndex = rnd() % (rowLength * colLength);
  int flipBit = rnd() % high; // randomly pick a bit to flip
  uint32_t* orig = reinterpret_cast<uint32_t*>(&mat[randIndex]);
  uint32_t shifted = *orig >> flipBit;
  if ((shifted & 1) == 1) {
    // flip 1 to 0 by subtracting
    *orig = *orig - (1 << flipBit);
  } else {
    // flip 0 to 1 by addition
    *orig = *orig + (1 << flipBit);
  }
  mat[randIndex] = *reinterpret_cast<int32_t*>(orig);
  return;
}

// randomly flip a bit in matrix A
void randInjectA(int rowLength, int colLength, uint8_t* mat, int high) {
  uniform_int_distribution<uint32_t> unif;
  random_device rd;
  mt19937 engine(rd());
  function<uint32_t()> rnd = bind(unif, engine);
  int randIndex = rnd() % (rowLength * colLength);
  int flipBit = rnd() % high;
  uint8_t orig = mat[randIndex];
  uint32_t shifted = orig >> flipBit;
  if ((shifted & 1) == 1) {
    orig = orig - (1 << flipBit);
  } else {
    orig = orig + (1 << flipBit);
  }
  mat[randIndex] = orig;
  return;
}

// randomly flip a bit in matrix B
// note: current design encoding only A
// so bitflips in B will not be detected
void randInjectB(int rowLength, int colLength, int8_t* mat, int high) {
  uniform_int_distribution<uint32_t> unif;
  random_device rd;
  mt19937 engine(rd());
  function<uint32_t()> rnd = bind(unif, engine);
  int randIndex = rnd() % (rowLength * colLength);
  int flipBit = rnd() % high;
  uint8_t* orig = reinterpret_cast<uint8_t*>(&mat[randIndex]);
  uint32_t shifted = *orig >> flipBit;
  if ((shifted & 1) == 1) {
    *orig = *orig - (1 << flipBit);
  } else {
    *orig = *orig + (1 << flipBit);
  }
  mat[randIndex] = *reinterpret_cast<int8_t*>(orig);
  return;
}

// a toy function to test/debug random bit flip injector
void testRandBitFlipInjector() {
  int32_t testC = 24524;
  bitset<32> beforeC(testC);
  cout << "C before: " << beforeC << endl;
  randInjectC(1, 1, &testC, 32);
  bitset<32> afterC(testC);
  cout << "C a fter: " << afterC << endl;
  uint8_t testA = 234;
  bitset<8> beforeA(testA);
  cout << "A before: " << beforeA << endl;
  randInjectA(1, 1, &testA, 8);
  bitset<8> afterA(testA);
  cout << "A  after: " << afterA << endl;
  int8_t testB = -69;
  bitset<8> beforeB(testB);
  cout << "B before: " << beforeB << endl;
  randInjectB(1, 1, &testB, 8);
  bitset<8> afterB(testB);
  cout << "B  after: " << afterB << endl;
  return;
}

void performance_test() {
  // clang-format off
  static const vector<vector<int>> shapes = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    // m, n, k
    {64, 800, 320},
    {64, 768, 512},
    {16, 256, 512},
    {128, 128, 128},
    {256, 512, 256},
    {1024, 1024, 1024},
  };
  // clang-format on
  bool flush = true;
  std::vector<char> llc;

  if (flush) {
    llc.resize(128 * 1024 * 1024, 1.0);
  }

  constexpr int NWARMUP = 4;
  constexpr int NITER = 10;

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  cout << "WARNING: the timer may be inaccurate when used by multiple threads."
       << endl;
  cout << setw(8) << "M, " << setw(8) << "N, " << setw(8) << "K, " << setw(18)
       << "Type, " << setw(18) << "Packing (us), " << setw(18)
       << "Kernel (us), " << setw(18) << "Postproc (us), " << setw(18)
       << "Total (us), " << setw(5) << "GOPs" << endl;
#else
  cout << setw(8) << "M, " << setw(8) << "N, " << setw(8) << "K, " << setw(18)
       << "Type, " << setw(5) << "GOPS" << endl;
#endif

  chrono::time_point<chrono::high_resolution_clock> start, end, abft_start,
      abft_end; // add abft_start, abft_end to store elapsed time related to
                // abft operations
  double encodeTime = 0.0;
  double cksmCalcTime = 0.0;
  double verifyTime = 0.0;
  for (auto shape : shapes) {
    int m = shape[0];
    int n = shape[1];
    int k = shape[2];

    uint32_t mod = 255; // mod is a constant, 255 (it should be signed integer)

    float alpha = 1.f, beta = 0.f;
    aligned_vector<uint8_t> Aint8(m * k);

    aligned_vector<int8_t> Bint8(k * n);

    aligned_vector<float> Cfp32_mkl(m * n);
    aligned_vector<int32_t> Cint32_mkl(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_ref(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_fb_acc32(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_fb_acc16(Cfp32_mkl.size());
    aligned_vector<uint8_t> AColCksmArr(k); // column checksum for A
    aligned_vector<int32_t> CColCksmArr(n); // column checksum for C

    // A matrix
    randFill<uint8_t>(Aint8, 0, 5);
    aligned_vector<float> Afp32(Aint8.begin(), Aint8.end());

    randFill<int8_t>(Bint8, -4, 4);
    avoidOverflow(m, n, k, Aint8.data(), Bint8.data());

    // measure the encoding time repeatedly
    for (int i = 0; i < NWARMUP + NITER; i++) {
      abft_start = chrono::high_resolution_clock::now();
      encodeMatColCksm_vec(m, k, Aint8.data(), AColCksmArr.data(), mod);
      abft_end = chrono::high_resolution_clock::now();
      if (i >= NWARMUP) {
        auto dur =
            chrono::duration_cast<chrono::nanoseconds>(abft_end - abft_start);
        encodeTime += dur.count();
      }
    }

    aligned_vector<float> Bfp32(Bint8.begin(), Bint8.end());

    double nops = 2.0 * m * n * k;
    double ttot = 0.0;
    string runType;
#ifdef USE_MKL
    runType = "MKL_fp32";
    ttot = measureWithWarmup(
        [&]() {
          cblas_sgemm(
              CblasRowMajor,
              CblasNoTrans,
              CblasNoTrans,
              m,
              n,
              k,
              alpha,
              Afp32.data(),
              k,
              Bfp32.data(),
              n,
              beta,
              Cfp32_mkl.data(),
              n);
        },
        NWARMUP,
        NITER,
        [&]() {
          if (flush) {
            llc_flush(llc);
          }
        });
    ttot *= 1e9; // convert to ns

    ((volatile char*)(llc.data()));

    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << setw(16) << runType << ", "
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
         << setw(16) << 0 << ", " << setw(16) << 0 << ", " << setw(16) << 0
         << ", " << setw(16) << 0 << ", "
#endif
         << setw(5) << fixed << setw(5) << setprecision(1) << nops / ttot
         << endl;

    for (auto i = 0; i < Cfp32_mkl.size(); ++i) {
      Cint32_mkl[i] = (int32_t)Cfp32_mkl[i];
    }
#endif

    vector<int32_t> row_offsets(m);

    matmul_u8i8acc32_ref(
        m, n, k, k, n, n, Aint8.data(), Bint8.data(), Cint32_ref.data());

    // printMatrix(matrix_op_t::NoTranspose, Bint8.data(), k, n, n, "B
    // unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Aint8.data(), m, k, k,
    // "A unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Cint32_ref.data(),
    // m, n, n, "C int32");

    PackBMatrix<int8_t> packedB_int32(
        matrix_op_t::NoTranspose, k, n, Bint8.data(), n, nullptr, 1);

    ttot = 0.0;
    runType = "FBGEMM_i8_acc32";
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    double total_packing_time = 0.0;
    double total_computing_time = 0.0;
    double total_kernel_time = 0.0;
    double total_postprocessing_time = 0.0;
    double total_run_time = 0.0;
#endif
    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << setw(16) << runType;

    for (auto i = 0; i < NWARMUP + NITER; ++i) {
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
      packing_time = 0.0;
      computing_time = 0.0;
      kernel_time = 0.0;
      postprocessing_time = 0.0;
      run_time = 0.0;
#endif
      llc_flush(llc);
      start = chrono::high_resolution_clock::now();

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        PackAMatrix<uint8_t> packA_int32(
            matrix_op_t::NoTranspose, m, k, Aint8.data(), k, nullptr, 1);

        DoNothing<int32_t, int32_t> doNothing32BitObj;
        memCopy<> memcopyObj(doNothing32BitObj);
        int num_threads = fbgemm_get_num_threads();
        int tid = fbgemm_get_thread_num();
        // printf ( "tid: %d, num_threads: %d\n", tid, num_threads );
        fbgemmPacked(
            packA_int32,
            packedB_int32,
            Cint32_fb_acc32.data(),
            Cint32_fb_acc32.data(),
            n,
            memcopyObj,
            tid,
            num_threads);
      }

      end = chrono::high_resolution_clock::now();

      if (i >= NWARMUP) {
        auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start);
        ttot += dur.count();
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
        total_packing_time += packing_time;
        total_computing_time += computing_time;
        total_kernel_time += kernel_time;
        total_postprocessing_time += postprocessing_time;
        total_run_time += run_time;
#endif
      }
    }
    ((volatile char*)(llc.data()));

    int errCnt = 0; // count the number of errors
    // calculate C's checksum by mutipling A's checksum with matrix B
    // this is a row vector multiplied by a matrix
    for (int i = 0; i < NWARMUP + NITER; i++) {
      abft_start = chrono::high_resolution_clock::now();
      // dimension of A's column checksun is 1*k
      PackAMatrix<uint8_t> packA_int32(
          matrix_op_t::NoTranspose, 1, k, AColCksmArr.data(), k, nullptr, 1);

      DoNothing<int32_t, int32_t> doNothing32BitObj;
      memCopy<> memcopyObj(doNothing32BitObj);
      int num_threads = fbgemm_get_num_threads();
      int tid = fbgemm_get_thread_num();
      // printf ( "tid: %d, num_threads: %d\n", tid, num_threads );
      fbgemmPacked(
          packA_int32,
          packedB_int32,
          CColCksmArr.data(),
          CColCksmArr.data(),
          n,
          memcopyObj,
          tid,
          num_threads);
      abft_end = chrono::high_resolution_clock::now();

      if (i >= NWARMUP) {
        auto dur =
            chrono::duration_cast<chrono::nanoseconds>(abft_end - abft_start);
        cksmCalcTime += dur.count();
      }

      // err injection by random bit flip
      // Cint32_fb_acc32[m/2 * n + n/2] ++;
      // randInjectC(m, n, Cint32_fb_acc32.data(), 32);

      abft_start = chrono::high_resolution_clock::now();
      errCnt += verifyMatColCksm_vec(
          m, n, Cint32_fb_acc32.data(), CColCksmArr.data(), mod);
      abft_end = chrono::high_resolution_clock::now();

      if (i >= NWARMUP) {
        auto dur =
            chrono::duration_cast<chrono::nanoseconds>(abft_end - abft_start);
        verifyTime += dur.count();
      }
    }

    // printMatrix(matrix_op_t::NoTranspose, Bint8.data(), k, n, n, "B
    // unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Aint8.data(), m, k, k,
    // "A unpacked");
    // printMatrix(matrix_op_t::NoTranspose,
    // Cint8_fb.data(), m, n, n, "C fb");

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    cout << ", " << setw(16) << total_packing_time / (double)NITER / 1e3 << ", "
         << setw(16) << total_kernel_time / (double)NITER / 1e3 << ", "
         << setw(16) << total_postprocessing_time / (double)NITER / 1e3 << ", "
         << setw(16) << total_run_time / (double)NITER / 1e3;
#endif
    // show abft performance overheads
    cout << ", " << setw(5) << fixed << setw(5) << setprecision(1)
         << NITER * nops / ttot << setw(5) << setprecision(5) << endl
         << "M N K: " << m << " " << n << " " << k
         << " total time with abft (ms): "
         << (double)(ttot + encodeTime + verifyTime + cksmCalcTime) /
            (double)1e6
         << ", MM time: " << (double)ttot / (double)1e6
         << ", encodeTime: " << (double)encodeTime / (double)1e6
         << ", cksmCalcTime: " << (double)cksmCalcTime / (double)1e6
         << ", verifyTime: " << (double)verifyTime / (double)1e6
         << ", overall overheads: "
         << (double)(encodeTime + cksmCalcTime + verifyTime) /
            (double)(ttot)*100.0
         << "%" << endl;

    // show the number of errors (muliplied by repeated times)
    cout << "Number of wrong columns in C: " << errCnt << endl << endl;
    compare_buffers(Cint32_ref.data(), Cint32_fb_acc32.data(), m, n, n, 5);
  }
}

int main(int /* unused */, char** /* unused */) {
#ifdef _OPENMP
  // Use 1 thread unless OMP_NUM_THREADS is explicit set.
  const char* val = getenv("OMP_NUM_THREADS");
  if (val == nullptr || !*val) {
    omp_set_num_threads(1);
  }
#endif
  testRandBitFlipInjector();
  performance_test();
  return 0;
}
