/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "fbgemm/Utils.h"
#include <cpuinfo.h>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include "TransposeUtils.h"

namespace fbgemm {

/**
 * @brief Compare the reference and test result matrix to check the correctness.
 * @param ref The buffer for the reference result matrix.
 * @param test The buffer for the test result matrix.
 * @param m The height of the reference and test result matrix.
 * @param n The width of the reference and test result matrix.
 * @param ld The leading dimension of the reference and test result matrix.
 * @param max_mismatches_to_report The maximum number of tolerable mismatches to
 * report.
 * @param atol The tolerable error.
 * @retval false If the number of mismatches for reference and test result
 * matrix exceeds max_mismatches_to_report.
 * @retval true If the number of mismatches for reference and test result matrix
 * is tolerable.
 */
template <typename T>
int compare_buffers(
    const T* ref,
    const T* test,
    int m,
    int n,
    int ld,
    int max_mismatches_to_report,
    float atol /*=1e-3*/) {
  size_t mismatches = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      T reference = ref[i * ld + j], actual = test[i * ld + j];
      if (std::abs(reference - actual) > atol) {
        std::cout << "\tmismatch at (" << i << ", " << j << ")" << std::endl;
        if (std::is_integral<T>::value) {
          std::cout << "\t  reference:" << static_cast<int64_t>(reference)
                    << " test:" << static_cast<int64_t>(actual) << std::endl;
        } else {
          std::cout << "\t  reference:" << reference << " test:" << actual
                    << std::endl;
        }

        mismatches++;
        if (mismatches > max_mismatches_to_report) {
          return 1;
        }
      }
    }
  }
  return 0;
}

/**
 * @brief Print the matrix.
 * @param op Transpose type of the matrix.
 * @param R The height of the matrix.
 * @param C The width of the matrix.
 * @param ld The leading dimension of the matrix.
 * @param name The prefix string before printing the matrix.
 */
template <typename T>
void printMatrix(
    matrix_op_t op,
    const T* inp,
    size_t R,
    size_t C,
    size_t ld,
    std::string name) {
  // R: number of rows in op(inp)
  // C: number of cols in op(inp)
  // ld: leading dimension in inp
  std::cout << name << ":"
            << "[" << R << ", " << C << "]" << std::endl;
  bool tr = (op == matrix_op_t::Transpose);
  for (auto r = 0; r < R; ++r) {
    for (auto c = 0; c < C; ++c) {
      T res = tr ? inp[c * ld + r] : inp[r * ld + c];
      if (std::is_integral<T>::value) {
        std::cout << std::setw(5) << static_cast<int64_t>(res) << " ";
      } else {
        std::cout << std::setw(5) << res << " ";
      }
    }
    std::cout << std::endl;
  }
}

template int compare_buffers<float>(
    const float* ref,
    const float* test,
    int m,
    int n,
    int ld,
    int max_mismatches_to_report,
    float atol);

template int compare_buffers<int32_t>(
    const int32_t* ref,
    const int32_t* test,
    int m,
    int n,
    int ld,
    int max_mismatches_to_report,
    float atol);

template int compare_buffers<uint8_t>(
    const uint8_t* ref,
    const uint8_t* test,
    int m,
    int n,
    int ld,
    int max_mismatches_to_report,
    float atol);

template void printMatrix<float>(
    matrix_op_t op,
    const float* inp,
    size_t R,
    size_t C,
    size_t ld,
    std::string name);
template void printMatrix<int8_t>(
    matrix_op_t op,
    const int8_t* inp,
    size_t R,
    size_t C,
    size_t ld,
    std::string name);
template void printMatrix<uint8_t>(
    matrix_op_t op,
    const uint8_t* inp,
    size_t R,
    size_t C,
    size_t ld,
    std::string name);
template void printMatrix<int32_t>(
    matrix_op_t op,
    const int32_t* inp,
    size_t R,
    size_t C,
    size_t ld,
    std::string name);

void transpose_ref(
    int M,
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  } // for each output row
}

void transpose_simd(
    int M,
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst) {
  if ((M == 1 && ld_dst == 1) || (N == 1 && ld_src == 1)) {
    if (dst != src) {
      memcpy(dst, src, M * N * sizeof(float));
    }
    return;
  }
  // Run time CPU detection
  if (cpuinfo_initialize()) {
    if (fbgemmHasAvx512Support()) {
      internal::transpose_avx512(M, N, src, ld_src, dst, ld_dst);
    } else if (fbgemmHasAvx2Support()) {
      internal::transpose_avx2(M, N, src, ld_src, dst, ld_dst);
    } else {
      transpose_ref(M, N, src, ld_src, dst, ld_dst);
      return;
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
}

bool fbgemmHasAvx512Support() {
  return (
      cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512bw() &&
      cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512vl());
}

bool fbgemmHasAvx2Support() {
  return (cpuinfo_has_x86_avx2());
}

bool fbgemmHasAvx512VnniSupport() {
  return (cpuinfo_has_x86_avx512vnni());
}

void fbgemmPartition1D(
    int thread_id,
    int num_threads,
    int total_work,
    int& start,
    int& end) {
  int work_per_thread = (total_work + num_threads - 1) / num_threads;
  start = std::min(thread_id * work_per_thread, total_work);
  end = std::min((thread_id + 1) * work_per_thread, total_work);
}

void fbgemmPartition1DBlocked(
    int thread_id,
    int num_threads,
    int total_work,
    int block_size,
    int& start,
    int& end) {
  if (block_size == 1) {
    return fbgemmPartition1D(thread_id, num_threads, total_work, start, end);
  }
  int total_work_in_blocks = total_work / block_size;
  int start_block, end_block;
  fbgemmPartition1D(
      thread_id, num_threads, total_work_in_blocks, start_block, end_block);
  start = std::min(start_block * block_size, total_work);
  end = thread_id == num_threads - 1
      ? std::max(end_block * block_size, total_work)
      : std::min(end_block * block_size, total_work);
}

void* fbgemmAlignedAlloc(
    size_t align,
    size_t size,
    bool raiseException /*=false*/) {
  void* aligned_mem;
  if (posix_memalign(&aligned_mem, align, size)) {
    if (raiseException) {
      throw std::bad_alloc();
    }
    return nullptr;
  }
  return aligned_mem;
}

int fbgemmGet2DPartition(
    int m,
    int n,
    int nthreads,
    int n_align,
    double aspect_ratio) {
  // mb: number of thread blocks within a socket along m.
  // nb: number of thread blocks along n.
  // mb * nb = nthreads.
  // bm: number of rows assigned per thread block (bm = ceil(m/mb)).
  // bn: number of cols assigned per thread block (bn = ceil(n/nb)).
  // find mb and nb such that bm / bn is as close as possible to aspect_ratio.
  int mb = 1;
  int nb = nthreads / mb;
  int bm = (m + mb - 1) / mb;
  int bn = ((n + n_align - 1) / n_align + nb - 1) / nb * n_align;
  double best_delta = std::abs(static_cast<double>(bm) / bn - aspect_ratio);
  for (int mb_candidate = 2; mb_candidate <= nthreads; mb_candidate++) {
    if (nthreads % mb_candidate != 0) {
      continue;
    }
    int nb_candidate = nthreads / mb_candidate;
    if ((n + nb_candidate - 1) / nb_candidate <= n_align / 2) {
      continue;
    }
    int bm_candidate = (m + mb_candidate - 1) / mb_candidate;
    int bn_candidate = ((n + n_align - 1) / n_align + nb_candidate - 1) /
        nb_candidate * n_align;
    double delta = std::abs(
        static_cast<double>(bm_candidate) / bn_candidate - aspect_ratio);
    if (delta < best_delta) {
      best_delta = delta;
      mb = mb_candidate;
    } else {
      break;
    }
  }
  return mb;
}

thread_type_t fbgemmGetThreadPartition(
    int g,
    int m,
    int n,
    int thread_id,
    int num_threads,
    int n_align) {
  assert(num_threads >= 1);

  // Fast path for the single thread case.
  if (num_threads == 1) {
    return thread_type_t{1, 1, 1, 0, 0, 0};
  }

  thread_type_t th_info;

  // Heuristic for determine the thread partitions for parallelizing across g, m
  // or n dimensions.
  // TODO: more smart ways for thread partitions considering the
  // grain size (MR, NR) parameters
  if (g > num_threads) {
    // TODO: when G == nthreads + 1, we'll have a big load imbalance because
    // only one thread will get 2 groups.
    th_info.g_num_threads = num_threads;
  } else {
    if (num_threads % g == 0) {
      th_info.g_num_threads = g;
    } else {
      th_info.g_num_threads = 1;
    }
  }
  num_threads /= th_info.g_num_threads;

  // We favor the parallelization on the m dimension compared to the n
  // dimension, so we set aspect_ratio to 0.5 here.
  th_info.m_num_threads = fbgemmGet2DPartition(m, n, num_threads, n_align, 0.5);

  assert(num_threads % (th_info.m_num_threads) == 0);
  th_info.n_num_threads = num_threads / th_info.m_num_threads;

  // When there are 12 threads (num_threads = 12) and g_nthreads = 2, m_nthreads
  // = 2, the threads will be organized as the following 2x2x3 layout (thread is
  // partitioned in the last-dim index (i.e., n, m, g, row-major for 2D) major
  // order):
  //
  // thread 0, thread 1, thread 2      thread 6, thread 7,  thread 8
  // thread 3, thread 4, thread 5      thread 9, thread 10, thread 11
  //
  // And the corresponding (g_thread_id, m_thread_id, n_thread_id) for
  // each thread is listed as the following:
  //
  // (0, 0, 0), (0, 0, 1), (0, 0, 2)            (1, 0, 0), (1, 0, 1), (1, 0, 2)
  // (0, 1, 0), (0, 1, 1), (0, 1, 2)            (1, 1, 0), (1, 1, 1), (1, 1, 2)

  // We can view the thread as the ternary with 3-dim base: {g,m,n}_num_threads.
  th_info.n_thread_id = thread_id % th_info.n_num_threads;
  thread_id /= th_info.n_num_threads;
  th_info.m_thread_id = thread_id % th_info.m_num_threads;
  thread_id /= th_info.m_num_threads;
  th_info.g_thread_id = thread_id % th_info.g_num_threads;

  return th_info;
}

} // namespace fbgemm
