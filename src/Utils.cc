/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Intel Corporation.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/Utils.h"
#include <cpuinfo.h>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#endif

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
    size_t max_mismatches_to_report,
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
  for (size_t r = 0; r < R; ++r) {
    for (size_t c = 0; c < C; ++c) {
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
    size_t max_mismatches_to_report,
    float atol);

template int compare_buffers<int32_t>(
    const int32_t* ref,
    const int32_t* test,
    int m,
    int n,
    int ld,
    size_t max_mismatches_to_report,
    float atol);

template int compare_buffers<uint8_t>(
    const uint8_t* ref,
    const uint8_t* test,
    int m,
    int n,
    int ld,
    size_t max_mismatches_to_report,
    float atol);

template int compare_buffers<int64_t>(
    const int64_t* ref,
    const int64_t* test,
    int m,
    int n,
    int ld,
    size_t max_mismatches_to_report,
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

namespace {
inst_set_t g_forced_isa = inst_set_t::anyarch;
bool g_Avx512_Ymm_enabled = false;

inst_set_t fbgemmEnvGetIsa() {
  static const char* isa_env = "FBGEMM_ENABLE_INSTRUCTIONS";
  static const std::unordered_map<std::string, inst_set_t> isaMap = {
      {"AVX2", inst_set_t::avx2},
      {"AVX512", inst_set_t::avx512},
      {"AVX512_E1", inst_set_t::avx512_vnni},
      {"AVX512_256", inst_set_t::avx512_ymm},
      {"AVX512_E1_256", inst_set_t::avx512_vnni_ymm},
  };
  const char* env = std::getenv(isa_env);
  if (env == nullptr) {
    return inst_set_t::anyarch;
  }

#ifdef __aarch64__
#ifdef VLOG
  VLOG(0) << "[" << env << "] not supported on aarch64";
#endif
  return inst_set_t::anyarch;
#endif

  std::string val(env);
  std::transform(val.begin(), val.end(), val.begin(), ::toupper);
  auto it = isaMap.find(val);
  return it == isaMap.end() ? inst_set_t::anyarch : it->second;
}

bool fbgemmEnvAvx512_256Enabled() {
  static const char* isa_env = "FBGEMM_ENABLE_AVX512_256";
  const char* env = std::getenv(isa_env);
  if (env == nullptr) {
    return false;
  }

#ifdef __aarch64__
#ifdef VLOG
  VLOG(0) << "[" << env << "] not supported on aarch64";
#endif
  return false;
#endif

  std::string val(env);
  std::transform(val.begin(), val.end(), val.begin(), ::tolower);
  return val == "true" || val == "1";
}

// This is require for build by older compilers GCC 5.4 and C++11
struct inst_set_t_hash {
  std::size_t operator()(inst_set_t t) const {
    return static_cast<std::size_t>(t);
  }
};

std::unordered_map<
    inst_set_t,
    std::unordered_set<inst_set_t, inst_set_t_hash>,
    inst_set_t_hash>
    isaSupportMap = {
        {inst_set_t::anyarch, {inst_set_t::anyarch}},
        {inst_set_t::avx2, {inst_set_t::avx2, inst_set_t::anyarch}},
        {inst_set_t::avx512,
         {inst_set_t::avx512, inst_set_t::avx512_ymm, inst_set_t::avx2}},
        {inst_set_t::avx512_ymm,
         {inst_set_t::avx512, inst_set_t::avx512_ymm, inst_set_t::avx2}},
        {inst_set_t::avx512_vnni,
         {inst_set_t::avx512_vnni,
          inst_set_t::avx512_vnni_ymm,
          inst_set_t::avx512,
          inst_set_t::avx512_ymm,
          inst_set_t::avx2}},
        {inst_set_t::avx512_vnni_ymm,
         {inst_set_t::avx512_vnni,
          inst_set_t::avx512_vnni_ymm,
          inst_set_t::avx512,
          inst_set_t::avx512_ymm,
          inst_set_t::avx2}},
};

} // namespace

/**
 * @brief Force specific architecure to for GEMM kernel execution
 *        overides FBGEMM_ENABLE_AVX512_256 env. variable
 * @param isa the ISA to enforce, supported optionsi
 *     AVX2          inst_set_t::avx2
 *     AVX512        inst_set_t::avx512
 *     AVX512_E1     inst_set_t::avx512_vnni
 *     AVX512_256    inst_set_t::avx512_ymm
 *     AVX512_E1_256 inst_set_t::avx512_vnni_ymm
 */
void fbgemmForceIsa(inst_set_t isa) {
  g_forced_isa = isa;
#ifdef __aarch64__
#ifdef VLOG
  VLOG(0) << "[anyarch] forced on aarch64";
#endif
  g_forced_isa = inst_set_t::anyarch;
#endif
}

/**
 * @brief Enables AVX512-256 if appriate. Inteded for Skylake based Xeon-D
 *        processors, wherein AXV512-256 is preferred due to higher
 *        Turbo frequencis
 * @param flag True enables / False disables
 */
void fbgemmEnableAvx512Ymm(bool flag) {
  g_Avx512_Ymm_enabled = flag;
}

/**
 * @brief Determine the best available x86 machine ISA to be used for
 *        GEMM kernels.
 *        FBGEMM_ENABLE_AVX512_256 env. or fbgemmForceIsa() are set
 *        forces to specific architecture if supported by the processor.
 *        Enforcing on Skylake to AVX2 will execute AVX2 version of the kernel
 *        However, enforcing AVX512-256 on Broadwell will fail, and AVX2 version
 *        of the kernels will be executed.
 */
inst_set_t fbgemmInstructionSet() {
  static const inst_set_t env_forced_isa = fbgemmEnvGetIsa();
  static const bool isAvx512_Ymm_enabled = fbgemmEnvAvx512_256Enabled();

  inst_set_t forced_isa =
      g_forced_isa != inst_set_t::anyarch ? g_forced_isa : env_forced_isa;
  static const inst_set_t detected_isa = ([]() {
    inst_set_t isa = inst_set_t::anyarch;
    // Check environment
    if (cpuinfo_initialize()) {
      const bool isXeonD = fbgemmIsIntelXeonD() &&
          (g_Avx512_Ymm_enabled || isAvx512_Ymm_enabled);
      if (fbgemmHasAvx512VnniSupport()) {
        if (isXeonD) {
          isa = inst_set_t::avx512_vnni_ymm;
        } else {
          isa = inst_set_t::avx512_vnni;
        }
      } else if (fbgemmHasAvx512Support()) {
        if (isXeonD) {
          isa = inst_set_t::avx512_ymm;
        } else {
          isa = inst_set_t::avx512;
        }
      } else if (fbgemmHasAvx2Support()) {
        isa = inst_set_t::avx2;
      }
    }
    return isa;
  })();

  if (forced_isa == inst_set_t::anyarch) {
    return detected_isa;
  }
  const auto supported_isa = isaSupportMap.find(detected_isa);
  assert(
      supported_isa != isaSupportMap.end() &&
      "Detected ISA can't be located in Supported ISA map");
  if (supported_isa == isaSupportMap.end()) {
    return detected_isa;
  }
  return supported_isa->second.count(forced_isa) ? forced_isa : detected_isa;
}

bool isZmm(inst_set_t isa) {
  return isa == inst_set_t::avx512 || isa == inst_set_t::avx512_vnni;
}

bool isYmm(inst_set_t isa) {
  return isa == inst_set_t::avx512_ymm || isa == inst_set_t::avx512_vnni_ymm ||
      isa == inst_set_t::avx2;
}

bool fbgemmIsIntelXeonD() {
  auto const pkgInfo = cpuinfo_get_packages();
  if (strstr(pkgInfo->name, "Intel Xeon D-") ||
      cpuinfo_get_packages_count() == 1) {
    return true;
  }
  return false;
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

bool fbgemmHasArmNeonSupport() {
  return (cpuinfo_has_arm_neon());
}

void fbgemmPartition1D(
    int thread_id,
    int num_threads,
    int64_t total_work,
    int64_t& start,
    int64_t& end) {
  // if num_threads == 0,
  // this threads should not perform any work
  if (num_threads == 0) {
    start = end = 0;
    return;
  }
  int64_t work_per_thread = (total_work + num_threads - 1) / num_threads;
  start = std::min(thread_id * work_per_thread, total_work);
  end = std::min((thread_id + 1) * work_per_thread, total_work);
}

void fbgemmPartition1DBlocked(
    int thread_id,
    int num_threads,
    int64_t total_work,
    int block_size,
    int64_t& start,
    int64_t& end) {
  if (block_size == 1) {
    return fbgemmPartition1D(thread_id, num_threads, total_work, start, end);
  }
  int64_t total_work_in_blocks = total_work / block_size;
  int64_t start_block, end_block;
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
  void* aligned_mem = nullptr;
  int ret;
#ifdef _MSC_VER
  aligned_mem = _aligned_malloc(size, align);
  ret = 0;
#else
  ret = posix_memalign(&aligned_mem, align, size);
#endif
  // Throw std::bad_alloc in the case of memory allocation failure.
  if (raiseException || ret || aligned_mem == nullptr) {
    throw std::bad_alloc();
  }
  return aligned_mem;
}

void fbgemmAlignedFree(void* p) {
#ifdef _MSC_VER
  _aligned_free(p);
#else
  free(p);
#endif
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

  // for large thread numbers, we would like to reduce the aspect_ratio ---
  // if the matrix is short-and-fat
  // this allows us to assign more parallelism to i-dimension
  if (nthreads > 16 && m / n < 0.2) {
    aspect_ratio = 0.2;
  }

  int mb = 1;
  int nb = nthreads / mb;
  int bm = (m + mb - 1) / mb;
  int bn = ((n + n_align - 1) / n_align + nb - 1) / nb * n_align;
  double best_delta = std::abs(static_cast<double>(bm) / bn - aspect_ratio);
  for (int mb_candidate = 2; mb_candidate <= nthreads; mb_candidate++) {
    // mb does not need to divide nthreads
    // here nthreads % mb_candidate!=0 constraint is removed for nthreads>16
    if (nthreads % mb_candidate != 0 && nthreads <= 16) {
      continue;
    }
    int nb_candidate = nthreads / mb_candidate;
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
    if (g != 0 && num_threads % g == 0) {
      th_info.g_num_threads = g;
    } else {
      th_info.g_num_threads = 1;
    }
  }
  num_threads /= th_info.g_num_threads;

  // We favor the parallelization on the m dimension compared to the n
  // dimension, so we set aspect_ratio to 0.5 here.
  th_info.m_num_threads = fbgemmGet2DPartition(m, n, num_threads, n_align, 0.5);

  // when num_threads >16, m_num_threads may not divide num_threads
  if (num_threads <= 16) {
    assert(num_threads % (th_info.m_num_threads) == 0);
  }
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

  // thread can be inactive,
  // meaning they are launched, but will not be assigned any work
  if (thread_id >=
      th_info.g_num_threads * th_info.m_num_threads * th_info.n_num_threads) {
    th_info.m_thread_id = 0;
    th_info.n_thread_id = 0;
    th_info.g_thread_id = 0;
    th_info.m_num_threads = 0;
    th_info.n_num_threads = 0;
    th_info.g_num_threads = 0;
    return th_info;
  }

  // We can view the thread as the ternary with 3-dim base: {g,m,n}_num_threads.
  th_info.n_thread_id = thread_id % th_info.n_num_threads;
  thread_id /= th_info.n_num_threads;
  th_info.m_thread_id = thread_id % th_info.m_num_threads;
  thread_id /= th_info.m_num_threads;
  th_info.g_thread_id = thread_id % th_info.g_num_threads;

  return th_info;
}

namespace {

// implementation taken from pytorch/c10/util/llvmMathExtras.h
template <typename T>
size_t count_leading_zeros(T val) {
  if (!val)
    return std::numeric_limits<T>::digits;

  // Use bisection method
  size_t zero_bits = 0;
  for (auto shift = std::numeric_limits<T>::digits >> 1; shift; shift >>= 1) {
    const auto tmp = val >> shift;
    if (tmp)
      val = tmp;
    else
      zero_bits |= shift;
  }
  return zero_bits;
}

// histogram size per thread
constexpr int RDX_HIST_SIZE = 256;

void update_prefsum_and_offset_in_range(
    int& offset,
    const int bins_beg,
    const int bins_end,
    const int nthreads,
    const int* const histogram,
    int* const histogram_ps) {
  for (int bins = bins_beg; bins < bins_end; ++bins) {
    for (int t = 0; t < nthreads; ++t) {
      histogram_ps[t * RDX_HIST_SIZE + bins] = offset;
      offset += histogram[t * RDX_HIST_SIZE + bins];
    }
  }
}

void combine_prefix_sum(
    const int nthreads,
    const int elements_count,
    const int* const histogram,
    int* const histogram_ps) {
  int offset = 0;
  update_prefsum_and_offset_in_range(
      offset, 0, RDX_HIST_SIZE, nthreads, histogram, histogram_ps);
  histogram_ps[RDX_HIST_SIZE * nthreads] = offset;
  // TODO(DamianSzwichtenberg): Is assert sufficient? In most cases, it will
  // work only in debug build.
  assert(offset == elements_count);
  // Suppress unused variable warning
  (void)elements_count;
}

void combine_prefix_sum_for_msb(
    const int nthreads,
    const int elements_count,
    const int* const histogram,
    int* const histogram_ps) {
  int offset = 0;
  update_prefsum_and_offset_in_range(
      offset, 128, RDX_HIST_SIZE, nthreads, histogram, histogram_ps);
  update_prefsum_and_offset_in_range(
      offset, 0, 128, nthreads, histogram, histogram_ps);
  histogram_ps[RDX_HIST_SIZE * (nthreads - 1) + 127] = offset;
  // TODO(DamianSzwichtenberg): Is assert sufficient? In most cases, it will
  // work only in debug build.
  assert(offset == elements_count);
  // Suppress unused variable warning
  (void)elements_count;
}

template <typename K, typename V>
void radix_sort_kernel(
    const K* const input_keys,
    const V* const input_values,
    K* const output_keys,
    V* const output_values,
    const int elements_count,
    int* const histogram,
    int* const histogram_ps,
    const int pass,
    const bool pass_with_sign_bit = false) {
  const auto tid = omp_get_thread_num();
  const auto nthreads = omp_get_num_threads();
  const auto elements_count_4 = elements_count / 4 * 4;

  int* const local_histogram = &histogram[RDX_HIST_SIZE * tid];
  int* const local_histogram_ps = &histogram_ps[RDX_HIST_SIZE * tid];

  // Step 1: compute histogram
  for (int i = 0; i < RDX_HIST_SIZE; i++) {
    local_histogram[i] = 0;
  }

#pragma omp for schedule(static)
  for (int64_t i = 0; i < elements_count_4; i += 4) {
    const auto key_1 = input_keys[i];
    const auto key_2 = input_keys[i + 1];
    const auto key_3 = input_keys[i + 2];
    const auto key_4 = input_keys[i + 3];

    local_histogram[(key_1 >> (pass * 8)) & 0xFF]++;
    local_histogram[(key_2 >> (pass * 8)) & 0xFF]++;
    local_histogram[(key_3 >> (pass * 8)) & 0xFF]++;
    local_histogram[(key_4 >> (pass * 8)) & 0xFF]++;
  }
  if (tid == (nthreads - 1)) {
    for (int64_t i = elements_count_4; i < elements_count; ++i) {
      const auto key = input_keys[i];
      local_histogram[(key >> (pass * 8)) & 0xFF]++;
    }
  }
#pragma omp barrier
  // Step 2: prefix sum
  if (tid == 0) {
    if (pass_with_sign_bit) {
      combine_prefix_sum_for_msb(
          nthreads, elements_count, histogram, histogram_ps);

    } else {
      combine_prefix_sum(nthreads, elements_count, histogram, histogram_ps);
    }
  }
#pragma omp barrier

  // Step 3: scatter
#pragma omp for schedule(static)
  for (int64_t i = 0; i < elements_count_4; i += 4) {
    const auto key_1 = input_keys[i];
    const auto key_2 = input_keys[i + 1];
    const auto key_3 = input_keys[i + 2];
    const auto key_4 = input_keys[i + 3];

    const int bin_1 = (key_1 >> (pass * 8)) & 0xFF;
    const int bin_2 = (key_2 >> (pass * 8)) & 0xFF;
    const int bin_3 = (key_3 >> (pass * 8)) & 0xFF;
    const int bin_4 = (key_4 >> (pass * 8)) & 0xFF;

    const int pos_1 = local_histogram_ps[bin_1]++;
    const int pos_2 = local_histogram_ps[bin_2]++;
    const int pos_3 = local_histogram_ps[bin_3]++;
    const int pos_4 = local_histogram_ps[bin_4]++;

    output_keys[pos_1] = key_1;
    output_values[pos_1] = input_values[i];
    output_keys[pos_2] = key_2;
    output_values[pos_2] = input_values[i + 1];
    output_keys[pos_3] = key_3;
    output_values[pos_3] = input_values[i + 2];
    output_keys[pos_4] = key_4;
    output_values[pos_4] = input_values[i + 3];
  }
  if (tid == (nthreads - 1)) {
    for (int64_t i = elements_count_4; i < elements_count; ++i) {
      const auto key = input_keys[i];
      const int pos = local_histogram_ps[(key >> (pass * 8)) & 0xFF]++;
      output_keys[pos] = key;
      output_values[pos] = input_values[i];
    }
  }
}

} // namespace

template <typename K, typename V>
std::pair<K*, V*> radix_sort_parallel(
    K* const inp_key_buf,
    V* const inp_value_buf,
    K* const tmp_key_buf,
    V* const tmp_value_buf,
    const int64_t elements_count,
    const int64_t max_value,
    const bool maybe_with_neg_vals) {
  if (max_value == 0) {
    return {inp_key_buf, inp_value_buf};
  }

  const auto maxthreads = omp_get_max_threads();
#ifdef _MSC_VER
  const size_t array_size = (size_t)RDX_HIST_SIZE * maxthreads;
  // fixes MSVC error C2131
  int* const histogram = static_cast<int*>(
      fbgemm::fbgemmAlignedAlloc(64, (array_size) * sizeof(int)));
  int* const histogram_ps = static_cast<int*>(
      fbgemm::fbgemmAlignedAlloc(64, (array_size + 1) * sizeof(int)));

#else
  alignas(64) int histogram[RDX_HIST_SIZE * maxthreads];
  alignas(64) int histogram_ps[RDX_HIST_SIZE * maxthreads + 1];
#endif
  // If negative values are present, we want to perform all passes
  // up to a sign bit
  int num_bits = sizeof(K) * 8;
  if (!maybe_with_neg_vals)
    // __builtin_clz is not portable, std::countl_zero is available in C++20
    num_bits -= count_leading_zeros(
        static_cast<typename std::make_unsigned<K>::type>(max_value));

  const unsigned int num_passes = (num_bits + 7) / 8;

#pragma omp parallel
  {
    K* input_keys = inp_key_buf;
    V* input_values = inp_value_buf;
    K* output_keys = tmp_key_buf;
    V* output_values = tmp_value_buf;

    for (unsigned int pass = 0; pass < num_passes; pass++) {
      radix_sort_kernel(
          input_keys,
          input_values,
          output_keys,
          output_values,
          elements_count,
          histogram,
          histogram_ps,
          pass,
          maybe_with_neg_vals && pass == num_passes - 1);

      std::swap(input_keys, output_keys);
      std::swap(input_values, output_values);
#pragma omp barrier
    }
  }
#ifdef _MSC_VER
  fbgemm::fbgemmAlignedFree(histogram);
  fbgemm::fbgemmAlignedFree(histogram_ps);
#endif
  return (
      num_passes % 2 == 0 ? std::make_pair(inp_key_buf, inp_value_buf)
                          : std::make_pair(tmp_key_buf, tmp_value_buf));
}

#define FORALL_INT_TYPES_AND_KEY(key_t, _) \
  _(key_t, uint8_t)                        \
  _(key_t, int8_t)                         \
  _(key_t, int16_t)                        \
  _(key_t, int)                            \
  _(key_t, int64_t)

#define INSTANTIATE(key_t, val_t)                                    \
  template FBGEMM_API std::pair<key_t*, val_t*> radix_sort_parallel( \
      key_t* const inp_key_buf,                                      \
      val_t* const inp_value_buf,                                    \
      key_t* const tmp_key_buf,                                      \
      val_t* const tmp_value_buf,                                    \
      const int64_t elements_count,                                  \
      const int64_t max_value,                                       \
      const bool maybe_with_neg_vals);

FORALL_INT_TYPES_AND_KEY(uint8_t, INSTANTIATE);
FORALL_INT_TYPES_AND_KEY(int8_t, INSTANTIATE);
FORALL_INT_TYPES_AND_KEY(int16_t, INSTANTIATE);
FORALL_INT_TYPES_AND_KEY(int, INSTANTIATE);
FORALL_INT_TYPES_AND_KEY(int64_t, INSTANTIATE);

using pair_int_double = std::pair<int, double>;
using pair_int_float = std::pair<int, float>;
INSTANTIATE(int, pair_int_double);
INSTANTIATE(int, pair_int_float);

bool is_radix_sort_accelerated_with_openmp() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

} // namespace fbgemm
